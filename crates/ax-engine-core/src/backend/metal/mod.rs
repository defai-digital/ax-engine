use rustc_hash::FxHashMap;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

#[cfg(test)]
use super::KvPrecisionPolicy;
use super::{Backend, RuntimePolicy};
use crate::gguf::tensor::GgmlType;
use anyhow::Context;

// v2: GpuKv is owned by InferenceModel via ModelKv, not stored in MetalOps.
use crate::model::shared::{
    encode_dequant_matvec_pair_with_config, encode_dequant_silu_down_matvec_with_config,
    env_flag_override,
};
use crate::model::{ModelFingerprint, config::ModelConfig};
use ax_engine_metal::{
    AttentionDispatchConfig, AttentionKernels, DequantDispatchConfig, DequantKernels,
    ElementwiseKernels, GdnKernels, KernelProfile, MatmulKernels, MetalBuffer, MetalDevice,
};

mod qwen35;
#[cfg(test)]
mod tests;

pub use qwen35::Qwen35RecurrentBatchPerfCounters;
pub(crate) use qwen35::{Qwen35MetalSlotBuffers, Qwen35SlotBufferSyncOutcome};

use qwen35::{
    Qwen35BatchLogitsScratchBufferKey, Qwen35BatchProjectionScratchBufferKey,
    Qwen35MetalBatchLogitsScratchBuffers, Qwen35MetalBatchProjectionScratchBuffers,
    Qwen35MetalQkvHandoffScratchBuffers, Qwen35MetalRecurrentProjectionScratchBuffers,
    Qwen35MetalRecurrentScratchBuffers, Qwen35QkvHandoffScratchBufferKey,
    Qwen35RecurrentBatchPerfState, Qwen35RecurrentScratchBufferKey, Qwen35RecurrentSlotBufferKey,
    qwen35_gate_up_scratch_dims, qwen35_recurrent_sequence_with_backend,
};

fn qwen35_selected_weighted_down_enabled(
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    if let Some(enabled) = env_flag_override("AX_QWEN35_SELECTED_WEIGHTED_DOWN") {
        return enabled
            && matches!(
                down_dtype,
                GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
            );
    }

    match down_dtype {
        // Q4_K routed down still benefits from the weighted resident path.
        GgmlType::Q4K => true,
        // Q5_K routed down is currently faster with the classic selected-down
        // plus weighted-reduce path. Keep the weighted resident kernel opt-in.
        GgmlType::Q5K => false,
        // Q6_K/Q8_0 are wins for fully-quantized expert stacks and Q4 mixed
        // layouts, but Q5 gate+up mixed with Q6/Q8 down still regresses a bit.
        GgmlType::Q6K | GgmlType::Q8_0 => {
            !(gate_dtype == GgmlType::Q5K && up_dtype == GgmlType::Q5K)
        }
        _ => false,
    }
}

fn qwen35_selected_fused_silu_down_q5_k_enabled() -> bool {
    !matches!(
        env_flag_override("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K"),
        Some(false)
    )
}

fn qwen35_selected_fused_silu_down_q5_k_slots8_enabled() -> bool {
    !matches!(
        env_flag_override("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_SLOTS8"),
        Some(false)
    )
}

fn qwen35_selected_fused_silu_down_q5_k_nr2_enabled() -> bool {
    matches!(
        env_flag_override("AX_QWEN35_SELECTED_FUSED_SILU_DOWN_Q5K_NR2"),
        Some(true)
    )
}

fn qwen35_selected_q4_k_matvec_enabled() -> bool {
    !matches!(
        env_flag_override("AX_QWEN35_SELECTED_Q4K_MATVEC"),
        Some(false)
    )
}

fn qwen35_selected_q5_k_matvec_enabled() -> bool {
    matches!(
        env_flag_override("AX_QWEN35_SELECTED_Q5K_MATVEC"),
        Some(true)
    )
}

fn qwen35_selected_pair_q4_k_matvec_enabled() -> bool {
    !matches!(
        env_flag_override("AX_QWEN35_SELECTED_PAIR_Q4K_MATVEC"),
        Some(false)
    )
}

fn qwen35_selected_pair_q5_k_matvec_enabled_for_layout(
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    match env_flag_override("AX_QWEN35_SELECTED_PAIR_Q5K_MATVEC") {
        Some(value) => value,
        None => {
            gate_dtype == GgmlType::Q5K && up_dtype == GgmlType::Q5K && down_dtype == GgmlType::Q6K
        }
    }
}

fn qwen35_selected_single_token_gate_up_dtype_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
    )
}

fn qwen35_selected_single_token_gate_up_path_supported(
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    qwen35_selected_single_token_gate_up_dtype_supported(gate_dtype)
        && qwen35_selected_single_token_gate_up_dtype_supported(up_dtype)
        && matches!(
            down_dtype,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
        )
}

fn qwen35_selected_single_token_down_supported(down_dtype: GgmlType) -> bool {
    matches!(
        down_dtype,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
    )
}

fn qwen35_selected_single_token_down_falls_back_to_mul_mat_id(down_dtype: GgmlType) -> bool {
    matches!(down_dtype, GgmlType::Q6K | GgmlType::Q8_0)
}

fn qwen35_selected_single_token_default_enabled(
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    if !qwen35_selected_single_token_gate_up_path_supported(gate_dtype, up_dtype, down_dtype) {
        return false;
    }
    if matches!(down_dtype, GgmlType::Q4K | GgmlType::Q5K) {
        return true;
    }
    if gate_dtype == GgmlType::Q5K && up_dtype == GgmlType::Q5K && down_dtype == GgmlType::Q6K {
        return qwen35_selected_pair_q5_k_matvec_enabled_for_layout(
            gate_dtype, up_dtype, down_dtype,
        );
    }
    if qwen35_selected_weighted_down_enabled(gate_dtype, up_dtype, down_dtype) {
        return true;
    }
    gate_dtype == GgmlType::Q4K
        && up_dtype == GgmlType::Q4K
        && qwen35_selected_pair_q4_k_matvec_enabled()
}

fn qwen35_shared_gate_inp_fused_enabled() -> bool {
    !matches!(
        env_flag_override("AX_QWEN35_SHARED_GATE_INP_FUSED"),
        Some(false)
    )
}

fn qwen35_selected_expert_pair_enabled(
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    match env_flag_override("AX_QWEN35_SELECTED_EXPERT_PAIR") {
        Some(value) => value,
        None => {
            if gate_dtype != up_dtype
                || !matches!(
                    gate_dtype,
                    GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
                )
            {
                return false;
            }
            if down_dtype == GgmlType::Q5K && qwen35_selected_fused_silu_down_q5_k_enabled() {
                return gate_dtype == GgmlType::Q4K && qwen35_selected_pair_q4_k_matvec_enabled();
            }
            true
        }
    }
}

#[cfg(test)]
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
/// Resolve whether GPU KV cache should use f16 storage for this model.
///
/// Controlled by `AX_METAL_F16_KV_CACHE`:
/// - `1` / `true` / `on`  -> force enable
/// - `0` / `false` / `off` -> force disable
/// - `auto` or unset       -> enable when context length >= 256
pub fn metal_f16_kv_cache_enabled(context_len: usize) -> bool {
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
}

impl Backend for MetalBackend {
    fn note_qwen35_prepared_state_batch_kind(
        &self,
        kind: crate::kv::qwen35_kv::Qwen3_5PreparedRecurrentStateBatchKind,
    ) {
        match kind {
            crate::kv::qwen35_kv::Qwen3_5PreparedRecurrentStateBatchKind::CpuDirect => {
                self.ops
                    .record_qwen35_recurrent_batch_state_batch_cpu_direct();
            }
            crate::kv::qwen35_kv::Qwen3_5PreparedRecurrentStateBatchKind::CpuDirectMaterializedFromBackend => {
                self.ops
                    .record_qwen35_recurrent_batch_state_batch_cpu_direct_materialized_from_backend();
            }
            crate::kv::qwen35_kv::Qwen3_5PreparedRecurrentStateBatchKind::CpuGathered => {
                self.ops
                    .record_qwen35_recurrent_batch_state_batch_cpu_gathered();
            }
            crate::kv::qwen35_kv::Qwen3_5PreparedRecurrentStateBatchKind::CpuGatheredMaterializedFromBackend => {
                self.ops
                    .record_qwen35_recurrent_batch_state_batch_cpu_gathered_materialized_from_backend();
            }
        }
    }

    fn configure_for_fingerprint(&self, fingerprint: &ModelFingerprint) -> anyhow::Result<()> {
        let profile = KernelProfile::load(&fingerprint.model_name, &fingerprint.predominant_quant);
        self.ops
            .apply_runtime_policy(RuntimePolicy::from_kernel_profile_for_arch(
                profile,
                &fingerprint.architecture,
            ));
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
                GgmlType::Q5_0 => {
                    self.fused_matvec_q5_0(a_quant, b, c, m, k);
                    return;
                }
                GgmlType::Q5_1 => {
                    self.fused_matvec_q5_1(a_quant, b, c, m, k);
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
        let mut uses_quant_cache: Vec<bool> = Vec::with_capacity(ops.len());
        for (a_quant, dtype, _) in ops {
            let key = a_quant.as_ptr() as usize;
            weight_keys.push(key);
            if *dtype == GgmlType::F32 {
                let mut cache = self.weight_cache.lock().unwrap();
                cache
                    .entry(key)
                    .or_insert_with(|| create_mmap_weight_buffer_from_bytes(&self.device, a_quant));
                uses_quant_cache.push(false);
            } else {
                self.ensure_quant_weight_cached(key, a_quant);
                uses_quant_cache.push(true);
            }
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
        let dense_cache = self.weight_cache.lock().unwrap();
        let quant_cache = self.quant_weight_cache.lock().unwrap();
        let dispatch_config = self.ops.dequant_dispatch_config();
        self.device
            .execute_sync(|encoder| {
                for (i, (_, dtype, m)) in ops.iter().enumerate() {
                    let buf_a = if uses_quant_cache[i] {
                        quant_cache.get(&weight_keys[i]).unwrap()
                    } else {
                        dense_cache.get(&weight_keys[i]).unwrap()
                    };
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
        drop(quant_cache);
        drop(dense_cache);

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
        state_batch: &mut super::Qwen3_5RecurrentStateBatch<'_>,
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
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
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

    fn sync_qwen35_kv(&self, qwen_kv: &mut crate::kv::Qwen3_5Kv) {
        if qwen_kv.has_gpu_recurrent_state() {
            let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
            let recurrent_state_stride = qwen_kv.recurrent_state_len();
            for slot_idx in 0..qwen_kv.recurrent_slot_capacity() {
                if !qwen_kv.has_recurrent_slot(slot_idx) {
                    continue;
                }
                for layer_idx in 0..qwen_kv.layer_count() {
                    if !qwen_kv.is_recurrent_layer(layer_idx) {
                        continue;
                    }
                    let need_conv_sync = qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx);
                    let need_recurrent_sync =
                        qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx);
                    if !need_conv_sync && !need_recurrent_sync {
                        continue;
                    }
                    let Some((conv_buf, rec_buf)) =
                        qwen_kv.gpu_recurrent_buffers(slot_idx, layer_idx)
                    else {
                        continue;
                    };
                    let conv_copy = need_conv_sync.then(|| unsafe {
                        conv_buf.as_slice::<f32>()[..conv_state_stride].to_vec()
                    });
                    let rec_copy = need_recurrent_sync.then(|| unsafe {
                        rec_buf.as_slice::<f32>()[..recurrent_state_stride].to_vec()
                    });
                    if let Some(conv_state) = conv_copy.as_deref() {
                        let generation = qwen_kv.conv_state_generation(slot_idx, layer_idx);
                        qwen_kv.sync_conv_state_from_backend(
                            slot_idx, layer_idx, conv_state, generation,
                        );
                    }
                    if let Some(recurrent_state) = rec_copy.as_deref() {
                        let generation = qwen_kv.recurrent_state_generation(slot_idx, layer_idx);
                        qwen_kv.sync_recurrent_state_from_backend(
                            slot_idx,
                            layer_idx,
                            recurrent_state,
                            generation,
                        );
                    }
                }
            }
        }
        let slot_cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let kv_identity = qwen_kv as *const crate::kv::Qwen3_5Kv as usize;
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
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
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
                // Interleaved ordering: matches repeat_heads_into and
                // the GPU shader qwen35_prepare_multi_token_gdn_qk_f32.
                let dst_head = rep * group_count + src_head;
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
            // Interleaved ordering: matches repeat_heads_into and
            // the GPU shader qwen35_prepare_multi_token_gdn_qk_f32.
            let dst_head = rep * group_count + src_head;
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
    /// Get or create a cached Metal buffer for raw quantized weight data.
    ///
    /// Weight tensors from mmap'd GGUF files have stable pointer addresses,
    /// so we use the data pointer as the cache key. On first access the data
    /// is aliased via a no-copy Metal shared-mode buffer when possible;
    /// otherwise it falls back to a copied buffer. Subsequent calls return
    /// the cached buffer with zero overhead.
    fn get_quant_weight_buffer(
        &self,
        data: &[u8],
    ) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        let key = data.as_ptr() as usize;
        let mut cache = self.quant_weight_cache.lock().unwrap();
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
        let cache = self.get_quant_weight_buffer(a_quant);
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
        let cache = self.get_quant_weight_buffer(a_quant);
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

    fn fused_matvec_q5_0(&self, a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
        let y_bytes = m * std::mem::size_of::<f32>();
        let input_guard = self.prepare_input(x);
        let output_guard = self.prepare_output(y_bytes);

        let key = a_quant.as_ptr() as usize;
        let cache = self.get_quant_weight_buffer(a_quant);
        let buf_a = cache.get(&key).unwrap();

        self.device
            .execute_sync(|encoder| {
                self.dequant_kernels.encode_fused_matvec_q5_0(
                    encoder,
                    buf_a,
                    input_guard.0.as_ref().unwrap(),
                    output_guard.0.as_ref().unwrap(),
                    m as u32,
                    k as u32,
                );
                Ok(())
            })
            .expect("Metal fused Q5_0 matvec failed");

        drop(cache);

        unsafe {
            std::ptr::copy_nonoverlapping(
                output_guard.0.as_ref().unwrap().contents().as_ptr() as *const f32,
                y.as_mut_ptr(),
                m,
            );
        }
    }

    fn fused_matvec_q5_1(&self, a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
        let y_bytes = m * std::mem::size_of::<f32>();
        let input_guard = self.prepare_input(x);
        let output_guard = self.prepare_output(y_bytes);

        let key = a_quant.as_ptr() as usize;
        let cache = self.get_quant_weight_buffer(a_quant);
        let buf_a = cache.get(&key).unwrap();

        self.device
            .execute_sync(|encoder| {
                self.dequant_kernels.encode_fused_matvec_q5_1(
                    encoder,
                    buf_a,
                    input_guard.0.as_ref().unwrap(),
                    output_guard.0.as_ref().unwrap(),
                    m as u32,
                    k as u32,
                );
                Ok(())
            })
            .expect("Metal fused Q5_1 matvec failed");

        drop(cache);

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
        let cache = self.get_quant_weight_buffer(a_quant);
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
        let cache = self.get_quant_weight_buffer(a_quant);
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
    // Gemma4-specific
    pub v_equals_k: bool,
    pub layer_output_scale: Option<usize>,
    pub gemma4_moe_router_scale: Option<usize>,
    pub gemma4_moe_pre_ffw_norm_2: Option<usize>,
    pub gemma4_moe_post_ffw_norm_1: Option<usize>,
    pub gemma4_moe_post_ffw_norm_2: Option<usize>,
    pub gemma4_moe_expert_scales: Option<usize>,
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
    pub moe_expert_gate_dtype: Option<GgmlType>,
    pub moe_expert_up_dtype: Option<GgmlType>,
    pub moe_expert_down_dtype: Option<GgmlType>,
    pub moe_expert_gate_stride: Option<usize>,
    pub moe_expert_up_stride: Option<usize>,
    pub moe_expert_down_stride: Option<usize>,
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
    pub rope_freqs: Option<usize>,
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
    /// Fused gate_up output [N × n_expert_used × (2 * expert_inter_dim)].
    pub moe_fused_gate_up: Option<MetalBuffer>,
    /// Down matmul output [N × n_expert_used × dim].
    pub moe_down_out: Option<MetalBuffer>,
    /// Tokens per expert [n_expert] u32.
    pub moe_tpe: Option<MetalBuffer>,
    /// Routing index [n_expert × N] i32.
    pub moe_hids: Option<MetalBuffer>,
    /// Expert IDs [N × n_expert_used] i32.
    pub moe_expert_ids: Option<MetalBuffer>,
    /// Expert weights [N × n_expert_used] f32.
    pub moe_expert_weights: Option<MetalBuffer>,
    /// Active expert metadata: count + compact expert list [(n_expert + 1)] u32.
    pub moe_active_experts: Option<MetalBuffer>,
}

#[derive(Clone, Copy)]
pub(crate) struct MoeBatchScratchView {
    pub(crate) norm: *const MetalBuffer,
    pub(crate) router: *const MetalBuffer,
    pub(crate) accum: *const MetalBuffer,
    pub(crate) gate_out: *const MetalBuffer,
    pub(crate) up_out: *const MetalBuffer,
    pub(crate) down_out: *const MetalBuffer,
    pub(crate) matmul_in_f16: *const MetalBuffer,
    pub(crate) tpe: *const MetalBuffer,
    pub(crate) hids: *const MetalBuffer,
    pub(crate) active: *mut MetalBuffer,
    pub(crate) ids: *mut MetalBuffer,
    pub(crate) weights: *mut MetalBuffer,
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
            matmul_in_f16: &bs.matmul_in_f16 as *const _,
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
    /// Backend-local runtime policy for this model/backend instance.
    runtime_policy: RwLock<RuntimePolicy>,
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
    /// Dedicated recurrent decode scratch for single/multi-token GDN kernels.
    qwen35_recurrent_scratch_buffers:
        Mutex<FxHashMap<Qwen35RecurrentScratchBufferKey, Qwen35MetalRecurrentScratchBuffers>>,
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
    fn max_decode_dims(config: &ModelConfig) -> (usize, usize, usize, usize, usize) {
        let dim = config.embedding_dim as usize;
        let n_heads = config.n_heads as usize;
        let max_head_dim = config
            .gemma4_head_dim_global
            .unwrap_or(config.head_dim)
            .max(config.gemma4_head_dim_swa.unwrap_or(config.head_dim))
            as usize;
        let max_n_kv_heads = config
            .gemma4_n_kv_heads_swa
            .unwrap_or(config.n_kv_heads)
            .max(config.gemma4_n_kv_heads_global.unwrap_or(config.n_kv_heads))
            as usize;
        let q_dim = n_heads * max_head_dim;
        let kv_dim = max_n_kv_heads * max_head_dim;
        (dim, n_heads, max_head_dim, q_dim, kv_dim)
    }

    #[inline]
    fn quant_view_cache_key(raw_key: usize) -> usize {
        const QUANT_VIEW_TAG: usize = 1usize << (usize::BITS - 1);
        debug_assert_eq!(
            raw_key & QUANT_VIEW_TAG,
            0,
            "quant view cache key tag collided with raw pointer key space"
        );
        raw_key | QUANT_VIEW_TAG
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
        let runtime_policy = RuntimePolicy::resolved_defaults();
        Ok(Self {
            device,
            elementwise,
            dequant,
            attention,
            matmul,
            gdn,
            runtime_policy: RwLock::new(runtime_policy),
            f32_weight_cache: Mutex::new(FxHashMap::default()),
            fused_qkv_weight_cache: Mutex::new(FxHashMap::default()),
            precomputed_f16_weight_cache: Mutex::new(FxHashMap::default()),
            scratches: Mutex::new(None),
            batch_scratches: Mutex::new(None),
            qwen35_recurrent_slot_buffers: Mutex::new(FxHashMap::default()),
            qwen35_recurrent_scratch_buffers: Mutex::new(FxHashMap::default()),
            qwen35_qkv_handoff_scratch_buffers: Mutex::new(FxHashMap::default()),
            qwen35_recurrent_projection_scratch_buffers: Mutex::new(FxHashMap::default()),
            qwen35_batch_projection_scratch_buffers: Mutex::new(FxHashMap::default()),
            qwen35_batch_logits_scratch_buffers: Mutex::new(FxHashMap::default()),
            qwen35_recurrent_batch_perf: Qwen35RecurrentBatchPerfState::default(),
            cached_model_keys: Mutex::new(None),
            moe_quant_cache: Mutex::new(FxHashMap::default()),
        })
    }

    pub fn apply_runtime_policy(&self, runtime_policy: RuntimePolicy) {
        *self.runtime_policy.write().unwrap() = runtime_policy;
    }

    pub fn runtime_policy(&self) -> RuntimePolicy {
        self.runtime_policy.read().unwrap().clone()
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
        _attend_len: u32,
    ) -> (DequantDispatchConfig, AttentionDispatchConfig) {
        let policy = self.runtime_policy.read().unwrap();
        (
            policy.dequant_dispatch_config(),
            policy.attention_dispatch_config(),
        )
    }

    pub fn metal_batch_f16_io_enabled(&self) -> bool {
        self.runtime_policy
            .read()
            .unwrap()
            .batch_prefill_prefers_f16_io()
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

    pub fn metal_q8_batch_native_shape_enabled(&self, m: u32, n: u32, k: u32) -> bool {
        self.runtime_policy
            .read()
            .unwrap()
            .q8_batch_native_shape_enabled(m, n, k)
    }

    pub fn init_scratches(&self, config: &ModelConfig) {
        let mut guard = self.scratches.lock().unwrap();
        if guard.is_some() {
            return;
        }

        let (dim, n_heads, head_dim, q_dim, kv_dim) = Self::max_decode_dims(config);
        let inter_dim = config.intermediate_dim as usize;
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
        for buf in self.moe_quant_cache.lock().unwrap().values() {
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
        let mut guard = self.batch_scratches.lock().unwrap();
        if let Some(ref existing) = *guard
            && existing.batch_size >= n_tokens
        {
            return;
        }

        let (dim, _n_heads, _head_dim, q_dim, kv_dim) = Self::max_decode_dims(config);
        let inter_dim = config.intermediate_dim as usize;
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
            moe_fused_gate_up: None,
            moe_down_out: None,
            moe_tpe: None,
            moe_hids: None,
            moe_expert_ids: None,
            moe_expert_weights: None,
            moe_active_experts: None,
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
            bs.moe_fused_gate_up = Some(alloc_bytes(
                n_tokens * n_expert_used * 2 * expert_inter_dim * f4,
            ));
            bs.moe_down_out = Some(alloc_bytes(n_tokens * n_expert_used * dim * f4));
            bs.moe_tpe = Some(alloc_bytes(n_expert * u4));
            bs.moe_hids = Some(alloc_bytes(n_expert * n_tokens * i4));
            bs.moe_expert_ids = Some(alloc_bytes(n_tokens * n_expert_used * i4));
            bs.moe_expert_weights = Some(alloc_bytes(n_tokens * n_expert_used * f4));
            bs.moe_active_experts = Some(alloc_bytes((n_expert + 1) * u4));
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

    /// Ensure a precomputed dense f16 copy of an F32 weight buffer exists.
    ///
    /// The f32→f16 cast is encoded on the provided encoder, so this is safe
    /// to call from inside an `execute_sync` closure without nesting.
    fn ensure_precomputed_f32_f16_on_encoder(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        weight_buf: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let key = Self::quant_buf_key(weight_buf);
        if self
            .precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .contains_key(&key)
        {
            return Ok(());
        }

        let dense_f16 = MetalBuffer::new(
            self.device.device(),
            (m as usize) * (k as usize) * std::mem::size_of::<half::f16>(),
        )?;
        self.elementwise
            .encode_cast_f32_to_f16(encoder, weight_buf, &dense_f16, m * k);
        ax_engine_metal::barrier_buffers(encoder);
        self.precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .insert(key, dense_f16);
        tracing::info!(
            key,
            m,
            k,
            "Cached precomputed dense f16 weight (F32, on-encoder)"
        );
        Ok(())
    }

    /// Ensure a precomputed dense f16 copy exists for an F32 weight buffer.
    ///
    /// Unlike `ensure_precomputed_f32_f16_on_encoder`, this variant performs
    /// the cast eagerly on its own command buffer so first-use hot paths do
    /// not pay the conversion cost inside the measured encode.
    pub fn ensure_precomputed_f32_f16(
        &self,
        weight_buf: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        if self.has_precomputed_weight(weight_buf) {
            return Ok(());
        }

        self.device.execute_sync(|encoder| {
            self.ensure_precomputed_f32_f16_on_encoder(encoder, weight_buf, m, k)
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_token_major_batch_matmul_f32(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        weight_buf: &MetalBuffer,
        input_f32: &MetalBuffer,
        input_f16: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        // Use on-encoder precompute to avoid nested execute_sync when called
        // from inside another execute_sync closure (e.g. MoE FFN dispatch).
        self.ensure_precomputed_f32_f16_on_encoder(encoder, weight_buf, m, k)?;
        self.elementwise
            .encode_cast_f32_to_f16(encoder, input_f32, input_f16, n * k);
        ax_engine_metal::barrier_buffers(encoder);
        anyhow::ensure!(
            self.encode_precomputed_batch_if_available(
                encoder, weight_buf, input_f16, output, m, n, k
            ),
            "missing precomputed dense f16 cache for F32 token-major batch matmul"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_quant_matvec(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        dtype: GgmlType,
    ) -> anyhow::Result<()> {
        let config = self.dequant_dispatch_config();
        match dtype {
            GgmlType::F32 => self
                .matmul
                .encode_matvec(encoder, weights, input, output, m, k),
            GgmlType::Q5_0 => self
                .dequant
                .encode_fused_matvec_q5_0(encoder, weights, input, output, m, k),
            GgmlType::Q5_1 => self
                .dequant
                .encode_fused_matvec_q5_1(encoder, weights, input, output, m, k),
            GgmlType::Q4K => self.dequant.encode_fused_matvec_q4_k_with_config(
                encoder, weights, input, output, m, k, config,
            ),
            GgmlType::Q5K => self.dequant.encode_fused_matvec_q5_k_with_config(
                encoder, weights, input, output, m, k, config,
            ),
            GgmlType::Q6K => self.dequant.encode_fused_matvec_q6_k_with_config(
                encoder, weights, input, output, m, k, config,
            ),
            GgmlType::Q8_0 => self.dequant.encode_fused_matvec_q8_0_with_config(
                encoder, weights, input, output, m, k, config,
            ),
            _ => anyhow::bail!("unsupported Metal MoE matvec dtype: {dtype:?}"),
        }
        Ok(())
    }

    fn validate_moe_mul_mat_id_dtype(dtype: GgmlType, role: &str) -> anyhow::Result<()> {
        if matches!(
            dtype,
            GgmlType::Q4K
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8_0
        ) {
            Ok(())
        } else {
            anyhow::bail!(
                "Metal MoE mul_mat_id path only supports Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0 {role} weights today; got {dtype:?}",
            )
        }
    }

    fn validate_moe_selected_dtype(dtype: GgmlType, role: &str) -> anyhow::Result<()> {
        if matches!(
            dtype,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
        ) {
            Ok(())
        } else {
            anyhow::bail!(
                "Metal MoE selected path only supports Q4_K/Q5_K/Q6_K/Q8_0 {role} weights today; got {dtype:?}",
            )
        }
    }

    fn validate_moe_selected_weighted_dtype(dtype: GgmlType, role: &str) -> anyhow::Result<()> {
        if matches!(
            dtype,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
        ) {
            Ok(())
        } else {
            anyhow::bail!(
                "Metal MoE selected weighted path only supports Q4_K/Q5_K/Q6_K/Q8_0 {role} weights today; got {dtype:?}",
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
        allow_blocked_input_is_hid: bool,
        input_is_hid: bool,
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
                input_is_hid,
            ),
            GgmlType::Q5_0 => self.dequant.encode_moe_mul_mat_id_q5_0(
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
                input_is_hid,
            ),
            GgmlType::Q5_1 => self.dequant.encode_moe_mul_mat_id_q5_1(
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
                input_is_hid,
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
                input_is_hid,
            ),
            GgmlType::Q6K => self.dequant.encode_moe_mul_mat_id_q6_k(
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
                allow_blocked_input_is_hid,
                input_is_hid,
            ),
            GgmlType::Q8_0 => self.dequant.encode_moe_mul_mat_id_q8_0(
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
                allow_blocked_input_is_hid,
                input_is_hid,
            ),
            _ => anyhow::bail!(
                "unsupported Metal MoE mul_mat_id dtype for routed experts: {dtype:?}"
            ),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_moe_mul_mat_selected_single_token(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output: &MetalBuffer,
        dtype: GgmlType,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
        input_is_slot_major: bool,
    ) -> anyhow::Result<()> {
        Self::validate_moe_selected_dtype(dtype, "selected expert")?;
        match dtype {
            GgmlType::Q4K => self.dequant.encode_moe_mul_mat_selected_q4_k(
                encoder,
                weights,
                input,
                selected_experts,
                output,
                m,
                k,
                n_selected,
                weight_stride,
                input_is_slot_major,
                qwen35_selected_q4_k_matvec_enabled(),
            ),
            GgmlType::Q5K => self.dequant.encode_moe_mul_mat_selected_q5_k(
                encoder,
                weights,
                input,
                selected_experts,
                output,
                m,
                k,
                n_selected,
                weight_stride,
                input_is_slot_major,
                qwen35_selected_q5_k_matvec_enabled(),
            ),
            GgmlType::Q6K => self.dequant.encode_moe_mul_mat_selected_q6_k(
                encoder,
                weights,
                input,
                selected_experts,
                output,
                m,
                k,
                n_selected,
                weight_stride,
                input_is_slot_major,
                self.dequant_dispatch_config(),
            ),
            GgmlType::Q8_0 => self.dequant.encode_moe_mul_mat_selected_q8_0(
                encoder,
                weights,
                input,
                selected_experts,
                output,
                m,
                k,
                n_selected,
                weight_stride,
                input_is_slot_major,
                self.dequant_dispatch_config(),
            ),
            _ => unreachable!("validated selected expert dtype"),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_moe_mul_mat_selected_single_token_pair(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        weights0: &MetalBuffer,
        weights1: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output0: &MetalBuffer,
        output1: &MetalBuffer,
        dtype: GgmlType,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride0: u32,
        weight_stride1: u32,
        input_is_slot_major: bool,
        use_q5_pair_matvec: bool,
    ) -> anyhow::Result<()> {
        Self::validate_moe_selected_dtype(dtype, "selected expert pair")?;
        match dtype {
            GgmlType::Q4K => self.dequant.encode_moe_mul_mat_selected_pair_q4_k(
                encoder,
                weights0,
                weights1,
                input,
                selected_experts,
                output0,
                output1,
                m,
                k,
                n_selected,
                weight_stride0,
                weight_stride1,
                input_is_slot_major,
                qwen35_selected_pair_q4_k_matvec_enabled(),
            ),
            GgmlType::Q5K => self.dequant.encode_moe_mul_mat_selected_pair_q5_k(
                encoder,
                weights0,
                weights1,
                input,
                selected_experts,
                output0,
                output1,
                m,
                k,
                n_selected,
                weight_stride0,
                weight_stride1,
                input_is_slot_major,
                use_q5_pair_matvec,
            ),
            GgmlType::Q6K => self.dequant.encode_moe_mul_mat_selected_pair_q6_k(
                encoder,
                weights0,
                weights1,
                input,
                selected_experts,
                output0,
                output1,
                m,
                k,
                n_selected,
                weight_stride0,
                weight_stride1,
                input_is_slot_major,
                self.dequant_dispatch_config(),
            ),
            GgmlType::Q8_0 => self.dequant.encode_moe_mul_mat_selected_pair_q8_0(
                encoder,
                weights0,
                weights1,
                input,
                selected_experts,
                output0,
                output1,
                m,
                k,
                n_selected,
                weight_stride0,
                weight_stride1,
                input_is_slot_major,
                self.dequant_dispatch_config(),
            ),
            _ => unreachable!("validated selected expert pair dtype"),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_moe_mul_mat_selected_single_token_weighted(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        expert_weights: &MetalBuffer,
        output: &MetalBuffer,
        dtype: GgmlType,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
    ) -> anyhow::Result<()> {
        Self::validate_moe_selected_weighted_dtype(dtype, "selected weighted expert")?;
        match dtype {
            GgmlType::Q4K => self.dequant.encode_moe_mul_mat_selected_weighted_q4_k(
                encoder,
                weights,
                input,
                selected_experts,
                expert_weights,
                output,
                m,
                k,
                n_selected,
                weight_stride,
            ),
            GgmlType::Q5K => self.dequant.encode_moe_mul_mat_selected_weighted_q5_k(
                encoder,
                weights,
                input,
                selected_experts,
                expert_weights,
                output,
                m,
                k,
                n_selected,
                weight_stride,
            ),
            GgmlType::Q6K => self.dequant.encode_moe_mul_mat_selected_weighted_q6_k(
                encoder,
                weights,
                input,
                selected_experts,
                expert_weights,
                output,
                m,
                k,
                n_selected,
                weight_stride,
                self.dequant_dispatch_config(),
            ),
            GgmlType::Q8_0 => self.dequant.encode_moe_mul_mat_selected_weighted_q8_0(
                encoder,
                weights,
                input,
                selected_experts,
                expert_weights,
                output,
                m,
                k,
                n_selected,
                weight_stride,
                self.dequant_dispatch_config(),
            ),
            _ => unreachable!("validated selected weighted expert dtype"),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_moe_fused_silu_down_selected_single_token_q5_k(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        weights: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        selected_experts: &MetalBuffer,
        expert_weights: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
    ) -> anyhow::Result<()> {
        let use_slots8_kernel =
            n_selected == 8 && qwen35_selected_fused_silu_down_q5_k_slots8_enabled();
        let use_nr2_kernel = qwen35_selected_fused_silu_down_q5_k_nr2_enabled();
        self.dequant
            .encode_moe_fused_silu_down_selected_weighted_q5_k(
                encoder,
                weights,
                gate,
                up,
                selected_experts,
                expert_weights,
                output,
                m,
                k,
                n_selected,
                weight_stride,
                use_slots8_kernel,
                use_nr2_kernel,
            );
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
        self.encode_moe_ffn_gpu_resident_cached_with_policy(
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
            explicit_barriers,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_ffn_gpu_resident_cached_with_policy(
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
        allow_blocked_q6q8_down: bool,
    ) -> anyhow::Result<()> {
        let scratch = self.moe_batch_scratch_view()?;
        self.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
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
            allow_blocked_q6q8_down,
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
        self.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
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
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
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
        allow_blocked_q6q8_down: bool,
    ) -> anyhow::Result<()> {
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
        let matmul_in_f16 = unsafe { &*scratch.matmul_in_f16 };
        let tpe_gpu = unsafe { &*scratch.tpe };
        let hids_gpu = unsafe { &*scratch.hids };
        let active_buf = unsafe { &mut *scratch.active };
        let ids_gpu = unsafe { &mut *scratch.ids };
        let weights_gpu = unsafe { &mut *scratch.weights };
        let pre_hidden_debug = if std::env::var("AX_DEBUG_MOE_NATIVE").is_ok() && n_tokens == 1 {
            Some(unsafe { hidden_gpu.as_slice::<f32>()[..4.min(dim)].to_vec() })
        } else {
            None
        };
        let mut sb = ax_engine_metal::SmartBarrier::new(encoder);
        macro_rules! sb_pre {
            ($reads:expr, $writes:expr) => {
                if explicit_barriers {
                    sb.pre_dispatch($reads, $writes);
                }
            };
        }
        macro_rules! sb_post {
            ($reads:expr, $writes:expr) => {
                if explicit_barriers {
                    sb.post_dispatch($reads, $writes);
                }
            };
        }
        macro_rules! sb_flush {
            () => {
                if explicit_barriers {
                    sb.flush();
                }
            };
        }

        // This scratch buffer is reused across resident-MoE layers. When decode
        // coalesces multiple recurrent layers into one command buffer, CPU-side
        // writes here would all happen before GPU execution starts, so later
        // layers would not see an in-sequence reset. Encode the zeroing on GPU
        // so every resident tail gets its own accumulator reset in command order.
        sb_pre!(&[accum_gpu], &[accum_gpu]);
        self.elementwise.encode_gen_scale(
            encoder,
            accum_gpu,
            accum_gpu,
            0.0,
            (n_tokens * dim) as u32,
        );
        sb_post!(&[accum_gpu], &[accum_gpu]);

        let nt = n_tokens as u32;
        let ne = n_expert as u32;
        let neu = n_expert_used as u32;
        let m_inter = expert_inter_dim as u32;
        let m_dim = dim as u32;
        let selected_expert_single_token_supported = n_tokens == 1
            && qwen35_selected_single_token_gate_up_path_supported(
                gate_dtype, up_dtype, down_dtype,
            );
        let selected_expert_single_token_default_enabled = n_tokens == 1
            && qwen35_selected_single_token_default_enabled(gate_dtype, up_dtype, down_dtype);
        let use_selected_expert_single_token_path =
            match env_flag_override("AX_QWEN35_SELECTED_EXPERT_SINGLE_TOKEN") {
                Some(enabled) => enabled && selected_expert_single_token_supported,
                None => selected_expert_single_token_default_enabled,
            };

        sb_pre!(&[hidden_gpu], &[norm_gpu]);
        self.elementwise.encode_rms_norm_out_batch(
            encoder,
            hidden_gpu,
            ffn_norm_w_buf,
            norm_gpu,
            dim as u32,
            n_tokens as u32,
            eps,
        );
        sb_post!(&[hidden_gpu], &[norm_gpu]);

        if n_tokens == 1 {
            sb_pre!(&[norm_gpu], &[router_gpu]);
            self.encode_quant_matvec(
                encoder,
                router_buf,
                norm_gpu,
                router_gpu,
                n_expert as u32,
                dim as u32,
                router_dtype,
            )?;
            sb_post!(&[norm_gpu], &[router_gpu]);
        } else if router_dtype == GgmlType::F32 {
            sb_pre!(&[norm_gpu], &[router_gpu]);
            self.encode_token_major_batch_matmul_f32(
                encoder,
                router_buf,
                norm_gpu,
                matmul_in_f16,
                router_gpu,
                n_expert as u32,
                nt,
                m_dim,
            )?;
            sb_post!(&[norm_gpu], &[router_gpu]);
        } else {
            sb_pre!(&[norm_gpu], &[router_gpu]);
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
            sb_post!(&[norm_gpu], &[router_gpu]);
        }
        sb_pre!(&[router_gpu], &[ids_gpu, weights_gpu]);

        self.elementwise.encode_moe_softmax_topk(
            encoder,
            router_gpu,
            ids_gpu,
            weights_gpu,
            n_tokens as u32,
            n_expert as u32,
            n_expert_used as u32,
        );
        sb_post!(&[router_gpu], &[ids_gpu, weights_gpu]);

        let profile_skip_routed_expert = matches!(
            env_flag_override("AX_QWEN35_PROFILE_SKIP_ROUTED_EXPERT"),
            Some(true)
        );
        if !profile_skip_routed_expert && use_selected_expert_single_token_path {
            let use_selected_expert_pair =
                qwen35_selected_expert_pair_enabled(gate_dtype, up_dtype, down_dtype);
            let use_q5_pair_matvec = qwen35_selected_pair_q5_k_matvec_enabled_for_layout(
                gate_dtype, up_dtype, down_dtype,
            );
            if use_selected_expert_pair {
                sb_pre!(&[norm_gpu, ids_gpu], &[gate_out_gpu, up_out_gpu]);
                self.encode_moe_mul_mat_selected_single_token_pair(
                    encoder,
                    gate_buf,
                    up_buf,
                    norm_gpu,
                    ids_gpu,
                    gate_out_gpu,
                    up_out_gpu,
                    gate_dtype,
                    m_inter,
                    m_dim,
                    neu,
                    blocks_per_expert_gate,
                    blocks_per_expert_up,
                    false,
                    use_q5_pair_matvec,
                )?;
                sb_post!(&[norm_gpu, ids_gpu], &[gate_out_gpu, up_out_gpu]);
            } else {
                sb_pre!(&[norm_gpu, ids_gpu], &[gate_out_gpu]);
                self.encode_moe_mul_mat_selected_single_token(
                    encoder,
                    gate_buf,
                    norm_gpu,
                    ids_gpu,
                    gate_out_gpu,
                    gate_dtype,
                    m_inter,
                    m_dim,
                    neu,
                    blocks_per_expert_gate,
                    false,
                )?;
                sb_post!(&[norm_gpu, ids_gpu], &[gate_out_gpu]);
                sb_pre!(&[norm_gpu, ids_gpu], &[up_out_gpu]);
                self.encode_moe_mul_mat_selected_single_token(
                    encoder,
                    up_buf,
                    norm_gpu,
                    ids_gpu,
                    up_out_gpu,
                    up_dtype,
                    m_inter,
                    m_dim,
                    neu,
                    blocks_per_expert_up,
                    false,
                )?;
                sb_post!(&[norm_gpu, ids_gpu], &[up_out_gpu]);
            }

            let profile_skip_selected_down = matches!(
                env_flag_override("AX_QWEN35_PROFILE_SKIP_SELECTED_DOWN"),
                Some(true)
            );
            let selected_down_single_token_supported =
                qwen35_selected_single_token_down_supported(down_dtype);
            let selected_down_falls_back_to_mul_mat_id =
                qwen35_selected_single_token_down_falls_back_to_mul_mat_id(down_dtype);
            let use_selected_weighted_down =
                qwen35_selected_weighted_down_enabled(gate_dtype, up_dtype, down_dtype);
            let use_selected_fused_silu_down_q5_k =
                down_dtype == GgmlType::Q5K && qwen35_selected_fused_silu_down_q5_k_enabled();
            if profile_skip_selected_down {
                sb_pre!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
                self.elementwise.encode_silu_elementwise_mul_batch(
                    encoder,
                    gate_out_gpu,
                    up_out_gpu,
                    m_inter,
                    nt * neu,
                );
                sb_post!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
                // diagnostic only: isolates routed gate/up + SiLU cost
            } else if use_selected_fused_silu_down_q5_k {
                sb_pre!(
                    &[gate_out_gpu, up_out_gpu, ids_gpu, weights_gpu],
                    &[accum_gpu]
                );
                self.encode_moe_fused_silu_down_selected_single_token_q5_k(
                    encoder,
                    down_buf,
                    gate_out_gpu,
                    up_out_gpu,
                    ids_gpu,
                    weights_gpu,
                    accum_gpu,
                    m_dim,
                    m_inter,
                    neu,
                    blocks_per_expert_down,
                )?;
                sb_post!(
                    &[gate_out_gpu, up_out_gpu, ids_gpu, weights_gpu],
                    &[accum_gpu]
                );
            } else if use_selected_weighted_down {
                sb_pre!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
                self.elementwise.encode_silu_elementwise_mul_batch(
                    encoder,
                    gate_out_gpu,
                    up_out_gpu,
                    m_inter,
                    nt * neu,
                );
                sb_post!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
                sb_pre!(&[gate_out_gpu, ids_gpu, weights_gpu], &[accum_gpu]);
                self.encode_moe_mul_mat_selected_single_token_weighted(
                    encoder,
                    down_buf,
                    gate_out_gpu,
                    ids_gpu,
                    weights_gpu,
                    accum_gpu,
                    down_dtype,
                    m_dim,
                    m_inter,
                    neu,
                    blocks_per_expert_down,
                )?;
                sb_post!(&[gate_out_gpu, ids_gpu, weights_gpu], &[accum_gpu]);
            } else if selected_down_falls_back_to_mul_mat_id {
                sb_pre!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
                self.elementwise.encode_silu_elementwise_mul_batch(
                    encoder,
                    gate_out_gpu,
                    up_out_gpu,
                    m_inter,
                    nt * neu,
                );
                sb_post!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
                // Mixed routed-expert layouts such as Qwen3-Coder's Q4/Q5
                // gate+up with Q6/Q8 down still benefit from the selected
                // gate/up path. Reuse the generic routed down kernel for the
                // final projection instead of disabling the whole fast path.
                sb_pre!(&[ids_gpu], &[tpe_gpu, hids_gpu, active_buf]);
                self.dequant.encode_moe_map0(
                    encoder,
                    ids_gpu,
                    tpe_gpu,
                    hids_gpu,
                    active_buf,
                    n_tokens as u32,
                    n_expert_used as u32,
                    n_expert as u32,
                );
                sb_post!(&[ids_gpu], &[tpe_gpu, hids_gpu, active_buf]);
                let max_active_experts = ne.min(nt.saturating_mul(neu));
                sb_pre!(
                    &[gate_out_gpu, tpe_gpu, hids_gpu, active_buf],
                    &[down_out_gpu]
                );
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
                    max_active_experts,
                    allow_blocked_q6q8_down,
                    true,
                )?;
                sb_post!(
                    &[gate_out_gpu, tpe_gpu, hids_gpu, active_buf],
                    &[down_out_gpu]
                );
                sb_pre!(&[down_out_gpu, weights_gpu], &[accum_gpu]);
                self.elementwise.encode_moe_weighted_reduce_slots(
                    encoder,
                    down_out_gpu,
                    weights_gpu,
                    accum_gpu,
                    m_dim,
                    nt,
                    neu,
                );
                sb_post!(&[down_out_gpu, weights_gpu], &[accum_gpu]);
            } else if selected_down_single_token_supported {
                sb_pre!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
                self.elementwise.encode_silu_elementwise_mul_batch(
                    encoder,
                    gate_out_gpu,
                    up_out_gpu,
                    m_inter,
                    nt * neu,
                );
                sb_post!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
                sb_pre!(&[gate_out_gpu, ids_gpu], &[down_out_gpu]);
                self.encode_moe_mul_mat_selected_single_token(
                    encoder,
                    down_buf,
                    gate_out_gpu,
                    ids_gpu,
                    down_out_gpu,
                    down_dtype,
                    m_dim,
                    m_inter,
                    neu,
                    blocks_per_expert_down,
                    true,
                )?;
                sb_post!(&[gate_out_gpu, ids_gpu], &[down_out_gpu]);
                sb_pre!(&[down_out_gpu, weights_gpu], &[accum_gpu]);
                self.elementwise.encode_moe_weighted_reduce_slots(
                    encoder,
                    down_out_gpu,
                    weights_gpu,
                    accum_gpu,
                    m_dim,
                    nt,
                    neu,
                );
                sb_post!(&[down_out_gpu, weights_gpu], &[accum_gpu]);
            } else {
                anyhow::bail!(
                    "unsupported routed down dtype {:?} for selected single-token MoE path",
                    down_dtype
                );
            }
        } else if !profile_skip_routed_expert {
            let use_selected_weighted_down = n_tokens == 1
                && qwen35_selected_weighted_down_enabled(gate_dtype, up_dtype, down_dtype);
            let max_active_experts = ne.min(nt.saturating_mul(neu));
            sb_pre!(&[ids_gpu], &[tpe_gpu, hids_gpu, active_buf]);
            self.dequant.encode_moe_map0(
                encoder,
                ids_gpu,
                tpe_gpu,
                hids_gpu,
                active_buf,
                n_tokens as u32,
                n_expert_used as u32,
                n_expert as u32,
            );
            sb_post!(&[ids_gpu], &[tpe_gpu, hids_gpu, active_buf]);
            sb_pre!(&[norm_gpu, tpe_gpu, hids_gpu, active_buf], &[gate_out_gpu]);
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
                max_active_experts,
                false,
                false,
            )?;
            sb_post!(&[norm_gpu, tpe_gpu, hids_gpu, active_buf], &[gate_out_gpu]);
            sb_pre!(&[norm_gpu, tpe_gpu, hids_gpu, active_buf], &[up_out_gpu]);
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
                max_active_experts,
                false,
                false,
            )?;
            sb_post!(&[norm_gpu, tpe_gpu, hids_gpu, active_buf], &[up_out_gpu]);

            sb_pre!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
            self.elementwise.encode_silu_elementwise_mul_batch(
                encoder,
                gate_out_gpu,
                up_out_gpu,
                m_inter,
                nt * neu,
            );
            sb_post!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);

            if use_selected_weighted_down {
                sb_pre!(&[gate_out_gpu, ids_gpu, weights_gpu], &[accum_gpu]);
                self.encode_moe_mul_mat_selected_single_token_weighted(
                    encoder,
                    down_buf,
                    gate_out_gpu,
                    ids_gpu,
                    weights_gpu,
                    accum_gpu,
                    down_dtype,
                    m_dim,
                    m_inter,
                    neu,
                    blocks_per_expert_down,
                )?;
                sb_post!(&[gate_out_gpu, ids_gpu, weights_gpu], &[accum_gpu]);
            } else {
                sb_pre!(
                    &[gate_out_gpu, tpe_gpu, hids_gpu, active_buf],
                    &[down_out_gpu]
                );
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
                    max_active_experts,
                    allow_blocked_q6q8_down,
                    true,
                )?;
                sb_post!(
                    &[gate_out_gpu, tpe_gpu, hids_gpu, active_buf],
                    &[down_out_gpu]
                );

                sb_pre!(&[down_out_gpu, weights_gpu], &[accum_gpu]);
                self.elementwise.encode_moe_weighted_reduce_slots(
                    encoder,
                    down_out_gpu,
                    weights_gpu,
                    accum_gpu,
                    m_dim,
                    nt,
                    neu,
                );
                sb_post!(&[down_out_gpu, weights_gpu], &[accum_gpu]);
            }
        }

        let profile_skip_shared_expert = matches!(
            env_flag_override("AX_QWEN35_PROFILE_SKIP_SHARED_EXPERT"),
            Some(true)
        );
        if !profile_skip_shared_expert && let Some(se) = shared_expert {
            let se_inter = se.inter_dim as u32;
            let dequant_config = self.dequant_dispatch_config();
            let single_token = n_tokens == 1;
            let encoded_shared_pair = if single_token {
                sb_pre!(&[norm_gpu], &[gate_out_gpu, up_out_gpu]);
                encode_dequant_matvec_pair_with_config(
                    self,
                    encoder,
                    se.gate,
                    se.up,
                    norm_gpu,
                    gate_out_gpu,
                    up_out_gpu,
                    se_inter,
                    m_dim,
                    se.dtype,
                    se.dtype,
                    dequant_config,
                    true,
                )
            } else {
                false
            };
            if encoded_shared_pair {
                sb_post!(&[norm_gpu], &[gate_out_gpu, up_out_gpu]);
            }

            if encoded_shared_pair {
                // pair kernel already wrote gate_out/up_out
            } else if single_token {
                sb_pre!(&[norm_gpu], &[gate_out_gpu]);
                self.encode_quant_matvec(
                    encoder,
                    se.gate,
                    norm_gpu,
                    gate_out_gpu,
                    se_inter,
                    m_dim,
                    se.dtype,
                )?;
                sb_post!(&[norm_gpu], &[gate_out_gpu]);
                sb_pre!(&[norm_gpu], &[up_out_gpu]);
                self.encode_quant_matvec(
                    encoder, se.up, norm_gpu, up_out_gpu, se_inter, m_dim, se.dtype,
                )?;
                sb_post!(&[norm_gpu], &[up_out_gpu]);
            } else if se.dtype == GgmlType::Q8_0 {
                sb_pre!(&[norm_gpu], &[matmul_in_f16]);
                self.elementwise.encode_cast_f32_to_f16(
                    encoder,
                    norm_gpu,
                    matmul_in_f16,
                    m_dim * nt,
                );
                sb_post!(&[norm_gpu], &[matmul_in_f16]);
                sb_pre!(&[matmul_in_f16], &[gate_out_gpu]);
                self.dequant.encode_fused_batch_q8_0_f16in_with_config(
                    encoder,
                    se.gate,
                    matmul_in_f16,
                    gate_out_gpu,
                    se_inter,
                    nt,
                    m_dim,
                    dequant_config,
                );
                sb_post!(&[matmul_in_f16], &[gate_out_gpu]);
                sb_pre!(&[matmul_in_f16], &[up_out_gpu]);
                self.dequant.encode_fused_batch_q8_0_f16in_with_config(
                    encoder,
                    se.up,
                    matmul_in_f16,
                    up_out_gpu,
                    se_inter,
                    nt,
                    m_dim,
                    dequant_config,
                );
                sb_post!(&[matmul_in_f16], &[up_out_gpu]);
            } else if se.dtype == GgmlType::F32 {
                sb_pre!(&[norm_gpu], &[gate_out_gpu]);
                self.encode_token_major_batch_matmul_f32(
                    encoder,
                    se.gate,
                    norm_gpu,
                    matmul_in_f16,
                    gate_out_gpu,
                    se_inter,
                    nt,
                    m_dim,
                )?;
                sb_post!(&[norm_gpu], &[gate_out_gpu]);
                sb_pre!(&[norm_gpu], &[up_out_gpu]);
                self.encode_token_major_batch_matmul_f32(
                    encoder,
                    se.up,
                    norm_gpu,
                    matmul_in_f16,
                    up_out_gpu,
                    se_inter,
                    nt,
                    m_dim,
                )?;
                sb_post!(&[norm_gpu], &[up_out_gpu]);
            } else {
                sb_pre!(&[norm_gpu], &[gate_out_gpu]);
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
                sb_post!(&[norm_gpu], &[gate_out_gpu]);
                sb_pre!(&[norm_gpu], &[up_out_gpu]);
                self.encode_quant_batch_matmul(
                    encoder, se.up, norm_gpu, up_out_gpu, se_inter, nt, m_dim, se.dtype,
                )?;
                sb_post!(&[norm_gpu], &[up_out_gpu]);
            }

            let encoded_shared_down = if single_token {
                sb_pre!(&[gate_out_gpu, up_out_gpu], &[down_out_gpu]);
                encode_dequant_silu_down_matvec_with_config(
                    self,
                    encoder,
                    se.down,
                    gate_out_gpu,
                    up_out_gpu,
                    down_out_gpu,
                    m_dim,
                    se_inter,
                    se.dtype,
                    dequant_config,
                    true,
                )
            } else {
                false
            };
            if encoded_shared_down {
                sb_post!(&[gate_out_gpu, up_out_gpu], &[down_out_gpu]);
            }

            if !encoded_shared_down {
                sb_pre!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
                self.elementwise.encode_silu_elementwise_mul_batch(
                    encoder,
                    gate_out_gpu,
                    up_out_gpu,
                    se_inter,
                    nt,
                );
                sb_post!(&[gate_out_gpu, up_out_gpu], &[gate_out_gpu]);
            }

            if encoded_shared_down {
                // fused down kernel already wrote down_out
            } else if single_token {
                sb_pre!(&[gate_out_gpu], &[down_out_gpu]);
                self.encode_quant_matvec(
                    encoder,
                    se.down,
                    gate_out_gpu,
                    down_out_gpu,
                    m_dim,
                    se_inter,
                    se.dtype,
                )?;
                sb_post!(&[gate_out_gpu], &[down_out_gpu]);
            } else if se.dtype == GgmlType::Q8_0 {
                sb_pre!(&[gate_out_gpu], &[matmul_in_f16]);
                self.elementwise.encode_cast_f32_to_f16(
                    encoder,
                    gate_out_gpu,
                    matmul_in_f16,
                    se_inter * nt,
                );
                sb_post!(&[gate_out_gpu], &[matmul_in_f16]);
                sb_pre!(&[matmul_in_f16], &[down_out_gpu]);
                self.dequant.encode_fused_batch_q8_0_f16in_with_config(
                    encoder,
                    se.down,
                    matmul_in_f16,
                    down_out_gpu,
                    m_dim,
                    nt,
                    se_inter,
                    dequant_config,
                );
                sb_post!(&[matmul_in_f16], &[down_out_gpu]);
            } else if se.dtype == GgmlType::F32 {
                sb_pre!(&[gate_out_gpu], &[down_out_gpu]);
                self.encode_token_major_batch_matmul_f32(
                    encoder,
                    se.down,
                    gate_out_gpu,
                    matmul_in_f16,
                    down_out_gpu,
                    m_dim,
                    nt,
                    se_inter,
                )?;
                sb_post!(&[gate_out_gpu], &[down_out_gpu]);
            } else {
                sb_pre!(&[gate_out_gpu], &[down_out_gpu]);
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
                sb_post!(&[gate_out_gpu], &[down_out_gpu]);
            }

            if let Some(gate_inp) = se.gate_inp {
                let profile_skip_shared_gate_inp = matches!(
                    env_flag_override("AX_QWEN35_PROFILE_SKIP_SHARED_GATE_INP"),
                    Some(true)
                );
                if !profile_skip_shared_gate_inp {
                    let gate_inp_dtype = se
                        .gate_inp_dtype
                        .ok_or_else(|| anyhow::anyhow!("missing shared expert gate input dtype"))?;
                    let use_fused_shared_gate_inp = single_token
                        && se.gate_inp_rows == 1
                        && gate_inp_dtype == GgmlType::F32
                        && qwen35_shared_gate_inp_fused_enabled();
                    if use_fused_shared_gate_inp {
                        sb_pre!(&[norm_gpu, down_out_gpu], &[down_out_gpu]);
                        self.elementwise.encode_dense_row_dot_sigmoid_mul_inplace(
                            encoder,
                            gate_inp,
                            norm_gpu,
                            down_out_gpu,
                            m_dim,
                            m_dim,
                        );
                        sb_post!(&[norm_gpu, down_out_gpu], &[down_out_gpu]);
                    } else if n_tokens == 1 {
                        sb_pre!(&[norm_gpu], &[router_gpu]);
                        self.encode_quant_matvec(
                            encoder,
                            gate_inp,
                            norm_gpu,
                            router_gpu,
                            se.gate_inp_rows as u32,
                            m_dim,
                            gate_inp_dtype,
                        )?;
                        sb_post!(&[norm_gpu], &[router_gpu]);
                    } else if gate_inp_dtype == GgmlType::F32 {
                        sb_pre!(&[norm_gpu], &[router_gpu]);
                        self.encode_token_major_batch_matmul_f32(
                            encoder,
                            gate_inp,
                            norm_gpu,
                            matmul_in_f16,
                            router_gpu,
                            se.gate_inp_rows as u32,
                            nt,
                            m_dim,
                        )?;
                        sb_post!(&[norm_gpu], &[router_gpu]);
                    } else {
                        sb_pre!(&[norm_gpu], &[router_gpu]);
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
                        sb_post!(&[norm_gpu], &[router_gpu]);
                    }
                    if use_fused_shared_gate_inp {
                        // fused kernel already applied sigmoid(gate_inp) to down_out
                    } else if se.gate_inp_rows == dim {
                        sb_pre!(&[router_gpu, down_out_gpu], &[down_out_gpu]);
                        self.elementwise.encode_sigmoid_elementwise_mul(
                            encoder,
                            router_gpu,
                            down_out_gpu,
                            m_dim * nt,
                        );
                        sb_post!(&[router_gpu, down_out_gpu], &[down_out_gpu]);
                    } else {
                        anyhow::ensure!(
                            se.gate_inp_rows == 1,
                            "shared expert gate width {} is not supported by the resident Metal MoE path (expected 1 or {})",
                            se.gate_inp_rows,
                            dim,
                        );
                        if single_token {
                            sb_pre!(&[router_gpu, down_out_gpu], &[down_out_gpu]);
                            self.elementwise.encode_sigmoid_scalar_mul_inplace(
                                encoder,
                                router_gpu,
                                down_out_gpu,
                                m_dim,
                            );
                            sb_post!(&[router_gpu, down_out_gpu], &[down_out_gpu]);
                        } else {
                            sb_pre!(&[router_gpu], &[router_gpu]);
                            self.elementwise
                                .encode_sigmoid_inplace(encoder, router_gpu, nt);
                            sb_post!(&[router_gpu], &[router_gpu]);
                            sb_pre!(&[down_out_gpu, router_gpu], &[down_out_gpu]);
                            self.gdn.encode_broadcast_mul(
                                encoder,
                                down_out_gpu,
                                router_gpu,
                                down_out_gpu,
                                m_dim,
                                m_dim * nt,
                            );
                            sb_post!(&[down_out_gpu, router_gpu], &[down_out_gpu]);
                        }
                    }
                }
            }

            sb_pre!(&[accum_gpu, down_out_gpu], &[accum_gpu]);
            self.elementwise.encode_elementwise_add_batch(
                encoder,
                accum_gpu,
                down_out_gpu,
                m_dim,
                nt,
            );
            sb_post!(&[accum_gpu, down_out_gpu], &[accum_gpu]);
        }

        sb_pre!(&[hidden_gpu, accum_gpu], &[hidden_gpu]);
        self.elementwise
            .encode_elementwise_add_batch(encoder, hidden_gpu, accum_gpu, m_dim, nt);
        sb_post!(&[hidden_gpu, accum_gpu], &[hidden_gpu]);
        // When decode continues in the same command buffer, the updated hidden
        // state becomes the input to the next layer (or output head) immediately.
        // Keep the final residual add ordered with subsequent reads under the
        // explicit-barrier decode plan.
        sb_flush!();
        if std::env::var("AX_DEBUG_MOE_NATIVE").is_ok() && n_tokens == 1 {
            let norm = unsafe { &norm_gpu.as_slice::<f32>()[..4.min(dim)] };
            let accum = unsafe { &accum_gpu.as_slice::<f32>()[..4.min(dim)] };
            let hidden = unsafe { &hidden_gpu.as_slice::<f32>()[..4.min(dim)] };
            eprintln!(
                "[MOE NATIVE DEBUG] use_selected={use_selected_expert_single_token_path} supported_selected={selected_expert_single_token_supported} gate_dtype={gate_dtype:?} up_dtype={up_dtype:?} down_dtype={down_dtype:?} pre_hidden[0..4]={:?} norm[0..4]={norm:?} accum[0..4]={accum:?} hidden[0..4]={hidden:?}",
                pre_hidden_debug.as_deref().unwrap_or(&[])
            );
        }
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
        let scratch = self.moe_batch_scratch_view()?;
        let pre_hidden_debug = if std::env::var("AX_DEBUG_MOE_NATIVE").is_ok() && n_tokens == 1 {
            Some(unsafe { hidden_gpu.as_slice::<f32>()[..4.min(dim)].to_vec() })
        } else {
            None
        };
        self.device.execute_sync(|encoder| {
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
                false,
            )
        })?;
        if std::env::var("AX_DEBUG_MOE_NATIVE").is_ok() && n_tokens == 1 {
            let ids = unsafe { &(*scratch.ids).as_slice::<i32>()[..n_tokens * n_expert_used] };
            let weights =
                unsafe { &(*scratch.weights).as_slice::<f32>()[..n_tokens * n_expert_used] };
            let norm = unsafe { &(*scratch.norm).as_slice::<f32>()[..4.min(dim)] };
            let accum = unsafe { &(*scratch.accum).as_slice::<f32>()[..4.min(dim)] };
            let hidden = unsafe { &hidden_gpu.as_slice::<f32>()[..4.min(dim)] };
            eprintln!(
                "[MOE NATIVE DEBUG] pre_hidden[0..4]={:?} ids={ids:?} weights={weights:?} norm[0..4]={norm:?} accum[0..4]={accum:?} hidden[0..4]={hidden:?}",
                pre_hidden_debug.as_deref().unwrap_or(&[])
            );
        }
        Ok(())
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
        let hids_gpu = MetalBuffer::new(dev, n_expert * n_tokens * i4)?;
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

        let mut active_buf = MetalBuffer::new(dev, (n_active as usize + 1).max(1) * u4)?;
        unsafe {
            let active_meta = active_buf.as_mut_slice::<u32>();
            active_meta[0] = n_active;
            active_meta[1..1 + active_list.len()].copy_from_slice(&active_list);
        }

        // Single CB: map0 + 3× mul_mat_id + SiLU + scatter.
        self.device.execute_sync(|encoder| {
            // Build routing index on GPU.
            self.dequant.encode_moe_map0(
                encoder,
                &ids_gpu,
                &tpe_gpu,
                &hids_gpu,
                &active_buf,
                nt,
                neu,
                ne,
            );
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
                false,
                false,
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
                false,
                false,
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
                false,
                true,
            )?;

            // Stage 5: Reduce routed expert slots into per-token accumulators.
            self.elementwise.encode_moe_weighted_reduce_slots(
                encoder,
                &down_out_gpu,
                &weights_gpu,
                &accum_gpu,
                m_dim,
                nt,
                neu,
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

    /// Execute Gemma4-style routed experts with fused `gate_up` weights.
    ///
    /// This mirrors llama.cpp's merged `gate_up_exps` path: one routed
    /// matmul produces `[gate | up]`, a split GELU-mul kernel materializes the
    /// activated expert hidden state, and a routed down matmul plus weighted
    /// slot reduction accumulates per-token outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_gate_up_gelu_dispatch(
        &self,
        norm_buf: &[f32],
        accum_out: &mut [f32],
        expert_ids: &[i32],
        expert_weights_flat: &[f32],
        gate_up_exps_raw: &[u8],
        down_exps_raw: &[u8],
        gate_up_dtype: GgmlType,
        down_dtype: GgmlType,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        dim: usize,
        expert_inter_dim: usize,
        gate_up_stride: usize,
        down_stride: usize,
    ) -> anyhow::Result<()> {
        let f4 = std::mem::size_of::<f32>();
        let i4 = std::mem::size_of::<i32>();
        let u4 = std::mem::size_of::<u32>();
        let dev = self.device.device();
        let fused_dim = expert_inter_dim
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Gemma4 fused gate_up dimension overflow"))?;
        let blocks_per_expert_gate_up =
            Self::moe_blocks_per_expert(gate_up_stride, gate_up_dtype, "expert gate_up")?;
        let blocks_per_expert_down =
            Self::moe_blocks_per_expert(down_stride, down_dtype, "expert down")?;

        let gate_up_key = self.ensure_moe_quant_cached(gate_up_exps_raw);
        let down_key = self.ensure_moe_quant_cached(down_exps_raw);

        let mut input_gpu = MetalBuffer::new(dev, n_tokens * dim * f4)?;
        let mut accum_gpu = MetalBuffer::new(dev, n_tokens * dim * f4)?;
        let mut ids_gpu = MetalBuffer::new(dev, n_tokens * n_expert_used * i4)?;
        let mut weights_gpu = MetalBuffer::new(dev, n_tokens * n_expert_used * f4)?;
        let tpe_gpu = MetalBuffer::new(dev, n_expert * u4)?;
        let hids_gpu = MetalBuffer::new(dev, n_expert * n_tokens * i4)?;
        let gate_up_out_gpu = MetalBuffer::new(dev, n_tokens * n_expert_used * fused_dim * f4)?;
        let activated_gpu =
            MetalBuffer::new(dev, n_tokens * n_expert_used * expert_inter_dim * f4)?;
        let down_out_gpu = MetalBuffer::new(dev, n_tokens * n_expert_used * dim * f4)?;

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

        let cache = self.moe_quant_cache.lock().unwrap();
        let gate_up_buf = cache.get(&gate_up_key).unwrap();
        let down_buf = cache.get(&down_key).unwrap();

        let nt = n_tokens as u32;
        let ne = n_expert as u32;
        let neu = n_expert_used as u32;
        let m_fused = fused_dim as u32;
        let m_dim = dim as u32;
        let k_dim = dim as u32;
        let k_inter = expert_inter_dim as u32;

        let mut active_set = std::collections::BTreeSet::new();
        for &eid in expert_ids {
            if eid >= 0 {
                active_set.insert(eid as u32);
            }
        }
        let active_list: Vec<u32> = active_set.into_iter().collect();
        let n_active = active_list.len() as u32;

        let mut active_buf = MetalBuffer::new(dev, (n_active as usize + 1).max(1) * u4)?;
        unsafe {
            let active_meta = active_buf.as_mut_slice::<u32>();
            active_meta[0] = n_active;
            active_meta[1..1 + active_list.len()].copy_from_slice(&active_list);
        }

        self.device.execute_sync(|encoder| {
            self.dequant.encode_moe_map0(
                encoder,
                &ids_gpu,
                &tpe_gpu,
                &hids_gpu,
                &active_buf,
                nt,
                neu,
                ne,
            );
            self.encode_moe_mul_mat_id(
                encoder,
                gate_up_buf,
                &input_gpu,
                &tpe_gpu,
                &hids_gpu,
                &gate_up_out_gpu,
                gate_up_dtype,
                m_fused,
                k_dim,
                nt,
                neu,
                ne,
                blocks_per_expert_gate_up,
                &active_buf,
                n_active,
                false,
                false,
            )?;
            self.elementwise.encode_gelu_split_mul_batch(
                encoder,
                &gate_up_out_gpu,
                &activated_gpu,
                k_inter,
                nt * neu,
            );
            self.encode_moe_mul_mat_id(
                encoder,
                down_buf,
                &activated_gpu,
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
                false,
                true,
            )?;
            self.elementwise.encode_moe_weighted_reduce_slots(
                encoder,
                &down_out_gpu,
                &weights_gpu,
                &accum_gpu,
                m_dim,
                nt,
                neu,
            );
            Ok(())
        })?;

        drop(cache);

        unsafe {
            accum_out[..n_tokens * dim]
                .copy_from_slice(&accum_gpu.as_slice::<f32>()[..n_tokens * dim]);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gemma4_routed_moe_fused_gate_up(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        bs: &mut GpuBatchScratchBuffers,
        router_scale: &MetalBuffer,
        router_buf: &MetalBuffer,
        router_dtype: GgmlType,
        pre_ff2_w: &MetalBuffer,
        post_ff2_w: &MetalBuffer,
        expert_scales: &MetalBuffer,
        gate_up_buf: &MetalBuffer,
        gate_up_dtype: GgmlType,
        down_buf: &MetalBuffer,
        down_dtype: GgmlType,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        dim: usize,
        expert_inter_dim: usize,
        gate_up_stride: usize,
        down_stride: usize,
        eps: f32,
    ) -> anyhow::Result<()> {
        let expert_input_gpu = bs
            .moe_norm
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_norm scratch"))?;
        let router_logits_gpu = bs
            .moe_router_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_router_out scratch"))?;
        let accum_gpu = bs
            .moe_accum
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_accum scratch"))?;
        let gate_out_gpu = bs
            .moe_gate_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_gate_out scratch"))?;
        let fused_gate_up_gpu = bs
            .moe_fused_gate_up
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_fused_gate_up scratch"))?;
        let down_out_gpu = bs
            .moe_down_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_down_out scratch"))?;
        let tpe_gpu = bs
            .moe_tpe
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_tpe scratch"))?;
        let hids_gpu = bs
            .moe_hids
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_hids scratch"))?;
        let ids_gpu = bs
            .moe_expert_ids
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_expert_ids scratch"))?;
        let weights_gpu = bs
            .moe_expert_weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_expert_weights scratch"))?;
        let active_buf = bs
            .moe_active_experts
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 MoE missing moe_active_experts scratch"))?;

        let blocks_per_expert_gate_up =
            Self::moe_blocks_per_expert(gate_up_stride, gate_up_dtype, "expert gate_up")?;
        let blocks_per_expert_down =
            Self::moe_blocks_per_expert(down_stride, down_dtype, "expert down")?;
        let nt = n_tokens as u32;
        let ne = n_expert as u32;
        let neu = n_expert_used as u32;
        let max_active_experts = ne.min(nt.saturating_mul(neu));
        let fused_dim = expert_inter_dim
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Gemma4 fused gate_up dimension overflow"))?;
        let m_fused = fused_dim as u32;
        let m_dim = dim as u32;
        let k_dim = dim as u32;
        let k_inter = expert_inter_dim as u32;
        let router_input_scale = (dim as f32).sqrt().recip();
        let hidden = &bs.hidden;

        unsafe {
            active_buf.as_mut_slice::<u32>()[0] = 0;
        }

        let mut sb = ax_engine_metal::SmartBarrier::new(encoder);
        sb.pre_dispatch(&[hidden], &[expert_input_gpu]);
        self.elementwise.encode_rms_norm_out_batch(
            encoder,
            hidden,
            pre_ff2_w,
            expert_input_gpu,
            dim as u32,
            nt,
            eps,
        );
        sb.post_dispatch(&[hidden], &[expert_input_gpu]);

        sb.pre_dispatch(&[hidden], &[accum_gpu]);
        self.elementwise.encode_rms_norm_out_batch(
            encoder,
            hidden,
            router_scale,
            accum_gpu,
            dim as u32,
            nt,
            eps,
        );
        sb.post_dispatch(&[hidden], &[accum_gpu]);

        sb.pre_dispatch(&[accum_gpu], &[accum_gpu]);
        self.elementwise.encode_gen_scale(
            encoder,
            accum_gpu,
            accum_gpu,
            router_input_scale,
            u32::try_from(n_tokens * dim)
                .map_err(|_| anyhow::anyhow!("Gemma4 router input element count overflow"))?,
        );
        sb.post_dispatch(&[accum_gpu], &[accum_gpu]);

        sb.pre_dispatch(&[accum_gpu], &[router_logits_gpu]);
        if router_dtype == GgmlType::F32 {
            self.encode_token_major_batch_matmul_f32(
                encoder,
                router_buf,
                accum_gpu,
                &bs.matmul_in_f16,
                router_logits_gpu,
                ne,
                nt,
                k_dim,
            )?;
        } else {
            self.encode_quant_batch_matmul(
                encoder,
                router_buf,
                accum_gpu,
                router_logits_gpu,
                ne,
                nt,
                k_dim,
                router_dtype,
            )?;
        }
        sb.post_dispatch(&[accum_gpu], &[router_logits_gpu]);

        sb.pre_dispatch(&[router_logits_gpu], &[ids_gpu, weights_gpu]);
        self.elementwise.encode_moe_softmax_topk(
            encoder,
            router_logits_gpu,
            ids_gpu,
            weights_gpu,
            nt,
            ne,
            neu,
        );
        sb.post_dispatch(&[router_logits_gpu], &[ids_gpu, weights_gpu]);

        sb.pre_dispatch(&[ids_gpu, weights_gpu, expert_scales], &[weights_gpu]);
        self.elementwise.encode_moe_apply_expert_scales(
            encoder,
            ids_gpu,
            weights_gpu,
            expert_scales,
            nt,
            neu,
        );
        sb.post_dispatch(&[ids_gpu, weights_gpu, expert_scales], &[weights_gpu]);

        sb.pre_dispatch(&[accum_gpu], &[accum_gpu]);
        self.elementwise.encode_gen_scale(
            encoder,
            accum_gpu,
            accum_gpu,
            0.0,
            u32::try_from(n_tokens * dim)
                .map_err(|_| anyhow::anyhow!("Gemma4 MoE accum element count overflow"))?,
        );
        sb.post_dispatch(&[accum_gpu], &[accum_gpu]);

        sb.pre_dispatch(&[ids_gpu, active_buf], &[tpe_gpu, hids_gpu]);
        self.dequant
            .encode_moe_map0(encoder, ids_gpu, tpe_gpu, hids_gpu, active_buf, nt, neu, ne);
        sb.post_dispatch(&[ids_gpu, active_buf], &[tpe_gpu, hids_gpu]);

        sb.pre_dispatch(
            &[expert_input_gpu, tpe_gpu, hids_gpu, active_buf],
            &[fused_gate_up_gpu],
        );
        self.encode_moe_mul_mat_id(
            encoder,
            gate_up_buf,
            expert_input_gpu,
            tpe_gpu,
            hids_gpu,
            fused_gate_up_gpu,
            gate_up_dtype,
            m_fused,
            k_dim,
            nt,
            neu,
            ne,
            blocks_per_expert_gate_up,
            active_buf,
            max_active_experts,
            false,
            false,
        )?;
        sb.post_dispatch(
            &[expert_input_gpu, tpe_gpu, hids_gpu, active_buf],
            &[fused_gate_up_gpu],
        );

        sb.pre_dispatch(&[fused_gate_up_gpu], &[gate_out_gpu]);
        self.elementwise.encode_gelu_split_mul_batch(
            encoder,
            fused_gate_up_gpu,
            gate_out_gpu,
            k_inter,
            nt * neu,
        );
        sb.post_dispatch(&[fused_gate_up_gpu], &[gate_out_gpu]);

        sb.pre_dispatch(
            &[gate_out_gpu, tpe_gpu, hids_gpu, active_buf],
            &[down_out_gpu],
        );
        self.encode_moe_mul_mat_id(
            encoder,
            down_buf,
            gate_out_gpu,
            tpe_gpu,
            hids_gpu,
            down_out_gpu,
            down_dtype,
            m_dim,
            k_inter,
            nt,
            neu,
            ne,
            blocks_per_expert_down,
            active_buf,
            max_active_experts,
            false,
            true,
        )?;
        sb.post_dispatch(
            &[gate_out_gpu, tpe_gpu, hids_gpu, active_buf],
            &[down_out_gpu],
        );

        sb.pre_dispatch(&[down_out_gpu, weights_gpu], &[accum_gpu]);
        self.elementwise.encode_moe_weighted_reduce_slots(
            encoder,
            down_out_gpu,
            weights_gpu,
            accum_gpu,
            m_dim,
            nt,
            neu,
        );
        sb.post_dispatch(&[down_out_gpu, weights_gpu], &[accum_gpu]);

        sb.pre_dispatch(&[accum_gpu], &[accum_gpu]);
        self.elementwise
            .encode_rms_norm_batch(encoder, accum_gpu, post_ff2_w, m_dim, nt, eps);
        sb.post_dispatch(&[accum_gpu], &[accum_gpu]);
        sb.flush();
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
    ///
    /// Cached-model paths look resident buffers up through `lock_weight_cache`.
    /// Store quantized tensors under a tagged key so they cannot collide with
    /// dense/f32 slices that share the same mmap address.
    pub fn ensure_quant_cached(&self, data: &[u8]) -> usize {
        let raw_key = data.as_ptr() as usize;
        let view_key = Self::quant_view_cache_key(raw_key);
        let mut cache = self.f32_weight_cache.lock().unwrap();
        cache
            .entry(view_key)
            .or_insert_with(|| create_mmap_weight_buffer_from_bytes(&self.device, data));
        view_key
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

    // v2: GpuKv is owned by InferenceModel via ModelKv::Gpu. No init_gpu_kv or gpu_kv() here.
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
        state_batch: &mut super::Qwen3_5RecurrentStateBatch<'_>,
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

    fn sync_qwen35_kv(&self, qwen_kv: &mut crate::kv::Qwen3_5Kv) {
        if self.decode_on_cpu {
            self.cpu.sync_qwen35_kv(qwen_kv);
        } else {
            self.metal.sync_qwen35_kv(qwen_kv);
        }
    }

    fn try_clone_qwen35_recurrent_slot(
        &self,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
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
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
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
        state_batch: &mut super::Qwen3_5RecurrentStateBatch<'_>,
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
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
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

    fn sync_qwen35_kv(&self, qwen_kv: &mut crate::kv::Qwen3_5Kv) {
        self.cpu.sync_qwen35_kv(qwen_kv);
    }

    fn try_clone_qwen35_recurrent_slot(
        &self,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
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
impl MetalOps {
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
}
