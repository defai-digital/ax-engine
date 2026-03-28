use rustc_hash::FxHashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::time::Instant;

use super::{Backend, RuntimePolicy};
use crate::gguf::tensor::GgmlType;
use anyhow::Context;

// v2: GpuKv is owned by LlamaModel via ModelKv, not stored in MetalOps.
use crate::model::config::ModelConfig;
use ax_engine_metal::{
    AttentionDispatchConfig, AttentionKernels, DequantDispatchConfig, DequantKernels,
    ElementwiseKernels, GdnKernels, MatmulKernels, MetalBuffer, MetalDevice,
};

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

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen35RecurrentBatchPerfCounters {
    pub conv_ns: u64,
    pub pack_ns: u64,
    pub gated_delta_ns: u64,
    pub unpack_ns: u64,
}

#[derive(Default)]
struct Qwen35RecurrentBatchPerfState {
    conv_ns: AtomicU64,
    pack_ns: AtomicU64,
    gated_delta_ns: AtomicU64,
    unpack_ns: AtomicU64,
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
            panic!(
                "qwen35 conv state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer lost generation {conv_generation}"
            );
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
            panic!(
                "qwen35 recurrent state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer lost generation {recurrent_generation}"
            );
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
        let layer_idx = state_batch.layer_idx();
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
            let slot_idx = state_batch.slot_index(batch_idx);
            let token_start = batch_idx * tokens_per_slot;
            let token_end = token_start + tokens_per_slot;
            let qkv_start = token_start * cfg.conv_dim;
            let qkv_end = token_end * cfg.conv_dim;
            let gate_start = token_start * cfg.time_step_rank;
            let gate_end = token_end * cfg.time_step_rank;
            let out_start = token_start * value_dim;
            let out_end = token_end * value_dim;

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
            slot_buffers.source_kv_identity = None;

            {
                let conv_state_src = state_batch.conv_state_for_slot_mut(batch_idx);
                unsafe {
                    slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_src.len()]
                        .copy_from_slice(conv_state_src);
                }
            }
            {
                let recurrent_state_src = state_batch.recurrent_state_for_slot_mut(batch_idx);
                unsafe {
                    slot_buffers.recurrent_state.as_mut_slice::<f32>()[..recurrent_state_src.len()]
                        .copy_from_slice(recurrent_state_src);
                }
            }

            unsafe {
                scratch_buffers.input.as_mut_slice::<f32>()[..qkv_end - qkv_start]
                    .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
            }
            self.gdn_kernels
                .causal_conv_sequence(
                    &self.device,
                    &scratch_buffers.input,
                    buf_kernel,
                    &slot_buffers.conv_state,
                    &scratch_buffers.conv_out,
                    tokens_per_slot as u32,
                    cfg.conv_cache_len as u32,
                    cfg.conv_dim as u32,
                )
                .expect("Metal qwen35 recurrent causal conv dispatch failed");

            let conv_out = unsafe { scratch_buffers.conv_out.as_slice::<f32>() };
            unsafe {
                if tokens_per_slot == 1 {
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
                } else {
                    prepare_multi_token_gdn_bh_buffers(
                        conv_out,
                        &alpha_batch[gate_start..gate_end],
                        &beta_batch[gate_start..gate_end],
                        &mut scratch_buffers.q.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                        &mut scratch_buffers.k.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                        &mut scratch_buffers.v.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                        &mut scratch_buffers.gate.as_mut_slice::<f32>()
                            [..tokens_per_slot * cfg.time_step_rank],
                        &mut scratch_buffers.beta.as_mut_slice::<f32>()
                            [..tokens_per_slot * cfg.time_step_rank],
                        tokens_per_slot,
                        cfg.group_count,
                        cfg.time_step_rank,
                        cfg.state_size,
                        cfg.rms_norm_eps,
                    );
                }
            }
            self.gdn_kernels
                .gated_delta_sequence(
                    &self.device,
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

            let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
            if tokens_per_slot == 1 {
                output_batch[out_start..out_end].copy_from_slice(&out_bhsv[..value_dim]);
            } else {
                unpack_bhsk_to_token_major(
                    out_bhsv,
                    &mut output_batch[out_start..out_end],
                    tokens_per_slot,
                    cfg.time_step_rank,
                    cfg.state_size,
                );
            }

            {
                let conv_state_dst = state_batch.conv_state_for_slot_mut(batch_idx);
                let conv_state_out = unsafe { slot_buffers.conv_state.as_slice::<f32>() };
                conv_state_dst.copy_from_slice(&conv_state_out[..conv_state_dst.len()]);
            }
            {
                let recurrent_state_dst = state_batch.recurrent_state_for_slot_mut(batch_idx);
                let recurrent_state_out = unsafe { slot_buffers.recurrent_state.as_slice::<f32>() };
                recurrent_state_dst
                    .copy_from_slice(&recurrent_state_out[..recurrent_state_dst.len()]);
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

                // Run conv1d on CPU for single-token decode (GPU dispatch
                // overhead exceeds the compute savings for seq_len=1).
                // Write to KV's CPU conv state, then sync to GPU slot buffer.
                {
                    let conv_out_dst = unsafe {
                        &mut scratch_buffers.conv_out.as_mut_slice::<f32>()[..cfg.conv_dim]
                    };
                    let conv_state_cpu = qwen_kv.conv_state_for_slot_mut(slot_idx, layer_idx);
                    crate::compute::gdn::depthwise_conv1d_step(
                        conv_state_cpu,
                        &qkv_batch[qkv_start..qkv_end],
                        conv_kernel,
                        cfg.conv_cache_len,
                        cfg.conv_dim,
                        conv_out_dst,
                    );
                    // Sync updated conv state to GPU slot buffer.
                    unsafe {
                        slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                            .copy_from_slice(conv_state_cpu);
                    }
                    slot_buffers.conv_synced_generation =
                        Some(qwen_kv.conv_state_generation(slot_idx, layer_idx));
                }

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
                self.gdn_kernels
                    .gated_delta_sequence(
                        &self.device,
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

                let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
                output_batch[out_start..out_end].copy_from_slice(&out_bhsv[..value_dim]);

                // Mark recurrent state as backend-owned (GPU has the latest).
                let backend_generation =
                    qwen_kv.note_backend_recurrent_state_update(slot_idx, layer_idx);
                slot_buffers.recurrent_synced_generation = Some(backend_generation);
                slot_buffers.source_kv_identity =
                    Some(qwen_kv as *const crate::kv::Qwen35Kv as usize);
            } else {
                self.sync_qwen35_slot_from_backend_if_needed(qwen_kv, layer_idx, slot_idx);
                let (conv_state, recurrent_state) =
                    qwen_kv.recurrent_buffers_for_slot_mut(slot_idx, layer_idx);
                let conv_state_buf = unsafe {
                    MetalBuffer::from_mut_slice_no_copy(self.device.device(), conv_state)
                }
                .expect("Failed to alias qwen35 conv state into Metal buffer");
                let recurrent_state_buf = unsafe {
                    MetalBuffer::from_mut_slice_no_copy(self.device.device(), recurrent_state)
                }
                .expect("Failed to alias qwen35 recurrent state into Metal buffer");
                unsafe {
                    scratch_buffers.input.as_mut_slice::<f32>()[..qkv_end - qkv_start]
                        .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
                }

                let conv_t = Instant::now();
                self.gdn_kernels
                    .causal_conv_sequence(
                        &self.device,
                        &scratch_buffers.input,
                        buf_kernel,
                        &conv_state_buf,
                        &scratch_buffers.conv_out,
                        tokens_per_slot as u32,
                        cfg.conv_cache_len as u32,
                        cfg.conv_dim as u32,
                    )
                    .expect("Metal qwen35 recurrent causal conv dispatch failed");
                self.ops
                    .record_qwen35_recurrent_batch_conv(conv_t.elapsed());

                let conv_out = unsafe { scratch_buffers.conv_out.as_slice::<f32>() };
                let pack_t = Instant::now();
                unsafe {
                    prepare_multi_token_gdn_bh_buffers(
                        conv_out,
                        &alpha_batch[gate_start..gate_end],
                        &beta_batch[gate_start..gate_end],
                        &mut scratch_buffers.q.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                        &mut scratch_buffers.k.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                        &mut scratch_buffers.v.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                        &mut scratch_buffers.gate.as_mut_slice::<f32>()
                            [..tokens_per_slot * cfg.time_step_rank],
                        &mut scratch_buffers.beta.as_mut_slice::<f32>()
                            [..tokens_per_slot * cfg.time_step_rank],
                        tokens_per_slot,
                        cfg.group_count,
                        cfg.time_step_rank,
                        cfg.state_size,
                        cfg.rms_norm_eps,
                    );
                }
                self.ops
                    .record_qwen35_recurrent_batch_pack(pack_t.elapsed());
                let gated_delta_t = Instant::now();
                self.gdn_kernels
                    .gated_delta_sequence(
                        &self.device,
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
                self.ops
                    .record_qwen35_recurrent_batch_gated_delta(gated_delta_t.elapsed());

                let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
                let unpack_t = Instant::now();
                unpack_bhsk_to_token_major(
                    out_bhsv,
                    &mut output_batch[out_start..out_end],
                    tokens_per_slot,
                    cfg.time_step_rank,
                    cfg.state_size,
                );
                self.ops
                    .record_qwen35_recurrent_batch_unpack(unpack_t.elapsed());
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
                GgmlType::Q4_0 => {
                    self.fused_matvec_q4_0(a_quant, b, c, m, k);
                    return;
                }
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
                    GgmlType::F32
                        | GgmlType::Q4_0
                        | GgmlType::Q8_0
                        | GgmlType::Q4K
                        | GgmlType::Q5K
                        | GgmlType::Q6K
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
                        GgmlType::Q4_0 => {
                            self.dequant_kernels.encode_fused_matvec_q4_0(
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

        self.gdn_kernels
            .causal_conv_sequence(
                &self.device,
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

        self.gdn_kernels
            .gated_delta_sequence(
                &self.device,
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
            if let Some(generation) = slot_buffers.recurrent_synced_generation {
                if qwen_kv.recurrent_state_cpu_stale(slot_key.slot_idx, slot_key.layer_idx)
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
            }
            // Sync conv state GPU → CPU if backend-owned.
            if let Some(generation) = slot_buffers.conv_synced_generation {
                if qwen_kv.conv_state_cpu_stale(slot_key.slot_idx, slot_key.layer_idx)
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
        }
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

fn unpack_bhsk_to_token_major(
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
fn prepare_multi_token_gdn_bh_buffers(
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

    fn fused_matvec_q4_0(&self, a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
        let y_bytes = m * std::mem::size_of::<f32>();
        let input_guard = self.prepare_input(x);
        let output_guard = self.prepare_output(y_bytes);

        let key = a_quant.as_ptr() as usize;
        let cache = self.get_weight_buffer(a_quant);
        let buf_a = cache.get(&key).unwrap();

        self.dequant_kernels
            .fused_matvec_q4_0(
                &self.device,
                buf_a,
                input_guard.0.as_ref().unwrap(),
                output_guard.0.as_ref().unwrap(),
                m as u32,
                k as u32,
            )
            .expect("Metal fused Q4_0 matvec failed");

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
    /// Backend-local profiling for Qwen3.5 recurrent multi-token batch phases.
    qwen35_recurrent_batch_perf: Qwen35RecurrentBatchPerfState,
    /// One-time runtime tuning state for f16in batch kernel routing.
    f16in_route_tuned: AtomicBool,
    /// Pre-computed weight cache keys, built once on first forward call.
    cached_model_keys: Mutex<Option<CachedModelKeys>>,
}

impl MetalOps {
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
            qwen35_recurrent_batch_perf: Qwen35RecurrentBatchPerfState::default(),
            f16in_route_tuned: AtomicBool::new(false),
            cached_model_keys: Mutex::new(None),
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

    pub fn metal_batch_f16_io_enabled(&self) -> bool {
        self.runtime_policy
            .read()
            .unwrap()
            .batch_prefill_prefers_f16_io()
    }

    fn record_qwen35_recurrent_batch_conv(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .conv_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    fn record_qwen35_recurrent_batch_pack(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .pack_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    fn record_qwen35_recurrent_batch_gated_delta(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .gated_delta_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    fn record_qwen35_recurrent_batch_unpack(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .unpack_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
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
            gate_buf: alloc(inter_dim),
            up_buf: alloc(inter_dim),
            down_buf: alloc(dim),
            logits_buf: alloc(vocab_size),
            splitk_partial_out: alloc(max_chunks * n_heads * head_dim),
            splitk_partial_lse: alloc(max_chunks * n_heads),
        };
        tracing::info!(
            dim,
            q_dim,
            kv_dim,
            inter_dim,
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

        let max_output_dim = dim.max(inter_dim).max(q_dim).max(kv_dim);

        let bs = GpuBatchScratchBuffers {
            hidden: alloc(n_tokens * dim),
            qkv_buf: alloc(n_tokens * (q_dim + 2 * kv_dim)),
            q_buf: alloc(n_tokens * q_dim),
            k_buf: alloc(n_tokens * kv_dim),
            v_buf: alloc(n_tokens * kv_dim),
            attn_out: alloc(n_tokens * q_dim),
            norm_buf: alloc(n_tokens * dim),
            proj_buf: alloc(n_tokens * dim),
            gate_buf: alloc(n_tokens * inter_dim),
            up_buf: alloc(n_tokens * inter_dim),
            matmul_in_f16: alloc_f16(n_tokens * max_input_dim),
            matmul_out_f16: alloc_f16(n_tokens * max_output_dim),
            batch_size: n_tokens,
        };
        tracing::info!(
            n_tokens,
            dim,
            q_dim,
            kv_dim,
            inter_dim,
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

        let dim = config.embedding_dim;
        let inter = config.intermediate_dim;
        let k = dim;
        let n_candidates = [32u32, 48u32, 64u32];
        let m_candidates = [dim, inter];

        // Keep this cheap: two shapes × three N values × small reps.
        let mut wins: Vec<(u32, u32)> = Vec::new();
        for &m in &m_candidates {
            for &n in &n_candidates {
                let t_large = self.bench_q4k_f16in_variant(m, n, k, false);
                let t_small = self.bench_q4k_f16in_variant(m, n, k, true);
                if let (Some(large), Some(small)) = (t_large, t_small)
                    && small < (large * 98 / 100)
                {
                    wins.push((m, n));
                }
            }
        }

        let old_policy = self.runtime_policy();
        let old_config = old_policy.dequant_dispatch_config();
        let (n_threshold, m_max) = if wins.is_empty() {
            (1u32, 0u32)
        } else {
            let max_n = wins.iter().map(|(_, n)| *n).max().unwrap_or(32);
            let max_m = wins.iter().map(|(m, _)| *m).max().unwrap_or(dim);
            (max_n + 1, max_m)
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
            "Autotuned f16in batch kernel routing"
        );
    }

    fn bench_q4k_f16in_variant(&self, m: u32, n: u32, k: u32, use_small: bool) -> Option<u128> {
        if !k.is_multiple_of(256) {
            return None;
        }

        const Q4K_BLOCK_VALUES: u32 = 256;
        const Q4K_BYTES_PER_BLOCK: usize = 144;
        let blocks_per_row = k / Q4K_BLOCK_VALUES;
        if blocks_per_row == 0 {
            return None;
        }
        let a_bytes = (m as usize)
            .checked_mul(blocks_per_row as usize)?
            .checked_mul(Q4K_BYTES_PER_BLOCK)?;
        let b_len = (n as usize).checked_mul(k as usize)?;
        let c_bytes = (n as usize)
            .checked_mul(m as usize)?
            .checked_mul(std::mem::size_of::<f32>())?;

        let a_data = vec![0u8; a_bytes];
        let b_data = vec![half::f16::from_f32(0.0); b_len];
        let a = MetalBuffer::from_bytes(self.device.device(), &a_data).ok()?;
        let b = MetalBuffer::from_slice(self.device.device(), &b_data).ok()?;
        let c = MetalBuffer::new(self.device.device(), c_bytes).ok()?;

        let old_config = self.dequant_dispatch_config();
        let mut bench_config = old_config;
        if use_small {
            bench_config.batch_f16in_small_n_threshold = u32::MAX;
            bench_config.batch_f16in_small_m_max = m;
        } else {
            bench_config.batch_f16in_small_n_threshold = 1;
            bench_config.batch_f16in_small_m_max = 0;
        }

        // Warmup
        let _ = self.device.execute_sync(|encoder| {
            self.dequant.encode_fused_batch_q4_k_f16in_with_config(
                encoder,
                &a,
                &b,
                &c,
                m,
                n,
                k,
                bench_config,
            );
            Ok(())
        });

        let reps = 3;
        let t0 = Instant::now();
        for _ in 0..reps {
            if self
                .device
                .execute_sync(|encoder| {
                    self.dequant.encode_fused_batch_q4_k_f16in_with_config(
                        encoder,
                        &a,
                        &b,
                        &c,
                        m,
                        n,
                        k,
                        bench_config,
                    );
                    Ok(())
                })
                .is_err()
            {
                return None;
            }
        }
        Some(t0.elapsed().as_nanos() / reps as u128)
    }

    /// Access GPU batch scratch buffers (must call init_batch_scratches first).
    pub fn batch_scratches(&self) -> std::sync::MutexGuard<'_, Option<GpuBatchScratchBuffers>> {
        self.batch_scratches.lock().unwrap()
    }

    pub(crate) fn sync_qwen35_slot_buffers_from_kv(
        &self,
        qwen_kv: &crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_idx: usize,
    ) {
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let slot_key = Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        };
        let recurrent_generation = qwen_kv.recurrent_state_generation(slot_idx, layer_idx);
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
                unsafe {
                    slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                        .copy_from_slice(qwen_kv.conv_state_for_slot(slot_idx, layer_idx));
                }
                slot_buffers.conv_synced_generation = Some(conv_generation);
            }
        } else if slot_buffers.conv_synced_generation != Some(conv_generation) {
            panic!(
                "qwen35 conv state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer lost generation {conv_generation}"
            );
        }

        if !qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx) {
            if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
                unsafe {
                    slot_buffers.recurrent_state.as_mut_slice::<f32>()[..recurrent_state_stride]
                        .copy_from_slice(qwen_kv.recurrent_state_for_slot(slot_idx, layer_idx));
                }
                slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
            }
        } else if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
            panic!(
                "qwen35 recurrent state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer lost generation {recurrent_generation}"
            );
        }
        slot_buffers.source_kv_identity = Some(kv_identity);
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

        assert_eq!(actual_q, expected_q);
        assert_eq!(actual_k, expected_k);
        assert_eq!(actual_v, expected_v);
        assert_eq!(actual_gate, expected_gate);
        assert_eq!(actual_beta, expected_beta);
    }

    #[test]
    fn test_metal_backend_batch_dequant_matvec_mixed_dtypes_share_command_buffer() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;

        let k = 32;
        let mut q4_block = [0u8; 18];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        q4_block[0] = d_bytes[0];
        q4_block[1] = d_bytes[1];
        q4_block[2..18].fill(0x99);

        let f32_weight = [0.5f32; 32];
        let f32_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                f32_weight.as_ptr() as *const u8,
                std::mem::size_of_val(&f32_weight),
            )
        };

        let x = [1.0f32; 32];
        let mut q4_out = [0.0f32; 1];
        let mut f32_out = [0.0f32; 1];
        let mut expected_q4 = [0.0f32; 1];
        let mut expected_f32 = [0.0f32; 1];

        cpu.dequant_matmul(&q4_block, GgmlType::Q4_0, &x, &mut expected_q4, 1, 1, k);
        cpu.dequant_matmul(f32_bytes, GgmlType::F32, &x, &mut expected_f32, 1, 1, k);

        backend.device.reset_perf_counters();
        backend.batch_dequant_matvec(
            &[
                (&q4_block, GgmlType::Q4_0, 1),
                (f32_bytes, GgmlType::F32, 1),
            ],
            &x,
            k,
            &mut [&mut q4_out, &mut f32_out],
        );
        let counters = backend.device.perf_counters();

        assert!((q4_out[0] - expected_q4[0]).abs() < 1e-2);
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
        let mut q4_block = [0u8; 18];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        q4_block[0] = d_bytes[0];
        q4_block[1] = d_bytes[1];
        q4_block[2..18].fill(0x99);

        let f32_weight = [0.5f32; 32];
        let f32_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                f32_weight.as_ptr() as *const u8,
                std::mem::size_of_val(&f32_weight),
            )
        };

        let x = [1.0f32; 32];
        let mut q4_out = [0.0f32; 1];
        let mut f32_out = [0.0f32; 1];
        let mut expected_q4 = [0.0f32; 1];
        let mut expected_f32 = [0.0f32; 1];

        cpu.dequant_matmul(&q4_block, GgmlType::Q4_0, &x, &mut expected_q4, 1, 1, k);
        cpu.dequant_matmul(f32_bytes, GgmlType::F32, &x, &mut expected_f32, 1, 1, k);

        backend.device.reset_perf_counters();
        backend.safe_batch_dequant_matvec(
            &[
                (&q4_block, GgmlType::Q4_0, 1),
                (f32_bytes, GgmlType::F32, 1),
            ],
            &x,
            k,
            &mut [&mut q4_out, &mut f32_out],
        );
        let counters = backend.device.perf_counters();

        assert!((q4_out[0] - expected_q4[0]).abs() < 1e-2);
        assert!((f32_out[0] - expected_f32[0]).abs() < 1e-3);
        assert_eq!(
            counters.command_buffers, 1,
            "safe mixed-dtype batch matvec should use one command buffer"
        );
    }

    #[test]
    fn test_metal_backend_fused_q4_0_matvec() {
        let backend = MetalBackend::new().unwrap();

        // Create Q4_0 data: 2 rows × 32 cols (1 block per row)
        let m = 2;
        let k = 32;
        let mut quant_data = Vec::new();
        for row in 0..m {
            let d = half::f16::from_f32((row as f32 + 1.0) * 0.5).to_le_bytes();
            let mut block = vec![0u8; 18];
            block[0] = d[0];
            block[1] = d[1];
            block[2..18].fill(0x99); // nibble 9: (9-8)=1
            quant_data.extend(block);
        }

        // Input vector
        let x: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();

        // CPU reference: dequant then matmul
        let mut weights = vec![0.0f32; m * k];
        crate::quant::q4_0::dequantize(&quant_data, &mut weights);
        let mut expected = vec![0.0f32; m];
        crate::compute::matmul::matmul_f32(&weights, &x, &mut expected, m, 1, k);

        // GPU fused dequant+matvec
        let mut result = vec![0.0f32; m];
        backend.dequant_matmul(&quant_data, GgmlType::Q4_0, &x, &mut result, m, 1, k);

        let diff = max_abs_diff(&result, &expected);
        assert!(
            diff < 1e-2,
            "Fused Q4_0 matvec mismatch: max_diff={diff}, result={result:?}, expected={expected:?}"
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

        let first_ptrs = {
            let cache = backend.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
            assert_eq!(cache.len(), 1);
            let buffers = cache.get(&slot_key).unwrap();
            (
                buffers.conv_state.ptr_id(),
                buffers.recurrent_state.ptr_id(),
            )
        };
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
        assert_eq!(cache.len(), 1);
        let buffers = cache.get(&slot_key).unwrap();
        assert_eq!(buffers.conv_state.ptr_id(), first_ptrs.0);
        assert_eq!(buffers.recurrent_state.ptr_id(), first_ptrs.1);
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
    fn test_metal_backend_qwen35_recurrent_sequence_for_kv_multi_token_matches_cpu_without_slot_cache()
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
            backend
                .ops
                .qwen35_recurrent_slot_buffers
                .lock()
                .unwrap()
                .is_empty(),
            "multi-token qwen35 recurrent KV path should bypass cached Metal slot buffers"
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
    fn test_metal_backend_qwen35_recurrent_sequence_for_kv_multi_slot_multi_token_syncs_backend_owned_state()
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
            !actual_kv.recurrent_state_cpu_stale(0, layer_idx),
            "multi-token slot 0 path should materialize recurrent state on CPU after syncing from backend"
        );
        assert!(
            !actual_kv.recurrent_state_cpu_stale(slot1, layer_idx),
            "multi-token slot 1 path should materialize recurrent state on CPU after syncing from backend"
        );
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
