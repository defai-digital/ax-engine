use rustc_hash::FxHashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::{Backend, RuntimePolicy};
use crate::gguf::tensor::GgmlType;
use anyhow::Context;

// v2: GpuKv is owned by LlamaModel via ModelKv, not stored in MetalOps.
use crate::model::config::ModelConfig;
use ax_metal::{
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
/// - `1` / `true` / `on` -> enabled
/// - unset / any other value -> disabled (default)
pub fn metal_q8_batch_native_enabled() -> bool {
    RuntimePolicy::resolved_defaults().q8_batch_native_enabled()
}

/// Whether native Q8_0 batch kernel should be used for a specific shape.
///
/// Requires `AX_METAL_Q8_BATCH_NATIVE=1` to be globally enabled, then applies
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
/// 2) `AX_METAL_BATCH_SIMD` (via `ax_metal::batch_simd_enabled()`)
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
    /// Cache of Metal buffers for weight tensors, keyed by data pointer address.
    /// Weight data from mmap'd GGUF files has stable addresses across calls.
    weight_cache: Mutex<FxHashMap<usize, MetalBuffer>>,
    /// Pre-allocated input vector buffer (grows on demand).
    input_buf: Mutex<(Option<MetalBuffer>, usize)>,
    /// Pre-allocated output vector buffer (grows on demand).
    output_buf: Mutex<(Option<MetalBuffer>, usize)>,
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
            input_buf: Mutex::new((None, 0)),
            output_buf: Mutex::new((None, 0)),
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

        // Check if all ops use the same quant type for batched GPU dispatch
        let first_dtype = ops[0].1;
        let all_same_dtype = ops.iter().all(|(_, dtype, _)| *dtype == first_dtype);
        let can_batch = all_same_dtype
            && (first_dtype == GgmlType::Q4_0
                || first_dtype == GgmlType::Q8_0
                || first_dtype == GgmlType::Q4K
                || first_dtype == GgmlType::Q5K
                || first_dtype == GgmlType::Q6K)
            && ops.iter().all(|(_, _, _)| true); // n=1 implied

        if !can_batch || ops.len() < 2 {
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
            cache.entry(key).or_insert_with(|| {
                MetalBuffer::from_bytes(self.device.device(), a_quant)
                    .expect("Failed to create Metal buffer for weight tensor")
            });
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
                for (i, (_, _, m)) in ops.iter().enumerate() {
                    let buf_a = cache.get(&weight_keys[i]).unwrap();
                    match first_dtype {
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
                        _ => unreachable!(),
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
    for head in 0..n_heads {
        for token_idx in 0..n_tokens {
            let src = head * n_tokens * head_dim + token_idx * head_dim;
            let dst = token_idx * n_heads * head_dim + head * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
        }
    }
}

impl MetalBackend {
    /// Get or create a cached Metal buffer for weight data.
    ///
    /// Weight tensors from mmap'd GGUF files have stable pointer addresses,
    /// so we use the data pointer as the cache key. On first access the data
    /// is copied into a Metal shared-mode buffer; subsequent calls return
    /// the cached buffer with zero overhead.
    fn get_weight_buffer(
        &self,
        data: &[u8],
    ) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        let key = data.as_ptr() as usize;
        let mut cache = self.weight_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            MetalBuffer::from_bytes(self.device.device(), data)
                .expect("Failed to create Metal buffer for weight tensor")
        });
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
        let runtime_policy = RuntimePolicy::resolved_defaults();
        Ok(Self {
            device,
            elementwise,
            dequant,
            attention,
            matmul,
            runtime_policy: RwLock::new(runtime_policy),
            f32_weight_cache: Mutex::new(FxHashMap::default()),
            fused_qkv_weight_cache: Mutex::new(FxHashMap::default()),
            precomputed_f16_weight_cache: Mutex::new(FxHashMap::default()),
            scratches: Mutex::new(None),
            batch_scratches: Mutex::new(None),
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

        let vocab_size = config.vocab_size as usize;
        let scratches = GpuScratchBuffers {
            hidden: alloc(dim),
            norm_buf: alloc(dim),
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

    /// Get or create a cached MetalBuffer for an f32 weight slice.
    /// Keyed by the data pointer address (stable for mmap'd weights).
    pub fn get_f32_weight_buffer(
        &self,
        data: &[f32],
    ) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        let key = data.as_ptr() as usize;
        let mut cache = self.f32_weight_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            MetalBuffer::from_slice(self.device.device(), data)
                .expect("Failed to create Metal buffer for f32 weight")
        });
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
        cache.entry(key).or_insert_with(|| {
            MetalBuffer::from_bytes(self.device.device(), data)
                .expect("Failed to create Metal buffer for quantized weight")
        });
        cache
    }

    /// Ensure an f32 weight slice is cached as a MetalBuffer.
    /// Returns the cache key (pointer address) for later lookup via `lock_weight_cache`.
    pub fn ensure_f32_cached(&self, data: &[f32]) -> usize {
        let key = data.as_ptr() as usize;
        let mut cache = self.f32_weight_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            MetalBuffer::from_slice(self.device.device(), data)
                .expect("Failed to create Metal buffer for f32 weight")
        });
        key
    }

    /// Ensure quantized weight data is cached as a MetalBuffer.
    /// Returns the cache key (pointer address) for later lookup via `lock_weight_cache`.
    pub fn ensure_quant_cached(&self, data: &[u8]) -> usize {
        let key = data.as_ptr() as usize;
        let mut cache = self.f32_weight_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            MetalBuffer::from_bytes(self.device.device(), data)
                .expect("Failed to create Metal buffer for quantized weight")
        });
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
        encoder: &ax_metal::MetalEncoder,
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
        encoder: &ax_metal::MetalEncoder,
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
    pub fn encode_precomputed_q4k_matvec_if_available(
        &self,
        encoder: &ax_metal::MetalEncoder,
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

    fn batch_dequant_matvec(
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
