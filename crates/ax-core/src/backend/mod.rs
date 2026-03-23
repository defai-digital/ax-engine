pub mod cpu;
pub mod metal;
pub mod neon;

use crate::gguf::tensor::GgmlType;

/// Backend trait for compute dispatch.
///
/// Provides matmul and fused dequant+matmul operations.
/// CpuBackend uses Accelerate BLAS. MetalBackend offloads to GPU.
pub trait Backend {
    /// Matrix multiply: C = A × B (all f32, row-major).
    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize);

    /// Dequantize quantized weights and multiply: C = dequant(A) × B.
    ///
    /// Default: dequantize to f32 on CPU, then call matmul.
    /// MetalBackend overrides with fused GPU kernel for supported types.
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
        let mut a_f32 = vec![0.0f32; m * k];
        crate::quant::dequantize(dtype, a_quant, &mut a_f32);
        self.matmul(&a_f32, b, c, m, n, k);
    }

    /// Batched dequant+matvec: computes y_i = dequant(A_i) × x for multiple
    /// weight matrices sharing the same input vector x. All N=1 (decode).
    #[allow(clippy::too_many_arguments)]
    fn batch_dequant_matvec(
        &self,
        ops: &[(&[u8], GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        debug_assert_eq!(ops.len(), outputs.len());
        for (i, (a_quant, dtype, m)) in ops.iter().enumerate() {
            self.dequant_matmul(a_quant, *dtype, x, outputs[i], *m, 1, k);
        }
    }

    /// Multi-head attention for prefill (multiple tokens with causal masking).
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
        let params = crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim);
        crate::compute::attention::multi_head_attention_prefill(q, k, v, output, n_tokens, &params);
    }

    /// Access Metal GPU operations for GPU-accelerated forward pass dispatch.
    ///
    /// Returns `None` for CPU-only backends. Metal and Hybrid backends return `Some`.
    fn metal_ops(&self) -> Option<&metal::MetalOps> {
        None
    }

    /// Whether this backend routes single-token decode (n=1) to the GPU.
    ///
    /// This is the single source of truth for `forward_batch`: if this returns
    /// `false`, `forward_batch` must NOT use the GPU batch path, because the
    /// forward pass will use CPU decode. Using GPU batch prefill with CPU decode
    /// is architecturally unsound (KV data ends up in GPU buffers only).
    ///
    /// Default: true iff `metal_ops()` is Some.
    fn use_gpu_decode(&self) -> bool {
        self.metal_ops().is_some()
    }
}

/// Backend configuration.
///
/// Controls which compute path is used for prefill (multi-token) and decode (single-token).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendConfig {
    /// CPU only (Accelerate BLAS). Both prefill and decode run on CPU.
    Cpu,
    /// Full Metal GPU. Both prefill and decode run on GPU.
    Metal,
    /// Hybrid: prefill (N > 1) on Metal, decode (N = 1) on Metal.
    ///
    /// This is the production default. Both prefill and decode use GPU command
    /// buffers. The KV cache is GPU-resident (`ModelKv::Gpu`).
    ///
    /// Note: in v1 this was described as "CPU decode" but the actual runtime
    /// default was Metal decode since Phase 1. v2 fixes the doc to match reality.
    #[default]
    Hybrid,
    /// Hybrid CPU decode: prefill (N > 1) on Metal, decode (N = 1) on CPU.
    ///
    /// Prefill uses serial `forward_single` (not the GPU batch path), because
    /// the KV cache is CPU-resident (`ModelKv::Cpu`) and GPU batch prefill
    /// cannot write to a CPU KV cache.
    ///
    /// Use this for latency-critical single-token experiments or debugging.
    HybridCpuDecode,
}

/// Create a boxed backend from configuration.
pub fn create_backend(config: BackendConfig) -> anyhow::Result<Box<dyn Backend>> {
    match config {
        BackendConfig::Cpu => Ok(Box::new(cpu::CpuBackend)),
        BackendConfig::Metal => Ok(Box::new(metal::MetalBackend::new()?)),
        BackendConfig::Hybrid => Ok(Box::new(metal::HybridBackend::new()?)),
        BackendConfig::HybridCpuDecode => Ok(Box::new(metal::HybridCpuDecodeBackend::new()?)),
    }
}

/// Create the `ModelKv` appropriate for a given backend configuration.
///
/// The KV variant must match the backend's `use_gpu_decode()` return value —
/// GPU decode requires `ModelKv::Gpu`, CPU decode requires `ModelKv::Cpu`.
pub fn create_model_kv(
    config: BackendConfig,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    page_size: usize,
    device: Option<&ax_metal::MetalDevice>,
) -> anyhow::Result<crate::kv::ModelKv> {
    use crate::kv::gpu_kv::GpuKvDtype;
    use crate::kv::page::{KvCacheConfig, KvDtype};
    use crate::kv::{CpuKv, GpuKv, ModelKv};

    match config {
        BackendConfig::Cpu | BackendConfig::HybridCpuDecode => {
            let cfg = KvCacheConfig {
                n_layers,
                n_kv_heads,
                head_dim,
                max_seq_len,
                page_size,
                dtype: KvDtype::F32,
            };
            Ok(ModelKv::Cpu(CpuKv::with_config(&cfg)))
        }
        BackendConfig::Metal | BackendConfig::Hybrid => {
            let device =
                device.ok_or_else(|| anyhow::anyhow!("Metal device required for GPU KV"))?;
            // Auto-select f16 KV for long contexts (same policy as v1)
            let dtype = match std::env::var("AX_METAL_F16_KV_CACHE").as_deref() {
                Ok("off") => GpuKvDtype::F32,
                Ok("on" | "1") => GpuKvDtype::F16,
                _ /* auto */ => {
                    if max_seq_len >= 256 { GpuKvDtype::F16 } else { GpuKvDtype::F32 }
                }
            };
            let gpu_kv = GpuKv::new_with_dtype(
                device,
                n_layers,
                n_kv_heads,
                head_dim,
                max_seq_len,
                page_size,
                dtype,
            )?;
            Ok(ModelKv::Gpu(gpu_kv))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_default() {
        assert_eq!(BackendConfig::default(), BackendConfig::Hybrid);
    }

    #[test]
    fn test_create_cpu_backend() {
        let backend = create_backend(BackendConfig::Cpu).unwrap();
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        backend.matmul(&a, &b, &mut c, 2, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-4);
    }

    #[test]
    fn test_hybrid_cpu_decode_does_not_use_gpu_decode() {
        let backend = create_backend(BackendConfig::HybridCpuDecode).unwrap();
        assert!(
            !backend.use_gpu_decode(),
            "HybridCpuDecode must return use_gpu_decode()=false"
        );
    }

    #[test]
    fn test_hybrid_uses_gpu_decode() {
        let backend = create_backend(BackendConfig::Hybrid).unwrap();
        assert!(
            backend.use_gpu_decode(),
            "Hybrid must return use_gpu_decode()=true"
        );
    }
}
