pub mod cpu;
pub mod kv_plan;
pub mod metal;
pub mod neon;
pub mod policy;

use crate::gguf::tensor::GgmlType;
use crate::model::config::ModelConfig;

pub use kv_plan::{
    CpuKvPlan, GpuKvPlan, KvCapacityPolicy, KvPlan, KvPlanKind, KvPlanMemoryEstimate, KvPlanner,
    KvPlannerRequirements, KvRollbackPolicy, Qwen35KvPlan,
};
pub use policy::{KvPrecisionPolicy, RuntimePolicy};

/// Backend trait for compute dispatch.
///
/// Provides matmul and fused dequant+matmul operations.
/// CpuBackend uses Accelerate BLAS. MetalBackend offloads to GPU.
pub trait Backend {
    /// Configure backend-local model policy after GGUF metadata is available.
    ///
    /// CPU backends ignore this. Metal-backed implementations should treat
    /// `model_name`, `quant`, and `architecture` as the source for any
    /// backend-local policy resolution.
    fn configure_for_model(
        &self,
        _model_name: &str,
        _quant: &str,
        _architecture: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Return the resolved backend-local runtime policy when available.
    ///
    /// Metal-backed implementations use this to expose the typed policy object
    /// that already owns routing and KV precision decisions.
    fn runtime_policy(&self) -> Option<RuntimePolicy> {
        None
    }

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

    /// Qwen3.5 recurrent GDN primitive for one or more sequence tokens.
    ///
    /// The default implementation uses the scalar CPU helper. GPU backends can
    /// override this with sequence-aware recurrent kernels later.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_recurrent_sequence(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        conv_state: &mut [f32],
        recurrent_state: &mut [f32],
        output_batch: &mut [f32],
        n_tokens: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        crate::compute::gdn::qwen35_recurrent_sequence(
            qkv_batch,
            beta_batch,
            alpha_batch,
            dt_bias,
            a,
            conv_kernel,
            conv_state,
            recurrent_state,
            output_batch,
            n_tokens,
            cfg,
        );
    }

    /// Qwen3.5 depthwise causal conv sequence primitive.
    #[allow(clippy::too_many_arguments)]
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
        crate::compute::gdn::depthwise_conv1d_sequence(
            input_batch,
            kernel,
            conv_state,
            output_batch,
            n_tokens,
            conv_cache_len,
            conv_dim,
        );
    }

    /// Qwen3.5 gated-delta recurrence primitive.
    #[allow(clippy::too_many_arguments)]
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
        crate::compute::gdn::gated_delta_rule_sequence(
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

/// Create the `ModelKv` appropriate for a given model/backend pair.
///
/// This is the single constructor path for current KV allocation semantics.
/// The planner decides the variant and allocation policy; the plan then builds
/// the final `ModelKv`, including the current GPU-allocation fallback to CPU.
pub fn create_model_kv(backend: &dyn Backend, config: &ModelConfig) -> crate::kv::ModelKv {
    KvPlanner::plan(backend, config).build(backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;
    use crate::model::config::{GateActivation, RopeScaling};

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            architecture: "llama".into(),
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            embedding_dim: 4096,
            head_dim: 128,
            intermediate_dim: 11008,
            context_length: 4096,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
        }
    }

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

    #[test]
    fn test_create_model_kv_uses_cpu_kv_for_hybrid_cpu_decode() {
        let backend = create_backend(BackendConfig::HybridCpuDecode).unwrap();
        let kv = create_model_kv(backend.as_ref(), &test_model_config());
        assert!(matches!(kv, crate::kv::ModelKv::Cpu(_)));
    }
}
