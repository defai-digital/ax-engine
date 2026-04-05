//! Backend abstraction and shared helpers for CPU/Metal execution.

pub mod cpu;
pub mod kv_plan;
pub mod metal;
pub mod neon;
pub mod policy;
pub mod qwen3_5;

use std::sync::{Mutex, OnceLock};

use crate::gguf::tensor::GgmlType;
use crate::kv::Qwen3_5Kv;
use crate::model::{ModelFingerprint, config::ModelConfig};

pub use kv_plan::{
    CpuKvPlan, GpuKvPlan, KvCapacityPolicy, KvPlan, KvPlanKind, KvPlanMemoryEstimate, KvPlanner,
    KvPlannerRequirements, KvRollbackPolicy, Qwen3_5KvPlan,
};
pub use policy::{KvPrecisionPolicy, RuntimePolicy};
#[allow(non_camel_case_types)]
pub type Qwen35KvPlan = Qwen3_5KvPlan;

#[derive(Default)]
struct TokenMajorMatmulScratch {
    input_t: Vec<f32>,
    output_mn: Vec<f32>,
}

fn token_major_matmul_scratch() -> &'static Mutex<TokenMajorMatmulScratch> {
    static SCRATCH: OnceLock<Mutex<TokenMajorMatmulScratch>> = OnceLock::new();
    SCRATCH.get_or_init(|| Mutex::new(TokenMajorMatmulScratch::default()))
}

pub use qwen3_5::{
    Qwen3_5RecurrentStateBatch, Qwen35RecurrentStateBatch, qwen3_5_recurrent_sequence_via_backend,
    qwen35_recurrent_sequence_via_backend,
};

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

    /// Configure backend-local model policy from a richer fingerprint.
    ///
    /// Metal-backed implementations should prefer this over
    /// `configure_for_model`, since it carries the model shape and quant
    /// layout needed for persistent tuning and cache-safe policy selection.
    fn configure_for_fingerprint(&self, fingerprint: &ModelFingerprint) -> anyhow::Result<()> {
        self.configure_for_model(
            &fingerprint.model_name,
            &fingerprint.predominant_quant,
            &fingerprint.architecture,
        )
    }

    /// Return the resolved backend-local runtime policy when available.
    ///
    /// Metal-backed implementations use this to expose the typed policy object
    /// that already owns routing and KV precision decisions.
    fn runtime_policy(&self) -> Option<RuntimePolicy> {
        None
    }

    /// Observe which prepared recurrent-state batch shape was used for Qwen3.5.
    ///
    /// Backends with performance instrumentation can override this to
    /// distinguish backend-native slot-buffer execution from CPU direct and
    /// gathered fallback paths.
    fn note_qwen35_prepared_state_batch_kind(
        &self,
        _kind: crate::kv::qwen35_kv::Qwen3_5PreparedRecurrentStateBatchKind,
    ) {
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

    /// Safe batched decode matvec: computes y_i = W_i × x for one or more
    /// projections that share the same single-token input vector.
    ///
    /// Default behavior reuses the standard grouped decode route. Metal-backed
    /// implementations may override this to avoid fused quantized `n=1` kernels
    /// while still batching multiple projections into one command buffer.
    #[allow(clippy::too_many_arguments)]
    fn safe_batch_dequant_matvec(
        &self,
        ops: &[(&[u8], GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        self.batch_dequant_matvec(ops, x, k, outputs);
    }

    /// Batched dequant matmul with token-major layout.
    ///
    /// Computes `output[n_tokens × out_dim] = input[n_tokens × in_dim] × W^T`
    /// where W = dequant(a_quant) is `[out_dim × in_dim]`.
    ///
    /// Both input and output are in token-major (row-major `[n_tokens × dim]`)
    /// layout. The default implementation transposes to column-major for the
    /// standard `dequant_matmul` path. GPU backends override with fused kernels
    /// whose Metal shaders natively accept token-major B `[N×K]`.
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
        let mut scratch = token_major_matmul_scratch()
            .lock()
            .expect("token-major matmul scratch mutex should not be poisoned");
        let TokenMajorMatmulScratch { input_t, output_mn } = &mut *scratch;
        input_t.resize(n_tokens * in_dim, 0.0);
        output_mn.resize(out_dim * n_tokens, 0.0);

        // Default: transpose to column-major, matmul, transpose back.
        for row in 0..n_tokens {
            for col in 0..in_dim {
                input_t[col * n_tokens + row] = input_token_major[row * in_dim + col];
            }
        }
        self.dequant_matmul(
            a_quant, dtype, input_t, output_mn, out_dim, n_tokens, in_dim,
        );
        for row in 0..out_dim {
            for col in 0..n_tokens {
                output_token_major[col * out_dim + row] = output_mn[row * n_tokens + col];
            }
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
        state_batch: &mut Qwen3_5RecurrentStateBatch<'_>,
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        qwen35_recurrent_sequence_via_backend(
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
    }

    /// Qwen3.5 recurrent GDN primitive operating directly on the hybrid KV.
    ///
    /// This keeps `Qwen3_5Forward` agnostic to how a backend stages recurrent
    /// state. Backends can override this to use backend-owned mirrors or
    /// device-resident pools without forcing another model-side refactor.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_recurrent_sequence_for_kv(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        qwen_kv: &mut Qwen3_5Kv,
        layer_idx: usize,
        slot_indices: &[usize],
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        let mut prepared_state_batch =
            qwen_kv.prepare_recurrent_state_batch(slot_indices, layer_idx);
        self.note_qwen35_prepared_state_batch_kind(prepared_state_batch.kind());
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

    /// Materialize any backend-owned Qwen3.5 recurrent state back into the KV.
    ///
    /// Most backends keep Qwen3.5 recurrent state in CPU-owned slices, so the
    /// default implementation is a no-op. Backends with slot-buffer mirrors can
    /// override this to flush the latest recurrent state before snapshotting or
    /// other host-side control paths.
    fn sync_qwen35_kv(&self, _qwen_kv: &mut Qwen3_5Kv) {}

    /// Attempt to clone a Qwen3.5 recurrent slot without first materializing
    /// backend-owned recurrent state back onto CPU.
    ///
    /// The default implementation returns `false`, which tells callers to fall
    /// back to `sync_qwen35_kv()` followed by a CPU-visible snapshot/restore.
    fn try_clone_qwen35_recurrent_slot(
        &self,
        _qwen_kv: &mut Qwen3_5Kv,
        _src_slot_idx: usize,
        _dst_slot_idx: usize,
    ) -> bool {
        false
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

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"))
}

/// Resolve the runtime backend configuration from supported environment overrides.
///
/// `AX_CPU_ONLY=1` forces [`BackendConfig::Cpu`].
/// `AX_HYBRID_DECODE=cpu` forces [`BackendConfig::HybridCpuDecode`].
/// Otherwise the production default [`BackendConfig::Hybrid`] is used.
pub fn resolve_backend_config_from_env() -> BackendConfig {
    if env_flag_enabled("AX_CPU_ONLY") {
        return BackendConfig::Cpu;
    }

    match std::env::var("AX_HYBRID_DECODE")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("cpu") => return BackendConfig::HybridCpuDecode,
        Some("metal") | Some("gpu") => return BackendConfig::default(),
        _ => {}
    }

    // Default: Hybrid (GPU prefill + GPU decode).
    //
    // This keeps the CLI/runtime aligned with the production fast path and with
    // `BackendConfig::default()`. Users can still force the older CPU-decode
    // path explicitly via `AX_HYBRID_DECODE=cpu`.
    BackendConfig::Hybrid
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
mod tests;
