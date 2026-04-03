pub mod cpu;
pub mod kv_plan;
pub mod metal;
pub mod neon;
pub mod policy;

use std::sync::{Mutex, OnceLock};

use crate::gguf::tensor::GgmlType;
use crate::kv::Qwen35Kv;
use crate::model::{ModelFingerprint, config::ModelConfig};

pub use kv_plan::{
    CpuKvPlan, GpuKvPlan, KvCapacityPolicy, KvPlan, KvPlanKind, KvPlanMemoryEstimate, KvPlanner,
    KvPlannerRequirements, KvRollbackPolicy, Qwen35KvPlan,
};
pub use policy::{KvPrecisionPolicy, RuntimePolicy};

#[derive(Default)]
struct TokenMajorMatmulScratch {
    input_t: Vec<f32>,
    output_mn: Vec<f32>,
}

fn token_major_matmul_scratch() -> &'static Mutex<TokenMajorMatmulScratch> {
    static SCRATCH: OnceLock<Mutex<TokenMajorMatmulScratch>> = OnceLock::new();
    SCRATCH.get_or_init(|| Mutex::new(TokenMajorMatmulScratch::default()))
}

/// Slot-indexed recurrent-state batch for Qwen3.5 hybrid layers.
///
/// The batch is expressed as one or more live recurrent slots, each with an
/// equal number of new tokens. The underlying state buffers stay flattened for
/// the current backend implementations, but slot ownership is explicit at the
/// API boundary so the runtime can evolve toward true `state_indices` execution
/// without another backend refactor.
#[derive(Debug)]
pub struct Qwen35RecurrentStateBatch<'a> {
    layer_idx: usize,
    slot_indices: &'a [usize],
    conv_state_batch: &'a mut [f32],
    recurrent_state_batch: &'a mut [f32],
    conv_state_stride: usize,
    recurrent_state_stride: usize,
}

impl<'a> Qwen35RecurrentStateBatch<'a> {
    pub fn new(
        layer_idx: usize,
        slot_indices: &'a [usize],
        conv_state_batch: &'a mut [f32],
        recurrent_state_batch: &'a mut [f32],
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) -> Self {
        assert!(
            !slot_indices.is_empty(),
            "qwen35 recurrent state batch requires at least one slot"
        );
        assert!(
            conv_state_stride > 0,
            "qwen35 recurrent conv state stride must be > 0"
        );
        assert!(
            recurrent_state_stride > 0,
            "qwen35 recurrent state stride must be > 0"
        );
        for (i, &slot_idx) in slot_indices.iter().enumerate() {
            for &other_slot_idx in &slot_indices[..i] {
                assert!(
                    slot_idx != other_slot_idx,
                    "qwen35 recurrent state batch must not contain duplicate slots"
                );
            }
        }
        assert_eq!(
            conv_state_batch.len(),
            slot_indices.len() * conv_state_stride,
            "qwen35 recurrent conv state batch has wrong length"
        );
        assert_eq!(
            recurrent_state_batch.len(),
            slot_indices.len() * recurrent_state_stride,
            "qwen35 recurrent state batch has wrong length"
        );
        Self {
            layer_idx,
            slot_indices,
            conv_state_batch,
            recurrent_state_batch,
            conv_state_stride,
            recurrent_state_stride,
        }
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn slot_count(&self) -> usize {
        self.slot_indices.len()
    }

    pub fn slot_indices(&self) -> &[usize] {
        self.slot_indices
    }

    pub fn slot_index(&self, batch_idx: usize) -> usize {
        self.slot_indices[batch_idx]
    }

    pub fn conv_state_stride(&self) -> usize {
        self.conv_state_stride
    }

    pub fn recurrent_state_stride(&self) -> usize {
        self.recurrent_state_stride
    }

    pub fn conv_state_for_slot_mut(&mut self, batch_idx: usize) -> &mut [f32] {
        let start = batch_idx * self.conv_state_stride;
        let end = start + self.conv_state_stride;
        &mut self.conv_state_batch[start..end]
    }

    pub fn recurrent_state_for_slot_mut(&mut self, batch_idx: usize) -> &mut [f32] {
        let start = batch_idx * self.recurrent_state_stride;
        let end = start + self.recurrent_state_stride;
        &mut self.recurrent_state_batch[start..end]
    }

    pub fn recurrent_buffers_for_slot_mut(&mut self, batch_idx: usize) -> (&mut [f32], &mut [f32]) {
        let conv_start = batch_idx * self.conv_state_stride;
        let conv_end = conv_start + self.conv_state_stride;
        let rec_start = batch_idx * self.recurrent_state_stride;
        let rec_end = rec_start + self.recurrent_state_stride;
        (
            &mut self.conv_state_batch[conv_start..conv_end],
            &mut self.recurrent_state_batch[rec_start..rec_end],
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_recurrent_sequence_via_backend(
    backend: &(impl Backend + ?Sized),
    qkv_batch: &[f32],
    beta_batch: &mut [f32],
    alpha_batch: &mut [f32],
    dt_bias: &[f32],
    a: &[f32],
    conv_kernel: &[f32],
    state_batch: &mut Qwen35RecurrentStateBatch<'_>,
    output_batch: &mut [f32],
    tokens_per_slot: usize,
    cfg: crate::compute::gdn::Qwen35RecurrentConfig,
) {
    assert!(
        tokens_per_slot > 0,
        "qwen35 recurrent sequence requires tokens_per_slot > 0"
    );
    let slot_count = state_batch.slot_count();
    let total_tokens = slot_count * tokens_per_slot;
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
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
        state_batch.conv_state_stride(),
        cfg.conv_cache_len * cfg.conv_dim,
        "qwen35 recurrent conv state stride mismatch"
    );
    assert_eq!(
        state_batch.recurrent_state_stride(),
        cfg.time_step_rank * cfg.state_size * cfg.state_size,
        "qwen35 recurrent state stride mismatch"
    );

    let mut conv_out_batch = vec![0.0f32; total_tokens * cfg.conv_dim];
    let mut q_batch = vec![0.0f32; total_tokens * value_dim];
    let mut k_batch = vec![0.0f32; total_tokens * value_dim];
    let mut v_batch = vec![0.0f32; total_tokens * value_dim];

    crate::compute::gdn::prepare_alpha_beta(alpha_batch, beta_batch, dt_bias, a);

    for batch_idx in 0..slot_count {
        let _slot_idx = state_batch.slot_index(batch_idx);
        let token_start = batch_idx * tokens_per_slot;
        let token_end = token_start + tokens_per_slot;
        let qkv_start = token_start * cfg.conv_dim;
        let qkv_end = token_end * cfg.conv_dim;
        backend.qwen35_causal_conv_sequence(
            &qkv_batch[qkv_start..qkv_end],
            conv_kernel,
            state_batch.conv_state_for_slot_mut(batch_idx),
            &mut conv_out_batch[qkv_start..qkv_end],
            tokens_per_slot,
            cfg.conv_cache_len,
            cfg.conv_dim,
        );
    }

    let mut q_lin = vec![0.0f32; key_dim];
    let mut k_lin = vec![0.0f32; key_dim];
    let mut q_rep = vec![0.0f32; value_dim];
    let mut k_rep = vec![0.0f32; value_dim];

    for token_idx in 0..total_tokens {
        let conv_start = token_idx * cfg.conv_dim;
        let conv_end = conv_start + cfg.conv_dim;
        let conv_out = &conv_out_batch[conv_start..conv_end];
        q_lin.copy_from_slice(&conv_out[..key_dim]);
        k_lin.copy_from_slice(&conv_out[key_dim..2 * key_dim]);
        let v_lin = &conv_out[2 * key_dim..2 * key_dim + value_dim];

        crate::compute::gdn::l2_norm_heads(
            &mut q_lin,
            cfg.group_count,
            cfg.state_size,
            cfg.rms_norm_eps,
        );
        crate::compute::gdn::l2_norm_heads(
            &mut k_lin,
            cfg.group_count,
            cfg.state_size,
            cfg.rms_norm_eps,
        );

        crate::compute::gdn::repeat_heads_into(
            &mut q_rep,
            &q_lin,
            cfg.group_count,
            cfg.time_step_rank,
            cfg.state_size,
        );
        crate::compute::gdn::repeat_heads_into(
            &mut k_rep,
            &k_lin,
            cfg.group_count,
            cfg.time_step_rank,
            cfg.state_size,
        );
        let out_start = token_idx * value_dim;
        let out_end = out_start + value_dim;
        q_batch[out_start..out_end].copy_from_slice(&q_rep);
        k_batch[out_start..out_end].copy_from_slice(&k_rep);
        v_batch[out_start..out_end].copy_from_slice(v_lin);
    }

    for batch_idx in 0..slot_count {
        let _slot_idx = state_batch.slot_index(batch_idx);
        let token_start = batch_idx * tokens_per_slot;
        let token_end = token_start + tokens_per_slot;
        let qkv_start = token_start * value_dim;
        let qkv_end = token_end * value_dim;
        let gate_start = token_start * cfg.time_step_rank;
        let gate_end = token_end * cfg.time_step_rank;
        backend.qwen35_gated_delta_sequence(
            &q_batch[qkv_start..qkv_end],
            &k_batch[qkv_start..qkv_end],
            &v_batch[qkv_start..qkv_end],
            &alpha_batch[gate_start..gate_end],
            &beta_batch[gate_start..gate_end],
            state_batch.recurrent_state_for_slot_mut(batch_idx),
            &mut output_batch[qkv_start..qkv_end],
            tokens_per_slot,
            cfg.time_step_rank,
            cfg.state_size,
        );
    }
}

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
        _kind: crate::kv::qwen35_kv::Qwen35PreparedRecurrentStateBatchKind,
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
        state_batch: &mut Qwen35RecurrentStateBatch<'_>,
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
    /// This keeps `Qwen35Forward` agnostic to how a backend stages recurrent
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
        qwen_kv: &mut Qwen35Kv,
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
    fn sync_qwen35_kv(&self, _qwen_kv: &mut Qwen35Kv) {}

    /// Attempt to clone a Qwen3.5 recurrent slot without first materializing
    /// backend-owned recurrent state back onto CPU.
    ///
    /// The default implementation returns `false`, which tells callers to fall
    /// back to `sync_qwen35_kv()` followed by a CPU-visible snapshot/restore.
    fn try_clone_qwen35_recurrent_slot(
        &self,
        _qwen_kv: &mut Qwen35Kv,
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

    // Default: HybridCpuDecode. GPU pipelined decode has a computation
    // error that affects all architectures (produces garbage output).
    // GPU batch prefill works correctly. Set AX_HYBRID_DECODE=metal to
    // force full GPU decode for benchmarking.
    BackendConfig::HybridCpuDecode
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
    use crate::backend::cpu::CpuBackend;
    use crate::model::ModelConfig;
    use crate::model::config::{GateActivation, RopeScaling};

    struct EnvVarRestore {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            match &self.previous {
                Some(prev) => unsafe { std::env::set_var(self.key, prev) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock()
    }

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
            expert_intermediate_dim: None,
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

    #[test]
    fn test_resolve_backend_config_from_env_defaults_to_hybrid() {
        let _guard = env_lock();
        let _restore_cpu = EnvVarRestore {
            key: "AX_CPU_ONLY",
            previous: std::env::var_os("AX_CPU_ONLY"),
        };
        let _restore_decode = EnvVarRestore {
            key: "AX_HYBRID_DECODE",
            previous: std::env::var_os("AX_HYBRID_DECODE"),
        };
        unsafe {
            std::env::remove_var("AX_CPU_ONLY");
            std::env::remove_var("AX_HYBRID_DECODE");
        }

        assert_eq!(resolve_backend_config_from_env(), BackendConfig::Hybrid);
    }

    #[test]
    fn test_resolve_backend_config_from_env_honors_cpu_only() {
        let _guard = env_lock();
        let _restore_cpu = EnvVarRestore {
            key: "AX_CPU_ONLY",
            previous: std::env::var_os("AX_CPU_ONLY"),
        };
        unsafe {
            std::env::set_var("AX_CPU_ONLY", "1");
        }

        assert_eq!(resolve_backend_config_from_env(), BackendConfig::Cpu);
    }

    #[test]
    fn test_resolve_backend_config_from_env_honors_hybrid_cpu_decode() {
        let _guard = env_lock();
        let _restore_decode = EnvVarRestore {
            key: "AX_HYBRID_DECODE",
            previous: std::env::var_os("AX_HYBRID_DECODE"),
        };
        unsafe {
            std::env::set_var("AX_HYBRID_DECODE", "cpu");
        }

        assert_eq!(
            resolve_backend_config_from_env(),
            BackendConfig::HybridCpuDecode
        );
    }

    #[test]
    fn test_resolve_backend_config_from_env_cpu_only_wins() {
        let _guard = env_lock();
        let _restore_cpu = EnvVarRestore {
            key: "AX_CPU_ONLY",
            previous: std::env::var_os("AX_CPU_ONLY"),
        };
        let _restore_decode = EnvVarRestore {
            key: "AX_HYBRID_DECODE",
            previous: std::env::var_os("AX_HYBRID_DECODE"),
        };
        unsafe {
            std::env::set_var("AX_CPU_ONLY", "1");
            std::env::set_var("AX_HYBRID_DECODE", "cpu");
        }

        assert_eq!(resolve_backend_config_from_env(), BackendConfig::Cpu);
    }

    #[test]
    fn test_qwen35_recurrent_state_batch_requires_matching_lengths() {
        let slot_indices = [0usize, 1usize];
        let mut conv_state = vec![0.0f32; 8];
        let mut recurrent_state = vec![0.0f32; 8];
        let batch = Qwen35RecurrentStateBatch::new(
            0,
            &slot_indices,
            &mut conv_state,
            &mut recurrent_state,
            4,
            4,
        );

        assert_eq!(batch.slot_count(), 2);
        assert_eq!(batch.layer_idx(), 0);
        assert_eq!(batch.slot_indices(), &slot_indices);
        assert_eq!(batch.conv_state_stride(), 4);
        assert_eq!(batch.recurrent_state_stride(), 4);
    }

    #[test]
    #[should_panic(expected = "qwen35 recurrent state batch must not contain duplicate slots")]
    fn test_qwen35_recurrent_state_batch_rejects_duplicate_slots() {
        let slot_indices = [0usize, 0usize];
        let mut conv_state = vec![0.0f32; 8];
        let mut recurrent_state = vec![0.0f32; 8];
        let _ = Qwen35RecurrentStateBatch::new(
            0,
            &slot_indices,
            &mut conv_state,
            &mut recurrent_state,
            4,
            4,
        );
    }

    #[test]
    #[should_panic(
        expected = "qwen35 recurrent slot batch slot 1 has seqlen_offset 1 != shared attention seq_len 0"
    )]
    fn test_qwen35_recurrent_sequence_for_kv_rejects_misaligned_slot_batch() {
        let backend = cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 2,
            conv_dim: 6,
            group_count: 1,
            state_size: 2,
            time_step_rank: 1,
            rms_norm_eps: 1e-5,
        };
        let mut kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1 = kv.allocate_recurrent_slot();
        kv.set_recurrent_seqlen_offset(slot1, 1);
        let slot_indices = [0usize, slot1];
        let tokens_per_slot = 1usize;
        let total_tokens = slot_indices.len() * tokens_per_slot;
        let qkv = vec![0.0f32; total_tokens * cfg.conv_dim];
        let mut alpha = vec![0.0f32; total_tokens * cfg.time_step_rank];
        let mut beta = vec![0.0f32; total_tokens * cfg.time_step_rank];
        let dt_bias = vec![0.0f32; cfg.time_step_rank];
        let a = vec![0.0f32; cfg.time_step_rank];
        let kernel = vec![0.0f32; (cfg.conv_cache_len + 1) * cfg.conv_dim];
        let mut out = vec![0.0f32; total_tokens * cfg.value_dim()];

        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut beta,
            &mut alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut kv,
            0,
            &slot_indices,
            &mut out,
            tokens_per_slot,
            cfg,
        );
    }

    #[test]
    fn test_qwen35_recurrent_sequence_via_backend_matches_per_slot_cpu_execution() {
        let backend = cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 2,
            conv_dim: 6,
            group_count: 1,
            state_size: 2,
            time_step_rank: 1,
            rms_norm_eps: 1e-5,
        };
        let slot_indices = [3usize, 7usize];
        let tokens_per_slot = 2;
        let total_tokens = slot_indices.len() * tokens_per_slot;
        let conv_state_stride = cfg.conv_cache_len * cfg.conv_dim;
        let recurrent_state_stride = cfg.time_step_rank * cfg.state_size * cfg.state_size;
        let qkv = vec![
            0.5, 0.1, -0.2, 0.7, 0.3, 0.4, //
            -0.1, 0.2, 0.6, 0.8, -0.4, 0.9, //
            0.2, -0.3, 0.1, -0.5, 0.6, 0.7, //
            0.8, 0.4, -0.6, 0.2, 0.1, -0.2,
        ];
        let alpha_input = vec![0.3, 0.4, 0.2, 0.1];
        let beta_input = vec![0.1, 0.2, 0.3, 0.4];
        let dt_bias = vec![0.05];
        let a = vec![0.7];
        let kernel = vec![
            0.1, 0.2, 0.1, 0.2, 0.1, 0.2, //
            0.2, 0.1, 0.2, 0.1, 0.2, 0.1, //
            0.3, 0.4, 0.3, 0.4, 0.3, 0.4,
        ];
        let mut conv_state = vec![0.0f32; slot_indices.len() * conv_state_stride];
        let mut recurrent_state = vec![0.0f32; slot_indices.len() * recurrent_state_stride];
        let mut alpha = alpha_input.clone();
        let mut beta = beta_input.clone();
        let mut out = vec![0.0f32; total_tokens * cfg.value_dim()];

        {
            let mut state_batch = Qwen35RecurrentStateBatch::new(
                0,
                &slot_indices,
                &mut conv_state,
                &mut recurrent_state,
                conv_state_stride,
                recurrent_state_stride,
            );
            backend.qwen35_recurrent_sequence(
                &qkv,
                &mut beta,
                &mut alpha,
                &dt_bias,
                &a,
                &kernel,
                &mut state_batch,
                &mut out,
                tokens_per_slot,
                cfg,
            );
        }

        for batch_idx in 0..slot_indices.len() {
            let token_start = batch_idx * tokens_per_slot;
            let token_end = token_start + tokens_per_slot;
            let qkv_start = token_start * cfg.conv_dim;
            let qkv_end = token_end * cfg.conv_dim;
            let gate_start = token_start * cfg.time_step_rank;
            let gate_end = token_end * cfg.time_step_rank;
            let out_start = token_start * cfg.value_dim();
            let out_end = token_end * cfg.value_dim();

            let mut expected_alpha = alpha_input[gate_start..gate_end].to_vec();
            let mut expected_beta = beta_input[gate_start..gate_end].to_vec();
            let mut expected_conv_state = vec![0.0f32; conv_state_stride];
            let mut expected_recurrent_state = vec![0.0f32; recurrent_state_stride];
            let mut expected_out = vec![0.0f32; tokens_per_slot * cfg.value_dim()];

            crate::compute::gdn::qwen35_recurrent_sequence(
                &qkv[qkv_start..qkv_end],
                &mut expected_beta,
                &mut expected_alpha,
                &dt_bias,
                &a,
                &kernel,
                &mut expected_conv_state,
                &mut expected_recurrent_state,
                &mut expected_out,
                tokens_per_slot,
                cfg,
            );

            for (actual, expected) in alpha[gate_start..gate_end]
                .iter()
                .zip(expected_alpha.iter())
            {
                assert!((actual - expected).abs() < 1e-5);
            }
            for (actual, expected) in beta[gate_start..gate_end].iter().zip(expected_beta.iter()) {
                assert!((actual - expected).abs() < 1e-5);
            }
            for (actual, expected) in out[out_start..out_end].iter().zip(expected_out.iter()) {
                assert!((actual - expected).abs() < 1e-5);
            }
            let conv_start = batch_idx * conv_state_stride;
            let conv_end = conv_start + conv_state_stride;
            for (actual, expected) in conv_state[conv_start..conv_end]
                .iter()
                .zip(expected_conv_state.iter())
            {
                assert!((actual - expected).abs() < 1e-5);
            }
            let state_start = batch_idx * recurrent_state_stride;
            let state_end = state_start + recurrent_state_stride;
            for (actual, expected) in recurrent_state[state_start..state_end]
                .iter()
                .zip(expected_recurrent_state.iter())
            {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_qwen35_recurrent_sequence_for_kv_matches_gathered_state_batch_path() {
        let backend = cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let tokens_per_slot = 2;
        let mut actual_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut expected_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1 = actual_kv.allocate_recurrent_slot();
        let expected_slot1 = expected_kv.allocate_recurrent_slot();
        assert_eq!(slot1, expected_slot1);
        let slot_indices = [0usize, slot1];
        let total_tokens = slot_indices.len() * tokens_per_slot;
        let conv_state_stride = actual_kv.conv_cache_len() * actual_kv.conv_dim();
        let recurrent_state_stride = actual_kv.recurrent_state_len();

        for (slot_pos, &slot_idx) in slot_indices.iter().enumerate() {
            actual_kv
                .conv_state_for_slot_mut(slot_idx, layer_idx)
                .iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = (slot_pos as f32 + 1.0) * 0.1 + i as f32 * 0.001);
            actual_kv
                .recurrent_state_for_slot_mut(slot_idx, layer_idx)
                .iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = (slot_pos as f32 + 1.0) * 0.2 + i as f32 * 0.002);
            expected_kv
                .conv_state_for_slot_mut(slot_idx, layer_idx)
                .copy_from_slice(actual_kv.conv_state_for_slot(slot_idx, layer_idx));
            expected_kv
                .recurrent_state_for_slot_mut(slot_idx, layer_idx)
                .copy_from_slice(actual_kv.recurrent_state_for_slot(slot_idx, layer_idx));
        }

        let qkv: Vec<f32> = (0..total_tokens * cfg.conv_dim)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.02)
            .collect();
        let alpha_input: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.05)
            .collect();
        let beta_input: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
            .map(|i| 0.1 + (i % 11) as f32 * 0.02)
            .collect();
        let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
        let a = vec![0.2, 0.25, 0.3, 0.35];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();

        let mut expected_conv_state = vec![0.0f32; slot_indices.len() * conv_state_stride];
        let mut expected_recurrent_state =
            vec![0.0f32; slot_indices.len() * recurrent_state_stride];
        expected_kv.gather_recurrent_state_batch(
            &slot_indices,
            layer_idx,
            &mut expected_conv_state,
            &mut expected_recurrent_state,
        );
        let mut expected_alpha = alpha_input.clone();
        let mut expected_beta = beta_input.clone();
        let mut expected_out = vec![0.0f32; total_tokens * cfg.value_dim()];
        {
            let mut state_batch = Qwen35RecurrentStateBatch::new(
                layer_idx,
                &slot_indices,
                &mut expected_conv_state,
                &mut expected_recurrent_state,
                conv_state_stride,
                recurrent_state_stride,
            );
            backend.qwen35_recurrent_sequence(
                &qkv,
                &mut expected_beta,
                &mut expected_alpha,
                &dt_bias,
                &a,
                &kernel,
                &mut state_batch,
                &mut expected_out,
                tokens_per_slot,
                cfg,
            );
        }
        expected_kv.scatter_recurrent_state_batch(
            &slot_indices,
            layer_idx,
            &expected_conv_state,
            &expected_recurrent_state,
        );

        let mut actual_alpha = alpha_input.clone();
        let mut actual_beta = beta_input.clone();
        let mut actual_out = vec![0.0f32; total_tokens * cfg.value_dim()];
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

        for (actual, expected) in actual_alpha.iter().zip(expected_alpha.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in actual_beta.iter().zip(expected_beta.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in actual_out.iter().zip(expected_out.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for &slot_idx in &slot_indices {
            for (actual, expected) in actual_kv
                .conv_state_for_slot(slot_idx, layer_idx)
                .iter()
                .zip(expected_kv.conv_state_for_slot(slot_idx, layer_idx).iter())
            {
                assert!((actual - expected).abs() < 1e-5);
            }
            for (actual, expected) in actual_kv
                .recurrent_state_for_slot(slot_idx, layer_idx)
                .iter()
                .zip(
                    expected_kv
                        .recurrent_state_for_slot(slot_idx, layer_idx)
                        .iter(),
                )
            {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_dequant_matmul_token_major_matches_f32_reference() {
        let backend = CpuBackend;
        let n_tokens = 2usize;
        let in_dim = 3usize;
        let out_dim = 2usize;
        let weights = [
            1.0f32, 2.0, 3.0, //
            -1.0, 0.5, 4.0,
        ];
        let input = [
            0.5f32, -1.0, 2.0, //
            3.0, 0.25, -2.0,
        ];
        let mut weight_bytes = Vec::with_capacity(weights.len() * std::mem::size_of::<f32>());
        for value in weights {
            weight_bytes.extend_from_slice(&value.to_le_bytes());
        }

        let mut actual = vec![0.0f32; n_tokens * out_dim];
        backend.dequant_matmul_token_major(
            &weight_bytes,
            GgmlType::F32,
            &input,
            &mut actual,
            n_tokens,
            out_dim,
            in_dim,
        );

        let expected = [
            1.0 * 0.5 - 2.0 * 1.0 + 3.0 * 2.0,
            -(1.0 * 0.5) - 0.5 * 1.0 + 4.0 * 2.0,
            1.0 * 3.0 + 2.0 * 0.25 - 3.0 * 2.0,
            -(1.0 * 3.0) + 0.5 * 0.25 - 4.0 * 2.0,
        ];

        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "token-major matmul mismatch at {idx}: actual={actual}, expected={expected}"
            );
        }
    }
}
