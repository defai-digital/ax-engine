//! Architecture-agnostic forward pass trait.
//!
//! Each model architecture (LLaMA, Qwen3, Gemma3) implements `ForwardPass`
//! to define its own transformer forward logic.
//!
//! # v2 changes vs v1
//! - `kv_cache: &mut KvCache` → `kv: &mut ModelKv`
//!   The forward pass receives the single-owner KV. It checks `kv.as_gpu_mut()`
//!   to decide whether the GPU batch path is viable.
//!
//! - `forward_batch` default uses `ctx.backend.use_gpu_decode()` as the gate
//!   for the GPU batch path. This means `HybridCpuDecodeBackend` (which returns
//!   `use_gpu_decode() = false`) automatically gets serial prefill without any
//!   additional env-var checks in the architecture code.
//!
//! - Removed the `AX_CPU_ONLY` env-var from the trait default. The backend
//!   choice (`BackendConfig`) is the source of truth. `AX_CPU_ONLY=1` maps to
//!   `BackendConfig::Cpu` at startup rather than being checked per-token.

use crate::backend::Backend;
use crate::compute::attention::AttentionParams;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::model::config::ModelConfig;
use crate::model::weights::WeightStore;

/// Shared context passed to forward pass implementations.
pub struct ForwardContext<'a> {
    pub config: &'a ModelConfig,
    pub attn_params: &'a AttentionParams,
    pub backend: &'a dyn Backend,
}

/// Object-safe trait for architecture-specific forward passes.
///
/// Implementors are `Send + Sync` so they can be stored in `LlamaModel`
/// which may be shared across threads.
pub trait ForwardPass: Send + Sync + std::fmt::Debug {
    /// Run a single decode step: one token in, logits out.
    ///
    /// When `ops` is `Some`, per-operation timing is recorded.
    #[allow(clippy::too_many_arguments)]
    fn forward_single(
        &self,
        ctx: &ForwardContext,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()>;

    /// Run batched prefill: process all tokens, return only last token's logits.
    ///
    /// # v2 default: gate on `use_gpu_decode()`
    ///
    /// The default implementation (used when an architecture does not override)
    /// falls back to serial `forward_single`. Architectures that override this
    /// MUST follow the v2 rule:
    ///
    /// ```rust,ignore
    /// let can_gpu_batch = !force_serial
    ///     && ctx.backend.use_gpu_decode()   // ← REQUIRED: single gate
    ///     && kv.as_gpu_mut().is_some()      // ← KV must be GPU-resident
    ///     && token_ids.len() > 1;
    /// ```
    ///
    /// This ensures that `HybridCpuDecodeBackend` (use_gpu_decode=false) always
    /// takes the serial path, writing KV to the CPU-resident `ModelKv::Cpu`.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, None)?;
        }
        Ok(())
    }

    /// Run batched prefill while recording coarse operation timing.
    ///
    /// The default implementation falls back to profiled sequential
    /// `forward_single` calls so architectures can opt in to true profiled GPU
    /// batch prefill incrementally.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_profiled(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, Some(ops))?;
        }
        Ok(())
    }

    /// Run a batched forward pass and return logits for every token position.
    ///
    /// The default implementation falls back to sequential `forward_single`
    /// calls so architectures can opt in to a true batched GPU path
    /// incrementally.
    fn forward_batch_all_logits(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        let vocab = ctx.config.vocab_size as usize;
        logits_all.resize(token_ids.len() * vocab, 0.0);

        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            let slot = &mut logits_all[i * vocab..(i + 1) * vocab];
            slot.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, slot, None)?;
        }
        Ok(())
    }

    /// Returns true when this forward implementation provides an explicit
    /// pipelined decode implementation for the current backend.
    fn supports_pipelined_decode(&self, _ctx: &ForwardContext) -> bool {
        false
    }

    /// Prepare the initial hidden-state buffer for a pipelined decode step.
    ///
    /// Implementations should write the token embedding or architecture-specific
    /// equivalent of the decode-step input into `hidden_buf`.
    fn embed_pipelined_token(
        &self,
        _ctx: &ForwardContext,
        _token_id: u32,
        _hidden_buf: &ax_engine_metal::MetalBuffer,
        _weights: &WeightStore,
    ) -> anyhow::Result<()> {
        anyhow::bail!("pipelined decode is not supported for this architecture")
    }

    /// Encode a single token decode step into a pending Metal frame.
    ///
    /// Implementations that do not support pipelined decode should return
    /// `Ok(None)`.
    fn encode_pending_decode_step(
        &self,
        _ctx: &ForwardContext,
        _hidden_buf: &ax_engine_metal::MetalBuffer,
        _position: usize,
        _kv: &mut ModelKv,
        _weights: &WeightStore,
    ) -> anyhow::Result<Option<ax_engine_metal::PendingFrame>> {
        Ok(None)
    }

    /// Whether this forward pass supports fused argmax in `encode_pending_decode_step_with_argmax`.
    /// Architectures that don't override should return false (the default).
    fn supports_fused_argmax(&self) -> bool {
        false
    }

    /// Like [`encode_pending_decode_step`] but also appends a GPU argmax
    /// dispatch at the end of the command buffer, so the caller can read
    /// the greedy token index without a separate CB round-trip.
    fn encode_pending_decode_step_with_argmax(
        &self,
        ctx: &ForwardContext,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
    ) -> anyhow::Result<Option<ax_engine_metal::PendingFrame>> {
        // Default: fall back to the non-argmax path (callers must use
        // separate gpu_argmax_logits if the architecture doesn't override).
        self.encode_pending_decode_step(ctx, hidden_buf, position, kv, weights)
    }

    /// Apply architecture-specific postprocessing to logits produced by a
    /// pipelined decode step after GPU readback.
    fn postprocess_pipelined_logits(
        &self,
        _ctx: &ForwardContext,
        _logits: &mut [f32],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Validate that the model config is compatible with this architecture.
    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()>;

    /// Return the architecture name (e.g. "llama", "qwen3", "gemma3").
    fn arch_name(&self) -> &str;
}
