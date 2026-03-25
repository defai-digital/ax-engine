//! LLaMA-family transformer forward pass.
//!
//! Implements the full inference pipeline:
//!   1. Token embedding lookup
//!   2. For each layer:
//!      a. RMSNorm (attention norm)
//!      b. Q/K/V projections (matmul)
//!      c. RoPE on Q and K
//!      d. KV cache update
//!      e. Multi-head attention
//!      f. Output projection (matmul)
//!      g. Residual add
//!      h. RMSNorm (FFN norm)
//!      i. Gate/Up projections (matmul)
//!      j. SiLU activation + element-wise multiply
//!      k. Down projection (matmul)
//!      l. Residual add
//!   3. Final RMSNorm
//!   4. LM head projection -> logits
//!
//! # v2 API changes
//! - `kv_cache: &mut KvCache` → `kv: &mut ModelKv`
//! - GPU decode path uses `kv.as_gpu_mut()` directly, no Mutex locking
//! - `forward_batch` GPU gate uses `ctx.backend.use_gpu_decode()` instead of AX_CPU_ONLY
//! - `gpu_kv.advance()` → `gpu_kv.finalize_token()`
//! - `gpu_kv.advance_by(n)` → `gpu_kv.finalize_batch(n)` (no CPU mirror sync)
//! - Paged KV deferred to v2.1 (stubs kept but disabled)

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::backend::metal::MetalOps;
use crate::compute::attention::{self, AttentionParams};
use crate::compute::rms_norm;
use crate::compute::rope;
use crate::compute::silu;
use crate::gguf::tensor::GgmlType;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::decode::DecodeIntent;
use crate::model::execution_plan::{
    DecodeBarrierPlan, DecodeExecutionPlan, DecodeScratchPlan, GpuBatchPrefillExecutionPlan,
    GpuDecodeExecutionPlan, LlamaLayerQkvPlan, PrefillAttentionPlan, PrefillExecutionPlan,
    PrefillFfnActivationPlan, PrefillLogitsPlan, PrefillMode, PrefillProjectionInputPlan,
    PrefillResidualHandoffPlan, PrefillWoInputPlan, llama_layer_plan_for_gpu,
};
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    encode_batch_logits, encode_dequant_batch, encode_dequant_batch_f16in,
    encode_dequant_batch_pair_f16in, encode_dequant_matvec, encode_dequant_matvec_with_config,
    gpu_decode_quant_supported, gpu_prefill_experimental_q5k_small_n_auto_eligible,
    gpu_prefill_uses_experimental_q5k,
};
use crate::model::weights::WeightStore;
use std::sync::OnceLock;

/// Timing helper: records elapsed time into `ops.$field` if ops is Some.
macro_rules! timed {
    ($ops:expr, $field:ident, $body:expr) => {{
        if let Some(ref mut ops) = $ops {
            let _t = OpTimer::start();
            let _r = $body;
            ops.$field += _t.elapsed();
            _r
        } else {
            $body
        }
    }};
}

/// Whether explicit Metal barriers are enabled for single-token decode path.
///
/// Controlled by `AX_METAL_DECODE_BARRIERS`:
/// - `1` / `true` / `on`            -> enabled
/// - unset / `0` / `false` / `off`  -> disabled (default)
///
/// Default OFF: llama.cpp uses zero barriers in decode. The sequential
/// Metal encoder on Apple Silicon UMA provides ordering guarantees —
/// each dispatch completes before the next starts, and writes are
/// immediately visible via cache coherency. Barriers are redundant.
pub(crate) fn metal_decode_barriers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_DECODE_BARRIERS") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false, // llama.cpp uses 0 barriers; safe on Apple Silicon UMA
    })
}

/// Whether mistral-style HD128 prefill should write attention output in f16.
///
/// Controlled by `AX_METAL_PREFILL_MISTRAL_F16OUT`:
/// - `1` / `true` / `on` -> enabled
/// - unset / other       -> disabled (default)
fn metal_prefill_attn_f16out_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_MISTRAL_F16OUT") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn metal_prefill_use_cached0_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_USE_CACHED0") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn metal_prefill_split_rope_append_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("AX_METAL_PREFILL_SPLIT_ROPE_APPEND") {
            Ok(v) => {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "on"
            }
            Err(_) => true,
        },
    )
}

/// Encode all transformer layers + final norm + LM head into an already-opened
/// Metal compute encoder.
///
/// Extracted from `forward_single_gpu_unified` so both the synchronous path
/// (execute_sync) and the pipelined path (encode_frame) can share the same
/// GPU command sequence.
///
/// `hidden_buf` replaces `s.hidden` as the running accumulator, allowing the
/// pipelined path to double-buffer the hidden state across tokens.
#[allow(clippy::too_many_arguments)]
fn encode_llama_gpu_layers_only(
    encoder: &ax_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_metal::MetalBuffer,
    cfg: &ModelConfig,
    kv_offset: u32,
    rope_position: f32,
    full_seq_len: usize,
    exec_plan: &GpuDecodeExecutionPlan,
    gpu_kv: &crate::kv::GpuKv,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_metal::MetalBuffer>,
    fused_qkv_cache: &rustc_hash::FxHashMap<(usize, usize, usize), ax_metal::MetalBuffer>,
    mut ops: Option<&mut OpBreakdown>,
) -> anyhow::Result<()> {
    let dim = cfg.embedding_dim as usize;
    let n_layers = cfg.n_layers as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let inter_dim = cfg.intermediate_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let eps = cfg.rms_norm_eps;
    let layer_plans: Vec<_> = cached
        .layers
        .iter()
        .map(|lw| llama_layer_plan_for_gpu(exec_plan, lw.wq_dtype, lw.wk_dtype, lw.wv_dtype))
        .collect();
    let decode_barrier = |encoder: &ax_metal::MetalEncoder| {
        if exec_plan.barriers == DecodeBarrierPlan::Explicit {
            ax_metal::barrier_buffers(encoder);
        }
    };

    for (layer, &layer_plan) in layer_plans.iter().enumerate().take(n_layers) {
        let lw = &cached.layers[layer];

        let norm_w_buf = weight_cache.get(&lw.attn_norm).unwrap();
        let wq_buf = weight_cache.get(&lw.wq).unwrap();
        let wk_buf = weight_cache.get(&lw.wk).unwrap();
        let wv_buf = weight_cache.get(&lw.wv).unwrap();
        let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
        let fused_qkv_buf = if layer_plan.qkv == LlamaLayerQkvPlan::Fused {
            fused_qkv_cache.get(&fused_qkv_key)
        } else {
            None
        };

        // 1. RMSNorm: hidden → norm_buf
        let t = OpTimer::start();
        metal_ops.elementwise.encode_rms_norm_out(
            encoder,
            hidden_buf,
            norm_w_buf,
            &s.norm_buf,
            dim as u32,
            eps,
        );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_norm += t.elapsed();
        }
        decode_barrier(encoder);

        // 2. QKV matmul
        if let Some(fused_w) = fused_qkv_buf {
            let t = OpTimer::start();
            encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                fused_w,
                &s.norm_buf,
                &s.qkv_buf,
                (q_dim + 2 * kv_dim) as u32,
                dim as u32,
                lw.wq_dtype,
                exec_plan.dequant_dispatch,
            );
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_encode_layer_qkv += t.elapsed();
            }
            decode_barrier(encoder);

            let t = OpTimer::start();
            metal_ops.elementwise.encode_qkv_split_rope_append_kv_batch(
                encoder,
                &s.qkv_buf,
                &s.q_buf,
                &s.k_buf,
                &s.v_buf,
                gpu_kv.k_buffer(layer),
                gpu_kv.v_buffer(layer),
                exec_plan.kv_f16,
                1,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                rope_position,
                1.0,
                cfg.rope_freq_base,
                kv_offset,
                kv_dim as u32,
            );
            let split_d = t.elapsed();
            if let Some(ref mut ops_ref) = ops {
                // Split attribution evenly between rope and KV append for this fused op.
                ops_ref.gpu_encode_layer_rope += split_d / 2;
                ops_ref.gpu_encode_layer_kv_append += split_d / 2;
            }
            decode_barrier(encoder);
        } else {
            let t = OpTimer::start();
            encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                wq_buf,
                &s.norm_buf,
                &s.q_buf,
                q_dim as u32,
                dim as u32,
                lw.wq_dtype,
                exec_plan.dequant_dispatch,
            );
            encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                wk_buf,
                &s.norm_buf,
                &s.k_buf,
                kv_dim as u32,
                dim as u32,
                lw.wk_dtype,
                exec_plan.dequant_dispatch,
            );
            encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                wv_buf,
                &s.norm_buf,
                &s.v_buf,
                kv_dim as u32,
                dim as u32,
                lw.wv_dtype,
                exec_plan.dequant_dispatch,
            );
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_encode_layer_qkv += t.elapsed();
            }
            decode_barrier(encoder);

            // 3. RoPE
            let t = OpTimer::start();
            metal_ops.elementwise.encode_rope(
                encoder,
                &s.q_buf,
                &s.k_buf,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                rope_position,
                cfg.rope_freq_base,
            );
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_encode_layer_rope += t.elapsed();
            }
            decode_barrier(encoder);

            // 4. KV append
            let t = OpTimer::start();
            metal_ops.elementwise.encode_kv_append(
                encoder,
                &s.k_buf,
                gpu_kv.k_buffer(layer),
                exec_plan.kv_f16,
                kv_offset,
                kv_dim as u32,
            );
            metal_ops.elementwise.encode_kv_append(
                encoder,
                &s.v_buf,
                gpu_kv.v_buffer(layer),
                exec_plan.kv_f16,
                kv_offset,
                kv_dim as u32,
            );
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_encode_layer_kv_append += t.elapsed();
            }
            decode_barrier(encoder);
        }

        // 5. Decode attention
        let t = OpTimer::start();
        metal_ops
            .attention
            .encode_attention_decode_with_scratch_and_config(
                encoder,
                &s.q_buf,
                gpu_kv.k_buffer(layer),
                gpu_kv.v_buffer(layer),
                &s.attn_out,
                &s.splitk_partial_out,
                &s.splitk_partial_lse,
                exec_plan.kv_f16,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                0,
                full_seq_len as u32,
                exec_plan.attention_dispatch,
            );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_attention += t.elapsed();
        }
        decode_barrier(encoder);

        // 6. Output projection
        let wo_buf = weight_cache.get(&lw.wo).unwrap();
        let t = OpTimer::start();
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wo_buf,
            &s.attn_out,
            &s.proj_buf,
            dim as u32,
            q_dim as u32,
            lw.wo_dtype,
            exec_plan.dequant_dispatch,
        );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_out_proj += t.elapsed();
        }
        decode_barrier(encoder);

        // 7. Residual: hidden += proj
        let t = OpTimer::start();
        metal_ops
            .elementwise
            .encode_elementwise_add(encoder, hidden_buf, &s.proj_buf, dim as u32);
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_residual += t.elapsed();
        }
        decode_barrier(encoder);

        // 8. FFN norm: hidden → norm_buf
        let ffn_nw_buf = weight_cache.get(&lw.ffn_norm).unwrap();
        let t = OpTimer::start();
        metal_ops.elementwise.encode_rms_norm_out(
            encoder,
            hidden_buf,
            ffn_nw_buf,
            &s.norm_buf,
            dim as u32,
            eps,
        );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_norm += t.elapsed();
        }
        decode_barrier(encoder);

        // 9. Gate + Up
        let wg_buf = weight_cache.get(&lw.wg).unwrap();
        let wu_buf = weight_cache.get(&lw.wu).unwrap();
        let t = OpTimer::start();
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wg_buf,
            &s.norm_buf,
            &s.gate_buf,
            inter_dim as u32,
            dim as u32,
            lw.wg_dtype,
            exec_plan.dequant_dispatch,
        );
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wu_buf,
            &s.norm_buf,
            &s.up_buf,
            inter_dim as u32,
            dim as u32,
            lw.wu_dtype,
            exec_plan.dequant_dispatch,
        );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_ffn += t.elapsed();
        }
        decode_barrier(encoder);

        // 10. SiLU(gate) * up
        let t = OpTimer::start();
        metal_ops.elementwise.encode_silu_elementwise_mul(
            encoder,
            &s.gate_buf,
            &s.up_buf,
            inter_dim as u32,
        );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_ffn += t.elapsed();
        }
        decode_barrier(encoder);

        // 11. Down projection
        let wd_buf = weight_cache.get(&lw.wd).unwrap();
        let t = OpTimer::start();
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wd_buf,
            &s.gate_buf,
            &s.down_buf,
            dim as u32,
            inter_dim as u32,
            lw.wd_dtype,
            exec_plan.dequant_dispatch,
        );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_ffn += t.elapsed();
        }
        decode_barrier(encoder);

        // 12. Residual: hidden += down
        let t = OpTimer::start();
        metal_ops
            .elementwise
            .encode_elementwise_add(encoder, hidden_buf, &s.down_buf, dim as u32);
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_residual += t.elapsed();
        }
        if layer + 1 < n_layers {
            decode_barrier(encoder);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_llama_gpu_output_head(
    encoder: &ax_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_metal::MetalBuffer,
    cfg: &ModelConfig,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_metal::MetalBuffer>,
    exec_plan: &GpuDecodeExecutionPlan,
) {
    let dim = cfg.embedding_dim as usize;
    let vocab_size = cfg.vocab_size as usize;
    let eps = cfg.rms_norm_eps;

    let decode_barrier = |encoder: &ax_metal::MetalEncoder| {
        if exec_plan.barriers == DecodeBarrierPlan::Explicit {
            ax_metal::barrier_buffers(encoder);
        }
    };

    // Final RMSNorm (in-place on hidden)
    let fnw_buf = weight_cache.get(&cached.output_norm).unwrap();
    decode_barrier(encoder);
    metal_ops
        .elementwise
        .encode_rms_norm(encoder, hidden_buf, fnw_buf, dim as u32, eps);

    // LM head → logits
    decode_barrier(encoder);
    let lm_buf = weight_cache.get(&cached.lm_head).unwrap();
    encode_dequant_matvec_with_config(
        metal_ops,
        encoder,
        lm_buf,
        hidden_buf,
        &s.logits_buf,
        vocab_size as u32,
        dim as u32,
        cached.lm_head_dtype,
        exec_plan.dequant_dispatch,
    );
}

/// Encode a LLaMA single-token decode step into a [`ax_metal::PendingFrame`]
/// without committing to the GPU.
///
/// Unlike `forward_single_gpu_unified` this function:
/// - Uses an external `hidden_buf` (for double-buffering in pipelined loops).
/// - Uses the explicit `position` for kv_offset/full_seq_len rather than
///   `gpu_kv.seq_len()`, which may lag by 1 in a pipelined context.
/// - Does NOT call `gpu_kv.finalize_token()` — the caller must call
///   [`LlamaModel::advance_gpu_kv_token`] after `wait_frame` completes.
///
/// **Precondition**: the caller must have pre-allocated KV capacity via
/// [`LlamaModel::prewarm_kv_capacity`] before entering the decode loop, so
/// that `ensure_capacity` is a guaranteed no-op here and safe while a prior
/// command buffer may still be executing.
fn encode_llama_pending_step(
    metal_ops: &MetalOps,
    cfg: &ModelConfig,
    hidden_buf: &ax_metal::MetalBuffer,
    position: usize,
    gpu_kv: &mut crate::kv::GpuKv,
    weights: &WeightStore,
) -> anyhow::Result<ax_metal::PendingFrame> {
    let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as usize;
    let exec_plan = DecodeExecutionPlan::llama_pipelined(
        metal_ops,
        gpu_kv,
        cfg.embedding_dim,
        cfg.head_dim,
        position + 1,
    );
    debug_assert_eq!(
        exec_plan.dequant_dispatch,
        metal_ops.dequant_dispatch_config(),
        "llama decode execution plan must match current Metal dequant dispatch config"
    );
    debug_assert_eq!(
        exec_plan.attention_dispatch,
        metal_ops.attention_dispatch_config(),
        "llama decode execution plan must match current Metal attention dispatch config"
    );

    // Pre-condition: capacity must already be reserved by caller.
    // This call is a guaranteed no-op if prewarm_kv_capacity was called.
    gpu_kv.ensure_capacity(&metal_ops.device, position + 1)?;

    // Init scratch buffers (no-op after first call)
    match DecodeScratchPlan::SharedGpuScratch {
        DecodeScratchPlan::SharedGpuScratch => metal_ops.init_scratches(cfg),
        DecodeScratchPlan::CpuScratch => anyhow::bail!("pipelined GPU decode requires GPU scratch"),
    }

    let scratch_guard = metal_ops.scratches();
    let s = scratch_guard.as_ref().unwrap();

    // Build weight key cache on first call
    if !metal_ops.has_cached_model_keys() {
        LlamaForward::build_cached_model_keys_llama(metal_ops, weights, cfg)?;
    }

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();

    // Use explicit position (NOT gpu_kv.seq_len()) for correct offset in pipeline
    let kv_offset = (position * kv_dim) as u32;
    let full_seq_len = position + 1;
    let rope_position = match cfg.rope_scaling {
        crate::model::config::RopeScaling::Linear(factor) => position as f32 / factor,
        crate::model::config::RopeScaling::None => position as f32,
    };

    let weight_cache = metal_ops.lock_weight_cache();
    let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

    metal_ops.device.encode_frame(|encoder| {
        encode_llama_gpu_layers_only(
            encoder,
            metal_ops,
            s,
            hidden_buf,
            cfg,
            kv_offset,
            rope_position,
            full_seq_len,
            &exec_plan,
            gpu_kv,
            cached,
            &weight_cache,
            &fused_qkv_cache,
            None,
        )?;
        encode_llama_gpu_output_head(
            encoder,
            metal_ops,
            s,
            hidden_buf,
            cfg,
            cached,
            &weight_cache,
            &exec_plan,
        );
        Ok(())
    })
}

/// LLaMA-family forward pass implementation.
///
/// Used for llama, mistral, and any architecture that follows
/// the standard LLaMA transformer pattern (SwiGLU FFN, no QKV bias).
#[derive(Debug)]
pub struct LlamaForward;

impl LlamaForward {
    /// Build and store pre-computed weight cache keys for all layers (LLaMA architecture).
    /// Called once on first forward pass; subsequent calls are skipped via `has_cached_model_keys()`.
    fn build_cached_model_keys_llama(
        metal_ops: &MetalOps,
        weights: &WeightStore,
        cfg: &ModelConfig,
    ) -> anyhow::Result<()> {
        use crate::backend::metal::{CachedLayerKeys, CachedModelKeys};

        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let dim = cfg.embedding_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let use_precomputed_f16 = metal_ops.metal_precompute_f16_enabled();

        let mut layers = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            let attn_norm_key = metal_ops.ensure_f32_cached(attn_norm_w);
            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
            let wq_key = metal_ops.ensure_quant_cached(wq_raw);
            let wk_key = metal_ops.ensure_quant_cached(wk_raw);
            let wv_key = metal_ops.ensure_quant_cached(wv_raw);
            if use_precomputed_f16
                && metal_ops.metal_fused_qkv_enabled()
                && wq_dtype == wk_dtype
                && wq_dtype == wv_dtype
                && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0)
            {
                metal_ops.ensure_qkv_fused_quant_cached(wq_raw, wk_raw, wv_raw);
                if wq_dtype == GgmlType::Q4K {
                    metal_ops.ensure_precomputed_q4k_f16_fused_qkv(
                        wq_raw,
                        wk_raw,
                        wv_raw,
                        (q_dim + 2 * kv_dim) as u32,
                        dim as u32,
                    )?;
                }
                if wq_dtype == GgmlType::Q8_0 {
                    metal_ops.ensure_precomputed_q8_0_f16_fused_qkv(
                        wq_raw,
                        wk_raw,
                        wv_raw,
                        (q_dim + 2 * kv_dim) as u32,
                        dim as u32,
                    )?;
                }
            }
            if use_precomputed_f16 && wq_dtype == GgmlType::Q4K {
                metal_ops.ensure_precomputed_q4k_f16_from_raw(wq_raw, q_dim as u32, dim as u32)?;
                metal_ops.ensure_precomputed_q4k_f16_from_raw(wk_raw, kv_dim as u32, dim as u32)?;
                metal_ops.ensure_precomputed_q4k_f16_from_raw(wv_raw, kv_dim as u32, dim as u32)?;
            }
            if use_precomputed_f16 && wq_dtype == GgmlType::Q6K {
                metal_ops.ensure_precomputed_q6k_f16_from_raw(wq_raw, q_dim as u32, dim as u32)?;
                metal_ops.ensure_precomputed_q6k_f16_from_raw(wk_raw, kv_dim as u32, dim as u32)?;
                metal_ops.ensure_precomputed_q6k_f16_from_raw(wv_raw, kv_dim as u32, dim as u32)?;
            }
            if use_precomputed_f16 && wq_dtype == GgmlType::Q8_0 {
                metal_ops.ensure_precomputed_q8_0_f16_from_raw(wq_raw, q_dim as u32, dim as u32)?;
                metal_ops.ensure_precomputed_q8_0_f16_from_raw(
                    wk_raw,
                    kv_dim as u32,
                    dim as u32,
                )?;
                metal_ops.ensure_precomputed_q8_0_f16_from_raw(
                    wv_raw,
                    kv_dim as u32,
                    dim as u32,
                )?;
            }
            let (wo_raw, wo_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            let (wg_raw, wg_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            let (wd_raw, wd_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
            let wo_key = metal_ops.ensure_quant_cached(wo_raw);
            let wg_key = metal_ops.ensure_quant_cached(wg_raw);
            let wu_key = metal_ops.ensure_quant_cached(wu_raw);
            let wd_key = metal_ops.ensure_quant_cached(wd_raw);
            if use_precomputed_f16 && wo_dtype == GgmlType::Q4K {
                metal_ops.ensure_precomputed_q4k_f16_from_raw(wo_raw, dim as u32, dim as u32)?;
            }
            if use_precomputed_f16 && wo_dtype == GgmlType::Q6K {
                metal_ops.ensure_precomputed_q6k_f16_from_raw(wo_raw, dim as u32, dim as u32)?;
            }
            if use_precomputed_f16 && wo_dtype == GgmlType::Q8_0 {
                metal_ops.ensure_precomputed_q8_0_f16_from_raw(wo_raw, dim as u32, dim as u32)?;
            }
            if use_precomputed_f16 && wg_dtype == GgmlType::Q4K {
                metal_ops.ensure_precomputed_q4k_f16_from_raw(
                    wg_raw,
                    inter_dim as u32,
                    dim as u32,
                )?;
            }
            if use_precomputed_f16 && wg_dtype == GgmlType::Q6K {
                metal_ops.ensure_precomputed_q6k_f16_from_raw(
                    wg_raw,
                    inter_dim as u32,
                    dim as u32,
                )?;
            }
            if use_precomputed_f16 && wg_dtype == GgmlType::Q8_0 {
                metal_ops.ensure_precomputed_q8_0_f16_from_raw(
                    wg_raw,
                    inter_dim as u32,
                    dim as u32,
                )?;
            }
            if use_precomputed_f16 && wu_dtype == GgmlType::Q4K {
                metal_ops.ensure_precomputed_q4k_f16_from_raw(
                    wu_raw,
                    inter_dim as u32,
                    dim as u32,
                )?;
            }
            if use_precomputed_f16 && wu_dtype == GgmlType::Q6K {
                metal_ops.ensure_precomputed_q6k_f16_from_raw(
                    wu_raw,
                    inter_dim as u32,
                    dim as u32,
                )?;
            }
            if use_precomputed_f16 && wu_dtype == GgmlType::Q8_0 {
                metal_ops.ensure_precomputed_q8_0_f16_from_raw(
                    wu_raw,
                    inter_dim as u32,
                    dim as u32,
                )?;
            }
            if use_precomputed_f16 && wd_dtype == GgmlType::Q4K {
                metal_ops.ensure_precomputed_q4k_f16_from_raw(
                    wd_raw,
                    dim as u32,
                    inter_dim as u32,
                )?;
            }
            if use_precomputed_f16 && wd_dtype == GgmlType::Q6K {
                metal_ops.ensure_precomputed_q6k_f16_from_raw(
                    wd_raw,
                    dim as u32,
                    inter_dim as u32,
                )?;
            }
            if use_precomputed_f16 && wd_dtype == GgmlType::Q8_0 {
                metal_ops.ensure_precomputed_q8_0_f16_from_raw(
                    wd_raw,
                    dim as u32,
                    inter_dim as u32,
                )?;
            }
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);
            layers.push(CachedLayerKeys {
                attn_norm: attn_norm_key,
                wq: wq_key,
                wq_dtype,
                wk: wk_key,
                wk_dtype,
                wv: wv_key,
                wv_dtype,
                wo: wo_key,
                wo_dtype,
                ffn_norm: ffn_norm_key,
                wg: wg_key,
                wg_dtype,
                wu: wu_key,
                wu_dtype,
                wd: wd_key,
                wd_dtype,
                attn_q_norm: None,
                attn_k_norm: None,
                post_attn_norm: None,
                post_ffn_norm: None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                wo_bias: None,
                gate_bias: None,
                up_bias: None,
                down_bias: None,
            });
        }
        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        let output_norm_key = metal_ops.ensure_f32_cached(final_norm_w);
        let lm_weight_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        let lm_head_key = metal_ops.ensure_quant_cached(lm_raw);
        if use_precomputed_f16 && lm_dtype == GgmlType::Q4K {
            metal_ops.ensure_precomputed_q4k_f16_from_raw(
                lm_raw,
                cfg.vocab_size,
                cfg.embedding_dim,
            )?;
        }
        if use_precomputed_f16 && lm_dtype == GgmlType::Q6K {
            metal_ops.ensure_precomputed_q6k_f16_from_raw(
                lm_raw,
                cfg.vocab_size,
                cfg.embedding_dim,
            )?;
        }
        if use_precomputed_f16 && lm_dtype == GgmlType::Q8_0 {
            metal_ops.ensure_precomputed_q8_0_f16_from_raw(
                lm_raw,
                cfg.vocab_size,
                cfg.embedding_dim,
            )?;
        }
        metal_ops.set_cached_model_keys(CachedModelKeys {
            layers,
            output_norm: output_norm_key,
            lm_head: lm_head_key,
            lm_head_dtype: lm_dtype,
        });
        Ok(())
    }

    /// Unified GPU forward pass: all layers in a single command buffer.
    ///
    /// v2: receives `gpu_kv: &mut GpuKv` directly (no Mutex, no advance_by).
    #[allow(clippy::too_many_arguments)]
    fn forward_single_gpu_unified(
        &self,
        ctx: &ForwardContext,
        metal_ops: &MetalOps,
        token_id: u32,
        position: usize,
        gpu_kv: &mut crate::kv::GpuKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as usize;
        let exec_plan = DecodeExecutionPlan::llama_single_cb(
            metal_ops,
            gpu_kv,
            cfg.embedding_dim,
            cfg.head_dim,
            gpu_kv.seq_len() + 1,
        );
        debug_assert_eq!(
            exec_plan.dequant_dispatch,
            metal_ops.dequant_dispatch_config(),
            "llama decode execution plan must match current Metal dequant dispatch config"
        );
        debug_assert_eq!(
            exec_plan.attention_dispatch,
            metal_ops.attention_dispatch_config(),
            "llama decode execution plan must match current Metal attention dispatch config"
        );

        assert!(logits.len() >= vocab_size);

        match DecodeScratchPlan::SharedGpuScratch {
            DecodeScratchPlan::SharedGpuScratch => metal_ops.init_scratches(cfg),
            DecodeScratchPlan::CpuScratch => {
                anyhow::bail!("single-CB GPU decode requires GPU scratch")
            }
        }

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();

        let next_seq = gpu_kv.seq_len() + 1;
        gpu_kv.ensure_capacity(&metal_ops.device, next_seq)?;

        // Token embedding + setup on host side before GPU execute.
        let setup_t = OpTimer::start();
        {
            let hidden_cpu = unsafe {
                std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim)
            };
            weights.dequantize_row("token_embd.weight", token_id as usize, hidden_cpu)?;
        }

        let rope_position = match cfg.rope_scaling {
            crate::model::config::RopeScaling::Linear(factor) => position as f32 / factor,
            crate::model::config::RopeScaling::None => position as f32,
        };

        // Pre-cache ALL weights and build cached keys (first call only)
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_llama(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();

        let cur_seq_len = gpu_kv.seq_len();
        let kv_offset = (cur_seq_len * kv_dim) as u32;
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        // GPU execution. Profiling must not alter the command-buffer shape, so
        // even when `ops` is present we keep the same single execute_sync path.
        {
            let weight_cache = metal_ops.lock_weight_cache();
            let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();
            if let Some(ref mut ops_ref) = ops {
                let exec_t = OpTimer::start();
                metal_ops.device.execute_sync(|encoder| {
                    encode_llama_gpu_layers_only(
                        encoder,
                        metal_ops,
                        s,
                        &s.hidden,
                        cfg,
                        kv_offset,
                        rope_position,
                        cur_seq_len + 1,
                        &exec_plan,
                        gpu_kv,
                        cached,
                        &weight_cache,
                        &fused_qkv_cache,
                        Some(ops_ref),
                    )?;
                    encode_llama_gpu_output_head(
                        encoder,
                        metal_ops,
                        s,
                        &s.hidden,
                        cfg,
                        cached,
                        &weight_cache,
                        &exec_plan,
                    );
                    Ok(())
                })?;
                ops_ref.gpu_execute += exec_t.elapsed();
            } else {
                let exec_t = OpTimer::start();
                metal_ops.device.execute_sync(|encoder| {
                    encode_llama_gpu_layers_only(
                        encoder,
                        metal_ops,
                        s,
                        &s.hidden,
                        cfg,
                        kv_offset,
                        rope_position,
                        cur_seq_len + 1,
                        &exec_plan,
                        gpu_kv,
                        cached,
                        &weight_cache,
                        &fused_qkv_cache,
                        None,
                    )?;
                    encode_llama_gpu_output_head(
                        encoder,
                        metal_ops,
                        s,
                        &s.hidden,
                        cfg,
                        cached,
                        &weight_cache,
                        &exec_plan,
                    );
                    Ok(())
                })?;
                let _ = exec_t;
            }
        }

        // v2: advance GPU KV only — no CPU mirror to sync
        gpu_kv.finalize_token();

        // Copy logits from GPU buffer to CPU
        let rb_t = OpTimer::start();
        let logits_gpu = unsafe {
            std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab_size)
        };
        logits[..vocab_size].copy_from_slice(logits_gpu);
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }

        Ok(())
    }

    /// Batched GPU prefill: process N tokens through all layers in minimal command buffers.
    ///
    /// Uses batch scratch buffers [N × dim] for hidden, Q, K, V, attn_out.
    /// Attention runs either batch-local prefill (empty prefix) or cache-backed prefill
    /// (when KV prefix already exists).
    ///
    /// v2: receives `gpu_kv: &mut GpuKv` directly. Ends with `gpu_kv.finalize_batch(n_tokens)`.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_gpu_unified(
        &self,
        ctx: &ForwardContext,
        metal_ops: &MetalOps,
        token_ids: &[u32],
        gpu_kv: &mut crate::kv::GpuKv,
        weights: &WeightStore,
        last_logits: Option<&mut [f32]>,
        logits_all: Option<&mut Vec<f32>>,
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let n_tokens = token_ids.len();
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let eps = cfg.rms_norm_eps;
        let emit_all_logits = logits_all.is_some();

        if let Some(logits) = last_logits.as_ref() {
            assert!(logits.len() >= vocab_size);
        }

        // Initialize buffers
        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, n_tokens);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let batch_guard = metal_ops.batch_scratches();
        let bs = batch_guard.as_ref().unwrap();

        // Ensure capacity for all N tokens
        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + n_tokens)?;

        // Embed all N tokens into batch hidden buffer via CPU UMA write
        {
            let batch_hidden_cpu = unsafe {
                std::slice::from_raw_parts_mut(
                    bs.hidden.contents().as_ptr() as *mut f32,
                    n_tokens * dim,
                )
            };
            for (i, &tid) in token_ids.iter().enumerate() {
                weights.dequantize_row(
                    "token_embd.weight",
                    tid as usize,
                    &mut batch_hidden_cpu[i * dim..(i + 1) * dim],
                )?;
            }
        }

        // Pre-cache weights and build cached keys (first call only)
        // The forward_single_gpu_unified path may have already built them.
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_llama(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();

        let base_seq_len = gpu_kv.seq_len();
        let all_logits_buf = if emit_all_logits {
            Some(ax_metal::MetalBuffer::new(
                metal_ops.device.device(),
                n_tokens * vocab_size * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        };

        // Single command buffer: all layers + final norm + LM head
        {
            let weight_cache = metal_ops.lock_weight_cache();
            let has_q8_weights = cached.layers.iter().any(|lw| {
                lw.wq_dtype == GgmlType::Q8_0
                    || lw.wk_dtype == GgmlType::Q8_0
                    || lw.wv_dtype == GgmlType::Q8_0
                    || lw.wo_dtype == GgmlType::Q8_0
                    || lw.wg_dtype == GgmlType::Q8_0
                    || lw.wu_dtype == GgmlType::Q8_0
                    || lw.wd_dtype == GgmlType::Q8_0
            }) || matches!(cached.lm_head_dtype, GgmlType::Q8_0);
            let has_q5k_weights = gpu_prefill_uses_experimental_q5k(weights);
            let q5k_small_n_auto_eligible =
                gpu_prefill_experimental_q5k_small_n_auto_eligible(weights);
            let prefill_plan: GpuBatchPrefillExecutionPlan = DecodeExecutionPlan::llama_prefill(
                metal_ops,
                gpu_kv,
                base_seq_len,
                n_tokens as u32,
                cfg.head_dim,
                has_q8_weights,
                has_q5k_weights,
                q5k_small_n_auto_eligible,
                metal_prefill_attn_f16out_enabled(),
                metal_prefill_use_cached0_enabled(),
                metal_prefill_split_rope_append_enabled(),
            );
            let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

            // ── Graph IR path: pre-computed dispatch schedule ──
            if super::prefill_schedule::prefill_graph_ir_enabled() {
                let schedule = super::prefill_schedule::build_llama_prefill_schedule(
                    cfg,
                    &prefill_plan,
                    cached,
                    &weight_cache,
                    bs,
                    s,
                    gpu_kv,
                    base_seq_len,
                    n_tokens,
                    all_logits_buf.as_ref(),
                    &fused_qkv_cache,
                );
                return super::prefill_schedule::execute_prefill_multi_cb(
                    &metal_ops.device,
                    &schedule,
                    metal_ops,
                );
            }

            // ── Inline path (default): existing SmartBarrier encoding ──
            metal_ops.device.execute_sync_concurrent(|encoder| {
                let nt = n_tokens as u32;
                let mut sb = ax_metal::SmartBarrier::new(encoder);

                // First layer's Phase 1a: RMSNorm (standalone, before loop).
                // Subsequent layers' Phase 1a is fused with the previous layer's
                // Phase 3f residual add (saves 1 dispatch + 1 barrier per layer).
                {
                    let norm_w_buf = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
                    if prefill_plan.use_f16_batch_io {
                        metal_ops.elementwise.encode_rms_norm_out_batch_f16(
                            encoder,
                            &bs.hidden,
                            norm_w_buf,
                            &bs.matmul_in_f16,
                            dim as u32,
                            nt,
                            eps,
                        );
                        sb.post_dispatch(&[&bs.hidden], &[&bs.matmul_in_f16]);
                    } else {
                        metal_ops.elementwise.encode_rms_norm_out_batch(
                            encoder,
                            &bs.hidden,
                            norm_w_buf,
                            &bs.norm_buf,
                            dim as u32,
                            nt,
                            eps,
                        );
                        sb.post_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
                    }
                }

                for layer in 0..n_layers {
                    let lw = &cached.layers[layer];
                    let (rope_start, rope_step) = match cfg.rope_scaling {
                        crate::model::config::RopeScaling::Linear(f) => {
                            (base_seq_len as f32 / f, 1.0f32 / f)
                        }
                        crate::model::config::RopeScaling::None => (base_seq_len as f32, 1.0f32),
                    };

                    let wq_buf = weight_cache.get(&lw.wq).unwrap();
                    let wk_buf = weight_cache.get(&lw.wk).unwrap();
                    let wv_buf = weight_cache.get(&lw.wv).unwrap();
                    let fused_qkv_m = q_dim + 2 * kv_dim;
                    let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
                    let qkv_layer_plan = DecodeExecutionPlan::llama_prefill_qkv_layer(
                        &prefill_plan,
                        lw.wq_dtype,
                        lw.wk_dtype,
                        lw.wv_dtype,
                    );
                    let fused_qkv_buf = if qkv_layer_plan.use_fused_projection {
                        fused_qkv_cache.get(&fused_qkv_key)
                    } else {
                        None
                    };

                    // Phase 1a (RMSNorm) is done before loop for layer 0, and
                    // fused with previous layer's Phase 3f for layers 1+.
                    // norm_buf / matmul_in_f16 is already populated.

                    // ── Phase 1b: Batched QKV matmul ──
                    if let Some(fused_w) = fused_qkv_buf {
                        let cache_offset = (base_seq_len * kv_dim) as u32;
                        let qkv_input = if prefill_plan.use_f16_batch_io {
                            &bs.matmul_in_f16
                        } else {
                            &bs.norm_buf
                        };
                        sb.pre_dispatch(&[qkv_input], &[&bs.qkv_buf]);
                        match qkv_layer_plan.input {
                            PrefillProjectionInputPlan::MatmulScratchF16 => {
                                encode_dequant_batch_f16in(
                                    metal_ops,
                                    encoder,
                                    fused_w,
                                    &bs.matmul_in_f16,
                                    &bs.qkv_buf,
                                    fused_qkv_m as u32,
                                    nt,
                                    dim as u32,
                                    lw.wq_dtype,
                                );
                            }
                            PrefillProjectionInputPlan::NormBufF32 => {
                                encode_dequant_batch(
                                    &metal_ops.dequant,
                                    &metal_ops.elementwise,
                                    encoder,
                                    fused_w,
                                    &bs.norm_buf,
                                    &bs.qkv_buf,
                                    &bs.matmul_in_f16,
                                    fused_qkv_m as u32,
                                    nt,
                                    dim as u32,
                                    lw.wq_dtype,
                                    false,
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.experimental_q5k_prefill_small_n,
                                );
                            }
                        }
                        sb.post_dispatch(&[qkv_input], &[&bs.qkv_buf]);
                        if qkv_layer_plan.llama_post
                            == crate::model::execution_plan::LlamaPrefillQkvPostPlan::FusedSplitRopeAppendKv
                        {
                            let kv_k = gpu_kv.k_buffer(layer);
                            let kv_v = gpu_kv.v_buffer(layer);
                            sb.pre_dispatch(
                                &[&bs.qkv_buf],
                                &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v],
                            );
                            metal_ops.elementwise.encode_qkv_split_rope_append_kv_batch(
                                encoder,
                                &bs.qkv_buf,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                kv_k,
                                kv_v,
                                prefill_plan.kv_f16,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                rope_start,
                                rope_step,
                                cfg.rope_freq_base,
                                cache_offset,
                                kv_dim as u32,
                            );
                            sb.post_dispatch(
                                &[&bs.qkv_buf],
                                &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v],
                            );
                        } else {
                            sb.pre_dispatch(
                                &[&bs.qkv_buf],
                                &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            );
                            metal_ops.elementwise.encode_qkv_split_rope_batch(
                                encoder,
                                &bs.qkv_buf,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                rope_start,
                                rope_step,
                                cfg.rope_freq_base,
                            );
                            sb.post_dispatch(
                                &[&bs.qkv_buf],
                                &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            );
                        }
                    } else {
                        // Separate Q/K/V projections — these can overlap when
                        // all three read norm_buf and write different outputs.
                        match qkv_layer_plan.input {
                            PrefillProjectionInputPlan::MatmulScratchF16 => {
                                // f16 path: all three share matmul_in_f16, can't overlap.
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.q_buf]);
                                encode_dequant_batch_f16in(
                                    metal_ops, encoder, wq_buf,
                                    &bs.matmul_in_f16, &bs.q_buf,
                                    q_dim as u32, nt, dim as u32, lw.wq_dtype,
                                );
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.q_buf]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.k_buf]);
                                encode_dequant_batch_f16in(
                                    metal_ops, encoder, wk_buf,
                                    &bs.matmul_in_f16, &bs.k_buf,
                                    kv_dim as u32, nt, dim as u32, lw.wk_dtype,
                                );
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.k_buf]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.v_buf]);
                                encode_dequant_batch_f16in(
                                    metal_ops, encoder, wv_buf,
                                    &bs.matmul_in_f16, &bs.v_buf,
                                    kv_dim as u32, nt, dim as u32, lw.wv_dtype,
                                );
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.v_buf]);
                            }
                            PrefillProjectionInputPlan::NormBufF32 => {
                                // f32 path: Q/K/V all read norm_buf, write
                                // different buffers → SmartBarrier skips
                                // barriers between them (GPU can overlap).
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.q_buf]);
                                encode_dequant_batch(
                                    &metal_ops.dequant, &metal_ops.elementwise,
                                    encoder, wq_buf, &bs.norm_buf, &bs.q_buf,
                                    &bs.matmul_in_f16, q_dim as u32, nt,
                                    dim as u32, lw.wq_dtype, false,
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.experimental_q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.q_buf]);
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.k_buf]);
                                encode_dequant_batch(
                                    &metal_ops.dequant, &metal_ops.elementwise,
                                    encoder, wk_buf, &bs.norm_buf, &bs.k_buf,
                                    &bs.matmul_in_f16, kv_dim as u32, nt,
                                    dim as u32, lw.wk_dtype, false,
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.experimental_q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.k_buf]);
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.v_buf]);
                                encode_dequant_batch(
                                    &metal_ops.dequant, &metal_ops.elementwise,
                                    encoder, wv_buf, &bs.norm_buf, &bs.v_buf,
                                    &bs.matmul_in_f16, kv_dim as u32, nt,
                                    dim as u32, lw.wv_dtype, false,
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.experimental_q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.v_buf]);
                            }
                        }
                    }

                    // ── Phase 1c: Batched RoPE + batched KV cache append ──
                    if qkv_layer_plan.llama_post
                        == crate::model::execution_plan::LlamaPrefillQkvPostPlan::Separate
                    {
                        sb.pre_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);
                        metal_ops.elementwise.encode_rope_batch(
                            encoder,
                            &bs.q_buf,
                            &bs.k_buf,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            rope_start,
                            rope_step,
                            cfg.rope_freq_base,
                        );
                        sb.post_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);
                    }

                    if qkv_layer_plan.llama_post
                        != crate::model::execution_plan::LlamaPrefillQkvPostPlan::FusedSplitRopeAppendKv
                    {
                        let cache_offset = (base_seq_len * kv_dim) as u32;
                        let kv_k = gpu_kv.k_buffer(layer);
                        let kv_v = gpu_kv.v_buffer(layer);
                        // K and V appends write to different KV buffers —
                        // SmartBarrier skips the barrier between them.
                        sb.pre_dispatch(&[&bs.k_buf], &[kv_k]);
                        metal_ops.elementwise.encode_kv_append_batch(
                            encoder,
                            &bs.k_buf,
                            kv_k,
                            prefill_plan.kv_f16,
                            cache_offset,
                            kv_dim as u32,
                            kv_dim as u32,
                            nt,
                        );
                        sb.post_dispatch(&[&bs.k_buf], &[kv_k]);
                        sb.pre_dispatch(&[&bs.v_buf], &[kv_v]);
                        metal_ops.elementwise.encode_kv_append_batch(
                            encoder,
                            &bs.v_buf,
                            kv_v,
                            prefill_plan.kv_f16,
                            cache_offset,
                            kv_dim as u32,
                            kv_dim as u32,
                            nt,
                        );
                        sb.post_dispatch(&[&bs.v_buf], &[kv_v]);
                    }

                    // ── Phase 2: Batched attention ──
                    if prefill_plan.attention == PrefillAttentionPlan::BatchLocalF16OutHd128 {
                        sb.pre_dispatch(
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            &[&bs.matmul_in_f16],
                        );
                        metal_ops.attention.encode_attention_prefill_f16out_hd128(
                            encoder,
                            &bs.q_buf,
                            &bs.k_buf,
                            &bs.v_buf,
                            &bs.matmul_in_f16,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                        );
                        sb.post_dispatch(
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            &[&bs.matmul_in_f16],
                        );
                    } else if prefill_plan.attention == PrefillAttentionPlan::BatchLocal {
                        sb.pre_dispatch(
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            &[&bs.attn_out],
                        );
                        metal_ops.attention.encode_attention_prefill_with_config(
                            encoder,
                            &bs.q_buf,
                            &bs.k_buf,
                            &bs.v_buf,
                            &bs.attn_out,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            prefill_plan.attention_dispatch,
                        );
                        sb.post_dispatch(
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            &[&bs.attn_out],
                        );
                    } else {
                        let kv_k = gpu_kv.k_buffer(layer);
                        let kv_v = gpu_kv.v_buffer(layer);
                        sb.pre_dispatch(&[&bs.q_buf, kv_k, kv_v], &[&bs.attn_out]);
                        metal_ops
                            .attention
                            .encode_attention_prefill_cached_with_config(
                                encoder,
                                &bs.q_buf,
                                kv_k,
                                kv_v,
                                &bs.attn_out,
                                prefill_plan.kv_f16,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                base_seq_len as u32,
                                0,
                                prefill_plan.attention_dispatch,
                            );
                        sb.post_dispatch(&[&bs.q_buf, kv_k, kv_v], &[&bs.attn_out]);
                    }

                    // ── Phase 3a: Batched output projection ──
                    let wo_buf = weight_cache.get(&lw.wo).unwrap();
                    let wo_input = if prefill_plan.use_f16_batch_io {
                        &bs.matmul_in_f16
                    } else {
                        &bs.attn_out
                    };
                    sb.pre_dispatch(&[wo_input], &[&bs.proj_buf]);
                    if prefill_plan.use_f16_batch_io {
                        if prefill_plan.wo_input == PrefillWoInputPlan::AttentionOutF32 {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.attn_out,
                                &bs.matmul_in_f16,
                                nt * q_dim as u32,
                            );
                        }
                        encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wo_buf,
                            &bs.matmul_in_f16,
                            &bs.proj_buf,
                            dim as u32,
                            nt,
                            q_dim as u32,
                            lw.wo_dtype,
                        );
                    } else {
                        encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wo_buf,
                            &bs.attn_out,
                            &bs.proj_buf,
                            &bs.matmul_in_f16,
                            dim as u32,
                            nt,
                            q_dim as u32,
                            lw.wo_dtype,
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.experimental_q5k_prefill_small_n,
                        );
                    }
                    sb.post_dispatch(&[wo_input], &[&bs.proj_buf]);

                    // ── Phase 3b: Batched residual + FFN norm ──
                    let ffn_nw_buf = weight_cache.get(&lw.ffn_norm).unwrap();
                    let ffn_norm_out = if prefill_plan.use_f16_batch_io {
                        &bs.matmul_in_f16
                    } else {
                        &bs.norm_buf
                    };
                    sb.pre_dispatch(
                        &[&bs.hidden, &bs.proj_buf],
                        &[&bs.hidden, ffn_norm_out],
                    );
                    if prefill_plan.use_f16_batch_io {
                        metal_ops
                            .elementwise
                            .encode_residual_add_rms_norm_out_batch_f16(
                                encoder,
                                &bs.hidden,
                                &bs.proj_buf,
                                ffn_nw_buf,
                                &bs.matmul_in_f16,
                                dim as u32,
                                nt,
                                eps,
                            );
                    } else {
                        metal_ops
                            .elementwise
                            .encode_residual_add_rms_norm_out_batch(
                                encoder,
                                &bs.hidden,
                                &bs.proj_buf,
                                ffn_nw_buf,
                                &bs.norm_buf,
                                dim as u32,
                                nt,
                                eps,
                            );
                    }
                    sb.post_dispatch(
                        &[&bs.hidden, &bs.proj_buf],
                        &[&bs.hidden, ffn_norm_out],
                    );

                    // ── Phase 3c: Batched gate + up ──
                    let wg_buf = weight_cache.get(&lw.wg).unwrap();
                    let wu_buf = weight_cache.get(&lw.wu).unwrap();
                    let ffn_layer_plan = DecodeExecutionPlan::llama_prefill_ffn_layer(
                        &prefill_plan,
                        lw.wg_dtype,
                        lw.wu_dtype,
                    );

                    let ffn_input = if prefill_plan.use_f16_batch_io {
                        &bs.matmul_in_f16
                    } else {
                        &bs.norm_buf
                    };
                    sb.pre_dispatch(&[ffn_input], &[&bs.gate_buf, &bs.up_buf]);
                    match ffn_layer_plan.input {
                        PrefillProjectionInputPlan::MatmulScratchF16 => {
                            if ffn_layer_plan.use_pair_kernel {
                                encode_dequant_batch_pair_f16in(
                                    &metal_ops.dequant,
                                    encoder,
                                    wg_buf,
                                    wu_buf,
                                    &bs.matmul_in_f16,
                                    &bs.gate_buf,
                                    &bs.up_buf,
                                    inter_dim as u32,
                                    nt,
                                    dim as u32,
                                    lw.wg_dtype,
                                );
                            } else {
                                encode_dequant_batch_f16in(
                                    metal_ops,
                                    encoder,
                                    wg_buf,
                                    &bs.matmul_in_f16,
                                    &bs.gate_buf,
                                    inter_dim as u32,
                                    nt,
                                    dim as u32,
                                    lw.wg_dtype,
                                );
                                encode_dequant_batch_f16in(
                                    metal_ops,
                                    encoder,
                                    wu_buf,
                                    &bs.matmul_in_f16,
                                    &bs.up_buf,
                                    inter_dim as u32,
                                    nt,
                                    dim as u32,
                                    lw.wu_dtype,
                                );
                            }
                        }
                        PrefillProjectionInputPlan::NormBufF32 => {
                            encode_dequant_batch(
                                &metal_ops.dequant,
                                &metal_ops.elementwise,
                                encoder,
                                wg_buf,
                                &bs.norm_buf,
                                &bs.gate_buf,
                                &bs.matmul_in_f16,
                                inter_dim as u32,
                                nt,
                                dim as u32,
                                lw.wg_dtype,
                                false,
                                prefill_plan.use_batch_simd,
                                prefill_plan.experimental_q5k_prefill_small_n,
                            );
                            encode_dequant_batch(
                                &metal_ops.dequant,
                                &metal_ops.elementwise,
                                encoder,
                                wu_buf,
                                &bs.norm_buf,
                                &bs.up_buf,
                                &bs.matmul_in_f16,
                                inter_dim as u32,
                                nt,
                                dim as u32,
                                lw.wu_dtype,
                                false,
                                prefill_plan.use_batch_simd,
                                prefill_plan.experimental_q5k_prefill_small_n,
                            );
                        }
                    }
                    sb.post_dispatch(&[ffn_input], &[&bs.gate_buf, &bs.up_buf]);

                    // ── Phase 3d: Batched activation ──
                    let act_out = match ffn_layer_plan.activation {
                        PrefillFfnActivationPlan::SiluMulScratchF16 => &bs.matmul_in_f16,
                        _ => &bs.gate_buf,
                    };
                    sb.pre_dispatch(&[&bs.gate_buf, &bs.up_buf], &[act_out]);
                    match ffn_layer_plan.activation {
                        PrefillFfnActivationPlan::SiluMulScratchF16 => {
                            metal_ops.elementwise.encode_silu_elementwise_mul_batch_f16(
                                encoder,
                                &bs.gate_buf,
                                &bs.up_buf,
                                &bs.matmul_in_f16,
                                inter_dim as u32,
                                nt,
                            );
                        }
                        PrefillFfnActivationPlan::SiluMulGateF32 => {
                            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                                encoder,
                                &bs.gate_buf,
                                &bs.up_buf,
                                inter_dim as u32,
                                nt,
                            );
                        }
                        PrefillFfnActivationPlan::GeluMulGateF32 => unreachable!(),
                    }
                    sb.post_dispatch(&[&bs.gate_buf, &bs.up_buf], &[act_out]);

                    // ── Phase 3e: Batched down projection ──
                    let wd_buf = weight_cache.get(&lw.wd).unwrap();
                    sb.pre_dispatch(&[act_out], &[&bs.proj_buf]);
                    match ffn_layer_plan.activation {
                        PrefillFfnActivationPlan::SiluMulScratchF16 => {
                            encode_dequant_batch_f16in(
                                metal_ops,
                                encoder,
                                wd_buf,
                                &bs.matmul_in_f16,
                                &bs.proj_buf,
                                dim as u32,
                                nt,
                                inter_dim as u32,
                                lw.wd_dtype,
                            );
                        }
                        PrefillFfnActivationPlan::SiluMulGateF32 => {
                            encode_dequant_batch(
                                &metal_ops.dequant,
                                &metal_ops.elementwise,
                                encoder,
                                wd_buf,
                                &bs.gate_buf,
                                &bs.proj_buf,
                                &bs.matmul_in_f16,
                                dim as u32,
                                nt,
                                inter_dim as u32,
                                lw.wd_dtype,
                                false,
                                prefill_plan.use_batch_simd,
                                prefill_plan.experimental_q5k_prefill_small_n,
                            );
                        }
                        PrefillFfnActivationPlan::GeluMulGateF32 => unreachable!(),
                    }
                    sb.post_dispatch(&[act_out], &[&bs.proj_buf]);

                    // ── Phase 3f: Batched residual (+ next layer's norm if not last) ──
                    let residual_plan = DecodeExecutionPlan::llama_prefill_residual_handoff(
                        &prefill_plan,
                        layer + 1 == n_layers,
                    );
                    let residual_norm_out = match residual_plan {
                        PrefillResidualHandoffPlan::ResidualOnly => None,
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF32 => {
                            Some(&bs.norm_buf)
                        }
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => {
                            Some(&bs.matmul_in_f16)
                        }
                    };
                    if let Some(nout) = residual_norm_out {
                        sb.pre_dispatch(
                            &[&bs.hidden, &bs.proj_buf],
                            &[&bs.hidden, nout],
                        );
                    } else {
                        sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    }
                    match residual_plan {
                        PrefillResidualHandoffPlan::ResidualOnly => {
                            metal_ops.elementwise.encode_elementwise_add_batch(
                                encoder,
                                &bs.hidden,
                                &bs.proj_buf,
                                dim as u32,
                                nt,
                            );
                        }
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF32 => {
                            let next_norm_w = weight_cache
                                .get(&cached.layers[layer + 1].attn_norm)
                                .unwrap();
                            metal_ops
                                .elementwise
                                .encode_residual_add_rms_norm_out_batch(
                                    encoder,
                                    &bs.hidden,
                                    &bs.proj_buf,
                                    next_norm_w,
                                    &bs.norm_buf,
                                    dim as u32,
                                    nt,
                                    eps,
                                );
                        }
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => {
                            let next_norm_w = weight_cache
                                .get(&cached.layers[layer + 1].attn_norm)
                                .unwrap();
                            metal_ops
                                .elementwise
                                .encode_residual_add_rms_norm_out_batch_f16(
                                    encoder,
                                    &bs.hidden,
                                    &bs.proj_buf,
                                    next_norm_w,
                                    &bs.matmul_in_f16,
                                    dim as u32,
                                    nt,
                                    eps,
                                );
                        }
                    }
                    if let Some(nout) = residual_norm_out {
                        sb.post_dispatch(
                            &[&bs.hidden, &bs.proj_buf],
                            &[&bs.hidden, nout],
                        );
                    } else {
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    }
                }

                // ── Post-loop: Final norm + LM head ──
                let fnw_buf = weight_cache.get(&cached.output_norm).unwrap();
                let lm_buf = weight_cache.get(&cached.lm_head).unwrap();
                match DecodeExecutionPlan::prefill_logits_plan(all_logits_buf.is_some()) {
                    PrefillLogitsPlan::BatchAllLogits => {
                        let logits_buf = all_logits_buf.as_ref().unwrap();
                        sb.pre_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
                        metal_ops.elementwise.encode_rms_norm_out_batch(
                            encoder,
                            &bs.hidden,
                            fnw_buf,
                            &bs.norm_buf,
                            dim as u32,
                            nt,
                            eps,
                        );
                        sb.post_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
                        sb.pre_dispatch(&[&bs.norm_buf], &[logits_buf]);
                        encode_batch_logits(
                            metal_ops,
                            encoder,
                            lm_buf,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            logits_buf,
                            vocab_size as u32,
                            nt,
                            dim as u32,
                            cached.lm_head_dtype,
                            prefill_plan.use_f16_batch_io,
                            prefill_plan.use_batch_simd,
                        );
                    }
                    PrefillLogitsPlan::LastTokenMatvec => {
                        sb.pre_dispatch(&[&bs.hidden], &[&s.hidden]);
                        let last_off = (n_tokens - 1) * dim * 4;
                        metal_ops.elementwise.encode_buffer_copy(
                            encoder, &bs.hidden, last_off, &s.hidden, 0, dim as u32,
                        );
                        sb.post_dispatch(&[&bs.hidden], &[&s.hidden]);
                        sb.pre_dispatch(&[&s.hidden], &[&s.hidden]);
                        metal_ops
                            .elementwise
                            .encode_rms_norm(encoder, &s.hidden, fnw_buf, dim as u32, eps);
                        sb.post_dispatch(&[&s.hidden], &[&s.hidden]);
                        sb.pre_dispatch(&[&s.hidden], &[&s.logits_buf]);
                        encode_dequant_matvec(
                            metal_ops,
                            encoder,
                            lm_buf,
                            &s.hidden,
                            &s.logits_buf,
                            vocab_size as u32,
                            dim as u32,
                            cached.lm_head_dtype,
                        );
                    }
                }

                Ok(())
            })?;
        }

        // v2: advance GPU KV only — no CPU mirror to sync
        gpu_kv.finalize_batch(n_tokens);

        if let Some(logits_all) = logits_all {
            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    all_logits_buf
                        .as_ref()
                        .expect("batch logits buffer must exist for all-logits path")
                        .contents()
                        .as_ptr() as *const f32,
                    n_tokens * vocab_size,
                )
            };
            logits_all.resize(n_tokens * vocab_size, 0.0);
            logits_all.copy_from_slice(logits_gpu);
        } else if let Some(logits) = last_logits {
            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    s.logits_buf.contents().as_ptr() as *const f32,
                    vocab_size,
                )
            };
            logits[..vocab_size].copy_from_slice(logits_gpu);
        }

        Ok(())
    }
}

impl ForwardPass for LlamaForward {
    fn forward_single(
        &self,
        ctx: &ForwardContext,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        // v2: GPU gate uses use_gpu_decode() + kv.as_gpu_mut() — no AX_CPU_ONLY check
        if ctx.backend.use_gpu_decode()
            && let Some(metal_ops) = ctx.backend.metal_ops()
            && let Some(gpu_kv) = kv.as_gpu_mut()
            && gpu_decode_quant_supported(weights)
        {
            if let Some(ops_ref) = ops.as_deref_mut() {
                let t = OpTimer::start();
                let r = self.forward_single_gpu_unified(
                    ctx,
                    metal_ops,
                    token_id,
                    position,
                    gpu_kv,
                    weights,
                    logits,
                    Some(ops_ref),
                );
                ops_ref.gpu += t.elapsed();
                return r;
            }
            return self.forward_single_gpu_unified(
                ctx, metal_ops, token_id, position, gpu_kv, weights, logits, None,
            );
        }

        // CPU fallback path
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let vocab_size = cfg.vocab_size as usize;

        assert!(logits.len() >= vocab_size);

        let cpu_kv = kv
            .as_cpu_mut()
            .expect("LlamaForward CPU path requires ModelKv::Cpu");

        // --- Step 1: Token embedding (single-row dequant) ---
        let mut hidden = vec![0.0f32; dim];
        timed!(
            ops,
            dequant,
            weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?
        );

        // Scratch buffers (reused across layers)
        let mut norm_buf = vec![0.0f32; dim];
        let mut q_buf = vec![0.0f32; n_heads * head_dim];
        let mut k_buf = vec![0.0f32; n_kv_heads * head_dim];
        let mut v_buf = vec![0.0f32; n_kv_heads * head_dim];
        let mut attn_out = vec![0.0f32; n_heads * head_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; dim];

        // --- Step 2: Transformer layers ---
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            // 2a. Attention norm (RMSNorm)
            let attn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?
            );
            timed!(
                ops,
                norm,
                rms_norm::rms_norm_out(&hidden, attn_norm_w, &mut norm_buf, cfg.rms_norm_eps)
            );

            // 2b. Q/K/V projections
            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;

            timed!(ops, matmul, {
                ctx.backend.batch_dequant_matvec(
                    &[
                        (wq_raw, wq_dtype, n_heads * head_dim),
                        (wk_raw, wk_dtype, n_kv_heads * head_dim),
                        (wv_raw, wv_dtype, n_kv_heads * head_dim),
                    ],
                    &norm_buf,
                    dim,
                    &mut [&mut q_buf, &mut k_buf, &mut v_buf],
                );
            });

            // 2c. RoPE on Q and K (apply linear scaling if configured)
            let rope_position = match cfg.rope_scaling {
                crate::model::config::RopeScaling::Linear(factor) => position as f32 / factor,
                crate::model::config::RopeScaling::None => position as f32,
            };
            timed!(
                ops,
                rope,
                rope::apply_rope_multi_head_scaled(
                    &mut q_buf,
                    &mut k_buf,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_position,
                    cfg.rope_freq_base,
                )
            );

            // 2d. Update KV cache (v2 CPU path: append_and_advance per layer, finalize after last)
            cpu_kv.append_and_advance(layer, &k_buf, &v_buf);

            // 2e. Multi-head attention
            let seq_len = cpu_kv.seq_len() + 1;
            timed!(
                ops,
                attention,
                attention::multi_head_attention(
                    &q_buf,
                    cpu_kv.k_slice_including_current(layer, seq_len),
                    cpu_kv.v_slice_including_current(layer, seq_len),
                    &mut attn_out,
                    ctx.attn_params,
                    seq_len,
                )
            );

            // 2f. Output projection via backend
            let (wo_raw, wo_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            timed!(
                ops,
                matmul,
                ctx.backend.dequant_matmul(
                    wo_raw,
                    wo_dtype,
                    &attn_out,
                    &mut proj_buf,
                    dim,
                    1,
                    n_heads * head_dim,
                )
            );

            // 2g. Residual add
            silu::elementwise_add(&mut hidden, &proj_buf);

            // 2h. FFN norm
            let ffn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?
            );
            timed!(
                ops,
                norm,
                rms_norm::rms_norm_out(&hidden, ffn_norm_w, &mut norm_buf, cfg.rms_norm_eps)
            );

            // 2i. Gate and Up projections
            let (wg_raw, wg_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;

            timed!(ops, matmul, {
                ctx.backend.batch_dequant_matvec(
                    &[(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)],
                    &norm_buf,
                    dim,
                    &mut [&mut gate_buf, &mut up_buf],
                );
            });

            // 2j. SiLU(gate) * up
            silu::silu_elementwise_mul(&mut gate_buf, &up_buf);

            // 2k. Down projection via backend
            let (wd_raw, wd_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            timed!(
                ops,
                matmul,
                ctx.backend.dequant_matmul(
                    wd_raw,
                    wd_dtype,
                    &gate_buf,
                    &mut down_buf,
                    dim,
                    1,
                    inter_dim,
                )
            );

            // 2l. Residual add
            silu::elementwise_add(&mut hidden, &down_buf);
        }

        // Advance CPU KV cache after all layers processed
        cpu_kv.finalize_token();

        // --- Step 3: Final RMSNorm ---
        let final_norm_w = timed!(ops, dequant, weights.f32_slice("output_norm.weight")?);
        timed!(
            ops,
            norm,
            rms_norm::rms_norm(&mut hidden, final_norm_w, cfg.rms_norm_eps)
        );

        // --- Step 4: LM head -> logits via backend ---
        let lm_weight_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        timed!(
            ops,
            matmul,
            ctx.backend
                .dequant_matmul(lm_raw, lm_dtype, &hidden, logits, vocab_size, 1, dim)
        );

        Ok(())
    }

    fn forward_batch(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let prefill_plan =
            PrefillExecutionPlan::for_forward_batch(ctx, kv, weights, token_ids.len(), false)?;

        if prefill_plan.mode == PrefillMode::GpuBatch {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
            match self.forward_batch_gpu_unified(
                ctx,
                metal_ops,
                token_ids,
                gpu_kv,
                weights,
                Some(logits),
                None,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!("GPU batch prefill failed, falling back to serial: {e}");
                }
            }
        }

        // Fallback: serial forward_single
        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, None)?;
        }
        Ok(())
    }

    fn forward_batch_all_logits(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        let prefill_plan =
            PrefillExecutionPlan::for_forward_batch(ctx, kv, weights, token_ids.len(), true)?;

        if prefill_plan.mode == PrefillMode::GpuBatch {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
            match self.forward_batch_gpu_unified(
                ctx,
                metal_ops,
                token_ids,
                gpu_kv,
                weights,
                None,
                Some(logits_all),
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!(
                        "GPU batch all-logits prefill failed, falling back to serial: {e}"
                    );
                }
            }
        }

        ForwardPass::forward_batch_all_logits(self, ctx, token_ids, kv, weights, logits_all)
    }

    fn validate_config(&self, _config: &ModelConfig) -> anyhow::Result<()> {
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "llama"
    }

    fn supports_pipelined_decode(&self, ctx: &ForwardContext) -> bool {
        ctx.backend.metal_ops().is_some()
    }

    fn embed_pipelined_token(
        &self,
        ctx: &ForwardContext,
        token_id: u32,
        hidden_buf: &ax_metal::MetalBuffer,
        weights: &WeightStore,
    ) -> anyhow::Result<()> {
        let dim = ctx.config.embedding_dim as usize;
        let hidden = unsafe {
            std::slice::from_raw_parts_mut(hidden_buf.contents().as_ptr() as *mut f32, dim)
        };
        weights
            .dequantize_row("token_embd.weight", token_id as usize, hidden)
            .map(|_| ())
    }

    fn encode_pending_decode_step(
        &self,
        ctx: &ForwardContext,
        hidden_buf: &ax_metal::MetalBuffer,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
    ) -> anyhow::Result<Option<ax_metal::PendingFrame>> {
        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(None);
        };
        let Some(gpu_kv) = kv.as_gpu_mut() else {
            return Ok(None);
        };
        let frame = encode_llama_pending_step(
            metal_ops, ctx.config, hidden_buf, position, gpu_kv, weights,
        )?;
        Ok(Some(frame))
    }
}

/// LLaMA model state for inference.
///
/// Internally delegates to an architecture-specific `ForwardPass` implementation.
/// For "llama" architecture, this uses `LlamaForward`. Other architectures
/// (Qwen3, Gemma3) use their own implementations selected via the arch registry.
pub struct LlamaModel {
    pub config: ModelConfig,
    attn_params: AttentionParams,
    backend: Box<dyn Backend>,
    forward: Box<dyn ForwardPass>,
}

impl LlamaModel {
    fn forward_context(&self) -> ForwardContext<'_> {
        ForwardContext {
            config: &self.config,
            attn_params: &self.attn_params,
            backend: &*self.backend,
        }
    }

    /// Create a new LLaMA model with CPU backend (default).
    pub fn new(config: ModelConfig) -> Self {
        Self::with_backend(config, Box::new(CpuBackend))
    }

    /// Create a new LLaMA model with a specific compute backend.
    ///
    /// The forward pass implementation is selected based on `config.architecture`
    /// via the architecture registry.
    ///
    /// # Panics
    /// Panics if the architecture is not supported. Use `arch_registry::forward_for_arch`
    /// directly if you need fallible construction.
    pub fn with_backend(config: ModelConfig, backend: Box<dyn Backend>) -> Self {
        let forward = crate::model::arch_registry::forward_for_arch(&config.architecture)
            .unwrap_or_else(|e| {
                panic!(
                    "unsupported model architecture '{}': {e}",
                    config.architecture
                )
            });

        if let Err(e) = forward.validate_config(&config) {
            tracing::warn!(
                arch = config.architecture,
                "Model config validation warning: {e}"
            );
        }

        let attn_params = AttentionParams::new(
            config.n_heads as usize,
            config.n_kv_heads as usize,
            config.head_dim as usize,
        );
        Self {
            config,
            attn_params,
            backend,
            forward,
        }
    }

    fn prefill_plan_has_q8_weights(&self, weights: &WeightStore) -> anyhow::Result<bool> {
        for layer in 0..self.config.n_layers as usize {
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                let name = format!("blk.{layer}.{suffix}");
                let (_, dtype) = weights.raw_with_dtype(&name)?;
                if dtype == GgmlType::Q8_0 {
                    return Ok(true);
                }
            }
        }

        let lm_weight_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (_, lm_head_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        Ok(lm_head_dtype == GgmlType::Q8_0)
    }

    fn prefill_attention_route(
        &self,
        plan: GpuBatchPrefillExecutionPlan,
        base_seq_len: usize,
        n_tokens: usize,
        head_dim: u32,
        sliding_window: u32,
    ) -> String {
        match plan.attention {
            PrefillAttentionPlan::BatchLocalF16OutHd128 => {
                "mistral_f16out_hd128/profile_preferred".to_string()
            }
            PrefillAttentionPlan::BatchLocal => {
                let selection = plan
                    .attention_dispatch
                    .prefill_local_candidate_selection(n_tokens as u32, head_dim);
                format!("{}/{}", selection.label(), selection.stability.label())
            }
            PrefillAttentionPlan::Cached => {
                let selection = plan.attention_dispatch.prefill_cached_candidate_selection(
                    plan.kv_f16,
                    n_tokens as u32,
                    head_dim,
                    base_seq_len as u32,
                    sliding_window,
                );
                format!("{}/{}", selection.label(), selection.stability.label())
            }
        }
    }

    /// Create a KV cache sized for this model.
    ///
    /// v2: returns `ModelKv` via the backend-side planner.
    /// Paged KV remains deferred to v2.1.
    pub fn create_model_kv(&self) -> ModelKv {
        self.kv_plan().build(self.backend.as_ref())
    }

    /// Create a KV cache matched to the active decode support of the loaded weights.
    ///
    /// Mixed-quant models can resolve to CPU decode even when the backend default is
    /// GPU decode. In that case we must allocate CPU KV up front to avoid a
    /// GPU-KV/CPU-decode mismatch at runtime.
    pub fn create_model_kv_for_weights(&self, weights: &WeightStore) -> ModelKv {
        self.kv_plan()
            .build_decode_compatible(self.backend.as_ref(), gpu_decode_quant_supported(weights))
    }

    /// Resolve the backend-side KV allocation plan for this model.
    pub fn kv_plan(&self) -> crate::backend::KvPlan {
        crate::backend::KvPlanner::plan(self.backend.as_ref(), &self.config)
    }

    pub fn kv_plan_with_requirements(
        &self,
        requirements: crate::backend::KvPlannerRequirements,
    ) -> anyhow::Result<crate::backend::KvPlan> {
        crate::backend::KvPlanner::plan_with_requirements(
            self.backend.as_ref(),
            &self.config,
            requirements,
        )
    }

    pub fn decode_plan_summary(
        &self,
        kv: &ModelKv,
        intent: DecodeIntent,
        allow_pipelined: bool,
    ) -> String {
        DecodeExecutionPlan::for_model(self, kv, intent, allow_pipelined).summary_label()
    }

    pub fn prefill_plan_summary(
        &self,
        weights: &WeightStore,
        kv: &ModelKv,
        n_tokens: usize,
    ) -> anyhow::Result<String> {
        let ctx = self.forward_context();
        let mode_plan =
            match PrefillExecutionPlan::for_forward_batch(&ctx, kv, weights, n_tokens, false) {
                Ok(plan) => plan,
                Err(_) if self.arch_name() == "qwen3" => {
                    return Ok("mode=serial reason=unsupported_qwen3_layout".to_string());
                }
                Err(e) => return Err(e),
            };
        if mode_plan.mode == PrefillMode::Serial {
            return Ok(mode_plan.summary_label());
        }

        let gpu_kv = kv.as_gpu().unwrap();
        let metal_ops = self.metal_ops().unwrap();

        let base_seq_len = gpu_kv.seq_len();
        let summary = match self.arch_name() {
            "llama" => {
                let plan = DecodeExecutionPlan::llama_prefill(
                    metal_ops,
                    gpu_kv,
                    base_seq_len,
                    n_tokens as u32,
                    self.config.head_dim,
                    self.prefill_plan_has_q8_weights(weights)?,
                    gpu_prefill_uses_experimental_q5k(weights),
                    gpu_prefill_experimental_q5k_small_n_auto_eligible(weights),
                    metal_prefill_attn_f16out_enabled(),
                    metal_prefill_use_cached0_enabled(),
                    metal_prefill_split_rope_append_enabled(),
                );
                let route = self.prefill_attention_route(
                    plan,
                    base_seq_len,
                    n_tokens,
                    self.config.head_dim,
                    0,
                );
                plan.summary_label(
                    mode_plan.summary_label().trim_start_matches("mode="),
                    &route,
                )
            }
            "qwen3" => {
                let sliding_window = self.config.sliding_window_size.unwrap_or(0);
                let plan = DecodeExecutionPlan::qwen3_prefill(
                    metal_ops,
                    gpu_kv,
                    base_seq_len,
                    n_tokens as u32,
                    self.config.head_dim,
                    sliding_window,
                    gpu_prefill_uses_experimental_q5k(weights),
                    gpu_prefill_experimental_q5k_small_n_auto_eligible(weights),
                );
                let route = self.prefill_attention_route(
                    plan,
                    base_seq_len,
                    n_tokens,
                    self.config.head_dim,
                    plan.attention_sliding_window,
                );
                plan.summary_label(
                    mode_plan.summary_label().trim_start_matches("mode="),
                    &route,
                )
            }
            "gemma3" => {
                let plan = DecodeExecutionPlan::gemma3_prefill(
                    metal_ops,
                    gpu_kv,
                    n_tokens as u32,
                    gpu_prefill_uses_experimental_q5k(weights),
                    gpu_prefill_experimental_q5k_small_n_auto_eligible(weights),
                );
                let route = self.prefill_attention_route(
                    plan,
                    base_seq_len,
                    n_tokens,
                    self.config.head_dim,
                    0,
                );
                plan.summary_label(
                    mode_plan.summary_label().trim_start_matches("mode="),
                    &route,
                )
            }
            _ => mode_plan.summary_label(),
        };

        Ok(summary)
    }

    /// Run a single decode step: one token in, logits out.
    pub fn forward_single(
        &self,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let ctx = self.forward_context();
        self.forward
            .forward_single(&ctx, token_id, position, kv, weights, logits, None)
    }

    /// Run batched prefill: process all tokens, return only last token's logits.
    ///
    /// Uses GPU batched attention when available, otherwise falls back to serial.
    pub fn forward_batch(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let ctx = self.forward_context();
        self.forward
            .forward_batch(&ctx, token_ids, kv, weights, logits)
    }

    /// Profiled decode step: same as `forward_single` but records per-operation
    /// timing into the provided `OpBreakdown`.
    pub fn forward_single_profiled(
        &self,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        let ctx = self.forward_context();
        self.forward
            .forward_single(&ctx, token_id, position, kv, weights, logits, Some(ops))
    }

    /// Run a forward pass on N tokens and return logits for every position.
    ///
    /// `logits_all` is resized to `token_ids.len() * vocab_size` on return.
    /// `logits_all[i * vocab_size .. (i+1) * vocab_size]` contains the unnormalized
    /// logit vector for token position `kv.seq_len() + i` (before the call).
    ///
    /// Used by speculative decoding to obtain target model probabilities for
    /// all K+1 tokens (last accepted + K draft candidates) in one call.
    ///
    /// Implementation note (v2.0): runs N sequential `forward_single` calls.
    /// v2.1 will replace this with a single GPU batch dispatch that runs LM-head
    /// on every token position.
    pub fn forward_batch_all_logits(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        let ctx = self.forward_context();
        self.forward
            .forward_batch_all_logits(&ctx, token_ids, kv, weights, logits_all)
    }

    /// Get the name of the architecture used by this model's forward pass.
    pub fn arch_name(&self) -> &str {
        self.forward.arch_name()
    }

    /// True when this model can safely use the current pipelined decode implementation.
    pub fn supports_pipelined_decode(&self) -> bool {
        self.forward
            .supports_pipelined_decode(&self.forward_context())
    }

    // ── Pipelining helpers (PERF-002) ────────────────────────────────────────

    /// Return the Metal device if this model uses a GPU backend, else `None`.
    pub fn metal_device(&self) -> Option<&ax_metal::MetalDevice> {
        self.backend.metal_ops().map(|m| &m.device)
    }

    pub(crate) fn metal_ops(&self) -> Option<&MetalOps> {
        self.backend.metal_ops()
    }

    /// Reset backend-local Metal performance counters for this model.
    pub fn reset_metal_perf_counters(&self) {
        if let Some(device) = self.metal_device() {
            device.reset_perf_counters();
        }
    }

    /// Read backend-local Metal performance counters for this model.
    pub fn read_metal_perf_counters(&self) -> ax_metal::PerfCounters {
        self.metal_device()
            .map(ax_metal::MetalDevice::perf_counters)
            .unwrap_or_default()
    }

    /// Allocate a Metal shared-memory buffer of `bytes` bytes.
    ///
    /// Returns `Err` if no Metal device is available.
    pub fn alloc_metal_buf(&self, bytes: usize) -> anyhow::Result<ax_metal::MetalBuffer> {
        let m = self
            .backend
            .metal_ops()
            .ok_or_else(|| anyhow::anyhow!("No Metal backend available"))?;
        ax_metal::MetalBuffer::new(m.device.device(), bytes)
    }

    /// Write the embedding for `token_id` into a Metal buffer (zero-copy UMA write).
    ///
    /// `buf` must be at least `embedding_dim * 4` bytes (f32 per element).
    pub fn embed_token_into(
        &self,
        token_id: u32,
        buf: &ax_metal::MetalBuffer,
        weights: &WeightStore,
    ) -> anyhow::Result<()> {
        self.forward
            .embed_pipelined_token(&self.forward_context(), token_id, buf, weights)
    }

    /// Encode a single decode step into a [`ax_metal::PendingFrame`] without committing.
    ///
    /// Returns `Some(frame)` if the model uses a GPU backend and `kv` is a GPU KV
    /// cache, otherwise `None` (caller should fall back to `forward_single`).
    ///
    /// **Pipelining contract** — the caller is responsible for:
    /// 1. Calling [`prewarm_kv_capacity`] before the decode loop (prevents
    ///    reallocation while a prior frame may be inflight on the GPU).
    /// 2. Writing the embedding into `hidden_buf` via [`embed_token_into`]
    ///    *after* encoding but *before* committing the frame.
    /// 3. Calling [`advance_gpu_kv_token`] after [`ax_metal::MetalDevice::wait_frame`].
    pub fn encode_pending_decode_step(
        &self,
        hidden_buf: &ax_metal::MetalBuffer,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
    ) -> anyhow::Result<Option<ax_metal::PendingFrame>> {
        self.forward.encode_pending_decode_step(
            &self.forward_context(),
            hidden_buf,
            position,
            kv,
            weights,
        )
    }

    /// Pre-allocate GPU KV capacity for at least `needed` positions.
    ///
    /// Must be called once before the pipelined decode loop starts, so that
    /// `ensure_capacity` is guaranteed to be a no-op (and therefore safe to
    /// call while a command buffer is inflight).
    pub fn prewarm_kv_capacity(&self, kv: &mut ModelKv, needed: usize) -> anyhow::Result<()> {
        let Some(metal_ops) = self.backend.metal_ops() else {
            return Ok(());
        };
        let Some(gpu_kv) = kv.as_gpu_mut() else {
            return Ok(());
        };
        gpu_kv.ensure_capacity(&metal_ops.device, needed)
    }

    /// Advance the GPU KV cache seq_len by 1 after a pipelined decode step completes.
    ///
    /// Must be called after [`ax_metal::MetalDevice::wait_frame`], not before.
    pub fn advance_gpu_kv_token(&self, kv: &mut ModelKv) {
        if let Some(gpu_kv) = kv.as_gpu_mut() {
            gpu_kv.finalize_token();
        }
    }

    /// Copy logits from the GPU scratch buffer into the provided CPU slice.
    ///
    /// Must be called after the inflight frame has completed (after `wait_frame`).
    /// `logits` must have length >= `vocab_size`.
    pub fn read_gpu_logits(&self, logits: &mut [f32]) -> anyhow::Result<()> {
        let Some(metal_ops) = self.backend.metal_ops() else {
            anyhow::bail!("read_gpu_logits: no Metal backend");
        };
        metal_ops.init_scratches(&self.config);
        let guard = metal_ops.scratches();
        let s = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU scratch buffers not initialized"))?;
        let vocab = self.config.vocab_size as usize;
        let logits_gpu = unsafe {
            std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab)
        };
        logits[..vocab].copy_from_slice(logits_gpu);
        self.forward
            .postprocess_pipelined_logits(&self.forward_context(), &mut logits[..vocab])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: "llama".into(),
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            embedding_dim: 8,
            head_dim: 4,
            intermediate_dim: 16,
            context_length: 32,
            vocab_size: 4,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
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
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
        }
    }

    #[test]
    fn test_llama_model_creation() {
        let config = tiny_config();
        let model = LlamaModel::new(config.clone());
        assert_eq!(model.config.n_layers, 1);
        assert_eq!(model.attn_params.n_heads, 2);
        assert_eq!(model.attn_params.n_kv_heads, 2);
        assert_eq!(model.attn_params.head_dim, 4);
    }

    #[test]
    fn test_llama_model_with_backend() {
        let config = tiny_config();
        let backend: Box<dyn Backend> = Box::new(CpuBackend);
        let model = LlamaModel::with_backend(config.clone(), backend);
        assert_eq!(model.config.n_layers, 1);
    }

    #[test]
    fn test_model_kv_creation_cpu() {
        let config = tiny_config();
        let model = LlamaModel::new(config);
        let kv = model.create_model_kv();
        assert_eq!(kv.seq_len(), 0);
        assert!(!kv.is_gpu());
    }

    #[test]
    fn test_model_kv_creation_hybrid_cpu_decode_uses_cpu_kv() {
        let Ok(backend) = crate::backend::metal::HybridCpuDecodeBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config(), Box::new(backend));
        let kv = model.create_model_kv();
        assert!(
            !kv.is_gpu(),
            "HybridCpuDecode must allocate a CPU KV cache because decode runs on CPU"
        );
    }

    #[test]
    fn test_profiled_forward_same_signature() {
        let mut ops = crate::metrics::OpBreakdown::new();
        assert_eq!(ops.total(), std::time::Duration::ZERO);
        ops.matmul += std::time::Duration::from_nanos(1);
        assert!(ops.total() > std::time::Duration::ZERO);
    }

    #[test]
    fn test_attention_params_from_config() {
        let config = ModelConfig {
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
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
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
        };
        let model = LlamaModel::new(config);
        assert_eq!(model.attn_params.n_heads, 32);
        assert_eq!(model.attn_params.n_kv_heads, 8);
        assert_eq!(model.attn_params.head_dim, 128);
    }

    #[test]
    fn test_llama_forward_arch_name() {
        let fwd = LlamaForward;
        assert_eq!(fwd.arch_name(), "llama");
    }

    #[test]
    fn test_model_selects_llama_forward() {
        let config = tiny_config();
        let model = LlamaModel::new(config);
        assert_eq!(model.arch_name(), "llama");
    }
}
