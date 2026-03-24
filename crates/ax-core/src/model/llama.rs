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
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    encode_batch_logits, encode_dequant_batch, encode_dequant_batch_f16in,
    encode_dequant_batch_pair_f16in, encode_dequant_matvec, gpu_batch_logits_supported,
    gpu_decode_quant_supported, gpu_quant_supported,
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
/// - unset / `1` / `true` / `on`   -> enabled (default)
/// - `0` / `false` / `off`         -> disabled
fn metal_decode_barriers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_DECODE_BARRIERS") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => true,
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
    kv_f16: bool,
    gpu_kv: &crate::kv::GpuKv,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_metal::MetalBuffer>,
    fused_qkv_cache: &rustc_hash::FxHashMap<(usize, usize, usize), ax_metal::MetalBuffer>,
    decode_barriers: bool,
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
    let use_fused_qkv = crate::backend::metal::metal_fused_qkv_enabled_for_arch(&cfg.architecture);

    let decode_barrier = |encoder: &ax_metal::MetalEncoder| {
        if decode_barriers {
            ax_metal::barrier_buffers(encoder);
        }
    };

    for layer in 0..n_layers {
        let lw = &cached.layers[layer];

        let norm_w_buf = weight_cache.get(&lw.attn_norm).unwrap();
        let wq_buf = weight_cache.get(&lw.wq).unwrap();
        let wk_buf = weight_cache.get(&lw.wk).unwrap();
        let wv_buf = weight_cache.get(&lw.wv).unwrap();
        let fused_qkv_dtype_ok = lw.wq_dtype == lw.wk_dtype
            && lw.wq_dtype == lw.wv_dtype
            && matches!(lw.wq_dtype, GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0);
        let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
        let fused_qkv_buf = if use_fused_qkv && fused_qkv_dtype_ok {
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
            encode_dequant_matvec(
                metal_ops,
                encoder,
                fused_w,
                &s.norm_buf,
                &s.qkv_buf,
                (q_dim + 2 * kv_dim) as u32,
                dim as u32,
                lw.wq_dtype,
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
                kv_f16,
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
            encode_dequant_matvec(
                metal_ops,
                encoder,
                wq_buf,
                &s.norm_buf,
                &s.q_buf,
                q_dim as u32,
                dim as u32,
                lw.wq_dtype,
            );
            encode_dequant_matvec(
                metal_ops,
                encoder,
                wk_buf,
                &s.norm_buf,
                &s.k_buf,
                kv_dim as u32,
                dim as u32,
                lw.wk_dtype,
            );
            encode_dequant_matvec(
                metal_ops,
                encoder,
                wv_buf,
                &s.norm_buf,
                &s.v_buf,
                kv_dim as u32,
                dim as u32,
                lw.wv_dtype,
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
                kv_f16,
                kv_offset,
                kv_dim as u32,
            );
            metal_ops.elementwise.encode_kv_append(
                encoder,
                &s.v_buf,
                gpu_kv.v_buffer(layer),
                kv_f16,
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
        metal_ops.attention.encode_attention_decode_with_scratch(
            encoder,
            &s.q_buf,
            gpu_kv.k_buffer(layer),
            gpu_kv.v_buffer(layer),
            &s.attn_out,
            &s.splitk_partial_out,
            &s.splitk_partial_lse,
            kv_f16,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            0,
            full_seq_len as u32,
        );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_attention += t.elapsed();
        }
        decode_barrier(encoder);

        // 6. Output projection
        let wo_buf = weight_cache.get(&lw.wo).unwrap();
        let t = OpTimer::start();
        encode_dequant_matvec(
            metal_ops,
            encoder,
            wo_buf,
            &s.attn_out,
            &s.proj_buf,
            dim as u32,
            q_dim as u32,
            lw.wo_dtype,
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
        encode_dequant_matvec(
            metal_ops,
            encoder,
            wg_buf,
            &s.norm_buf,
            &s.gate_buf,
            inter_dim as u32,
            dim as u32,
            lw.wg_dtype,
        );
        encode_dequant_matvec(
            metal_ops,
            encoder,
            wu_buf,
            &s.norm_buf,
            &s.up_buf,
            inter_dim as u32,
            dim as u32,
            lw.wu_dtype,
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
        encode_dequant_matvec(
            metal_ops,
            encoder,
            wd_buf,
            &s.gate_buf,
            &s.down_buf,
            dim as u32,
            inter_dim as u32,
            lw.wd_dtype,
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
    decode_barriers: bool,
) {
    let dim = cfg.embedding_dim as usize;
    let vocab_size = cfg.vocab_size as usize;
    let eps = cfg.rms_norm_eps;

    let decode_barrier = |encoder: &ax_metal::MetalEncoder| {
        if decode_barriers {
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
    encode_dequant_matvec(
        metal_ops,
        encoder,
        lm_buf,
        hidden_buf,
        &s.logits_buf,
        vocab_size as u32,
        dim as u32,
        cached.lm_head_dtype,
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
    let kv_f16 = gpu_kv.is_f16();

    // Pre-condition: capacity must already be reserved by caller.
    // This call is a guaranteed no-op if prewarm_kv_capacity was called.
    gpu_kv.ensure_capacity(&metal_ops.device, position + 1)?;

    // Init scratch buffers (no-op after first call)
    metal_ops.init_scratches(cfg);

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
    let decode_barriers = metal_decode_barriers_enabled();

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
            kv_f16,
            gpu_kv,
            cached,
            &weight_cache,
            &fused_qkv_cache,
            decode_barriers,
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
            decode_barriers,
        );
        Ok(())
    })
}

/// LLaMA-family forward pass implementation.
///
/// Used for llama, mistral, codellama, and any architecture that follows
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
        let use_precomputed_f16 =
            crate::backend::metal::metal_precompute_f16_enabled_for_model(cfg);

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
                && crate::backend::metal::metal_fused_qkv_enabled()
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

        assert!(logits.len() >= vocab_size);

        metal_ops.init_scratches(cfg);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();

        let kv_f16 = gpu_kv.is_f16();

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
            let decode_barriers = metal_decode_barriers_enabled();
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
                        kv_f16,
                        gpu_kv,
                        cached,
                        &weight_cache,
                        &fused_qkv_cache,
                        decode_barriers,
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
                        decode_barriers,
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
                        kv_f16,
                        gpu_kv,
                        cached,
                        &weight_cache,
                        &fused_qkv_cache,
                        decode_barriers,
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
                        decode_barriers,
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

        let kv_f16 = gpu_kv.is_f16();

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
            let use_f16_batch_io =
                crate::backend::metal::metal_batch_f16_io_enabled_for_arch(&cfg.architecture)
                    || has_q8_weights;
            let use_f16_pair =
                crate::backend::metal::metal_batch_f16_pair_enabled_for_arch(&cfg.architecture);
            let use_fused_qkv =
                crate::backend::metal::metal_fused_qkv_enabled_for_arch(&cfg.architecture);
            let use_batch_simd =
                crate::backend::metal::metal_batch_simd_enabled_for_arch(&cfg.architecture);
            let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

            metal_ops.device.execute_sync_concurrent(|encoder| {
                let nt = n_tokens as u32;

                // First layer's Phase 1a: RMSNorm (standalone, before loop).
                // Subsequent layers' Phase 1a is fused with the previous layer's
                // Phase 3f residual add (saves 1 dispatch + 1 barrier per layer).
                {
                    let norm_w_buf = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
                    if use_f16_batch_io {
                        metal_ops.elementwise.encode_rms_norm_out_batch_f16(
                            encoder,
                            &bs.hidden,
                            norm_w_buf,
                            &bs.matmul_in_f16,
                            dim as u32,
                            nt,
                            eps,
                        );
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
                    }
                    ax_metal::barrier_buffers(encoder);
                }

                for layer in 0..n_layers {
                    let lw = &cached.layers[layer];
                    let (rope_start, rope_step) = match cfg.rope_scaling {
                        crate::model::config::RopeScaling::Linear(f) => {
                            (base_seq_len as f32 / f, 1.0f32 / f)
                        }
                        crate::model::config::RopeScaling::None => (base_seq_len as f32, 1.0f32),
                    };
                    let mut rope_done = false;
                    let mut kv_appended = false;

                    let wq_buf = weight_cache.get(&lw.wq).unwrap();
                    let wk_buf = weight_cache.get(&lw.wk).unwrap();
                    let wv_buf = weight_cache.get(&lw.wv).unwrap();
                    let fused_qkv_m = q_dim + 2 * kv_dim;
                    let fused_qkv_dtype_ok = lw.wq_dtype == lw.wk_dtype
                        && lw.wq_dtype == lw.wv_dtype
                        && matches!(lw.wq_dtype, GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0);
                    let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
                    let fused_qkv_buf = if use_fused_qkv && fused_qkv_dtype_ok {
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
                        if use_f16_batch_io {
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
                        } else {
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
                                use_batch_simd,
                            );
                        }
                        if metal_prefill_split_rope_append_enabled() {
                            metal_ops.elementwise.encode_qkv_split_rope_append_kv_batch(
                                encoder,
                                &bs.qkv_buf,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                gpu_kv.k_buffer(layer),
                                gpu_kv.v_buffer(layer),
                                kv_f16,
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
                            rope_done = true;
                            kv_appended = true;
                        } else {
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
                            rope_done = true;
                        }
                    } else if use_f16_batch_io {
                        encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wq_buf,
                            &bs.matmul_in_f16,
                            &bs.q_buf,
                            q_dim as u32,
                            nt,
                            dim as u32,
                            lw.wq_dtype,
                        );
                        encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wk_buf,
                            &bs.matmul_in_f16,
                            &bs.k_buf,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            lw.wk_dtype,
                        );
                        encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wv_buf,
                            &bs.matmul_in_f16,
                            &bs.v_buf,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            lw.wv_dtype,
                        );
                    } else {
                        encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wq_buf,
                            &bs.norm_buf,
                            &bs.q_buf,
                            &bs.matmul_in_f16,
                            q_dim as u32,
                            nt,
                            dim as u32,
                            lw.wq_dtype,
                            false,
                            use_batch_simd,
                        );
                        encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wk_buf,
                            &bs.norm_buf,
                            &bs.k_buf,
                            &bs.matmul_in_f16,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            lw.wk_dtype,
                            false,
                            use_batch_simd,
                        );
                        encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wv_buf,
                            &bs.norm_buf,
                            &bs.v_buf,
                            &bs.matmul_in_f16,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            lw.wv_dtype,
                            false,
                            use_batch_simd,
                        );
                    }
                    ax_metal::barrier_buffers(encoder);

                    // ── Phase 1c: Batched RoPE + batched KV cache append ──
                    if !rope_done {
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
                        ax_metal::barrier_buffers(encoder);
                    }

                    if !kv_appended {
                        let cache_offset = (base_seq_len * kv_dim) as u32;
                        metal_ops.elementwise.encode_kv_append_batch(
                            encoder,
                            &bs.k_buf,
                            gpu_kv.k_buffer(layer),
                            kv_f16,
                            cache_offset,
                            kv_dim as u32,
                            kv_dim as u32,
                            nt,
                        );
                        metal_ops.elementwise.encode_kv_append_batch(
                            encoder,
                            &bs.v_buf,
                            gpu_kv.v_buffer(layer),
                            kv_f16,
                            cache_offset,
                            kv_dim as u32,
                            kv_dim as u32,
                            nt,
                        );
                        ax_metal::barrier_buffers(encoder);
                    }

                    // ── Phase 2: Batched attention ──
                    let use_attn_f16out = use_f16_batch_io
                        && base_seq_len == 0
                        && head_dim == 128
                        && metal_prefill_attn_f16out_enabled();
                    if base_seq_len == 0 && !metal_prefill_use_cached0_enabled() {
                        if use_attn_f16out {
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
                        } else {
                            metal_ops.attention.encode_attention_prefill(
                                encoder,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                &bs.attn_out,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                            );
                        }
                    } else {
                        metal_ops.attention.encode_attention_prefill_cached(
                            encoder,
                            &bs.q_buf,
                            gpu_kv.k_buffer(layer),
                            gpu_kv.v_buffer(layer),
                            &bs.attn_out,
                            kv_f16,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            base_seq_len as u32,
                            0,
                        );
                    }
                    ax_metal::barrier_buffers(encoder);

                    // ── Phase 3a: Batched output projection ──
                    let wo_buf = weight_cache.get(&lw.wo).unwrap();
                    if use_f16_batch_io {
                        if !use_attn_f16out {
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
                            use_batch_simd,
                        );
                    }
                    ax_metal::barrier_buffers(encoder);

                    // ── Phase 3b: Batched residual + FFN norm ──
                    let ffn_nw_buf = weight_cache.get(&lw.ffn_norm).unwrap();
                    if use_f16_batch_io {
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
                    ax_metal::barrier_buffers(encoder);

                    // ── Phase 3c: Batched gate + up ──
                    let wg_buf = weight_cache.get(&lw.wg).unwrap();
                    let wu_buf = weight_cache.get(&lw.wu).unwrap();

                    if use_f16_batch_io {
                        let use_pair_this_layer = (use_f16_pair || lw.wg_dtype == GgmlType::Q8_0)
                            && lw.wg_dtype == lw.wu_dtype;
                        if use_pair_this_layer {
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
                    } else {
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
                            use_batch_simd,
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
                            use_batch_simd,
                        );
                    }
                    ax_metal::barrier_buffers(encoder);

                    // ── Phase 3d: Batched activation ──
                    if use_f16_batch_io {
                        metal_ops.elementwise.encode_silu_elementwise_mul_batch_f16(
                            encoder,
                            &bs.gate_buf,
                            &bs.up_buf,
                            &bs.matmul_in_f16,
                            inter_dim as u32,
                            nt,
                        );
                    } else {
                        metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                            encoder,
                            &bs.gate_buf,
                            &bs.up_buf,
                            inter_dim as u32,
                            nt,
                        );
                    }
                    ax_metal::barrier_buffers(encoder);

                    // ── Phase 3e: Batched down projection ──
                    let wd_buf = weight_cache.get(&lw.wd).unwrap();
                    if use_f16_batch_io {
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
                    } else {
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
                            use_batch_simd,
                        );
                    }
                    ax_metal::barrier_buffers(encoder);

                    // ── Phase 3f: Batched residual (+ next layer's norm if not last) ──
                    if layer + 1 < n_layers {
                        // Fuse residual add with next layer's Phase 1a RMSNorm.
                        // hidden += proj_buf; norm_buf = RMSNorm(hidden, next_attn_norm)
                        let next_norm_w = weight_cache
                            .get(&cached.layers[layer + 1].attn_norm)
                            .unwrap();
                        if use_f16_batch_io {
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
                        } else {
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
                    } else {
                        // Last layer: standalone residual (no norm needed for logits path).
                        metal_ops.elementwise.encode_elementwise_add_batch(
                            encoder,
                            &bs.hidden,
                            &bs.proj_buf,
                            dim as u32,
                            nt,
                        );
                    }
                    ax_metal::barrier_buffers(encoder);
                }

                let fnw_buf = weight_cache.get(&cached.output_norm).unwrap();
                let lm_buf = weight_cache.get(&cached.lm_head).unwrap();
                if let Some(logits_buf) = all_logits_buf.as_ref() {
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        fnw_buf,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        eps,
                    );
                    ax_metal::barrier_buffers(encoder);
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
                        use_f16_batch_io,
                        use_batch_simd,
                    );
                } else {
                    // Final RMSNorm on last token's hidden
                    let last_off = (n_tokens - 1) * dim * 4;
                    metal_ops.elementwise.encode_buffer_copy(
                        encoder, &bs.hidden, last_off, &s.hidden, 0, dim as u32,
                    );
                    ax_metal::barrier_buffers(encoder);
                    metal_ops
                        .elementwise
                        .encode_rms_norm(encoder, &s.hidden, fnw_buf, dim as u32, eps);

                    // LM head (folded into same command buffer)
                    ax_metal::barrier_buffers(encoder);
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
        // v2 GPU gate: use_gpu_decode() + kv.as_gpu_mut() (no AX_CPU_ONLY check)
        let force_serial = std::env::var("AX_SERIAL_PREFILL").is_ok();
        let can_gpu_batch = !force_serial
            && ctx.backend.use_gpu_decode()
            && token_ids.len() > 1
            && kv.as_gpu_mut().is_some()
            && (gpu_quant_supported(weights) || gpu_decode_quant_supported(weights));

        if can_gpu_batch && let Some(metal_ops) = ctx.backend.metal_ops() {
            // Extract gpu_kv after the is_some() check above — guaranteed Some
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
        let force_serial = std::env::var("AX_SERIAL_PREFILL").is_ok();
        let lm_weight_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_head_dtype_supported = matches!(
            weights.raw_with_dtype(lm_weight_name),
            Ok((_, dtype)) if gpu_batch_logits_supported(dtype)
        );
        let can_gpu_batch = !force_serial
            && ctx.backend.use_gpu_decode()
            && token_ids.len() > 1
            && kv.as_gpu_mut().is_some()
            && lm_head_dtype_supported
            && (gpu_quant_supported(weights) || gpu_decode_quant_supported(weights));

        if can_gpu_batch && let Some(metal_ops) = ctx.backend.metal_ops() {
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

    /// Create a KV cache sized for this model.
    ///
    /// v2: returns `ModelKv`. Backend selects CPU or GPU variant automatically.
    /// f16 KV is used when context_length >= 256 (auto policy, same as v1).
    /// Paged KV deferred to v2.1.
    pub fn create_model_kv(&self) -> ModelKv {
        if self.config.architecture == "qwen35" {
            return ModelKv::Qwen35(crate::kv::Qwen35Kv::new(
                self.config.n_layers as usize,
                self.config.n_kv_heads as usize,
                self.config.head_dim as usize,
                self.config.context_length as usize,
                self.config.qwen35_full_attention_interval.unwrap_or(4) as usize,
                self.config.qwen35_ssm_conv_kernel.unwrap_or(4) as usize,
                self.config.qwen35_ssm_inner_size.unwrap_or(0) as usize,
                self.config.qwen35_ssm_state_size.unwrap_or(0) as usize,
                self.config.qwen35_ssm_time_step_rank.unwrap_or(0) as usize,
                self.config.qwen35_ssm_group_count.unwrap_or(0) as usize,
            ));
        }

        if self.backend.use_gpu_decode() {
            if let Some(metal_ops) = self.backend.metal_ops() {
                // GPU variant: use Metal buffers, f16 when context_length >= 256
                let cfg = &self.config;
                let kv_dtype = if crate::backend::metal::metal_f16_kv_cache_enabled(
                    cfg.context_length as usize,
                ) {
                    crate::kv::GpuKvDtype::F16
                } else {
                    crate::kv::GpuKvDtype::F32
                };
                match crate::kv::GpuKv::new_with_dtype(
                    &metal_ops.device,
                    cfg.n_layers as usize,
                    cfg.n_kv_heads as usize,
                    cfg.head_dim as usize,
                    cfg.context_length as usize,
                    256, // page_size
                    kv_dtype,
                ) {
                    Ok(gpu_kv) => {
                        tracing::info!(kv_dtype = ?kv_dtype, "Initialized GPU KV cache");
                        return ModelKv::Gpu(gpu_kv);
                    }
                    Err(e) => {
                        tracing::warn!("GPU KV allocation failed, falling back to CPU: {e}");
                    }
                }
            } else {
                tracing::warn!(
                    "Backend reported GPU decode support without Metal ops; falling back to CPU KV"
                );
            }
        }
        // CPU variant
        ModelKv::Cpu(crate::kv::CpuKv::new(
            self.config.n_layers as usize,
            self.config.n_kv_heads as usize,
            self.config.head_dim as usize,
            self.config.context_length as usize,
        ))
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
