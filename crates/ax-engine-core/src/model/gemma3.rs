//! Gemma3-family transformer forward pass.
//!
//! Key differences from LLaMA:
//!   1. Q/K projection sizes use n_heads * head_dim (head_dim may differ from embedding_dim / n_heads)
//!   2. Sliding window attention for most layers (every Nth layer is global)
//!   3. GELU activation in FFN instead of SiLU
//!   4. Weight tying: LM head reuses token_embd.weight when output.weight is absent
//!   5. Per-head RMSNorm on Q and K vectors after projection, before RoPE (qk_norm)
//!   6. Post-attention and post-FFN RMSNorm (before residual add)
//!   7. Different RoPE freq base for local (sliding window) vs global layers
//!   8. Embedding scaling by sqrt(embedding_dim)
//!
//! # v2 API changes
//! - `kv_cache: &mut KvCache` → `kv: &mut ModelKv`
//! - GPU decode path uses `kv.as_gpu_mut()` directly, no Mutex locking
//! - `forward_batch` GPU gate uses `ctx.backend.use_gpu_decode()` instead of AX_CPU_ONLY
//! - `gpu_kv.advance()` → `gpu_kv.finalize_token()`
//! - `gpu_kv.advance_by(n)` → `gpu_kv.finalize_batch(n)` (no CPU mirror sync)

use crate::backend::metal::MetalOps;
use crate::compute::attention;
use crate::compute::rms_norm;
use crate::compute::rope;
use crate::compute::silu;
use crate::gguf::tensor::GgmlType;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::execution_plan::{
    DecodeBarrierPlan, DecodeExecutionPlan, GpuBatchPrefillExecutionPlan, GpuDecodeExecutionPlan,
    PrefillAttentionPlan, PrefillExecutionPlan, PrefillFfnActivationPlan, PrefillLogitsPlan,
    PrefillMode, PrefillProjectionInputPlan, PrefillResidualHandoffPlan,
};
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    encode_batch_logits, encode_dequant_batch, encode_dequant_batch_f16in,
    encode_dequant_batch_pair_f16in, encode_dequant_matvec, encode_dequant_matvec_with_config,
    gpu_decode_quant_supported, gpu_prefill_q5k_small_n_auto_eligible, gpu_prefill_uses_q5k,
    per_head_rms_norm,
};
use crate::model::weights::WeightStore;

/// Timing helper.
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

/// Gemma3-family forward pass implementation.
#[derive(Debug)]
pub struct Gemma3Forward;

/// Gemma3 GPU decode strategy: per-head QK norm + per-layer RoPE base +
/// sliding window attention + optional post-attention norm.
struct Gemma3DecodeStrategy<'a> {
    cfg: &'a ModelConfig,
    kv_offset: u32,
    rope_position: f32,
    rope_base: f32,
    full_seq_len: usize,
    is_local: bool,
    dims: &'a crate::model::shared::GpuLayerDims,
}

impl crate::model::shared::GpuDecodeLayerStrategy for Gemma3DecodeStrategy<'_> {
    fn encode_qkv_post_attend_residual(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        metal_ops: &MetalOps,
        s: &crate::backend::metal::GpuScratchBuffers,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        lw: &crate::backend::metal::CachedLayerKeys,
        weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
        gpu_kv: &crate::kv::GpuKv,
        layer: usize,
        exec_plan: &GpuDecodeExecutionPlan,
        barrier_fn: &dyn Fn(&ax_engine_metal::MetalEncoder),
        used_fused_qkv: bool,
    ) {
        let d = self.dims;
        let eps = d.eps;

        if used_fused_qkv && let (Some(qn_key), Some(kn_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
            let qn = weight_cache.get(&qn_key).unwrap();
            let kn = weight_cache.get(&kn_key).unwrap();
            metal_ops
                .elementwise
                .encode_qkv_split_qk_norm_rope_append_kv_batch(
                    encoder,
                    &s.qkv_buf,
                    &s.q_buf,
                    &s.k_buf,
                    &s.v_buf,
                    qn,
                    kn,
                    gpu_kv.k_buffer(layer),
                    gpu_kv.v_buffer(layer),
                    exec_plan.kv_f16,
                    1,
                    d.n_heads,
                    d.n_kv_heads,
                    d.head_dim,
                    eps,
                    self.rope_position,
                    1.0,
                    self.rope_base,
                    self.kv_offset,
                    d.kv_dim,
                );
            barrier_fn(encoder);
        } else {
            // Separate: optional QK norm → RoPE → KV append
            if let (Some(qn_key), Some(kn_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
                let qn = weight_cache.get(&qn_key).unwrap();
                let kn = weight_cache.get(&kn_key).unwrap();
                metal_ops
                    .elementwise
                    .encode_per_head_rms_norm(encoder, &s.q_buf, qn, d.n_heads, d.head_dim, eps);
                metal_ops.elementwise.encode_per_head_rms_norm(
                    encoder,
                    &s.k_buf,
                    kn,
                    d.n_kv_heads,
                    d.head_dim,
                    eps,
                );
                barrier_fn(encoder);
            }
            metal_ops.elementwise.encode_rope(
                encoder,
                &s.q_buf,
                &s.k_buf,
                d.n_heads,
                d.n_kv_heads,
                d.head_dim,
                self.rope_position,
                self.rope_base,
            );
            barrier_fn(encoder);
            metal_ops.elementwise.encode_kv_append(
                encoder,
                &s.k_buf,
                gpu_kv.k_buffer(layer),
                exec_plan.kv_f16,
                self.kv_offset,
                d.kv_dim,
            );
            metal_ops.elementwise.encode_kv_append(
                encoder,
                &s.v_buf,
                gpu_kv.v_buffer(layer),
                exec_plan.kv_f16,
                self.kv_offset,
                d.kv_dim,
            );
            barrier_fn(encoder);
        }

        // Attention with sliding window support
        let (attend_start, attend_len) = if self.is_local {
            if let Some(window) = self.cfg.sliding_window_size {
                let window = window as usize;
                let alen = self.full_seq_len.min(window);
                let astart = self.full_seq_len.saturating_sub(alen);
                (astart as u32, alen as u32)
            } else {
                (0u32, self.full_seq_len as u32)
            }
        } else {
            (0u32, self.full_seq_len as u32)
        };
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
                d.n_heads,
                d.n_kv_heads,
                d.head_dim,
                attend_start,
                attend_len,
                exec_plan.attention_dispatch,
            );
        barrier_fn(encoder);

        // WO projection
        let wo_buf = weight_cache.get(&lw.wo).unwrap();
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wo_buf,
            &s.attn_out,
            &s.proj_buf,
            d.dim,
            d.q_dim,
            lw.wo_dtype,
            exec_plan.dequant_dispatch,
        );
        barrier_fn(encoder);

        // Residual + FFN norm (with optional post-attention norm for Gemma3)
        let ffn_nw = weight_cache.get(&lw.ffn_norm).unwrap();
        if let Some(post_attn_key) = lw.post_attn_norm {
            let post_nw = weight_cache.get(&post_attn_key).unwrap();
            metal_ops
                .elementwise
                .encode_post_attn_norm_residual_add_rms_norm_out_batch(
                    encoder,
                    hidden_buf,
                    &s.proj_buf,
                    post_nw,
                    ffn_nw,
                    &s.norm_buf,
                    d.dim,
                    1,
                    eps,
                );
        } else {
            metal_ops
                .elementwise
                .encode_residual_add_rms_norm_out_batch(
                    encoder,
                    hidden_buf,
                    &s.proj_buf,
                    ffn_nw,
                    &s.norm_buf,
                    d.dim,
                    1,
                    eps,
                );
        }
        barrier_fn(encoder);
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_gemma3_gpu_layers_only(
    encoder: &ax_engine_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_engine_metal::MetalBuffer,
    cfg: &ModelConfig,
    position: usize,
    kv_offset: u32,
    full_seq_len: usize,
    exec_plan: &GpuDecodeExecutionPlan,
    gpu_kv: &crate::kv::GpuKv,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
    fused_qkv_cache: &rustc_hash::FxHashMap<(usize, usize, usize), ax_engine_metal::MetalBuffer>,
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
    let _use_fused_decode_qkv = exec_plan.qkv == crate::model::execution_plan::DecodeQkvPlan::Fused;
    let decode_barrier = |encoder: &ax_engine_metal::MetalEncoder| {
        if exec_plan.barriers == DecodeBarrierPlan::Explicit {
            ax_engine_metal::barrier_buffers(encoder);
        }
    };

    // Layer 0 starts with a standalone attention norm. Later layers reuse the
    // next-layer handoff written by the previous FFN residual stage.
    {
        let norm_w_buf = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
        metal_ops.elementwise.encode_rms_norm_out(
            encoder,
            hidden_buf,
            norm_w_buf,
            &s.norm_buf,
            dim as u32,
            eps,
        );
        decode_barrier(encoder);
    }

    let gpu_dims = crate::model::shared::GpuLayerDims {
        dim: dim as u32,
        q_dim: q_dim as u32,
        kv_dim: kv_dim as u32,
        inter_dim: inter_dim as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        eps,
    };

    for layer in 0..n_layers {
        let lw = &cached.layers[layer];
        let is_local = Gemma3Forward::use_sliding_window(layer, cfg);
        let rope_base = if is_local {
            cfg.rope_freq_base_local.unwrap_or(cfg.rope_freq_base)
        } else {
            cfg.rope_freq_base
        };
        let rope_position = if is_local {
            position as f32
        } else {
            cfg.rope_scaling.scaled_position(position)
        };
        let strategy = Gemma3DecodeStrategy {
            cfg,
            kv_offset,
            rope_position,
            rope_base,
            full_seq_len,
            is_local,
            dims: &gpu_dims,
        };
        let next_attn_norm = if layer + 1 < n_layers {
            Some(cached.layers[layer + 1].attn_norm)
        } else {
            None
        };
        let post_ffn_nw = lw.post_ffn_norm.and_then(|k| weight_cache.get(&k));
        crate::model::shared::encode_gpu_decode_layer(
            encoder,
            metal_ops,
            s,
            hidden_buf,
            lw,
            weight_cache,
            fused_qkv_cache,
            gpu_kv,
            layer,
            n_layers,
            exec_plan,
            next_attn_norm,
            &gpu_dims,
            &strategy,
            crate::model::layer_ops::FfnActivation::GELU,
            post_ffn_nw,
            &decode_barrier,
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_gemma3_gpu_output_head(
    encoder: &ax_engine_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_engine_metal::MetalBuffer,
    cfg: &ModelConfig,
    exec_plan: &GpuDecodeExecutionPlan,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
) {
    crate::model::shared::encode_gpu_output_head(
        encoder,
        metal_ops,
        s,
        hidden_buf,
        exec_plan,
        cached,
        weight_cache,
        cfg.embedding_dim,
        cfg.vocab_size,
        cfg.rms_norm_eps,
    );
}

fn encode_gemma3_pending_step(
    metal_ops: &MetalOps,
    cfg: &ModelConfig,
    hidden_buf: &ax_engine_metal::MetalBuffer,
    position: usize,
    gpu_kv: &mut crate::kv::GpuKv,
    weights: &WeightStore,
) -> anyhow::Result<ax_engine_metal::PendingFrame> {
    let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as usize;

    gpu_kv.ensure_capacity(&metal_ops.device, position + 1)?;
    metal_ops.init_scratches(cfg);

    let scratch_guard = metal_ops.scratches();
    let s = scratch_guard.as_ref().unwrap();

    if !metal_ops.has_cached_model_keys() {
        Gemma3Forward::build_cached_model_keys_gemma3(metal_ops, weights, cfg)?;
    }

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let weight_cache = metal_ops.lock_weight_cache();
    let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

    let kv_offset = (position * kv_dim) as u32;
    let full_seq_len = position + 1;
    let exec_plan = DecodeExecutionPlan::gemma3_pipelined(
        metal_ops,
        gpu_kv,
        cfg.embedding_dim,
        cfg.head_dim,
        full_seq_len,
    );

    metal_ops.device.encode_frame(|encoder| {
        encode_gemma3_gpu_layers_only(
            encoder,
            metal_ops,
            s,
            hidden_buf,
            cfg,
            position,
            kv_offset,
            full_seq_len,
            &exec_plan,
            gpu_kv,
            cached,
            &weight_cache,
            &fused_qkv_cache,
        )?;
        encode_gemma3_gpu_output_head(
            encoder,
            metal_ops,
            s,
            hidden_buf,
            cfg,
            &exec_plan,
            cached,
            &weight_cache,
        );
        Ok(())
    })
}

impl Gemma3Forward {
    /// Check if a layer uses sliding window (local) attention.
    ///
    /// Gemma3: most layers use sliding window. Every Nth layer (where N = pattern)
    /// uses global (full) attention. Specifically, the last layer in each group
    /// of N layers is global: layer % pattern == pattern - 1.
    ///
    /// For pattern=6: layers 0-4 = local, layer 5 = global, layers 6-10 = local, etc.
    pub(crate) fn use_sliding_window(layer: usize, config: &ModelConfig) -> bool {
        match (config.sliding_window_size, config.sliding_window_pattern) {
            (Some(_size), Some(pattern)) if pattern > 0 => {
                // Local (sliding window) for all layers except every Nth
                layer % (pattern as usize) != (pattern as usize - 1)
            }
            _ => false,
        }
    }

    pub(crate) fn gpu_prefill_chunk_len(config: &ModelConfig, n_tokens: usize) -> Option<usize> {
        match config.sliding_window_size {
            Some(window) if n_tokens > window as usize => Some(window as usize),
            _ => None,
        }
    }

    /// Build and store pre-computed weight cache keys for all layers (Gemma3 architecture).
    /// Called once on first forward pass; subsequent calls are skipped via `has_cached_model_keys()`.
    fn build_cached_model_keys_gemma3(
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
        let use_fused_decode_qkv = metal_ops.metal_decode_fused_qkv_enabled();

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
            if use_fused_decode_qkv
                && wq_dtype == wk_dtype
                && wq_dtype == wv_dtype
                && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K)
            {
                metal_ops.ensure_qkv_fused_quant_cached(wq_raw, wk_raw, wv_raw);
            }
            if use_precomputed_f16
                && metal_ops.metal_fused_qkv_enabled()
                && wq_dtype == wk_dtype
                && wq_dtype == wv_dtype
                && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K)
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

            // QK norm weights (Gemma3-specific)
            let attn_q_norm_key = if weights.has(&format!("{prefix}.attn_q_norm.weight")) {
                let qw = weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?;
                let q_norm_key = metal_ops.ensure_f32_cached(qw);
                Some(q_norm_key)
            } else {
                None
            };
            let attn_k_norm_key = if weights.has(&format!("{prefix}.attn_k_norm.weight")) {
                let kw = weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?;
                let k_norm_key = metal_ops.ensure_f32_cached(kw);
                Some(k_norm_key)
            } else {
                None
            };

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
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);

            // Post-norms (Gemma3-specific)
            let post_attn_norm_key = if weights.has(&format!("{prefix}.post_attention_norm.weight"))
            {
                let w = weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
                Some(metal_ops.ensure_f32_cached(w))
            } else {
                None
            };
            let post_ffn_norm_key = if weights.has(&format!("{prefix}.post_ffw_norm.weight")) {
                let w = weights.f32_slice(&format!("{prefix}.post_ffw_norm.weight"))?;
                Some(metal_ops.ensure_f32_cached(w))
            } else {
                None
            };

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
                attn_q_norm: attn_q_norm_key,
                attn_k_norm: attn_k_norm_key,
                post_attn_norm: post_attn_norm_key,
                post_ffn_norm: post_ffn_norm_key,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                wo_bias: None,
                gate_bias: None,
                up_bias: None,
                down_bias: None,
                moe_router: None,
                moe_router_dtype: None,
                moe_expert_gate: None,
                moe_expert_up: None,
                moe_expert_down: None,
                moe_expert_dtype: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_dtype: None,
            });
        }
        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        let output_norm_key = metal_ops.ensure_f32_cached(final_norm_w);

        // Weight tying: use output.weight if it exists, else token_embd.weight
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
    /// Eliminates the CPU attention sync point by using GPU decode attention
    /// and GPU KV cache. Reduces from 69 command buffers to 2 per token.
    ///
    /// v2: receives `gpu_kv: &mut GpuKv` directly. Ends with `gpu_kv.finalize_token()`.
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
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let kv_dim = n_kv_heads * head_dim;

        assert!(logits.len() >= vocab_size);

        // Initialize GPU scratch buffers on first call
        metal_ops.init_scratches(cfg);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();

        // Ensure GPU KV cache has capacity for this token
        let next_seq = gpu_kv.seq_len() + 1;
        gpu_kv.ensure_capacity(&metal_ops.device, next_seq)?;

        let setup_t = OpTimer::start();
        // Token embedding (CPU dequant → GPU scratch)
        {
            let hidden_cpu = unsafe {
                std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim)
            };
            weights.dequantize_row("token_embd.weight", token_id as usize, hidden_cpu)?;
            if cfg.embed_scale {
                let embd_scale = (dim as f32).sqrt();
                for h in hidden_cpu.iter_mut() {
                    *h *= embd_scale;
                }
            }
        }

        // Pre-cache ALL weights and build cached keys (first call only)
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_gemma3(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();

        // Current sequence position (before appending this token)
        let cur_seq_len = gpu_kv.seq_len();
        let kv_offset = (cur_seq_len * kv_dim) as u32;
        let exec_plan = DecodeExecutionPlan::gemma3_single_cb(
            metal_ops,
            gpu_kv,
            cfg.embedding_dim,
            cfg.head_dim,
            cur_seq_len + 1,
        );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        // ── Single command buffer: all layers + final norm + LM head ──
        {
            let weight_cache = metal_ops.lock_weight_cache();
            let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();
            let exec_t = OpTimer::start();
            metal_ops.device.execute_sync(|encoder| {
                encode_gemma3_gpu_layers_only(
                    encoder,
                    metal_ops,
                    s,
                    &s.hidden,
                    cfg,
                    position,
                    kv_offset,
                    cur_seq_len + 1,
                    &exec_plan,
                    gpu_kv,
                    cached,
                    &weight_cache,
                    &fused_qkv_cache,
                )?;
                encode_gemma3_gpu_output_head(
                    encoder,
                    metal_ops,
                    s,
                    &s.hidden,
                    cfg,
                    &exec_plan,
                    cached,
                    &weight_cache,
                );
                Ok(())
            })?;
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_execute += exec_t.elapsed();
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

        // Logit scaling
        if let Some(scale) = cfg.logit_scale {
            for l in logits[..vocab_size].iter_mut() {
                *l *= scale;
            }
        }

        Ok(())
    }

    /// Batched GPU prefill for Gemma3.
    ///
    /// Uses per-token loops for QKV/RoPE/QK-norm and output-proj/FFN phases
    /// (required for correctness with Gemma3-specific per-head norms and
    /// post-attention/post-FFN norms), with batched attention in between.
    /// Weight cache keys are pre-computed to avoid format!/HashMap overhead.
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
        mut ops: Option<&mut OpBreakdown>,
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

        // Gemma3 sliding window guard: batch prefill only works when n_tokens <= window_size
        if let Some(window) = cfg.sliding_window_size {
            anyhow::ensure!(
                n_tokens <= window as usize,
                "Gemma3 batch prefill: n_tokens ({}) > sliding_window_size ({}); use serial",
                n_tokens,
                window
            );
        }

        let setup_t = OpTimer::start();

        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, n_tokens);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let batch_guard = metal_ops.batch_scratches();
        let bs = batch_guard.as_ref().unwrap();

        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + n_tokens)?;

        // Embed all N tokens (with scaling)
        {
            let batch_hidden_cpu = unsafe {
                std::slice::from_raw_parts_mut(
                    bs.hidden.contents().as_ptr() as *mut f32,
                    n_tokens * dim,
                )
            };
            for (i, &tid) in token_ids.iter().enumerate() {
                let slice = &mut batch_hidden_cpu[i * dim..(i + 1) * dim];
                weights.dequantize_row("token_embd.weight", tid as usize, slice)?;
            }
            if cfg.embed_scale {
                let scale = (dim as f32).sqrt();
                for h in batch_hidden_cpu.iter_mut() {
                    *h *= scale;
                }
            }
        }

        // Pre-cache weights and build cached keys (first call only)
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_gemma3(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();

        let base_seq_len = gpu_kv.seq_len();
        let all_logits_buf = if emit_all_logits {
            Some(ax_engine_metal::MetalBuffer::new(
                metal_ops.device.device(),
                n_tokens * vocab_size * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        };

        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        // Single command buffer: all layers + final norm + LM head
        {
            let weight_cache = metal_ops.lock_weight_cache();
            let has_q5k_weights = gpu_prefill_uses_q5k(weights);
            let q5k_small_n_auto_eligible = gpu_prefill_q5k_small_n_auto_eligible(weights);
            let prefill_plan: GpuBatchPrefillExecutionPlan = DecodeExecutionPlan::gemma3_prefill(
                metal_ops,
                gpu_kv,
                n_tokens as u32,
                has_q5k_weights,
                q5k_small_n_auto_eligible,
            );
            let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

            let exec_t = OpTimer::start();
            metal_ops.device.execute_sync_concurrent(|encoder| {
                let nt = n_tokens as u32;
                let mut sb = ax_engine_metal::SmartBarrier::new(encoder);

                // First layer's Phase 1a: standalone RMSNorm before loop.
                {
                    let t = OpTimer::start();
                    let norm_w_buf = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
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
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_norm += t.elapsed();
                    }
                }

                for layer in 0..n_layers {
                    let lw = &cached.layers[layer];
                    let layer_plan = DecodeExecutionPlan::gemma3_prefill_layer(
                        cfg,
                        layer,
                        base_seq_len,
                        prefill_plan.use_f16_batch_io,
                    );

                    let wq_buf = weight_cache.get(&lw.wq).unwrap();
                    let wk_buf = weight_cache.get(&lw.wk).unwrap();
                    let wv_buf = weight_cache.get(&lw.wv).unwrap();

                    let fused_qkv_m = q_dim + 2 * kv_dim;
                    let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
                    let qkv_layer_plan = DecodeExecutionPlan::gemma3_prefill_qkv_layer(
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

                    // Phase 1a: norm_buf already populated (first layer before loop,
                    // subsequent layers fused with previous Phase 3h).

                    // ── Phase 1b: Batched QKV matmul ──
                    let qkv_t = OpTimer::start();
                    if let Some(fused_w) = fused_qkv_buf {
                        let qkv_input = &bs.norm_buf;
                        sb.pre_dispatch(&[qkv_input], &[&bs.qkv_buf]);
                        match qkv_layer_plan.input {
                            PrefillProjectionInputPlan::MatmulScratchF16 => {
                                metal_ops.elementwise.encode_cast_f32_to_f16(
                                    encoder,
                                    &bs.norm_buf,
                                    &bs.matmul_in_f16,
                                    nt * dim as u32,
                                );
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
                                    prefill_plan.q5k_prefill_small_n,
                                );
                            }
                        }
                        sb.post_dispatch(&[qkv_input], &[&bs.qkv_buf]);
                        sb.pre_dispatch(&[&bs.qkv_buf], &[&bs.q_buf, &bs.k_buf, &bs.v_buf]);
                        metal_ops.elementwise.encode_qkv_split_batch(
                            encoder,
                            &bs.qkv_buf,
                            &bs.q_buf,
                            &bs.k_buf,
                            &bs.v_buf,
                            nt,
                            q_dim as u32,
                            kv_dim as u32,
                        );
                        sb.post_dispatch(&[&bs.qkv_buf], &[&bs.q_buf, &bs.k_buf, &bs.v_buf]);
                    } else {
                        match qkv_layer_plan.input {
                            PrefillProjectionInputPlan::MatmulScratchF16 => {
                                // f16 path: all three share matmul_in_f16, can't overlap.
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.matmul_in_f16]);
                                metal_ops.elementwise.encode_cast_f32_to_f16(
                                    encoder,
                                    &bs.norm_buf,
                                    &bs.matmul_in_f16,
                                    nt * dim as u32,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.matmul_in_f16]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.q_buf]);
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
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.q_buf]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.k_buf]);
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
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.k_buf]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.v_buf]);
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
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.v_buf]);
                            }
                            PrefillProjectionInputPlan::NormBufF32 => {
                                // f32 path: Q/K/V all read norm_buf, write
                                // different buffers -> SmartBarrier skips
                                // barriers between them (GPU can overlap).
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.q_buf]);
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
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.q_buf]);
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.k_buf]);
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
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.k_buf]);
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.v_buf]);
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
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.v_buf]);
                            }
                        }
                    }
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_qkv += qkv_t.elapsed();
                    }

                    let cache_offset = (base_seq_len * kv_dim) as u32;
                    let kv_k = gpu_kv.k_buffer(layer);
                    let kv_v = gpu_kv.v_buffer(layer);
                    let fused_qkv_post = fused_qkv_buf.is_some()
                        && lw.attn_q_norm.is_some()
                        && lw.attn_k_norm.is_some();

                    // ── Phase 1c+1d: Fused split + QK norm + RoPE + KV append when eligible ──
                    let rope_kv_t = OpTimer::start();
                    if fused_qkv_post {
                        let q_nw = weight_cache.get(&lw.attn_q_norm.unwrap()).unwrap();
                        let k_nw = weight_cache.get(&lw.attn_k_norm.unwrap()).unwrap();
                        sb.pre_dispatch(
                            &[&bs.qkv_buf],
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v],
                        );
                        metal_ops
                            .elementwise
                            .encode_qkv_split_qk_norm_rope_append_kv_batch(
                                encoder,
                                &bs.qkv_buf,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                q_nw,
                                k_nw,
                                kv_k,
                                kv_v,
                                prefill_plan.kv_f16,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                eps,
                                layer_plan.rope_start,
                                layer_plan.rope_step,
                                layer_plan.rope_base,
                                cache_offset,
                                kv_dim as u32,
                            );
                        sb.post_dispatch(
                            &[&bs.qkv_buf],
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v],
                        );
                    } else {
                        sb.pre_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);
                        if let (Some(q_norm_key), Some(k_norm_key)) =
                            (lw.attn_q_norm, lw.attn_k_norm)
                        {
                            let q_nw = weight_cache.get(&q_norm_key).unwrap();
                            let k_nw = weight_cache.get(&k_norm_key).unwrap();
                            metal_ops.elementwise.encode_qk_norm_rope_batch(
                                encoder,
                                &bs.q_buf,
                                &bs.k_buf,
                                q_nw,
                                k_nw,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                eps,
                                layer_plan.rope_start,
                                layer_plan.rope_step,
                                layer_plan.rope_base,
                            );
                        } else {
                            metal_ops.elementwise.encode_rope_batch(
                                encoder,
                                &bs.q_buf,
                                &bs.k_buf,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                layer_plan.rope_start,
                                layer_plan.rope_step,
                                layer_plan.rope_base,
                            );
                        }
                        sb.post_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);

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
                    if let Some(ref mut ops_ref) = ops {
                        let elapsed = rope_kv_t.elapsed();
                        ops_ref.gpu_encode_layer_rope += elapsed / 2;
                        ops_ref.gpu_encode_layer_kv_append += elapsed / 2;
                    }

                    // ── Phase 2: Batched attention ──
                    let attn_t = OpTimer::start();
                    if layer_plan.attention == PrefillAttentionPlan::BatchLocal {
                        sb.pre_dispatch(&[&bs.q_buf, &bs.k_buf, &bs.v_buf], &[&bs.attn_out]);
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
                        sb.post_dispatch(&[&bs.q_buf, &bs.k_buf, &bs.v_buf], &[&bs.attn_out]);
                    } else {
                        let attn_kv_k = gpu_kv.k_buffer(layer);
                        let attn_kv_v = gpu_kv.v_buffer(layer);
                        sb.pre_dispatch(&[&bs.q_buf, attn_kv_k, attn_kv_v], &[&bs.attn_out]);
                        metal_ops
                            .attention
                            .encode_attention_prefill_cached_with_config(
                                encoder,
                                &bs.q_buf,
                                attn_kv_k,
                                attn_kv_v,
                                &bs.attn_out,
                                prefill_plan.kv_f16,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                base_seq_len as u32,
                                layer_plan.sliding_window,
                                prefill_plan.attention_dispatch,
                            );
                        sb.post_dispatch(&[&bs.q_buf, attn_kv_k, attn_kv_v], &[&bs.attn_out]);
                    }
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_attention += attn_t.elapsed();
                    }

                    // ── Phase 3a: Batched output projection ──
                    let out_proj_t = OpTimer::start();
                    let wo_buf = weight_cache.get(&lw.wo).unwrap();
                    sb.pre_dispatch(&[&bs.attn_out], &[&bs.proj_buf]);
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
                        prefill_plan.use_f16_batch_io,
                        prefill_plan.use_batch_simd,
                        prefill_plan.q5k_prefill_small_n,
                    );
                    sb.post_dispatch(&[&bs.attn_out], &[&bs.proj_buf]);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_out_proj += out_proj_t.elapsed();
                    }

                    let ffn_nw_buf = weight_cache.get(&lw.ffn_norm).unwrap();
                    let post_attn_and_residual_t = OpTimer::start();
                    if let Some(post_attn_key) = lw.post_attn_norm {
                        let post_attn_nw_buf = weight_cache.get(&post_attn_key).unwrap();
                        sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                        metal_ops
                            .elementwise
                            .encode_post_attn_norm_residual_add_rms_norm_out_batch(
                                encoder,
                                &bs.hidden,
                                &bs.proj_buf,
                                post_attn_nw_buf,
                                ffn_nw_buf,
                                &bs.norm_buf,
                                dim as u32,
                                nt,
                                eps,
                            );
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                    } else {
                        sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
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
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                    }
                    if let Some(ref mut ops_ref) = ops {
                        let elapsed = post_attn_and_residual_t.elapsed();
                        ops_ref.gpu_encode_layer_residual += elapsed / 2;
                        ops_ref.gpu_encode_layer_norm += elapsed / 2;
                    }

                    // ── Phase 3d: Batched gate + up ──
                    let gate_up_t = OpTimer::start();
                    let wg_buf = weight_cache.get(&lw.wg).unwrap();
                    let wu_buf = weight_cache.get(&lw.wu).unwrap();
                    let ffn_layer_plan = DecodeExecutionPlan::gemma3_prefill_ffn_layer(
                        &prefill_plan,
                        lw.wg_dtype,
                        lw.wu_dtype,
                    );

                    sb.pre_dispatch(&[&bs.norm_buf], &[&bs.gate_buf, &bs.up_buf]);
                    match ffn_layer_plan.input {
                        PrefillProjectionInputPlan::MatmulScratchF16 => {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.norm_buf,
                                &bs.matmul_in_f16,
                                nt * dim as u32,
                            );
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
                                prefill_plan.q5k_prefill_small_n,
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
                                prefill_plan.q5k_prefill_small_n,
                            );
                        }
                    }
                    sb.post_dispatch(&[&bs.norm_buf], &[&bs.gate_buf, &bs.up_buf]);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_ffn += gate_up_t.elapsed();
                    }

                    // ── Phase 3e: Batched GELU activation ──
                    let activation_t = OpTimer::start();
                    sb.pre_dispatch(&[&bs.gate_buf, &bs.up_buf], &[&bs.gate_buf]);
                    match ffn_layer_plan.activation {
                        PrefillFfnActivationPlan::GeluMulGateF32 => {
                            metal_ops.elementwise.encode_gelu_elementwise_mul_batch(
                                encoder,
                                &bs.gate_buf,
                                &bs.up_buf,
                                inter_dim as u32,
                                nt,
                            );
                        }
                        PrefillFfnActivationPlan::SiluMulGateF32
                        | PrefillFfnActivationPlan::SiluMulScratchF16 => unreachable!(),
                    }
                    sb.post_dispatch(&[&bs.gate_buf, &bs.up_buf], &[&bs.gate_buf]);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_ffn += activation_t.elapsed();
                    }

                    // ── Phase 3f: Batched down projection ──
                    let down_proj_t = OpTimer::start();
                    let wd_buf = weight_cache.get(&lw.wd).unwrap();
                    sb.pre_dispatch(&[&bs.gate_buf], &[&bs.proj_buf]);
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
                        prefill_plan.use_f16_batch_io,
                        prefill_plan.use_batch_simd,
                        prefill_plan.q5k_prefill_small_n,
                    );
                    sb.post_dispatch(&[&bs.gate_buf], &[&bs.proj_buf]);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_ffn += down_proj_t.elapsed();
                    }

                    let residual_plan =
                        DecodeExecutionPlan::gemma3_prefill_residual_handoff(layer + 1 == n_layers);
                    let fused_post_ffn_handoff = lw.post_ffn_norm.is_some()
                        && matches!(
                            residual_plan,
                            PrefillResidualHandoffPlan::ResidualAddRmsNormF32
                        );

                    // ── Phase 3g: Post-FFN RMSNorm (Gemma3-specific) ──
                    let post_ffn_norm_t = OpTimer::start();
                    if !fused_post_ffn_handoff {
                        if let Some(post_ffn_key) = lw.post_ffn_norm {
                            let nw = weight_cache.get(&post_ffn_key).unwrap();
                            sb.pre_dispatch(&[&bs.proj_buf], &[&bs.proj_buf]);
                            metal_ops.elementwise.encode_rms_norm_batch(
                                encoder,
                                &bs.proj_buf,
                                nw,
                                dim as u32,
                                nt,
                                eps,
                            );
                            sb.post_dispatch(&[&bs.proj_buf], &[&bs.proj_buf]);
                        }
                        if let Some(ref mut ops_ref) = ops {
                            ops_ref.gpu_encode_layer_norm += post_ffn_norm_t.elapsed();
                        }
                    }

                    // ── Phase 3h: Batched residual (+ next layer's norm if not last) ──
                    let residual_handoff_t = OpTimer::start();
                    let residual_norm_out = match residual_plan {
                        PrefillResidualHandoffPlan::ResidualOnly => None,
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF32 => Some(&bs.norm_buf),
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => unreachable!(),
                    };
                    if let Some(nout) = residual_norm_out {
                        sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, nout]);
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
                            if let Some(post_ffn_key) = lw.post_ffn_norm {
                                let post_ffn_nw = weight_cache.get(&post_ffn_key).unwrap();
                                metal_ops
                                    .elementwise
                                    .encode_post_ffn_norm_residual_add_rms_norm_out_batch(
                                        encoder,
                                        &bs.hidden,
                                        &bs.proj_buf,
                                        post_ffn_nw,
                                        next_norm_w,
                                        &bs.norm_buf,
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
                        }
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => unreachable!(),
                    }
                    if let Some(nout) = residual_norm_out {
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, nout]);
                    } else {
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    }
                    if let Some(ref mut ops_ref) = ops {
                        let elapsed = residual_handoff_t.elapsed();
                        match residual_plan {
                            PrefillResidualHandoffPlan::ResidualOnly => {
                                ops_ref.gpu_encode_layer_residual += elapsed;
                            }
                            PrefillResidualHandoffPlan::ResidualAddRmsNormF32 => {
                                ops_ref.gpu_encode_layer_residual += elapsed / 2;
                                ops_ref.gpu_encode_layer_norm += elapsed / 2;
                            }
                            PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => unreachable!(),
                        }
                    }
                }

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
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_execute += exec_t.elapsed();
            }
        }

        // v2: advance GPU KV only — no CPU mirror to sync
        gpu_kv.finalize_batch(n_tokens);

        let rb_t = OpTimer::start();
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
            if let Some(scale) = cfg.logit_scale {
                for l in logits_all.iter_mut() {
                    *l *= scale;
                }
            }
        } else if let Some(logits) = last_logits {
            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    s.logits_buf.contents().as_ptr() as *const f32,
                    vocab_size,
                )
            };
            logits[..vocab_size].copy_from_slice(logits_gpu);
            if let Some(scale) = cfg.logit_scale {
                for l in logits[..vocab_size].iter_mut() {
                    *l *= scale;
                }
            }
        }
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }

        Ok(())
    }
}

impl ForwardPass for Gemma3Forward {
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
            if let Some(ops_ref) = ops {
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

        // Gemma3: Q/K/V projection output sizes based on head_dim (not necessarily dim)
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        assert!(logits.len() >= vocab_size);

        let cpu_kv = kv
            .as_cpu_mut()
            .expect("Gemma3Forward CPU path requires ModelKv::Cpu");

        // --- Step 1: Token embedding (single-row dequant) ---
        let mut hidden = vec![0.0f32; dim];
        timed!(
            ops,
            dequant,
            weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?
        );

        // Gemma: scale embeddings by sqrt(embedding_dim)
        if cfg.embed_scale {
            let embd_scale = (dim as f32).sqrt();
            for h in hidden.iter_mut() {
                *h *= embd_scale;
            }
        }

        // Scratch buffers
        let mut norm_buf = vec![0.0f32; dim];
        let mut q_buf = vec![0.0f32; q_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; dim];

        // --- Step 2: Transformer layers ---
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            let is_local = Self::use_sliding_window(layer, cfg);

            // 2a. Pre-attention RMSNorm
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

            // 2b. Q/K/V projections (Gemma3: output dims based on head_dim)
            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;

            timed!(ops, matmul, {
                ctx.backend.batch_dequant_matvec(
                    &[
                        (wq_raw, wq_dtype, q_dim),
                        (wk_raw, wk_dtype, kv_dim),
                        (wv_raw, wv_dtype, kv_dim),
                    ],
                    &norm_buf,
                    dim,
                    &mut [&mut q_buf, &mut k_buf, &mut v_buf],
                );
            });

            // 2c. Per-head QK normalization (Gemma3-specific)
            // Apply RMSNorm independently to each head's Q and K vectors
            if weights.has(&format!("{prefix}.attn_q_norm.weight")) {
                let q_norm_w = timed!(
                    ops,
                    dequant,
                    weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?
                );
                let k_norm_w = timed!(
                    ops,
                    dequant,
                    weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?
                );
                timed!(ops, norm, {
                    per_head_rms_norm(&mut q_buf, n_heads, head_dim, q_norm_w, cfg.rms_norm_eps);
                    per_head_rms_norm(&mut k_buf, n_kv_heads, head_dim, k_norm_w, cfg.rms_norm_eps);
                });
            }

            // 2d. RoPE on Q and K
            // Gemma3: local layers use a different (lower) RoPE freq base than global layers
            let rope_base = if is_local {
                cfg.rope_freq_base_local.unwrap_or(cfg.rope_freq_base)
            } else {
                cfg.rope_freq_base
            };
            // Gemma3 per-layer RoPE: global layers use linear scaling,
            // SWA (local) layers use raw position (no scaling).
            let rope_position = if is_local {
                position as f32
            } else {
                cfg.rope_scaling.scaled_position(position)
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
                    rope_base,
                )
            );

            // 2e. Update KV cache (v2 CPU path)
            cpu_kv.append_and_advance(layer, &k_buf, &v_buf);

            // 2f. Multi-head attention (with sliding window for local layers)
            let full_seq_len = cpu_kv.seq_len() + 1;
            let seq_len = if is_local {
                if let Some(window) = cfg.sliding_window_size {
                    full_seq_len.min(window as usize)
                } else {
                    full_seq_len
                }
            } else {
                full_seq_len
            };

            // For sliding window, we need the most recent `seq_len` tokens
            let k_start = full_seq_len.saturating_sub(seq_len);
            let k_slice = &cpu_kv.k_slice_including_current(layer, full_seq_len)
                [k_start * kv_dim..full_seq_len * kv_dim];
            let v_slice = &cpu_kv.v_slice_including_current(layer, full_seq_len)
                [k_start * kv_dim..full_seq_len * kv_dim];

            timed!(
                ops,
                attention,
                attention::multi_head_attention(
                    &q_buf,
                    k_slice,
                    v_slice,
                    &mut attn_out,
                    ctx.attn_params,
                    seq_len,
                )
            );

            // 2g. Output projection
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
                    q_dim,
                )
            );

            // 2h. Post-attention RMSNorm (Gemma3-specific: applied before residual add)
            if weights.has(&format!("{prefix}.post_attention_norm.weight")) {
                let post_attn_norm_w = timed!(
                    ops,
                    dequant,
                    weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?
                );
                timed!(
                    ops,
                    norm,
                    rms_norm::rms_norm(&mut proj_buf, post_attn_norm_w, cfg.rms_norm_eps)
                );
            }

            // 2i. Residual add
            silu::elementwise_add(&mut hidden, &proj_buf);

            // 2j-2o. FFN: norm → gate/up → GELU → down → [post-FFN norm] → residual
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
            crate::model::layer_ops::apply_ffn_single(
                ctx.backend,
                weights,
                &prefix,
                &mut hidden,
                &mut norm_buf,
                &mut gate_buf,
                &mut up_buf,
                &mut down_buf,
                dim,
                inter_dim,
                ffn_norm_w,
                cfg.rms_norm_eps,
                crate::model::layer_ops::FfnActivation::GELU,
            );
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

        // --- Step 4: LM head ---
        // Weight tying: use token_embd.weight if output.weight doesn't exist.
        // Many Gemma GGUF files omit output.weight entirely (implicit tying)
        // even when the tie_word_embeddings flag is absent or false.
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

        // Logit scaling (if configured)
        if let Some(scale) = cfg.logit_scale {
            for l in logits[..vocab_size].iter_mut() {
                *l *= scale;
            }
        }

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

        if matches!(
            prefill_plan.mode,
            PrefillMode::GpuBatch | PrefillMode::GpuChunked
        ) {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
            if prefill_plan.mode == PrefillMode::GpuChunked {
                let chunk_len = prefill_plan.chunk_len.unwrap();
                for chunk in token_ids.chunks(chunk_len) {
                    match self.forward_batch_gpu_unified(
                        ctx,
                        metal_ops,
                        chunk,
                        gpu_kv,
                        weights,
                        Some(logits),
                        None,
                        None,
                    ) {
                        Ok(()) => {}
                        Err(e) => {
                            tracing::warn!(
                                "Gemma3 chunked GPU batch prefill failed, falling back to serial: {e}"
                            );
                            let start_pos = kv.seq_len();
                            for (i, &tid) in token_ids.iter().enumerate() {
                                logits.fill(0.0);
                                self.forward_single(
                                    ctx,
                                    tid,
                                    start_pos + i,
                                    kv,
                                    weights,
                                    logits,
                                    None,
                                )?;
                            }
                            return Ok(());
                        }
                    }
                }
                return Ok(());
            }
            match self.forward_batch_gpu_unified(
                ctx,
                metal_ops,
                token_ids,
                gpu_kv,
                weights,
                Some(logits),
                None,
                None,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!("GPU batch prefill failed, falling back to serial: {e}");
                }
            }
        }
        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, None)?;
        }
        Ok(())
    }

    fn forward_batch_profiled(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        let prefill_plan =
            PrefillExecutionPlan::for_forward_batch(ctx, kv, weights, token_ids.len(), false)?;

        if matches!(
            prefill_plan.mode,
            PrefillMode::GpuBatch | PrefillMode::GpuChunked
        ) {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
            if prefill_plan.mode == PrefillMode::GpuChunked {
                let total_t = OpTimer::start();
                let chunk_len = prefill_plan.chunk_len.unwrap();
                for chunk in token_ids.chunks(chunk_len) {
                    self.forward_batch_gpu_unified(
                        ctx,
                        metal_ops,
                        chunk,
                        gpu_kv,
                        weights,
                        Some(logits),
                        None,
                        Some(&mut *ops),
                    )?;
                }
                ops.gpu += total_t.elapsed();
                return Ok(());
            }

            let total_t = OpTimer::start();
            let result = self.forward_batch_gpu_unified(
                ctx,
                metal_ops,
                token_ids,
                gpu_kv,
                weights,
                Some(logits),
                None,
                Some(&mut *ops),
            );
            ops.gpu += total_t.elapsed();
            return result;
        }

        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, Some(ops))?;
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
                None,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!(
                        "Gemma3 GPU batch all-logits prefill failed, falling back to serial: {e}"
                    );
                }
            }
        }

        ForwardPass::forward_batch_all_logits(self, ctx, token_ids, kv, weights, logits_all)
    }

    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()> {
        // Gemma3 should have GELU gate activation
        if config.gate_activation != crate::model::config::GateActivation::GELU {
            tracing::warn!(
                "Gemma3Forward selected but gate_activation is {:?}, expected GELU",
                config.gate_activation
            );
        }
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "gemma3"
    }

    fn supports_pipelined_decode(&self, _ctx: &ForwardContext) -> bool {
        true
    }

    fn embed_pipelined_token(
        &self,
        ctx: &ForwardContext,
        token_id: u32,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        weights: &WeightStore,
    ) -> anyhow::Result<()> {
        let dim = ctx.config.embedding_dim as usize;
        let hidden = unsafe {
            std::slice::from_raw_parts_mut(hidden_buf.contents().as_ptr() as *mut f32, dim)
        };
        weights.dequantize_row("token_embd.weight", token_id as usize, hidden)?;
        if ctx.config.embed_scale {
            let embd_scale = (dim as f32).sqrt();
            for h in hidden.iter_mut() {
                *h *= embd_scale;
            }
        }
        Ok(())
    }

    fn encode_pending_decode_step(
        &self,
        ctx: &ForwardContext,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
    ) -> anyhow::Result<Option<ax_engine_metal::PendingFrame>> {
        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(None);
        };
        let Some(gpu_kv) = kv.as_gpu_mut() else {
            return Ok(None);
        };
        let frame = encode_gemma3_pending_step(
            metal_ops, ctx.config, hidden_buf, position, gpu_kv, weights,
        )?;
        Ok(Some(frame))
    }

    fn postprocess_pipelined_logits(
        &self,
        ctx: &ForwardContext,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        if let Some(scale) = ctx.config.logit_scale {
            for l in logits.iter_mut() {
                *l *= scale;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma3_forward_arch_name() {
        let fwd = Gemma3Forward;
        assert_eq!(fwd.arch_name(), "gemma3");
    }

    #[test]
    fn test_sliding_window_detection() {
        let mut config = ModelConfig {
            architecture: "gemma3".into(),
            n_layers: 12,
            n_heads: 8,
            n_kv_heads: 4,
            embedding_dim: 2560,
            head_dim: 256,
            intermediate_dim: 10240,
            context_length: 8192,
            vocab_size: 256000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1000000.0,
            has_qkv_bias: false,
            sliding_window_size: Some(1024),
            sliding_window_pattern: Some(6),
            gate_activation: crate::model::config::GateActivation::GELU,
            tie_word_embeddings: true,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: true,
            rope_freq_base_local: Some(10000.0),
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            expert_intermediate_dim: None,
        };

        // Pattern=6: layers 0-4 = local (sliding window), layer 5 = global
        assert!(Gemma3Forward::use_sliding_window(0, &config)); // local
        assert!(Gemma3Forward::use_sliding_window(1, &config)); // local
        assert!(Gemma3Forward::use_sliding_window(2, &config)); // local
        assert!(Gemma3Forward::use_sliding_window(3, &config)); // local
        assert!(Gemma3Forward::use_sliding_window(4, &config)); // local
        assert!(!Gemma3Forward::use_sliding_window(5, &config)); // global (5 % 6 == 5 == 6-1)
        assert!(Gemma3Forward::use_sliding_window(6, &config)); // local
        assert!(!Gemma3Forward::use_sliding_window(11, &config)); // global (11 % 6 == 5 == 6-1)

        // No sliding window when config doesn't specify it
        config.sliding_window_size = None;
        config.sliding_window_pattern = None;
        assert!(!Gemma3Forward::use_sliding_window(0, &config));
        assert!(!Gemma3Forward::use_sliding_window(1, &config));
    }

    #[test]
    fn test_gpu_prefill_chunk_len() {
        let mut config = ModelConfig {
            architecture: "gemma3".into(),
            n_layers: 12,
            n_heads: 8,
            n_kv_heads: 4,
            embedding_dim: 2560,
            head_dim: 256,
            intermediate_dim: 10240,
            context_length: 8192,
            vocab_size: 256000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1000000.0,
            has_qkv_bias: false,
            sliding_window_size: Some(1024),
            sliding_window_pattern: Some(6),
            gate_activation: crate::model::config::GateActivation::GELU,
            tie_word_embeddings: true,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: true,
            rope_freq_base_local: Some(10000.0),
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            expert_intermediate_dim: None,
        };

        assert_eq!(Gemma3Forward::gpu_prefill_chunk_len(&config, 512), None);
        assert_eq!(Gemma3Forward::gpu_prefill_chunk_len(&config, 1024), None);
        assert_eq!(
            Gemma3Forward::gpu_prefill_chunk_len(&config, 2048),
            Some(1024)
        );

        config.sliding_window_size = None;
        assert_eq!(Gemma3Forward::gpu_prefill_chunk_len(&config, 4096), None);
    }

    #[test]
    fn test_per_head_rms_norm() {
        // 2 heads, head_dim=4, weights all 1.0
        let mut buf = vec![2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];

        per_head_rms_norm(&mut buf, 2, 4, &weight, 1e-6);

        // Head 0: all 2.0 → RMS = 2.0, result ≈ [1,1,1,1]
        for (i, &v) in buf[..4].iter().enumerate() {
            assert!((v - 1.0).abs() < 0.01, "head0[{i}]: {v} != 1.0");
        }
        // Head 1: all 4.0 → RMS = 4.0, result ≈ [1,1,1,1]
        for (i, &v) in buf[4..8].iter().enumerate() {
            assert!((v - 1.0).abs() < 0.01, "head1[{i}]: {v} != 1.0");
        }
    }
}
