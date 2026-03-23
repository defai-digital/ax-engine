//! Falcon-family transformer forward pass.
//!
//! Key differences from LLaMA:
//!   1. LayerNorm with bias (original Falcon) or RMSNorm (Falcon-2)
//!   2. Parallel attention: attn and FFN share the same norm output (original Falcon)
//!   3. GELU activation in FFN
//!   4. No gate projection in FFN for original Falcon (just up → GELU → down)
//!   5. Falcon-2 variants may have gated FFN (gate+up → GeGLU → down)
//!
//! Detection strategy:
//!   - Parallel attention: detected by absence of `blk.0.ffn_norm.weight`
//!   - LayerNorm bias: detected by presence of `blk.0.attn_norm.bias`
//!   - Gated FFN: detected by presence of `blk.0.ffn_gate.weight`
//!
//! GPU support:
//!   - Sequential attention + no norm bias: full GPU path (RMSNorm on GPU)
//!   - Parallel attention or norm bias: CPU fallback (no GPU LayerNorm kernel)

use crate::backend::metal::MetalOps;
use crate::compute::attention;
use crate::compute::gelu;
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

/// CPU LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
fn layer_norm_out(x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// Falcon-family forward pass implementation.
///
/// Supports both original Falcon (parallel attention, LayerNorm, no-gate FFN)
/// and Falcon-2 (sequential, RMSNorm, gated FFN).
#[derive(Debug)]
pub struct FalconForward;

/// Encode all GPU layers for sequential (Falcon-2 style) models.
///
/// Only used when no norm bias is present (can use RMSNorm on GPU).
#[allow(clippy::too_many_arguments)]
fn encode_falcon_gpu_layers_sequential(
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
    has_gate: bool,
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

    for layer in 0..n_layers {
        let lw = &cached.layers[layer];

        let norm_w_buf = weight_cache.get(&lw.attn_norm).unwrap();
        let wq_buf = weight_cache.get(&lw.wq).unwrap();
        let wk_buf = weight_cache.get(&lw.wk).unwrap();
        let wv_buf = weight_cache.get(&lw.wv).unwrap();

        // 1. RMSNorm: hidden → norm_buf
        metal_ops.elementwise.encode_rms_norm_out(
            encoder,
            hidden_buf,
            norm_w_buf,
            &s.norm_buf,
            dim as u32,
            eps,
        );
        ax_metal::barrier_buffers(encoder);

        // 2. QKV matmul
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
        ax_metal::barrier_buffers(encoder);

        // 3. RoPE
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
        ax_metal::barrier_buffers(encoder);

        // 4. KV cache append
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
        ax_metal::barrier_buffers(encoder);

        // 5. Attention
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
        ax_metal::barrier_buffers(encoder);

        // 6. Output projection + residual
        let wo_buf = weight_cache.get(&lw.wo).unwrap();
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
        ax_metal::barrier_buffers(encoder);
        metal_ops
            .elementwise
            .encode_elementwise_add(encoder, hidden_buf, &s.proj_buf, dim as u32);
        ax_metal::barrier_buffers(encoder);

        // 7. FFN norm + FFN
        let ffn_nw_buf = weight_cache.get(&lw.ffn_norm).unwrap();
        metal_ops.elementwise.encode_rms_norm_out(
            encoder,
            hidden_buf,
            ffn_nw_buf,
            &s.norm_buf,
            dim as u32,
            eps,
        );
        ax_metal::barrier_buffers(encoder);

        if has_gate {
            // Gated FFN: GELU(gate) * up → down (Falcon-2 / GeGLU)
            let wg_buf = weight_cache.get(&lw.wg).unwrap();
            let wu_buf = weight_cache.get(&lw.wu).unwrap();
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
            ax_metal::barrier_buffers(encoder);
            metal_ops.elementwise.encode_gelu_elementwise_mul(
                encoder,
                &s.gate_buf,
                &s.up_buf,
                inter_dim as u32,
            );
        } else {
            // No-gate FFN: up → GELU → down (original Falcon)
            let wu_buf = weight_cache.get(&lw.wu).unwrap();
            encode_dequant_matvec(
                metal_ops,
                encoder,
                wu_buf,
                &s.norm_buf,
                &s.gate_buf,
                inter_dim as u32,
                dim as u32,
                lw.wu_dtype,
            );
            ax_metal::barrier_buffers(encoder);
            metal_ops
                .elementwise
                .encode_gelu_inplace(encoder, &s.gate_buf, inter_dim as u32);
        }
        ax_metal::barrier_buffers(encoder);

        // 8. Down projection + residual
        let wd_buf = weight_cache.get(&lw.wd).unwrap();
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
        ax_metal::barrier_buffers(encoder);
        metal_ops
            .elementwise
            .encode_elementwise_add(encoder, hidden_buf, &s.down_buf, dim as u32);
        if layer + 1 < n_layers {
            ax_metal::barrier_buffers(encoder);
        }
    }

    Ok(())
}

fn encode_falcon_gpu_output_head(
    encoder: &ax_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_metal::MetalBuffer,
    cfg: &ModelConfig,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_metal::MetalBuffer>,
) {
    let dim = cfg.embedding_dim as usize;
    let vocab_size = cfg.vocab_size as usize;
    let eps = cfg.rms_norm_eps;

    let fnw_buf = weight_cache.get(&cached.output_norm).unwrap();
    ax_metal::barrier_buffers(encoder);
    metal_ops
        .elementwise
        .encode_rms_norm(encoder, hidden_buf, fnw_buf, dim as u32, eps);
    ax_metal::barrier_buffers(encoder);
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

fn encode_falcon_pending_step(
    metal_ops: &MetalOps,
    cfg: &ModelConfig,
    hidden_buf: &ax_metal::MetalBuffer,
    position: usize,
    gpu_kv: &mut crate::kv::GpuKv,
    weights: &WeightStore,
    has_gate: bool,
) -> anyhow::Result<ax_metal::PendingFrame> {
    let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as usize;
    let kv_f16 = gpu_kv.is_f16();

    gpu_kv.ensure_capacity(&metal_ops.device, position + 1)?;
    metal_ops.init_scratches(cfg);

    let scratch_guard = metal_ops.scratches();
    let s = scratch_guard.as_ref().unwrap();

    if !metal_ops.has_cached_model_keys() {
        FalconForward::build_cached_model_keys(metal_ops, weights, cfg)?;
    }

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();

    let kv_offset = (position * kv_dim) as u32;
    let full_seq_len = position + 1;
    let rope_position = match cfg.rope_scaling {
        crate::model::config::RopeScaling::Linear(factor) => position as f32 / factor,
        crate::model::config::RopeScaling::None => position as f32,
    };

    let weight_cache = metal_ops.lock_weight_cache();

    metal_ops.device.encode_frame(|encoder| {
        encode_falcon_gpu_layers_sequential(
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
            has_gate,
        )?;
        encode_falcon_gpu_output_head(
            encoder,
            metal_ops,
            s,
            hidden_buf,
            cfg,
            cached,
            &weight_cache,
        );
        Ok(())
    })
}

impl FalconForward {
    /// Detect if model uses parallel attention (no separate FFN norm).
    fn is_parallel_attention(weights: &WeightStore) -> bool {
        !weights.has("blk.0.ffn_norm.weight")
    }

    /// Detect if model has LayerNorm bias (original Falcon).
    fn has_norm_bias(weights: &WeightStore) -> bool {
        weights.has("blk.0.attn_norm.bias")
    }

    /// Detect if model has gated FFN (Falcon-2 style).
    fn has_gated_ffn(weights: &WeightStore) -> bool {
        weights.has("blk.0.ffn_gate.weight")
    }

    /// Whether this model can use the GPU path.
    /// GPU requires: sequential attention + no norm bias (so RMSNorm works).
    fn can_gpu(weights: &WeightStore) -> bool {
        !Self::is_parallel_attention(weights) && !Self::has_norm_bias(weights)
    }

    /// Build GPU weight cache keys (sequential Falcon only).
    fn build_cached_model_keys(
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
        let has_gate = Self::has_gated_ffn(weights);
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

            let (wo_raw, wo_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            let wo_key = metal_ops.ensure_quant_cached(wo_raw);
            if use_precomputed_f16 && wo_dtype == GgmlType::Q4K {
                metal_ops.ensure_precomputed_q4k_f16_from_raw(wo_raw, dim as u32, dim as u32)?;
            }
            if use_precomputed_f16 && wo_dtype == GgmlType::Q6K {
                metal_ops.ensure_precomputed_q6k_f16_from_raw(wo_raw, dim as u32, dim as u32)?;
            }

            // FFN weights
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);

            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            let wu_key = metal_ops.ensure_quant_cached(wu_raw);
            let (wd_raw, wd_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            let wd_key = metal_ops.ensure_quant_cached(wd_raw);

            // Gate (optional, Falcon-2 only)
            let (wg_key, wg_dtype) = if has_gate {
                let (wg_raw, wg_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
                let key = metal_ops.ensure_quant_cached(wg_raw);
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
                (key, wg_dtype)
            } else {
                // Placeholder — wg won't be used for no-gate path
                (wu_key, wu_dtype)
            };

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

        metal_ops.set_cached_model_keys(CachedModelKeys {
            layers,
            output_norm: output_norm_key,
            lm_head: lm_head_key,
            lm_head_dtype: lm_dtype,
        });
        Ok(())
    }

    /// GPU decode for sequential (Falcon-2 style) models.
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
        let has_gate = Self::has_gated_ffn(weights);

        assert!(logits.len() >= vocab_size);

        metal_ops.init_scratches(cfg);
        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();

        let kv_f16 = gpu_kv.is_f16();
        let next_seq = gpu_kv.seq_len() + 1;
        gpu_kv.ensure_capacity(&metal_ops.device, next_seq)?;

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

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();

        let cur_seq_len = gpu_kv.seq_len();
        let kv_offset = (cur_seq_len * kv_dim) as u32;
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        {
            let weight_cache = metal_ops.lock_weight_cache();
            let exec_t = OpTimer::start();
            metal_ops.device.execute_sync(|encoder| {
                encode_falcon_gpu_layers_sequential(
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
                    has_gate,
                )?;
                encode_falcon_gpu_output_head(
                    encoder,
                    metal_ops,
                    s,
                    &s.hidden,
                    cfg,
                    cached,
                    &weight_cache,
                );
                Ok(())
            })?;
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_execute += exec_t.elapsed();
            }
        }

        gpu_kv.finalize_token();

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

    /// Batched GPU prefill for sequential Falcon.
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
        let has_gate = Self::has_gated_ffn(weights);
        let emit_all_logits = logits_all.is_some();

        if let Some(logits) = last_logits.as_ref() {
            assert!(logits.len() >= vocab_size);
        }

        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, n_tokens);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let batch_guard = metal_ops.batch_scratches();
        let bs = batch_guard.as_ref().unwrap();

        let kv_f16 = gpu_kv.is_f16();
        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + n_tokens)?;

        // Embed all N tokens
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

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys(metal_ops, weights, cfg)?;
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

        {
            let weight_cache = metal_ops.lock_weight_cache();
            let use_f16_batch_io =
                crate::backend::metal::metal_batch_f16_io_enabled_for_arch(&cfg.architecture);
            let use_f16_pair =
                crate::backend::metal::metal_batch_f16_pair_enabled_for_arch(&cfg.architecture);
            let use_batch_simd =
                crate::backend::metal::metal_batch_simd_enabled_for_arch(&cfg.architecture);

            metal_ops.device.execute_sync(|encoder| {
                let nt = n_tokens as u32;

                for layer in 0..n_layers {
                    let lw = &cached.layers[layer];

                    let norm_w_buf = weight_cache.get(&lw.attn_norm).unwrap();
                    let wq_buf = weight_cache.get(&lw.wq).unwrap();
                    let wk_buf = weight_cache.get(&lw.wk).unwrap();
                    let wv_buf = weight_cache.get(&lw.wv).unwrap();

                    // Phase 1: Batched RMSNorm
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        norm_w_buf,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        eps,
                    );
                    ax_metal::barrier_buffers(encoder);

                    // Phase 2: Batched QKV matmul
                    if use_f16_batch_io {
                        metal_ops.elementwise.encode_cast_f32_to_f16(
                            encoder,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            nt * dim as u32,
                        );
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

                    // Phase 3: Batched RoPE + KV append
                    let (rope_start, rope_step) = match cfg.rope_scaling {
                        crate::model::config::RopeScaling::Linear(f) => {
                            (base_seq_len as f32 / f, 1.0f32 / f)
                        }
                        crate::model::config::RopeScaling::None => (base_seq_len as f32, 1.0f32),
                    };
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

                    // Phase 4: Batched attention
                    if base_seq_len == 0 {
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

                    // Phase 5: Output proj + residual
                    let wo_buf = weight_cache.get(&lw.wo).unwrap();
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
                        use_f16_batch_io,
                        use_batch_simd,
                    );
                    ax_metal::barrier_buffers(encoder);

                    // Phase 6: FFN norm + FFN
                    let ffn_nw_buf = weight_cache.get(&lw.ffn_norm).unwrap();
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
                    ax_metal::barrier_buffers(encoder);

                    if has_gate {
                        let wg_buf = weight_cache.get(&lw.wg).unwrap();
                        let wu_buf = weight_cache.get(&lw.wu).unwrap();
                        if use_f16_batch_io {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.norm_buf,
                                &bs.matmul_in_f16,
                                nt * dim as u32,
                            );
                            let use_pair_this_layer = (use_f16_pair
                                || (lw.wg_dtype == GgmlType::Q8_0 && n_tokens <= 256))
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
                        metal_ops.elementwise.encode_gelu_elementwise_mul_batch(
                            encoder,
                            &bs.gate_buf,
                            &bs.up_buf,
                            inter_dim as u32,
                            nt,
                        );
                    } else {
                        // No-gate FFN
                        let wu_buf = weight_cache.get(&lw.wu).unwrap();
                        if use_f16_batch_io {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.norm_buf,
                                &bs.matmul_in_f16,
                                nt * dim as u32,
                            );
                            encode_dequant_batch_f16in(
                                metal_ops,
                                encoder,
                                wu_buf,
                                &bs.matmul_in_f16,
                                &bs.gate_buf,
                                inter_dim as u32,
                                nt,
                                dim as u32,
                                lw.wu_dtype,
                            );
                        } else {
                            encode_dequant_batch(
                                &metal_ops.dequant,
                                &metal_ops.elementwise,
                                encoder,
                                wu_buf,
                                &bs.norm_buf,
                                &bs.gate_buf,
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
                        metal_ops.elementwise.encode_gelu_inplace_batch(
                            encoder,
                            &bs.gate_buf,
                            inter_dim as u32,
                            nt,
                        );
                    }
                    ax_metal::barrier_buffers(encoder);

                    // Phase 7: Down proj + residual
                    let wd_buf = weight_cache.get(&lw.wd).unwrap();
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
                        use_f16_batch_io,
                        use_batch_simd,
                    );
                    ax_metal::barrier_buffers(encoder);
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                    );
                    ax_metal::barrier_buffers(encoder);
                }

                // Output head
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
                    let last_off = (n_tokens - 1) * dim * 4;
                    metal_ops.elementwise.encode_buffer_copy(
                        encoder, &bs.hidden, last_off, &s.hidden, 0, dim as u32,
                    );
                    ax_metal::barrier_buffers(encoder);
                    metal_ops
                        .elementwise
                        .encode_rms_norm(encoder, &s.hidden, fnw_buf, dim as u32, eps);
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

        gpu_kv.finalize_batch(n_tokens);

        if let Some(logits_all) = logits_all {
            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    all_logits_buf.as_ref().unwrap().contents().as_ptr() as *const f32,
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

impl ForwardPass for FalconForward {
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
        // GPU path: only for sequential, no-bias models
        if Self::can_gpu(weights)
            && ctx.backend.use_gpu_decode()
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

        // CPU path (all Falcon variants)
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let parallel_attn = Self::is_parallel_attention(weights);
        let has_norm_bias = Self::has_norm_bias(weights);
        let has_gate = Self::has_gated_ffn(weights);

        assert!(logits.len() >= vocab_size);

        let cpu_kv = kv
            .as_cpu_mut()
            .expect("FalconForward CPU path requires ModelKv::Cpu");

        // Step 1: Token embedding
        let mut hidden = vec![0.0f32; dim];
        timed!(
            ops,
            dequant,
            weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?
        );

        // Scratch buffers
        let mut norm_buf = vec![0.0f32; dim];
        let mut q_buf = vec![0.0f32; n_heads * head_dim];
        let mut k_buf = vec![0.0f32; n_kv_heads * head_dim];
        let mut v_buf = vec![0.0f32; n_kv_heads * head_dim];
        let mut attn_out = vec![0.0f32; n_heads * head_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = if has_gate {
            vec![0.0f32; inter_dim]
        } else {
            Vec::new()
        };
        let mut down_buf = vec![0.0f32; dim];

        // Step 2: Transformer layers
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            // 2a. Attention norm (LayerNorm with bias, or RMSNorm)
            let attn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?
            );
            if has_norm_bias {
                let attn_norm_b = timed!(
                    ops,
                    dequant,
                    weights.f32_slice(&format!("{prefix}.attn_norm.bias"))?
                );
                timed!(
                    ops,
                    norm,
                    layer_norm_out(
                        &hidden,
                        attn_norm_w,
                        attn_norm_b,
                        &mut norm_buf,
                        cfg.rms_norm_eps
                    )
                );
            } else {
                timed!(
                    ops,
                    norm,
                    rms_norm::rms_norm_out(&hidden, attn_norm_w, &mut norm_buf, cfg.rms_norm_eps)
                );
            }

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

            // 2c. RoPE
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

            // 2d. KV cache
            cpu_kv.append_and_advance(layer, &k_buf, &v_buf);

            // 2e. Attention
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

            // 2f. Output projection
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

            if parallel_attn {
                // ── Parallel attention: FFN uses same norm_buf, both added to residual ──

                // FFN: up → GELU → down (no gate for original Falcon)
                let (wu_raw, wu_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
                timed!(
                    ops,
                    matmul,
                    ctx.backend.dequant_matmul(
                        wu_raw,
                        wu_dtype,
                        &norm_buf,
                        &mut gate_buf,
                        inter_dim,
                        1,
                        dim,
                    )
                );
                gelu::gelu(&mut gate_buf);

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

                // Residual: hidden += attn_proj + ffn_down
                silu::elementwise_add(&mut hidden, &proj_buf);
                silu::elementwise_add(&mut hidden, &down_buf);
            } else {
                // ── Sequential attention (Falcon-2 style) ──

                // Residual from attention
                silu::elementwise_add(&mut hidden, &proj_buf);

                // FFN norm
                let ffn_norm_w = timed!(
                    ops,
                    dequant,
                    weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?
                );
                if has_norm_bias && weights.has(&format!("{prefix}.ffn_norm.bias")) {
                    let ffn_norm_b = timed!(
                        ops,
                        dequant,
                        weights.f32_slice(&format!("{prefix}.ffn_norm.bias"))?
                    );
                    timed!(
                        ops,
                        norm,
                        layer_norm_out(
                            &hidden,
                            ffn_norm_w,
                            ffn_norm_b,
                            &mut norm_buf,
                            cfg.rms_norm_eps
                        )
                    );
                } else {
                    timed!(
                        ops,
                        norm,
                        rms_norm::rms_norm_out(
                            &hidden,
                            ffn_norm_w,
                            &mut norm_buf,
                            cfg.rms_norm_eps
                        )
                    );
                }

                if has_gate {
                    // Gated FFN: GELU(gate) * up → down
                    let (wg_raw, wg_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
                    let (wu_raw, wu_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
                    timed!(ops, matmul, {
                        ctx.backend.batch_dequant_matvec(
                            &[(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)],
                            &norm_buf,
                            dim,
                            &mut [&mut gate_buf, &mut up_buf],
                        );
                    });
                    gelu::gelu_elementwise_mul(&mut gate_buf, &up_buf);
                } else {
                    // No-gate FFN: up → GELU → down
                    let (wu_raw, wu_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
                    timed!(
                        ops,
                        matmul,
                        ctx.backend.dequant_matmul(
                            wu_raw,
                            wu_dtype,
                            &norm_buf,
                            &mut gate_buf,
                            inter_dim,
                            1,
                            dim,
                        )
                    );
                    gelu::gelu(&mut gate_buf);
                }

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

                silu::elementwise_add(&mut hidden, &down_buf);
            }
        }

        cpu_kv.finalize_token();

        // Step 3: Final RMSNorm
        let final_norm_w = timed!(ops, dequant, weights.f32_slice("output_norm.weight")?);
        if has_norm_bias && weights.has("output_norm.bias") {
            let final_norm_b = timed!(ops, dequant, weights.f32_slice("output_norm.bias")?);
            layer_norm_out(
                &hidden,
                final_norm_w,
                final_norm_b,
                &mut norm_buf,
                cfg.rms_norm_eps,
            );
            hidden.copy_from_slice(&norm_buf);
        } else {
            timed!(
                ops,
                norm,
                rms_norm::rms_norm(&mut hidden, final_norm_w, cfg.rms_norm_eps)
            );
        }

        // Step 4: LM head
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
        let force_serial = std::env::var("AX_SERIAL_PREFILL").is_ok();
        let can_gpu_batch = !force_serial
            && Self::can_gpu(weights)
            && ctx.backend.use_gpu_decode()
            && token_ids.len() > 1
            && kv.as_gpu_mut().is_some()
            && gpu_quant_supported(weights);

        if can_gpu_batch && let Some(metal_ops) = ctx.backend.metal_ops() {
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
            && Self::can_gpu(weights)
            && ctx.backend.use_gpu_decode()
            && token_ids.len() > 1
            && kv.as_gpu_mut().is_some()
            && lm_head_dtype_supported
            && gpu_quant_supported(weights);

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
        "falcon"
    }

    fn supports_pipelined_decode(&self, _ctx: &ForwardContext) -> bool {
        // Only sequential (Falcon-2 style) can pipeline on GPU
        true
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
        if !Self::can_gpu(weights) {
            return Ok(None);
        }
        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(None);
        };
        let Some(gpu_kv) = kv.as_gpu_mut() else {
            return Ok(None);
        };
        let has_gate = Self::has_gated_ffn(weights);
        let frame = encode_falcon_pending_step(
            metal_ops, ctx.config, hidden_buf, position, gpu_kv, weights, has_gate,
        )?;
        Ok(Some(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_falcon_forward_arch_name() {
        let fwd = FalconForward;
        assert_eq!(fwd.arch_name(), "falcon");
    }

    #[test]
    fn test_layer_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0f32; 4];
        layer_norm_out(&x, &w, &b, &mut out, 1e-5);
        // Mean = 2.5, Var = 1.25
        let mean = 2.5f32;
        let var = 1.25f32;
        let inv_std = 1.0 / (var + 1e-5f32).sqrt();
        for i in 0..4 {
            let expected = (x[i] - mean) * inv_std;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "layer_norm mismatch at {i}: got {} expected {}",
                out[i],
                expected
            );
        }
    }
}
