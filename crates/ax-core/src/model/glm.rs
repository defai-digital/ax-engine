//! GLM-family (ChatGLM-4/5) transformer forward pass.
//!
//! GLM is a hybrid of Qwen3 and Gemma3 features:
//!   1. QKV bias terms (shared with Qwen3)
//!   2. SwiGLU activation (SiLU gated FFN, same as LLaMA/Qwen3)
//!   3. Post-attention and post-FFN RMSNorm (shared with Gemma3)
//!   4. Optional per-head QK normalization
//!   5. RoPE positional encoding
//!
//! Detection strategy (from GGUF weights):
//!   - QKV bias: `blk.0.attn_q.bias` present
//!   - Post-layer norms: `blk.0.post_attention_norm.weight` present
//!   - Per-head QK norm: `blk.0.attn_q_norm.weight` present

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
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    encode_dequant_matvec, gpu_decode_quant_supported, gpu_quant_supported, per_head_rms_norm,
};
use crate::model::weights::WeightStore;

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

/// GLM-family forward pass implementation.
#[derive(Debug)]
pub struct GlmForward;

// ── GPU single-token decode ─────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn encode_glm_gpu_layers_only(
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

        // 1. Attention norm
        let norm_w_buf = weight_cache.get(&lw.attn_norm).unwrap();
        metal_ops.elementwise.encode_rms_norm_out(
            encoder, hidden_buf, norm_w_buf, &s.norm_buf, dim as u32, eps,
        );
        ax_metal::barrier_buffers(encoder);

        // 2. QKV matmul
        let wq_buf = weight_cache.get(&lw.wq).unwrap();
        let wk_buf = weight_cache.get(&lw.wk).unwrap();
        let wv_buf = weight_cache.get(&lw.wv).unwrap();
        encode_dequant_matvec(metal_ops, encoder, wq_buf, &s.norm_buf, &s.q_buf, q_dim as u32, dim as u32, lw.wq_dtype);
        encode_dequant_matvec(metal_ops, encoder, wk_buf, &s.norm_buf, &s.k_buf, kv_dim as u32, dim as u32, lw.wk_dtype);
        encode_dequant_matvec(metal_ops, encoder, wv_buf, &s.norm_buf, &s.v_buf, kv_dim as u32, dim as u32, lw.wv_dtype);
        ax_metal::barrier_buffers(encoder);

        // 3. QKV bias
        if let (Some(qb_key), Some(kb_key), Some(vb_key)) = (lw.q_bias, lw.k_bias, lw.v_bias) {
            let qb_buf = weight_cache.get(&qb_key).unwrap();
            let kb_buf = weight_cache.get(&kb_key).unwrap();
            let vb_buf = weight_cache.get(&vb_key).unwrap();
            metal_ops.elementwise.encode_elementwise_add(encoder, &s.q_buf, qb_buf, q_dim as u32);
            metal_ops.elementwise.encode_elementwise_add(encoder, &s.k_buf, kb_buf, kv_dim as u32);
            metal_ops.elementwise.encode_elementwise_add(encoder, &s.v_buf, vb_buf, kv_dim as u32);
            ax_metal::barrier_buffers(encoder);
        }

        // 4. Per-head QK norm (optional)
        if let (Some(q_norm_key), Some(k_norm_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
            let q_nw_buf = weight_cache.get(&q_norm_key).unwrap();
            let k_nw_buf = weight_cache.get(&k_norm_key).unwrap();
            metal_ops.elementwise.encode_per_head_rms_norm(encoder, &s.q_buf, q_nw_buf, n_heads as u32, head_dim as u32, eps);
            metal_ops.elementwise.encode_per_head_rms_norm(encoder, &s.k_buf, k_nw_buf, n_kv_heads as u32, head_dim as u32, eps);
            ax_metal::barrier_buffers(encoder);
        }

        // 5. RoPE
        metal_ops.elementwise.encode_rope(
            encoder, &s.q_buf, &s.k_buf,
            n_heads as u32, n_kv_heads as u32, head_dim as u32,
            rope_position, cfg.rope_freq_base,
        );
        ax_metal::barrier_buffers(encoder);

        // 6. KV cache
        metal_ops.elementwise.encode_kv_append(encoder, &s.k_buf, gpu_kv.k_buffer(layer), kv_f16, kv_offset, kv_dim as u32);
        metal_ops.elementwise.encode_kv_append(encoder, &s.v_buf, gpu_kv.v_buffer(layer), kv_f16, kv_offset, kv_dim as u32);
        ax_metal::barrier_buffers(encoder);

        // 7. Attention
        metal_ops.attention.encode_attention_decode_with_scratch(
            encoder, &s.q_buf, gpu_kv.k_buffer(layer), gpu_kv.v_buffer(layer),
            &s.attn_out, &s.splitk_partial_out, &s.splitk_partial_lse,
            kv_f16, n_heads as u32, n_kv_heads as u32, head_dim as u32, 0, full_seq_len as u32,
        );
        ax_metal::barrier_buffers(encoder);

        // 8. Output projection
        let wo_buf = weight_cache.get(&lw.wo).unwrap();
        encode_dequant_matvec(metal_ops, encoder, wo_buf, &s.attn_out, &s.proj_buf, dim as u32, q_dim as u32, lw.wo_dtype);
        ax_metal::barrier_buffers(encoder);

        // 9. Post-attention norm (GLM-specific)
        if let Some(post_attn_key) = lw.post_attn_norm {
            let nw = weight_cache.get(&post_attn_key).unwrap();
            metal_ops.elementwise.encode_rms_norm(encoder, &s.proj_buf, nw, dim as u32, eps);
            ax_metal::barrier_buffers(encoder);
        }

        // 10. Residual
        metal_ops.elementwise.encode_elementwise_add(encoder, hidden_buf, &s.proj_buf, dim as u32);
        ax_metal::barrier_buffers(encoder);

        // 11. FFN norm
        let ffn_nw_buf = weight_cache.get(&lw.ffn_norm).unwrap();
        metal_ops.elementwise.encode_rms_norm_out(encoder, hidden_buf, ffn_nw_buf, &s.norm_buf, dim as u32, eps);
        ax_metal::barrier_buffers(encoder);

        // 12. Gate + Up → SiLU*mul → Down
        let wg_buf = weight_cache.get(&lw.wg).unwrap();
        let wu_buf = weight_cache.get(&lw.wu).unwrap();
        encode_dequant_matvec(metal_ops, encoder, wg_buf, &s.norm_buf, &s.gate_buf, inter_dim as u32, dim as u32, lw.wg_dtype);
        encode_dequant_matvec(metal_ops, encoder, wu_buf, &s.norm_buf, &s.up_buf, inter_dim as u32, dim as u32, lw.wu_dtype);
        ax_metal::barrier_buffers(encoder);

        metal_ops.elementwise.encode_silu_elementwise_mul(encoder, &s.gate_buf, &s.up_buf, inter_dim as u32);
        ax_metal::barrier_buffers(encoder);

        let wd_buf = weight_cache.get(&lw.wd).unwrap();
        encode_dequant_matvec(metal_ops, encoder, wd_buf, &s.gate_buf, &s.down_buf, dim as u32, inter_dim as u32, lw.wd_dtype);
        ax_metal::barrier_buffers(encoder);

        // 13. Post-FFN norm (GLM-specific)
        if let Some(post_ffn_key) = lw.post_ffn_norm {
            let nw = weight_cache.get(&post_ffn_key).unwrap();
            metal_ops.elementwise.encode_rms_norm(encoder, &s.down_buf, nw, dim as u32, eps);
            ax_metal::barrier_buffers(encoder);
        }

        // 14. Residual
        metal_ops.elementwise.encode_elementwise_add(encoder, hidden_buf, &s.down_buf, dim as u32);
        if layer + 1 < n_layers {
            ax_metal::barrier_buffers(encoder);
        }
    }

    Ok(())
}

fn encode_glm_gpu_output_head(
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
    metal_ops.elementwise.encode_rms_norm(encoder, hidden_buf, fnw_buf, dim as u32, eps);
    ax_metal::barrier_buffers(encoder);
    let lm_buf = weight_cache.get(&cached.lm_head).unwrap();
    encode_dequant_matvec(metal_ops, encoder, lm_buf, hidden_buf, &s.logits_buf, vocab_size as u32, dim as u32, cached.lm_head_dtype);
}

fn encode_glm_pending_step(
    metal_ops: &MetalOps,
    cfg: &ModelConfig,
    hidden_buf: &ax_metal::MetalBuffer,
    position: usize,
    gpu_kv: &mut crate::kv::GpuKv,
    weights: &WeightStore,
) -> anyhow::Result<ax_metal::PendingFrame> {
    let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as usize;
    let kv_f16 = gpu_kv.is_f16();
    gpu_kv.ensure_capacity(&metal_ops.device, position + 1)?;
    metal_ops.init_scratches(cfg);
    let scratch_guard = metal_ops.scratches();
    let s = scratch_guard.as_ref().unwrap();
    if !metal_ops.has_cached_model_keys() {
        GlmForward::build_cached_model_keys(metal_ops, weights, cfg)?;
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
        encode_glm_gpu_layers_only(encoder, metal_ops, s, hidden_buf, cfg, kv_offset, rope_position, full_seq_len, kv_f16, gpu_kv, cached, &weight_cache)?;
        encode_glm_gpu_output_head(encoder, metal_ops, s, hidden_buf, cfg, cached, &weight_cache);
        Ok(())
    })
}

impl GlmForward {
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
        let use_precomputed_f16 = crate::backend::metal::metal_precompute_f16_enabled_for_model(cfg);

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

            for (raw, m, k) in [(wq_raw, q_dim as u32, dim as u32), (wk_raw, kv_dim as u32, dim as u32), (wv_raw, kv_dim as u32, dim as u32)] {
                if use_precomputed_f16 && wq_dtype == GgmlType::Q4K { metal_ops.ensure_precomputed_q4k_f16_from_raw(raw, m, k)?; }
                if use_precomputed_f16 && wq_dtype == GgmlType::Q6K { metal_ops.ensure_precomputed_q6k_f16_from_raw(raw, m, k)?; }
            }

            // QKV bias
            let q_bias_key = if weights.has(&format!("{prefix}.attn_q.bias")) { Some(metal_ops.ensure_f32_cached(weights.f32_slice(&format!("{prefix}.attn_q.bias"))?)) } else { None };
            let k_bias_key = if weights.has(&format!("{prefix}.attn_k.bias")) { Some(metal_ops.ensure_f32_cached(weights.f32_slice(&format!("{prefix}.attn_k.bias"))?)) } else { None };
            let v_bias_key = if weights.has(&format!("{prefix}.attn_v.bias")) { Some(metal_ops.ensure_f32_cached(weights.f32_slice(&format!("{prefix}.attn_v.bias"))?)) } else { None };

            // Per-head QK norm
            let attn_q_norm_key = if weights.has(&format!("{prefix}.attn_q_norm.weight")) { Some(metal_ops.ensure_f32_cached(weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?)) } else { None };
            let attn_k_norm_key = if weights.has(&format!("{prefix}.attn_k_norm.weight")) { Some(metal_ops.ensure_f32_cached(weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?)) } else { None };

            // Post-layer norms (GLM-specific)
            let post_attn_norm_key = if weights.has(&format!("{prefix}.post_attention_norm.weight")) {
                Some(metal_ops.ensure_f32_cached(weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?))
            } else { None };
            let post_ffn_norm_key = if weights.has(&format!("{prefix}.post_ffw_norm.weight")) {
                Some(metal_ops.ensure_f32_cached(weights.f32_slice(&format!("{prefix}.post_ffw_norm.weight"))?))
            } else { None };

            let (wo_raw, wo_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            let wo_key = metal_ops.ensure_quant_cached(wo_raw);
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);
            let (wg_raw, wg_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            let (wd_raw, wd_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            let wg_key = metal_ops.ensure_quant_cached(wg_raw);
            let wu_key = metal_ops.ensure_quant_cached(wu_raw);
            let wd_key = metal_ops.ensure_quant_cached(wd_raw);

            for (raw, dtype, m, k) in [
                (wo_raw, wo_dtype, dim as u32, q_dim as u32),
                (wg_raw, wg_dtype, inter_dim as u32, dim as u32),
                (wu_raw, wu_dtype, inter_dim as u32, dim as u32),
                (wd_raw, wd_dtype, dim as u32, inter_dim as u32),
            ] {
                if use_precomputed_f16 && dtype == GgmlType::Q4K { metal_ops.ensure_precomputed_q4k_f16_from_raw(raw, m, k)?; }
                if use_precomputed_f16 && dtype == GgmlType::Q6K { metal_ops.ensure_precomputed_q6k_f16_from_raw(raw, m, k)?; }
            }

            layers.push(CachedLayerKeys {
                attn_norm: attn_norm_key,
                wq: wq_key, wq_dtype, wk: wk_key, wk_dtype, wv: wv_key, wv_dtype,
                wo: wo_key, wo_dtype,
                ffn_norm: ffn_norm_key,
                wg: wg_key, wg_dtype, wu: wu_key, wu_dtype, wd: wd_key, wd_dtype,
                attn_q_norm: attn_q_norm_key, attn_k_norm: attn_k_norm_key,
                post_attn_norm: post_attn_norm_key, post_ffn_norm: post_ffn_norm_key,
                q_bias: q_bias_key, k_bias: k_bias_key, v_bias: v_bias_key,
            });
        }

        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        let output_norm_key = metal_ops.ensure_f32_cached(final_norm_w);
        let lm_name = if weights.has("output.weight") { "output.weight" } else { "token_embd.weight" };
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_name)?;
        let lm_head_key = metal_ops.ensure_quant_cached(lm_raw);
        if use_precomputed_f16 && lm_dtype == GgmlType::Q4K { metal_ops.ensure_precomputed_q4k_f16_from_raw(lm_raw, cfg.vocab_size, cfg.embedding_dim)?; }
        if use_precomputed_f16 && lm_dtype == GgmlType::Q6K { metal_ops.ensure_precomputed_q6k_f16_from_raw(lm_raw, cfg.vocab_size, cfg.embedding_dim)?; }

        metal_ops.set_cached_model_keys(CachedModelKeys { layers, output_norm: output_norm_key, lm_head: lm_head_key, lm_head_dtype: lm_dtype });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_single_gpu_unified(&self, ctx: &ForwardContext, metal_ops: &MetalOps, token_id: u32, position: usize, gpu_kv: &mut crate::kv::GpuKv, weights: &WeightStore, logits: &mut [f32], mut ops: Option<&mut OpBreakdown>) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let kv_dim = n_kv_heads * head_dim;

        assert!(logits.len() >= vocab_size);
        metal_ops.init_scratches(cfg);
        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let kv_f16 = gpu_kv.is_f16();
        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + 1)?;

        let setup_t = OpTimer::start();
        { let h = unsafe { std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim) }; weights.dequantize_row("token_embd.weight", token_id as usize, h)?; }
        let rope_position = match cfg.rope_scaling { crate::model::config::RopeScaling::Linear(f) => position as f32 / f, crate::model::config::RopeScaling::None => position as f32 };
        if !metal_ops.has_cached_model_keys() { Self::build_cached_model_keys(metal_ops, weights, cfg)?; }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let cur_seq_len = gpu_kv.seq_len();
        let kv_offset = (cur_seq_len * kv_dim) as u32;
        if let Some(ref mut o) = ops { o.gpu_encode += setup_t.elapsed(); }

        { let wc = metal_ops.lock_weight_cache(); let t = OpTimer::start();
            metal_ops.device.execute_sync(|encoder| {
                encode_glm_gpu_layers_only(encoder, metal_ops, s, &s.hidden, cfg, kv_offset, rope_position, cur_seq_len + 1, kv_f16, gpu_kv, cached, &wc)?;
                encode_glm_gpu_output_head(encoder, metal_ops, s, &s.hidden, cfg, cached, &wc);
                Ok(())
            })?;
            if let Some(ref mut o) = ops { o.gpu_execute += t.elapsed(); }
        }

        gpu_kv.finalize_token();
        let t = OpTimer::start();
        let lg = unsafe { std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab_size) };
        logits[..vocab_size].copy_from_slice(lg);
        if let Some(ref mut o) = ops { o.gpu_readback += t.elapsed(); }
        Ok(())
    }
}

impl ForwardPass for GlmForward {
    fn forward_single(&self, ctx: &ForwardContext, token_id: u32, position: usize, kv: &mut ModelKv, weights: &WeightStore, logits: &mut [f32], mut ops: Option<&mut OpBreakdown>) -> anyhow::Result<()> {
        if ctx.backend.use_gpu_decode() && let Some(metal_ops) = ctx.backend.metal_ops() && let Some(gpu_kv) = kv.as_gpu_mut() && gpu_decode_quant_supported(weights) {
            if let Some(ops_ref) = ops { let t = OpTimer::start(); let r = self.forward_single_gpu_unified(ctx, metal_ops, token_id, position, gpu_kv, weights, logits, Some(ops_ref)); ops_ref.gpu += t.elapsed(); return r; }
            return self.forward_single_gpu_unified(ctx, metal_ops, token_id, position, gpu_kv, weights, logits, None);
        }

        // CPU fallback
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        assert!(logits.len() >= vocab_size);
        let cpu_kv = kv.as_cpu_mut().expect("GlmForward CPU path requires ModelKv::Cpu");

        let mut hidden = vec![0.0f32; dim];
        timed!(ops, dequant, weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?);

        let mut norm_buf = vec![0.0f32; dim];
        let mut q_buf = vec![0.0f32; n_heads * head_dim];
        let mut k_buf = vec![0.0f32; n_kv_heads * head_dim];
        let mut v_buf = vec![0.0f32; n_kv_heads * head_dim];
        let mut attn_out = vec![0.0f32; n_heads * head_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; dim];

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            // Attention norm
            let attn_norm_w = timed!(ops, dequant, weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?);
            timed!(ops, norm, rms_norm::rms_norm_out(&hidden, attn_norm_w, &mut norm_buf, cfg.rms_norm_eps));

            // QKV projections
            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
            timed!(ops, matmul, { ctx.backend.batch_dequant_matvec(&[(wq_raw, wq_dtype, n_heads * head_dim), (wk_raw, wk_dtype, n_kv_heads * head_dim), (wv_raw, wv_dtype, n_kv_heads * head_dim)], &norm_buf, dim, &mut [&mut q_buf, &mut k_buf, &mut v_buf]); });

            // QKV bias
            if weights.has(&format!("{prefix}.attn_q.bias")) { let b = weights.dequantize(&format!("{prefix}.attn_q.bias"))?; silu::elementwise_add(&mut q_buf, &b); }
            if weights.has(&format!("{prefix}.attn_k.bias")) { let b = weights.dequantize(&format!("{prefix}.attn_k.bias"))?; silu::elementwise_add(&mut k_buf, &b); }
            if weights.has(&format!("{prefix}.attn_v.bias")) { let b = weights.dequantize(&format!("{prefix}.attn_v.bias"))?; silu::elementwise_add(&mut v_buf, &b); }

            // Per-head QK norm
            if weights.has(&format!("{prefix}.attn_q_norm.weight")) {
                let qnw = weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?;
                let knw = weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?;
                per_head_rms_norm(&mut q_buf, n_heads, head_dim, qnw, cfg.rms_norm_eps);
                per_head_rms_norm(&mut k_buf, n_kv_heads, head_dim, knw, cfg.rms_norm_eps);
            }

            // RoPE
            let rope_pos = match cfg.rope_scaling { crate::model::config::RopeScaling::Linear(f) => position as f32 / f, crate::model::config::RopeScaling::None => position as f32 };
            timed!(ops, rope, rope::apply_rope_multi_head_scaled(&mut q_buf, &mut k_buf, n_heads, n_kv_heads, head_dim, rope_pos, cfg.rope_freq_base));

            // KV cache + attention
            cpu_kv.append_and_advance(layer, &k_buf, &v_buf);
            let seq_len = cpu_kv.seq_len() + 1;
            timed!(ops, attention, attention::multi_head_attention(&q_buf, cpu_kv.k_slice_including_current(layer, seq_len), cpu_kv.v_slice_including_current(layer, seq_len), &mut attn_out, ctx.attn_params, seq_len));

            // Output projection
            let (wo_raw, wo_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            timed!(ops, matmul, ctx.backend.dequant_matmul(wo_raw, wo_dtype, &attn_out, &mut proj_buf, dim, 1, n_heads * head_dim));

            // Post-attention norm (GLM-specific)
            if weights.has(&format!("{prefix}.post_attention_norm.weight")) {
                let nw = weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
                timed!(ops, norm, rms_norm::rms_norm(&mut proj_buf, nw, cfg.rms_norm_eps));
            }

            silu::elementwise_add(&mut hidden, &proj_buf);

            // FFN norm
            let ffn_norm_w = timed!(ops, dequant, weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?);
            timed!(ops, norm, rms_norm::rms_norm_out(&hidden, ffn_norm_w, &mut norm_buf, cfg.rms_norm_eps));

            // Gate + Up → SiLU*mul → Down
            let (wg_raw, wg_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            timed!(ops, matmul, { ctx.backend.batch_dequant_matvec(&[(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)], &norm_buf, dim, &mut [&mut gate_buf, &mut up_buf]); });
            silu::silu_elementwise_mul(&mut gate_buf, &up_buf);

            let (wd_raw, wd_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            timed!(ops, matmul, ctx.backend.dequant_matmul(wd_raw, wd_dtype, &gate_buf, &mut down_buf, dim, 1, inter_dim));

            // Post-FFN norm (GLM-specific)
            if weights.has(&format!("{prefix}.post_ffw_norm.weight")) {
                let nw = weights.f32_slice(&format!("{prefix}.post_ffw_norm.weight"))?;
                timed!(ops, norm, rms_norm::rms_norm(&mut down_buf, nw, cfg.rms_norm_eps));
            }

            silu::elementwise_add(&mut hidden, &down_buf);
        }

        cpu_kv.finalize_token();

        let final_norm_w = timed!(ops, dequant, weights.f32_slice("output_norm.weight")?);
        timed!(ops, norm, rms_norm::rms_norm(&mut hidden, final_norm_w, cfg.rms_norm_eps));

        let lm_name = if weights.has("output.weight") { "output.weight" } else { "token_embd.weight" };
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_name)?;
        timed!(ops, matmul, ctx.backend.dequant_matmul(lm_raw, lm_dtype, &hidden, logits, vocab_size, 1, dim));
        Ok(())
    }

    fn forward_batch(&self, ctx: &ForwardContext, token_ids: &[u32], kv: &mut ModelKv, weights: &WeightStore, logits: &mut [f32]) -> anyhow::Result<()> {
        let force_serial = std::env::var("AX_SERIAL_PREFILL").is_ok();
        let can_gpu = !force_serial && ctx.backend.use_gpu_decode() && token_ids.len() > 1 && kv.as_gpu_mut().is_some() && gpu_quant_supported(weights);
        if can_gpu { /* TODO: batched GPU prefill for GLM */ }
        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() { logits.fill(0.0); self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, None)?; }
        Ok(())
    }

    fn validate_config(&self, _config: &ModelConfig) -> anyhow::Result<()> { Ok(()) }
    fn arch_name(&self) -> &str { "glm" }
    fn supports_pipelined_decode(&self, _ctx: &ForwardContext) -> bool { true }

    fn embed_pipelined_token(&self, ctx: &ForwardContext, token_id: u32, hidden_buf: &ax_metal::MetalBuffer, weights: &WeightStore) -> anyhow::Result<()> {
        let dim = ctx.config.embedding_dim as usize;
        let h = unsafe { std::slice::from_raw_parts_mut(hidden_buf.contents().as_ptr() as *mut f32, dim) };
        weights.dequantize_row("token_embd.weight", token_id as usize, h).map(|_| ())
    }

    fn encode_pending_decode_step(&self, ctx: &ForwardContext, hidden_buf: &ax_metal::MetalBuffer, position: usize, kv: &mut ModelKv, weights: &WeightStore) -> anyhow::Result<Option<ax_metal::PendingFrame>> {
        let Some(metal_ops) = ctx.backend.metal_ops() else { return Ok(None) };
        let Some(gpu_kv) = kv.as_gpu_mut() else { return Ok(None) };
        Ok(Some(encode_glm_pending_step(metal_ops, ctx.config, hidden_buf, position, gpu_kv, weights)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glm_forward_arch_name() {
        let fwd = GlmForward;
        assert_eq!(fwd.arch_name(), "glm");
    }
}
