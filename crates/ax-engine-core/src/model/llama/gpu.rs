/// Metal compute encoder.
///
/// Extracted from `forward_single_gpu_unified` so both the synchronous path
/// (execute_sync) and the pipelined path (encode_frame) can share the same
/// GPU command sequence.
///
/// `hidden_buf` replaces `s.hidden` as the running accumulator, allowing the
/// pipelined path to double-buffer the hidden state across tokens.
/// LLaMA GPU decode strategy: plain split + RoPE + KV append (no bias, no QK norm).
struct LlamaDecodeStrategy<'a> {
    cfg: &'a ModelConfig,
    kv_offset: u32,
    rope_position: f32,
    full_seq_len: usize,
    dims: &'a crate::model::shared::GpuLayerDims,
}

impl crate::model::shared::GpuDecodeLayerStrategy for LlamaDecodeStrategy<'_> {
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
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        used_fused_qkv: bool,
    ) {
        let d = self.dims;
        let eps = d.eps;
        let kv_k = gpu_kv.k_buffer(layer);
        let kv_v = gpu_kv.v_buffer(layer);

        if used_fused_qkv {
            // Fused: split + RoPE + KV append in one kernel
            barrier.pre_dispatch(
                &[&s.qkv_buf],
                &[&s.q_buf, &s.k_buf, &s.v_buf, kv_k, kv_v],
            );
            metal_ops.elementwise.encode_qkv_split_rope_append_kv_batch(
                encoder,
                &s.qkv_buf,
                &s.q_buf,
                &s.k_buf,
                &s.v_buf,
                kv_k,
                kv_v,
                exec_plan.kv_f16,
                1,
                d.n_heads,
                d.n_kv_heads,
                d.head_dim,
                self.rope_position,
                1.0,
                self.cfg.rope_freq_base,
                self.kv_offset,
                d.kv_dim,
            );
            barrier.post_dispatch(
                &[&s.qkv_buf],
                &[&s.q_buf, &s.k_buf, &s.v_buf, kv_k, kv_v],
            );
            barrier.step(encoder);
        } else {
            // Separate: RoPE then KV append
            barrier.pre_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
            metal_ops.elementwise.encode_rope(
                encoder,
                &s.q_buf,
                &s.k_buf,
                d.n_heads,
                d.n_kv_heads,
                d.head_dim,
                self.rope_position,
                self.cfg.rope_freq_base,
            );
            barrier.post_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
            barrier.step(encoder);
            // K/V appends write to different KV buffers → can overlap.
            barrier.pre_dispatch(&[&s.k_buf], &[kv_k]);
            metal_ops.elementwise.encode_kv_append(
                encoder,
                &s.k_buf,
                kv_k,
                exec_plan.kv_f16,
                self.kv_offset,
                d.kv_dim,
            );
            barrier.post_dispatch(&[&s.k_buf], &[kv_k]);
            barrier.pre_dispatch(&[&s.v_buf], &[kv_v]);
            metal_ops.elementwise.encode_kv_append(
                encoder,
                &s.v_buf,
                kv_v,
                exec_plan.kv_f16,
                self.kv_offset,
                d.kv_dim,
            );
            barrier.post_dispatch(&[&s.v_buf], &[kv_v]);
            barrier.step(encoder);
        }

        // Attention
        barrier.pre_dispatch(&[&s.q_buf, kv_k, kv_v], &[&s.attn_out]);
        if exec_plan.kv_q4 {
            metal_ops.attention.encode_attention_decode_q4kv(
                encoder,
                &s.q_buf,
                kv_k,
                kv_v,
                &s.attn_out,
                d.n_heads,
                d.n_kv_heads,
                d.head_dim,
                0,
                self.full_seq_len as u32,
            );
        } else if exec_plan.kv_q8 {
            if d.head_dim == 128 {
                metal_ops.attention.encode_attention_decode_q8kv(
                    encoder,
                    &s.q_buf,
                    kv_k,
                    kv_v,
                    &s.attn_out,
                    d.n_heads,
                    d.n_kv_heads,
                    d.head_dim,
                    0,
                    self.full_seq_len as u32,
                );
            } else if d.head_dim == 256 {
                metal_ops.attention.encode_attention_decode_q8kv_hd256(
                    encoder,
                    &s.q_buf,
                    kv_k,
                    kv_v,
                    &s.attn_out,
                    d.n_heads,
                    d.n_kv_heads,
                    d.head_dim,
                    0,
                    self.full_seq_len as u32,
                );
            }
        } else {
            metal_ops
                .attention
                .encode_attention_decode_with_scratch_and_config(
                    encoder,
                    &s.q_buf,
                    kv_k,
                    kv_v,
                    &s.attn_out,
                    &s.splitk_partial_out,
                    &s.splitk_partial_lse,
                    exec_plan.kv_f16,
                    d.n_heads,
                    d.n_kv_heads,
                    d.head_dim,
                    0,
                    self.full_seq_len as u32,
                    exec_plan.attention_dispatch,
                );
        }
        barrier.post_dispatch(&[&s.q_buf, kv_k, kv_v], &[&s.attn_out]);
        barrier.step(encoder);

        // WO projection
        let wo_buf = weight_cache.get(&lw.wo).unwrap();
        barrier.pre_dispatch(&[&s.attn_out], &[&s.proj_buf]);
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
        barrier.post_dispatch(&[&s.attn_out], &[&s.proj_buf]);
        barrier.step(encoder);

        // Residual + FFN norm
        let ffn_nw = weight_cache.get(&lw.ffn_norm).unwrap();
        barrier.pre_dispatch(&[hidden_buf, &s.proj_buf], &[hidden_buf, &s.norm_buf]);
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
        barrier.post_dispatch(&[hidden_buf, &s.proj_buf], &[hidden_buf, &s.norm_buf]);
        barrier.step(encoder);
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_llama_gpu_layers_only(
    encoder: &ax_engine_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_engine_metal::MetalBuffer,
    cfg: &ModelConfig,
    kv_offset: u32,
    rope_position: f32,
    full_seq_len: usize,
    exec_plan: &GpuDecodeExecutionPlan,
    gpu_kv: &crate::kv::GpuKv,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
    fused_qkv_cache: &rustc_hash::FxHashMap<(usize, usize, usize), ax_engine_metal::MetalBuffer>,
    mut ops: Option<&mut OpBreakdown>,
    barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
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
    // Layer 0 starts with a standalone attn norm. Subsequent layers receive
    // their attn norm from the previous layer's fused FFN residual handoff.
    {
        let t = OpTimer::start();
        let norm_w_buf = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
        barrier.pre_dispatch(&[hidden_buf], &[&s.norm_buf]);
        metal_ops.elementwise.encode_rms_norm_out(
            encoder,
            hidden_buf,
            norm_w_buf,
            &s.norm_buf,
            dim as u32,
            eps,
        );
        barrier.post_dispatch(&[hidden_buf], &[&s.norm_buf]);
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode_layer_norm += t.elapsed();
        }
        barrier.step(encoder);
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
        let strategy = LlamaDecodeStrategy {
            cfg,
            kv_offset,
            rope_position,
            full_seq_len,
            dims: &gpu_dims,
        };
        let next_attn_norm = if layer + 1 < n_layers {
            Some(cached.layers[layer + 1].attn_norm)
        } else {
            None
        };
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
            crate::model::layer_ops::FfnActivation::SiLU,
            None,
            barrier,
        );
        // NOTE: per-op GPU timing (OpTimer) is not available when using the
        // shared decode layer path. The pipelined path (encode_llama_gpu_layer_sequence)
        // retains individual timing and is the primary production path.
        let _ = &ops;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_llama_gpu_output_head(
    encoder: &ax_engine_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_engine_metal::MetalBuffer,
    cfg: &ModelConfig,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
    exec_plan: &GpuDecodeExecutionPlan,
    barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
) {
    crate::model::shared::encode_gpu_output_head(
        encoder,
        metal_ops,
        s,
        hidden_buf,
        exec_plan,
        cached,
        weight_cache,
        barrier,
        cfg.embedding_dim,
        cfg.vocab_size,
        cfg.rms_norm_eps,
    );
}

/// Encode a LLaMA single-token decode step into a [`ax_engine_metal::PendingFrame`]
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
    hidden_buf: &ax_engine_metal::MetalBuffer,
    position: usize,
    gpu_kv: &mut crate::kv::GpuKv,
    weights: &WeightStore,
) -> anyhow::Result<ax_engine_metal::PendingFrame> {
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
        DecodeScratchPlan::CpuScratch | DecodeScratchPlan::HybridBackendOwned => {
            anyhow::bail!("pipelined GPU decode requires GPU scratch")
        }
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
    let rope_position = cfg.rope_scaling.scaled_position(position);

    let weight_cache = metal_ops.lock_weight_cache();
    let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

    let encode_body = |encoder: &ax_engine_metal::MetalEncoder| -> anyhow::Result<()> {
        let barrier =
            crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
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
            &barrier,
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
            &barrier,
        );
        barrier.flush();
        Ok(())
    };
    if exec_plan.encoder == DecodeEncoderPlan::Concurrent {
        metal_ops.device.encode_frame_concurrent(encode_body)
    } else {
        metal_ops.device.encode_frame(encode_body)
    }
}
