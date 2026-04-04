struct Qwen3DecodeStrategy<'a> {
    cfg: &'a ModelConfig,
    kv_offset: u32,
    rope_position: f32,
    full_seq_len: usize,
    layer_plan: &'a Qwen3DecodeLayerPlan,
    dims: &'a crate::model::shared::GpuLayerDims,
}

impl crate::model::shared::GpuDecodeLayerStrategy for Qwen3DecodeStrategy<'_> {
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

        // --- QKV post-processing ---
        let kv_k = gpu_kv.k_buffer(layer);
        let kv_v = gpu_kv.v_buffer(layer);
        if used_fused_qkv {
            barrier.pre_dispatch(
                &[&s.qkv_buf],
                &[&s.q_buf, &s.k_buf, &s.v_buf, kv_k, kv_v],
            );
            match self.layer_plan.qwen3_post {
                Qwen3PrefillQkvPost::FusedBiasQkNorm => {
                    let qb_buf = weight_cache.get(&lw.q_bias.unwrap()).unwrap();
                    let kb_buf = weight_cache.get(&lw.k_bias.unwrap()).unwrap();
                    let vb_buf = weight_cache.get(&lw.v_bias.unwrap()).unwrap();
                    let q_nw = weight_cache.get(&lw.attn_q_norm.unwrap()).unwrap();
                    let k_nw = weight_cache.get(&lw.attn_k_norm.unwrap()).unwrap();
                    metal_ops
                        .elementwise
                        .encode_qkv_split_bias_qknorm_rope_append_kv_batch(
                            encoder,
                            &s.qkv_buf,
                            &s.q_buf,
                            &s.k_buf,
                            &s.v_buf,
                            q_nw,
                            k_nw,
                            kv_k,
                            kv_v,
                            exec_plan.kv_f16,
                            qb_buf,
                            kb_buf,
                            vb_buf,
                            1,
                            d.n_heads,
                            d.n_kv_heads,
                            d.head_dim,
                            eps,
                            self.rope_position,
                            1.0,
                            self.cfg.rope_freq_base,
                            self.kv_offset,
                            d.kv_dim,
                        );
                }
                Qwen3PrefillQkvPost::FusedQkNorm => {
                    let q_nw = weight_cache.get(&lw.attn_q_norm.unwrap()).unwrap();
                    let k_nw = weight_cache.get(&lw.attn_k_norm.unwrap()).unwrap();
                    metal_ops
                        .elementwise
                        .encode_qkv_split_qk_norm_rope_append_kv_batch(
                            encoder,
                            &s.qkv_buf,
                            &s.q_buf,
                            &s.k_buf,
                            &s.v_buf,
                            q_nw,
                            k_nw,
                            kv_k,
                            kv_v,
                            exec_plan.kv_f16,
                            1,
                            d.n_heads,
                            d.n_kv_heads,
                            d.head_dim,
                            eps,
                            self.rope_position,
                            1.0,
                            self.cfg.rope_freq_base,
                            self.kv_offset,
                            d.kv_dim,
                        );
                }
                _ => unreachable!("qwen3 fused decode path requires fused qwen3 post plan"),
            }
            barrier.post_dispatch(
                &[&s.qkv_buf],
                &[&s.q_buf, &s.k_buf, &s.v_buf, kv_k, kv_v],
            );
            barrier.step(encoder);
        } else {
            // Separate path: optional bias → optional QK norm → RoPE → KV append
            if let (Some(qb_key), Some(kb_key), Some(vb_key)) = (lw.q_bias, lw.k_bias, lw.v_bias) {
                let qb = weight_cache.get(&qb_key).unwrap();
                let kb = weight_cache.get(&kb_key).unwrap();
                let vb = weight_cache.get(&vb_key).unwrap();
                barrier.pre_dispatch(&[&s.q_buf], &[&s.q_buf]);
                metal_ops
                    .elementwise
                    .encode_elementwise_add(encoder, &s.q_buf, qb, d.q_dim);
                barrier.post_dispatch(&[&s.q_buf], &[&s.q_buf]);
                barrier.pre_dispatch(&[&s.k_buf], &[&s.k_buf]);
                metal_ops
                    .elementwise
                    .encode_elementwise_add(encoder, &s.k_buf, kb, d.kv_dim);
                barrier.post_dispatch(&[&s.k_buf], &[&s.k_buf]);
                barrier.pre_dispatch(&[&s.v_buf], &[&s.v_buf]);
                metal_ops
                    .elementwise
                    .encode_elementwise_add(encoder, &s.v_buf, vb, d.kv_dim);
                barrier.post_dispatch(&[&s.v_buf], &[&s.v_buf]);
                barrier.step(encoder);
            }
            if let (Some(qn_key), Some(kn_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
                let qn = weight_cache.get(&qn_key).unwrap();
                let kn = weight_cache.get(&kn_key).unwrap();
                barrier.pre_dispatch(&[&s.q_buf], &[&s.q_buf]);
                metal_ops
                    .elementwise
                    .encode_per_head_rms_norm(encoder, &s.q_buf, qn, d.n_heads, d.head_dim, eps);
                barrier.post_dispatch(&[&s.q_buf], &[&s.q_buf]);
                barrier.pre_dispatch(&[&s.k_buf], &[&s.k_buf]);
                metal_ops.elementwise.encode_per_head_rms_norm(
                    encoder,
                    &s.k_buf,
                    kn,
                    d.n_kv_heads,
                    d.head_dim,
                    eps,
                );
                barrier.post_dispatch(&[&s.k_buf], &[&s.k_buf]);
                barrier.step(encoder);
            }
            barrier.pre_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
            metal_ops.elementwise.encode_rope(
                encoder,
                &s.q_buf,
                &s.k_buf,
                d.n_heads,
                d.n_kv_heads,
                d.head_dim,
                d.head_dim,
                self.rope_position,
                self.cfg.rope_freq_base,
            );
            barrier.post_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
            barrier.step(encoder);
            let kv_k = gpu_kv.k_buffer(layer);
            let kv_v = gpu_kv.v_buffer(layer);
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

        // --- Attention ---
        barrier.pre_dispatch(&[&s.q_buf, kv_k, kv_v], &[&s.attn_out]);
        if exec_plan.kv_q8 {
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

        // --- WO projection ---
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

        // --- Residual + FFN norm ---
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
fn encode_qwen3_gpu_layers_only(
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
        .map(|lw| {
            qwen3_layer_plan_for_gpu(
                exec_plan,
                lw.wq_dtype,
                lw.wk_dtype,
                lw.wv_dtype,
                lw.q_bias.is_some() && lw.k_bias.is_some() && lw.v_bias.is_some(),
                lw.attn_q_norm.is_some() && lw.attn_k_norm.is_some(),
            )
        })
        .collect();
    let barrier = crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);

    // Layer 0 starts with a standalone attention norm. Later layers reuse the
    // next-layer handoff written by the previous FFN residual stage.
    {
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

    for (layer, layer_plan) in layer_plans.iter().enumerate().take(n_layers) {
        let lw = &cached.layers[layer];
        let strategy = Qwen3DecodeStrategy {
            cfg,
            kv_offset,
            rope_position,
            full_seq_len,
            layer_plan,
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
            &barrier,
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen3_gpu_output_head(
    encoder: &ax_engine_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_engine_metal::MetalBuffer,
    cfg: &ModelConfig,
    exec_plan: &GpuDecodeExecutionPlan,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
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

fn encode_qwen3_pending_step(
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
        Qwen3Forward::build_cached_model_keys_qwen3(metal_ops, weights, cfg)?;
    }

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();

    let kv_offset = (position * kv_dim) as u32;
    let full_seq_len = position + 1;
    let exec_plan = DecodeExecutionPlan::qwen3_pipelined(
        metal_ops,
        gpu_kv,
        cfg.embedding_dim,
        cfg.head_dim,
        full_seq_len,
    );
    let rope_position = cfg.rope_scaling.scaled_position(position);

    let weight_cache = metal_ops.lock_weight_cache();
    let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

    let encode_body = |encoder: &ax_engine_metal::MetalEncoder| -> anyhow::Result<()> {
        let barrier = crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
        encode_qwen3_gpu_layers_only(
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
        )?;
        encode_qwen3_gpu_output_head(
            encoder,
            metal_ops,
            s,
            hidden_buf,
            cfg,
            &exec_plan,
            cached,
            &weight_cache,
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
