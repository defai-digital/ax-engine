impl Qwen35Forward {
    fn encode_qwen35_pending_step(
        metal_ops: &crate::backend::metal::MetalOps,
        cfg: &ModelConfig,
        hidden_buf: &MetalBuffer,
        position: usize,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        weights: &WeightStore,
    ) -> anyhow::Result<ax_engine_metal::PendingFrame> {
        Self::encode_qwen35_pending_step_inner(
            metal_ops, cfg, hidden_buf, position, qwen_kv, weights, false,
        )
    }

    fn encode_qwen35_pending_step_with_argmax(
        metal_ops: &crate::backend::metal::MetalOps,
        cfg: &ModelConfig,
        hidden_buf: &MetalBuffer,
        position: usize,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        weights: &WeightStore,
    ) -> anyhow::Result<ax_engine_metal::PendingFrame> {
        Self::encode_qwen35_pending_step_inner(
            metal_ops, cfg, hidden_buf, position, qwen_kv, weights, true,
        )
    }

    fn encode_qwen35_pending_step_inner(
        metal_ops: &crate::backend::metal::MetalOps,
        cfg: &ModelConfig,
        hidden_buf: &MetalBuffer,
        position: usize,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        weights: &WeightStore,
        fuse_argmax: bool,
    ) -> anyhow::Result<ax_engine_metal::PendingFrame> {
        use crate::model::shared::encode_dequant_matvec_with_config;

        let dims = Self::recurrent_dims(cfg)?;
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let recurrent_slot = qwen_kv.active_slot();
        let eps = cfg.rms_norm_eps;
        let conv_cache_len = qwen_kv.conv_cache_len();
        let full_seq_len = position + 1;

        anyhow::ensure!(
            qwen_kv.ensure_gpu_attention_capacity_for(full_seq_len),
            "qwen35 pending decode requires GPU attention KV capacity"
        );

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_qwen35(metal_ops, weights, cfg)?;
        }
        metal_ops.init_scratches(cfg);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let gpu_layer_keys = Self::cached_gpu_layer_keys(cached.lm_head)
            .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 gpu layer keys"))?;
        let weight_cache = metal_ops.lock_weight_cache();

        let exec_plan = Self::qwen35_decode_plan(
            metal_ops,
            qwen_kv.gpu_attention().ok_or_else(|| {
                anyhow::anyhow!("qwen35 pending decode requires GPU attention KV")
            })?,
            cfg.embedding_dim,
            cfg.head_dim,
            full_seq_len,
            true,
        );
        let rope_position = Self::rope_position(cfg, position);
        let kv_offset = (position * kv_dim) as u32;
        let use_concurrent = exec_plan.encoder
            == crate::model::execution_plan::DecodeEncoderPlan::Concurrent;
        // Macro to avoid duplicating the entire closure for serial vs concurrent.
        macro_rules! with_encoder {
            ($device:expr, $body:expr) => {
                if use_concurrent {
                    $device.encode_frame_concurrent($body)
                } else {
                    $device.encode_frame($body)
                }
            };
        }
        with_encoder!(metal_ops.device, |encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                exec_plan.barriers,
            );
            let norm_w = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
            barrier.pre_dispatch(&[hidden_buf], &[&s.norm_buf]);
            metal_ops.elementwise.encode_rms_norm_out(
                encoder,
                hidden_buf,
                norm_w,
                &s.norm_buf,
                dim as u32,
                eps,
            );
            barrier.post_dispatch(&[hidden_buf], &[&s.norm_buf]);
            barrier.step(encoder);

            for layer in 0..n_layers {
                let lw = &cached.layers[layer];
                let next_norm = if layer + 1 < n_layers {
                    Some(
                        weight_cache
                            .get(&cached.layers[layer + 1].attn_norm)
                            .unwrap(),
                    )
                } else {
                    None
                };

                if !cfg.qwen35_is_recurrent_layer(layer) {
                    let gpu_attn = qwen_kv.gpu_attention().ok_or_else(|| {
                        anyhow::anyhow!("qwen35 pending decode requires GPU attention KV")
                    })?;
                    // Q/K/V matvecs all read norm_buf, write different outputs
                    // → SmartBarrier skips barriers between them.
                    let wq = weight_cache.get(&lw.wq).unwrap();
                    let wk = weight_cache.get(&lw.wk).unwrap();
                    let wv = weight_cache.get(&lw.wv).unwrap();
                    barrier.pre_dispatch(&[&s.norm_buf], &[&s.gate_buf]);
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wq,
                        &s.norm_buf,
                        &s.gate_buf,
                        (q_dim * 2) as u32,
                        dim as u32,
                        lw.wq_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    barrier.post_dispatch(&[&s.norm_buf], &[&s.gate_buf]);
                    barrier.pre_dispatch(&[&s.norm_buf], &[&s.k_buf]);
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wk,
                        &s.norm_buf,
                        &s.k_buf,
                        kv_dim as u32,
                        dim as u32,
                        lw.wk_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    barrier.post_dispatch(&[&s.norm_buf], &[&s.k_buf]);
                    barrier.pre_dispatch(&[&s.norm_buf], &[&s.v_buf]);
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wv,
                        &s.norm_buf,
                        &s.v_buf,
                        kv_dim as u32,
                        dim as u32,
                        lw.wv_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    barrier.post_dispatch(&[&s.norm_buf], &[&s.v_buf]);
                    barrier.pre_dispatch(&[&s.gate_buf], &[&s.q_buf, &s.up_buf]);
                    metal_ops.elementwise.encode_split_qgate_batch(
                        encoder,
                        &s.gate_buf,
                        &s.q_buf,
                        &s.up_buf,
                        1,
                        q_dim as u32,
                    );
                    barrier.post_dispatch(&[&s.gate_buf], &[&s.q_buf, &s.up_buf]);
                    barrier.step(encoder);

                    if let (Some(q_key), Some(k_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
                        let q_nw = weight_cache.get(&q_key).unwrap();
                        let k_nw = weight_cache.get(&k_key).unwrap();
                        barrier.pre_dispatch(&[&s.q_buf], &[&s.q_buf]);
                        metal_ops.elementwise.encode_per_head_rms_norm_batch(
                            encoder,
                            &s.q_buf,
                            q_nw,
                            1,
                            n_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        barrier.post_dispatch(&[&s.q_buf], &[&s.q_buf]);
                        barrier.pre_dispatch(&[&s.k_buf], &[&s.k_buf]);
                        metal_ops.elementwise.encode_per_head_rms_norm_batch(
                            encoder,
                            &s.k_buf,
                            k_nw,
                            1,
                            n_kv_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        barrier.post_dispatch(&[&s.k_buf], &[&s.k_buf]);
                        barrier.step(encoder);
                    }

                    barrier.pre_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
                    metal_ops.elementwise.encode_rope_batch(
                        encoder,
                        &s.q_buf,
                        &s.k_buf,
                        1,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        rope_position,
                        0.0,
                        cfg.rope_freq_base,
                    );
                    barrier.post_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
                    barrier.step(encoder);

                    // K/V appends write different KV buffers → can overlap.
                    let kv_k = gpu_attn.k_buffer(layer);
                    let kv_v = gpu_attn.v_buffer(layer);
                    barrier.pre_dispatch(&[&s.k_buf], &[kv_k]);
                    metal_ops.elementwise.encode_kv_append(
                        encoder,
                        &s.k_buf,
                        kv_k,
                        gpu_attn.is_f16(),
                        kv_offset,
                        kv_dim as u32,
                    );
                    barrier.post_dispatch(&[&s.k_buf], &[kv_k]);
                    barrier.pre_dispatch(&[&s.v_buf], &[kv_v]);
                    metal_ops.elementwise.encode_kv_append(
                        encoder,
                        &s.v_buf,
                        kv_v,
                        gpu_attn.is_f16(),
                        kv_offset,
                        kv_dim as u32,
                    );
                    barrier.post_dispatch(&[&s.v_buf], &[kv_v]);
                    barrier.step(encoder);

                    barrier.pre_dispatch(&[&s.q_buf, kv_k, kv_v], &[&s.attn_out]);
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
                            gpu_attn.is_f16(),
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            0,
                            full_seq_len as u32,
                            exec_plan.attention_dispatch,
                        );
                    barrier.post_dispatch(&[&s.q_buf, kv_k, kv_v], &[&s.attn_out]);
                    barrier.step(encoder);

                    barrier.pre_dispatch(&[&s.up_buf, &s.attn_out], &[&s.attn_out]);
                    metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                        encoder,
                        &s.up_buf,
                        &s.attn_out,
                        q_dim as u32,
                    );
                    barrier.post_dispatch(&[&s.up_buf, &s.attn_out], &[&s.attn_out]);
                    barrier.step(encoder);

                    barrier.pre_dispatch(&[&s.attn_out], &[&s.proj_buf]);
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        weight_cache.get(&lw.wo).unwrap(),
                        &s.attn_out,
                        &s.proj_buf,
                        dim as u32,
                        q_dim as u32,
                        lw.wo_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    barrier.post_dispatch(&[&s.attn_out], &[&s.proj_buf]);
                    barrier.step(encoder);

                    let ffn_nw = weight_cache.get(&lw.ffn_norm).unwrap();
                    barrier.pre_dispatch(
                        &[hidden_buf, &s.proj_buf],
                        &[hidden_buf, &s.norm_buf],
                    );
                    metal_ops
                        .elementwise
                        .encode_residual_add_rms_norm_out_batch(
                            encoder,
                            hidden_buf,
                            &s.proj_buf,
                            ffn_nw,
                            &s.norm_buf,
                            dim as u32,
                            1,
                            eps,
                        );
                    barrier.post_dispatch(
                        &[hidden_buf, &s.proj_buf],
                        &[hidden_buf, &s.norm_buf],
                    );
                    barrier.step(encoder);

                    // Gate/Up: both read norm_buf, write different outputs → can overlap.
                    let wg = weight_cache.get(&lw.wg).unwrap();
                    let wu = weight_cache.get(&lw.wu).unwrap();
                    barrier.pre_dispatch(&[&s.norm_buf], &[&s.gate_buf, &s.up_buf]);
                    if !crate::model::shared::encode_dequant_matvec_pair_with_config(
                        metal_ops,
                        encoder,
                        wg,
                        wu,
                        &s.norm_buf,
                        &s.gate_buf,
                        &s.up_buf,
                        inter_dim as u32,
                        dim as u32,
                        lw.wg_dtype,
                        lw.wu_dtype,
                        exec_plan.dequant_dispatch,
                        exec_plan.use_pair_matvec,
                    ) {
                        encode_dequant_matvec_with_config(
                            metal_ops,
                            encoder,
                            wg,
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
                            wu,
                            &s.norm_buf,
                            &s.up_buf,
                            inter_dim as u32,
                            dim as u32,
                            lw.wu_dtype,
                            exec_plan.dequant_dispatch,
                        );
                    }
                    barrier.post_dispatch(&[&s.norm_buf], &[&s.gate_buf, &s.up_buf]);
                    barrier.step(encoder);

                    crate::model::shared::encode_gpu_ffn_decode_tail(
                        metal_ops,
                        encoder,
                        s,
                        hidden_buf,
                        weight_cache.get(&lw.wd).unwrap(),
                        lw.wd_dtype,
                        dim as u32,
                        inter_dim as u32,
                        eps,
                        exec_plan.dequant_dispatch,
                        exec_plan.use_fused_silu_down,
                        crate::model::layer_ops::FfnActivation::SiLU,
                        None,
                        next_norm,
                        &barrier,
                    );
                } else {
                    let recurrent_keys = match &gpu_layer_keys[layer] {
                        Qwen35GpuLayerKeys::Recurrent(keys) => keys,
                        Qwen35GpuLayerKeys::FullAttention => {
                            anyhow::bail!("expected recurrent qwen35 GPU keys for layer {layer}")
                        }
                    };
                    debug_assert!(kv_dim >= dims.time_step_rank);
                    let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
                    let recurrent_state_stride = qwen_kv.recurrent_state_len();
                    metal_ops.with_qwen35_recurrent_slot_buffer_for_kv(
                        qwen_kv,
                        layer,
                        recurrent_slot,
                        conv_state_stride,
                        recurrent_state_stride,
                        |slot_buffers| {
                            let norm_w = weight_cache.get(&lw.attn_norm).unwrap();
                            barrier.pre_dispatch(&[hidden_buf], &[&s.norm_buf]);
                            metal_ops.elementwise.encode_rms_norm_out(
                                encoder, hidden_buf, norm_w, &s.norm_buf,
                                dim as u32, eps,
                            );
                            barrier.post_dispatch(&[hidden_buf], &[&s.norm_buf]);
                            barrier.step(encoder);

                            // Recurrent projections: all read norm_buf, write
                            // different outputs → can overlap.
                            barrier.pre_dispatch(&[&s.norm_buf], &[&s.gate_buf]);
                            encode_dequant_matvec_with_config(
                                metal_ops, encoder,
                                weight_cache.get(&recurrent_keys.wqkv).unwrap(),
                                &s.norm_buf, &s.gate_buf,
                                dims.conv_dim() as u32, dim as u32,
                                recurrent_keys.wqkv_dtype, exec_plan.dequant_dispatch,
                            );
                            barrier.post_dispatch(&[&s.norm_buf], &[&s.gate_buf]);
                            barrier.pre_dispatch(&[&s.norm_buf], &[&s.attn_out]);
                            encode_dequant_matvec_with_config(
                                metal_ops, encoder,
                                weight_cache.get(&recurrent_keys.wgate).unwrap(),
                                &s.norm_buf, &s.attn_out,
                                dims.inner_size as u32, dim as u32,
                                recurrent_keys.wgate_dtype, exec_plan.dequant_dispatch,
                            );
                            barrier.post_dispatch(&[&s.norm_buf], &[&s.attn_out]);
                            let wbeta = weight_cache.get(&recurrent_keys.wbeta).unwrap();
                            let walpha = weight_cache.get(&recurrent_keys.walpha).unwrap();
                            barrier.pre_dispatch(&[&s.norm_buf], &[&s.v_buf, &s.proj_buf]);
                            if !Self::encode_recurrent_pair_matvec_with_config(
                                metal_ops, encoder, wbeta, walpha,
                                &s.norm_buf, &s.v_buf, &s.proj_buf,
                                dims.time_step_rank as u32, dim as u32,
                                recurrent_keys.wbeta_dtype, recurrent_keys.walpha_dtype,
                            ) {
                                encode_dequant_matvec_with_config(
                                    metal_ops, encoder, wbeta, &s.norm_buf, &s.v_buf,
                                    dims.time_step_rank as u32, dim as u32,
                                    recurrent_keys.wbeta_dtype, exec_plan.dequant_dispatch,
                                );
                                encode_dequant_matvec_with_config(
                                    metal_ops, encoder, walpha, &s.norm_buf, &s.proj_buf,
                                    dims.time_step_rank as u32, dim as u32,
                                    recurrent_keys.walpha_dtype, exec_plan.dequant_dispatch,
                                );
                            }
                            barrier.post_dispatch(&[&s.norm_buf], &[&s.v_buf, &s.proj_buf]);
                            barrier.step(encoder);

                            // softplus(beta) and sigmoid(alpha) operate on different bufs.
                            let dt_bias_buf = weight_cache.get(&recurrent_keys.dt_bias).unwrap();
                            let ssm_a_buf = weight_cache.get(&recurrent_keys.ssm_a).unwrap();
                            barrier.pre_dispatch(&[&s.proj_buf], &[&s.proj_buf]);
                            metal_ops.elementwise.encode_softplus_bias_mul(
                                encoder, &s.proj_buf, dt_bias_buf, ssm_a_buf,
                                dims.time_step_rank as u32,
                            );
                            barrier.post_dispatch(&[&s.proj_buf], &[&s.proj_buf]);
                            barrier.pre_dispatch(&[&s.v_buf], &[&s.v_buf]);
                            metal_ops.elementwise.encode_sigmoid_inplace(
                                encoder, &s.v_buf, dims.time_step_rank as u32,
                            );
                            barrier.post_dispatch(&[&s.v_buf], &[&s.v_buf]);
                            barrier.step(encoder);

                            barrier.pre_dispatch(
                                &[&s.gate_buf, &slot_buffers.conv_state],
                                &[&s.up_buf, &slot_buffers.conv_state],
                            );
                            metal_ops.gdn.encode_causal_conv_sequence(
                                encoder, &s.gate_buf,
                                weight_cache.get(&recurrent_keys.conv_kernel).unwrap(),
                                &slot_buffers.conv_state, &s.up_buf,
                                1, conv_cache_len as u32, dims.conv_dim() as u32,
                            );
                            barrier.post_dispatch(
                                &[&s.gate_buf, &slot_buffers.conv_state],
                                &[&s.up_buf, &slot_buffers.conv_state],
                            );
                            barrier.step(encoder);

                            barrier.pre_dispatch(
                                &[&s.up_buf, &s.proj_buf, &s.v_buf, &slot_buffers.recurrent_state],
                                &[&s.q_buf, &slot_buffers.recurrent_state],
                            );
                            if metal_ops.gdn.encode_single_token_gated_delta_fused(
                                encoder, &s.up_buf, &s.proj_buf, &s.v_buf,
                                &slot_buffers.recurrent_state, &s.q_buf,
                                dims.group_count as u32, dims.time_step_rank as u32,
                                dims.state_size as u32, eps,
                            ) {
                                barrier.post_dispatch(
                                    &[&s.up_buf, &s.proj_buf, &s.v_buf, &slot_buffers.recurrent_state],
                                    &[&s.q_buf, &slot_buffers.recurrent_state],
                                );
                                barrier.step(encoder);
                            } else {
                                barrier.post_dispatch(
                                    &[&s.up_buf], &[&s.q_buf, &s.norm_buf, &s.down_buf],
                                );
                                metal_ops.gdn.encode_prepare_single_token_qkv(
                                    encoder, &s.up_buf, &s.q_buf, &s.norm_buf, &s.down_buf,
                                    dims.group_count as u32, dims.time_step_rank as u32,
                                    dims.state_size as u32, eps,
                                );
                                barrier.post_dispatch(
                                    &[&s.up_buf], &[&s.q_buf, &s.norm_buf, &s.down_buf],
                                );
                                barrier.step(encoder);

                                barrier.pre_dispatch(
                                    &[&s.q_buf, &s.norm_buf, &s.down_buf, &s.proj_buf,
                                      &s.v_buf, &slot_buffers.recurrent_state],
                                    &[&s.q_buf, &slot_buffers.recurrent_state],
                                );
                                metal_ops.gdn.encode_gated_delta_sequence(
                                    encoder, &s.q_buf, &s.norm_buf, &s.down_buf,
                                    &s.proj_buf, &s.v_buf, &slot_buffers.recurrent_state,
                                    &s.q_buf, 1, dims.time_step_rank as u32,
                                    dims.state_size as u32,
                                );
                                barrier.post_dispatch(
                                    &[&s.q_buf, &s.norm_buf, &s.down_buf, &s.proj_buf,
                                      &s.v_buf, &slot_buffers.recurrent_state],
                                    &[&s.q_buf, &slot_buffers.recurrent_state],
                                );
                                barrier.step(encoder);
                            }

                            let ssm_norm = weight_cache.get(&recurrent_keys.ssm_norm).unwrap();
                            barrier.pre_dispatch(&[&s.q_buf], &[&s.q_buf]);
                            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                                encoder, &s.q_buf, ssm_norm, 1,
                                dims.time_step_rank as u32, dims.state_size as u32, eps,
                            );
                            barrier.post_dispatch(&[&s.q_buf], &[&s.q_buf]);
                            barrier.step(encoder);

                            barrier.pre_dispatch(&[&s.attn_out, &s.q_buf], &[&s.attn_out]);
                            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                                encoder, &s.attn_out, &s.q_buf,
                                dims.inner_size as u32, 1,
                            );
                            barrier.post_dispatch(&[&s.attn_out, &s.q_buf], &[&s.attn_out]);
                            barrier.step(encoder);

                            barrier.pre_dispatch(&[&s.attn_out], &[&s.proj_buf]);
                            encode_dequant_matvec_with_config(
                                metal_ops, encoder,
                                weight_cache.get(&recurrent_keys.wssm_out).unwrap(),
                                &s.attn_out, &s.proj_buf,
                                dim as u32, dims.inner_size as u32,
                                recurrent_keys.wssm_out_dtype, exec_plan.dequant_dispatch,
                            );
                            barrier.post_dispatch(&[&s.attn_out], &[&s.proj_buf]);
                            barrier.step(encoder);

                            let ffn_nw = weight_cache.get(&lw.ffn_norm).unwrap();
                            barrier.pre_dispatch(
                                &[hidden_buf, &s.proj_buf], &[hidden_buf, &s.norm_buf],
                            );
                            metal_ops.elementwise.encode_residual_add_rms_norm_out_batch(
                                encoder, hidden_buf, &s.proj_buf, ffn_nw,
                                &s.norm_buf, dim as u32, 1, eps,
                            );
                            barrier.post_dispatch(
                                &[hidden_buf, &s.proj_buf], &[hidden_buf, &s.norm_buf],
                            );
                            barrier.step(encoder);

                            // Gate/Up: both read norm_buf, write different → can overlap.
                            let wg = weight_cache.get(&lw.wg).unwrap();
                            let wu = weight_cache.get(&lw.wu).unwrap();
                            barrier.pre_dispatch(&[&s.norm_buf], &[&s.gate_buf, &s.up_buf]);
                            if !crate::model::shared::encode_dequant_matvec_pair_with_config(
                                metal_ops, encoder, wg, wu,
                                &s.norm_buf, &s.gate_buf, &s.up_buf,
                                inter_dim as u32, dim as u32,
                                lw.wg_dtype, lw.wu_dtype,
                                exec_plan.dequant_dispatch, exec_plan.use_pair_matvec,
                            ) {
                                encode_dequant_matvec_with_config(
                                    metal_ops, encoder, wg, &s.norm_buf, &s.gate_buf,
                                    inter_dim as u32, dim as u32,
                                    lw.wg_dtype, exec_plan.dequant_dispatch,
                                );
                                encode_dequant_matvec_with_config(
                                    metal_ops, encoder, wu, &s.norm_buf, &s.up_buf,
                                    inter_dim as u32, dim as u32,
                                    lw.wu_dtype, exec_plan.dequant_dispatch,
                                );
                            }
                            barrier.post_dispatch(&[&s.norm_buf], &[&s.gate_buf, &s.up_buf]);
                            barrier.step(encoder);

                            crate::model::shared::encode_gpu_ffn_decode_tail(
                                metal_ops,
                                encoder,
                                s,
                                hidden_buf,
                                weight_cache.get(&lw.wd).unwrap(),
                                lw.wd_dtype,
                                dim as u32,
                                inter_dim as u32,
                                eps,
                                exec_plan.dequant_dispatch,
                                exec_plan.use_fused_silu_down,
                                crate::model::layer_ops::FfnActivation::SiLU,
                                None,
                                next_norm,
                                &barrier,
                            );
                        },
                    );
                }
            }

            crate::model::shared::encode_gpu_output_head(
                encoder,
                metal_ops,
                s,
                hidden_buf,
                &exec_plan,
                cached,
                &weight_cache,
                &barrier,
                dim as u32,
                vocab_size as u32,
                eps,
            );
            barrier.flush();
            if fuse_argmax {
                metal_ops.elementwise.encode_argmax_f32(
                    encoder,
                    &s.logits_buf,
                    &s.argmax_idx,
                    &s.argmax_val,
                    cfg.vocab_size,
                );
            }
            Ok(())
        })
    }

    /// GPU-accelerated single-token decode for Qwen3.5.
    ///
    /// Encodes full-attention layers using the shared GPU layer encoder (1 CB
    /// per layer) and recurrent layers with minimized CPU round-trips. This is
    /// the default single-token decode path when Metal recurrent support and
    /// GPU attention KV are available.
    ///
    /// Returns `Ok(true)` if GPU path was used, `Ok(false)` to fall back.
    #[allow(clippy::too_many_arguments)]
    fn try_forward_single_gpu(
        ctx: &ForwardContext,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<bool> {
        use crate::model::shared::encode_dequant_matvec_with_config;

        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(false);
        };
        let cfg = ctx.config;
        // MoE single-token decode (both GPU and CPU paths) produces incorrect
        // output — expert weight tensor layout handling needs investigation.
        // Disable GPU decode for MoE; CPU path also has the same issue but
        // at least doesn't crash.
        if Self::qwen35_is_moe(cfg) {
            return Ok(false);
        }
        let qwen_kv = kv
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen35Forward requires ModelKv::Qwen35"))?;
        if qwen_kv.gpu_attention().is_none() {
            return Ok(false);
        }

        let dims = Self::recurrent_dims(cfg)?;
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let recurrent_slot = qwen_kv.active_slot();
        let eps = cfg.rms_norm_eps;
        let conv_cache_len = qwen_kv.conv_cache_len();

        // Cache weights on GPU.
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_qwen35(metal_ops, weights, cfg)?;
        }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let gpu_layer_keys = Self::cached_gpu_layer_keys(cached.lm_head)
            .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 gpu layer keys"))?;
        let moe_layer_keys = if Self::qwen35_is_moe(cfg) {
            let layer_keys = Self::cached_moe_layer_keys(cached.lm_head)
                .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 moe layer keys"))?;
            if cached
                .layers
                .iter()
                .zip(layer_keys.iter())
                .any(|(layer, moe)| layer.wg == 0 && moe.is_none())
            {
                return Ok(false);
            }
            Some(layer_keys)
        } else {
            None
        };
        metal_ops.init_scratches(cfg);
        let moe_scratch = if moe_layer_keys.is_some() {
            metal_ops.init_batch_scratches(cfg, 1);
            Some(metal_ops.moe_batch_scratch_view()?)
        } else {
            None
        };
        let mut scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_mut().unwrap();
        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = moe_layer_keys
            .as_ref()
            .map(|_| metal_ops.lock_moe_weight_cache());

        let seq_len = qwen_kv.seq_len();
        let full_seq_len = seq_len + 1;
        if !qwen_kv.ensure_gpu_attention_capacity_for(full_seq_len) {
            return Ok(false);
        }

        // Build execution plan.
        let exec_plan = Self::qwen35_decode_plan(
            metal_ops,
            qwen_kv.gpu_attention().unwrap(),
            cfg.embedding_dim,
            cfg.head_dim,
            full_seq_len,
            false,
        );
        let setup_t = OpTimer::start();

        // Embed token.
        {
            let h = unsafe {
                std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim)
            };
            weights.dequantize_row("token_embd.weight", token_id as usize, h)?;
        }

        let rope_position = Self::rope_position(cfg, position);
        let kv_offset = (seq_len * kv_dim) as u32;

        if !qwen_kv.has_gpu_recurrent_state() {
            for layer in 0..n_layers {
                if cfg.qwen35_is_recurrent_layer(layer) {
                    metal_ops.sync_qwen35_slot_buffers_from_kv(qwen_kv, layer, recurrent_slot);
                }
            }
        }

        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        let exec_t = OpTimer::start();
        let use_concurrent_sc = exec_plan.encoder
            == crate::model::execution_plan::DecodeEncoderPlan::Concurrent;
        macro_rules! exec_sync {
            ($device:expr, $body:expr) => {
                if use_concurrent_sc {
                    $device.execute_sync_concurrent($body)
                } else {
                    $device.execute_sync($body)
                }
            };
        }
        exec_sync!(metal_ops.device, |encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                exec_plan.barriers,
            );
            // Layer 0 attention norm.
            let layer0_norm_t = OpTimer::start();
            let norm_w = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
            metal_ops.elementwise.encode_rms_norm_out(
                encoder,
                &s.hidden,
                norm_w,
                &s.norm_buf,
                dim as u32,
                eps,
            );
            barrier.step(encoder);
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_encode_layer_norm += layer0_norm_t.elapsed();
            }

            macro_rules! encode_post_attention_ffn {
                ($lw:expr, $moe_keys:expr, $next_norm:expr) => {{
                    let residual_norm_t = OpTimer::start();
                    if let Some(moe_keys) = $moe_keys {
                        metal_ops.elementwise.encode_elementwise_add_batch(
                            encoder,
                            &s.hidden,
                            &s.proj_buf,
                            dim as u32,
                            1,
                        );
                        barrier.step(encoder);
                        if let Some(ref mut ops_ref) = ops {
                            ops_ref.gpu_encode_layer_residual += residual_norm_t.elapsed();
                        }

                        let moe_t = OpTimer::start();
                        let moe_cache = moe_weight_cache.as_ref().unwrap();
                        let shared_expert = moe_keys.shared_expert.map(|shared| {
                            crate::backend::metal::SharedExpertCachedBuffers {
                                gate: moe_cache.get(&shared.gate).unwrap(),
                                up: moe_cache.get(&shared.up).unwrap(),
                                down: moe_cache.get(&shared.down).unwrap(),
                                gate_inp: shared
                                    .gate_inp
                                    .map(|gate_inp| moe_cache.get(&gate_inp).unwrap()),
                                gate_inp_dtype: shared.gate_inp_dtype,
                                dtype: shared.dtype,
                                inter_dim: shared.inter_dim,
                                gate_inp_rows: shared.gate_inp_rows,
                            }
                        });
                        metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                            encoder,
                            moe_scratch.expect("qwen35 decode MoE scratch view missing"),
                            &s.hidden,
                            weight_cache.get(&$lw.ffn_norm).unwrap(),
                            moe_cache.get(&moe_keys.router).unwrap(),
                            moe_keys.router_dtype,
                            moe_cache.get(&moe_keys.gate).unwrap(),
                            moe_keys.gate_dtype,
                            moe_cache.get(&moe_keys.up).unwrap(),
                            moe_keys.up_dtype,
                            moe_cache.get(&moe_keys.down).unwrap(),
                            moe_keys.down_dtype,
                            1,
                            moe_keys.n_expert,
                            moe_keys.n_expert_used,
                            dim,
                            moe_keys.expert_inter_dim,
                            moe_keys.gate_stride,
                            moe_keys.up_stride,
                            moe_keys.down_stride,
                            eps,
                            shared_expert.as_ref(),
                            exec_plan.barriers
                                == crate::model::execution_plan::DecodeBarrierPlan::Explicit,
                        )?;
                        if let Some(next_norm) = $next_norm {
                            metal_ops.elementwise.encode_rms_norm_out(
                                encoder,
                                &s.hidden,
                                next_norm,
                                &s.norm_buf,
                                dim as u32,
                                eps,
                            );
                            barrier.step(encoder);
                        }
                        if let Some(ref mut ops_ref) = ops {
                            ops_ref.gpu_encode_layer_ffn += moe_t.elapsed();
                        }
                    } else {
                        let ffn_nw = weight_cache.get(&$lw.ffn_norm).unwrap();
                        metal_ops
                            .elementwise
                            .encode_residual_add_rms_norm_out_batch(
                                encoder,
                                &s.hidden,
                                &s.proj_buf,
                                ffn_nw,
                                &s.norm_buf,
                                dim as u32,
                                1,
                                eps,
                            );
                        barrier.step(encoder);
                        if let Some(ref mut ops_ref) = ops {
                            let elapsed = residual_norm_t.elapsed();
                            ops_ref.gpu_encode_layer_residual += elapsed / 2;
                            ops_ref.gpu_encode_layer_norm += elapsed / 2;
                        }

                        let gate_up_t = OpTimer::start();
                        let wg = weight_cache.get(&$lw.wg).unwrap();
                        let wu = weight_cache.get(&$lw.wu).unwrap();
                        if !crate::model::shared::encode_dequant_matvec_pair_with_config(
                            metal_ops,
                            encoder,
                            wg,
                            wu,
                            &s.norm_buf,
                            &s.gate_buf,
                            &s.up_buf,
                            inter_dim as u32,
                            dim as u32,
                            $lw.wg_dtype,
                            $lw.wu_dtype,
                            exec_plan.dequant_dispatch,
                            exec_plan.use_pair_matvec,
                        ) {
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                wg,
                                &s.norm_buf,
                                &s.gate_buf,
                                inter_dim as u32,
                                dim as u32,
                                $lw.wg_dtype,
                                exec_plan.dequant_dispatch,
                            );
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                wu,
                                &s.norm_buf,
                                &s.up_buf,
                                inter_dim as u32,
                                dim as u32,
                                $lw.wu_dtype,
                                exec_plan.dequant_dispatch,
                            );
                        }
                        barrier.step(encoder);
                        if let Some(ref mut ops_ref) = ops {
                            ops_ref.gpu_encode_layer_ffn += gate_up_t.elapsed();
                        }

                        let ffn_tail_t = OpTimer::start();
                        crate::model::shared::encode_gpu_ffn_decode_tail(
                            metal_ops,
                            encoder,
                            s,
                            &s.hidden,
                            weight_cache.get(&$lw.wd).unwrap(),
                            $lw.wd_dtype,
                            dim as u32,
                            inter_dim as u32,
                            eps,
                            exec_plan.dequant_dispatch,
                            exec_plan.use_fused_silu_down,
                            crate::model::layer_ops::FfnActivation::SiLU,
                            None,
                            $next_norm,
                            &barrier,
                        );
                        if let Some(ref mut ops_ref) = ops {
                            ops_ref.gpu_encode_layer_ffn += ffn_tail_t.elapsed();
                        }
                    }
                }};
            }

            for layer in 0..n_layers {
                let lw = &cached.layers[layer];
                let moe_keys = moe_layer_keys
                    .as_ref()
                    .and_then(|layer_keys| layer_keys[layer].as_ref());
                let next_norm = if layer + 1 < n_layers {
                    Some(
                        weight_cache
                            .get(&cached.layers[layer + 1].attn_norm)
                            .unwrap(),
                    )
                } else {
                    None
                };

                if !cfg.qwen35_is_recurrent_layer(layer) {
                    let q_norm_key = lw.attn_q_norm;
                    let k_norm_key = lw.attn_k_norm;
                    let gpu_attn = qwen_kv.gpu_attention().unwrap();

                    let qkv_t = OpTimer::start();
                    let wq = weight_cache.get(&lw.wq).unwrap();
                    let wk = weight_cache.get(&lw.wk).unwrap();
                    let wv = weight_cache.get(&lw.wv).unwrap();
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wq,
                        &s.norm_buf,
                        &s.gate_buf,
                        (q_dim * 2) as u32,
                        dim as u32,
                        lw.wq_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wk,
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
                        wv,
                        &s.norm_buf,
                        &s.v_buf,
                        kv_dim as u32,
                        dim as u32,
                        lw.wv_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    metal_ops.elementwise.encode_split_qgate_batch(
                        encoder,
                        &s.gate_buf,
                        &s.q_buf,
                        &s.up_buf,
                        1,
                        q_dim as u32,
                    );
                    barrier.step(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_qkv += qkv_t.elapsed();
                    }

                    if let (Some(q_key), Some(k_key)) = (q_norm_key, k_norm_key) {
                        let qk_norm_t = OpTimer::start();
                        let q_nw = weight_cache.get(&q_key).unwrap();
                        let k_nw = weight_cache.get(&k_key).unwrap();
                        metal_ops.elementwise.encode_per_head_rms_norm_batch(
                            encoder,
                            &s.q_buf,
                            q_nw,
                            1,
                            n_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        metal_ops.elementwise.encode_per_head_rms_norm_batch(
                            encoder,
                            &s.k_buf,
                            k_nw,
                            1,
                            n_kv_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        barrier.step(encoder);
                        if let Some(ref mut ops_ref) = ops {
                            ops_ref.gpu_encode_layer_norm += qk_norm_t.elapsed();
                        }
                    }

                    let rope_t = OpTimer::start();
                    metal_ops.elementwise.encode_rope_batch(
                        encoder,
                        &s.q_buf,
                        &s.k_buf,
                        1,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        rope_position,
                        0.0,
                        cfg.rope_freq_base,
                    );
                    barrier.step(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_rope += rope_t.elapsed();
                    }

                    let kv_append_t = OpTimer::start();
                    metal_ops.elementwise.encode_kv_append(
                        encoder,
                        &s.k_buf,
                        gpu_attn.k_buffer(layer),
                        gpu_attn.is_f16(),
                        kv_offset,
                        kv_dim as u32,
                    );
                    metal_ops.elementwise.encode_kv_append(
                        encoder,
                        &s.v_buf,
                        gpu_attn.v_buffer(layer),
                        gpu_attn.is_f16(),
                        kv_offset,
                        kv_dim as u32,
                    );
                    barrier.step(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_kv_append += kv_append_t.elapsed();
                    }

                    let attn_t = OpTimer::start();
                    metal_ops
                        .attention
                        .encode_attention_decode_with_scratch_and_config(
                            encoder,
                            &s.q_buf,
                            gpu_attn.k_buffer(layer),
                            gpu_attn.v_buffer(layer),
                            &s.attn_out,
                            &s.splitk_partial_out,
                            &s.splitk_partial_lse,
                            gpu_attn.is_f16(),
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            0,
                            full_seq_len as u32,
                            exec_plan.attention_dispatch,
                        );
                    barrier.step(encoder);

                    metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                        encoder,
                        &s.up_buf,
                        &s.attn_out,
                        q_dim as u32,
                    );
                    barrier.step(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_attention += attn_t.elapsed();
                    }

                    let out_proj_t = OpTimer::start();
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        weight_cache.get(&lw.wo).unwrap(),
                        &s.attn_out,
                        &s.proj_buf,
                        dim as u32,
                        q_dim as u32,
                        lw.wo_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    barrier.step(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_out_proj += out_proj_t.elapsed();
                    }

                    encode_post_attention_ffn!(lw, moe_keys, next_norm);
                } else {
                    let recurrent_keys = match &gpu_layer_keys[layer] {
                        Qwen35GpuLayerKeys::Recurrent(keys) => keys,
                        Qwen35GpuLayerKeys::FullAttention => {
                            anyhow::bail!("expected recurrent qwen35 GPU keys for layer {layer}")
                        }
                    };
                    debug_assert!(kv_dim >= dims.time_step_rank);
                    let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
                    let recurrent_state_stride = qwen_kv.recurrent_state_len();
                    metal_ops.with_qwen35_recurrent_slot_buffer_for_kv(
                        qwen_kv,
                        layer,
                        recurrent_slot,
                        conv_state_stride,
                        recurrent_state_stride,
                        |slot_buffers| -> anyhow::Result<()> {
                            let recurrent_norm_t = OpTimer::start();
                            let norm_w = weight_cache.get(&lw.attn_norm).unwrap();
                            metal_ops.elementwise.encode_rms_norm_out(
                                encoder,
                                &s.hidden,
                                norm_w,
                                &s.norm_buf,
                                dim as u32,
                                eps,
                            );
                            barrier.step(encoder);
                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_norm += recurrent_norm_t.elapsed();
                            }

                            let recurrent_input_t = OpTimer::start();
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                weight_cache.get(&recurrent_keys.wqkv).unwrap(),
                                &s.norm_buf,
                                &s.gate_buf,
                                dims.conv_dim() as u32,
                                dim as u32,
                                recurrent_keys.wqkv_dtype,
                                exec_plan.dequant_dispatch,
                            );
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                weight_cache.get(&recurrent_keys.wgate).unwrap(),
                                &s.norm_buf,
                                &s.attn_out,
                                dims.inner_size as u32,
                                dim as u32,
                                recurrent_keys.wgate_dtype,
                                exec_plan.dequant_dispatch,
                            );
                            let wbeta = weight_cache.get(&recurrent_keys.wbeta).unwrap();
                            let walpha = weight_cache.get(&recurrent_keys.walpha).unwrap();
                            if !Self::encode_recurrent_pair_matvec_with_config(
                                metal_ops,
                                encoder,
                                wbeta,
                                walpha,
                                &s.norm_buf,
                                &s.v_buf,
                                &s.proj_buf,
                                dims.time_step_rank as u32,
                                dim as u32,
                                recurrent_keys.wbeta_dtype,
                                recurrent_keys.walpha_dtype,
                            ) {
                                encode_dequant_matvec_with_config(
                                    metal_ops,
                                    encoder,
                                    wbeta,
                                    &s.norm_buf,
                                    &s.v_buf,
                                    dims.time_step_rank as u32,
                                    dim as u32,
                                    recurrent_keys.wbeta_dtype,
                                    exec_plan.dequant_dispatch,
                                );
                                encode_dequant_matvec_with_config(
                                    metal_ops,
                                    encoder,
                                    walpha,
                                    &s.norm_buf,
                                    &s.proj_buf,
                                    dims.time_step_rank as u32,
                                    dim as u32,
                                    recurrent_keys.walpha_dtype,
                                    exec_plan.dequant_dispatch,
                                );
                            }
                            barrier.step(encoder);

                            // Apply prepare_alpha_beta on GPU: softplus(alpha + dt_bias) * A
                            // for the decay factor, and sigmoid(beta) for the input gate.
                            let dt_bias_buf = weight_cache.get(&recurrent_keys.dt_bias).unwrap();
                            let ssm_a_buf = weight_cache.get(&recurrent_keys.ssm_a).unwrap();
                            metal_ops.elementwise.encode_softplus_bias_mul(
                                encoder,
                                &s.proj_buf,
                                dt_bias_buf,
                                ssm_a_buf,
                                dims.time_step_rank as u32,
                            );
                            metal_ops.elementwise.encode_sigmoid_inplace(
                                encoder,
                                &s.v_buf,
                                dims.time_step_rank as u32,
                            );
                            barrier.step(encoder);

                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_qkv += recurrent_input_t.elapsed();
                            }

                            let recurrent_core_t = OpTimer::start();
                            metal_ops.gdn.encode_causal_conv_sequence(
                                encoder,
                                &s.gate_buf,
                                weight_cache.get(&recurrent_keys.conv_kernel).unwrap(),
                                &slot_buffers.conv_state,
                                &s.up_buf,
                                1,
                                conv_cache_len as u32,
                                dims.conv_dim() as u32,
                            );
                            barrier.step(encoder);

                            if metal_ops.gdn.encode_single_token_gated_delta_fused(
                                encoder,
                                &s.up_buf,
                                &s.proj_buf,
                                &s.v_buf,
                                &slot_buffers.recurrent_state,
                                &s.q_buf,
                                dims.group_count as u32,
                                dims.time_step_rank as u32,
                                dims.state_size as u32,
                                eps,
                            ) {
                                barrier.step(encoder);
                            } else {
                                metal_ops.gdn.encode_prepare_single_token_qkv(
                                    encoder,
                                    &s.up_buf,
                                    &s.q_buf,
                                    &s.norm_buf,
                                    &s.down_buf,
                                    dims.group_count as u32,
                                    dims.time_step_rank as u32,
                                    dims.state_size as u32,
                                    eps,
                                );
                                barrier.step(encoder);

                                metal_ops.gdn.encode_gated_delta_sequence(
                                    encoder,
                                    &s.q_buf,
                                    &s.norm_buf,
                                    &s.down_buf,
                                    &s.proj_buf,
                                    &s.v_buf,
                                    &slot_buffers.recurrent_state,
                                    &s.q_buf,
                                    1,
                                    dims.time_step_rank as u32,
                                    dims.state_size as u32,
                                );
                                barrier.step(encoder);
                            }

                            let ssm_norm = weight_cache.get(&recurrent_keys.ssm_norm).unwrap();
                            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                                encoder,
                                &s.q_buf,
                                ssm_norm,
                                1,
                                dims.time_step_rank as u32,
                                dims.state_size as u32,
                                eps,
                            );
                            barrier.step(encoder);

                            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                                encoder,
                                &s.attn_out,
                                &s.q_buf,
                                dims.inner_size as u32,
                                1,
                            );
                            barrier.step(encoder);
                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_attention += recurrent_core_t.elapsed();
                            }

                            let recurrent_out_proj_t = OpTimer::start();
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                weight_cache.get(&recurrent_keys.wssm_out).unwrap(),
                                &s.attn_out,
                                &s.proj_buf,
                                dim as u32,
                                dims.inner_size as u32,
                                recurrent_keys.wssm_out_dtype,
                                exec_plan.dequant_dispatch,
                            );
                            barrier.step(encoder);
                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_out_proj += recurrent_out_proj_t.elapsed();
                            }

                            encode_post_attention_ffn!(lw, moe_keys, next_norm);
                            Ok(())
                        },
                    )?;
                }
            }

            crate::model::shared::encode_gpu_output_head(
                encoder,
                metal_ops,
                s,
                &s.hidden,
                &exec_plan,
                cached,
                &weight_cache,
                &barrier,
                dim as u32,
                vocab_size as u32,
                eps,
            );
            barrier.flush();
            Ok(())
        })?;
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_execute += exec_t.elapsed();
        }

        drop(weight_cache);
        drop(cached_guard);

        let rb_t = OpTimer::start();
        for layer in 0..n_layers {
            if cfg.qwen35_is_recurrent_layer(layer) {
                let _ = qwen_kv.note_backend_conv_state_update(recurrent_slot, layer);
                let _ = qwen_kv.note_backend_recurrent_state_update(recurrent_slot, layer);
                if !qwen_kv.has_gpu_recurrent_state() {
                    let conv_generation =
                        qwen_kv.conv_state_generation(recurrent_slot, layer);
                    let recurrent_generation =
                        qwen_kv.recurrent_state_generation(recurrent_slot, layer);
                    let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
                    let recurrent_state_stride = qwen_kv.recurrent_state_len();
                    metal_ops.with_qwen35_recurrent_slot_buffer(
                        layer,
                        recurrent_slot,
                        conv_state_stride,
                        recurrent_state_stride,
                        |slot_buffers| {
                            slot_buffers.conv_synced_generation = Some(conv_generation);
                            slot_buffers.recurrent_synced_generation =
                                Some(recurrent_generation);
                        },
                    );
                }
            }
        }
        qwen_kv.mark_attention_cpu_dirty();

        qwen_kv.finalize_token();

        let logits_gpu = unsafe { &s.logits_buf.as_slice::<f32>()[..vocab_size] };
        logits[..vocab_size].copy_from_slice(logits_gpu);
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }
        Ok(true)
    }
}
