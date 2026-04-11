impl Qwen3MoeForward {
    fn qwen3moe_decode_layers_per_command_buffer(
        layer_count: usize,
        gate_dtype: crate::gguf::tensor::GgmlType,
        up_dtype: crate::gguf::tensor::GgmlType,
        down_dtype: crate::gguf::tensor::GgmlType,
    ) -> usize {
        if let Ok(value) = std::env::var("AX_QWEN3MOE_GPU_DECODE_LAYERS_PER_CB")
            && let Ok(parsed) = value.trim().parse::<usize>()
            && parsed > 0
        {
            return parsed.min(layer_count.max(1));
        }

        // Qwen3-MoE now has selected routed-expert decode kernels across the
        // shipped quant families, and short-run plus 2k-context checks show
        // that coalescing the full routed stack is the best default. Keep the
        // env override above for quick A/B and fallback control.
        if Self::moe_gpu_expert_dtype_supported(gate_dtype)
            && Self::moe_gpu_expert_dtype_supported(up_dtype)
            && Self::moe_gpu_expert_dtype_supported(down_dtype)
        {
            return layer_count.max(1);
        }

        1
    }

    fn qwen3moe_prefill_split_layer(
        layer_count: usize,
        gate_dtype: crate::gguf::tensor::GgmlType,
        up_dtype: crate::gguf::tensor::GgmlType,
        down_dtype: crate::gguf::tensor::GgmlType,
        multi_cb_enabled: bool,
    ) -> usize {
        if !multi_cb_enabled {
            return layer_count;
        }

        // Reuse the Qwen3.5-style two-command-buffer overlap only where it
        // actually wins. Repeated Qwen3-Coder A/B runs improved Q4/Q5/Q6
        // prefill, but regressed the fully-Q8 routed-expert stack, so keep Q8
        // on the single-command-buffer path until it has a graph-IR schedule
        // that earns the split.
        let fully_q8 = gate_dtype == crate::gguf::tensor::GgmlType::Q8_0
            && up_dtype == crate::gguf::tensor::GgmlType::Q8_0
            && down_dtype == crate::gguf::tensor::GgmlType::Q8_0;
        if fully_q8 {
            return layer_count;
        }

        layer_count / 2
    }

    fn build_cached_model_keys(
        metal_ops: &MetalOps,
        weights: &WeightStore,
        cfg: &ModelConfig,
    ) -> anyhow::Result<()> {
        use crate::backend::metal::{CachedLayerKeys, CachedModelKeys};

        let dim = cfg.embedding_dim as usize;
        let use_precomputed_f16 = metal_ops.metal_precompute_f16_enabled();
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);

        for layer in 0..cfg.n_layers as usize {
            let prefix = format!("blk.{layer}");
            let n_heads = cfg.n_heads as usize;
            let head_dim = cfg.head_dim as usize;
            let n_kv_heads = cfg.n_kv_heads as usize;
            let q_dim = n_heads * head_dim;
            let kv_dim = n_kv_heads * head_dim;

            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            let attn_norm_key = metal_ops.ensure_f32_cached(attn_norm_w);

            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
            let (wo_raw, wo_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;

            let wq_key = metal_ops.ensure_quant_cached(wq_raw);
            let wk_key = metal_ops.ensure_quant_cached(wk_raw);
            let wv_key = metal_ops.ensure_quant_cached(wv_raw);
            let wo_key = metal_ops.ensure_quant_cached(wo_raw);

            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);

            if use_precomputed_f16 {
                crate::model::shared::ensure_precomputed_linear_f16(metal_ops, wq_raw, wq_dtype, q_dim as u32, dim as u32)?;
                crate::model::shared::ensure_precomputed_linear_f16(metal_ops, wk_raw, wk_dtype, kv_dim as u32, dim as u32)?;
                crate::model::shared::ensure_precomputed_linear_f16(metal_ops, wv_raw, wv_dtype, kv_dim as u32, dim as u32)?;
                crate::model::shared::ensure_precomputed_linear_f16(metal_ops, wo_raw, wo_dtype, dim as u32, q_dim as u32)?;
            }

            let (attn_q_norm_key, attn_k_norm_key) = cache_attention_qk_norm_keys(metal_ops, weights, &prefix)?;

            let moe_layer =
                crate::model::shared::build_routed_moe_resident_layer_keys(
                    metal_ops, cfg, weights, &prefix, dim,
                )?
                .ok_or_else(|| anyhow::anyhow!("qwen3moe layer {layer} lacks GPU-routed MoE support"))?;

            layers.push(CachedLayerKeys {
                attn_norm: attn_norm_key,
                wq: wq_key,
                wq_dtype,
                wk: wk_key,
                wk_dtype,
                wv: wv_key,
                wv_dtype,
                wo: wo_key, wo_dtype,
                ffn_norm: ffn_norm_key,
                // Qwen3-MoE layers do not expose dense FFN tensors; keep these
                // slots invalid and route FFN execution through the GPU-resident
                // routed-MoE path below.
                wg: 0,
                wg_dtype: crate::gguf::tensor::GgmlType::F32,
                wu: 0,
                wu_dtype: crate::gguf::tensor::GgmlType::F32,
                wd: 0,
                wd_dtype: crate::gguf::tensor::GgmlType::F32,
                attn_q_norm: attn_q_norm_key,
                attn_k_norm: attn_k_norm_key,
                post_attn_norm: None,
                post_ffn_norm: None,
                v_equals_k: false,
                layer_output_scale: None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                wo_bias: None,
                gate_bias: None,
                up_bias: None,
                down_bias: None,
                moe_router: Some(moe_layer.router),
                moe_router_dtype: Some(moe_layer.router_dtype),
                moe_expert_gate: Some(vec![moe_layer.gate]),
                moe_expert_up: Some(vec![moe_layer.up]),
                moe_expert_down: Some(vec![moe_layer.down]),
                moe_expert_dtype: Some(moe_layer.down_dtype),
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_dtype: None,
            });
        }

        let (output_norm_key, lm_raw, lm_dtype, lm_head_key) = cache_output_head_keys(metal_ops, weights)?;
        if use_precomputed_f16 {
            crate::model::shared::ensure_precomputed_lm_head_f16(metal_ops, lm_raw, lm_dtype, cfg.vocab_size, cfg.embedding_dim)?;
        }

        metal_ops.set_cached_model_keys(CachedModelKeys {
            layers, output_norm: output_norm_key, lm_head: lm_head_key, lm_head_dtype: lm_dtype, rope_freqs: None,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_single_gpu_unified(
        &self, ctx: &ForwardContext, metal_ops: &MetalOps, token_id: u32, position: usize,
        gpu_kv: &mut crate::kv::GpuKv, weights: &WeightStore, logits: &mut [f32],
        _ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let n_expert = cfg.n_expert.unwrap_or(0) as usize;
        let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
        let eps = cfg.rms_norm_eps;
        anyhow::ensure!(logits.len() >= vocab_size, "logits buffer too small");

        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, 1); // MoE FFN needs batch scratch even for n=1
        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + 1)?;

        {
            let hidden_cpu = unsafe { std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim) };
            weights.dequantize_row("token_embd.weight", token_id as usize, hidden_cpu)?;
        }

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys(metal_ops, weights, cfg)?;
        }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let cur_seq_len = gpu_kv.seq_len();
        let full_seq_len = cur_seq_len + 1;
        let base_plan = DecodeExecutionPlan::gemma4_single_cb(metal_ops, gpu_kv, cfg.embedding_dim, cfg.head_dim, full_seq_len);

        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();

        let (_, gate_dtype) = weights.raw_with_dtype("blk.0.ffn_gate_exps.weight")?;
        let (_, up_dtype) = weights.raw_with_dtype("blk.0.ffn_up_exps.weight")?;
        let (_, down_dtype) = weights.raw_with_dtype("blk.0.ffn_down_exps.weight")?;
        let gate_stride =
            crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
        let up_stride =
            crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
        let layers_per_command_buffer =
            Self::qwen3moe_decode_layers_per_command_buffer(
                cfg.n_layers as usize,
                gate_dtype,
                up_dtype,
                down_dtype,
            );
        let kv_f16 = gpu_kv.is_f16();

        // Keep routed-MoE decode coalesced to minimize submission overhead.
        // The policy above uses the full layer stack for supported quantized
        // layouts, while the env override can still split ranges for A/B runs
        // or fallback if a model/device combination needs it.
        for layer_start in (0..cfg.n_layers as usize).step_by(layers_per_command_buffer) {
            let layer_end =
                (layer_start + layers_per_command_buffer).min(cfg.n_layers as usize);
            metal_ops.device.execute_sync(|encoder| {
                let barrier =
                    crate::model::shared::DecodeBarrierCtx::new(encoder, base_plan.barriers);
                for layer in layer_start..layer_end {
                    let lw = &cached.layers[layer];
                    let attn_norm_w = weight_cache.get(&lw.attn_norm).unwrap();
                    let wq_buf = weight_cache.get(&lw.wq).unwrap();
                    let wk_buf = weight_cache.get(&lw.wk).unwrap();
                    let wv_buf = weight_cache.get(&lw.wv).unwrap();
                    let wo_buf = weight_cache.get(&lw.wo).unwrap();
                    let ffn_norm_w = weight_cache.get(&lw.ffn_norm).unwrap();
                    let kv_k = gpu_kv.k_buffer(layer);
                    let kv_v = gpu_kv.v_buffer(layer);
                    let kv_stride = gpu_kv.kv_stride_for_layer(layer);
                    let kv_offset = (cur_seq_len * kv_stride) as u32;

                    barrier.pre_dispatch(&[&s.hidden], &[&s.norm_buf]);
                    metal_ops.elementwise.encode_rms_norm_out(
                        encoder,
                        &s.hidden,
                        attn_norm_w,
                        &s.norm_buf,
                        dim as u32,
                        eps,
                    );
                    barrier.post_dispatch(&[&s.hidden], &[&s.norm_buf]);
                    barrier.step(encoder);

                    barrier.pre_dispatch(&[&s.norm_buf], &[&s.q_buf]);
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wq_buf,
                        &s.norm_buf,
                        &s.q_buf,
                        q_dim as u32,
                        dim as u32,
                        lw.wq_dtype,
                        base_plan.dequant_dispatch,
                    );
                    barrier.post_dispatch(&[&s.norm_buf], &[&s.q_buf]);
                    barrier.pre_dispatch(&[&s.norm_buf], &[&s.k_buf]);
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wk_buf,
                        &s.norm_buf,
                        &s.k_buf,
                        kv_dim as u32,
                        dim as u32,
                        lw.wk_dtype,
                        base_plan.dequant_dispatch,
                    );
                    barrier.post_dispatch(&[&s.norm_buf], &[&s.k_buf]);
                    barrier.pre_dispatch(&[&s.norm_buf], &[&s.v_buf]);
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wv_buf,
                        &s.norm_buf,
                        &s.v_buf,
                        kv_dim as u32,
                        dim as u32,
                        lw.wv_dtype,
                        base_plan.dequant_dispatch,
                    );
                    barrier.post_dispatch(&[&s.norm_buf], &[&s.v_buf]);
                    barrier.step(encoder);

                    if let (Some(qn_key), Some(kn_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
                        let qn = weight_cache.get(&qn_key).unwrap();
                        let kn = weight_cache.get(&kn_key).unwrap();
                        barrier.pre_dispatch(&[&s.q_buf], &[&s.q_buf]);
                        metal_ops.elementwise.encode_per_head_rms_norm(
                            encoder,
                            &s.q_buf,
                            qn,
                            n_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        barrier.post_dispatch(&[&s.q_buf], &[&s.q_buf]);
                        barrier.pre_dispatch(&[&s.k_buf], &[&s.k_buf]);
                        metal_ops.elementwise.encode_per_head_rms_norm(
                            encoder,
                            &s.k_buf,
                            kn,
                            n_kv_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        barrier.post_dispatch(&[&s.k_buf], &[&s.k_buf]);
                    }

                    barrier.pre_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
                    metal_ops.elementwise.encode_rope_batch_neox_partial(
                        encoder,
                        &s.q_buf,
                        &s.k_buf,
                        1,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        head_dim as u32,
                        position as f32,
                        1.0,
                        cfg.rope_freq_base,
                    );
                    barrier.post_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
                    barrier.step(encoder);

                    barrier.pre_dispatch(&[&s.k_buf], &[kv_k]);
                    metal_ops.elementwise.encode_kv_append(
                        encoder,
                        &s.k_buf,
                        kv_k,
                        kv_f16,
                        kv_offset,
                        kv_dim as u32,
                    );
                    barrier.post_dispatch(&[&s.k_buf], &[kv_k]);
                    barrier.pre_dispatch(&[&s.v_buf], &[kv_v]);
                    metal_ops.elementwise.encode_kv_append(
                        encoder,
                        &s.v_buf,
                        kv_v,
                        kv_f16,
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
                            kv_f16,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            0,
                            full_seq_len as u32,
                            base_plan.attention_dispatch,
                        );
                    barrier.post_dispatch(&[&s.q_buf, kv_k, kv_v], &[&s.attn_out]);
                    barrier.step(encoder);

                    barrier.pre_dispatch(&[&s.attn_out], &[&s.proj_buf]);
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wo_buf,
                        &s.attn_out,
                        &s.proj_buf,
                        dim as u32,
                        q_dim as u32,
                        lw.wo_dtype,
                        base_plan.dequant_dispatch,
                    );
                    barrier.post_dispatch(&[&s.attn_out], &[&s.proj_buf]);
                    barrier.step(encoder);

                    barrier.pre_dispatch(&[&s.hidden, &s.proj_buf], &[&s.hidden]);
                    metal_ops.elementwise.encode_elementwise_add(
                        encoder,
                        &s.hidden,
                        &s.proj_buf,
                        dim as u32,
                    );
                    barrier.post_dispatch(&[&s.hidden, &s.proj_buf], &[&s.hidden]);
                    barrier.step(encoder);

                    let router_buf = moe_weight_cache.get(&lw.moe_router.unwrap()).unwrap();
                    let moe_gate_buf = moe_weight_cache
                        .get(&lw.moe_expert_gate.as_ref().unwrap()[0])
                        .unwrap();
                    let moe_up_buf = moe_weight_cache
                        .get(&lw.moe_expert_up.as_ref().unwrap()[0])
                        .unwrap();
                    let moe_down_buf = moe_weight_cache
                        .get(&lw.moe_expert_down.as_ref().unwrap()[0])
                        .unwrap();
                    metal_ops.encode_moe_ffn_gpu_resident_cached(
                        encoder,
                        &s.hidden,
                        ffn_norm_w,
                        router_buf,
                        lw.moe_router_dtype.unwrap(),
                        moe_gate_buf,
                        gate_dtype,
                        moe_up_buf,
                        up_dtype,
                        moe_down_buf,
                        down_dtype,
                        1,
                        n_expert,
                        n_expert_used,
                        dim,
                        expert_inter_dim,
                        gate_stride,
                        up_stride,
                        down_stride,
                        eps,
                        None,
                        base_plan.barriers
                            != crate::model::execution_plan::DecodeBarrierPlan::Implicit,
                    )?;
                }

                barrier.flush();
                Ok(())
            })?;
        }

        // (debug removed)
        // Output head in its own CB
        metal_ops.device.execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(encoder, base_plan.barriers);
            crate::model::shared::encode_gpu_output_head(encoder, metal_ops, s, &s.hidden, &base_plan, cached, &weight_cache, &barrier, dim as u32, vocab_size as u32, eps);
            barrier.flush();
            Ok(())
        })?;

        let logits_gpu = unsafe { std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab_size) };
        logits[..vocab_size].copy_from_slice(logits_gpu);
        gpu_kv.finalize_batch(1);
        Ok(())
    }

    /// GPU unified batch prefill — all N tokens processed in one command buffer.
    ///
    /// For each layer: batch norm → batch Q/K/V matmul → batch QK norm →
    /// batch RoPE → batch KV append → prefill attention → batch output proj →
    /// residual add → batch MoE FFN. Then last-token logits via matvec.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_gpu_unified(
        &self,
        ctx: &ForwardContext,
        metal_ops: &MetalOps,
        token_ids: &[u32],
        gpu_kv: &mut crate::kv::GpuKv,
        weights: &WeightStore,
        last_logits: Option<&mut [f32]>,
        _logits_all: Option<&mut Vec<f32>>,
        _ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let n_tokens = token_ids.len();
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let n_expert = cfg.n_expert.unwrap_or(0) as usize;
        let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
        let eps = cfg.rms_norm_eps;

        if let Some(logits) = last_logits.as_ref() {
            anyhow::ensure!(logits.len() >= vocab_size, "logits buffer too small");
        }

        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, n_tokens);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let batch_guard = metal_ops.batch_scratches();
        let bs = batch_guard.as_ref().unwrap();

        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + n_tokens)?;

        // Token embedding
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

        let (_, gate_dtype) = weights.raw_with_dtype("blk.0.ffn_gate_exps.weight")?;
        let (_, up_dtype) = weights.raw_with_dtype("blk.0.ffn_up_exps.weight")?;
        let (_, down_dtype) = weights.raw_with_dtype("blk.0.ffn_down_exps.weight")?;
        let gate_stride =
            crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
        let up_stride =
            crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();

        let has_q5k = crate::model::shared::gpu_prefill_uses_q5k(weights);
        let q5k_small_n = crate::model::shared::gpu_prefill_q5k_small_n_auto_eligible(weights);
        let prefill_plan = DecodeExecutionPlan::gemma3_prefill(
            metal_ops,
            gpu_kv,
            n_tokens as u32,
            has_q5k,
            q5k_small_n,
        );

        let nt = n_tokens as u32;
        let split_layer = Self::qwen3moe_prefill_split_layer(
            n_layers,
            gate_dtype,
            up_dtype,
            down_dtype,
            crate::model::prefill_schedule::prefill_multi_cb_enabled(),
        );

        let encode_layer_range = |
            encoder: &ax_engine_metal::MetalEncoder,
            layer_range: std::ops::Range<usize>,
            include_output_head: bool,
        | -> anyhow::Result<()> {
            let barrier_on = true;
            let barrier = |enc: &ax_engine_metal::MetalEncoder| {
                if barrier_on {
                    ax_engine_metal::barrier_buffers(enc);
                }
            };

            for layer in layer_range {
                let lw = &cached.layers[layer];
                let attn_norm_w = weight_cache.get(&lw.attn_norm).unwrap();
                let wq_buf = weight_cache.get(&lw.wq).unwrap();
                let wk_buf = weight_cache.get(&lw.wk).unwrap();
                let wv_buf = weight_cache.get(&lw.wv).unwrap();
                let wo_buf = weight_cache.get(&lw.wo).unwrap();
                let ffn_norm_w = weight_cache.get(&lw.ffn_norm).unwrap();
                let kv_k = gpu_kv.k_buffer(layer);
                let kv_v = gpu_kv.v_buffer(layer);
                let kv_layer_stride = gpu_kv.kv_stride_for_layer(layer);
                let kv_offset = (base_seq_len * kv_layer_stride) as u32;
                let kv_stride = kv_layer_stride as u32;

                // ── Attention norm (batch) ──
                metal_ops.elementwise.encode_rms_norm_out_batch(
                    encoder, &bs.hidden, attn_norm_w, &bs.norm_buf,
                    dim as u32, nt, eps,
                );
                barrier(encoder);

                // ── Q/K/V batch projections ──
                let use_f16 = prefill_plan.use_f16_batch_io;
                if use_f16 {
                    metal_ops.elementwise.encode_cast_f32_to_f16(
                        encoder, &bs.norm_buf, &bs.matmul_in_f16, nt * dim as u32,
                    );
                    barrier(encoder);
                    crate::model::shared::encode_dequant_batch_f16in(
                        metal_ops, encoder, wq_buf, &bs.matmul_in_f16, &bs.q_buf,
                        q_dim as u32, nt, dim as u32, lw.wq_dtype,
                    );
                    crate::model::shared::encode_dequant_batch_f16in(
                        metal_ops, encoder, wk_buf, &bs.matmul_in_f16, &bs.k_buf,
                        kv_dim as u32, nt, dim as u32, lw.wk_dtype,
                    );
                    crate::model::shared::encode_dequant_batch_f16in(
                        metal_ops, encoder, wv_buf, &bs.matmul_in_f16, &bs.v_buf,
                        kv_dim as u32, nt, dim as u32, lw.wv_dtype,
                    );
                } else {
                    crate::model::shared::encode_dequant_batch(
                        &metal_ops.dequant, &metal_ops.elementwise,
                        encoder, wq_buf, &bs.norm_buf, &bs.q_buf, &bs.matmul_in_f16,
                        q_dim as u32, nt, dim as u32, lw.wq_dtype,
                        false, prefill_plan.use_batch_simd, prefill_plan.q5k_prefill_small_n,
                    );
                    crate::model::shared::encode_dequant_batch(
                        &metal_ops.dequant, &metal_ops.elementwise,
                        encoder, wk_buf, &bs.norm_buf, &bs.k_buf, &bs.matmul_in_f16,
                        kv_dim as u32, nt, dim as u32, lw.wk_dtype,
                        false, prefill_plan.use_batch_simd, prefill_plan.q5k_prefill_small_n,
                    );
                    crate::model::shared::encode_dequant_batch(
                        &metal_ops.dequant, &metal_ops.elementwise,
                        encoder, wv_buf, &bs.norm_buf, &bs.v_buf, &bs.matmul_in_f16,
                        kv_dim as u32, nt, dim as u32, lw.wv_dtype,
                        false, prefill_plan.use_batch_simd, prefill_plan.q5k_prefill_small_n,
                    );
                }
                barrier(encoder);

                // ── QK norm (batch) ──
                if let (Some(qn_key), Some(kn_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
                    let qn = weight_cache.get(&qn_key).unwrap();
                    let kn = weight_cache.get(&kn_key).unwrap();
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder, &bs.q_buf, qn, n_heads as u32, head_dim as u32, nt, eps,
                    );
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder, &bs.k_buf, kn, n_kv_heads as u32, head_dim as u32, nt, eps,
                    );
                    barrier(encoder);
                }

                // ── RoPE (batch) ──
                metal_ops.elementwise.encode_rope_batch_neox_partial(
                    encoder, &bs.q_buf, &bs.k_buf,
                    nt, n_heads as u32, n_kv_heads as u32,
                    head_dim as u32, head_dim as u32,
                    base_seq_len as f32, 1.0, cfg.rope_freq_base,
                );
                barrier(encoder);

                // ── KV append (batch pair) ──
                metal_ops.elementwise.encode_kv_append_batch_pair(
                    encoder, &bs.k_buf, &bs.v_buf, kv_k, kv_v,
                    prefill_plan.kv_f16, kv_offset, kv_stride,
                    kv_dim as u32, nt,
                );
                barrier(encoder);

                // ── Prefill attention ──
                metal_ops.attention.encode_attention_prefill_cached_with_config(
                    encoder, &bs.q_buf, kv_k, kv_v, &bs.attn_out,
                    prefill_plan.kv_f16, nt,
                    n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    base_seq_len as u32, 0,
                    prefill_plan.attention_dispatch,
                );
                barrier(encoder);

                // ── Output projection (batch) ──
                crate::model::shared::encode_dequant_batch(
                    &metal_ops.dequant, &metal_ops.elementwise,
                    encoder, wo_buf, &bs.attn_out, &bs.proj_buf, &bs.matmul_in_f16,
                    dim as u32, nt, q_dim as u32, lw.wo_dtype,
                    prefill_plan.use_f16_batch_io, prefill_plan.use_batch_simd,
                    prefill_plan.q5k_prefill_small_n,
                );
                barrier(encoder);

                // ── Residual add (batch) ──
                metal_ops.elementwise.encode_elementwise_add_batch(
                    encoder, &bs.hidden, &bs.proj_buf, dim as u32, nt,
                );
                barrier(encoder);

                // ── MoE FFN (batch, GPU resident) ──
                let router_buf = moe_weight_cache.get(&lw.moe_router.unwrap()).unwrap();
                let moe_gate_buf = moe_weight_cache
                    .get(&lw.moe_expert_gate.as_ref().unwrap()[0]).unwrap();
                let moe_up_buf = moe_weight_cache
                    .get(&lw.moe_expert_up.as_ref().unwrap()[0]).unwrap();
                let moe_down_buf = moe_weight_cache
                    .get(&lw.moe_expert_down.as_ref().unwrap()[0]).unwrap();

                {
                    let moe_scratch = crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs)?;
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                        encoder, moe_scratch, &bs.hidden, ffn_norm_w,
                        router_buf, lw.moe_router_dtype.unwrap(),
                        moe_gate_buf, gate_dtype,
                        moe_up_buf, up_dtype,
                        moe_down_buf, down_dtype,
                        n_tokens, n_expert, n_expert_used,
                        dim, expert_inter_dim,
                        gate_stride, up_stride, down_stride,
                        eps, None, true,
                    )?;
                }
            }

            if include_output_head {
                // ── Output head (last token) ──
                let last_off = (n_tokens - 1) * dim * std::mem::size_of::<f32>();
                metal_ops.elementwise.encode_buffer_copy(
                    encoder, &bs.hidden, last_off, &s.hidden, 0, dim as u32,
                );
                barrier(encoder);

                let output_norm_w = weight_cache.get(&cached.output_norm).unwrap();
                metal_ops.elementwise.encode_rms_norm(
                    encoder, &s.hidden, output_norm_w, dim as u32, eps,
                );
                barrier(encoder);

                let lm_head_buf = weight_cache.get(&cached.lm_head).unwrap();
                crate::model::shared::encode_dequant_matvec(
                    metal_ops, encoder, lm_head_buf, &s.hidden, &s.logits_buf,
                    vocab_size as u32, dim as u32, cached.lm_head_dtype,
                );
                barrier(encoder);
            }
            Ok(())
        };

        if split_layer > 0 && split_layer < n_layers {
            let _cb1 = metal_ops.device.execute_async(|encoder| {
                encode_layer_range(encoder, 0..split_layer, false)
            })?;
            metal_ops.device.execute_sync(|encoder| {
                encode_layer_range(encoder, split_layer..n_layers, true)
            })?;
        } else {
            metal_ops
                .device
                .execute_sync(|encoder| encode_layer_range(encoder, 0..n_layers, true))?;
        }

        // Read back logits
        if let Some(logits) = last_logits {
            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    s.logits_buf.contents().as_ptr() as *const f32,
                    vocab_size,
                )
            };
            logits[..vocab_size].copy_from_slice(logits_gpu);
        }

        gpu_kv.finalize_batch(n_tokens);
        Ok(())
    }
}
