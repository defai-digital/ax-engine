impl ForwardPass for Qwen3MoeForward {
    fn prepare_runtime(
        &self,
        ctx: &ForwardContext,
        weights: &WeightStore,
    ) -> anyhow::Result<()> {
        if let Some(metal_ops) = ctx.backend.metal_ops()
            && Self::moe_gpu_decode_supported(ctx.config, weights)
            && Self::moe_gpu_expert_dispatch_supported(weights)
            && !metal_ops.has_cached_model_keys()
        {
            Self::build_cached_model_keys(metal_ops, weights, ctx.config)?;
        }

        Ok(())
    }

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
        // GPU decode path — requires both attention and MoE expert weights
        // to be GPU-compatible. Attention weights must be Q4K/Q5K/Q6K/Q8_0,
        // and routed expert weights may mix Q4K/Q5K/Q6K/Q8_0 across gate/up/down.
        if ctx.backend.use_gpu_decode()
            && let Some(metal_ops) = ctx.backend.metal_ops()
            && let Some(gpu_kv) = kv.as_gpu_mut()
            && Self::moe_gpu_decode_supported(ctx.config, weights)
            && Self::moe_gpu_expert_dispatch_supported(weights)
        {
            return self.forward_single_gpu_unified(
                ctx, metal_ops, token_id, position, gpu_kv, weights, logits, ops,
            );
        }

        let cfg = ctx.config;
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

        assert!(logits.len() >= vocab_size);

        let cpu_kv = kv
            .as_cpu_mut()
            .expect("Qwen3MoeForward CPU path requires ModelKv::Cpu");

        // --- Step 1: Token embedding ---
        let mut hidden = vec![0.0f32; dim];
        timed!(
            ops,
            dequant,
            weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?
        );

        // Scratch buffers
        let mut norm_buf = vec![0.0f32; dim];
        let mut q_buf = vec![0.0f32; q_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut moe_scratch = MoeSingleScratch {
            gate_buf: vec![0.0f32; expert_inter_dim],
            up_buf: vec![0.0f32; expert_inter_dim],
            down_buf: vec![0.0f32; dim],
            accum_buf: vec![0.0f32; dim],
            router_logits: vec![0.0f32; n_expert],
        };

        // --- Step 2: Transformer layers ---
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            // 2a. Pre-attention RMSNorm
            apply_attention_norm_single(
                weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;

            // 2b. Q/K/V projections (separate)
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
                    &mut [
                        &mut q_buf[..q_dim],
                        &mut k_buf[..kv_dim],
                        &mut v_buf[..kv_dim],
                    ],
                );
            });

            // 2c. Per-head QK normalization
            apply_optional_attention_qk_norm_single(
                weights,
                &prefix,
                &mut q_buf[..q_dim],
                &mut k_buf[..kv_dim],
                n_heads,
                n_kv_heads,
                head_dim,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;

            // 2d. RoPE (NeoX style)
            timed!(
                ops,
                rope,
                rope::apply_rope_multi_head_neox_partial_scaled(
                    &mut q_buf[..q_dim],
                    &mut k_buf[..kv_dim],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    head_dim,
                    position as f32,
                    cfg.rope_freq_base,
                )
            );

            // 2e. Update KV cache
            cpu_kv.append_and_advance(layer, &k_buf[..kv_dim], &v_buf[..kv_dim]);

            // 2f. Multi-head attention
            let full_seq_len = cpu_kv.seq_len() + 1;
            let k_cache = cpu_kv.k_slice_including_current(layer, full_seq_len);
            let v_cache = cpu_kv.v_slice_including_current(layer, full_seq_len);
            let attn_params = attention::AttentionParams::new(n_heads, n_kv_heads, head_dim);
            timed!(
                ops,
                attention,
                attention::multi_head_attention(
                    &q_buf[..q_dim],
                    k_cache,
                    v_cache,
                    &mut attn_out[..q_dim],
                    &attn_params,
                    full_seq_len,
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
                    &attn_out[..q_dim],
                    &mut proj_buf,
                    dim,
                    1,
                    q_dim,
                )
            );

            // 2h. Residual add
            silu::elementwise_add(&mut hidden, &proj_buf);

            // 2i. MoE FFN
            Self::apply_moe_ffn_single(
                ctx.backend,
                weights,
                &prefix,
                &mut hidden,
                &mut norm_buf,
                &mut moe_scratch,
                dim,
                n_expert,
                n_expert_used,
                expert_inter_dim,
                cfg.rms_norm_eps,
            )?;
        }

        // Advance CPU KV cache
        cpu_kv.finalize_token();

        // --- Step 3: Final RMSNorm ---
        apply_output_norm_single(weights, &mut hidden, cfg.rms_norm_eps, ops.as_deref_mut())?;

        // --- Step 4: LM head ---
        write_normalized_single_logits_with_breakdown(
            ctx.backend,
            &hidden,
            dim,
            vocab_size,
            weights,
            logits,
            ops,
        )?;

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
        let n_tokens = token_ids.len();
        if n_tokens <= 1 {
            if n_tokens == 1 {
                return self.forward_single(ctx, token_ids[0], kv.seq_len(), kv, weights, logits, None);
            }
            return Ok(());
        }

        // GPU batch prefill path
        let prefill_plan = crate::model::execution_plan::PrefillExecutionPlan::for_forward_batch(
            ctx,
            kv,
            weights,
            n_tokens,
            false,
        )?;
        if matches!(
            prefill_plan.mode,
            crate::model::execution_plan::PrefillMode::GpuBatch
                | crate::model::execution_plan::PrefillMode::GpuChunked
        ) {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
            return self.forward_batch_gpu_unified(
                ctx, metal_ops, token_ids, gpu_kv, weights, Some(logits), None, None,
            );
        }

        // CPU batch path
        self.forward_batch_token_major(ctx, token_ids, kv, weights, logits, None)
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
        let n_tokens = token_ids.len();
        if n_tokens <= 1 {
            if n_tokens == 1 {
                return self.forward_single(
                    ctx,
                    token_ids[0],
                    kv.seq_len(),
                    kv,
                    weights,
                    logits,
                    Some(ops),
                );
            }
            return Ok(());
        }

        // Mirror the non-profiled forward_batch GPU dispatch logic.
        let prefill_plan = crate::model::execution_plan::PrefillExecutionPlan::for_forward_batch(
            ctx,
            kv,
            weights,
            n_tokens,
            false,
        )?;
        if matches!(
            prefill_plan.mode,
            crate::model::execution_plan::PrefillMode::GpuBatch
                | crate::model::execution_plan::PrefillMode::GpuChunked
        ) {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
            return self.forward_batch_gpu_unified(
                ctx,
                metal_ops,
                token_ids,
                gpu_kv,
                weights,
                Some(logits),
                None,
                Some(ops),
            );
        }

        self.forward_batch_token_major(ctx, token_ids, kv, weights, logits, Some(ops))
    }

    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.n_expert.is_some() && config.n_expert.unwrap() > 0,
            "qwen3moe requires n_expert > 0"
        );
        anyhow::ensure!(
            config.n_expert_used.is_some() && config.n_expert_used.unwrap() > 0,
            "qwen3moe requires n_expert_used > 0"
        );
        anyhow::ensure!(
            config.expert_intermediate_dim.is_some()
                && config.expert_intermediate_dim.unwrap() > 0,
            "qwen3moe requires expert_intermediate_dim > 0"
        );
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "qwen3moe"
    }
}

impl Qwen3MoeForward {
    /// Check if GPU decode is supported for MoE models.
    /// MoE models don't have dense FFN weights, so the standard check fails.
    /// This checks only the attention weights; MoE FFN dispatch handles its
    /// own quant support at encode time.
    pub fn moe_gpu_decode_supported(config: &ModelConfig, weights: &WeightStore) -> bool {
        use crate::model::shared::gpu_decode_quant_dtype_supported;
        let attn_suffixes = ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"];
        for layer in 0..config.n_layers {
            for suffix in &attn_suffixes {
                let name = format!("blk.{layer}.{suffix}");
                if let Ok((_, dtype)) = weights.raw_with_dtype(&name)
                    && !gpu_decode_quant_dtype_supported(dtype)
                {
                    return false;
                }
            }
        }
        true
    }

    pub(crate) fn moe_gpu_expert_dtype_supported(dtype: crate::gguf::tensor::GgmlType) -> bool {
        crate::model::shared::moe_routed_expert_dtype_supported(dtype)
    }

    /// Check if MoE expert weights support the GPU mul_mat_id path.
    pub fn moe_gpu_expert_dispatch_supported(weights: &WeightStore) -> bool {
        let check = |name: &str| -> bool {
            if let Ok((_, dtype)) = weights.raw_with_dtype(name) {
                Self::moe_gpu_expert_dtype_supported(dtype)
            } else {
                false
            }
        };
        check("blk.0.ffn_gate_exps.weight")
            && check("blk.0.ffn_up_exps.weight")
            && check("blk.0.ffn_down_exps.weight")
    }

    /// CPU batch token-major forward pass.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_token_major(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
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

        let mut scratch = Qwen3MoeBatchScratch::new(cfg, n_tokens);

        // Token embedding
        timed!(ops, dequant, {
            for (i, &tid) in token_ids.iter().enumerate() {
                let row = &mut scratch.hidden[i * dim..(i + 1) * dim];
                weights.dequantize_row("token_embd.weight", tid as usize, row)?;
            }
            anyhow::Ok(())
        })?;

        let base_seq_len = kv.seq_len();

        // Transformer layers
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            // Pre-attention RMSNorm
            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            timed!(ops, norm, {
                for t in 0..n_tokens {
                    let start = t * dim;
                    rms_norm::rms_norm_out(
                        &scratch.hidden[start..start + dim],
                        attn_norm_w,
                        &mut scratch.norm_buf[start..start + dim],
                        cfg.rms_norm_eps,
                    );
                }
            });

            // Q/K/V projections
            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
            timed!(ops, matmul, {
                ctx.backend.dequant_matmul_token_major(
                    wq_raw, wq_dtype, &scratch.norm_buf, &mut scratch.q_buf,
                    n_tokens, q_dim, dim,
                );
                ctx.backend.dequant_matmul_token_major(
                    wk_raw, wk_dtype, &scratch.norm_buf, &mut scratch.k_buf,
                    n_tokens, kv_dim, dim,
                );
                ctx.backend.dequant_matmul_token_major(
                    wv_raw, wv_dtype, &scratch.norm_buf, &mut scratch.v_buf,
                    n_tokens, kv_dim, dim,
                );
            });

            // Per-head QK norm
            let qk_norm_weights = crate::model::shared::maybe_attention_qk_norm_weights(
                weights, &prefix,
            )?;
            if let Some(norm_weights) = qk_norm_weights {
                timed!(ops, norm, {
                    for t in 0..n_tokens {
                        crate::model::shared::apply_attention_qk_norm(
                            &mut scratch.q_buf[t * q_dim..(t + 1) * q_dim],
                            &mut scratch.k_buf[t * kv_dim..(t + 1) * kv_dim],
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            norm_weights,
                            cfg.rms_norm_eps,
                        );
                    }
                });
            }

            // RoPE
            timed!(ops, rope, {
                for t in 0..n_tokens {
                    let position = base_seq_len + t;
                    rope::apply_rope_multi_head_neox_partial_scaled(
                        &mut scratch.q_buf[t * q_dim..(t + 1) * q_dim],
                        &mut scratch.k_buf[t * kv_dim..(t + 1) * kv_dim],
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        head_dim,
                        position as f32,
                        cfg.rope_freq_base,
                    );
                }
            });

            // Append to KV cache
            match kv {
                ModelKv::Cpu(cpu_kv) => {
                    cpu_kv.append_batch(layer, &scratch.k_buf, &scratch.v_buf, n_tokens);
                }
                ModelKv::Gpu(gpu_kv) => {
                    gpu_kv.append_layer_batch(layer, &scratch.k_buf, &scratch.v_buf, n_tokens);
                }
                ModelKv::Qwen35(_) => unreachable!("qwen3moe does not use qwen35 kv"),
            }

            // Multi-head attention (use backend prefill when no prefix context)
            if base_seq_len == 0 {
                timed!(ops, attention, ctx.backend.attention_prefill(
                    &scratch.q_buf,
                    &scratch.k_buf,
                    &scratch.v_buf,
                    &mut scratch.attn_out,
                    n_tokens,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                ));
            } else {
                let attn_params = attention::AttentionParams::new(n_heads, n_kv_heads, head_dim);
                match kv {
                    ModelKv::Cpu(cpu_kv) => {
                        let prefix_k = &cpu_kv.k_slice(layer, base_seq_len)[..base_seq_len * kv_dim];
                        let prefix_v = &cpu_kv.v_slice(layer, base_seq_len)[..base_seq_len * kv_dim];
                        timed!(ops, attention,
                            attention::multi_head_attention_prefill_with_prefix(
                                prefix_k,
                                prefix_v,
                                base_seq_len,
                                &scratch.q_buf,
                                &scratch.k_buf,
                                &scratch.v_buf,
                                &mut scratch.attn_out,
                                n_tokens,
                                &attn_params,
                            )
                        );
                    }
                    _ => {
                        // GPU KV: read back prefix and use CPU attention
                        timed!(ops, attention, ctx.backend.attention_prefill(
                            &scratch.q_buf,
                            &scratch.k_buf,
                            &scratch.v_buf,
                            &mut scratch.attn_out,
                            n_tokens,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                        ));
                    }
                }
            }

            // Output projection
            let (wo_raw, wo_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            timed!(ops, matmul, ctx.backend.dequant_matmul_token_major(
                wo_raw, wo_dtype, &scratch.attn_out, &mut scratch.proj_buf,
                n_tokens, dim, q_dim,
            ));

            // Residual add
            Self::parallel_elementwise_add(&mut scratch.hidden, &scratch.proj_buf);

            // MoE FFN
            Self::apply_moe_ffn_batch(
                ctx.backend,
                weights,
                &prefix,
                &mut scratch.hidden,
                &mut scratch.moe_norm_buf,
                &mut scratch.moe_accum_buf,
                &mut scratch.moe_scratch,
                n_tokens,
                dim,
                n_expert,
                n_expert_used,
                expert_inter_dim,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;
        }

        // Finalize KV
        match kv {
            ModelKv::Cpu(cpu_kv) => cpu_kv.finalize_batch(n_tokens),
            ModelKv::Gpu(gpu_kv) => gpu_kv.finalize_batch(n_tokens),
            ModelKv::Qwen35(_) => unreachable!(),
        }

        // Final RMSNorm + logits (last token only)
        let last_hidden = &scratch.hidden[(n_tokens - 1) * dim..n_tokens * dim];
        let mut final_hidden = last_hidden.to_vec();
        apply_output_norm_single(weights, &mut final_hidden, cfg.rms_norm_eps, ops.as_deref_mut())?;
        crate::model::shared::write_normalized_single_logits_with_breakdown(
            ctx.backend,
            &final_hidden,
            dim,
            vocab_size,
            weights,
            logits,
            ops,
        )?;

        Ok(())
    }
}
