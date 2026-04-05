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
            && gpu_decode_quant_supported(ctx.config, weights)
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
            apply_attention_norm_single(
                weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;

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
            apply_optional_attention_qk_norm_single(
                weights,
                &prefix,
                &mut q_buf,
                &mut k_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;

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
        apply_output_norm_single(weights, &mut hidden, cfg.rms_norm_eps, ops.as_deref_mut())?;

        // --- Step 4: LM head ---
        // Weight tying: use token_embd.weight if output.weight doesn't exist.
        // Many Gemma GGUF files omit output.weight entirely (implicit tying)
        // even when the tie_word_embeddings flag is absent or false.
        write_normalized_single_logits_with_breakdown(
            ctx.backend,
            &hidden,
            dim,
            vocab_size,
            weights,
            logits,
            ops,
        )?;

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
