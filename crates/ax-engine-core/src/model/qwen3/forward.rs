impl ForwardPass for Qwen3Forward {
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
        ensure_supported_qwen3_layout(weights, ctx.config)?;

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

        assert!(logits.len() >= vocab_size);

        let cpu_kv = kv
            .as_cpu_mut()
            .expect("Qwen3Forward CPU path requires ModelKv::Cpu");

        // --- Step 1: Token embedding (single-row dequant) ---
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
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; dim];

        // --- Step 2: Transformer layers ---
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            // 2a. Attention norm
            apply_attention_norm_single(
                weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;

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

            // 2c. Qwen3: add QKV bias terms (CPU path: dequantize bias and add in-place)
            if weights.has(&format!("{prefix}.attn_q.bias")) {
                let q_bias = weights.dequantize(&format!("{prefix}.attn_q.bias"))?;
                silu::elementwise_add(&mut q_buf, &q_bias);
            }
            if weights.has(&format!("{prefix}.attn_k.bias")) {
                let k_bias = weights.dequantize(&format!("{prefix}.attn_k.bias"))?;
                silu::elementwise_add(&mut k_buf, &k_bias);
            }
            if weights.has(&format!("{prefix}.attn_v.bias")) {
                let v_bias = weights.dequantize(&format!("{prefix}.attn_v.bias"))?;
                silu::elementwise_add(&mut v_buf, &v_bias);
            }

            // 2d. Per-head QK normalization (Qwen3: applied before RoPE, same as Gemma3)
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

            // 2e. RoPE on Q and K (apply linear scaling if configured)
            let rope_position = cfg.rope_scaling.scaled_position(position);
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

            // 2f. Update KV cache (v2 CPU path: append_and_advance per layer, finalize after last)
            cpu_kv.append_and_advance(layer, &k_buf, &v_buf);

            // 2g. Multi-head attention
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

            // 2h. Output projection
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

            // 2i. Residual add
            silu::elementwise_add(&mut hidden, &proj_buf);

            // 2j-2n. FFN: norm → gate/up → SiLU → down → residual
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
                crate::model::layer_ops::FfnActivation::SiLU,
            );
        }

        // Advance CPU KV cache after all layers processed
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
            ops.as_deref_mut(),
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
        let prefill_plan =
            PrefillExecutionPlan::for_forward_batch(ctx, kv, weights, token_ids.len(), false)?;

        if prefill_plan.mode == PrefillMode::GpuBatch {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
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

        // Fallback: serial forward_single calls
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

        if prefill_plan.mode == PrefillMode::GpuBatch {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
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
                        "GPU batch all-logits prefill failed, falling back to serial: {e}"
                    );
                }
            }
        }

        ForwardPass::forward_batch_all_logits(self, ctx, token_ids, kv, weights, logits_all)
    }

    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()> {
        if !config.has_qkv_bias {
            tracing::info!(
                "Qwen3Forward: no QKV bias configured (qwen3.attention.bias absent or false); \
                 bias weights will be used if present in GGUF tensors"
            );
        }
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "qwen3"
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
        weights
            .dequantize_row("token_embd.weight", token_id as usize, hidden)
            .map(|_| ())
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
        let frame = encode_qwen3_pending_step(
            metal_ops, ctx.config, hidden_buf, position, gpu_kv, weights,
        )?;
        Ok(Some(frame))
    }
}

