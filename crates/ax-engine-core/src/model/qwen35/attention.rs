use crate::compute::attention as compute_attention;

impl Qwen35Forward {
    #[allow(clippy::too_many_arguments)]
    fn full_attention_prefill_batch(
        backend: &dyn crate::backend::Backend,
        qwen_kv: &crate::kv::Qwen35Kv,
        layer: usize,
        q_batch: &[f32],
        k_batch: &[f32],
        v_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        params: &AttentionParams,
    ) {
        let prefix_len = qwen_kv.seq_len();
        if prefix_len == 0 {
            backend.attention_prefill(
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                params.n_heads,
                params.n_kv_heads,
                params.head_dim,
            );
        } else {
            compute_attention::multi_head_attention_prefill_with_prefix(
                qwen_kv.attention_k_slice_including_current(layer, prefix_len),
                qwen_kv.attention_v_slice_including_current(layer, prefix_len),
                prefix_len,
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                params,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_full_attention_prefill_gpu_with_prefix(
        backend: &dyn crate::backend::Backend,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        q_batch: &[f32],
        k_batch: &[f32],
        v_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        full_attn_params: &AttentionParams,
    ) -> anyhow::Result<bool> {
        let prefix_len = qwen_kv.seq_len();
        if prefix_len == 0 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };
        if qwen_kv.gpu_attention().is_none() {
            return Ok(false);
        }

        qwen_kv.attention_append_batch(layer, k_batch, v_batch, n_tokens);

        let q_len = n_tokens * full_attn_params.n_heads * full_attn_params.head_dim;
        debug_assert_eq!(q_batch.len(), q_len);
        debug_assert_eq!(attn_out_batch.len(), q_len);

        let Some(gpu_attention) = qwen_kv.gpu_attention() else {
            compute_attention::multi_head_attention_prefill_with_prefix(
                qwen_kv.attention_k_slice_including_current(layer, prefix_len),
                qwen_kv.attention_v_slice_including_current(layer, prefix_len),
                prefix_len,
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                full_attn_params,
            );
            return Ok(true);
        };

        let buf_q = MetalBuffer::from_slice(metal_ops.device.device(), &q_batch[..q_len])?;
        let buf_o = MetalBuffer::new(
            metal_ops.device.device(),
            q_len * std::mem::size_of::<f32>(),
        )?;
        metal_ops.device.execute_sync(|encoder| {
            if gpu_attention.is_q4() {
                metal_ops.attention.encode_attention_prefill_cached_q4kv(
                    encoder,
                    &buf_q,
                    gpu_attention.k_buffer(layer),
                    gpu_attention.v_buffer(layer),
                    &buf_o,
                    n_tokens as u32,
                    full_attn_params.n_heads as u32,
                    full_attn_params.n_kv_heads as u32,
                    full_attn_params.head_dim as u32,
                    prefix_len as u32,
                    0,
                );
            } else if gpu_attention.is_q8() {
                metal_ops.attention.encode_attention_prefill_cached_q8kv(
                    encoder,
                    &buf_q,
                    gpu_attention.k_buffer(layer),
                    gpu_attention.v_buffer(layer),
                    &buf_o,
                    n_tokens as u32,
                    full_attn_params.n_heads as u32,
                    full_attn_params.n_kv_heads as u32,
                    full_attn_params.head_dim as u32,
                    prefix_len as u32,
                    0,
                );
            } else {
                metal_ops
                    .attention
                    .encode_attention_prefill_cached_with_config(
                        encoder,
                        &buf_q,
                        gpu_attention.k_buffer(layer),
                        gpu_attention.v_buffer(layer),
                        &buf_o,
                        gpu_attention.is_f16(),
                        n_tokens as u32,
                        full_attn_params.n_heads as u32,
                        full_attn_params.n_kv_heads as u32,
                        full_attn_params.head_dim as u32,
                        prefix_len as u32,
                        0,
                        metal_ops.attention_dispatch_config(),
                    );
            }
            Ok(())
        })?;

        let result = unsafe { buf_o.as_slice::<f32>() };
        attn_out_batch[..q_len].copy_from_slice(&result[..q_len]);
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_full_attention_decode_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        qwen_kv: &crate::kv::Qwen35Kv,
        layer: usize,
        q: &[f32],
        attn_out: &mut [f32],
        full_attn_params: &AttentionParams,
    ) -> anyhow::Result<bool> {
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };
        let Some(gpu_attention) = qwen_kv.gpu_attention() else {
            return Ok(false);
        };

        metal_ops.init_scratches(cfg);
        let scratch_guard = metal_ops.scratches();
        let Some(scratches) = scratch_guard.as_ref() else {
            return Ok(false);
        };

        let q_len = full_attn_params.n_heads * full_attn_params.head_dim;
        debug_assert_eq!(q.len(), q_len);
        debug_assert_eq!(attn_out.len(), q_len);
        unsafe {
            let q_dst = scratches.q_buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(q.as_ptr(), q_dst, q_len);
        }

        metal_ops.device.execute_sync(|encoder| {
            if gpu_attention.is_q4() {
                metal_ops.attention.encode_attention_decode_q4kv(
                    encoder,
                    &scratches.q_buf,
                    gpu_attention.k_buffer(layer),
                    gpu_attention.v_buffer(layer),
                    &scratches.attn_out,
                    full_attn_params.n_heads as u32,
                    full_attn_params.n_kv_heads as u32,
                    full_attn_params.head_dim as u32,
                    0,
                    (qwen_kv.seq_len() + 1) as u32,
                );
            } else if gpu_attention.is_q8() {
                if full_attn_params.head_dim == 128 {
                    metal_ops.attention.encode_attention_decode_q8kv(
                        encoder,
                        &scratches.q_buf,
                        gpu_attention.k_buffer(layer),
                        gpu_attention.v_buffer(layer),
                        &scratches.attn_out,
                        full_attn_params.n_heads as u32,
                        full_attn_params.n_kv_heads as u32,
                        full_attn_params.head_dim as u32,
                        0,
                        (qwen_kv.seq_len() + 1) as u32,
                    );
                } else if full_attn_params.head_dim == 256 {
                    metal_ops.attention.encode_attention_decode_q8kv_hd256(
                        encoder,
                        &scratches.q_buf,
                        gpu_attention.k_buffer(layer),
                        gpu_attention.v_buffer(layer),
                        &scratches.attn_out,
                        full_attn_params.n_heads as u32,
                        full_attn_params.n_kv_heads as u32,
                        full_attn_params.head_dim as u32,
                        0,
                        (qwen_kv.seq_len() + 1) as u32,
                    );
                }
            } else {
                metal_ops
                    .attention
                    .encode_attention_decode_with_scratch_and_config(
                        encoder,
                        &scratches.q_buf,
                        gpu_attention.k_buffer(layer),
                        gpu_attention.v_buffer(layer),
                        &scratches.attn_out,
                        &scratches.splitk_partial_out,
                        &scratches.splitk_partial_lse,
                        gpu_attention.is_f16(),
                        full_attn_params.n_heads as u32,
                        full_attn_params.n_kv_heads as u32,
                        full_attn_params.head_dim as u32,
                        0,
                        (qwen_kv.seq_len() + 1) as u32,
                        metal_ops.attention_dispatch_config(),
                    );
            }
            Ok(())
        })?;

        unsafe {
            let attn_src = scratches.attn_out.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(attn_src, attn_out.as_mut_ptr(), q_len);
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_last_batch_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &mut [f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let last_hidden = &mut hidden[(n_tokens - 1) * dim..n_tokens * dim];
        Self::write_single_logits(
            backend,
            last_hidden,
            dim,
            vocab_size,
            rms_norm_eps,
            weights,
            logits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_post_attention_ffn_batch(
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<()> {
        let ffn_norm_w = weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
        crate::model::layer_ops::apply_ffn_batch(
            backend,
            weights,
            prefix,
            hidden,
            norm_buf,
            gate_buf,
            up_buf,
            down_buf,
            n_tokens,
            dim,
            inter_dim,
            ffn_norm_w,
            rms_norm_eps,
            crate::model::layer_ops::FfnActivation::SiLU,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_post_attention_ffn_single(
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        if ops.is_some() {
            // Profiled path: keep inline for per-op timing.
            let mut ops = ops;
            let ffn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?
            );
            timed!(
                ops,
                norm,
                rms_norm::rms_norm_out(hidden, ffn_norm_w, norm_buf, rms_norm_eps)
            );
            let input_ops = Self::ffn_input_ops(weights, prefix, inter_dim)?;
            timed_matmul_bucket!(ops, matmul_input_proj, {
                let mut outputs = [&mut *gate_buf, &mut *up_buf];
                Self::decode_project_ops_gpu_safe(backend, &input_ops, norm_buf, dim, &mut outputs);
            });
            silu::silu_elementwise_mul(gate_buf, up_buf);
            let (wd_raw, wd_dtype, _) =
                timed!(ops, dequant, Self::ffn_down_op(weights, prefix, dim)?);
            timed_matmul_bucket!(
                ops,
                matmul_output_proj,
                Self::decode_dequant_matmul_gpu_safe(
                    backend, wd_raw, wd_dtype, gate_buf, down_buf, dim, inter_dim
                )
            );
            silu::elementwise_add(hidden, down_buf);
        } else {
            // Non-profiled path: delegate to shared implementation.
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
            crate::model::layer_ops::apply_ffn_single(
                backend,
                weights,
                prefix,
                hidden,
                norm_buf,
                gate_buf,
                up_buf,
                down_buf,
                dim,
                inter_dim,
                ffn_norm_w,
                rms_norm_eps,
                crate::model::layer_ops::FfnActivation::SiLU,
            );
        }
        Ok(())
    }

    /// GPU-encoded attention norm + input projections in a single command buffer.
    ///
    /// Encodes: RMSNorm(hidden) → norm_buf, then for each (weight, out_dim):
    /// fused dequant matmul → output buffer. All in one execute_sync.
    ///
    /// Returns `true` if GPU path was used, `false` to fall back to CPU.
    #[allow(clippy::too_many_arguments)]
    fn try_norm_and_project_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &[f32],
        norm_buf: &mut [f32],
        projections: &[QuantOp<'_>],
        outputs: &mut [&mut [f32]],
        n_tokens: usize,
        dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<bool> {
        debug_assert_eq!(projections.len(), outputs.len());
        if n_tokens <= 1 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };

        if !projections.iter().all(|(_, dtype, out_dim)| {
            Self::qwen35_batch_projection_supported(
                metal_ops,
                *dtype,
                *out_dim as u32,
                n_tokens as u32,
                dim as u32,
            )
        }) {
            return Ok(false);
        }

        // Cache all weights + norm.
        let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
        let nw_key = metal_ops.ensure_f32_cached(attn_norm_w);
        let mut proj_keys = Vec::with_capacity(projections.len());
        for (raw, dtype, out_dim) in projections {
            Self::prepare_qwen35_batch_projection_weight(
                metal_ops,
                raw,
                *dtype,
                *out_dim as u32,
                dim as u32,
            )?;
            let key = raw.as_ptr() as usize;
            metal_ops.ensure_quant_cached(raw);
            proj_keys.push(key);
        }

        metal_ops.init_batch_scratches(cfg, n_tokens);

        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        // Copy hidden to GPU.
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(hidden);
        }

        let weight_cache = metal_ops.lock_weight_cache();
        let nw_buf = weight_cache.get(&nw_key).unwrap();

        let output_dims: Vec<usize> = projections.iter().map(|(_, _, out_dim)| *out_dim).collect();
        metal_ops.with_qwen35_batch_projection_scratch(n_tokens, &output_dims, |temp_scratch| {
            metal_ops.device.execute_sync(|encoder| {
                // 1. RMSNorm: hidden → norm_buf
                metal_ops.elementwise.encode_rms_norm_out_batch(
                    encoder,
                    &bs.hidden,
                    nw_buf,
                    &bs.norm_buf,
                    dim as u32,
                    n_tokens as u32,
                    rms_norm_eps,
                );
                if projections
                    .iter()
                    .any(|(_, dtype, _)| Self::qwen35_batch_projection_needs_f16_input(*dtype))
                {
                    metal_ops.elementwise.encode_cast_f32_to_f16(
                        encoder,
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        (n_tokens * dim) as u32,
                    );
                }
                // 2. All input projections from norm_buf.
                for (i, (_, dtype, out_dim)) in projections.iter().enumerate() {
                    let w_buf = weight_cache.get(&proj_keys[i]).unwrap();
                    Self::encode_qwen35_batch_projection(
                        metal_ops,
                        encoder,
                        w_buf,
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        &temp_scratch.outputs[i],
                        *out_dim as u32,
                        n_tokens as u32,
                        dim as u32,
                        *dtype,
                    );
                }
                Ok(())
            })?;

            // Read back norm_buf and projection outputs.
            let norm_result = unsafe { &bs.norm_buf.as_slice::<f32>()[..n_tokens * dim] };
            norm_buf[..n_tokens * dim].copy_from_slice(norm_result);

            for (i, (_, _, out_dim)) in projections.iter().enumerate() {
                let result =
                    unsafe { &temp_scratch.outputs[i].as_slice::<f32>()[..n_tokens * out_dim] };
                outputs[i][..n_tokens * out_dim].copy_from_slice(result);
            }
            Ok::<(), anyhow::Error>(())
        })?;
        drop(weight_cache);
        drop(bs_guard);

        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_norm_batch(
        weights: &WeightStore,
        prefix: &str,
        hidden: &[f32],
        norm_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<()> {
        let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
        Self::rms_norm_token_major(hidden, attn_norm_w, norm_buf, n_tokens, dim, rms_norm_eps);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// GPU-accelerated batch Q/K preparation: Q extraction + QK norm + RoPE.
    ///
    /// Replaces `prepare_full_attention_qk_batch` per-token CPU loops with
    /// GPU batch dispatches. Falls back to CPU when Metal is unavailable.
    #[allow(clippy::too_many_arguments)]
    fn try_prepare_full_attention_qk_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        q_gate_batch: &[f32],
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        n_tokens: usize,
        start_position: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<bool> {
        if n_tokens <= 1 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };

        // CPU: extract Q from Q+gate (strided memcpy — fast, ~1ms for 512 tokens).
        for token_idx in 0..n_tokens {
            let src = token_idx * q_dim * 2;
            let dst = token_idx * q_dim;
            q_batch[dst..dst + q_dim].copy_from_slice(&q_gate_batch[src..src + q_dim]);
        }

        // Cache QK norm weights on GPU if they exist.
        let norm_weights = Self::maybe_attention_qk_norm(weights, prefix)?;
        let q_norm_key = norm_weights.map(|nw| metal_ops.ensure_f32_cached(nw.q));
        let k_norm_key = norm_weights.map(|nw| metal_ops.ensure_f32_cached(nw.k));

        // Upload Q and K to GPU batch scratch buffers.
        metal_ops.init_batch_scratches(cfg, n_tokens);
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };
        unsafe {
            bs.q_buf.as_mut_slice::<f32>()[..n_tokens * q_dim]
                .copy_from_slice(&q_batch[..n_tokens * q_dim]);
            bs.k_buf.as_mut_slice::<f32>()[..n_tokens * kv_dim]
                .copy_from_slice(&k_batch[..n_tokens * kv_dim]);
        }

        let (rope_start, rope_step) = cfg.rope_scaling.scaled_start_step(start_position);
        let weight_cache = metal_ops.lock_weight_cache();

        metal_ops.device.execute_sync(|encoder| {
            // 1. Per-head QK norm (if applicable).
            if let (Some(q_key), Some(k_key)) = (q_norm_key, k_norm_key) {
                let q_nw = weight_cache.get(&q_key).unwrap();
                let k_nw = weight_cache.get(&k_key).unwrap();
                metal_ops.elementwise.encode_per_head_rms_norm_batch(
                    encoder,
                    &bs.q_buf,
                    q_nw,
                    n_tokens as u32,
                    n_heads as u32,
                    head_dim as u32,
                    cfg.rms_norm_eps,
                );
                metal_ops.elementwise.encode_per_head_rms_norm_batch(
                    encoder,
                    &bs.k_buf,
                    k_nw,
                    n_tokens as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    cfg.rms_norm_eps,
                );
            }

            // 2. Batch RoPE on Q and K.
            metal_ops.elementwise.encode_rope_batch(
                encoder,
                &bs.q_buf,
                &bs.k_buf,
                n_tokens as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                rope_start,
                rope_step,
                cfg.rope_freq_base,
            );
            Ok(())
        })?;
        drop(weight_cache);

        // Read back Q and K.
        unsafe {
            q_batch[..n_tokens * q_dim]
                .copy_from_slice(&bs.q_buf.as_slice::<f32>()[..n_tokens * q_dim]);
            k_batch[..n_tokens * kv_dim]
                .copy_from_slice(&bs.k_buf.as_slice::<f32>()[..n_tokens * kv_dim]);
        }
        drop(bs_guard);
        Ok(true)
    }

    /// GPU-accelerated attention gate: sigmoid(gate) × attn_out.
    ///
    /// Replaces `apply_attention_gate_batch` per-token CPU loops with a single
    /// GPU dispatch. Falls back to CPU when Metal is unavailable.
    #[allow(clippy::too_many_arguments)]
    fn try_apply_attention_gate_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        q_gate_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        q_dim: usize,
    ) -> anyhow::Result<bool> {
        if n_tokens <= 1 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };

        let total = n_tokens * q_dim;

        // CPU: extract gate values to contiguous buffer (strided copy).
        let mut gate_contiguous = vec![0.0f32; total];
        for token_idx in 0..n_tokens {
            let src = token_idx * q_dim * 2 + q_dim;
            let dst = token_idx * q_dim;
            gate_contiguous[dst..dst + q_dim].copy_from_slice(&q_gate_batch[src..src + q_dim]);
        }

        metal_ops.init_batch_scratches(cfg, n_tokens);
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        // Upload gate and attn_out to GPU scratch.
        // gate → gate_buf (inter_dim >= q_dim), attn_out → attn_out (q_dim).
        unsafe {
            bs.gate_buf.as_mut_slice::<f32>()[..total].copy_from_slice(&gate_contiguous);
            bs.attn_out.as_mut_slice::<f32>()[..total].copy_from_slice(attn_out_batch);
        }

        metal_ops.device.execute_sync(|encoder| {
            metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                encoder,
                &bs.gate_buf,
                &bs.attn_out,
                total as u32,
            );
            Ok(())
        })?;

        // Read back attn_out.
        unsafe {
            attn_out_batch[..total].copy_from_slice(&bs.attn_out.as_slice::<f32>()[..total]);
        }
        drop(bs_guard);
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_full_attention_batch_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        batch_position: usize,
        norm_buf: &[f32],
        q_gate_batch: &mut [f32],
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
        fused_input_batch: &mut [f32],
        attn_out_batch: &mut [f32],
        proj_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        full_attn_params: &AttentionParams,
        skip_input_projections: bool,
    ) -> anyhow::Result<()> {
        if !skip_input_projections {
            let input_plan = Self::full_attention_input_plan(weights, prefix, q_dim, kv_dim)?;
            Self::project_full_attention_inputs_batch(
                input_plan,
                q_gate_batch,
                k_batch,
                v_batch,
                fused_input_batch,
                n_tokens,
                |raw, dtype, rows, out| {
                    Self::batched_dequant_matmul_token_major(
                        backend, raw, dtype, norm_buf, out, n_tokens, rows, dim,
                    );
                },
            );
        }

        // GPU-accelerated Q extraction + QK norm + RoPE (replaces per-token CPU loops).
        let used_gpu_qk = Self::try_prepare_full_attention_qk_batch_gpu(
            cfg,
            backend,
            weights,
            prefix,
            q_gate_batch,
            q_batch,
            k_batch,
            n_tokens,
            qwen_kv.seq_len(),
            q_dim,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
        )?;
        if !used_gpu_qk {
            Self::prepare_full_attention_qk_batch(
                cfg,
                q_gate_batch,
                q_batch,
                k_batch,
                n_tokens,
                qwen_kv.seq_len(),
                q_dim,
                kv_dim,
                n_heads,
                n_kv_heads,
                head_dim,
                Self::maybe_attention_qk_norm(weights, prefix)?,
                cfg.rms_norm_eps,
            );
        }

        let used_gpu_prefix_kv = Self::try_full_attention_prefill_gpu_with_prefix(
            backend,
            qwen_kv,
            layer,
            q_batch,
            k_batch,
            v_batch,
            attn_out_batch,
            n_tokens,
            full_attn_params,
        )?;
        if !used_gpu_prefix_kv {
            Self::full_attention_prefill_batch(
                backend,
                qwen_kv,
                layer,
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                full_attn_params,
            );
        }

        // GPU-accelerated attention gate (replaces per-token CPU sigmoid+mul).
        let used_gpu_gate = Self::try_apply_attention_gate_batch_gpu(
            cfg,
            backend,
            q_gate_batch,
            attn_out_batch,
            n_tokens,
            q_dim,
        )?;
        if !used_gpu_gate {
            Self::apply_attention_gate_batch(q_gate_batch, attn_out_batch, n_tokens, q_dim);
        }

        if !used_gpu_prefix_kv {
            qwen_kv.attention_append_batch(layer, k_batch, v_batch, n_tokens);
        }

        let (wo_raw, wo_dtype, _) = Self::full_attention_output_op(weights, prefix, dim)?;
        Self::batched_dequant_matmul_token_major(
            backend,
            wo_raw,
            wo_dtype,
            attn_out_batch,
            proj_buf,
            n_tokens,
            dim,
            q_dim,
        );
        Self::assert_finite_if_enabled(
            "full_attention_proj_batch",
            proj_buf,
            layer,
            batch_position,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_full_attention_single_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        position: usize,
        norm_buf: &[f32],
        q_gate_buf: &mut [f32],
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        proj_buf: &mut [f32],
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        full_attn_params: &AttentionParams,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let input_ops = Self::full_attention_input_ops(weights, prefix, q_dim, kv_dim)?;

        timed_matmul_bucket!(ops, matmul_input_proj, {
            let mut outputs = [&mut *q_gate_buf, &mut *k_buf, &mut *v_buf];
            Self::decode_project_ops_gpu_safe(backend, &input_ops, norm_buf, dim, &mut outputs);
        });
        Self::extract_q_from_q_gate(q_gate_buf, q_buf);
        let gate_attn = &mut q_gate_buf[q_dim..];

        if let Some(norm_weights) = timed!(
            ops,
            dequant,
            Self::maybe_attention_qk_norm(weights, prefix)?
        ) {
            timed!(ops, norm, {
                Self::apply_attention_qk_norm(
                    q_buf,
                    k_buf,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    norm_weights,
                    cfg.rms_norm_eps,
                );
            });
        }

        timed!(
            ops,
            rope,
            Self::apply_rope(cfg, q_buf, k_buf, position, n_heads, n_kv_heads, head_dim,)
        );

        qwen_kv.attention_append(layer, k_buf, v_buf);
        let seq_len = qwen_kv.seq_len() + 1;
        let used_gpu_attention = timed!(
            ops,
            attention,
            Self::try_full_attention_decode_gpu(
                cfg,
                backend,
                qwen_kv,
                layer,
                q_buf,
                attn_out,
                full_attn_params,
            )?
        );
        if !used_gpu_attention {
            timed!(
                ops,
                attention,
                compute_attention::multi_head_attention(
                    q_buf,
                    qwen_kv.attention_k_slice_including_current(layer, seq_len),
                    qwen_kv.attention_v_slice_including_current(layer, seq_len),
                    attn_out,
                    full_attn_params,
                    seq_len,
                )
            );
        }

        Self::apply_attention_gate(gate_attn, attn_out);

        let (wo_raw, wo_dtype, _) = timed!(
            ops,
            dequant,
            Self::full_attention_output_op(weights, prefix, dim)?
        );
        timed_matmul_bucket!(
            ops,
            matmul_output_proj,
            Self::decode_dequant_matmul_gpu_safe(
                backend, wo_raw, wo_dtype, attn_out, proj_buf, dim, q_dim
            )
        );
        Self::assert_finite_if_enabled("full_attention_proj", proj_buf, layer, position)?;
        Ok(())
    }

}
