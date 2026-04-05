impl ForwardPass for Gemma4Forward {
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
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let vocab_size = cfg.vocab_size as usize;

        // Max dimensions across all layer types (for scratch buffer allocation)
        let max_hd = cfg
            .gemma4_head_dim_global
            .unwrap_or(cfg.head_dim)
            .max(cfg.gemma4_head_dim_swa.unwrap_or(cfg.head_dim))
            as usize;
        let max_q_dim = n_heads * max_hd;
        let max_kv_dim = cfg
            .gemma4_n_kv_heads_swa
            .unwrap_or(cfg.n_kv_heads)
            .max(cfg.gemma4_n_kv_heads_global.unwrap_or(cfg.n_kv_heads))
            as usize
            * max_hd;
        let inter_dim = cfg.intermediate_dim as usize;

        assert!(logits.len() >= vocab_size);

        let cpu_kv = kv
            .as_cpu_mut()
            .expect("Gemma4Forward CPU path requires ModelKv::Cpu");

        // --- Step 1: Token embedding ---
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

        // Scratch buffers (sized to max across layer types)
        let mut norm_buf = vec![0.0f32; dim];
        let mut q_buf = vec![0.0f32; max_q_dim];
        let mut k_buf = vec![0.0f32; max_kv_dim];
        let mut v_buf = vec![0.0f32; max_kv_dim];
        let mut attn_out = vec![0.0f32; max_q_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; dim];

        // KV cache stride (primary config = SWA = max stride)
        let kv_stride = cfg.n_kv_heads as usize * cfg.head_dim as usize;

        // Load rope_freqs for proportional RoPE on global layers
        let rope_freq_factors: Option<Vec<f32>> = if weights.has("rope_freqs.weight") {
            Some(weights.f32_slice("rope_freqs.weight")?.to_vec())
        } else {
            None
        };

        // --- Step 2: Transformer layers ---
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            let is_local = Self::use_sliding_window(layer, cfg);
            let (head_dim, n_kv_heads, q_dim, kv_dim) = Self::layer_dims(layer, cfg);
            let v_equals_k = Self::layer_v_equals_k(layer, cfg, weights);

            // 2a. Pre-attention RMSNorm
            apply_attention_norm_single(
                weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;

            // 2b. Q/K/V projections (per-layer variable dimensions)
            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;

            if v_equals_k {
                // Global layer: no V projection, V = K
                timed!(ops, matmul, {
                    ctx.backend.dequant_matmul(
                        wq_raw, wq_dtype, &norm_buf, &mut q_buf[..q_dim], q_dim, 1, dim,
                    );
                    ctx.backend.dequant_matmul(
                        wk_raw, wk_dtype, &norm_buf, &mut k_buf[..kv_dim], kv_dim, 1, dim,
                    );
                });
                // Copy K → V (before normalization)
                v_buf[..kv_dim].copy_from_slice(&k_buf[..kv_dim]);
            } else {
                // SWA layer: standard Q/K/V projections
                let (wv_raw, wv_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
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
            }

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

            // 2c'. V normalization (Gemma4: raw RMSNorm, no learned weight)
            for v_head in v_buf[..kv_dim].chunks_exact_mut(head_dim) {
                rms_norm::rms_norm_no_weight(v_head, cfg.rms_norm_eps);
            }

            // 2d. RoPE on Q and K (Gemma4 uses NeoX-style split-half pairs)
            let rope_base = if is_local {
                cfg.rope_freq_base_local.unwrap_or(cfg.rope_freq_base)
            } else {
                cfg.rope_freq_base
            };
            let rope_position = if is_local {
                position as f32
            } else {
                cfg.rope_scaling.scaled_position(position)
            };

            if !is_local && let Some(ref freq_factors) = rope_freq_factors {
                // Global layers: NeoX RoPE with freq_factors for proportional rotation
                timed!(
                    ops,
                    rope,
                    rope::apply_rope_neox_with_freq_factors(
                        &mut q_buf[..q_dim],
                        &mut k_buf[..kv_dim],
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        rope_position,
                        rope_base,
                        freq_factors,
                    )
                );
            } else {
                // SWA layers: standard NeoX RoPE with full rotation
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
                        rope_position,
                        rope_base,
                    )
                );
            }

            // 2e. Update KV cache (pad to full stride for global layers)
            if kv_dim < kv_stride {
                let mut k_padded = vec![0.0f32; kv_stride];
                let mut v_padded = vec![0.0f32; kv_stride];
                k_padded[..kv_dim].copy_from_slice(&k_buf[..kv_dim]);
                v_padded[..kv_dim].copy_from_slice(&v_buf[..kv_dim]);
                cpu_kv.append_and_advance(layer, &k_padded, &v_padded);
            } else {
                cpu_kv.append_and_advance(layer, &k_buf[..kv_dim], &v_buf[..kv_dim]);
            }

            // 2f. Multi-head attention
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

            let k_start = full_seq_len.saturating_sub(seq_len);
            let k_cache_raw = cpu_kv.k_slice_including_current(layer, full_seq_len);
            let v_cache_raw = cpu_kv.v_slice_including_current(layer, full_seq_len);

            // Repack KV data for global layers (cache stride != layer kv_dim)
            let (k_cache_slice, v_cache_slice);
            let mut k_compact;
            let mut v_compact;
            if kv_dim < kv_stride {
                k_compact = vec![0.0f32; seq_len * kv_dim];
                v_compact = vec![0.0f32; seq_len * kv_dim];
                for t in 0..seq_len {
                    let src = (k_start + t) * kv_stride;
                    k_compact[t * kv_dim..(t + 1) * kv_dim]
                        .copy_from_slice(&k_cache_raw[src..src + kv_dim]);
                    v_compact[t * kv_dim..(t + 1) * kv_dim]
                        .copy_from_slice(&v_cache_raw[src..src + kv_dim]);
                }
                k_cache_slice = &k_compact[..];
                v_cache_slice = &v_compact[..];
            } else {
                k_cache_slice = &k_cache_raw[k_start * kv_stride..full_seq_len * kv_stride];
                v_cache_slice = &v_cache_raw[k_start * kv_stride..full_seq_len * kv_stride];
            }

            // Gemma4: attention scale = 1.0 (QK norms handle scaling)
            timed!(
                ops,
                attention,
                attention::multi_head_attention_scaled(
                    &q_buf[..q_dim],
                    k_cache_slice,
                    v_cache_slice,
                    &mut attn_out[..q_dim],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    seq_len,
                    1.0,
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

            // 2h. Post-attention RMSNorm
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

            // 2p. Layer output scale (Gemma4-specific)
            if weights.has(&format!("{prefix}.layer_output_scale.weight")) {
                let scale =
                    weights.f32_slice(&format!("{prefix}.layer_output_scale.weight"))?[0];
                for h in hidden.iter_mut() {
                    *h *= scale;
                }
            }
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

        // --- Step 5: Logit softcapping ---
        if let Some(cap) = cfg.final_logit_softcapping {
            for l in logits[..vocab_size].iter_mut() {
                *l = (*l / cap).tanh() * cap;
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
        // Serial fallback: process tokens one at a time
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
        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, Some(ops))?;
        }
        Ok(())
    }

    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()> {
        if config.gate_activation != crate::model::config::GateActivation::GELU {
            tracing::warn!(
                "Gemma4Forward selected but gate_activation is {:?}, expected GELU",
                config.gate_activation
            );
        }
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "gemma4"
    }

    fn supports_pipelined_decode(&self, _ctx: &ForwardContext) -> bool {
        false
    }
}
