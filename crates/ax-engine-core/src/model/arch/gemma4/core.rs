impl Gemma4Forward {
    fn build_cached_model_keys_gemma4(
        metal_ops: &crate::backend::metal::MetalOps,
        weights: &WeightStore,
        cfg: &ModelConfig,
    ) -> anyhow::Result<()> {
        use crate::backend::metal::{CachedLayerKeys, CachedModelKeys};

        let dim = cfg.embedding_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let use_precomputed_f16 = metal_ops.metal_precompute_f16_enabled();
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);

        for layer in 0..cfg.n_layers as usize {
            let spec = Gemma4LayerSpec::new(layer, cfg, weights);
            let attn_norm_w = weights.f32_slice(&format!("{}.attn_norm.weight", spec.prefix))?;
            let attn_norm_key = metal_ops.ensure_f32_cached(attn_norm_w);

            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{}.attn_q.weight", spec.prefix))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{}.attn_k.weight", spec.prefix))?;
            let (wv_raw, wv_dtype) = if spec.v_equals_k {
                (wk_raw, wk_dtype)
            } else {
                weights.raw_with_dtype(&format!("{}.attn_v.weight", spec.prefix))?
            };
            let (wo_raw, wo_dtype) =
                weights.raw_with_dtype(&format!("{}.attn_output.weight", spec.prefix))?;
            let (wg_raw, wg_dtype) = weights.raw_with_dtype(&format!("{}.ffn_gate.weight", spec.prefix))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{}.ffn_up.weight", spec.prefix))?;
            let (wd_raw, wd_dtype) = weights.raw_with_dtype(&format!("{}.ffn_down.weight", spec.prefix))?;
            let ffn_norm_w = weights.f32_slice(&format!("{}.ffn_norm.weight", spec.prefix))?;

            let wq_key = metal_ops.ensure_quant_cached(wq_raw);
            let wk_key = metal_ops.ensure_quant_cached(wk_raw);
            let wv_key = metal_ops.ensure_quant_cached(wv_raw);
            let wo_key = metal_ops.ensure_quant_cached(wo_raw);
            let wg_key = metal_ops.ensure_quant_cached(wg_raw);
            let wu_key = metal_ops.ensure_quant_cached(wu_raw);
            let wd_key = metal_ops.ensure_quant_cached(wd_raw);
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);

            if use_precomputed_f16
                && metal_ops.metal_fused_qkv_enabled()
                && wq_dtype == wk_dtype
                && wq_dtype == wv_dtype
                && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0)
            {
                metal_ops.ensure_qkv_fused_quant_cached(wq_raw, wk_raw, wv_raw);
                if wq_dtype == GgmlType::Q4K {
                    metal_ops.ensure_precomputed_q4k_f16_fused_qkv(
                        wq_raw,
                        wk_raw,
                        wv_raw,
                        (spec.q_dim + 2 * spec.kv_dim) as u32,
                        dim as u32,
                    )?;
                }
                if wq_dtype == GgmlType::Q8_0 {
                    metal_ops.ensure_precomputed_q8_0_f16_fused_qkv(
                        wq_raw,
                        wk_raw,
                        wv_raw,
                        (spec.q_dim + 2 * spec.kv_dim) as u32,
                        dim as u32,
                    )?;
                }
            }

            if use_precomputed_f16 {
                crate::model::shared::ensure_precomputed_linear_f16(
                    metal_ops,
                    wq_raw,
                    wq_dtype,
                    spec.q_dim as u32,
                    dim as u32,
                )?;
                crate::model::shared::ensure_precomputed_linear_f16(
                    metal_ops,
                    wk_raw,
                    wk_dtype,
                    spec.kv_dim as u32,
                    dim as u32,
                )?;
                if !spec.v_equals_k {
                    crate::model::shared::ensure_precomputed_linear_f16(
                        metal_ops,
                        wv_raw,
                        wv_dtype,
                        spec.kv_dim as u32,
                        dim as u32,
                    )?;
                }
                crate::model::shared::ensure_precomputed_linear_f16(
                    metal_ops,
                    wo_raw,
                    wo_dtype,
                    dim as u32,
                    spec.q_dim as u32,
                )?;
                crate::model::shared::ensure_precomputed_linear_f16(
                    metal_ops,
                    wg_raw,
                    wg_dtype,
                    inter_dim as u32,
                    dim as u32,
                )?;
                crate::model::shared::ensure_precomputed_linear_f16(
                    metal_ops,
                    wu_raw,
                    wu_dtype,
                    inter_dim as u32,
                    dim as u32,
                )?;
                crate::model::shared::ensure_precomputed_linear_f16(
                    metal_ops,
                    wd_raw,
                    wd_dtype,
                    dim as u32,
                    inter_dim as u32,
                )?;
            }

            let (attn_q_norm_key, attn_k_norm_key) =
                crate::model::shared::cache_attention_qk_norm_keys(metal_ops, weights, &spec.prefix)?;
            let post_attn_norm = crate::model::shared::cache_optional_prefixed_f32_key(
                metal_ops,
                weights,
                &spec.prefix,
                "post_attention_norm.weight",
            )?;
            let layer_output_scale = crate::model::shared::cache_optional_prefixed_f32_key(
                metal_ops,
                weights,
                &spec.prefix,
                "layer_output_scale.weight",
            )?;

            layers.push(CachedLayerKeys {
                attn_norm: attn_norm_key,
                wq: wq_key,
                wq_dtype,
                wk: wk_key,
                wk_dtype,
                wv: wv_key,
                wv_dtype,
                wo: wo_key,
                wo_dtype,
                ffn_norm: ffn_norm_key,
                wg: wg_key,
                wg_dtype,
                wu: wu_key,
                wu_dtype,
                wd: wd_key,
                wd_dtype,
                attn_q_norm: attn_q_norm_key,
                attn_k_norm: attn_k_norm_key,
                post_attn_norm,
                post_ffn_norm: None,
                v_equals_k: spec.v_equals_k,
                layer_output_scale,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                wo_bias: None,
                gate_bias: None,
                up_bias: None,
                down_bias: None,
                moe_router: None,
                moe_router_dtype: None,
                moe_expert_gate: None,
                moe_expert_up: None,
                moe_expert_down: None,
                moe_expert_dtype: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_dtype: None,
            });
        }

        let (output_norm_key, lm_raw, lm_dtype, lm_head_key) =
            crate::model::shared::cache_output_head_keys(metal_ops, weights)?;
        if use_precomputed_f16 {
            crate::model::shared::ensure_precomputed_lm_head_f16(
                metal_ops,
                lm_raw,
                lm_dtype,
                cfg.vocab_size,
                cfg.embedding_dim,
            )?;
        }

        let rope_freqs = crate::model::shared::cache_optional_f32_key(
            metal_ops,
            weights,
            "rope_freqs.weight",
        )?;

        metal_ops.set_cached_model_keys(CachedModelKeys {
            layers,
            output_norm: output_norm_key,
            lm_head: lm_head_key,
            lm_head_dtype: lm_dtype,
            rope_freqs,
        });

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_gpu_unified(
        &self,
        ctx: &ForwardContext,
        metal_ops: &MetalOps,
        token_ids: &[u32],
        gpu_kv: &mut crate::kv::GpuKv,
        weights: &WeightStore,
        last_logits: Option<&mut [f32]>,
        logits_all: Option<&mut Vec<f32>>,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let n_tokens = token_ids.len();
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let eps = cfg.rms_norm_eps;
        let emit_all_logits = logits_all.is_some();

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
        anyhow::ensure!(
            !gpu_kv.is_q8(),
            "gemma4 gpu batch prefill does not support q8 kv cache yet"
        );

        let setup_t = OpTimer::start();
        {
            let batch_hidden_cpu = unsafe {
                std::slice::from_raw_parts_mut(
                    bs.hidden.contents().as_ptr() as *mut f32,
                    n_tokens * dim,
                )
            };
            for (i, &tid) in token_ids.iter().enumerate() {
                let slice = &mut batch_hidden_cpu[i * dim..(i + 1) * dim];
                weights.dequantize_row("token_embd.weight", tid as usize, slice)?;
            }
            if cfg.embed_scale {
                let scale = (dim as f32).sqrt();
                for h in batch_hidden_cpu.iter_mut() {
                    *h *= scale;
                }
            }
        }

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_gemma4(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let layer_specs: Vec<_> = (0..n_layers)
            .map(|layer| Gemma4LayerSpec::new(layer, cfg, weights))
            .collect();
        let all_logits_buf = if emit_all_logits {
            Some(ax_engine_metal::MetalBuffer::new(
                metal_ops.device.device(),
                n_tokens * vocab_size * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        };
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        let weight_cache = metal_ops.lock_weight_cache();
        let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();
        let has_q5k_weights = gpu_prefill_uses_q5k(weights);
        let q5k_small_n_auto_eligible = gpu_prefill_q5k_small_n_auto_eligible(weights);
        let prefill_plan = DecodeExecutionPlan::gemma3_prefill(
            metal_ops,
            gpu_kv,
            n_tokens as u32,
            has_q5k_weights,
            q5k_small_n_auto_eligible,
        );
        let base_seq_len = gpu_kv.seq_len();

        let exec_t = OpTimer::start();
        metal_ops.device.execute_sync_concurrent(|encoder| {
            let nt = n_tokens as u32;
            let mut sb = ax_engine_metal::SmartBarrier::new(encoder);
            let rope_freqs = cached.rope_freqs.and_then(|key| weight_cache.get(&key));

            let first_attn_norm = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
            sb.pre_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
            metal_ops.elementwise.encode_rms_norm_out_batch(
                encoder,
                &bs.hidden,
                first_attn_norm,
                &bs.norm_buf,
                dim as u32,
                nt,
                eps,
            );
            sb.post_dispatch(&[&bs.hidden], &[&bs.norm_buf]);

            for (layer, spec) in layer_specs.iter().enumerate() {
                let lw = &cached.layers[layer];
                let n_q_heads = (spec.q_dim / spec.head_dim) as u32;
                let kv_offset = (base_seq_len * gpu_kv.kv_stride()) as u32;
                let kv_stride = gpu_kv.kv_stride() as u32;
                let kv_k = gpu_kv.k_buffer(layer);
                let kv_v = gpu_kv.v_buffer(layer);
                let q_weight = lw.attn_q_norm.and_then(|key| weight_cache.get(&key));
                let k_weight = lw.attn_k_norm.and_then(|key| weight_cache.get(&key));
                let fused_qkv_m = spec.q_dim + 2 * spec.kv_dim;
                let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
                let qkv_layer_plan = DecodeExecutionPlan::gemma3_prefill_qkv_layer(
                    &prefill_plan,
                    lw.wq_dtype,
                    lw.wk_dtype,
                    lw.wv_dtype,
                );
                let fused_qkv_buf = if qkv_layer_plan.use_fused_projection {
                    fused_qkv_cache.get(&fused_qkv_key)
                } else {
                    None
                };

                let qkv_t = OpTimer::start();
                let q_input = &bs.norm_buf;
                let use_f16_input =
                    matches!(qkv_layer_plan.input, PrefillProjectionInputPlan::MatmulScratchF16);
                if use_f16_input {
                    sb.pre_dispatch(&[q_input], &[&bs.matmul_in_f16]);
                    metal_ops.elementwise.encode_cast_f32_to_f16(
                        encoder,
                        q_input,
                        &bs.matmul_in_f16,
                        nt * dim as u32,
                    );
                    sb.post_dispatch(&[q_input], &[&bs.matmul_in_f16]);
                }

                let wq_buf = weight_cache.get(&lw.wq).unwrap();
                let wk_buf = weight_cache.get(&lw.wk).unwrap();
                if let Some(fused_w) = fused_qkv_buf {
                    sb.pre_dispatch(&[q_input], &[&bs.qkv_buf]);
                    if use_f16_input {
                        encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            fused_w,
                            &bs.matmul_in_f16,
                            &bs.qkv_buf,
                            fused_qkv_m as u32,
                            nt,
                            dim as u32,
                            lw.wq_dtype,
                        );
                    } else {
                        encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            fused_w,
                            q_input,
                            &bs.qkv_buf,
                            &bs.matmul_in_f16,
                            fused_qkv_m as u32,
                            nt,
                            dim as u32,
                            lw.wq_dtype,
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.q5k_prefill_small_n,
                        );
                    }
                    sb.post_dispatch(&[q_input], &[&bs.qkv_buf]);
                    sb.pre_dispatch(&[&bs.qkv_buf], &[&bs.q_buf, &bs.k_buf, &bs.v_buf]);
                    metal_ops.elementwise.encode_qkv_split_batch(
                        encoder,
                        &bs.qkv_buf,
                        &bs.q_buf,
                        &bs.k_buf,
                        &bs.v_buf,
                        nt,
                        spec.q_dim as u32,
                        spec.kv_dim as u32,
                    );
                    sb.post_dispatch(&[&bs.qkv_buf], &[&bs.q_buf, &bs.k_buf, &bs.v_buf]);
                } else {
                    sb.pre_dispatch(&[q_input], &[&bs.q_buf]);
                    if use_f16_input {
                        encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wq_buf,
                            &bs.matmul_in_f16,
                            &bs.q_buf,
                            spec.q_dim as u32,
                            nt,
                            dim as u32,
                            lw.wq_dtype,
                        );
                    } else {
                        encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wq_buf,
                            q_input,
                            &bs.q_buf,
                            &bs.matmul_in_f16,
                            spec.q_dim as u32,
                            nt,
                            dim as u32,
                            lw.wq_dtype,
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.q5k_prefill_small_n,
                        );
                    }
                    sb.post_dispatch(&[q_input], &[&bs.q_buf]);

                    sb.pre_dispatch(&[q_input], &[&bs.k_buf]);
                    if use_f16_input {
                        encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wk_buf,
                            &bs.matmul_in_f16,
                            &bs.k_buf,
                            spec.kv_dim as u32,
                            nt,
                            dim as u32,
                            lw.wk_dtype,
                        );
                    } else {
                        encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wk_buf,
                            q_input,
                            &bs.k_buf,
                            &bs.matmul_in_f16,
                            spec.kv_dim as u32,
                            nt,
                            dim as u32,
                            lw.wk_dtype,
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.q5k_prefill_small_n,
                        );
                    }
                    sb.post_dispatch(&[q_input], &[&bs.k_buf]);

                    if spec.v_equals_k {
                        let copy_bytes = n_tokens * spec.kv_dim * std::mem::size_of::<f32>();
                        sb.pre_dispatch(&[&bs.k_buf], &[&bs.v_buf]);
                        metal_ops.elementwise.encode_buffer_copy(
                            encoder,
                            &bs.k_buf,
                            0,
                            &bs.v_buf,
                            0,
                            (copy_bytes / std::mem::size_of::<f32>()) as u32,
                        );
                        sb.post_dispatch(&[&bs.k_buf], &[&bs.v_buf]);
                    } else {
                        let wv_buf = weight_cache.get(&lw.wv).unwrap();
                        sb.pre_dispatch(&[q_input], &[&bs.v_buf]);
                        if use_f16_input {
                            encode_dequant_batch_f16in(
                                metal_ops,
                                encoder,
                                wv_buf,
                                &bs.matmul_in_f16,
                                &bs.v_buf,
                                spec.kv_dim as u32,
                                nt,
                                dim as u32,
                                lw.wv_dtype,
                            );
                        } else {
                            encode_dequant_batch(
                                &metal_ops.dequant,
                                &metal_ops.elementwise,
                                encoder,
                                wv_buf,
                                q_input,
                                &bs.v_buf,
                                &bs.matmul_in_f16,
                                spec.kv_dim as u32,
                                nt,
                                dim as u32,
                                lw.wv_dtype,
                                false,
                                prefill_plan.use_batch_simd,
                                prefill_plan.q5k_prefill_small_n,
                            );
                        }
                        sb.post_dispatch(&[q_input], &[&bs.v_buf]);
                    }
                }
                if let Some(ref mut ops_ref) = ops {
                    ops_ref.gpu_encode_layer_qkv += qkv_t.elapsed();
                }

                let qkv_post_t = OpTimer::start();
                if let Some(q_weight_buf) = q_weight {
                    sb.pre_dispatch(&[&bs.q_buf], &[&bs.q_buf]);
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder,
                        &bs.q_buf,
                        q_weight_buf,
                        nt,
                        n_q_heads,
                        spec.head_dim as u32,
                        eps,
                    );
                    sb.post_dispatch(&[&bs.q_buf], &[&bs.q_buf]);
                }
                if let Some(k_weight_buf) = k_weight {
                    sb.pre_dispatch(&[&bs.k_buf], &[&bs.k_buf]);
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder,
                        &bs.k_buf,
                        k_weight_buf,
                        nt,
                        spec.n_kv_heads as u32,
                        spec.head_dim as u32,
                        eps,
                    );
                    sb.post_dispatch(&[&bs.k_buf], &[&bs.k_buf]);
                }
                sb.pre_dispatch(&[&bs.v_buf], &[&bs.v_buf]);
                metal_ops.elementwise.encode_per_head_rms_norm_no_weight_batch(
                    encoder,
                    &bs.v_buf,
                    nt,
                    spec.n_kv_heads as u32,
                    spec.head_dim as u32,
                    eps,
                );
                sb.post_dispatch(&[&bs.v_buf], &[&bs.v_buf]);

                sb.pre_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);
                if !spec.is_sliding() {
                    if let Some(freq_factors) = rope_freqs {
                        metal_ops
                            .elementwise
                            .encode_rope_batch_neox_partial_with_freq_factors(
                                encoder,
                                &bs.q_buf,
                                &bs.k_buf,
                                freq_factors,
                                nt,
                                n_q_heads,
                                spec.n_kv_heads as u32,
                                spec.head_dim as u32,
                                spec.head_dim as u32,
                                base_seq_len as f32,
                                1.0,
                                spec.rope_base,
                            );
                    } else {
                        metal_ops.elementwise.encode_rope_batch(
                            encoder,
                            &bs.q_buf,
                            &bs.k_buf,
                            nt,
                            n_q_heads,
                            spec.n_kv_heads as u32,
                            spec.head_dim as u32,
                            spec.head_dim as u32,
                            base_seq_len as f32,
                            1.0,
                            spec.rope_base,
                        );
                    }
                } else {
                    metal_ops.elementwise.encode_rope_batch(
                        encoder,
                        &bs.q_buf,
                        &bs.k_buf,
                        nt,
                        n_q_heads,
                        spec.n_kv_heads as u32,
                        spec.head_dim as u32,
                        spec.head_dim as u32,
                        base_seq_len as f32,
                        1.0,
                        spec.rope_base,
                    );
                }
                sb.post_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);

                sb.pre_dispatch(&[&bs.k_buf, &bs.v_buf], &[kv_k, kv_v]);
                metal_ops.elementwise.encode_kv_append_batch_pair(
                    encoder,
                    &bs.k_buf,
                    &bs.v_buf,
                    kv_k,
                    kv_v,
                    prefill_plan.kv_f16,
                    kv_offset,
                    kv_stride,
                    spec.kv_dim as u32,
                    nt,
                );
                sb.post_dispatch(&[&bs.k_buf, &bs.v_buf], &[kv_k, kv_v]);
                if let Some(ref mut ops_ref) = ops {
                    let elapsed = qkv_post_t.elapsed();
                    ops_ref.gpu_encode_layer_rope += elapsed / 2;
                    ops_ref.gpu_encode_layer_kv_append += elapsed / 2;
                }

                let attn_t = OpTimer::start();
                sb.pre_dispatch(&[&bs.q_buf, kv_k, kv_v], &[&bs.attn_out]);
                if spec.kv_dim == gpu_kv.kv_stride() {
                    metal_ops
                        .attention
                        .encode_attention_prefill_cached_with_config(
                            encoder,
                            &bs.q_buf,
                            kv_k,
                            kv_v,
                            &bs.attn_out,
                            prefill_plan.kv_f16,
                            nt,
                            n_q_heads,
                            spec.n_kv_heads as u32,
                            spec.head_dim as u32,
                            base_seq_len as u32,
                            spec.local_window.unwrap_or(0) as u32,
                            prefill_plan.attention_dispatch,
                        );
                } else {
                    metal_ops
                        .attention
                        .encode_attention_prefill_cached_with_stride_and_config(
                            encoder,
                            &bs.q_buf,
                            kv_k,
                            kv_v,
                            &bs.attn_out,
                            prefill_plan.kv_f16,
                            nt,
                            n_q_heads,
                            spec.n_kv_heads as u32,
                            spec.head_dim as u32,
                            kv_stride,
                            base_seq_len as u32,
                            spec.local_window.unwrap_or(0) as u32,
                            prefill_plan.attention_dispatch,
                        );
                }
                sb.post_dispatch(&[&bs.q_buf, kv_k, kv_v], &[&bs.attn_out]);
                if let Some(ref mut ops_ref) = ops {
                    ops_ref.gpu_encode_layer_attention += attn_t.elapsed();
                }

                let out_proj_t = OpTimer::start();
                let wo_buf = weight_cache.get(&lw.wo).unwrap();
                sb.pre_dispatch(&[&bs.attn_out], &[&bs.proj_buf]);
                encode_dequant_batch(
                    &metal_ops.dequant,
                    &metal_ops.elementwise,
                    encoder,
                    wo_buf,
                    &bs.attn_out,
                    &bs.proj_buf,
                    &bs.matmul_in_f16,
                    dim as u32,
                    nt,
                    spec.q_dim as u32,
                    lw.wo_dtype,
                    prefill_plan.use_f16_batch_io,
                    prefill_plan.use_batch_simd,
                    prefill_plan.q5k_prefill_small_n,
                );
                sb.post_dispatch(&[&bs.attn_out], &[&bs.proj_buf]);
                if let Some(ref mut ops_ref) = ops {
                    ops_ref.gpu_encode_layer_out_proj += out_proj_t.elapsed();
                }

                let ffn_norm_buf = weight_cache.get(&lw.ffn_norm).unwrap();
                let post_attn_t = OpTimer::start();
                if let Some(post_attn_key) = lw.post_attn_norm {
                    let post_attn_buf = weight_cache.get(&post_attn_key).unwrap();
                    sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                    metal_ops
                        .elementwise
                        .encode_post_attn_norm_residual_add_rms_norm_out_batch(
                            encoder,
                            &bs.hidden,
                            &bs.proj_buf,
                            post_attn_buf,
                            ffn_norm_buf,
                            &bs.norm_buf,
                            dim as u32,
                            nt,
                            eps,
                        );
                    sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                } else {
                    sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                    metal_ops.elementwise.encode_residual_add_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        ffn_norm_buf,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        eps,
                    );
                    sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                }
                if let Some(ref mut ops_ref) = ops {
                    let elapsed = post_attn_t.elapsed();
                    ops_ref.gpu_encode_layer_residual += elapsed / 2;
                    ops_ref.gpu_encode_layer_norm += elapsed / 2;
                }

                let ffn_layer_plan = DecodeExecutionPlan::gemma3_prefill_ffn_layer(
                    &prefill_plan,
                    lw.wg_dtype,
                    lw.wu_dtype,
                );
                let wg_buf = weight_cache.get(&lw.wg).unwrap();
                let wu_buf = weight_cache.get(&lw.wu).unwrap();
                let gate_up_t = OpTimer::start();
                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.gate_buf, &bs.up_buf]);
                match ffn_layer_plan.input {
                    PrefillProjectionInputPlan::MatmulScratchF16 => {
                        metal_ops.elementwise.encode_cast_f32_to_f16(
                            encoder,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            nt * dim as u32,
                        );
                        if ffn_layer_plan.use_pair_kernel {
                            encode_dequant_batch_pair_f16in(
                                &metal_ops.dequant,
                                encoder,
                                wg_buf,
                                wu_buf,
                                &bs.matmul_in_f16,
                                &bs.gate_buf,
                                &bs.up_buf,
                                inter_dim as u32,
                                nt,
                                dim as u32,
                                lw.wg_dtype,
                            );
                        } else {
                            encode_dequant_batch_f16in(
                                metal_ops,
                                encoder,
                                wg_buf,
                                &bs.matmul_in_f16,
                                &bs.gate_buf,
                                inter_dim as u32,
                                nt,
                                dim as u32,
                                lw.wg_dtype,
                            );
                            encode_dequant_batch_f16in(
                                metal_ops,
                                encoder,
                                wu_buf,
                                &bs.matmul_in_f16,
                                &bs.up_buf,
                                inter_dim as u32,
                                nt,
                                dim as u32,
                                lw.wu_dtype,
                            );
                        }
                    }
                    PrefillProjectionInputPlan::NormBufF32 => {
                        encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wg_buf,
                            &bs.norm_buf,
                            &bs.gate_buf,
                            &bs.matmul_in_f16,
                            inter_dim as u32,
                            nt,
                            dim as u32,
                            lw.wg_dtype,
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.q5k_prefill_small_n,
                        );
                        encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wu_buf,
                            &bs.norm_buf,
                            &bs.up_buf,
                            &bs.matmul_in_f16,
                            inter_dim as u32,
                            nt,
                            dim as u32,
                            lw.wu_dtype,
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.q5k_prefill_small_n,
                        );
                    }
                }
                sb.post_dispatch(&[&bs.norm_buf], &[&bs.gate_buf, &bs.up_buf]);
                if let Some(ref mut ops_ref) = ops {
                    ops_ref.gpu_encode_layer_ffn += gate_up_t.elapsed();
                }

                let activation_t = OpTimer::start();
                sb.pre_dispatch(&[&bs.gate_buf, &bs.up_buf], &[&bs.gate_buf]);
                match ffn_layer_plan.activation {
                    PrefillFfnActivationPlan::GeluMulGateF32 => {
                        metal_ops.elementwise.encode_gelu_elementwise_mul_batch(
                            encoder,
                            &bs.gate_buf,
                            &bs.up_buf,
                            inter_dim as u32,
                            nt,
                        );
                    }
                    PrefillFfnActivationPlan::SiluMulGateF32
                    | PrefillFfnActivationPlan::SiluMulScratchF16 => unreachable!(),
                }
                sb.post_dispatch(&[&bs.gate_buf, &bs.up_buf], &[&bs.gate_buf]);
                if let Some(ref mut ops_ref) = ops {
                    ops_ref.gpu_encode_layer_ffn += activation_t.elapsed();
                }

                let down_t = OpTimer::start();
                let wd_buf = weight_cache.get(&lw.wd).unwrap();
                sb.pre_dispatch(&[&bs.gate_buf], &[&bs.proj_buf]);
                encode_dequant_batch(
                    &metal_ops.dequant,
                    &metal_ops.elementwise,
                    encoder,
                    wd_buf,
                    &bs.gate_buf,
                    &bs.proj_buf,
                    &bs.matmul_in_f16,
                    dim as u32,
                    nt,
                    inter_dim as u32,
                    lw.wd_dtype,
                    prefill_plan.use_f16_batch_io,
                    prefill_plan.use_batch_simd,
                    prefill_plan.q5k_prefill_small_n,
                );
                sb.post_dispatch(&[&bs.gate_buf], &[&bs.proj_buf]);
                if let Some(ref mut ops_ref) = ops {
                    ops_ref.gpu_encode_layer_ffn += down_t.elapsed();
                }

                let residual_t = OpTimer::start();
                if layer + 1 == n_layers {
                    sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                    );
                    sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    if let Some(scale_key) = lw.layer_output_scale {
                        let scale = unsafe { weight_cache.get(&scale_key).unwrap().as_slice::<f32>()[0] };
                        sb.pre_dispatch(&[&bs.hidden], &[&bs.hidden]);
                        metal_ops.elementwise.encode_gen_scale(
                            encoder,
                            &bs.hidden,
                            &bs.hidden,
                            scale,
                            (n_tokens * dim) as u32,
                        );
                        sb.post_dispatch(&[&bs.hidden], &[&bs.hidden]);
                    }
                } else if let Some(scale_key) = lw.layer_output_scale {
                    let scale = unsafe { weight_cache.get(&scale_key).unwrap().as_slice::<f32>()[0] };
                    let next_attn_norm = weight_cache.get(&cached.layers[layer + 1].attn_norm).unwrap();
                    sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                    );
                    sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    sb.pre_dispatch(&[&bs.hidden], &[&bs.hidden]);
                    metal_ops.elementwise.encode_gen_scale(
                        encoder,
                        &bs.hidden,
                        &bs.hidden,
                        scale,
                        (n_tokens * dim) as u32,
                    );
                    sb.post_dispatch(&[&bs.hidden], &[&bs.hidden]);
                    sb.pre_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        next_attn_norm,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        eps,
                    );
                    sb.post_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
                } else {
                    let next_attn_norm = weight_cache.get(&cached.layers[layer + 1].attn_norm).unwrap();
                    sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                    metal_ops.elementwise.encode_residual_add_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        next_attn_norm,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        eps,
                    );
                    sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                }
                if let Some(ref mut ops_ref) = ops {
                    let elapsed = residual_t.elapsed();
                    ops_ref.gpu_encode_layer_residual += elapsed / 2;
                    ops_ref.gpu_encode_layer_norm += elapsed / 2;
                }
            }

            let output_norm_buf = weight_cache.get(&cached.output_norm).unwrap();
            let lm_head_buf = weight_cache.get(&cached.lm_head).unwrap();
            match DecodeExecutionPlan::prefill_logits_plan(all_logits_buf.is_some()) {
                PrefillLogitsPlan::BatchAllLogits => {
                    let logits_buf = all_logits_buf.as_ref().unwrap();
                    sb.pre_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        output_norm_buf,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        eps,
                    );
                    sb.post_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
                    sb.pre_dispatch(&[&bs.norm_buf], &[logits_buf]);
                    encode_batch_logits(
                        metal_ops,
                        encoder,
                        lm_head_buf,
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        logits_buf,
                        vocab_size as u32,
                        nt,
                        dim as u32,
                        cached.lm_head_dtype,
                        prefill_plan.use_f16_batch_io,
                        prefill_plan.use_batch_simd,
                    );
                    sb.post_dispatch(&[&bs.norm_buf], &[logits_buf]);
                }
                PrefillLogitsPlan::LastTokenMatvec => {
                    let last_off = (n_tokens - 1) * dim * std::mem::size_of::<f32>();
                    sb.pre_dispatch(&[&bs.hidden], &[&s.hidden]);
                    metal_ops.elementwise.encode_buffer_copy(
                        encoder,
                        &bs.hidden,
                        last_off,
                        &s.hidden,
                        0,
                        dim as u32,
                    );
                    sb.post_dispatch(&[&bs.hidden], &[&s.hidden]);
                    sb.pre_dispatch(&[&s.hidden], &[&s.hidden]);
                    metal_ops.elementwise.encode_rms_norm(
                        encoder,
                        &s.hidden,
                        output_norm_buf,
                        dim as u32,
                        eps,
                    );
                    sb.post_dispatch(&[&s.hidden], &[&s.hidden]);
                    sb.pre_dispatch(&[&s.hidden], &[&s.logits_buf]);
                    encode_dequant_matvec(
                        metal_ops,
                        encoder,
                        lm_head_buf,
                        &s.hidden,
                        &s.logits_buf,
                        vocab_size as u32,
                        dim as u32,
                        cached.lm_head_dtype,
                    );
                    sb.post_dispatch(&[&s.hidden], &[&s.logits_buf]);
                }
            }

            sb.flush();
            Ok(())
        })?;
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_execute += exec_t.elapsed();
        }

        gpu_kv.finalize_batch(n_tokens);

        let rb_t = OpTimer::start();
        if let Some(logits_all) = logits_all {
            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    all_logits_buf
                        .as_ref()
                        .expect("batch logits buffer must exist for all-logits path")
                        .contents()
                        .as_ptr() as *const f32,
                    n_tokens * vocab_size,
                )
            };
            logits_all.resize(n_tokens * vocab_size, 0.0);
            logits_all.copy_from_slice(logits_gpu);
        } else if let Some(logits) = last_logits {
            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    s.logits_buf.contents().as_ptr() as *const f32,
                    vocab_size,
                )
            };
            logits[..vocab_size].copy_from_slice(logits_gpu);
            if let Some(cap) = cfg.final_logit_softcapping {
                for l in logits[..vocab_size].iter_mut() {
                    *l = (*l / cap).tanh() * cap;
                }
            }
        }
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_single_gpu_unified(
        &self,
        ctx: &ForwardContext,
        metal_ops: &crate::backend::metal::MetalOps,
        token_id: u32,
        position: usize,
        gpu_kv: &mut crate::kv::GpuKv,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        anyhow::ensure!(logits.len() >= vocab_size, "logits buffer too small");

        metal_ops.init_scratches(cfg);
        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();

        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + 1)?;

        let mut ops = ops;
        let setup_t = OpTimer::start();
        {
            let hidden_cpu = unsafe {
                std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim)
            };
            weights.dequantize_row("token_embd.weight", token_id as usize, hidden_cpu)?;
            if cfg.embed_scale {
                let embd_scale = (dim as f32).sqrt();
                for h in hidden_cpu.iter_mut() {
                    *h *= embd_scale;
                }
            }
        }
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_gemma4(metal_ops, weights, cfg)?;
        }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let cur_seq_len = gpu_kv.seq_len();
        let full_seq_len = cur_seq_len + 1;
        let kv_offset = (cur_seq_len * gpu_kv.kv_stride()) as u32;
        let kv_row_stride = gpu_kv.kv_stride() as u32;
        let max_head_dim = cfg
            .gemma4_head_dim_global
            .unwrap_or(cfg.head_dim)
            .max(cfg.gemma4_head_dim_swa.unwrap_or(cfg.head_dim));
        let base_plan = crate::model::execution_plan::DecodeExecutionPlan::gemma3_single_cb(
            metal_ops,
            gpu_kv,
            cfg.embedding_dim,
            max_head_dim,
            full_seq_len,
        );
        let layer_specs: Vec<_> = (0..cfg.n_layers as usize)
            .map(|layer| Gemma4LayerSpec::new(layer, cfg, weights))
            .collect();
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        let weight_cache = metal_ops.lock_weight_cache();
        let exec_t = OpTimer::start();
        let encode_body = |encoder: &ax_engine_metal::MetalEncoder| -> anyhow::Result<()> {
            let barrier =
                crate::model::shared::DecodeBarrierCtx::new(encoder, base_plan.barriers);
            let rope_freqs = cached.rope_freqs.and_then(|key| weight_cache.get(&key));

            for (layer, spec) in layer_specs.iter().enumerate() {
                let lw = &cached.layers[layer];
                let n_heads = (spec.q_dim / spec.head_dim) as u32;
                let layer_plan = crate::model::execution_plan::DecodeExecutionPlan::gemma3_single_cb(
                    metal_ops,
                    gpu_kv,
                    cfg.embedding_dim,
                    spec.head_dim as u32,
                    full_seq_len,
                );
                let attn_norm_w = weight_cache.get(&lw.attn_norm).unwrap();
                let wq_buf = weight_cache.get(&lw.wq).unwrap();
                let wk_buf = weight_cache.get(&lw.wk).unwrap();
                let wv_buf = weight_cache.get(&lw.wv).unwrap();
                let wo_buf = weight_cache.get(&lw.wo).unwrap();
                let ffn_norm_w = weight_cache.get(&lw.ffn_norm).unwrap();
                let wg_buf = weight_cache.get(&lw.wg).unwrap();
                let wu_buf = weight_cache.get(&lw.wu).unwrap();
                let wd_buf = weight_cache.get(&lw.wd).unwrap();
                let kv_k = gpu_kv.k_buffer(layer);
                let kv_v = gpu_kv.v_buffer(layer);

                barrier.pre_dispatch(&[&s.hidden], &[&s.norm_buf]);
                metal_ops.elementwise.encode_rms_norm_out(
                    encoder,
                    &s.hidden,
                    attn_norm_w,
                    &s.norm_buf,
                    dim as u32,
                    cfg.rms_norm_eps,
                );
                barrier.post_dispatch(&[&s.hidden], &[&s.norm_buf]);
                barrier.step(encoder);

                barrier.pre_dispatch(&[&s.norm_buf], &[&s.q_buf]);
                crate::model::shared::encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wq_buf,
                    &s.norm_buf,
                    &s.q_buf,
                    spec.q_dim as u32,
                    dim as u32,
                    lw.wq_dtype,
                    layer_plan.dequant_dispatch,
                );
                barrier.post_dispatch(&[&s.norm_buf], &[&s.q_buf]);
                barrier.pre_dispatch(&[&s.norm_buf], &[&s.k_buf]);
                crate::model::shared::encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wk_buf,
                    &s.norm_buf,
                    &s.k_buf,
                    spec.kv_dim as u32,
                    dim as u32,
                    lw.wk_dtype,
                    layer_plan.dequant_dispatch,
                );
                barrier.post_dispatch(&[&s.norm_buf], &[&s.k_buf]);
                barrier.pre_dispatch(&[&s.norm_buf], &[&s.v_buf]);
                crate::model::shared::encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wv_buf,
                    &s.norm_buf,
                    &s.v_buf,
                    spec.kv_dim as u32,
                    dim as u32,
                    lw.wv_dtype,
                    layer_plan.dequant_dispatch,
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
                        n_heads,
                        spec.head_dim as u32,
                        cfg.rms_norm_eps,
                    );
                    barrier.post_dispatch(&[&s.q_buf], &[&s.q_buf]);
                    barrier.pre_dispatch(&[&s.k_buf], &[&s.k_buf]);
                    metal_ops.elementwise.encode_per_head_rms_norm(
                        encoder,
                        &s.k_buf,
                        kn,
                        spec.n_kv_heads as u32,
                        spec.head_dim as u32,
                        cfg.rms_norm_eps,
                    );
                    barrier.post_dispatch(&[&s.k_buf], &[&s.k_buf]);
                }
                barrier.pre_dispatch(&[&s.v_buf], &[&s.v_buf]);
                metal_ops.elementwise.encode_per_head_rms_norm_no_weight(
                    encoder,
                    &s.v_buf,
                    spec.n_kv_heads as u32,
                    spec.head_dim as u32,
                    cfg.rms_norm_eps,
                );
                barrier.post_dispatch(&[&s.v_buf], &[&s.v_buf]);
                barrier.step(encoder);

                let rope_position = if spec.is_sliding() {
                    position as f32
                } else {
                    cfg.rope_scaling.scaled_position(position)
                };
                barrier.pre_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
                if !spec.is_sliding() {
                    if let Some(freq_factors) = rope_freqs {
                        metal_ops
                            .elementwise
                            .encode_rope_batch_neox_partial_with_freq_factors(
                                encoder,
                                &s.q_buf,
                                &s.k_buf,
                                freq_factors,
                                1,
                                n_heads,
                                spec.n_kv_heads as u32,
                                spec.head_dim as u32,
                                spec.head_dim as u32,
                                rope_position,
                                1.0,
                                spec.rope_base,
                            );
                    } else {
                        metal_ops.elementwise.encode_rope_batch_neox_partial(
                            encoder,
                            &s.q_buf,
                            &s.k_buf,
                            1,
                            n_heads,
                            spec.n_kv_heads as u32,
                            spec.head_dim as u32,
                            spec.head_dim as u32,
                            rope_position,
                            1.0,
                            spec.rope_base,
                        );
                    }
                } else {
                    metal_ops.elementwise.encode_rope_batch_neox_partial(
                        encoder,
                        &s.q_buf,
                        &s.k_buf,
                        1,
                        n_heads,
                        spec.n_kv_heads as u32,
                        spec.head_dim as u32,
                        spec.head_dim as u32,
                        rope_position,
                        1.0,
                        spec.rope_base,
                    );
                }
                barrier.post_dispatch(&[&s.q_buf, &s.k_buf], &[&s.q_buf, &s.k_buf]);
                barrier.step(encoder);

                barrier.pre_dispatch(&[&s.k_buf], &[kv_k]);
                metal_ops.elementwise.encode_kv_append(
                    encoder,
                    &s.k_buf,
                    kv_k,
                    layer_plan.kv_f16,
                    kv_offset,
                    spec.kv_dim as u32,
                );
                barrier.post_dispatch(&[&s.k_buf], &[kv_k]);
                barrier.pre_dispatch(&[&s.v_buf], &[kv_v]);
                metal_ops.elementwise.encode_kv_append(
                    encoder,
                    &s.v_buf,
                    kv_v,
                    layer_plan.kv_f16,
                    kv_offset,
                    spec.kv_dim as u32,
                );
                barrier.post_dispatch(&[&s.v_buf], &[kv_v]);
                barrier.step(encoder);

                let (attend_start, attend_len) = if let Some(window) = spec.local_window {
                    let len = full_seq_len.min(window);
                    (full_seq_len.saturating_sub(len) as u32, len as u32)
                } else {
                    (0u32, full_seq_len as u32)
                };
                barrier.pre_dispatch(&[&s.q_buf, kv_k, kv_v], &[&s.attn_out]);
                if spec.kv_dim == gpu_kv.kv_stride() {
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
                            layer_plan.kv_f16,
                            n_heads,
                            spec.n_kv_heads as u32,
                            spec.head_dim as u32,
                            attend_start,
                            attend_len,
                            layer_plan.attention_dispatch,
                        );
                } else {
                    metal_ops
                        .attention
                        .encode_attention_decode_with_stride_and_config(
                            encoder,
                            &s.q_buf,
                            kv_k,
                            kv_v,
                            &s.attn_out,
                            layer_plan.kv_f16,
                            n_heads,
                            spec.n_kv_heads as u32,
                            spec.head_dim as u32,
                            kv_row_stride,
                            attend_start,
                            attend_len,
                            layer_plan.attention_dispatch,
                        );
                }
                barrier.post_dispatch(&[&s.q_buf, kv_k, kv_v], &[&s.attn_out]);
                barrier.step(encoder);

                barrier.pre_dispatch(&[&s.attn_out], &[&s.proj_buf]);
                crate::model::shared::encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wo_buf,
                    &s.attn_out,
                    &s.proj_buf,
                    dim as u32,
                    spec.q_dim as u32,
                    lw.wo_dtype,
                    layer_plan.dequant_dispatch,
                );
                barrier.post_dispatch(&[&s.attn_out], &[&s.proj_buf]);
                barrier.step(encoder);

                if let Some(post_attn_key) = lw.post_attn_norm {
                    let post_attn_w = weight_cache.get(&post_attn_key).unwrap();
                    barrier.pre_dispatch(&[&s.proj_buf], &[&s.proj_buf]);
                    metal_ops.elementwise.encode_rms_norm(
                        encoder,
                        &s.proj_buf,
                        post_attn_w,
                        dim as u32,
                        cfg.rms_norm_eps,
                    );
                    barrier.post_dispatch(&[&s.proj_buf], &[&s.proj_buf]);
                    barrier.step(encoder);
                }

                barrier.pre_dispatch(&[&s.hidden, &s.proj_buf], &[&s.hidden]);
                metal_ops
                    .elementwise
                    .encode_elementwise_add(encoder, &s.hidden, &s.proj_buf, dim as u32);
                barrier.post_dispatch(&[&s.hidden, &s.proj_buf], &[&s.hidden]);
                barrier.step(encoder);

                barrier.pre_dispatch(&[&s.hidden], &[&s.norm_buf]);
                metal_ops.elementwise.encode_rms_norm_out(
                    encoder,
                    &s.hidden,
                    ffn_norm_w,
                    &s.norm_buf,
                    dim as u32,
                    cfg.rms_norm_eps,
                );
                barrier.post_dispatch(&[&s.hidden], &[&s.norm_buf]);
                barrier.step(encoder);

                barrier.pre_dispatch(&[&s.norm_buf], &[&s.gate_buf, &s.up_buf]);
                if !crate::model::shared::encode_dequant_matvec_pair_with_config(
                    metal_ops,
                    encoder,
                    wg_buf,
                    wu_buf,
                    &s.norm_buf,
                    &s.gate_buf,
                    &s.up_buf,
                    inter_dim as u32,
                    dim as u32,
                    lw.wg_dtype,
                    lw.wu_dtype,
                    layer_plan.dequant_dispatch,
                    layer_plan.use_pair_matvec,
                ) {
                    crate::model::shared::encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wg_buf,
                        &s.norm_buf,
                        &s.gate_buf,
                        inter_dim as u32,
                        dim as u32,
                        lw.wg_dtype,
                        layer_plan.dequant_dispatch,
                    );
                    crate::model::shared::encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wu_buf,
                        &s.norm_buf,
                        &s.up_buf,
                        inter_dim as u32,
                        dim as u32,
                        lw.wu_dtype,
                        layer_plan.dequant_dispatch,
                    );
                }
                barrier.post_dispatch(&[&s.norm_buf], &[&s.gate_buf, &s.up_buf]);
                barrier.step(encoder);

                barrier.pre_dispatch(&[&s.gate_buf, &s.up_buf], &[&s.down_buf]);
                if !crate::model::shared::encode_dequant_gelu_down_matvec_with_config(
                    metal_ops,
                    encoder,
                    wd_buf,
                    &s.gate_buf,
                    &s.up_buf,
                    &s.down_buf,
                    dim as u32,
                    inter_dim as u32,
                    lw.wd_dtype,
                    layer_plan.dequant_dispatch,
                ) {
                    metal_ops.elementwise.encode_gelu_elementwise_mul(
                        encoder,
                        &s.gate_buf,
                        &s.up_buf,
                        inter_dim as u32,
                    );
                    barrier.post_dispatch(&[&s.gate_buf, &s.up_buf], &[&s.gate_buf]);
                    barrier.step(encoder);
                    barrier.pre_dispatch(&[&s.gate_buf], &[&s.down_buf]);
                    crate::model::shared::encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wd_buf,
                        &s.gate_buf,
                        &s.down_buf,
                        dim as u32,
                        inter_dim as u32,
                        lw.wd_dtype,
                        layer_plan.dequant_dispatch,
                    );
                }
                barrier.post_dispatch(&[&s.gate_buf, &s.up_buf], &[&s.down_buf]);
                barrier.step(encoder);

                barrier.pre_dispatch(&[&s.hidden, &s.down_buf], &[&s.hidden]);
                metal_ops
                    .elementwise
                    .encode_elementwise_add(encoder, &s.hidden, &s.down_buf, dim as u32);
                barrier.post_dispatch(&[&s.hidden, &s.down_buf], &[&s.hidden]);
                barrier.step(encoder);

                if let Some(scale_key) = lw.layer_output_scale {
                    let scale_buf = weight_cache.get(&scale_key).unwrap();
                    let scale = unsafe { scale_buf.as_slice::<f32>()[0] };
                    barrier.pre_dispatch(&[&s.hidden], &[&s.hidden]);
                    metal_ops
                        .elementwise
                        .encode_gen_scale(encoder, &s.hidden, &s.hidden, scale, dim as u32);
                    barrier.post_dispatch(&[&s.hidden], &[&s.hidden]);
                    barrier.step(encoder);
                }
            }

            crate::model::shared::encode_gpu_output_head(
                encoder,
                metal_ops,
                s,
                &s.hidden,
                &base_plan,
                cached,
                &weight_cache,
                &barrier,
                dim as u32,
                vocab_size as u32,
                cfg.rms_norm_eps,
            );
            barrier.flush();
            Ok(())
        };

        if base_plan.encoder == crate::model::execution_plan::DecodeEncoderPlan::Concurrent {
            metal_ops.device.execute_sync_concurrent(encode_body)?;
        } else {
            metal_ops.device.execute_sync(encode_body)?;
        }
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_execute += exec_t.elapsed();
        }

        gpu_kv.finalize_token();

        let rb_t = OpTimer::start();
        let logits_gpu = unsafe {
            std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab_size)
        };
        logits[..vocab_size].copy_from_slice(logits_gpu);
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }

        if let Some(cap) = cfg.final_logit_softcapping {
            for l in logits[..vocab_size].iter_mut() {
                *l = (*l / cap).tanh() * cap;
            }
        }

        Ok(())
    }
}
