impl Gemma3Forward {
    /// Check if a layer uses sliding window (local) attention.
    ///
    /// Gemma3: most layers use sliding window. Every Nth layer (where N = pattern)
    /// uses global (full) attention. Specifically, the last layer in each group
    /// of N layers is global: layer % pattern == pattern - 1.
    ///
    /// For pattern=6: layers 0-4 = local, layer 5 = global, layers 6-10 = local, etc.
    pub(crate) fn use_sliding_window(layer: usize, config: &ModelConfig) -> bool {
        match (config.sliding_window_size, config.sliding_window_pattern) {
            (Some(_size), Some(pattern)) if pattern > 0 => {
                // Local (sliding window) for all layers except every Nth
                layer % (pattern as usize) != (pattern as usize - 1)
            }
            _ => false,
        }
    }

    pub(crate) fn gpu_prefill_chunk_len(config: &ModelConfig, n_tokens: usize) -> Option<usize> {
        match config.sliding_window_size {
            Some(window) if n_tokens > window as usize => Some(window as usize),
            _ => None,
        }
    }

    /// Build and store pre-computed weight cache keys for all layers (Gemma3 architecture).
    /// Called once on first forward pass; subsequent calls are skipped via `has_cached_model_keys()`.
    fn build_cached_model_keys_gemma3(
        metal_ops: &MetalOps,
        weights: &WeightStore,
        cfg: &ModelConfig,
    ) -> anyhow::Result<()> {
        use crate::backend::metal::{CachedLayerKeys, CachedModelKeys};

        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let dim = cfg.embedding_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let use_precomputed_f16 = metal_ops.metal_precompute_f16_enabled();
        let use_fused_decode_qkv = metal_ops.metal_decode_fused_qkv_enabled();

        let mut layers = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            let attn_norm_key = metal_ops.ensure_f32_cached(attn_norm_w);
            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
            let wq_key = metal_ops.ensure_quant_cached(wq_raw);
            let wk_key = metal_ops.ensure_quant_cached(wk_raw);
            let wv_key = metal_ops.ensure_quant_cached(wv_raw);
            if use_fused_decode_qkv
                && wq_dtype == wk_dtype
                && wq_dtype == wv_dtype
                && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K)
            {
                metal_ops.ensure_qkv_fused_quant_cached(wq_raw, wk_raw, wv_raw);
            }
            if use_precomputed_f16
                && metal_ops.metal_fused_qkv_enabled()
                && wq_dtype == wk_dtype
                && wq_dtype == wv_dtype
                && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K)
            {
                metal_ops.ensure_qkv_fused_quant_cached(wq_raw, wk_raw, wv_raw);
                if wq_dtype == GgmlType::Q4K {
                    metal_ops.ensure_precomputed_q4k_f16_fused_qkv(
                        wq_raw,
                        wk_raw,
                        wv_raw,
                        (q_dim + 2 * kv_dim) as u32,
                        dim as u32,
                    )?;
                }
            }
            if use_precomputed_f16 && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K) {
                ensure_precomputed_linear_f16_many(
                    metal_ops,
                    &[
                        (wq_raw, wq_dtype, q_dim as u32, dim as u32),
                        (wk_raw, wk_dtype, kv_dim as u32, dim as u32),
                        (wv_raw, wv_dtype, kv_dim as u32, dim as u32),
                    ],
                )?;
            }

            // QK norm weights (Gemma3-specific)
            let (attn_q_norm_key, attn_k_norm_key) =
                cache_attention_qk_norm_keys(metal_ops, weights, &prefix)?;

            let (wo_raw, wo_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            let (wg_raw, wg_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            let (wd_raw, wd_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
            let wo_key = metal_ops.ensure_quant_cached(wo_raw);
            let wg_key = metal_ops.ensure_quant_cached(wg_raw);
            let wu_key = metal_ops.ensure_quant_cached(wu_raw);
            let wd_key = metal_ops.ensure_quant_cached(wd_raw);
            if use_precomputed_f16 {
                ensure_precomputed_linear_f16(
                    metal_ops,
                    wo_raw,
                    wo_dtype,
                    dim as u32,
                    q_dim as u32,
                )?;
                ensure_precomputed_linear_f16(
                    metal_ops,
                    wg_raw,
                    wg_dtype,
                    inter_dim as u32,
                    dim as u32,
                )?;
                ensure_precomputed_linear_f16(
                    metal_ops,
                    wu_raw,
                    wu_dtype,
                    inter_dim as u32,
                    dim as u32,
                )?;
                ensure_precomputed_linear_f16(
                    metal_ops,
                    wd_raw,
                    wd_dtype,
                    dim as u32,
                    inter_dim as u32,
                )?;
            }
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);

            // Post-norms (Gemma3-specific)
            let post_attn_norm_key = cache_optional_prefixed_f32_key(
                metal_ops,
                weights,
                &prefix,
                "post_attention_norm.weight",
            )?;
            let post_ffn_norm_key = cache_optional_prefixed_f32_key(
                metal_ops,
                weights,
                &prefix,
                "post_ffw_norm.weight",
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
                post_attn_norm: post_attn_norm_key,
                post_ffn_norm: post_ffn_norm_key,
                v_equals_k: false,
                layer_output_scale: None,
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
            cache_output_head_keys(metal_ops, weights)?;
        if use_precomputed_f16 {
            ensure_precomputed_lm_head_f16(
                metal_ops,
                lm_raw,
                lm_dtype,
                cfg.vocab_size,
                cfg.embedding_dim,
            )?;
        }
        metal_ops.set_cached_model_keys(CachedModelKeys {
            layers,
            output_norm: output_norm_key,
            lm_head: lm_head_key,
            lm_head_dtype: lm_dtype,
            rope_freqs: None,
        });
        Ok(())
    }

    /// Unified GPU forward pass: all layers in a single command buffer.
    ///
    /// Eliminates the CPU attention sync point by using GPU decode attention
    /// and GPU KV cache. Reduces from 69 command buffers to 2 per token.
    ///
    /// v2: receives `gpu_kv: &mut GpuKv` directly. Ends with `gpu_kv.finalize_token()`.
    #[allow(clippy::too_many_arguments)]
    fn forward_single_gpu_unified(
        &self,
        ctx: &ForwardContext,
        metal_ops: &MetalOps,
        token_id: u32,
        position: usize,
        gpu_kv: &mut crate::kv::GpuKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let kv_dim = n_kv_heads * head_dim;

        assert!(logits.len() >= vocab_size);

        // Initialize GPU scratch buffers on first call
        metal_ops.init_scratches(cfg);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();

        // Ensure GPU KV cache has capacity for this token
        let next_seq = gpu_kv.seq_len() + 1;
        gpu_kv.ensure_capacity(&metal_ops.device, next_seq)?;

        let setup_t = OpTimer::start();
        // Token embedding (CPU dequant → GPU scratch)
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

        // Pre-cache ALL weights and build cached keys (first call only)
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_gemma3(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();

        // Current sequence position (before appending this token)
        let cur_seq_len = gpu_kv.seq_len();
        let kv_offset = (cur_seq_len * kv_dim) as u32;
        let exec_plan = DecodeExecutionPlan::gemma3_single_cb(
            metal_ops,
            gpu_kv,
            cfg.embedding_dim,
            cfg.head_dim,
            cur_seq_len + 1,
        );
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        // ── Single command buffer: all layers + final norm + LM head ──
        {
            let weight_cache = metal_ops.lock_weight_cache();
            let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();
            let exec_t = OpTimer::start();
            let encode_body =
                |encoder: &ax_engine_metal::MetalEncoder| -> anyhow::Result<()> {
                    let barrier = crate::model::shared::DecodeBarrierCtx::new(
                        encoder,
                        exec_plan.barriers,
                    );
                    encode_gemma3_gpu_layers_only(
                        encoder,
                        metal_ops,
                        s,
                        &s.hidden,
                        cfg,
                        position,
                        kv_offset,
                        cur_seq_len + 1,
                        &exec_plan,
                        gpu_kv,
                        cached,
                        &weight_cache,
                        &fused_qkv_cache,
                    )?;
                    encode_gemma3_gpu_output_head(
                        encoder,
                        metal_ops,
                        s,
                        &s.hidden,
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
                metal_ops.device.execute_sync_concurrent(encode_body)?;
            } else {
                metal_ops.device.execute_sync(encode_body)?;
            }
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_execute += exec_t.elapsed();
            }
        }

        // v2: advance GPU KV only — no CPU mirror to sync
        gpu_kv.finalize_token();

        // Copy logits from GPU buffer to CPU
        let rb_t = OpTimer::start();
        let logits_gpu = unsafe {
            std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab_size)
        };
        logits[..vocab_size].copy_from_slice(logits_gpu);
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }

        // Logit scaling
        if let Some(scale) = cfg.logit_scale {
            for l in logits[..vocab_size].iter_mut() {
                *l *= scale;
            }
        }

        Ok(())
    }

    /// Batched GPU prefill for Gemma3.
    ///
    /// Uses per-token loops for QKV/RoPE/QK-norm and output-proj/FFN phases
    /// (required for correctness with Gemma3-specific per-head norms and
    /// post-attention/post-FFN norms), with batched attention in between.
    /// Weight cache keys are pre-computed to avoid format!/HashMap overhead.
    ///
    /// v2: receives `gpu_kv: &mut GpuKv` directly. Ends with `gpu_kv.finalize_batch(n_tokens)`.
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
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let eps = cfg.rms_norm_eps;
        let emit_all_logits = logits_all.is_some();

        if let Some(logits) = last_logits.as_ref() {
            assert!(logits.len() >= vocab_size);
        }

        // Gemma3 sliding window guard: batch prefill only works when n_tokens <= window_size
        if let Some(window) = cfg.sliding_window_size {
            anyhow::ensure!(
                n_tokens <= window as usize,
                "Gemma3 batch prefill: n_tokens ({}) > sliding_window_size ({}); use serial",
                n_tokens,
                window
            );
        }

        let setup_t = OpTimer::start();

        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, n_tokens);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let batch_guard = metal_ops.batch_scratches();
        let bs = batch_guard.as_ref().unwrap();

        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + n_tokens)?;

        // Embed all N tokens (with scaling)
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

        // Pre-cache weights and build cached keys (first call only)
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_gemma3(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();

        let base_seq_len = gpu_kv.seq_len();
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

        // Single command buffer: all layers + final norm + LM head
        {
            let weight_cache = metal_ops.lock_weight_cache();
            let has_q5k_weights = gpu_prefill_uses_q5k(weights);
            let q5k_small_n_auto_eligible = gpu_prefill_q5k_small_n_auto_eligible(weights);
            let prefill_plan: GpuBatchPrefillExecutionPlan = DecodeExecutionPlan::gemma3_prefill(
                metal_ops,
                gpu_kv,
                n_tokens as u32,
                has_q5k_weights,
                q5k_small_n_auto_eligible,
            );
            let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

            let exec_t = OpTimer::start();
            metal_ops.device.execute_sync_concurrent(|encoder| {
                let nt = n_tokens as u32;
                let mut sb = ax_engine_metal::SmartBarrier::new(encoder);

                // First layer's Phase 1a: standalone RMSNorm before loop.
                {
                    let t = OpTimer::start();
                    let norm_w_buf = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        norm_w_buf,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        eps,
                    );
                    sb.post_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_norm += t.elapsed();
                    }
                }

                for layer in 0..n_layers {
                    let lw = &cached.layers[layer];
                    let layer_plan = DecodeExecutionPlan::gemma3_prefill_layer(
                        cfg,
                        layer,
                        base_seq_len,
                        prefill_plan.use_f16_batch_io,
                    );

                    let wq_buf = weight_cache.get(&lw.wq).unwrap();
                    let wk_buf = weight_cache.get(&lw.wk).unwrap();
                    let wv_buf = weight_cache.get(&lw.wv).unwrap();

                    let fused_qkv_m = q_dim + 2 * kv_dim;
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

                    // Phase 1a: norm_buf already populated (first layer before loop,
                    // subsequent layers fused with previous Phase 3h).

                    // ── Phase 1b: Batched QKV matmul ──
                    let qkv_t = OpTimer::start();
                    if let Some(fused_w) = fused_qkv_buf {
                        let qkv_input = &bs.norm_buf;
                        sb.pre_dispatch(&[qkv_input], &[&bs.qkv_buf]);
                        match qkv_layer_plan.input {
                            PrefillProjectionInputPlan::MatmulScratchF16 => {
                                metal_ops.elementwise.encode_cast_f32_to_f16(
                                    encoder,
                                    &bs.norm_buf,
                                    &bs.matmul_in_f16,
                                    nt * dim as u32,
                                );
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
                            }
                            PrefillProjectionInputPlan::NormBufF32 => {
                                encode_dequant_batch(
                                    &metal_ops.dequant,
                                    &metal_ops.elementwise,
                                    encoder,
                                    fused_w,
                                    &bs.norm_buf,
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
                        }
                        sb.post_dispatch(&[qkv_input], &[&bs.qkv_buf]);
                        sb.pre_dispatch(&[&bs.qkv_buf], &[&bs.q_buf, &bs.k_buf, &bs.v_buf]);
                        metal_ops.elementwise.encode_qkv_split_batch(
                            encoder,
                            &bs.qkv_buf,
                            &bs.q_buf,
                            &bs.k_buf,
                            &bs.v_buf,
                            nt,
                            q_dim as u32,
                            kv_dim as u32,
                        );
                        sb.post_dispatch(&[&bs.qkv_buf], &[&bs.q_buf, &bs.k_buf, &bs.v_buf]);
                    } else {
                        match qkv_layer_plan.input {
                            PrefillProjectionInputPlan::MatmulScratchF16 => {
                                // f16 path: all three share matmul_in_f16, can't overlap.
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.matmul_in_f16]);
                                metal_ops.elementwise.encode_cast_f32_to_f16(
                                    encoder,
                                    &bs.norm_buf,
                                    &bs.matmul_in_f16,
                                    nt * dim as u32,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.matmul_in_f16]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.q_buf]);
                                encode_dequant_batch_f16in(
                                    metal_ops,
                                    encoder,
                                    wq_buf,
                                    &bs.matmul_in_f16,
                                    &bs.q_buf,
                                    q_dim as u32,
                                    nt,
                                    dim as u32,
                                    lw.wq_dtype,
                                );
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.q_buf]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.k_buf]);
                                encode_dequant_batch_f16in(
                                    metal_ops,
                                    encoder,
                                    wk_buf,
                                    &bs.matmul_in_f16,
                                    &bs.k_buf,
                                    kv_dim as u32,
                                    nt,
                                    dim as u32,
                                    lw.wk_dtype,
                                );
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.k_buf]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.v_buf]);
                                encode_dequant_batch_f16in(
                                    metal_ops,
                                    encoder,
                                    wv_buf,
                                    &bs.matmul_in_f16,
                                    &bs.v_buf,
                                    kv_dim as u32,
                                    nt,
                                    dim as u32,
                                    lw.wv_dtype,
                                );
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.v_buf]);
                            }
                            PrefillProjectionInputPlan::NormBufF32 => {
                                // f32 path: Q/K/V all read norm_buf, write
                                // different buffers -> SmartBarrier skips
                                // barriers between them (GPU can overlap).
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.q_buf]);
                                encode_dequant_batch(
                                    &metal_ops.dequant,
                                    &metal_ops.elementwise,
                                    encoder,
                                    wq_buf,
                                    &bs.norm_buf,
                                    &bs.q_buf,
                                    &bs.matmul_in_f16,
                                    q_dim as u32,
                                    nt,
                                    dim as u32,
                                    lw.wq_dtype,
                                    false,
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.q_buf]);
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.k_buf]);
                                encode_dequant_batch(
                                    &metal_ops.dequant,
                                    &metal_ops.elementwise,
                                    encoder,
                                    wk_buf,
                                    &bs.norm_buf,
                                    &bs.k_buf,
                                    &bs.matmul_in_f16,
                                    kv_dim as u32,
                                    nt,
                                    dim as u32,
                                    lw.wk_dtype,
                                    false,
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.k_buf]);
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.v_buf]);
                                encode_dequant_batch(
                                    &metal_ops.dequant,
                                    &metal_ops.elementwise,
                                    encoder,
                                    wv_buf,
                                    &bs.norm_buf,
                                    &bs.v_buf,
                                    &bs.matmul_in_f16,
                                    kv_dim as u32,
                                    nt,
                                    dim as u32,
                                    lw.wv_dtype,
                                    false,
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.v_buf]);
                            }
                        }
                    }
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_qkv += qkv_t.elapsed();
                    }

                    let cache_offset = (base_seq_len * kv_dim) as u32;
                    let kv_k = gpu_kv.k_buffer(layer);
                    let kv_v = gpu_kv.v_buffer(layer);
                    let fused_qkv_post = fused_qkv_buf.is_some()
                        && lw.attn_q_norm.is_some()
                        && lw.attn_k_norm.is_some();

                    // ── Phase 1c+1d: Fused split + QK norm + RoPE + KV append when eligible ──
                    let rope_kv_t = OpTimer::start();
                    if fused_qkv_post {
                        let q_nw = weight_cache.get(&lw.attn_q_norm.unwrap()).unwrap();
                        let k_nw = weight_cache.get(&lw.attn_k_norm.unwrap()).unwrap();
                        sb.pre_dispatch(
                            &[&bs.qkv_buf],
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v],
                        );
                        metal_ops
                            .elementwise
                            .encode_qkv_split_qk_norm_rope_append_kv_batch(
                                encoder,
                                &bs.qkv_buf,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                q_nw,
                                k_nw,
                                kv_k,
                                kv_v,
                                prefill_plan.kv_f16,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                eps,
                                layer_plan.rope_start,
                                layer_plan.rope_step,
                                layer_plan.rope_base,
                                cache_offset,
                                kv_dim as u32,
                            );
                        sb.post_dispatch(
                            &[&bs.qkv_buf],
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v],
                        );
                    } else {
                        sb.pre_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);
                        if let (Some(q_norm_key), Some(k_norm_key)) =
                            (lw.attn_q_norm, lw.attn_k_norm)
                        {
                            let q_nw = weight_cache.get(&q_norm_key).unwrap();
                            let k_nw = weight_cache.get(&k_norm_key).unwrap();
                            metal_ops.elementwise.encode_qk_norm_rope_batch(
                                encoder,
                                &bs.q_buf,
                                &bs.k_buf,
                                q_nw,
                                k_nw,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                eps,
                                layer_plan.rope_start,
                                layer_plan.rope_step,
                                layer_plan.rope_base,
                            );
                        } else {
                            metal_ops.elementwise.encode_rope_batch(
                                encoder,
                                &bs.q_buf,
                                &bs.k_buf,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                head_dim as u32,
                                layer_plan.rope_start,
                                layer_plan.rope_step,
                                layer_plan.rope_base,
                            );
                        }
                        sb.post_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);

                        // K and V appends write to different KV buffers —
                        // SmartBarrier skips the barrier between them.
                        sb.pre_dispatch(&[&bs.k_buf], &[kv_k]);
                        if prefill_plan.kv_q8 {
                            metal_ops.elementwise.encode_kv_append_batch_q8(
                                encoder,
                                &bs.k_buf,
                                kv_k,
                                cache_offset,
                                kv_dim as u32 / 32,
                                kv_dim as u32,
                                nt,
                            );
                        } else {
                            metal_ops.elementwise.encode_kv_append_batch(
                                encoder,
                                &bs.k_buf,
                                kv_k,
                                prefill_plan.kv_f16,
                                cache_offset,
                                kv_dim as u32,
                                kv_dim as u32,
                                nt,
                            );
                        }
                        sb.post_dispatch(&[&bs.k_buf], &[kv_k]);
                        sb.pre_dispatch(&[&bs.v_buf], &[kv_v]);
                        if prefill_plan.kv_q8 {
                            metal_ops.elementwise.encode_kv_append_batch_q8(
                                encoder,
                                &bs.v_buf,
                                kv_v,
                                cache_offset,
                                kv_dim as u32 / 32,
                                kv_dim as u32,
                                nt,
                            );
                        } else {
                            metal_ops.elementwise.encode_kv_append_batch(
                                encoder,
                                &bs.v_buf,
                                kv_v,
                                prefill_plan.kv_f16,
                                cache_offset,
                                kv_dim as u32,
                                kv_dim as u32,
                                nt,
                            );
                        }
                        sb.post_dispatch(&[&bs.v_buf], &[kv_v]);
                    }
                    if let Some(ref mut ops_ref) = ops {
                        let elapsed = rope_kv_t.elapsed();
                        ops_ref.gpu_encode_layer_rope += elapsed / 2;
                        ops_ref.gpu_encode_layer_kv_append += elapsed / 2;
                    }

                    // ── Phase 2: Batched attention ──
                    let attn_t = OpTimer::start();
                    if layer_plan.attention == PrefillAttentionPlan::BatchLocal {
                        sb.pre_dispatch(&[&bs.q_buf, &bs.k_buf, &bs.v_buf], &[&bs.attn_out]);
                        metal_ops.attention.encode_attention_prefill_with_config(
                            encoder,
                            &bs.q_buf,
                            &bs.k_buf,
                            &bs.v_buf,
                            &bs.attn_out,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            prefill_plan.attention_dispatch,
                        );
                        sb.post_dispatch(&[&bs.q_buf, &bs.k_buf, &bs.v_buf], &[&bs.attn_out]);
                    } else {
                        let attn_kv_k = gpu_kv.k_buffer(layer);
                        let attn_kv_v = gpu_kv.v_buffer(layer);
                        sb.pre_dispatch(&[&bs.q_buf, attn_kv_k, attn_kv_v], &[&bs.attn_out]);
                        if prefill_plan.kv_q8 {
                            metal_ops
                                .attention
                                .encode_attention_prefill_cached_q8kv(
                                    encoder,
                                    &bs.q_buf,
                                    attn_kv_k,
                                    attn_kv_v,
                                    &bs.attn_out,
                                    nt,
                                    n_heads as u32,
                                    n_kv_heads as u32,
                                    head_dim as u32,
                                    base_seq_len as u32,
                                    layer_plan.sliding_window,
                                );
                        } else {
                            metal_ops
                                .attention
                                .encode_attention_prefill_cached_with_config(
                                    encoder,
                                    &bs.q_buf,
                                    attn_kv_k,
                                    attn_kv_v,
                                    &bs.attn_out,
                                    prefill_plan.kv_f16,
                                    nt,
                                    n_heads as u32,
                                    n_kv_heads as u32,
                                    head_dim as u32,
                                    base_seq_len as u32,
                                    layer_plan.sliding_window,
                                    prefill_plan.attention_dispatch,
                                );
                        }
                        sb.post_dispatch(&[&bs.q_buf, attn_kv_k, attn_kv_v], &[&bs.attn_out]);
                    }
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_attention += attn_t.elapsed();
                    }

                    // ── Phase 3a: Batched output projection ──
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
                        q_dim as u32,
                        lw.wo_dtype,
                        prefill_plan.use_f16_batch_io,
                        prefill_plan.use_batch_simd,
                        prefill_plan.q5k_prefill_small_n,
                    );
                    sb.post_dispatch(&[&bs.attn_out], &[&bs.proj_buf]);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_out_proj += out_proj_t.elapsed();
                    }

                    let ffn_nw_buf = weight_cache.get(&lw.ffn_norm).unwrap();
                    let post_attn_and_residual_t = OpTimer::start();
                    if let Some(post_attn_key) = lw.post_attn_norm {
                        let post_attn_nw_buf = weight_cache.get(&post_attn_key).unwrap();
                        sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                        metal_ops
                            .elementwise
                            .encode_post_attn_norm_residual_add_rms_norm_out_batch(
                                encoder,
                                &bs.hidden,
                                &bs.proj_buf,
                                post_attn_nw_buf,
                                ffn_nw_buf,
                                &bs.norm_buf,
                                dim as u32,
                                nt,
                                eps,
                            );
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                    } else {
                        sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                        metal_ops
                            .elementwise
                            .encode_residual_add_rms_norm_out_batch(
                                encoder,
                                &bs.hidden,
                                &bs.proj_buf,
                                ffn_nw_buf,
                                &bs.norm_buf,
                                dim as u32,
                                nt,
                                eps,
                            );
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, &bs.norm_buf]);
                    }
                    if let Some(ref mut ops_ref) = ops {
                        let elapsed = post_attn_and_residual_t.elapsed();
                        ops_ref.gpu_encode_layer_residual += elapsed / 2;
                        ops_ref.gpu_encode_layer_norm += elapsed / 2;
                    }

                    // ── Phase 3d: Batched gate + up ──
                    let gate_up_t = OpTimer::start();
                    let wg_buf = weight_cache.get(&lw.wg).unwrap();
                    let wu_buf = weight_cache.get(&lw.wu).unwrap();
                    let ffn_layer_plan = DecodeExecutionPlan::gemma3_prefill_ffn_layer(
                        &prefill_plan,
                        lw.wg_dtype,
                        lw.wu_dtype,
                    );

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

                    // ── Phase 3e: Batched GELU activation ──
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

                    // ── Phase 3f: Batched down projection ──
                    let down_proj_t = OpTimer::start();
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
                        ops_ref.gpu_encode_layer_ffn += down_proj_t.elapsed();
                    }

                    let residual_plan =
                        DecodeExecutionPlan::gemma3_prefill_residual_handoff(layer + 1 == n_layers);
                    let fused_post_ffn_handoff = lw.post_ffn_norm.is_some()
                        && matches!(
                            residual_plan,
                            PrefillResidualHandoffPlan::ResidualAddRmsNormF32
                        );

                    // ── Phase 3g: Post-FFN RMSNorm (Gemma3-specific) ──
                    let post_ffn_norm_t = OpTimer::start();
                    if !fused_post_ffn_handoff {
                        if let Some(post_ffn_key) = lw.post_ffn_norm {
                            let nw = weight_cache.get(&post_ffn_key).unwrap();
                            sb.pre_dispatch(&[&bs.proj_buf], &[&bs.proj_buf]);
                            metal_ops.elementwise.encode_rms_norm_batch(
                                encoder,
                                &bs.proj_buf,
                                nw,
                                dim as u32,
                                nt,
                                eps,
                            );
                            sb.post_dispatch(&[&bs.proj_buf], &[&bs.proj_buf]);
                        }
                        if let Some(ref mut ops_ref) = ops {
                            ops_ref.gpu_encode_layer_norm += post_ffn_norm_t.elapsed();
                        }
                    }

                    // ── Phase 3h: Batched residual (+ next layer's norm if not last) ──
                    let residual_handoff_t = OpTimer::start();
                    let residual_norm_out = match residual_plan {
                        PrefillResidualHandoffPlan::ResidualOnly => None,
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF32 => Some(&bs.norm_buf),
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => unreachable!(),
                    };
                    if let Some(nout) = residual_norm_out {
                        sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, nout]);
                    } else {
                        sb.pre_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    }
                    match residual_plan {
                        PrefillResidualHandoffPlan::ResidualOnly => {
                            metal_ops.elementwise.encode_elementwise_add_batch(
                                encoder,
                                &bs.hidden,
                                &bs.proj_buf,
                                dim as u32,
                                nt,
                            );
                        }
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF32 => {
                            let next_norm_w = weight_cache
                                .get(&cached.layers[layer + 1].attn_norm)
                                .unwrap();
                            if let Some(post_ffn_key) = lw.post_ffn_norm {
                                let post_ffn_nw = weight_cache.get(&post_ffn_key).unwrap();
                                metal_ops
                                    .elementwise
                                    .encode_post_ffn_norm_residual_add_rms_norm_out_batch(
                                        encoder,
                                        &bs.hidden,
                                        &bs.proj_buf,
                                        post_ffn_nw,
                                        next_norm_w,
                                        &bs.norm_buf,
                                        dim as u32,
                                        nt,
                                        eps,
                                    );
                            } else {
                                metal_ops
                                    .elementwise
                                    .encode_residual_add_rms_norm_out_batch(
                                        encoder,
                                        &bs.hidden,
                                        &bs.proj_buf,
                                        next_norm_w,
                                        &bs.norm_buf,
                                        dim as u32,
                                        nt,
                                        eps,
                                    );
                            }
                        }
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => unreachable!(),
                    }
                    if let Some(nout) = residual_norm_out {
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden, nout]);
                    } else {
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    }
                    if let Some(ref mut ops_ref) = ops {
                        let elapsed = residual_handoff_t.elapsed();
                        match residual_plan {
                            PrefillResidualHandoffPlan::ResidualOnly => {
                                ops_ref.gpu_encode_layer_residual += elapsed;
                            }
                            PrefillResidualHandoffPlan::ResidualAddRmsNormF32 => {
                                ops_ref.gpu_encode_layer_residual += elapsed / 2;
                                ops_ref.gpu_encode_layer_norm += elapsed / 2;
                            }
                            PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => unreachable!(),
                        }
                    }
                }

                let fnw_buf = weight_cache.get(&cached.output_norm).unwrap();
                let lm_buf = weight_cache.get(&cached.lm_head).unwrap();
                match DecodeExecutionPlan::prefill_logits_plan(all_logits_buf.is_some()) {
                    PrefillLogitsPlan::BatchAllLogits => {
                        let logits_buf = all_logits_buf.as_ref().unwrap();
                        sb.pre_dispatch(&[&bs.hidden], &[&bs.norm_buf]);
                        metal_ops.elementwise.encode_rms_norm_out_batch(
                            encoder,
                            &bs.hidden,
                            fnw_buf,
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
                            lm_buf,
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
                    }
                    PrefillLogitsPlan::LastTokenMatvec => {
                        sb.pre_dispatch(&[&bs.hidden], &[&s.hidden]);
                        let last_off = (n_tokens - 1) * dim * 4;
                        metal_ops.elementwise.encode_buffer_copy(
                            encoder, &bs.hidden, last_off, &s.hidden, 0, dim as u32,
                        );
                        sb.post_dispatch(&[&bs.hidden], &[&s.hidden]);
                        sb.pre_dispatch(&[&s.hidden], &[&s.hidden]);
                        metal_ops
                            .elementwise
                            .encode_rms_norm(encoder, &s.hidden, fnw_buf, dim as u32, eps);
                        sb.post_dispatch(&[&s.hidden], &[&s.hidden]);
                        sb.pre_dispatch(&[&s.hidden], &[&s.logits_buf]);
                        encode_dequant_matvec(
                            metal_ops,
                            encoder,
                            lm_buf,
                            &s.hidden,
                            &s.logits_buf,
                            vocab_size as u32,
                            dim as u32,
                            cached.lm_head_dtype,
                        );
                    }
                }

                Ok(())
            })?;
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_execute += exec_t.elapsed();
            }
        }

        // v2: advance GPU KV only — no CPU mirror to sync
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
            if let Some(scale) = cfg.logit_scale {
                for l in logits_all.iter_mut() {
                    *l *= scale;
                }
            }
        } else if let Some(logits) = last_logits {
            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    s.logits_buf.contents().as_ptr() as *const f32,
                    vocab_size,
                )
            };
            logits[..vocab_size].copy_from_slice(logits_gpu);
            if let Some(scale) = cfg.logit_scale {
                for l in logits[..vocab_size].iter_mut() {
                    *l *= scale;
                }
            }
        }
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }

        Ok(())
    }
}
