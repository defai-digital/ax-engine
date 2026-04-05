/// LLaMA-family forward pass implementation.
///
/// Used for llama, mistral, and any architecture that follows
/// the standard LLaMA transformer pattern (SwiGLU FFN, no QKV bias).
#[derive(Debug)]
pub struct LlamaForward;

impl LlamaForward {
    /// Build and store pre-computed weight cache keys for all layers (LLaMA architecture).
    /// Called once on first forward pass; subsequent calls are skipped via `has_cached_model_keys()`.
    fn build_cached_model_keys_llama(
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
                        (q_dim + 2 * kv_dim) as u32,
                        dim as u32,
                    )?;
                }
                if wq_dtype == GgmlType::Q8_0 {
                    metal_ops.ensure_precomputed_q8_0_f16_fused_qkv(
                        wq_raw,
                        wk_raw,
                        wv_raw,
                        (q_dim + 2 * kv_dim) as u32,
                        dim as u32,
                    )?;
                }
            }
            if use_precomputed_f16
                && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0)
            {
                ensure_precomputed_linear_f16_many(
                    metal_ops,
                    &[
                        (wq_raw, wq_dtype, q_dim as u32, dim as u32),
                        (wk_raw, wk_dtype, kv_dim as u32, dim as u32),
                        (wv_raw, wv_dtype, kv_dim as u32, dim as u32),
                    ],
                )?;
            }
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
                ensure_precomputed_linear_f16(metal_ops, wo_raw, wo_dtype, dim as u32, dim as u32)?;
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
                attn_q_norm: None,
                attn_k_norm: None,
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
    /// v2: receives `gpu_kv: &mut GpuKv` directly (no Mutex, no advance_by).
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
        let vocab_size = cfg.vocab_size as usize;
        let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as usize;
        let exec_plan = DecodeExecutionPlan::llama_single_cb(
            metal_ops,
            gpu_kv,
            cfg.embedding_dim,
            cfg.head_dim,
            gpu_kv.seq_len() + 1,
        );
        debug_assert_eq!(
            exec_plan.dequant_dispatch,
            metal_ops.dequant_dispatch_config(),
            "llama decode execution plan must match current Metal dequant dispatch config"
        );
        debug_assert_eq!(
            exec_plan.attention_dispatch,
            metal_ops.attention_dispatch_config(),
            "llama decode execution plan must match current Metal attention dispatch config"
        );

        assert!(logits.len() >= vocab_size);

        match DecodeScratchPlan::SharedGpuScratch {
            DecodeScratchPlan::SharedGpuScratch => metal_ops.init_scratches(cfg),
            DecodeScratchPlan::CpuScratch | DecodeScratchPlan::HybridBackendOwned => {
                anyhow::bail!("single-CB GPU decode requires GPU scratch")
            }
        }

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();

        let next_seq = gpu_kv.seq_len() + 1;
        gpu_kv.ensure_capacity(&metal_ops.device, next_seq)?;

        // Token embedding + setup on host side before GPU execute.
        let setup_t = OpTimer::start();
        {
            let hidden_cpu = unsafe {
                std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim)
            };
            weights.dequantize_row("token_embd.weight", token_id as usize, hidden_cpu)?;
        }

        let rope_position = cfg.rope_scaling.scaled_position(position);

        // Pre-cache ALL weights and build cached keys (first call only)
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_llama(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();

        let cur_seq_len = gpu_kv.seq_len();
        let kv_offset = (cur_seq_len * kv_dim) as u32;
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        // GPU execution. Profiling must not alter the command-buffer shape, so
        // even when `ops` is present we keep the same single execute_sync path.
        {
            let weight_cache = metal_ops.lock_weight_cache();
            let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();
            let exec_gpu = |ops_arg: Option<&mut OpBreakdown>,
                            encoder: &ax_engine_metal::MetalEncoder|
             -> anyhow::Result<()> {
                let barrier =
                    crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
                encode_llama_gpu_layers_only(
                    encoder,
                    metal_ops,
                    s,
                    &s.hidden,
                    cfg,
                    kv_offset,
                    rope_position,
                    cur_seq_len + 1,
                    &exec_plan,
                    gpu_kv,
                    cached,
                    &weight_cache,
                    &fused_qkv_cache,
                    ops_arg,
                    &barrier,
                )?;
                encode_llama_gpu_output_head(
                    encoder,
                    metal_ops,
                    s,
                    &s.hidden,
                    cfg,
                    cached,
                    &weight_cache,
                    &exec_plan,
                    &barrier,
                );
                barrier.flush();
                Ok(())
            };
            let use_concurrent = exec_plan.encoder == DecodeEncoderPlan::Concurrent;
            if let Some(ref mut ops_ref) = ops {
                let exec_t = OpTimer::start();
                if use_concurrent {
                    metal_ops
                        .device
                        .execute_sync_concurrent(|e| exec_gpu(Some(ops_ref), e))?;
                } else {
                    metal_ops
                        .device
                        .execute_sync(|e| exec_gpu(Some(ops_ref), e))?;
                }
                ops_ref.gpu_execute += exec_t.elapsed();
            } else {
                let exec_t = OpTimer::start();
                if use_concurrent {
                    metal_ops
                        .device
                        .execute_sync_concurrent(|e| exec_gpu(None, e))?;
                } else {
                    metal_ops.device.execute_sync(|e| exec_gpu(None, e))?;
                }
                let _ = exec_t;
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

        Ok(())
    }

    /// Batched GPU prefill: process N tokens through all layers in minimal command buffers.
    ///
    /// Uses batch scratch buffers [N × dim] for hidden, Q, K, V, attn_out.
    /// Attention runs either batch-local prefill (empty prefix) or cache-backed prefill
    /// (when KV prefix already exists).
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

        let setup_t = OpTimer::start();

        // Initialize buffers
        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, n_tokens);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let batch_guard = metal_ops.batch_scratches();
        let bs = batch_guard.as_ref().unwrap();

        // Ensure capacity for all N tokens
        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + n_tokens)?;

        // Embed all N tokens into batch hidden buffer via CPU UMA write
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

        // Pre-cache weights and build cached keys (first call only)
        // The forward_single_gpu_unified path may have already built them.
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_llama(metal_ops, weights, cfg)?;
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
            let has_q8_weights = cached.layers.iter().any(|lw| {
                lw.wq_dtype == GgmlType::Q8_0
                    || lw.wk_dtype == GgmlType::Q8_0
                    || lw.wv_dtype == GgmlType::Q8_0
                    || lw.wo_dtype == GgmlType::Q8_0
                    || lw.wg_dtype == GgmlType::Q8_0
                    || lw.wu_dtype == GgmlType::Q8_0
                    || lw.wd_dtype == GgmlType::Q8_0
            }) || matches!(cached.lm_head_dtype, GgmlType::Q8_0);
            let has_q5k_weights = gpu_prefill_uses_q5k(weights);
            let q5k_small_n_auto_eligible = gpu_prefill_q5k_small_n_auto_eligible(weights);
            let prefill_plan: GpuBatchPrefillExecutionPlan = DecodeExecutionPlan::llama_prefill(
                metal_ops,
                gpu_kv,
                base_seq_len,
                n_tokens as u32,
                cfg.head_dim,
                has_q8_weights,
                has_q5k_weights,
                q5k_small_n_auto_eligible,
                metal_prefill_attn_f16out_enabled(),
                metal_prefill_use_cached0_enabled(),
                metal_prefill_split_rope_append_enabled(),
            );
            let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

            // ── Graph IR path: pre-computed dispatch schedule ──
            if crate::model::prefill_schedule::prefill_graph_ir_enabled() {
                let schedule = crate::model::prefill_schedule::build_llama_prefill_schedule(
                    cfg,
                    &prefill_plan,
                    cached,
                    &weight_cache,
                    bs,
                    s,
                    gpu_kv,
                    base_seq_len,
                    n_tokens,
                    all_logits_buf.as_ref(),
                    &fused_qkv_cache,
                );
                let exec_t = OpTimer::start();
                let result = crate::model::prefill_schedule::execute_prefill_multi_cb(
                    &metal_ops.device,
                    &schedule,
                    metal_ops,
                );
                if let Some(ref mut ops_ref) = ops {
                    ops_ref.gpu_execute += exec_t.elapsed();
                }
                return result;
            }

            // ── Inline path (default): existing SmartBarrier encoding ──
            let exec_t = OpTimer::start();
            metal_ops.device.execute_sync_concurrent(|encoder| {
                let nt = n_tokens as u32;
                let mut sb = ax_engine_metal::SmartBarrier::new(encoder);

                // First layer's Phase 1a: RMSNorm (standalone, before loop).
                // Subsequent layers' Phase 1a is fused with the previous layer's
                // Phase 3f residual add (saves 1 dispatch + 1 barrier per layer).
                {
                    let t = OpTimer::start();
                    let norm_w_buf = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
                    if prefill_plan.use_f16_batch_io {
                        metal_ops.elementwise.encode_rms_norm_out_batch_f16(
                            encoder,
                            &bs.hidden,
                            norm_w_buf,
                            &bs.matmul_in_f16,
                            dim as u32,
                            nt,
                            eps,
                        );
                        sb.post_dispatch(&[&bs.hidden], &[&bs.matmul_in_f16]);
                    } else {
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
                    }
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_norm += t.elapsed();
                    }
                }

                for layer in 0..n_layers {
                    let lw = &cached.layers[layer];
                    let (rope_start, rope_step) = cfg.rope_scaling.scaled_start_step(base_seq_len);

                    let wq_buf = weight_cache.get(&lw.wq).unwrap();
                    let wk_buf = weight_cache.get(&lw.wk).unwrap();
                    let wv_buf = weight_cache.get(&lw.wv).unwrap();
                    let fused_qkv_m = q_dim + 2 * kv_dim;
                    let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
                    let qkv_layer_plan = DecodeExecutionPlan::llama_prefill_qkv_layer(
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

                    // Phase 1a (RMSNorm) is done before loop for layer 0, and
                    // fused with previous layer's Phase 3f for layers 1+.
                    // norm_buf / matmul_in_f16 is already populated.

                    // ── Phase 1b: Batched QKV matmul ──
                    let qkv_t = OpTimer::start();
                    if let Some(fused_w) = fused_qkv_buf {
                        let cache_offset = (base_seq_len * kv_dim) as u32;
                        let qkv_input = if matches!(
                            qkv_layer_plan.input,
                            PrefillProjectionInputPlan::MatmulScratchF16
                        ) {
                            &bs.matmul_in_f16
                        } else {
                            &bs.norm_buf
                        };
                        sb.pre_dispatch(&[qkv_input], &[&bs.qkv_buf]);
                        match qkv_layer_plan.input {
                            PrefillProjectionInputPlan::MatmulScratchF16 => {
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
                        if qkv_layer_plan.llama_post
                            == crate::model::execution_plan::LlamaPrefillQkvPostPlan::FusedSplitRopeAppendKv
                        {
                            let kv_k = gpu_kv.k_buffer(layer);
                            let kv_v = gpu_kv.v_buffer(layer);
                            sb.pre_dispatch(
                                &[&bs.qkv_buf],
                                &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v],
                            );
                            metal_ops.elementwise.encode_qkv_split_rope_append_kv_batch(
                                encoder,
                                &bs.qkv_buf,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                kv_k,
                                kv_v,
                                prefill_plan.kv_f16,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                rope_start,
                                rope_step,
                                cfg.rope_freq_base,
                                cache_offset,
                                kv_dim as u32,
                            );
                            sb.post_dispatch(
                                &[&bs.qkv_buf],
                                &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v],
                            );
                        } else {
                            sb.pre_dispatch(
                                &[&bs.qkv_buf],
                                &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            );
                            metal_ops.elementwise.encode_qkv_split_rope_batch(
                                encoder,
                                &bs.qkv_buf,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                rope_start,
                                rope_step,
                                cfg.rope_freq_base,
                            );
                            sb.post_dispatch(
                                &[&bs.qkv_buf],
                                &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            );
                        }
                    } else {
                        // Separate Q/K/V projections — these can overlap when
                        // all three read norm_buf and write different outputs.
                        match qkv_layer_plan.input {
                            PrefillProjectionInputPlan::MatmulScratchF16 => {
                                debug_assert!(prefill_plan.use_f16_batch_io);
                                // f16 path: all three share matmul_in_f16, can't overlap.
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.q_buf]);
                                encode_dequant_batch_f16in(
                                    metal_ops, encoder, wq_buf,
                                    &bs.matmul_in_f16, &bs.q_buf,
                                    q_dim as u32, nt, dim as u32, lw.wq_dtype,
                                );
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.q_buf]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.k_buf]);
                                encode_dequant_batch_f16in(
                                    metal_ops, encoder, wk_buf,
                                    &bs.matmul_in_f16, &bs.k_buf,
                                    kv_dim as u32, nt, dim as u32, lw.wk_dtype,
                                );
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.k_buf]);
                                sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.v_buf]);
                                encode_dequant_batch_f16in(
                                    metal_ops, encoder, wv_buf,
                                    &bs.matmul_in_f16, &bs.v_buf,
                                    kv_dim as u32, nt, dim as u32, lw.wv_dtype,
                                );
                                sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.v_buf]);
                            }
                            PrefillProjectionInputPlan::NormBufF32 => {
                                // f32 path: Q/K/V all read norm_buf, write
                                // different buffers → SmartBarrier skips
                                // barriers between them (GPU can overlap).
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.q_buf]);
                                encode_dequant_batch(
                                    &metal_ops.dequant, &metal_ops.elementwise,
                                    encoder, wq_buf, &bs.norm_buf, &bs.q_buf,
                                    &bs.matmul_in_f16, q_dim as u32, nt,
                                    dim as u32, lw.wq_dtype, false,
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.q_buf]);
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.k_buf]);
                                encode_dequant_batch(
                                    &metal_ops.dequant, &metal_ops.elementwise,
                                    encoder, wk_buf, &bs.norm_buf, &bs.k_buf,
                                    &bs.matmul_in_f16, kv_dim as u32, nt,
                                    dim as u32, lw.wk_dtype, false,
                                    prefill_plan.use_batch_simd,
                                    prefill_plan.q5k_prefill_small_n,
                                );
                                sb.post_dispatch(&[&bs.norm_buf], &[&bs.k_buf]);
                                sb.pre_dispatch(&[&bs.norm_buf], &[&bs.v_buf]);
                                encode_dequant_batch(
                                    &metal_ops.dequant, &metal_ops.elementwise,
                                    encoder, wv_buf, &bs.norm_buf, &bs.v_buf,
                                    &bs.matmul_in_f16, kv_dim as u32, nt,
                                    dim as u32, lw.wv_dtype, false,
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

                    // ── Phase 1c: Batched RoPE + batched KV cache append ──
                    let rope_kv_t = OpTimer::start();
                    if qkv_layer_plan.llama_post
                        == crate::model::execution_plan::LlamaPrefillQkvPostPlan::Separate
                    {
                        sb.pre_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);
                        metal_ops.elementwise.encode_rope_batch(
                            encoder,
                            &bs.q_buf,
                            &bs.k_buf,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            head_dim as u32,
                            rope_start,
                            rope_step,
                            cfg.rope_freq_base,
                        );
                        sb.post_dispatch(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);
                    }

                    if qkv_layer_plan.llama_post
                        != crate::model::execution_plan::LlamaPrefillQkvPostPlan::FusedSplitRopeAppendKv
                    {
                        let cache_offset = (base_seq_len * kv_dim) as u32;
                        let kv_k = gpu_kv.k_buffer(layer);
                        let kv_v = gpu_kv.v_buffer(layer);
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
                    if prefill_plan.attention == PrefillAttentionPlan::BatchLocalF16OutHd128 {
                        sb.pre_dispatch(
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            &[&bs.matmul_in_f16],
                        );
                        metal_ops.attention.encode_attention_prefill_f16out_hd128(
                            encoder,
                            &bs.q_buf,
                            &bs.k_buf,
                            &bs.v_buf,
                            &bs.matmul_in_f16,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                        );
                        sb.post_dispatch(
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            &[&bs.matmul_in_f16],
                        );
                    } else if prefill_plan.attention == PrefillAttentionPlan::BatchLocal {
                        sb.pre_dispatch(
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            &[&bs.attn_out],
                        );
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
                        sb.post_dispatch(
                            &[&bs.q_buf, &bs.k_buf, &bs.v_buf],
                            &[&bs.attn_out],
                        );
                    } else {
                        let kv_k = gpu_kv.k_buffer(layer);
                        let kv_v = gpu_kv.v_buffer(layer);
                        sb.pre_dispatch(&[&bs.q_buf, kv_k, kv_v], &[&bs.attn_out]);
                        if prefill_plan.kv_q8 {
                            metal_ops
                                .attention
                                .encode_attention_prefill_cached_q8kv(
                                    encoder,
                                    &bs.q_buf,
                                    kv_k,
                                    kv_v,
                                    &bs.attn_out,
                                    nt,
                                    n_heads as u32,
                                    n_kv_heads as u32,
                                    head_dim as u32,
                                    base_seq_len as u32,
                                    0,
                                );
                        } else {
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
                                    n_heads as u32,
                                    n_kv_heads as u32,
                                    head_dim as u32,
                                    base_seq_len as u32,
                                    0,
                                    prefill_plan.attention_dispatch,
                                );
                        }
                        sb.post_dispatch(&[&bs.q_buf, kv_k, kv_v], &[&bs.attn_out]);
                    }
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_attention += attn_t.elapsed();
                    }

                    // ── Phase 3a: Batched output projection ──
                    let out_proj_t = OpTimer::start();
                    let wo_buf = weight_cache.get(&lw.wo).unwrap();
                    let wo_input = if prefill_plan.use_f16_batch_io {
                        &bs.matmul_in_f16
                    } else {
                        &bs.attn_out
                    };
                    sb.pre_dispatch(&[wo_input], &[&bs.proj_buf]);
                    if prefill_plan.use_f16_batch_io {
                        if prefill_plan.wo_input == PrefillWoInputPlan::AttentionOutF32 {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.attn_out,
                                &bs.matmul_in_f16,
                                nt * q_dim as u32,
                            );
                        }
                        encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wo_buf,
                            &bs.matmul_in_f16,
                            &bs.proj_buf,
                            dim as u32,
                            nt,
                            q_dim as u32,
                            lw.wo_dtype,
                        );
                    } else {
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
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.q5k_prefill_small_n,
                        );
                    }
                    sb.post_dispatch(&[wo_input], &[&bs.proj_buf]);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_out_proj += out_proj_t.elapsed();
                    }

                    // ── Phase 3b: Batched residual + FFN norm ──
                    let residual_norm_t = OpTimer::start();
                    let ffn_nw_buf = weight_cache.get(&lw.ffn_norm).unwrap();
                    let ffn_norm_out = if prefill_plan.use_f16_batch_io {
                        &bs.matmul_in_f16
                    } else {
                        &bs.norm_buf
                    };
                    sb.pre_dispatch(
                        &[&bs.hidden, &bs.proj_buf],
                        &[&bs.hidden, ffn_norm_out],
                    );
                    if prefill_plan.use_f16_batch_io {
                        metal_ops
                            .elementwise
                            .encode_residual_add_rms_norm_out_batch_f16(
                                encoder,
                                &bs.hidden,
                                &bs.proj_buf,
                                ffn_nw_buf,
                                &bs.matmul_in_f16,
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
                                ffn_nw_buf,
                                &bs.norm_buf,
                                dim as u32,
                                nt,
                                eps,
                            );
                    }
                    sb.post_dispatch(
                        &[&bs.hidden, &bs.proj_buf],
                        &[&bs.hidden, ffn_norm_out],
                    );
                    if let Some(ref mut ops_ref) = ops {
                        let elapsed = residual_norm_t.elapsed();
                        ops_ref.gpu_encode_layer_residual += elapsed / 2;
                        ops_ref.gpu_encode_layer_norm += elapsed / 2;
                    }

                    // ── Phase 3c: Batched gate + up ──
                    let gate_up_t = OpTimer::start();
                    let wg_buf = weight_cache.get(&lw.wg).unwrap();
                    let wu_buf = weight_cache.get(&lw.wu).unwrap();
                    let ffn_layer_plan = DecodeExecutionPlan::llama_prefill_ffn_layer(
                        &prefill_plan,
                        lw.wg_dtype,
                        lw.wu_dtype,
                    );

                    let ffn_input = if prefill_plan.use_f16_batch_io {
                        &bs.matmul_in_f16
                    } else {
                        &bs.norm_buf
                    };
                    match ffn_layer_plan.input {
                        PrefillProjectionInputPlan::MatmulScratchF16 => {
                            debug_assert!(prefill_plan.use_f16_batch_io);
                            sb.pre_dispatch(&[&bs.matmul_in_f16], &[&bs.gate_buf, &bs.up_buf]);
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
                            sb.post_dispatch(&[&bs.matmul_in_f16], &[&bs.gate_buf, &bs.up_buf]);
                        }
                        PrefillProjectionInputPlan::NormBufF32 => {
                            sb.pre_dispatch(&[ffn_input], &[&bs.gate_buf, &bs.up_buf]);
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
                            sb.post_dispatch(&[ffn_input], &[&bs.gate_buf, &bs.up_buf]);
                        }
                    }
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_ffn += gate_up_t.elapsed();
                    }

                    // ── Phase 3d: Batched activation ──
                    let activation_t = OpTimer::start();
                    let act_out = match ffn_layer_plan.activation {
                        PrefillFfnActivationPlan::SiluMulScratchF16 => &bs.matmul_in_f16,
                        _ => &bs.gate_buf,
                    };
                    sb.pre_dispatch(&[&bs.gate_buf, &bs.up_buf], &[act_out]);
                    match ffn_layer_plan.activation {
                        PrefillFfnActivationPlan::SiluMulScratchF16 => {
                            metal_ops.elementwise.encode_silu_elementwise_mul_batch_f16(
                                encoder,
                                &bs.gate_buf,
                                &bs.up_buf,
                                &bs.matmul_in_f16,
                                inter_dim as u32,
                                nt,
                            );
                        }
                        PrefillFfnActivationPlan::SiluMulGateF32 => {
                            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                                encoder,
                                &bs.gate_buf,
                                &bs.up_buf,
                                inter_dim as u32,
                                nt,
                            );
                        }
                        PrefillFfnActivationPlan::GeluMulGateF32 => unreachable!(),
                    }
                    sb.post_dispatch(&[&bs.gate_buf, &bs.up_buf], &[act_out]);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_ffn += activation_t.elapsed();
                    }

                    // ── Phase 3e: Batched down projection ──
                    let down_proj_t = OpTimer::start();
                    let wd_buf = weight_cache.get(&lw.wd).unwrap();
                    sb.pre_dispatch(&[act_out], &[&bs.proj_buf]);
                    match ffn_layer_plan.activation {
                        PrefillFfnActivationPlan::SiluMulScratchF16 => {
                            encode_dequant_batch_f16in(
                                metal_ops,
                                encoder,
                                wd_buf,
                                &bs.matmul_in_f16,
                                &bs.proj_buf,
                                dim as u32,
                                nt,
                                inter_dim as u32,
                                lw.wd_dtype,
                            );
                        }
                        PrefillFfnActivationPlan::SiluMulGateF32 => {
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
                                false,
                                prefill_plan.use_batch_simd,
                                prefill_plan.q5k_prefill_small_n,
                            );
                        }
                        PrefillFfnActivationPlan::GeluMulGateF32 => unreachable!(),
                    }
                    sb.post_dispatch(&[act_out], &[&bs.proj_buf]);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_ffn += down_proj_t.elapsed();
                    }

                    // ── Phase 3f: Batched residual (+ next layer's norm if not last) ──
                    let residual_handoff_t = OpTimer::start();
                    let residual_plan = DecodeExecutionPlan::llama_prefill_residual_handoff(
                        &prefill_plan,
                        layer + 1 == n_layers,
                    );
                    let residual_norm_out = match residual_plan {
                        PrefillResidualHandoffPlan::ResidualOnly => None,
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF32 => {
                            Some(&bs.norm_buf)
                        }
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => {
                            Some(&bs.matmul_in_f16)
                        }
                    };
                    if let Some(nout) = residual_norm_out {
                        sb.pre_dispatch(
                            &[&bs.hidden, &bs.proj_buf],
                            &[&bs.hidden, nout],
                        );
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
                        PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => {
                            let next_norm_w = weight_cache
                                .get(&cached.layers[layer + 1].attn_norm)
                                .unwrap();
                            metal_ops
                                .elementwise
                                .encode_residual_add_rms_norm_out_batch_f16(
                                    encoder,
                                    &bs.hidden,
                                    &bs.proj_buf,
                                    next_norm_w,
                                    &bs.matmul_in_f16,
                                    dim as u32,
                                    nt,
                                    eps,
                                );
                        }
                    }
                    if let Some(nout) = residual_norm_out {
                        sb.post_dispatch(
                            &[&bs.hidden, &bs.proj_buf],
                            &[&bs.hidden, nout],
                        );
                    } else {
                        sb.post_dispatch(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                    }
                    if let Some(ref mut ops_ref) = ops {
                        let elapsed = residual_handoff_t.elapsed();
                        match residual_plan {
                            PrefillResidualHandoffPlan::ResidualOnly => {
                                ops_ref.gpu_encode_layer_residual += elapsed;
                            }
                            PrefillResidualHandoffPlan::ResidualAddRmsNormF32
                            | PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => {
                                ops_ref.gpu_encode_layer_residual += elapsed / 2;
                                ops_ref.gpu_encode_layer_norm += elapsed / 2;
                            }
                        }
                    }
                }

                // ── Post-loop: Final norm + LM head ──
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
        } else if let Some(logits) = last_logits {
            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    s.logits_buf.contents().as_ptr() as *const f32,
                    vocab_size,
                )
            };
            logits[..vocab_size].copy_from_slice(logits_gpu);
        }
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }

        Ok(())
    }
}
