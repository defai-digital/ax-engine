use crate::backend::Backend;
use crate::backend::metal::MetalOps;
use crate::model::prefill_schedule;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen35UnifiedRecurrentGpuPlan {
    qkv_gpu_fast_path_enabled: bool,
    keep_rec_z_on_gpu: bool,
    fused_gpu_recurrent_layer_candidate: bool,
    fused_gpu_recurrent_layer_uses_gpu_alpha_beta: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen35UnifiedRecurrentTailBufferPlan {
    use_unified_tail: bool,
    alias_rec_out: bool,
    alias_rec_z: bool,
}

impl Qwen3_5Forward {
    fn lock_cpu_batch_fallback_scratch()
    -> std::sync::MutexGuard<'static, Qwen3_5CpuBatchFallbackScratch> {
        Self::cpu_batch_fallback_scratch()
            .lock()
            .expect("qwen35 cpu batch fallback scratch mutex should not be poisoned")
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_serial_fallback(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        start_position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: Option<&mut [f32]>,
        logits_all: Option<&mut Vec<f32>>,
    ) -> anyhow::Result<()> {
        match (logits, logits_all) {
            (Some(logits), None) => {
                for (i, &tid) in token_ids.iter().enumerate() {
                    logits.fill(0.0);
                    self.forward_single(ctx, tid, start_position + i, kv, weights, logits, None)?;
                }
                Ok(())
            }
            (None, Some(logits_all)) => {
                let vocab_size = ctx.config.vocab_size as usize;
                logits_all.resize(token_ids.len() * vocab_size, 0.0);
                for (i, &tid) in token_ids.iter().enumerate() {
                    let slot = &mut logits_all[i * vocab_size..(i + 1) * vocab_size];
                    slot.fill(0.0);
                    self.forward_single(ctx, tid, start_position + i, kv, weights, slot, None)?;
                }
                Ok(())
            }
            _ => anyhow::bail!(
                "qwen35 batch forward requires either last logits or all logits output"
            ),
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn try_run_moe_ffn_gpu_resident_batch(
        metal_ops: &MetalOps,
        cached_layer: &crate::backend::metal::CachedLayerKeys,
        moe_layer: Option<&Qwen3_5MoeResidentLayerKeys>,
        n_tokens: usize,
        dim: usize,
        eps: f32,
    ) -> anyhow::Result<bool> {
        if cached_layer.wg != 0 {
            return Ok(false);
        }
        let Some(moe_layer) = moe_layer else {
            return Ok(false);
        };

        let hidden_ptr: *const ax_engine_metal::MetalBuffer = {
            let scratch_guard = metal_ops.batch_scratches();
            match scratch_guard.as_ref() {
                Some(scratches) => &scratches.hidden as *const _,
                None => std::ptr::null(),
            }
        };
        if hidden_ptr.is_null() {
            return Ok(false);
        }

        let hidden_gpu = unsafe { &*hidden_ptr };
        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        let shared_expert = moe_layer.shared_expert.map(|shared| {
            crate::backend::metal::SharedExpertCachedBuffers {
                gate: moe_weight_cache.get(&shared.gate).unwrap(),
                up: moe_weight_cache.get(&shared.up).unwrap(),
                down: moe_weight_cache.get(&shared.down).unwrap(),
                gate_inp: shared
                    .gate_inp
                    .map(|gate_inp| moe_weight_cache.get(&gate_inp).unwrap()),
                gate_inp_dtype: shared.gate_inp_dtype,
                dtype: shared.dtype,
                inter_dim: shared.inter_dim,
                gate_inp_rows: shared.gate_inp_rows,
            }
        });
        match metal_ops.moe_ffn_gpu_resident_cached(
            hidden_gpu,
            weight_cache.get(&cached_layer.ffn_norm).unwrap(),
            moe_weight_cache.get(&moe_layer.router).unwrap(),
            moe_layer.router_dtype,
            moe_weight_cache.get(&moe_layer.gate).unwrap(),
            moe_layer.gate_dtype,
            moe_weight_cache.get(&moe_layer.up).unwrap(),
            moe_layer.up_dtype,
            moe_weight_cache.get(&moe_layer.down).unwrap(),
            moe_layer.down_dtype,
            n_tokens,
            moe_layer.n_expert,
            moe_layer.n_expert_used,
            dim,
            moe_layer.expert_inter_dim,
            moe_layer.gate_stride,
            moe_layer.up_stride,
            moe_layer.down_stride,
            eps,
            shared_expert.as_ref(),
        ) {
            Ok(()) => Ok(true),
            Err(err) => {
                if std::env::var("AX_DEBUG_QWEN35_PREFILL_MOE").is_ok() {
                    eprintln!(
                        "[QWEN35 PREFILL MOE FALLBACK] n_tokens={n_tokens} dim={dim} n_expert={} n_expert_used={} gate_dtype={:?} up_dtype={:?} down_dtype={:?} err={err}",
                        moe_layer.n_expert,
                        moe_layer.n_expert_used,
                        moe_layer.gate_dtype,
                        moe_layer.up_dtype,
                        moe_layer.down_dtype,
                    );
                }
                tracing::debug!(
                    "qwen35 batch MoE GPU-resident FFN failed; falling back to CPU path: {err}",
                );
                Ok(false)
            }
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn run_post_attention_ffn_batch_with_moe_fallback(
        metal_ops: &MetalOps,
        cfg: &ModelConfig,
        backend: &dyn Backend,
        weights: &WeightStore,
        prefix: &str,
        cached_layer: &crate::backend::metal::CachedLayerKeys,
        moe_layer: Option<&Qwen3_5MoeResidentLayerKeys>,
        hidden: &mut [f32],
        norm_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        eps: f32,
    ) -> anyhow::Result<()> {
        if Self::try_run_moe_ffn_gpu_resident_batch(
            metal_ops,
            cached_layer,
            moe_layer,
            n_tokens,
            dim,
            eps,
        )? {
            return Ok(());
        }

        let mut ffn_gate = vec![0.0f32; n_tokens * inter_dim];
        let mut ffn_up = vec![0.0f32; n_tokens * inter_dim];
        let mut ffn_down = vec![0.0f32; n_tokens * dim];
        Self::apply_post_attention_ffn_batch(
            cfg,
            backend,
            weights,
            prefix,
            hidden,
            norm_buf,
            &mut ffn_gate,
            &mut ffn_up,
            &mut ffn_down,
            n_tokens,
            dim,
            inter_dim,
            eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_qwen35_full_attention_hidden_before_moe_gpu(
        metal_ops: &MetalOps,
        cfg: &ModelConfig,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        cached_layer: &crate::backend::metal::CachedLayerKeys,
        layer: usize,
        batch_position: usize,
        n_tokens: usize,
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        nt: u32,
        eps: f32,
        defer_kv_readback: bool,
    ) -> anyhow::Result<bool> {
        if cached_layer.wg != 0 {
            return Ok(false);
        }
        let wq_dtype = cached_layer.wq_dtype;
        let wk_dtype = cached_layer.wk_dtype;
        let wv_dtype = cached_layer.wv_dtype;
        let wo_dtype = cached_layer.wo_dtype;
        let gpu_batch_supported = |dt: crate::gguf::tensor::GgmlType, m: u32, k: u32| {
            Self::qwen35_batch_projection_supported(metal_ops, dt, m, n_tokens as u32, k)
        };
        if !gpu_batch_supported(wq_dtype, (q_dim * 2) as u32, dim as u32)
            || !gpu_batch_supported(wk_dtype, kv_dim as u32, dim as u32)
            || !gpu_batch_supported(wv_dtype, kv_dim as u32, dim as u32)
            || !gpu_batch_supported(wo_dtype, dim as u32, q_dim as u32)
        {
            return Ok(false);
        }

        let q_norm_key = cached_layer.attn_q_norm;
        let k_norm_key = cached_layer.attn_k_norm;
        let (rope_start, rope_step) = cfg.rope_scaling.scaled_start_step(batch_position);
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };
        let weight_cache = metal_ops.lock_weight_cache();
        let nw_buf = weight_cache.get(&cached_layer.attn_norm).unwrap();
        let wq_buf = weight_cache.get(&cached_layer.wq).unwrap();
        let wk_buf = weight_cache.get(&cached_layer.wk).unwrap();
        let wv_buf = weight_cache.get(&cached_layer.wv).unwrap();
        let wo_buf = weight_cache.get(&cached_layer.wo).unwrap();

        metal_ops.device.execute_sync(|encoder| {
            let qkv_uses_f16 = Self::qwen35_batch_projection_needs_f16_input(wq_dtype)
                || Self::qwen35_batch_projection_needs_f16_input(wk_dtype)
                || Self::qwen35_batch_projection_needs_f16_input(wv_dtype);
            metal_ops.elementwise.encode_rms_norm_out_batch(
                encoder,
                &bs.hidden,
                nw_buf,
                &bs.norm_buf,
                dim as u32,
                nt,
                eps,
            );
            ax_engine_metal::barrier_buffers(encoder);
            if qkv_uses_f16 {
                metal_ops.elementwise.encode_cast_f32_to_f16(
                    encoder,
                    &bs.norm_buf,
                    &bs.matmul_in_f16,
                    nt * dim as u32,
                );
                ax_engine_metal::barrier_buffers(encoder);
            }
            Self::encode_qwen35_batch_projection(
                metal_ops,
                encoder,
                wq_buf,
                &bs.norm_buf,
                &bs.matmul_in_f16,
                &bs.gate_buf,
                (q_dim * 2) as u32,
                nt,
                dim as u32,
                wq_dtype,
            );
            Self::encode_qwen35_batch_projection(
                metal_ops,
                encoder,
                wk_buf,
                &bs.norm_buf,
                &bs.matmul_in_f16,
                &bs.k_buf,
                kv_dim as u32,
                nt,
                dim as u32,
                wk_dtype,
            );
            Self::encode_qwen35_batch_projection(
                metal_ops,
                encoder,
                wv_buf,
                &bs.norm_buf,
                &bs.matmul_in_f16,
                &bs.v_buf,
                kv_dim as u32,
                nt,
                dim as u32,
                wv_dtype,
            );
            ax_engine_metal::barrier_buffers(encoder);
            metal_ops.elementwise.encode_split_qgate_batch(
                encoder,
                &bs.gate_buf,
                &bs.q_buf,
                &bs.up_buf,
                nt,
                q_dim as u32,
                head_dim as u32,
            );
            ax_engine_metal::barrier_buffers(encoder);
            if let (Some(q_key), Some(k_key)) = (q_norm_key, k_norm_key) {
                let q_nw = weight_cache.get(&q_key).unwrap();
                let k_nw = weight_cache.get(&k_key).unwrap();
                metal_ops.elementwise.encode_per_head_rms_norm_batch(
                    encoder,
                    &bs.q_buf,
                    q_nw,
                    nt,
                    n_heads as u32,
                    head_dim as u32,
                    eps,
                );
                ax_engine_metal::barrier_buffers(encoder);
                metal_ops.elementwise.encode_per_head_rms_norm_batch(
                    encoder,
                    &bs.k_buf,
                    k_nw,
                    nt,
                    n_kv_heads as u32,
                    head_dim as u32,
                    eps,
                );
                ax_engine_metal::barrier_buffers(encoder);
            }
            metal_ops.elementwise.encode_rope_batch_neox_partial(
                encoder,
                &bs.q_buf,
                &bs.k_buf,
                nt,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                (head_dim as u32).min(64),
                rope_start,
                rope_step,
                cfg.rope_freq_base,
            );
            ax_engine_metal::barrier_buffers(encoder);
            if let Some(gpu_attn) = qwen_kv.gpu_attention() {
                if gpu_attn.is_q8() {
                    let blocks_per_row = kv_dim / crate::kv::gpu_kv::Q8_0_BLOCK_VALUES;
                    let row_offset = (batch_position * blocks_per_row) as u32;
                    metal_ops.elementwise.encode_kv_append_batch_pair_q8(
                        encoder,
                        &bs.k_buf,
                        &bs.v_buf,
                        gpu_attn.k_buffer(layer),
                        gpu_attn.v_buffer(layer),
                        row_offset,
                        blocks_per_row as u32,
                        kv_dim as u32,
                        nt,
                    );
                } else {
                    let cache_offset = (batch_position * kv_dim) as u32;
                    metal_ops.elementwise.encode_kv_append_batch_pair(
                        encoder,
                        &bs.k_buf,
                        &bs.v_buf,
                        gpu_attn.k_buffer(layer),
                        gpu_attn.v_buffer(layer),
                        gpu_attn.is_f16(),
                        cache_offset,
                        kv_dim as u32,
                        kv_dim as u32,
                        nt,
                    );
                }
                ax_engine_metal::barrier_buffers(encoder);
            }
            if batch_position == 0 {
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
                    metal_ops.attention_dispatch_config(),
                );
            } else if let Some(gpu_attn) = qwen_kv.gpu_attention() {
                metal_ops
                    .attention
                    .encode_attention_prefill_cached_with_config(
                        encoder,
                        &bs.q_buf,
                        gpu_attn.k_buffer(layer),
                        gpu_attn.v_buffer(layer),
                        &bs.attn_out,
                        gpu_attn.is_f16(),
                        nt,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        batch_position as u32,
                        0,
                        metal_ops.attention_dispatch_config(),
                    );
            }
            ax_engine_metal::barrier_buffers(encoder);
            metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                encoder,
                &bs.up_buf,
                &bs.attn_out,
                (n_tokens * q_dim) as u32,
            );
            ax_engine_metal::barrier_buffers(encoder);
            if Self::qwen35_batch_projection_needs_f16_input(wo_dtype) {
                metal_ops.elementwise.encode_cast_f32_to_f16(
                    encoder,
                    &bs.attn_out,
                    &bs.matmul_in_f16,
                    nt * q_dim as u32,
                );
                ax_engine_metal::barrier_buffers(encoder);
            }
            Self::encode_qwen35_batch_projection(
                metal_ops,
                encoder,
                wo_buf,
                &bs.attn_out,
                &bs.matmul_in_f16,
                &bs.proj_buf,
                dim as u32,
                nt,
                q_dim as u32,
                wo_dtype,
            );
            ax_engine_metal::barrier_buffers(encoder);
            metal_ops.elementwise.encode_buffer_copy(
                encoder,
                &bs.hidden,
                0,
                &bs.norm_buf,
                0,
                nt * dim as u32,
            );
            ax_engine_metal::barrier_buffers(encoder);
            metal_ops.elementwise.encode_elementwise_add_batch(
                encoder,
                &bs.norm_buf,
                &bs.proj_buf,
                dim as u32,
                nt,
            );
            Ok(())
        })?;

        if !defer_kv_readback {
            let k_after_rope = unsafe { &bs.k_buf.as_slice::<f32>()[..n_tokens * kv_dim] };
            let v_slice = unsafe { &bs.v_buf.as_slice::<f32>()[..n_tokens * kv_dim] };
            qwen_kv.attention_append_batch_cpu_mirror(layer, k_after_rope, v_slice, n_tokens);
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_unified_full_attention_batch_layer(
        metal_ops: &MetalOps,
        cfg: &ModelConfig,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        _weights: &WeightStore,
        allow_graph_ir_schedule: bool,
        cached_layer: &crate::backend::metal::CachedLayerKeys,
        moe_layer: Option<&Qwen3_5MoeResidentLayerKeys>,
        _prefix: &str,
        layer: usize,
        batch_position: usize,
        n_tokens: usize,
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        nt: u32,
        eps: f32,
        ops: Option<&mut OpBreakdown>,
        defer_kv_readback: bool,
    ) -> anyhow::Result<bool> {
        let layer_is_moe = cached_layer.wg == 0;
        if layer_is_moe {
            let Some(moe_layer) = moe_layer else {
                return Ok(false);
            };
            if !Self::build_qwen35_full_attention_hidden_before_moe_gpu(
                metal_ops,
                cfg,
                qwen_kv,
                cached_layer,
                layer,
                batch_position,
                n_tokens,
                dim,
                q_dim,
                kv_dim,
                n_heads,
                n_kv_heads,
                head_dim,
                nt,
                eps,
                defer_kv_readback,
            )? {
                return Ok(false);
            }
            return Self::run_qwen35_projected_moe_resident_tail_from_staged_hidden(
                metal_ops,
                cached_layer.ffn_norm,
                moe_layer,
                n_tokens,
                dim,
                eps,
            );
        }
        let wq_dtype = cached_layer.wq_dtype;
        let wk_dtype = cached_layer.wk_dtype;
        let wv_dtype = cached_layer.wv_dtype;
        let wo_dtype = cached_layer.wo_dtype;
        let gpu_batch_supported = |dt: crate::gguf::tensor::GgmlType, m: u32, k: u32| {
            Self::qwen35_batch_projection_supported(metal_ops, dt, m, n_tokens as u32, k)
        };
        if !gpu_batch_supported(wq_dtype, (q_dim * 2) as u32, dim as u32)
            || !gpu_batch_supported(wk_dtype, kv_dim as u32, dim as u32)
            || !gpu_batch_supported(wv_dtype, kv_dim as u32, dim as u32)
            || !gpu_batch_supported(wo_dtype, dim as u32, q_dim as u32)
        {
            return Ok(false);
        }
        let dense_ffn = if layer_is_moe {
            if moe_layer.is_none() {
                return Ok(false);
            }
            None
        } else {
            let wg_dtype = cached_layer.wg_dtype;
            let wu_dtype = cached_layer.wu_dtype;
            let wd_dtype = cached_layer.wd_dtype;
            if !gpu_batch_supported(wg_dtype, inter_dim as u32, dim as u32)
                || !gpu_batch_supported(wu_dtype, inter_dim as u32, dim as u32)
                || !gpu_batch_supported(wd_dtype, dim as u32, inter_dim as u32)
            {
                return Ok(false);
            }
            Some((
                cached_layer.wg,
                wg_dtype,
                cached_layer.wu,
                wu_dtype,
                cached_layer.wd,
                wd_dtype,
            ))
        };
        let q_norm_key = cached_layer.attn_q_norm;
        let k_norm_key = cached_layer.attn_k_norm;

        let (rope_start, rope_step) = cfg.rope_scaling.scaled_start_step(batch_position);
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };
        let weight_cache = metal_ops.lock_weight_cache();
        let nw_buf = weight_cache.get(&cached_layer.attn_norm).unwrap();
        let wq_buf = weight_cache.get(&cached_layer.wq).unwrap();
        let wk_buf = weight_cache.get(&cached_layer.wk).unwrap();
        let wv_buf = weight_cache.get(&cached_layer.wv).unwrap();
        let wo_buf = weight_cache.get(&cached_layer.wo).unwrap();
        let ffn_nw_buf = weight_cache.get(&cached_layer.ffn_norm).unwrap();
        let moe_weight_cache = layer_is_moe.then(|| metal_ops.lock_moe_weight_cache());
        let moe_scratch = layer_is_moe
            .then(|| crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs))
            .transpose()?;
        let (wg_buf, wg_dtype, wu_buf, wu_dtype, wd_buf, wd_dtype) =
            if let Some((wg_key, wg_dtype, wu_key, wu_dtype, wd_key, wd_dtype)) = dense_ffn {
                (
                    Some(weight_cache.get(&wg_key).unwrap()),
                    wg_dtype,
                    Some(weight_cache.get(&wu_key).unwrap()),
                    wu_dtype,
                    Some(weight_cache.get(&wd_key).unwrap()),
                    wd_dtype,
                )
            } else {
                (
                    None,
                    crate::gguf::tensor::GgmlType::F32,
                    None,
                    crate::gguf::tensor::GgmlType::F32,
                    None,
                    crate::gguf::tensor::GgmlType::F32,
                )
            };

        let cb_t = OpTimer::start();
        if !layer_is_moe
            && !prefill_schedule::try_execute_qwen35_full_attention_prefill_schedule(
                &metal_ops.device,
                metal_ops,
                allow_graph_ir_schedule,
                cfg,
                cached_layer,
                &weight_cache,
                bs,
                qwen_kv.gpu_attention(),
                layer,
                batch_position,
                n_tokens,
                q_dim,
                kv_dim,
                inter_dim,
                metal_ops.attention_dispatch_config(),
            )?
        {
            metal_ops.device.execute_sync(|encoder| {
                let qkv_uses_f16 = Self::qwen35_batch_projection_needs_f16_input(wq_dtype)
                    || Self::qwen35_batch_projection_needs_f16_input(wk_dtype)
                    || Self::qwen35_batch_projection_needs_f16_input(wv_dtype);
                let ffn_input_uses_f16 = !layer_is_moe
                    && (Self::qwen35_batch_projection_needs_f16_input(wg_dtype)
                        || Self::qwen35_batch_projection_needs_f16_input(wu_dtype));
                metal_ops.elementwise.encode_rms_norm_out_batch(
                    encoder,
                    &bs.hidden,
                    nw_buf,
                    &bs.norm_buf,
                    dim as u32,
                    nt,
                    eps,
                );
                ax_engine_metal::barrier_buffers(encoder);
                if qkv_uses_f16 {
                    metal_ops.elementwise.encode_cast_f32_to_f16(
                        encoder,
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        nt * dim as u32,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                }
                Self::encode_qwen35_batch_projection(
                    metal_ops,
                    encoder,
                    wq_buf,
                    &bs.norm_buf,
                    &bs.matmul_in_f16,
                    &bs.gate_buf,
                    (q_dim * 2) as u32,
                    nt,
                    dim as u32,
                    wq_dtype,
                );
                Self::encode_qwen35_batch_projection(
                    metal_ops,
                    encoder,
                    wk_buf,
                    &bs.norm_buf,
                    &bs.matmul_in_f16,
                    &bs.k_buf,
                    kv_dim as u32,
                    nt,
                    dim as u32,
                    wk_dtype,
                );
                Self::encode_qwen35_batch_projection(
                    metal_ops,
                    encoder,
                    wv_buf,
                    &bs.norm_buf,
                    &bs.matmul_in_f16,
                    &bs.v_buf,
                    kv_dim as u32,
                    nt,
                    dim as u32,
                    wv_dtype,
                );
                ax_engine_metal::barrier_buffers(encoder);
                metal_ops.elementwise.encode_split_qgate_batch(
                    encoder,
                    &bs.gate_buf,
                    &bs.q_buf,
                    &bs.up_buf,
                    nt,
                    q_dim as u32,
                    head_dim as u32,
                );
                ax_engine_metal::barrier_buffers(encoder);
                if let (Some(q_key), Some(k_key)) = (q_norm_key, k_norm_key) {
                    let q_nw = weight_cache.get(&q_key).unwrap();
                    let k_nw = weight_cache.get(&k_key).unwrap();
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder,
                        &bs.q_buf,
                        q_nw,
                        nt,
                        n_heads as u32,
                        head_dim as u32,
                        eps,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder,
                        &bs.k_buf,
                        k_nw,
                        nt,
                        n_kv_heads as u32,
                        head_dim as u32,
                        eps,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                }
                metal_ops.elementwise.encode_rope_batch_neox_partial(
                    encoder,
                    &bs.q_buf,
                    &bs.k_buf,
                    nt,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    (head_dim as u32).min(64),
                    rope_start,
                    rope_step,
                    cfg.rope_freq_base,
                );
                ax_engine_metal::barrier_buffers(encoder);
                if let Some(gpu_attn) = qwen_kv.gpu_attention() {
                    if gpu_attn.is_q8() {
                        let blocks_per_row = kv_dim / crate::kv::gpu_kv::Q8_0_BLOCK_VALUES;
                        let row_offset = (batch_position * blocks_per_row) as u32;
                        metal_ops.elementwise.encode_kv_append_batch_pair_q8(
                            encoder,
                            &bs.k_buf,
                            &bs.v_buf,
                            gpu_attn.k_buffer(layer),
                            gpu_attn.v_buffer(layer),
                            row_offset,
                            blocks_per_row as u32,
                            kv_dim as u32,
                            nt,
                        );
                    } else {
                        let cache_offset = (batch_position * kv_dim) as u32;
                        metal_ops.elementwise.encode_kv_append_batch_pair(
                            encoder,
                            &bs.k_buf,
                            &bs.v_buf,
                            gpu_attn.k_buffer(layer),
                            gpu_attn.v_buffer(layer),
                            gpu_attn.is_f16(),
                            cache_offset,
                            kv_dim as u32,
                            kv_dim as u32,
                            nt,
                        );
                    }
                    ax_engine_metal::barrier_buffers(encoder);
                }
                if batch_position == 0 {
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
                        metal_ops.attention_dispatch_config(),
                    );
                } else if let Some(gpu_attn) = qwen_kv.gpu_attention() {
                    metal_ops
                        .attention
                        .encode_attention_prefill_cached_with_config(
                            encoder,
                            &bs.q_buf,
                            gpu_attn.k_buffer(layer),
                            gpu_attn.v_buffer(layer),
                            &bs.attn_out,
                            gpu_attn.is_f16(),
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            batch_position as u32,
                            0,
                            metal_ops.attention_dispatch_config(),
                        );
                }
                ax_engine_metal::barrier_buffers(encoder);
                metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                    encoder,
                    &bs.up_buf,
                    &bs.attn_out,
                    (n_tokens * q_dim) as u32,
                );
                ax_engine_metal::barrier_buffers(encoder);
                if Self::qwen35_batch_projection_needs_f16_input(wo_dtype) {
                    metal_ops.elementwise.encode_cast_f32_to_f16(
                        encoder,
                        &bs.attn_out,
                        &bs.matmul_in_f16,
                        nt * q_dim as u32,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                }
                Self::encode_qwen35_batch_projection(
                    metal_ops,
                    encoder,
                    wo_buf,
                    &bs.attn_out,
                    &bs.matmul_in_f16,
                    &bs.proj_buf,
                    dim as u32,
                    nt,
                    q_dim as u32,
                    wo_dtype,
                );
                ax_engine_metal::barrier_buffers(encoder);
                if let Some(moe_layer) = moe_layer {
                    let moe_cache = moe_weight_cache.as_ref().unwrap();
                    let shared_expert = moe_layer.shared_expert.map(|shared| {
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
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                        encoder,
                        moe_scratch.expect("qwen35 batch MoE scratch view missing"),
                        &bs.hidden,
                        ffn_nw_buf,
                        moe_cache.get(&moe_layer.router).unwrap(),
                        moe_layer.router_dtype,
                        moe_cache.get(&moe_layer.gate).unwrap(),
                        moe_layer.gate_dtype,
                        moe_cache.get(&moe_layer.up).unwrap(),
                        moe_layer.up_dtype,
                        moe_cache.get(&moe_layer.down).unwrap(),
                        moe_layer.down_dtype,
                        n_tokens,
                        moe_layer.n_expert,
                        moe_layer.n_expert_used,
                        dim,
                        moe_layer.expert_inter_dim,
                        moe_layer.gate_stride,
                        moe_layer.up_stride,
                        moe_layer.down_stride,
                        eps,
                        shared_expert.as_ref(),
                        true,
                    )?;
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
                    if ffn_input_uses_f16 {
                        metal_ops.elementwise.encode_cast_f32_to_f16(
                            encoder,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            nt * dim as u32,
                        );
                    }
                    Self::encode_qwen35_batch_projection(
                        metal_ops,
                        encoder,
                        wg_buf.unwrap(),
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        &bs.gate_buf,
                        inter_dim as u32,
                        nt,
                        dim as u32,
                        wg_dtype,
                    );
                    Self::encode_qwen35_batch_projection(
                        metal_ops,
                        encoder,
                        wu_buf.unwrap(),
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        &bs.up_buf,
                        inter_dim as u32,
                        nt,
                        dim as u32,
                        wu_dtype,
                    );
                    metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                        encoder,
                        &bs.gate_buf,
                        &bs.up_buf,
                        inter_dim as u32,
                        nt,
                    );
                    if Self::qwen35_batch_projection_needs_f16_input(wd_dtype) {
                        metal_ops.elementwise.encode_cast_f32_to_f16(
                            encoder,
                            &bs.gate_buf,
                            &bs.matmul_in_f16,
                            nt * inter_dim as u32,
                        );
                    }
                    Self::encode_qwen35_batch_projection(
                        metal_ops,
                        encoder,
                        wd_buf.unwrap(),
                        &bs.gate_buf,
                        &bs.matmul_in_f16,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                        inter_dim as u32,
                        wd_dtype,
                    );
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                    );
                }
                Ok(())
            })?;
        }
        if let Some(ops) = ops {
            let elapsed = cb_t.elapsed();
            ops.gpu_execute += elapsed;
            ops.gpu_execute_layers += elapsed;
        }

        if !defer_kv_readback {
            let k_after_rope = unsafe { &bs.k_buf.as_slice::<f32>()[..n_tokens * kv_dim] };
            let v_slice = unsafe { &bs.v_buf.as_slice::<f32>()[..n_tokens * kv_dim] };
            qwen_kv.attention_append_batch_cpu_mirror(layer, k_after_rope, v_slice, n_tokens);
        }
        Ok(true)
    }

    /// Flush a deferred K/V CPU mirror readback from a completed full-attention layer.
    /// Must be called before the next full-attention layer overwrites `bs.k_buf`/`bs.v_buf`.
    fn flush_deferred_kv_readback(
        metal_ops: &MetalOps,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        layer: usize,
        n_tokens: usize,
        kv_dim: usize,
    ) {
        let bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_ref() else {
            return;
        };
        let k_after_rope = unsafe { &bs.k_buf.as_slice::<f32>()[..n_tokens * kv_dim] };
        let v_slice = unsafe { &bs.v_buf.as_slice::<f32>()[..n_tokens * kv_dim] };
        qwen_kv.attention_append_batch_cpu_mirror(layer, k_after_rope, v_slice, n_tokens);
    }

    #[allow(clippy::too_many_arguments)]
    fn write_unified_batch_outputs(
        metal_ops: &MetalOps,
        backend: &dyn Backend,
        weights: &WeightStore,
        logits: Option<&mut [f32]>,
        logits_all: Option<&mut Vec<f32>>,
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        eps: f32,
    ) -> anyhow::Result<bool> {
        match (logits, logits_all) {
            (Some(logits), None) => {
                let mut bs_guard = metal_ops.batch_scratches();
                let Some(bs) = bs_guard.as_mut() else {
                    return Ok(false);
                };
                let mut scratch = Self::lock_cpu_batch_fallback_scratch();
                let Qwen3_5CpuBatchFallbackScratch { hidden, .. } = &mut *scratch;
                hidden.resize(dim, 0.0);
                hidden.copy_from_slice(unsafe {
                    &bs.hidden.as_slice::<f32>()[((n_tokens - 1) * dim)..n_tokens * dim]
                });
                drop(bs_guard);
                Self::write_single_logits(
                    backend,
                    hidden.as_mut_slice(),
                    dim,
                    vocab_size,
                    eps,
                    weights,
                    logits,
                )?;
                Ok(true)
            }
            (None, Some(logits_all)) => {
                let mut bs_guard = metal_ops.batch_scratches();
                let Some(bs) = bs_guard.as_mut() else {
                    return Ok(false);
                };
                let mut scratch = Self::lock_cpu_batch_fallback_scratch();
                let Qwen3_5CpuBatchFallbackScratch {
                    hidden,
                    final_hidden,
                    ..
                } = &mut *scratch;
                hidden.resize(n_tokens * dim, 0.0);
                hidden.copy_from_slice(unsafe { &bs.hidden.as_slice::<f32>()[..n_tokens * dim] });
                drop(bs_guard);
                Self::write_all_batch_logits_with_scratch(
                    backend,
                    hidden.as_slice(),
                    n_tokens,
                    dim,
                    vocab_size,
                    eps,
                    weights,
                    logits_all,
                    final_hidden,
                )?;
                Ok(true)
            }
            _ => unreachable!("validated by caller"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_unified_recurrent_tail_batch_layer(
        metal_ops: &MetalOps,
        allow_graph_ir_schedule: bool,
        dims: Qwen3_5RecurrentDims,
        temp_qkv: &ax_engine_metal::MetalBuffer,
        temp_z: &ax_engine_metal::MetalBuffer,
        rec_out_alias: Option<&ax_engine_metal::MetalBuffer>,
        rec_z_alias: Option<&ax_engine_metal::MetalBuffer>,
        recurrent_output_in_batch_scratch: bool,
        recurrent_output_in_qkv_temp: bool,
        keep_rec_z_on_gpu: bool,
        ssm_gpu: bool,
        ssm_result: Option<&[f32]>,
        ssm_norm_key: Option<usize>,
        ssm_key: usize,
        ssm_out_dtype: crate::gguf::tensor::GgmlType,
        ffn_nw_key: usize,
        moe_layer: Option<&Qwen3_5MoeResidentLayerKeys>,
        wg_key: usize,
        wu_key: usize,
        wd_key: usize,
        wg_dtype: crate::gguf::tensor::GgmlType,
        wu_dtype: crate::gguf::tensor::GgmlType,
        wd_dtype: crate::gguf::tensor::GgmlType,
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        nt: u32,
        eps: f32,
        ops: Option<&mut OpBreakdown>,
        pipelined: bool,
    ) -> anyhow::Result<(bool, Option<ax_engine_metal::InflightFrame>)> {
        let active_total_inner = n_tokens * dims.inner_size;
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok((false, None));
        };

        if let Some(ssm_result) = ssm_result {
            unsafe {
                bs.attn_out.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(ssm_result);
            }
        }
        let weight_cache = metal_ops.lock_weight_cache();
        let ffn_nw_buf = weight_cache.get(&ffn_nw_key).unwrap();
        let moe_weight_cache = moe_layer.map(|_| metal_ops.lock_moe_weight_cache());
        let moe_scratch = moe_layer
            .map(|_| crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs))
            .transpose()?;
        let wg_buf = moe_layer
            .is_none()
            .then(|| weight_cache.get(&wg_key).unwrap());
        let wu_buf = moe_layer
            .is_none()
            .then(|| weight_cache.get(&wu_key).unwrap());
        let wd_buf = moe_layer
            .is_none()
            .then(|| weight_cache.get(&wd_key).unwrap());
        let cb_t = OpTimer::start();

        let recurrent_output_is_bhsk_qkv =
            recurrent_output_in_batch_scratch && recurrent_output_in_qkv_temp;
        let rec_out_buf = if recurrent_output_in_batch_scratch {
            if recurrent_output_is_bhsk_qkv {
                &bs.gate_buf
            } else {
                &bs.proj_buf
            }
        } else {
            rec_out_alias.expect("gpu recurrent output alias missing")
        };
        let rec_z_buf = if keep_rec_z_on_gpu {
            temp_z
        } else {
            rec_z_alias.expect("gpu recurrent gate alias missing")
        };
        let recurrent_projection = if ssm_gpu {
            Some(prefill_schedule::Qwen35RecurrentTailProjection {
                rec_out: rec_out_buf,
                rec_z: rec_z_buf,
                ssm_norm: weight_cache.get(&ssm_norm_key.unwrap()).unwrap(),
                ssm_weight: weight_cache.get(&ssm_key).unwrap(),
                ssm_dtype: ssm_out_dtype,
                time_step_rank: dims.time_step_rank,
                state_size: dims.state_size,
                inner_dim: dims.inner_size,
            })
        } else {
            None
        };
        // --- Pipelined path: try async graph-IR schedule first ----------------
        if moe_layer.is_none()
            && pipelined
            && let Some(frame) =
                prefill_schedule::try_execute_qwen35_recurrent_tail_prefill_schedule_async(
                    &metal_ops.device,
                    metal_ops,
                    allow_graph_ir_schedule,
                    bs,
                    recurrent_projection,
                    ffn_nw_buf,
                    wg_buf.expect("dense recurrent tail gate buffer missing"),
                    wg_dtype,
                    wu_buf.expect("dense recurrent tail up buffer missing"),
                    wu_dtype,
                    wd_buf.expect("dense recurrent tail down buffer missing"),
                    wd_dtype,
                    n_tokens,
                    dim,
                    inter_dim,
                    eps,
                )?
        {
            drop(weight_cache);
            drop(bs_guard);
            return Ok((true, Some(frame)));
        }

        // --- Try blocking graph-IR schedule (non-pipelined) ------------------
        let used_schedule = if moe_layer.is_none() && !pipelined {
            prefill_schedule::try_execute_qwen35_recurrent_tail_prefill_schedule(
                &metal_ops.device,
                metal_ops,
                allow_graph_ir_schedule,
                bs,
                recurrent_projection,
                ffn_nw_buf,
                wg_buf.expect("dense recurrent tail gate buffer missing"),
                wg_dtype,
                wu_buf.expect("dense recurrent tail up buffer missing"),
                wu_dtype,
                wd_buf.expect("dense recurrent tail down buffer missing"),
                wd_dtype,
                n_tokens,
                dim,
                inter_dim,
                eps,
            )?
        } else {
            false // pipelined path already tried graph-IR above
        };

        // --- Inline encoding closure (shared between sync and async) ---------
        if !used_schedule {
            let encode_inline = |encoder: &ax_engine_metal::MetalEncoder| -> anyhow::Result<()> {
                if ssm_gpu {
                    if recurrent_output_is_bhsk_qkv {
                        metal_ops.gdn.encode_unpack_bhsk_to_token_major(
                            encoder,
                            temp_qkv,
                            rec_out_buf,
                            nt,
                            dims.time_step_rank as u32,
                            dims.state_size as u32,
                        );
                    }
                    let ssm_norm_buf = weight_cache.get(&ssm_norm_key.unwrap()).unwrap();
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder,
                        rec_out_buf,
                        ssm_norm_buf,
                        nt,
                        dims.time_step_rank as u32,
                        dims.state_size as u32,
                        eps,
                    );
                    metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                        encoder,
                        rec_z_buf,
                        rec_out_buf,
                        dims.inner_size as u32,
                        nt,
                    );
                    if Self::qwen35_batch_projection_needs_f16_input(ssm_out_dtype) {
                        metal_ops.elementwise.encode_cast_f32_to_f16(
                            encoder,
                            rec_z_buf,
                            &bs.matmul_in_f16,
                            active_total_inner as u32,
                        );
                    }
                    let ssm_buf = weight_cache.get(&ssm_key).unwrap();
                    Self::encode_qwen35_batch_projection(
                        metal_ops,
                        encoder,
                        ssm_buf,
                        rec_z_buf,
                        &bs.matmul_in_f16,
                        &bs.attn_out,
                        dim as u32,
                        nt,
                        dims.inner_size as u32,
                        ssm_out_dtype,
                    );
                }
                if let Some(moe_layer) = moe_layer {
                    let moe_cache = moe_weight_cache.as_ref().unwrap();
                    let shared_expert = moe_layer.shared_expert.map(|shared| {
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
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.attn_out,
                        dim as u32,
                        nt,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                        encoder,
                        moe_scratch.expect("qwen35 recurrent MoE scratch view missing"),
                        &bs.hidden,
                        ffn_nw_buf,
                        moe_cache.get(&moe_layer.router).unwrap(),
                        moe_layer.router_dtype,
                        moe_cache.get(&moe_layer.gate).unwrap(),
                        moe_layer.gate_dtype,
                        moe_cache.get(&moe_layer.up).unwrap(),
                        moe_layer.up_dtype,
                        moe_cache.get(&moe_layer.down).unwrap(),
                        moe_layer.down_dtype,
                        n_tokens,
                        moe_layer.n_expert,
                        moe_layer.n_expert_used,
                        dim,
                        moe_layer.expert_inter_dim,
                        moe_layer.gate_stride,
                        moe_layer.up_stride,
                        moe_layer.down_stride,
                        eps,
                        shared_expert.as_ref(),
                        true,
                    )?;
                } else {
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.attn_out,
                        dim as u32,
                        nt,
                    );
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        ffn_nw_buf,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        eps,
                    );
                    if Self::qwen35_batch_projection_needs_f16_input(wg_dtype)
                        || Self::qwen35_batch_projection_needs_f16_input(wu_dtype)
                    {
                        metal_ops.elementwise.encode_cast_f32_to_f16(
                            encoder,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            nt * dim as u32,
                        );
                    }
                    Self::encode_qwen35_batch_projection(
                        metal_ops,
                        encoder,
                        wg_buf.unwrap(),
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        &bs.gate_buf,
                        inter_dim as u32,
                        nt,
                        dim as u32,
                        wg_dtype,
                    );
                    Self::encode_qwen35_batch_projection(
                        metal_ops,
                        encoder,
                        wu_buf.unwrap(),
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        &bs.up_buf,
                        inter_dim as u32,
                        nt,
                        dim as u32,
                        wu_dtype,
                    );
                    metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                        encoder,
                        &bs.gate_buf,
                        &bs.up_buf,
                        inter_dim as u32,
                        nt,
                    );
                    if Self::qwen35_batch_projection_needs_f16_input(wd_dtype) {
                        metal_ops.elementwise.encode_cast_f32_to_f16(
                            encoder,
                            &bs.gate_buf,
                            &bs.matmul_in_f16,
                            nt * inter_dim as u32,
                        );
                    }
                    Self::encode_qwen35_batch_projection(
                        metal_ops,
                        encoder,
                        wd_buf.unwrap(),
                        &bs.gate_buf,
                        &bs.matmul_in_f16,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                        inter_dim as u32,
                        wd_dtype,
                    );
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                    );
                }
                Ok(())
            };

            if pipelined {
                let frame = metal_ops.device.execute_async(encode_inline)?;
                drop(weight_cache);
                drop(bs_guard);
                return Ok((true, Some(frame)));
            } else {
                metal_ops.device.execute_sync(encode_inline)?;
            }
        }
        if let Some(ops) = ops {
            let elapsed = cb_t.elapsed();
            ops.gpu_execute += elapsed;
            ops.gpu_execute_layers += elapsed;
        }
        Ok((true, None))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_unified_recurrent_projection_phase(
        metal_ops: &MetalOps,
        backend: &dyn Backend,
        input_ops: Option<[QuantOp<'_>; 4]>,
        gpu_proj_indices: &[usize],
        cpu_proj_indices: &[usize],
        proj_keys: &[usize],
        proj_dtypes: &[crate::gguf::tensor::GgmlType],
        proj_dims: &[usize],
        temp_qkv: &ax_engine_metal::MetalBuffer,
        temp_z: &ax_engine_metal::MetalBuffer,
        temp_beta: &ax_engine_metal::MetalBuffer,
        temp_alpha: &ax_engine_metal::MetalBuffer,
        gpu_plan: Qwen35UnifiedRecurrentGpuPlan,
        rec_qkv_batch: &mut [f32],
        rec_z_batch: &mut [f32],
        rec_beta_batch: &mut [f32],
        rec_alpha_batch: &mut [f32],
        cpu_norm_scratch: &mut Vec<f32>,
        n_tokens: usize,
        dim: usize,
        nt: u32,
        eps: f32,
        nw_key: usize,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<bool> {
        {
            let mut bs_guard = metal_ops.batch_scratches();
            let Some(bs) = bs_guard.as_mut() else {
                return Ok(false);
            };

            let weight_cache = metal_ops.lock_weight_cache();
            let nw_buf = weight_cache.get(&nw_key).unwrap();

            let cb_t = OpTimer::start();
            metal_ops.device.execute_sync(|encoder| {
                metal_ops.elementwise.encode_rms_norm_out_batch(
                    encoder,
                    &bs.hidden,
                    nw_buf,
                    &bs.norm_buf,
                    dim as u32,
                    nt,
                    eps,
                );
                if gpu_proj_indices
                    .iter()
                    .any(|&i| Self::qwen35_batch_projection_needs_f16_input(proj_dtypes[i]))
                {
                    metal_ops.elementwise.encode_cast_f32_to_f16(
                        encoder,
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        nt * dim as u32,
                    );
                }
                for &i in gpu_proj_indices {
                    let w_buf = weight_cache.get(&proj_keys[i]).unwrap();
                    let out_buf = match i {
                        0 => temp_qkv,
                        1 => temp_z,
                        2 => temp_beta,
                        3 => temp_alpha,
                        _ => unreachable!(),
                    };
                    Self::encode_qwen35_batch_projection(
                        metal_ops,
                        encoder,
                        w_buf,
                        &bs.norm_buf,
                        &bs.matmul_in_f16,
                        out_buf,
                        proj_dims[i] as u32,
                        nt,
                        dim as u32,
                        proj_dtypes[i],
                    );
                }
                Ok(())
            })?;
            if let Some(ref mut ops) = ops {
                let elapsed = cb_t.elapsed();
                ops.gpu_execute += elapsed;
                ops.gpu_execute_layers += elapsed;
            }

            let readback_t = OpTimer::start();
            unsafe {
                for &i in gpu_proj_indices {
                    let out_slice = match i {
                        0 => &temp_qkv.as_slice::<f32>()[..n_tokens * proj_dims[i]],
                        1 => &temp_z.as_slice::<f32>()[..n_tokens * proj_dims[i]],
                        2 => &temp_beta.as_slice::<f32>()[..n_tokens * proj_dims[i]],
                        3 => &temp_alpha.as_slice::<f32>()[..n_tokens * proj_dims[i]],
                        _ => unreachable!(),
                    };
                    if Self::qwen35_should_readback_recurrent_projection(i, gpu_plan) {
                        match i {
                            0 => rec_qkv_batch[..out_slice.len()].copy_from_slice(out_slice),
                            1 => rec_z_batch[..out_slice.len()].copy_from_slice(out_slice),
                            2 => rec_beta_batch[..out_slice.len()].copy_from_slice(out_slice),
                            3 => rec_alpha_batch[..out_slice.len()].copy_from_slice(out_slice),
                            _ => unreachable!(),
                        }
                    }
                }
                if !cpu_proj_indices.is_empty() {
                    cpu_norm_scratch.resize(n_tokens * dim, 0.0);
                    cpu_norm_scratch
                        .as_mut_slice()
                        .copy_from_slice(&bs.norm_buf.as_slice::<f32>()[..n_tokens * dim]);
                }
            }
            if let Some(ref mut ops) = ops {
                ops.gpu_readback += readback_t.elapsed();
            }
        }

        for &i in cpu_proj_indices {
            let input_ops = input_ops
                .as_ref()
                .expect("qwen35 recurrent CPU projection inputs missing");
            let (raw, dtype, out_dim) = input_ops[i];
            let dst = match i {
                0 => &mut rec_qkv_batch[..n_tokens * out_dim],
                1 => &mut rec_z_batch[..n_tokens * out_dim],
                2 => &mut rec_beta_batch[..n_tokens * out_dim],
                3 => &mut rec_alpha_batch[..n_tokens * out_dim],
                _ => unreachable!(),
            };
            timed_matmul_bucket!(
                ops,
                matmul_input_proj,
                Self::batched_dequant_matmul_token_major(
                    backend,
                    raw,
                    dtype,
                    cpu_norm_scratch.as_slice(),
                    dst,
                    n_tokens,
                    out_dim,
                    dim,
                )
            );
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_unified_recurrent_runtime_phase(
        metal_ops: &MetalOps,
        backend: &dyn Backend,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        layer: usize,
        recurrent_slot: usize,
        recurrent_slot_indices: &[usize],
        recurrent_keys: &Qwen3_5RecurrentLayerKeys,
        temp_qkv: &ax_engine_metal::MetalBuffer,
        recurrent_beta_gpu: Option<&ax_engine_metal::MetalBuffer>,
        recurrent_alpha_gpu: Option<&ax_engine_metal::MetalBuffer>,
        rec_qkv_batch: &mut [f32],
        rec_beta_batch: &mut [f32],
        rec_alpha_batch: &mut [f32],
        rec_out_batch: &mut [f32],
        n_tokens: usize,
        dims: Qwen3_5RecurrentDims,
        eps: f32,
        allow_gpu_qkv_handoff: bool,
        qkv_gpu_fast_path_enabled: bool,
        keep_recurrent_output_on_gpu: bool,
        force_backend_state_batch: bool,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<(bool, bool)> {
        let mut recurrent_output_in_batch_scratch = false;
        let mut recurrent_output_in_qkv_temp = false;

        let used_gpu_qkv_handoff = if !allow_gpu_qkv_handoff || force_backend_state_batch {
            None
        } else {
            timed!(
                ops,
                recurrent,
                Self::try_run_recurrent_batch_from_gpu_qkv_single_slot(
                    metal_ops,
                    qwen_kv,
                    layer,
                    recurrent_slot,
                    temp_qkv,
                    recurrent_beta_gpu,
                    recurrent_alpha_gpu,
                    recurrent_keys,
                    rec_beta_batch,
                    rec_alpha_batch,
                    rec_out_batch,
                    n_tokens,
                    dims,
                    eps,
                    keep_recurrent_output_on_gpu,
                    false,
                    false,
                )
            )?
        };
        let used_forced_backend_native_gpu_handoff =
            if allow_gpu_qkv_handoff && force_backend_state_batch {
                timed!(
                    ops,
                    recurrent,
                    Self::try_run_recurrent_batch_from_gpu_qkv_single_slot(
                        metal_ops,
                        qwen_kv,
                        layer,
                        recurrent_slot,
                        temp_qkv,
                        recurrent_beta_gpu,
                        recurrent_alpha_gpu,
                        recurrent_keys,
                        rec_beta_batch,
                        rec_alpha_batch,
                        rec_out_batch,
                        n_tokens,
                        dims,
                        eps,
                        keep_recurrent_output_on_gpu,
                        true,
                        true,
                    )
                )?
            } else {
                None
            };

        if let Some(stats) = used_gpu_qkv_handoff.or(used_forced_backend_native_gpu_handoff) {
            metal_ops.record_qwen35_recurrent_batch_qkv_handoff();
            recurrent_output_in_batch_scratch = keep_recurrent_output_on_gpu;
            recurrent_output_in_qkv_temp = keep_recurrent_output_on_gpu;
            if let Some(ref mut ops) = ops {
                ops.gpu_execute += stats.gpu_execute;
                ops.gpu_execute_layers += stats.gpu_execute;
                ops.gpu_readback += stats.gpu_readback;
            }
        } else {
            if qkv_gpu_fast_path_enabled {
                let readback_t = OpTimer::start();
                unsafe {
                    let qkv_slice = &temp_qkv.as_slice::<f32>()[..n_tokens * dims.conv_dim()];
                    rec_qkv_batch[..qkv_slice.len()].copy_from_slice(qkv_slice);
                }
                if let Some(ref mut ops) = ops {
                    ops.gpu_readback += readback_t.elapsed();
                }
            }
            rec_out_batch.fill(0.0);
            let weight_cache = metal_ops.lock_weight_cache();
            let dt_bias = unsafe {
                weight_cache
                    .get(&recurrent_keys.dt_bias)
                    .expect("qwen35 dt_bias buffer missing after cache")
                    .as_slice::<f32>()
            };
            let ssm_a = unsafe {
                weight_cache
                    .get(&recurrent_keys.ssm_a)
                    .expect("qwen35 ssm_a buffer missing after cache")
                    .as_slice::<f32>()
            };
            let conv_kernel = unsafe {
                weight_cache
                    .get(&recurrent_keys.conv_kernel)
                    .expect("qwen35 conv kernel buffer missing after cache")
                    .as_slice::<f32>()
            };
            timed!(
                ops,
                recurrent,
                backend.qwen35_recurrent_sequence_for_kv(
                    rec_qkv_batch,
                    rec_beta_batch,
                    rec_alpha_batch,
                    dt_bias,
                    ssm_a,
                    conv_kernel,
                    qwen_kv,
                    layer,
                    recurrent_slot_indices,
                    rec_out_batch,
                    n_tokens,
                    Self::qwen35_recurrent_config(qwen_kv, dims, eps),
                )
            );
            drop(weight_cache);
        }

        Ok((
            recurrent_output_in_batch_scratch,
            recurrent_output_in_qkv_temp,
        ))
    }

    fn with_qwen35_batch_hidden_slice_mut<R>(
        metal_ops: &MetalOps,
        n_tokens: usize,
        dim: usize,
        f: impl FnOnce(&mut [f32]) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let hidden_ptr = {
            let bs_guard = metal_ops.batch_scratches();
            let Some(bs) = bs_guard.as_ref() else {
                anyhow::bail!("missing batch scratches")
            };
            bs.hidden.contents().as_ptr() as *mut f32
        };
        let hidden = unsafe { std::slice::from_raw_parts_mut(hidden_ptr, n_tokens * dim) };
        f(hidden)
    }

    fn qwen35_should_use_unified_recurrent_runtime(unified_recurrent_enabled: bool) -> bool {
        unified_recurrent_enabled
    }

    fn qwen35_should_allow_gpu_qkv_runtime_handoff(layer_is_moe: bool) -> bool {
        let _ = layer_is_moe;
        true
    }

    fn qwen35_should_enable_gpu_qkv_fast_path(
        check: Option<Qwen35RecurrentQkvFastPathCheck>,
    ) -> bool {
        env_flag_override("AX_QWEN35_GPU_QKV_FAST_PATH").unwrap_or_else(|| {
            check
                .map(Qwen35RecurrentQkvFastPathCheck::is_eligible)
                .unwrap_or(false)
        })
    }

    fn qwen35_should_try_merged_projection_fused_recurrent_layer(layer_is_moe: bool) -> bool {
        if layer_is_moe {
            return env_flag_override("AX_QWEN35_MOE_MERGED_FUSED_RECURRENT").unwrap_or(true);
        }
        // Default-off until the dense merged-projection fused recurrent path
        // has full real-model parity. `AX_QWEN35_MERGED_FUSED_RECURRENT=1`
        // remains available for focused debugging.
        env_flag_override("AX_QWEN35_MERGED_FUSED_RECURRENT").unwrap_or(false)
    }

    fn qwen35_should_use_unified_recurrent_moe_gpu_tail() -> bool {
        env_flag_override("AX_QWEN35_UNIFIED_RECURRENT_MOE_GPU_TAIL").unwrap_or(false)
    }

    fn qwen35_should_try_resident_moe_gpu_tail(layer_is_moe: bool, ssm_gpu: bool) -> bool {
        layer_is_moe
            && ssm_gpu
            && env_flag_override("AX_QWEN35_RESIDENT_MOE_GPU_TAIL").unwrap_or(true)
    }

    fn qwen35_should_try_projected_moe_gpu_tail(n_tokens: usize) -> bool {
        if n_tokens <= 1 {
            return false;
        }
        env_flag_override("AX_QWEN35_PROJECTED_MOE_GPU_TAIL").unwrap_or(true)
    }

    fn build_qwen35_projected_hidden_before_moe_gpu(
        metal_ops: &MetalOps,
        n_tokens: usize,
        dim: usize,
        proj_buf: &[f32],
    ) -> anyhow::Result<bool> {
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };
        unsafe {
            bs.proj_buf.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(proj_buf);
        }
        metal_ops.device.execute_sync(|encoder| {
            metal_ops.elementwise.encode_buffer_copy(
                encoder,
                &bs.hidden,
                0,
                &bs.norm_buf,
                0,
                (n_tokens * dim) as u32,
            );
            ax_engine_metal::barrier_buffers(encoder);
            metal_ops.elementwise.encode_elementwise_add_batch(
                encoder,
                &bs.norm_buf,
                &bs.proj_buf,
                dim as u32,
                n_tokens as u32,
            );
            Ok(())
        })?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_qwen35_projected_moe_resident_tail_from_staged_hidden(
        metal_ops: &MetalOps,
        ffn_nw_key: usize,
        moe_layer: &Qwen3_5MoeResidentLayerKeys,
        n_tokens: usize,
        dim: usize,
        eps: f32,
    ) -> anyhow::Result<bool> {
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };
        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        let ffn_nw_buf = weight_cache
            .get(&ffn_nw_key)
            .expect("qwen35 recurrent projected MoE tail missing ffn_norm");
        let moe_scratch = crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs)?;
        let shared_expert = moe_layer.shared_expert.map(|shared| {
            crate::backend::metal::SharedExpertCachedBuffers {
                gate: moe_weight_cache.get(&shared.gate).unwrap(),
                up: moe_weight_cache.get(&shared.up).unwrap(),
                down: moe_weight_cache.get(&shared.down).unwrap(),
                gate_inp: shared
                    .gate_inp
                    .map(|gate_inp| moe_weight_cache.get(&gate_inp).unwrap()),
                gate_inp_dtype: shared.gate_inp_dtype,
                dtype: shared.dtype,
                inter_dim: shared.inter_dim,
                gate_inp_rows: shared.gate_inp_rows,
            }
        });
        metal_ops.device.execute_sync(|encoder| {
            metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                encoder,
                moe_scratch,
                &bs.norm_buf,
                ffn_nw_buf,
                moe_weight_cache.get(&moe_layer.router).unwrap(),
                moe_layer.router_dtype,
                moe_weight_cache.get(&moe_layer.gate).unwrap(),
                moe_layer.gate_dtype,
                moe_weight_cache.get(&moe_layer.up).unwrap(),
                moe_layer.up_dtype,
                moe_weight_cache.get(&moe_layer.down).unwrap(),
                moe_layer.down_dtype,
                n_tokens,
                moe_layer.n_expert,
                moe_layer.n_expert_used,
                dim,
                moe_layer.expert_inter_dim,
                moe_layer.gate_stride,
                moe_layer.up_stride,
                moe_layer.down_stride,
                eps,
                shared_expert.as_ref(),
                false,
            )?;
            ax_engine_metal::barrier_buffers(encoder);
            metal_ops.elementwise.encode_buffer_copy(
                encoder,
                &bs.norm_buf,
                0,
                &bs.hidden,
                0,
                (n_tokens * dim) as u32,
            );
            Ok(())
        })?;
        Ok(true)
    }

    fn qwen35_should_use_unified_recurrent_tail(
        unified_recurrent_enabled: bool,
        layer_is_moe: bool,
    ) -> bool {
        unified_recurrent_enabled
            && (!layer_is_moe || Self::qwen35_should_use_unified_recurrent_moe_gpu_tail())
    }

    fn qwen35_should_keep_recurrent_output_on_gpu_for_unified_tail(
        unified_recurrent_enabled: bool,
        layer_is_moe: bool,
        ssm_gpu: bool,
    ) -> bool {
        Self::qwen35_should_use_unified_recurrent_tail(unified_recurrent_enabled, layer_is_moe)
            && ssm_gpu
    }

    #[allow(clippy::too_many_arguments)]
    fn try_run_qwen35_recurrent_moe_gpu_tail_after_handoff(
        metal_ops: &MetalOps,
        dims: Qwen3_5RecurrentDims,
        temp_qkv: &ax_engine_metal::MetalBuffer,
        temp_z: &ax_engine_metal::MetalBuffer,
        recurrent_output_in_batch_scratch: bool,
        recurrent_output_in_qkv_temp: bool,
        keep_rec_z_on_gpu: bool,
        ssm_norm_key: usize,
        ssm_key: usize,
        ssm_out_dtype: crate::gguf::tensor::GgmlType,
        ffn_nw_key: usize,
        moe_layer: &Qwen3_5MoeResidentLayerKeys,
        n_tokens: usize,
        dim: usize,
        nt: u32,
        eps: f32,
    ) -> anyhow::Result<bool> {
        if !recurrent_output_in_batch_scratch || !recurrent_output_in_qkv_temp || !keep_rec_z_on_gpu
        {
            return Ok(false);
        }

        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        let ssm_norm_buf = weight_cache.get(&ssm_norm_key).unwrap();
        let ssm_buf = weight_cache.get(&ssm_key).unwrap();
        let ffn_nw_buf = weight_cache.get(&ffn_nw_key).unwrap();
        let moe_scratch = crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs)?;
        let shared_expert = moe_layer.shared_expert.map(|shared| {
            crate::backend::metal::SharedExpertCachedBuffers {
                gate: moe_weight_cache.get(&shared.gate).unwrap(),
                up: moe_weight_cache.get(&shared.up).unwrap(),
                down: moe_weight_cache.get(&shared.down).unwrap(),
                gate_inp: shared
                    .gate_inp
                    .map(|gate_inp| moe_weight_cache.get(&gate_inp).unwrap()),
                gate_inp_dtype: shared.gate_inp_dtype,
                dtype: shared.dtype,
                inter_dim: shared.inter_dim,
                gate_inp_rows: shared.gate_inp_rows,
            }
        });

        metal_ops.device.execute_sync(|encoder| {
            metal_ops.gdn.encode_unpack_bhsk_to_token_major(
                encoder,
                temp_qkv,
                &bs.gate_buf,
                nt,
                dims.time_step_rank as u32,
                dims.state_size as u32,
            );
            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                encoder,
                &bs.gate_buf,
                ssm_norm_buf,
                nt,
                dims.time_step_rank as u32,
                dims.state_size as u32,
                eps,
            );
            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                encoder,
                temp_z,
                &bs.gate_buf,
                dims.inner_size as u32,
                nt,
            );
            if Self::qwen35_batch_projection_needs_f16_input(ssm_out_dtype) {
                metal_ops.elementwise.encode_cast_f32_to_f16(
                    encoder,
                    temp_z,
                    &bs.matmul_in_f16,
                    (n_tokens * dims.inner_size) as u32,
                );
            }
            Self::encode_qwen35_batch_projection(
                metal_ops,
                encoder,
                ssm_buf,
                temp_z,
                &bs.matmul_in_f16,
                &bs.attn_out,
                dim as u32,
                nt,
                dims.inner_size as u32,
                ssm_out_dtype,
            );
            metal_ops.elementwise.encode_elementwise_add_batch(
                encoder,
                &bs.hidden,
                &bs.attn_out,
                dim as u32,
                nt,
            );
            metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                encoder,
                moe_scratch,
                &bs.hidden,
                ffn_nw_buf,
                moe_weight_cache.get(&moe_layer.router).unwrap(),
                moe_layer.router_dtype,
                moe_weight_cache.get(&moe_layer.gate).unwrap(),
                moe_layer.gate_dtype,
                moe_weight_cache.get(&moe_layer.up).unwrap(),
                moe_layer.up_dtype,
                moe_weight_cache.get(&moe_layer.down).unwrap(),
                moe_layer.down_dtype,
                n_tokens,
                moe_layer.n_expert,
                moe_layer.n_expert_used,
                dim,
                moe_layer.expert_inter_dim,
                moe_layer.gate_stride,
                moe_layer.up_stride,
                moe_layer.down_stride,
                eps,
                shared_expert.as_ref(),
                false,
            )
        })?;

        drop(moe_weight_cache);
        drop(weight_cache);
        drop(bs_guard);
        Ok(true)
    }

    fn qwen35_build_unified_recurrent_gpu_plan(
        recurrent_slot_count: usize,
        gpu_proj_indices: &[usize],
        cpu_proj_indices: &[usize],
        qkv_gpu_fast_path_enabled: bool,
        ssm_gpu: bool,
    ) -> Qwen35UnifiedRecurrentGpuPlan {
        let keep_rec_z_on_gpu =
            recurrent_slot_count == 1 && ssm_gpu && gpu_proj_indices.contains(&1);
        let fused_gpu_recurrent_layer_uses_gpu_alpha_beta =
            gpu_proj_indices.contains(&2) && gpu_proj_indices.contains(&3);
        let fused_gpu_recurrent_layer_candidate = Self::qwen35_fused_recurrent_gpu_candidate(
            recurrent_slot_count,
            qkv_gpu_fast_path_enabled,
            cpu_proj_indices,
            keep_rec_z_on_gpu,
            ssm_gpu,
        );
        Qwen35UnifiedRecurrentGpuPlan {
            qkv_gpu_fast_path_enabled,
            keep_rec_z_on_gpu,
            fused_gpu_recurrent_layer_candidate,
            fused_gpu_recurrent_layer_uses_gpu_alpha_beta,
        }
    }

    fn qwen35_should_readback_recurrent_projection(
        projection_index: usize,
        gpu_plan: Qwen35UnifiedRecurrentGpuPlan,
    ) -> bool {
        match projection_index {
            0 => !gpu_plan.qkv_gpu_fast_path_enabled,
            1 => !gpu_plan.keep_rec_z_on_gpu || !gpu_plan.fused_gpu_recurrent_layer_candidate,
            2 | 3 => {
                !(gpu_plan.fused_gpu_recurrent_layer_candidate
                    && gpu_plan.fused_gpu_recurrent_layer_uses_gpu_alpha_beta)
            }
            _ => false,
        }
    }

    fn qwen35_build_unified_recurrent_tail_buffer_plan(
        unified_recurrent_enabled: bool,
        layer_is_moe: bool,
        ssm_gpu: bool,
        recurrent_output_in_batch_scratch: bool,
        keep_rec_z_on_gpu: bool,
    ) -> Qwen35UnifiedRecurrentTailBufferPlan {
        let use_unified_tail =
            Self::qwen35_should_use_unified_recurrent_tail(unified_recurrent_enabled, layer_is_moe);
        Qwen35UnifiedRecurrentTailBufferPlan {
            use_unified_tail,
            alias_rec_out: use_unified_tail && ssm_gpu && !recurrent_output_in_batch_scratch,
            alias_rec_z: use_unified_tail && ssm_gpu && !keep_rec_z_on_gpu,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_qwen35_recurrent_layer_via_legacy_batch_after_projection(
        metal_ops: &MetalOps,
        backend: &dyn Backend,
        cfg: &ModelConfig,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        weights: &WeightStore,
        prefix: &str,
        layer: usize,
        batch_position: usize,
        recurrent_slot: usize,
        recurrent_slot_indices: &[usize],
        dims: Qwen3_5RecurrentDims,
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        eps: f32,
        norm_buf: &mut [f32],
        rec_qkv_batch: &mut [f32],
        rec_z_batch: &mut [f32],
        rec_beta_batch: &mut [f32],
        rec_alpha_batch: &mut [f32],
        rec_out_batch: &mut [f32],
        proj_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
    ) -> anyhow::Result<bool> {
        proj_buf.fill(0.0);
        gate_buf.fill(0.0);
        up_buf.fill(0.0);
        down_buf.fill(0.0);
        Self::run_recurrent_batch_layer(
            cfg,
            backend,
            weights,
            prefix,
            qwen_kv,
            recurrent_slot,
            layer,
            batch_position,
            dims,
            recurrent_slot_indices,
            norm_buf,
            rec_qkv_batch,
            rec_z_batch,
            rec_beta_batch,
            rec_alpha_batch,
            rec_out_batch,
            proj_buf,
            n_tokens,
            dim,
            true,
        )?;
        Self::with_qwen35_batch_hidden_slice_mut(metal_ops, n_tokens, dim, |hidden| {
            Self::apply_layer_tail_batch(
                cfg,
                backend,
                weights,
                prefix,
                hidden,
                proj_buf,
                norm_buf,
                gate_buf,
                up_buf,
                down_buf,
                n_tokens,
                dim,
                inter_dim,
                eps,
                layer,
                batch_position,
            )
        })?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn project_qwen35_recurrent_output_batch(
        backend: &dyn Backend,
        weights: &WeightStore,
        prefix: &str,
        recurrent_slot_indices: &[usize],
        recurrent_slot: usize,
        dims: Qwen3_5RecurrentDims,
        n_tokens: usize,
        dim: usize,
        ssm_out_dtype: crate::gguf::tensor::GgmlType,
        rec_out_batch: &[f32],
        proj_buf: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let active_rec_out = Self::slot_batch_slice(
            rec_out_batch,
            recurrent_slot_indices,
            recurrent_slot,
            n_tokens,
            dims.inner_size,
        );
        let (ssm_out_raw, _, _) = Self::recurrent_output_op(weights, prefix, dim)?;
        proj_buf.fill(0.0);
        timed_matmul_bucket!(
            ops,
            matmul_output_proj,
            Self::batched_dequant_matmul_token_major(
                backend,
                ssm_out_raw,
                ssm_out_dtype,
                active_rec_out,
                proj_buf,
                n_tokens,
                dim,
                dims.inner_size,
            )
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_run_qwen35_recurrent_projected_moe_gpu_tail(
        metal_ops: &MetalOps,
        ffn_nw_key: usize,
        moe_layer: &Qwen3_5MoeResidentLayerKeys,
        n_tokens: usize,
        dim: usize,
        eps: f32,
        proj_buf: &mut [f32],
    ) -> anyhow::Result<bool> {
        if !Self::qwen35_should_try_projected_moe_gpu_tail(n_tokens) {
            return Ok(false);
        }
        Self::build_qwen35_projected_hidden_before_moe_gpu(metal_ops, n_tokens, dim, proj_buf)?;
        Self::run_qwen35_projected_moe_resident_tail_from_staged_hidden(
            metal_ops, ffn_nw_key, moe_layer, n_tokens, dim, eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_qwen35_recurrent_moe_legacy_tail(
        metal_ops: &MetalOps,
        backend: &dyn Backend,
        cfg: &ModelConfig,
        weights: &WeightStore,
        prefix: &str,
        layer: usize,
        batch_position: usize,
        recurrent_slot_indices: &[usize],
        recurrent_slot: usize,
        dims: Qwen3_5RecurrentDims,
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        eps: f32,
        ssm_out_dtype: crate::gguf::tensor::GgmlType,
        rec_out_batch: &[f32],
        norm_buf: &mut [f32],
        proj_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<bool> {
        Self::project_qwen35_recurrent_output_batch(
            backend,
            weights,
            prefix,
            recurrent_slot_indices,
            recurrent_slot,
            dims,
            n_tokens,
            dim,
            ssm_out_dtype,
            rec_out_batch,
            proj_buf,
            ops,
        )?;
        Self::with_qwen35_batch_hidden_slice_mut(metal_ops, n_tokens, dim, |hidden| {
            Self::apply_layer_tail_batch(
                cfg,
                backend,
                weights,
                prefix,
                hidden,
                proj_buf,
                norm_buf,
                gate_buf,
                up_buf,
                down_buf,
                n_tokens,
                dim,
                inter_dim,
                eps,
                layer,
                batch_position,
            )
        })?;
        Ok(true)
    }

    fn qwen35_fused_recurrent_gpu_candidate(
        recurrent_slot_count: usize,
        qkv_gpu_fast_path_enabled: bool,
        cpu_proj_indices: &[usize],
        keep_rec_z_on_gpu: bool,
        ssm_gpu: bool,
    ) -> bool {
        let qkv_stays_gpu_resident = qkv_gpu_fast_path_enabled
            || cpu_proj_indices.iter().all(|&i| i == 2 || i == 3);
        recurrent_slot_count == 1
            && qkv_stays_gpu_resident
            && cpu_proj_indices.iter().all(|&i| i == 2 || i == 3)
            && keep_rec_z_on_gpu
            && ssm_gpu
    }

    #[allow(clippy::too_many_arguments)]
    fn run_unified_recurrent_batch_layer(
        metal_ops: &MetalOps,
        backend: &dyn Backend,
        cfg: &ModelConfig,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        weights: &WeightStore,
        has_projection_phase: bool,
        has_runtime_phase: bool,
        has_tail_graph_ir_schedule: bool,
        cached_layer: &crate::backend::metal::CachedLayerKeys,
        recurrent_keys: &Qwen3_5RecurrentLayerKeys,
        moe_layer: Option<&Qwen3_5MoeResidentLayerKeys>,
        prefix: &str,
        layer: usize,
        batch_position: usize,
        recurrent_slot: usize,
        recurrent_slot_indices: &[usize],
        recurrent_slot_count: usize,
        dims: Qwen3_5RecurrentDims,
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        nt: u32,
        eps: f32,
        force_backend_state_batch: bool,
        mut ops: Option<&mut OpBreakdown>,
        pipelined: bool,
    ) -> anyhow::Result<(Option<&'static str>, Option<ax_engine_metal::InflightFrame>)> {
        let unified_recurrent_enabled = Self::unified_recurrent_enabled();

        if !has_projection_phase {
            return Ok((Some("unified_recurrent_projection_phase_disabled"), None));
        }
        if !has_runtime_phase {
            return Ok((Some("unified_recurrent_runtime_phase_disabled"), None));
        }
        let layer_is_moe = cached_layer.wg == 0;
        let nw_key = cached_layer.attn_norm;
        let ffn_nw_key = cached_layer.ffn_norm;

        if layer_is_moe && moe_layer.is_none() {
            return Ok((Some("qwen35moe_recurrent_ffn_uses_cpu_path"), None));
        }

        let gpu_batch_supported = |dt: crate::gguf::tensor::GgmlType, m: u32, k: u32| {
            Self::qwen35_batch_projection_supported(metal_ops, dt, m, n_tokens as u32, k)
        };
        let (wg_key, wg_dtype, wu_key, wu_dtype, wd_key, wd_dtype) = if layer_is_moe {
            (
                0usize,
                crate::gguf::tensor::GgmlType::F32,
                0usize,
                crate::gguf::tensor::GgmlType::F32,
                0usize,
                crate::gguf::tensor::GgmlType::F32,
            )
        } else {
            let wg_dtype = cached_layer.wg_dtype;
            let wu_dtype = cached_layer.wu_dtype;
            let wd_dtype = cached_layer.wd_dtype;
            if !gpu_batch_supported(wg_dtype, inter_dim as u32, dim as u32)
                || !gpu_batch_supported(wu_dtype, inter_dim as u32, dim as u32)
                || !gpu_batch_supported(wd_dtype, dim as u32, inter_dim as u32)
            {
                return Ok((Some("unsupported_recurrent_ffn_projection_dtype"), None));
            }
            (
                cached_layer.wg,
                wg_dtype,
                cached_layer.wu,
                wu_dtype,
                cached_layer.wd,
                wd_dtype,
            )
        };

        let recurrent_total_tokens = n_tokens * recurrent_slot_count;
        let mut cpu_scratch = Self::lock_cpu_batch_fallback_scratch();
        cpu_scratch
            .rec_qkv_batch
            .resize(recurrent_total_tokens * dims.conv_dim(), 0.0);
        cpu_scratch
            .rec_z_batch
            .resize(recurrent_total_tokens * dims.inner_size, 0.0);
        cpu_scratch
            .rec_beta_batch
            .resize(recurrent_total_tokens * dims.time_step_rank, 0.0);
        cpu_scratch
            .rec_alpha_batch
            .resize(recurrent_total_tokens * dims.time_step_rank, 0.0);
        cpu_scratch
            .rec_out_batch
            .resize(recurrent_total_tokens * dims.inner_size, 0.0);
        cpu_scratch.norm_buf.resize(n_tokens * dim, 0.0);
        cpu_scratch.proj_buf.resize(n_tokens * dim, 0.0);
        cpu_scratch.gate_buf.resize(n_tokens * inter_dim, 0.0);
        cpu_scratch.up_buf.resize(n_tokens * inter_dim, 0.0);
        cpu_scratch.down_buf.resize(n_tokens * dim, 0.0);
        cpu_scratch.norm_buf.fill(0.0);
        cpu_scratch.rec_qkv_batch.fill(0.0);
        cpu_scratch.rec_z_batch.fill(0.0);
        cpu_scratch.rec_beta_batch.fill(0.0);
        cpu_scratch.rec_alpha_batch.fill(0.0);
        cpu_scratch.rec_out_batch.fill(0.0);
        cpu_scratch.proj_buf.fill(0.0);
        cpu_scratch.gate_buf.fill(0.0);
        cpu_scratch.up_buf.fill(0.0);
        cpu_scratch.down_buf.fill(0.0);
        let Qwen3_5CpuBatchFallbackScratch {
            norm_buf,
            proj_buf,
            gate_buf,
            up_buf,
            rec_qkv_batch,
            rec_z_batch,
            rec_beta_batch,
            rec_alpha_batch,
            rec_out_batch,
            down_buf,
            ..
        } = &mut *cpu_scratch;

        let mut gpu_proj_indices = [0usize; 4];
        let mut cpu_proj_indices = [0usize; 4];
        let mut gpu_proj_count = 0usize;
        let mut cpu_proj_count = 0usize;
        let proj_keys = [
            recurrent_keys.wqkv,
            recurrent_keys.wgate,
            recurrent_keys.wbeta,
            recurrent_keys.walpha,
        ];
        let proj_dtypes = [
            recurrent_keys.wqkv_dtype,
            recurrent_keys.wgate_dtype,
            recurrent_keys.wbeta_dtype,
            recurrent_keys.walpha_dtype,
        ];
        let proj_dims = [
            dims.conv_dim(),
            dims.inner_size,
            dims.time_step_rank,
            dims.time_step_rank,
        ];
        for i in 0..proj_keys.len() {
            if gpu_batch_supported(proj_dtypes[i], proj_dims[i] as u32, dim as u32) {
                gpu_proj_indices[gpu_proj_count] = i;
                gpu_proj_count += 1;
            } else {
                cpu_proj_indices[cpu_proj_count] = i;
                cpu_proj_count += 1;
            }
        }
        let gpu_proj_indices = &gpu_proj_indices[..gpu_proj_count];
        let cpu_proj_indices = &cpu_proj_indices[..cpu_proj_count];
        let input_ops = if cpu_proj_indices.is_empty() {
            None
        } else {
            Some(Self::recurrent_input_ops(weights, prefix, dims)?)
        };

        let ssm_out_dtype = recurrent_keys.wssm_out_dtype;
        let ssm_gpu = gpu_batch_supported(ssm_out_dtype, dim as u32, dims.inner_size as u32);
        if ssm_gpu {
            metal_ops.record_qwen35_recurrent_batch_gpu_ssm_projection();
        }
        let ssm_key = recurrent_keys.wssm_out;
        if gpu_proj_indices.contains(&0) {
            metal_ops.record_qwen35_recurrent_batch_qkv_gpu_projection();
        }
        let qkv_gpu_fast_path_check = (recurrent_slot_count == 1 && gpu_proj_indices.contains(&0))
            .then(|| {
                Self::recurrent_batch_from_gpu_qkv_single_slot_check(metal_ops, n_tokens, dims)
            });
        let qkv_gpu_fast_path_enabled =
            !layer_is_moe && Self::qwen35_should_enable_gpu_qkv_fast_path(qkv_gpu_fast_path_check);
        if let Some(check) = qkv_gpu_fast_path_check {
            if qkv_gpu_fast_path_enabled {
                metal_ops.record_qwen35_recurrent_batch_qkv_fast_path_eligible();
            } else {
                if check.state_size_too_large {
                    metal_ops.record_qwen35_recurrent_batch_qkv_fast_reject_state_size();
                }
                if check.group_divisibility_invalid {
                    metal_ops.record_qwen35_recurrent_batch_qkv_fast_reject_group_divisibility();
                }
                if check.missing_batch_scratches {
                    metal_ops
                        .record_qwen35_recurrent_batch_qkv_fast_reject_missing_batch_scratches();
                }
                if check.q_capacity_too_small {
                    metal_ops.record_qwen35_recurrent_batch_qkv_fast_reject_q_capacity();
                }
                if check.k_capacity_too_small {
                    metal_ops.record_qwen35_recurrent_batch_qkv_fast_reject_k_capacity();
                }
                if check.v_capacity_too_small {
                    metal_ops.record_qwen35_recurrent_batch_qkv_fast_reject_v_capacity();
                }
                if check.gate_capacity_too_small {
                    metal_ops.record_qwen35_recurrent_batch_qkv_fast_reject_gate_capacity();
                }
                if check.up_capacity_too_small {
                    metal_ops.record_qwen35_recurrent_batch_qkv_fast_reject_up_capacity();
                }
            }
        }

        let mut gpu_plan = Self::qwen35_build_unified_recurrent_gpu_plan(
            recurrent_slot_count,
            gpu_proj_indices,
            cpu_proj_indices,
            qkv_gpu_fast_path_enabled,
            ssm_gpu,
        );
        let use_unified_tail =
            Self::qwen35_should_use_unified_recurrent_tail(unified_recurrent_enabled, layer_is_moe);
        let keep_recurrent_output_on_gpu =
            Self::qwen35_should_keep_recurrent_output_on_gpu_for_unified_tail(
                unified_recurrent_enabled,
                layer_is_moe,
                ssm_gpu,
            ) || Self::qwen35_should_try_resident_moe_gpu_tail(layer_is_moe, ssm_gpu);
        gpu_plan.fused_gpu_recurrent_layer_candidate =
            gpu_plan.fused_gpu_recurrent_layer_candidate
                && (!layer_is_moe
                    || Self::qwen35_should_try_merged_projection_fused_recurrent_layer(true));
        if !gpu_plan.fused_gpu_recurrent_layer_candidate {
            tracing::debug!(
                layer,
                recurrent_slot_count,
                qkv_gpu_fast_path_enabled = gpu_plan.qkv_gpu_fast_path_enabled,
                keep_rec_z_on_gpu = gpu_plan.keep_rec_z_on_gpu,
                ssm_gpu,
                cpu_proj_count = cpu_proj_indices.len(),
                "recurrent layer NOT fused candidate"
            );
        }

        let (layer_completed, tail_inflight) = metal_ops.with_qwen35_recurrent_projection_scratch(
            n_tokens,
            dims.conv_dim(),
            dims.inner_size,
            dims.time_step_rank,
            |temp_scratch| -> anyhow::Result<(bool, Option<ax_engine_metal::InflightFrame>)> {
                let temp_qkv = &temp_scratch.qkv;
                let temp_z = &temp_scratch.z;
                let temp_beta = &temp_scratch.beta;
                let temp_alpha = &temp_scratch.alpha;
                let recurrent_beta_gpu = gpu_proj_indices.contains(&2).then_some(temp_beta);
                let recurrent_alpha_gpu = gpu_proj_indices.contains(&3).then_some(temp_alpha);
                if !Self::qwen35_should_use_unified_recurrent_runtime(unified_recurrent_enabled) {
                    if !Self::run_unified_recurrent_projection_phase(
                        metal_ops,
                        backend,
                        input_ops,
                        gpu_proj_indices,
                        cpu_proj_indices,
                        &proj_keys,
                        &proj_dtypes,
                        &proj_dims,
                        temp_qkv,
                        temp_z,
                        temp_beta,
                        temp_alpha,
                        Qwen35UnifiedRecurrentGpuPlan {
                            qkv_gpu_fast_path_enabled: false,
                            keep_rec_z_on_gpu: false,
                            fused_gpu_recurrent_layer_candidate: false,
                            fused_gpu_recurrent_layer_uses_gpu_alpha_beta: false,
                        },
                        rec_qkv_batch.as_mut_slice(),
                        rec_z_batch.as_mut_slice(),
                        rec_beta_batch.as_mut_slice(),
                        rec_alpha_batch.as_mut_slice(),
                        norm_buf,
                        n_tokens,
                        dim,
                        nt,
                        eps,
                        nw_key,
                        ops.as_deref_mut(),
                    )? {
                        return Ok((false, None));
                    }

                    Self::run_qwen35_recurrent_layer_via_legacy_batch_after_projection(
                        metal_ops,
                        backend,
                        cfg,
                        qwen_kv,
                        weights,
                        prefix,
                        layer,
                        batch_position,
                        recurrent_slot,
                        recurrent_slot_indices,
                        dims,
                        n_tokens,
                        dim,
                        inter_dim,
                        eps,
                        norm_buf.as_mut_slice(),
                        rec_qkv_batch.as_mut_slice(),
                        rec_z_batch.as_mut_slice(),
                        rec_beta_batch.as_mut_slice(),
                        rec_alpha_batch.as_mut_slice(),
                        rec_out_batch.as_mut_slice(),
                        proj_buf.as_mut_slice(),
                        gate_buf.as_mut_slice(),
                        up_buf.as_mut_slice(),
                        down_buf.as_mut_slice(),
                    )?;
                    return Ok((true, None));
                }
                // Try merged projection + fused recurrent in a single CB
                // when all projections are GPU-capable (no CPU fallback needed).
                if gpu_plan.fused_gpu_recurrent_layer_candidate
                    && cpu_proj_indices.is_empty()
                    && Self::qwen35_should_try_merged_projection_fused_recurrent_layer(layer_is_moe)
                {
                    let proj_params = FusedProjectionParams {
                        nw_key,
                        gpu_proj_indices,
                        proj_keys: &proj_keys,
                        proj_dtypes: &proj_dtypes,
                        proj_dims: &proj_dims,
                        dim,
                        eps,
                    };
                    let merged_stats = timed!(
                        ops,
                        recurrent,
                        Self::try_run_recurrent_batch_layer_fused_gpu_single_slot(
                            metal_ops,
                            qwen_kv,
                            layer,
                            recurrent_slot,
                            temp_qkv,
                            temp_z,
                            recurrent_beta_gpu,
                            recurrent_alpha_gpu,
                            recurrent_keys,
                            rec_beta_batch.as_slice(),
                            rec_alpha_batch.as_slice(),
                            n_tokens,
                            dim,
                            inter_dim,
                            dims,
                            eps,
                            ssm_key,
                            ssm_out_dtype,
                            ffn_nw_key,
                            moe_layer,
                            wg_key,
                            wg_dtype,
                            wu_key,
                            wu_dtype,
                            wd_key,
                            wd_dtype,
                            Some(&proj_params),
                            force_backend_state_batch,
                        )
                    )?;
                    if let Some(stats) = merged_stats {
                        if let Some(ref mut ops) = ops {
                            ops.gpu_execute += stats.gpu_execute;
                            ops.gpu_execute_layers += stats.gpu_execute;
                            ops.gpu_readback += stats.gpu_readback;
                        }
                        return Ok((true, None));
                    }
                }

                // Fallback: separate projection CB + fused/non-fused body CB.
                if !Self::run_unified_recurrent_projection_phase(
                    metal_ops,
                    backend,
                    input_ops,
                    gpu_proj_indices,
                    cpu_proj_indices,
                    &proj_keys,
                    &proj_dtypes,
                    &proj_dims,
                    temp_qkv,
                    temp_z,
                    temp_beta,
                    temp_alpha,
                    gpu_plan,
                    rec_qkv_batch.as_mut_slice(),
                    rec_z_batch.as_mut_slice(),
                    rec_beta_batch.as_mut_slice(),
                    rec_alpha_batch.as_mut_slice(),
                    norm_buf,
                    n_tokens,
                    dim,
                    nt,
                    eps,
                    nw_key,
                    ops.as_deref_mut(),
                )? {
                    return Ok((false, None));
                }
                if gpu_plan.fused_gpu_recurrent_layer_candidate {
                    let fused_stats = timed!(
                        ops,
                        recurrent,
                        Self::try_run_recurrent_batch_layer_fused_gpu_single_slot(
                            metal_ops,
                            qwen_kv,
                            layer,
                            recurrent_slot,
                            temp_qkv,
                            temp_z,
                            recurrent_beta_gpu,
                            recurrent_alpha_gpu,
                            recurrent_keys,
                            rec_beta_batch.as_slice(),
                            rec_alpha_batch.as_slice(),
                            n_tokens,
                            dim,
                            inter_dim,
                            dims,
                            eps,
                            ssm_key,
                            ssm_out_dtype,
                            ffn_nw_key,
                            moe_layer,
                            wg_key,
                            wg_dtype,
                            wu_key,
                            wu_dtype,
                            wd_key,
                            wd_dtype,
                            None,
                            force_backend_state_batch,
                        )
                    )?;
                    if let Some(stats) = fused_stats {
                        if let Some(ref mut ops) = ops {
                            ops.gpu_execute += stats.gpu_execute;
                            ops.gpu_execute_layers += stats.gpu_execute;
                            ops.gpu_readback += stats.gpu_readback;
                        }
                        return Ok((true, None));
                    }
                }

                let (recurrent_output_in_batch_scratch, recurrent_output_in_qkv_temp) =
                    Self::run_unified_recurrent_runtime_phase(
                        metal_ops,
                        backend,
                        qwen_kv,
                        layer,
                        recurrent_slot,
                        recurrent_slot_indices,
                        recurrent_keys,
                        temp_qkv,
                        recurrent_beta_gpu,
                        recurrent_alpha_gpu,
                        rec_qkv_batch.as_mut_slice(),
                        rec_beta_batch.as_mut_slice(),
                        rec_alpha_batch.as_mut_slice(),
                        rec_out_batch.as_mut_slice(),
                        n_tokens,
                        dims,
                        eps,
                        Self::qwen35_should_allow_gpu_qkv_runtime_handoff(layer_is_moe),
                        gpu_plan.qkv_gpu_fast_path_enabled,
                        keep_recurrent_output_on_gpu,
                        force_backend_state_batch,
                        ops.as_deref_mut(),
                    )?;

                let tail_plan = Self::qwen35_build_unified_recurrent_tail_buffer_plan(
                    use_unified_tail,
                    layer_is_moe,
                    ssm_gpu,
                    recurrent_output_in_batch_scratch,
                    gpu_plan.keep_rec_z_on_gpu,
                );
                let ssm_norm_key = ssm_gpu.then_some(recurrent_keys.ssm_norm);
                if !ssm_gpu {
                    let weight_cache = metal_ops.lock_weight_cache();
                    let ssm_norm = unsafe {
                        weight_cache
                            .get(&recurrent_keys.ssm_norm)
                            .expect("qwen35 ssm_norm buffer missing after cache")
                            .as_slice::<f32>()
                    };
                    Self::finalize_recurrent_output_batch(
                        rec_out_batch.as_mut_slice(),
                        rec_z_batch.as_slice(),
                        recurrent_total_tokens,
                        dims,
                        ssm_norm,
                        eps,
                    );
                    drop(weight_cache);
                }

                let ssm_result = if !ssm_gpu && tail_plan.use_unified_tail {
                    let (ssm_out_raw, _, _) = Self::recurrent_output_op(weights, prefix, dim)?;
                    let result = &mut down_buf[..n_tokens * dim];
                    result.fill(0.0);
                    let active_rec_out = Self::slot_batch_slice(
                        rec_out_batch.as_slice(),
                        recurrent_slot_indices,
                        recurrent_slot,
                        n_tokens,
                        dims.inner_size,
                    );
                    timed_matmul_bucket!(
                        ops,
                        matmul_output_proj,
                        Self::batched_dequant_matmul_token_major(
                            backend,
                            ssm_out_raw,
                            ssm_out_dtype,
                            active_rec_out,
                            result,
                            n_tokens,
                            dim,
                            dims.inner_size,
                        )
                    );
                    Some(&result[..])
                } else {
                    None
                };

                if !tail_plan.use_unified_tail {
                    if let Some(moe_layer) = moe_layer
                        && Self::qwen35_should_try_resident_moe_gpu_tail(layer_is_moe, ssm_gpu)
                        && timed!(
                            ops,
                            recurrent,
                            Self::try_run_qwen35_recurrent_moe_gpu_tail_after_handoff(
                                metal_ops,
                                dims,
                                temp_qkv,
                                temp_z,
                                recurrent_output_in_batch_scratch,
                                recurrent_output_in_qkv_temp,
                                gpu_plan.keep_rec_z_on_gpu,
                                ssm_norm_key.expect(
                                    "qwen35 recurrent resident MoE GPU tail missing ssm_norm",
                                ),
                                ssm_key,
                                ssm_out_dtype,
                                ffn_nw_key,
                                moe_layer,
                                n_tokens,
                                dim,
                                nt,
                                eps,
                            )
                        )?
                    {
                        return Ok((true, None));
                    }
                    if ssm_gpu {
                        let weight_cache = metal_ops.lock_weight_cache();
                        let ssm_norm = unsafe {
                            weight_cache
                                .get(
                                    &ssm_norm_key
                                        .expect("qwen35 recurrent legacy tail missing ssm_norm"),
                                )
                                .expect("qwen35 ssm_norm buffer missing after cache")
                                .as_slice::<f32>()
                        };
                        Self::finalize_recurrent_output_batch(
                            rec_out_batch.as_mut_slice(),
                            rec_z_batch.as_slice(),
                            recurrent_total_tokens,
                            dims,
                            ssm_norm,
                            eps,
                        );
                        drop(weight_cache);
                    }
                    if Self::qwen35_should_try_projected_moe_gpu_tail(n_tokens) {
                        Self::project_qwen35_recurrent_output_batch(
                            backend,
                            weights,
                            prefix,
                            recurrent_slot_indices,
                            recurrent_slot,
                            dims,
                            n_tokens,
                            dim,
                            ssm_out_dtype,
                            rec_out_batch.as_slice(),
                            proj_buf.as_mut_slice(),
                            ops.as_deref_mut(),
                        )?;
                        if timed!(
                            ops,
                            recurrent,
                            Self::try_run_qwen35_recurrent_projected_moe_gpu_tail(
                                metal_ops,
                                ffn_nw_key,
                                moe_layer.expect("qwen35 recurrent MoE layer keys missing"),
                                n_tokens,
                                dim,
                                eps,
                                proj_buf.as_mut_slice(),
                            )
                        )? {
                            return Ok((true, None));
                        }
                    }
                    return Self::run_qwen35_recurrent_moe_legacy_tail(
                        metal_ops,
                        backend,
                        cfg,
                        weights,
                        prefix,
                        layer,
                        batch_position,
                        recurrent_slot_indices,
                        recurrent_slot,
                        dims,
                        n_tokens,
                        dim,
                        inter_dim,
                        eps,
                        ssm_out_dtype,
                        rec_out_batch.as_slice(),
                        norm_buf.as_mut_slice(),
                        proj_buf.as_mut_slice(),
                        gate_buf.as_mut_slice(),
                        up_buf.as_mut_slice(),
                        down_buf.as_mut_slice(),
                        ops.as_deref_mut(),
                    )
                    .map(|_| (true, None));
                }

                let rec_out_alias = if tail_plan.alias_rec_out {
                    let active_rec_out = Self::slot_batch_slice_mut(
                        rec_out_batch.as_mut_slice(),
                        recurrent_slot_indices,
                        recurrent_slot,
                        n_tokens,
                        dims.inner_size,
                    );
                    Some(unsafe {
                        ax_engine_metal::MetalBuffer::from_mut_slice_no_copy(
                            metal_ops.device.device(),
                            active_rec_out,
                        )
                    }?)
                } else {
                    None
                };
                let rec_z_alias = if tail_plan.alias_rec_z {
                    let active_rec_z = Self::slot_batch_slice_mut(
                        rec_z_batch.as_mut_slice(),
                        recurrent_slot_indices,
                        recurrent_slot,
                        n_tokens,
                        dims.inner_size,
                    );
                    Some(unsafe {
                        ax_engine_metal::MetalBuffer::from_mut_slice_no_copy(
                            metal_ops.device.device(),
                            active_rec_z,
                        )
                    }?)
                } else {
                    None
                };

                let (tail_ok, tail_frame) = Self::run_unified_recurrent_tail_batch_layer(
                    metal_ops,
                    has_tail_graph_ir_schedule,
                    dims,
                    temp_qkv,
                    temp_z,
                    rec_out_alias.as_ref(),
                    rec_z_alias.as_ref(),
                    recurrent_output_in_batch_scratch,
                    recurrent_output_in_qkv_temp,
                    gpu_plan.keep_rec_z_on_gpu,
                    ssm_gpu,
                    ssm_result,
                    ssm_norm_key,
                    ssm_key,
                    ssm_out_dtype,
                    ffn_nw_key,
                    moe_layer,
                    wg_key,
                    wu_key,
                    wd_key,
                    wg_dtype,
                    wu_dtype,
                    wd_dtype,
                    n_tokens,
                    dim,
                    inter_dim,
                    nt,
                    eps,
                    ops,
                    pipelined,
                )?;
                if !tail_ok {
                    return Ok((false, None));
                }
                Ok((true, tail_frame))
            },
        )?;

        if layer_completed {
            return Ok((None, tail_inflight));
        }

        Ok((Some("unified_recurrent_layer_unavailable"), None))
    }

    /// GPU-unified prefill: keeps hidden on GPU, encodes full-attention layers
    /// in single command buffers. Falls back to `forward_batch_impl` on failure.
    #[allow(clippy::too_many_arguments)]
    fn try_forward_batch_gpu_unified(
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: Option<&mut [f32]>,
        logits_all: Option<&mut Vec<f32>>,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<bool> {
        let total_t = OpTimer::start();
        let fallback = |reason: &'static str| {
            Self::warn_prefill_unified_fallback_once(reason);
            Ok(false)
        };
        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return fallback("missing_metal_backend");
        };
        let cfg = ctx.config;
        let backend = ctx.backend;
        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            return fallback("missing_qwen35_kv");
        };

        let dims = Self::recurrent_dims(cfg)?;
        let n_tokens = token_ids.len();
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let eps = cfg.rms_norm_eps;
        let batch_position = qwen_kv.seq_len();
        let recurrent_slot = qwen_kv.active_slot();
        let recurrent_slot_indices = qwen_kv.recurrent_batch_slot_indices().into_owned();
        let recurrent_slot_count = recurrent_slot_indices.len();
        if qwen_kv.gpu_attention().is_some()
            && !qwen_kv.ensure_gpu_attention_capacity_for(batch_position + n_tokens)
        {
            eprintln!(
                "[BATCH FALLBACK] gpu_attention_capacity_unavailable: batch_position={batch_position} n_tokens={n_tokens}"
            );
            return fallback("gpu_attention_capacity_unavailable");
        }
        if qwen_kv.gpu_attention().is_none() {
            eprintln!("[BATCH FALLBACK] no GPU attention KV");
        }
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_qwen35(metal_ops, weights, cfg)?;
        }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 gpu layer keys"))?;
        let gpu_layer_keys = Self::cached_gpu_layer_keys(cached.lm_head)
            .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 recurrent gpu layer keys"))?;
        let moe_layer_keys = if Self::qwen35_is_moe(cfg) {
            Self::cached_moe_layer_keys(cached.lm_head)
        } else {
            None
        };

        // Check that all projection dtypes are GPU-supported for this batch shape.
        let gpu_batch_supported = |dt: crate::gguf::tensor::GgmlType, m: u32, k: u32| {
            Self::qwen35_batch_projection_supported(metal_ops, dt, m, n_tokens as u32, k)
        };

        // Quick check: verify at least one full-attention layer's weights are supported.
        for layer in 0..n_layers {
            if !cfg.qwen35_is_recurrent_layer(layer) {
                if !gpu_batch_supported(
                    cached.layers[layer].wq_dtype,
                    (q_dim * 2) as u32,
                    dim as u32,
                ) {
                    return fallback("unsupported_full_attention_q_projection_dtype");
                }
                break;
            }
        }

        // Init GPU batch scratch buffers.
        metal_ops.init_batch_scratches(cfg, n_tokens);

        let nt = n_tokens as u32;

        // Embed tokens directly into GPU hidden buffer (UMA write).
        {
            let mut bs_guard = metal_ops.batch_scratches();
            let Some(bs) = bs_guard.as_mut() else {
                return fallback("missing_batch_scratches_after_init");
            };
            let h = unsafe {
                std::slice::from_raw_parts_mut(
                    bs.hidden.contents().as_ptr() as *mut f32,
                    n_tokens * dim,
                )
            };
            timed!(
                ops,
                dequant,
                Self::dequantize_token_embeddings_batch(weights, token_ids, h, dim)
            )?;
        }

        let recurrent_layer_state_owners = (0..n_layers)
            .map(|layer| qwen_kv.layer_state_owner(recurrent_slot, layer))
            .collect::<Vec<_>>();
        let prefill_schedule = prefill_schedule::build_qwen35_prefill_schedule(
            cfg,
            batch_position,
            n_tokens,
            qwen_kv.gpu_attention().is_some(),
            Some(&recurrent_layer_state_owners),
        );
        let _prefill_schedule_summary =
            prefill_schedule::summarize_qwen35_prefill_schedule(&prefill_schedule);

        let pipelined = prefill_schedule::prefill_inter_step_pipelined_enabled();
        let max_layer = Self::debug_max_layer_inclusive();
        let mut inflight: Option<ax_engine_metal::InflightFrame> = None;
        let mut deferred_kv_layer: Option<usize> = None;

        let mut step_idx = 0usize;
        while step_idx < prefill_schedule.steps.len() {
            let step = prefill_schedule.steps[step_idx];
            if max_layer.is_some_and(|max_layer| step.layer > max_layer) {
                break;
            }
            let layer = step.layer;
            let prefix = format!("blk.{layer}");
            let recurrent_keys = match &gpu_layer_keys[layer] {
                Qwen3_5GpuLayerKeys::Recurrent(keys) => Some(keys),
                Qwen3_5GpuLayerKeys::FullAttention => None,
            };
            let moe_layer = moe_layer_keys
                .as_ref()
                .and_then(|layer_keys| layer_keys[layer].as_ref());
            match step.kind {
                prefill_schedule::Qwen35PrefillExecutionStepKind::FullAttention {
                    uses_graph_ir,
                    ..
                } => {
                    // Wait for any inflight recurrent tail before proceeding.
                    if let Some(frame) = inflight.take() {
                        metal_ops.device.wait_frame(frame)?;
                    }
                    // Flush deferred K/V readback before this FA overwrites bs.k_buf/v_buf.
                    if let Some(prev_fa_layer) = deferred_kv_layer.take() {
                        Self::flush_deferred_kv_readback(
                            metal_ops,
                            qwen_kv,
                            prev_fa_layer,
                            n_tokens,
                            kv_dim,
                        );
                    }
                    if !Self::run_unified_full_attention_batch_layer(
                        metal_ops,
                        cfg,
                        qwen_kv,
                        weights,
                        uses_graph_ir,
                        &cached.layers[layer],
                        moe_layer,
                        &prefix,
                        layer,
                        batch_position,
                        n_tokens,
                        dim,
                        q_dim,
                        kv_dim,
                        inter_dim,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        nt,
                        eps,
                        ops.as_deref_mut(),
                        pipelined,
                    )? {
                        // MoE layer: process on CPU using UMA access to hidden buffer.
                        if !Self::qwen35_layer_uses_moe(weights, &prefix) {
                            return fallback("unified_full_attention_layer_unavailable");
                        }
                        // Wait for any inflight GPU work.
                        if let Some(frame) = inflight.take() {
                            metal_ops.device.wait_frame(frame)?;
                        }
                        if let Some(prev_fa_layer) = deferred_kv_layer.take() {
                            Self::flush_deferred_kv_readback(
                                metal_ops,
                                qwen_kv,
                                prev_fa_layer,
                                n_tokens,
                                kv_dim,
                            );
                        }
                        // Access hidden buffer via UMA (drop guard to avoid
                        // deadlock with inner batch_scratches locks).
                        let hidden_ptr = {
                            let bs_guard = metal_ops.batch_scratches();
                            let Some(bs) = bs_guard.as_ref() else {
                                return fallback("moe_fa_missing_batch_scratches");
                            };
                            bs.hidden.contents().as_ptr() as *mut f32
                        };
                        {
                            let hidden = unsafe {
                                std::slice::from_raw_parts_mut(hidden_ptr, n_tokens * dim)
                            };
                            let full_attn_params =
                                AttentionParams::new(n_heads, n_kv_heads, head_dim);
                            let mut norm_buf = vec![0.0f32; n_tokens * dim];
                            let mut q_gate_batch = vec![0.0f32; n_tokens * q_dim * 2];
                            let mut q_batch = vec![0.0f32; n_tokens * q_dim];
                            let mut k_batch = vec![0.0f32; n_tokens * kv_dim];
                            let mut v_batch = vec![0.0f32; n_tokens * kv_dim];
                            let mut fused_input_batch = vec![0.0f32; n_tokens * q_dim * 2];
                            let mut attn_out_batch = vec![0.0f32; n_tokens * q_dim];
                            let mut proj_buf = vec![0.0f32; n_tokens * dim];

                            // Batch projections into 1 CB via try_norm_and_project_batch_gpu.
                            let input_plan =
                                Self::full_attention_input_plan(weights, &prefix, q_dim, kv_dim)?;
                            let gpu_projected = match &input_plan {
                                Qwen35FullAttentionInputPlan::Split(ops) => {
                                    let mut outs: [&mut [f32]; 3] =
                                        [&mut q_gate_batch, &mut k_batch, &mut v_batch];
                                    Self::try_norm_and_project_batch_gpu(
                                        cfg,
                                        backend,
                                        weights,
                                        &prefix,
                                        hidden,
                                        &mut norm_buf,
                                        ops,
                                        &mut outs,
                                        n_tokens,
                                        dim,
                                        eps,
                                    )?
                                }
                                _ => false,
                            };
                            if !gpu_projected {
                                Self::apply_attention_norm_batch(
                                    weights,
                                    &prefix,
                                    hidden,
                                    &mut norm_buf,
                                    n_tokens,
                                    dim,
                                    eps,
                                )?;
                            }
                            Self::run_full_attention_batch_layer(
                                cfg,
                                backend,
                                weights,
                                &prefix,
                                qwen_kv,
                                layer,
                                batch_position,
                                &norm_buf,
                                &mut q_gate_batch,
                                &mut q_batch,
                                &mut k_batch,
                                &mut v_batch,
                                &mut fused_input_batch,
                                &mut attn_out_batch,
                                &mut proj_buf,
                                n_tokens,
                                dim,
                                q_dim,
                                kv_dim,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                &full_attn_params,
                                gpu_projected,
                            )?;
                            let mut gate_buf = vec![0.0f32; n_tokens * inter_dim];
                            let mut up_buf = vec![0.0f32; n_tokens * inter_dim];
                            let mut down_buf = vec![0.0f32; n_tokens * dim];
                            Self::apply_layer_tail_batch(
                                cfg,
                                backend,
                                weights,
                                &prefix,
                                hidden,
                                &proj_buf,
                                &mut norm_buf,
                                &mut gate_buf,
                                &mut up_buf,
                                &mut down_buf,
                                n_tokens,
                                dim,
                                inter_dim,
                                eps,
                                layer,
                                batch_position,
                            )?;
                        }
                    }
                    if pipelined {
                        deferred_kv_layer = Some(layer);
                    }
                    step_idx += 1;
                }
                prefill_schedule::Qwen35PrefillExecutionStepKind::RecurrentProjection {
                    force_backend_state_batch,
                }
                | prefill_schedule::Qwen35PrefillExecutionStepKind::RecurrentRuntime {
                    force_backend_state_batch,
                }
                | prefill_schedule::Qwen35PrefillExecutionStepKind::RecurrentTail {
                    force_backend_state_batch,
                    ..
                } => {
                    let is_first_step_for_layer =
                        step_idx == 0 || prefill_schedule.steps[step_idx - 1].layer != layer;
                    if !is_first_step_for_layer {
                        step_idx += 1;
                        continue;
                    }
                    // Wait for any inflight recurrent tail from previous layer.
                    if let Some(frame) = inflight.take() {
                        metal_ops.device.wait_frame(frame)?;
                    }
                    let mut end_step = step_idx;
                    let mut has_projection_phase = false;
                    let mut has_runtime_phase = false;
                    let mut has_tail_graph_ir_schedule = false;
                    while end_step < prefill_schedule.steps.len()
                        && prefill_schedule.steps[end_step].layer == layer
                    {
                        match prefill_schedule.steps[end_step].kind {
                            prefill_schedule::Qwen35PrefillExecutionStepKind::FullAttention {
                                ..
                            } => break,
                            prefill_schedule::Qwen35PrefillExecutionStepKind::RecurrentProjection {
                                ..
                            } => has_projection_phase = true,
                            prefill_schedule::Qwen35PrefillExecutionStepKind::RecurrentRuntime {
                                ..
                            } => has_runtime_phase = true,
                            prefill_schedule::Qwen35PrefillExecutionStepKind::RecurrentTail {
                                uses_graph_ir,
                                ..
                            } => has_tail_graph_ir_schedule = uses_graph_ir,
                        }
                        end_step += 1;
                    }
                    let (reason, tail_frame) = Self::run_unified_recurrent_batch_layer(
                        metal_ops,
                        backend,
                        cfg,
                        qwen_kv,
                        weights,
                        has_projection_phase,
                        has_runtime_phase,
                        has_tail_graph_ir_schedule,
                        &cached.layers[layer],
                        recurrent_keys
                            .expect("qwen35 recurrent layer missing cached recurrent keys"),
                        moe_layer,
                        &prefix,
                        layer,
                        batch_position,
                        recurrent_slot,
                        &recurrent_slot_indices,
                        recurrent_slot_count,
                        dims,
                        n_tokens,
                        dim,
                        inter_dim,
                        nt,
                        eps,
                        force_backend_state_batch,
                        ops.as_deref_mut(),
                        pipelined,
                    )?;
                    if let Some(_reason) = reason {
                        // Recurrent fallback path: keep unified full-attention layers,
                        // but reuse the proven-correct legacy recurrent batch path.
                        // Wait for any inflight GPU work.
                        if let Some(frame) = inflight.take() {
                            metal_ops.device.wait_frame(frame)?;
                        }
                        // Access hidden buffer via UMA (drop guard for inner locks).
                        let rec_hidden_ptr = {
                            let bs_guard = metal_ops.batch_scratches();
                            let Some(bs) = bs_guard.as_ref() else {
                                return fallback("moe_rec_missing_batch_scratches");
                            };
                            bs.hidden.contents().as_ptr() as *mut f32
                        };
                        {
                            let hidden = unsafe {
                                std::slice::from_raw_parts_mut(rec_hidden_ptr, n_tokens * dim)
                            };
                            let mut norm_buf = vec![0.0f32; n_tokens * dim];
                            let mut rec_qkv = vec![0.0f32; n_tokens * dims.conv_dim()];
                            let mut rec_z = vec![0.0f32; n_tokens * dims.inner_size];
                            let mut rec_beta = vec![0.0f32; n_tokens * dims.time_step_rank];
                            let mut rec_alpha = vec![0.0f32; n_tokens * dims.time_step_rank];
                            let mut rec_out = vec![0.0f32; n_tokens * dims.inner_size];
                            let mut proj_buf = vec![0.0f32; n_tokens * dim];

                            // Batch recurrent projections into 1 CB.
                            let input_ops = Self::recurrent_input_ops(weights, &prefix, dims)?;
                            let mut rec_outs: [&mut [f32]; 4] =
                                [&mut rec_qkv, &mut rec_z, &mut rec_beta, &mut rec_alpha];
                            let gpu_projected = Self::try_norm_and_project_batch_gpu(
                                cfg,
                                backend,
                                weights,
                                &prefix,
                                hidden,
                                &mut norm_buf,
                                &input_ops,
                                &mut rec_outs,
                                n_tokens,
                                dim,
                                eps,
                            )?;
                            if !gpu_projected {
                                Self::apply_attention_norm_batch(
                                    weights,
                                    &prefix,
                                    hidden,
                                    &mut norm_buf,
                                    n_tokens,
                                    dim,
                                    eps,
                                )?;
                            }
                            Self::run_recurrent_batch_layer(
                                cfg,
                                backend,
                                weights,
                                &prefix,
                                qwen_kv,
                                recurrent_slot,
                                layer,
                                batch_position,
                                dims,
                                &recurrent_slot_indices,
                                &norm_buf,
                                &mut rec_qkv,
                                &mut rec_z,
                                &mut rec_beta,
                                &mut rec_alpha,
                                &mut rec_out,
                                &mut proj_buf,
                                n_tokens,
                                dim,
                                gpu_projected, // skip projections if already done on GPU
                            )?;
                            let mut gate_buf = vec![0.0f32; n_tokens * inter_dim];
                            let mut up_buf = vec![0.0f32; n_tokens * inter_dim];
                            let mut down_buf = vec![0.0f32; n_tokens * dim];
                            Self::apply_layer_tail_batch(
                                cfg,
                                backend,
                                weights,
                                &prefix,
                                hidden,
                                &proj_buf,
                                &mut norm_buf,
                                &mut gate_buf,
                                &mut up_buf,
                                &mut down_buf,
                                n_tokens,
                                dim,
                                inter_dim,
                                eps,
                                layer,
                                batch_position,
                            )?;
                        }
                    } else {
                        inflight = tail_frame;
                    }
                    step_idx = end_step;
                }
            }
        }

        // Wait for any remaining inflight tail frame.
        if let Some(frame) = inflight.take() {
            metal_ops.device.wait_frame(frame)?;
        }
        // Flush any remaining deferred K/V readback.
        if let Some(prev_fa_layer) = deferred_kv_layer.take() {
            Self::flush_deferred_kv_readback(metal_ops, qwen_kv, prev_fa_layer, n_tokens, kv_dim);
        }

        qwen_kv.finalize_batch(n_tokens);

        let writes_last_logits = logits.is_some();
        if !Self::write_unified_batch_outputs(
            metal_ops, backend, weights, logits, logits_all, n_tokens, dim, vocab_size, eps,
        )? {
            let reason = if writes_last_logits {
                "missing_batch_scratches_for_last_logits"
            } else {
                "missing_batch_scratches_for_all_logits"
            };
            return fallback(reason);
        }
        if let Some(ref mut ops) = ops {
            ops.gpu += total_t.elapsed();
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_impl(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        mut logits: Option<&mut [f32]>,
        mut logits_all: Option<&mut Vec<f32>>,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            !token_ids.is_empty(),
            "qwen35 forward_batch requires at least one token"
        );
        anyhow::ensure!(
            logits.is_some() ^ logits_all.is_some(),
            "qwen35 batch forward requires either last logits or all logits output"
        );

        let force_serial = env_flag_enabled("AX_SERIAL_PREFILL");
        if force_serial
            || !Self::gpu_batch_prefill_enabled_for_config(ctx.config)
            || !ctx.backend.use_gpu_decode()
            || token_ids.len() <= 1
        {
            return self.forward_batch_serial_fallback(
                ctx,
                token_ids,
                kv.seq_len(),
                kv,
                weights,
                logits,
                logits_all,
            );
        }

        if Self::unified_prefill_enabled()
            && let Some(logits) = logits.as_mut()
        {
            if Self::try_forward_batch_gpu_unified(
                ctx,
                token_ids,
                kv,
                weights,
                Some(&mut **logits),
                None,
                ops.as_deref_mut(),
            )? {
                return Ok(());
            }
        } else if Self::unified_prefill_enabled()
            && let Some(logits_all) = logits_all.as_mut()
            && Self::try_forward_batch_gpu_unified(
                ctx,
                token_ids,
                kv,
                weights,
                None,
                Some(&mut **logits_all),
                ops.as_deref_mut(),
            )?
        {
            return Ok(());
        }

        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            anyhow::bail!("Qwen3_5Forward requires ModelKv::Qwen35");
        };
        let recurrent_slot = qwen_kv.active_slot();

        let cfg = ctx.config;
        let backend = ctx.backend;
        let dims = Self::recurrent_dims(cfg)?;
        let n_tokens = token_ids.len();
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let conv_dim = dims.conv_dim();
        let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);
        let recurrent_slot_indices = qwen_kv.recurrent_batch_slot_indices().into_owned();
        let recurrent_slot_count = recurrent_slot_indices.len();
        let max_layer = Self::debug_max_layer_inclusive();

        let batch_position = qwen_kv.seq_len();
        let mut scratch = Self::lock_cpu_batch_fallback_scratch();
        scratch.ensure_lengths(
            n_tokens,
            dim,
            inter_dim,
            q_dim,
            kv_dim,
            recurrent_slot_count,
            conv_dim,
            dims.inner_size,
            dims.time_step_rank,
        );

        let Qwen3_5CpuBatchFallbackScratch {
            hidden,
            norm_buf,
            proj_buf,
            gate_buf,
            up_buf,
            down_buf,
            q_gate_batch,
            q_batch,
            k_batch,
            v_batch,
            fused_input_batch,
            attn_out_batch,
            rec_qkv_batch,
            rec_z_batch,
            rec_beta_batch,
            rec_alpha_batch,
            rec_out_batch,
            final_hidden,
        } = &mut *scratch;

        timed!(
            ops,
            dequant,
            Self::dequantize_token_embeddings_batch(weights, token_ids, hidden.as_mut_slice(), dim)
        )?;

        for layer in 0..n_layers {
            if max_layer.is_some_and(|max_layer| layer > max_layer) {
                break;
            }
            let prefix = format!("blk.{layer}");

            match Self::layer_type(cfg, layer) {
                Qwen35LayerType::FullAttention => {
                    let input_plan =
                        Self::full_attention_input_plan(weights, &prefix, q_dim, kv_dim)?;
                    let gpu_projected = match &input_plan {
                        Qwen35FullAttentionInputPlan::Split(ops) => {
                            let mut outs: [&mut [f32]; 3] = [
                                q_gate_batch.as_mut_slice(),
                                k_batch.as_mut_slice(),
                                v_batch.as_mut_slice(),
                            ];
                            Self::try_norm_and_project_batch_gpu(
                                cfg,
                                backend,
                                weights,
                                &prefix,
                                hidden.as_slice(),
                                norm_buf.as_mut_slice(),
                                ops,
                                &mut outs,
                                n_tokens,
                                dim,
                                cfg.rms_norm_eps,
                            )?
                        }
                        _ => false,
                    };
                    if !gpu_projected {
                        Self::apply_attention_norm_batch(
                            weights,
                            &prefix,
                            hidden.as_slice(),
                            norm_buf.as_mut_slice(),
                            n_tokens,
                            dim,
                            cfg.rms_norm_eps,
                        )?;
                    }
                    Self::assert_finite_if_enabled(
                        "attn_norm_output_batch",
                        norm_buf.as_slice(),
                        layer,
                        batch_position,
                    )?;
                    Self::run_full_attention_batch_layer(
                        cfg,
                        backend,
                        weights,
                        &prefix,
                        qwen_kv,
                        layer,
                        batch_position,
                        norm_buf.as_slice(),
                        q_gate_batch.as_mut_slice(),
                        q_batch.as_mut_slice(),
                        k_batch.as_mut_slice(),
                        v_batch.as_mut_slice(),
                        fused_input_batch.as_mut_slice(),
                        attn_out_batch.as_mut_slice(),
                        proj_buf.as_mut_slice(),
                        n_tokens,
                        dim,
                        q_dim,
                        kv_dim,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        &full_attn_params,
                        gpu_projected,
                    )?;
                }
                Qwen35LayerType::RecurrentGdn => {
                    let input_ops = Self::recurrent_input_ops(weights, &prefix, dims)?;
                    let mut rec_outs: [&mut [f32]; 4] = [
                        rec_qkv_batch.as_mut_slice(),
                        rec_z_batch.as_mut_slice(),
                        rec_beta_batch.as_mut_slice(),
                        rec_alpha_batch.as_mut_slice(),
                    ];
                    let gpu_projected = Self::try_norm_and_project_batch_gpu(
                        cfg,
                        backend,
                        weights,
                        &prefix,
                        hidden.as_slice(),
                        norm_buf.as_mut_slice(),
                        &input_ops,
                        &mut rec_outs,
                        n_tokens,
                        dim,
                        cfg.rms_norm_eps,
                    )?;
                    if !gpu_projected {
                        Self::apply_attention_norm_batch(
                            weights,
                            &prefix,
                            hidden.as_slice(),
                            norm_buf.as_mut_slice(),
                            n_tokens,
                            dim,
                            cfg.rms_norm_eps,
                        )?;
                    }
                    Self::assert_finite_if_enabled(
                        "attn_norm_output_batch",
                        norm_buf.as_slice(),
                        layer,
                        batch_position,
                    )?;
                    Self::run_recurrent_batch_layer(
                        cfg,
                        backend,
                        weights,
                        &prefix,
                        qwen_kv,
                        recurrent_slot,
                        layer,
                        batch_position,
                        dims,
                        &recurrent_slot_indices,
                        norm_buf.as_slice(),
                        rec_qkv_batch.as_mut_slice(),
                        rec_z_batch.as_mut_slice(),
                        rec_beta_batch.as_mut_slice(),
                        rec_alpha_batch.as_mut_slice(),
                        rec_out_batch.as_mut_slice(),
                        proj_buf.as_mut_slice(),
                        n_tokens,
                        dim,
                        gpu_projected,
                    )?;
                }
            }

            Self::apply_layer_tail_batch(
                cfg,
                backend,
                weights,
                &prefix,
                hidden.as_mut_slice(),
                proj_buf.as_slice(),
                norm_buf.as_mut_slice(),
                gate_buf.as_mut_slice(),
                up_buf.as_mut_slice(),
                down_buf.as_mut_slice(),
                n_tokens,
                dim,
                inter_dim,
                cfg.rms_norm_eps,
                layer,
                batch_position,
            )?;
        }

        qwen_kv.finalize_batch(n_tokens);

        match (logits, logits_all) {
            (Some(logits), None) => Self::write_last_batch_logits(
                backend,
                hidden.as_mut_slice(),
                n_tokens,
                dim,
                vocab_size,
                cfg.rms_norm_eps,
                weights,
                logits,
            ),
            (None, Some(logits_all)) => Self::write_all_batch_logits_with_scratch(
                backend,
                hidden.as_slice(),
                n_tokens,
                dim,
                vocab_size,
                cfg.rms_norm_eps,
                weights,
                logits_all,
                final_hidden,
            ),
            _ => unreachable!("validated above"),
        }
    }
}
