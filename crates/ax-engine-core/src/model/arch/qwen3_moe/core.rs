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

    fn qwen3moe_prefill_concurrent_enabled(
        gate_dtype: crate::gguf::tensor::GgmlType,
        up_dtype: crate::gguf::tensor::GgmlType,
        down_dtype: crate::gguf::tensor::GgmlType,
    ) -> bool {
        if let Some(enabled) =
            crate::model::shared::env_flag_override("AX_QWEN3MOE_PREFILL_CONCURRENT")
        {
            return enabled;
        }

        // Sequential encoding remains the better default for the fully-Q4
        // routed stack. Q5/Q6/Q8 all improved with the concurrent encoder
        // once the cold-path prep bugs were fixed.
        !(gate_dtype == crate::gguf::tensor::GgmlType::Q4K
            && up_dtype == crate::gguf::tensor::GgmlType::Q4K
            && down_dtype == crate::gguf::tensor::GgmlType::Q4K)
    }

    fn qwen3moe_blocked_q6q8_down_enabled() -> bool {
        !matches!(
            crate::model::shared::env_flag_override("AX_QWEN3MOE_BLOCKED_Q6Q8_DOWN"),
            Some(false)
        )
    }

    fn qwen3moe_gpu_pipelined_decode_enabled() -> bool {
        !matches!(
            crate::model::shared::env_flag_override("AX_QWEN3MOE_GPU_PIPELINED_DECODE"),
            Some(false)
        )
    }

    fn qwen3moe_concurrent_decode_enabled(
        gate_dtype: crate::gguf::tensor::GgmlType,
        up_dtype: crate::gguf::tensor::GgmlType,
        down_dtype: crate::gguf::tensor::GgmlType,
    ) -> bool {
        crate::model::shared::qwen3moe_concurrent_decode_enabled_for_layout(
            gate_dtype,
            up_dtype,
            down_dtype,
        )
    }

    pub(crate) fn qwen3moe_split_moe_decode_command_buffers(
        gate_dtype: crate::gguf::tensor::GgmlType,
        up_dtype: crate::gguf::tensor::GgmlType,
        down_dtype: crate::gguf::tensor::GgmlType,
    ) -> bool {
        match crate::model::shared::env_flag_override("AX_QWEN3MOE_SPLIT_MOE_DECODE") {
            Some(enabled) => enabled,
            None => {
                gate_dtype == crate::gguf::tensor::GgmlType::Q5K
                    && up_dtype == crate::gguf::tensor::GgmlType::Q5K
                    && down_dtype == crate::gguf::tensor::GgmlType::Q6K
            }
        }
    }

    fn qwen3moe_single_token_moe_scratch_summary(
        metal_ops: &MetalOps,
        dim: usize,
        n_expert_used: usize,
        expert_inter_dim: usize,
    ) -> Option<String> {
        let first_nonfinite = |values: &[f32]| {
            values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
        };
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref()?;
        let ids = unsafe {
            bs.moe_expert_ids
                .as_ref()?
                .as_slice::<i32>()[..n_expert_used]
                .to_vec()
        };
        let weights = unsafe {
            bs.moe_expert_weights
                .as_ref()?
                .as_slice::<f32>()[..n_expert_used]
                .to_vec()
        };
        let gate_out = unsafe {
            &bs.moe_gate_out
                .as_ref()?
                .as_slice::<f32>()[..n_expert_used * expert_inter_dim]
        };
        let up_out = unsafe {
            &bs.moe_up_out
                .as_ref()?
                .as_slice::<f32>()[..n_expert_used * expert_inter_dim]
        };
        let down_out = unsafe {
            &bs.moe_down_out
                .as_ref()?
                .as_slice::<f32>()[..n_expert_used * dim]
        };
        let accum = unsafe { &bs.moe_accum.as_ref()?.as_slice::<f32>()[..dim] };
        Some(format!(
            "ids={ids:?} weights={weights:?} gate_max_abs={} up_max_abs={} gate_nonfinite={:?} up_nonfinite={:?} down_nonfinite={:?} accum_nonfinite={:?}",
            gate_out.iter().copied().map(f32::abs).fold(0.0f32, f32::max),
            up_out.iter().copied().map(f32::abs).fold(0.0f32, f32::max),
            first_nonfinite(gate_out),
            first_nonfinite(up_out),
            first_nonfinite(down_out),
            first_nonfinite(accum),
        ))
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
            let use_fused_qkv = metal_ops.metal_fused_qkv_enabled()
                && wq_dtype == wk_dtype
                && wq_dtype == wv_dtype
                && matches!(
                    wq_dtype,
                    crate::gguf::tensor::GgmlType::Q4K | crate::gguf::tensor::GgmlType::Q6K
                );
            if use_fused_qkv {
                metal_ops.ensure_qkv_fused_quant_cached(wq_raw, wk_raw, wv_raw);
                if use_precomputed_f16 && wq_dtype == crate::gguf::tensor::GgmlType::Q4K {
                    metal_ops.ensure_precomputed_q4k_f16_fused_qkv(
                        wq_raw,
                        wk_raw,
                        wv_raw,
                        (q_dim + 2 * kv_dim) as u32,
                        dim as u32,
                    )?;
                }
            }

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
                gemma4_moe_router_scale: None,
                gemma4_moe_pre_ffw_norm_2: None,
                gemma4_moe_post_ffw_norm_1: None,
                gemma4_moe_post_ffw_norm_2: None,
                gemma4_moe_expert_scales: None,
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
                moe_expert_gate_dtype: Some(moe_layer.gate_dtype),
                moe_expert_up_dtype: Some(moe_layer.up_dtype),
                moe_expert_down_dtype: Some(moe_layer.down_dtype),
                moe_expert_gate_stride: Some(moe_layer.gate_stride),
                moe_expert_up_stride: Some(moe_layer.up_stride),
                moe_expert_down_stride: Some(moe_layer.down_stride),
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
    fn qwen3moe_decode_plan(
        metal_ops: &MetalOps,
        gpu_kv: &mut crate::kv::GpuKv,
        embedding_dim: u32,
        head_dim: u32,
        attend_len: usize,
        pipelined: bool,
        gate_dtype: crate::gguf::tensor::GgmlType,
        up_dtype: crate::gguf::tensor::GgmlType,
        down_dtype: crate::gguf::tensor::GgmlType,
    ) -> crate::model::execution_plan::GpuDecodeExecutionPlan {
        let mut plan = if pipelined {
            DecodeExecutionPlan::qwen3moe_pipelined(
                metal_ops,
                gpu_kv,
                embedding_dim,
                head_dim,
                attend_len,
            )
        } else {
            DecodeExecutionPlan::qwen3moe_single_cb(
                metal_ops,
                gpu_kv,
                embedding_dim,
                head_dim,
                attend_len,
            )
        };

        if plan.encoder != crate::model::execution_plan::DecodeEncoderPlan::Concurrent
            && Self::qwen3moe_concurrent_decode_enabled(gate_dtype, up_dtype, down_dtype)
        {
            plan.encoder = crate::model::execution_plan::DecodeEncoderPlan::Concurrent;
            plan.barriers = crate::model::execution_plan::DecodeBarrierPlan::Smart;
        }

        plan
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_qwen3moe_gpu_layer_range(
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        metal_ops: &MetalOps,
        cfg: &ModelConfig,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        s: &crate::backend::metal::GpuScratchBuffers,
        gpu_kv: &mut crate::kv::GpuKv,
        cached: &crate::backend::metal::CachedModelKeys,
        weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
        moe_weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
        exec_plan: &crate::model::execution_plan::GpuDecodeExecutionPlan,
        position: usize,
        layer_start: usize,
        layer_end: usize,
        gate_dtype: crate::gguf::tensor::GgmlType,
        up_dtype: crate::gguf::tensor::GgmlType,
        down_dtype: crate::gguf::tensor::GgmlType,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        ops: &mut Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        Self::encode_qwen3moe_gpu_layer_range_internal(
            encoder,
            barrier,
            metal_ops,
            cfg,
            hidden_buf,
            s,
            gpu_kv,
            cached,
            weight_cache,
            moe_weight_cache,
            exec_plan,
            position,
            layer_start,
            layer_end,
            gate_dtype,
            up_dtype,
            down_dtype,
            gate_stride,
            up_stride,
            down_stride,
            ops,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_qwen3moe_gpu_layer_range_internal(
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        metal_ops: &MetalOps,
        cfg: &ModelConfig,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        s: &crate::backend::metal::GpuScratchBuffers,
        gpu_kv: &mut crate::kv::GpuKv,
        cached: &crate::backend::metal::CachedModelKeys,
        weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
        moe_weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
        exec_plan: &crate::model::execution_plan::GpuDecodeExecutionPlan,
        position: usize,
        layer_start: usize,
        layer_end: usize,
        gate_dtype: crate::gguf::tensor::GgmlType,
        up_dtype: crate::gguf::tensor::GgmlType,
        down_dtype: crate::gguf::tensor::GgmlType,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        ops: &mut Option<&mut OpBreakdown>,
        run_moe: bool,
    ) -> anyhow::Result<()> {
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let n_expert = cfg.n_expert.unwrap_or(0) as usize;
        let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
        let eps = cfg.rms_norm_eps;
        let kv_f16 = gpu_kv.is_f16();
        let full_seq_len = u32::try_from(position.saturating_add(1))
            .map_err(|_| anyhow::anyhow!("qwen3moe decode sequence length overflow"))?;

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
            let kv_offset = u32::try_from(position.saturating_mul(kv_stride)).map_err(|_| {
                anyhow::anyhow!(
                    "qwen3moe KV offset overflow: position={}, kv_stride={}",
                    position,
                    kv_stride
                )
            })?;

            let layer_norm_t = OpTimer::start();
            barrier.pre_dispatch(&[hidden_buf], &[&s.norm_buf]);
            metal_ops.elementwise.encode_rms_norm_out(
                encoder,
                hidden_buf,
                attn_norm_w,
                &s.norm_buf,
                dim as u32,
                eps,
            );
            barrier.post_dispatch(&[hidden_buf], &[&s.norm_buf]);
            barrier.step(encoder);
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_norm += layer_norm_t.elapsed();
            }

            let qkv_t = OpTimer::start();
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
                exec_plan.dequant_dispatch,
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
                exec_plan.dequant_dispatch,
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
                exec_plan.dequant_dispatch,
            );
            barrier.post_dispatch(&[&s.norm_buf], &[&s.v_buf]);
            barrier.step(encoder);
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_qkv += qkv_t.elapsed();
            }

            if let (Some(qn_key), Some(kn_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
                let qk_norm_t = OpTimer::start();
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
                if let Some(ops_ref) = ops.as_deref_mut() {
                    ops_ref.gpu_encode_layer_norm += qk_norm_t.elapsed();
                }
            }

            let rope_t = OpTimer::start();
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
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_rope += rope_t.elapsed();
            }

            let kv_append_t = OpTimer::start();
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
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_kv_append += kv_append_t.elapsed();
            }

            let attn_t = OpTimer::start();
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
                    full_seq_len,
                    exec_plan.attention_dispatch,
            );
            barrier.post_dispatch(&[&s.q_buf, kv_k, kv_v], &[&s.attn_out]);
            barrier.step(encoder);
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_attention += attn_t.elapsed();
            }

            let out_proj_t = OpTimer::start();
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
                exec_plan.dequant_dispatch,
            );
            barrier.post_dispatch(&[&s.attn_out], &[&s.proj_buf]);
            barrier.step(encoder);
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_out_proj += out_proj_t.elapsed();
            }

            let residual_t = OpTimer::start();
            barrier.pre_dispatch(&[hidden_buf, &s.proj_buf], &[hidden_buf]);
            metal_ops
                .elementwise
                .encode_elementwise_add(encoder, hidden_buf, &s.proj_buf, dim as u32);
            barrier.post_dispatch(&[hidden_buf, &s.proj_buf], &[hidden_buf]);
            barrier.step(encoder);
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_residual += residual_t.elapsed();
            }

            if run_moe {
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
                let layer_gate_dtype = lw.moe_expert_gate_dtype.unwrap_or(gate_dtype);
                let layer_up_dtype = lw.moe_expert_up_dtype.unwrap_or(up_dtype);
                let layer_down_dtype = lw.moe_expert_down_dtype.unwrap_or(down_dtype);
                let layer_gate_stride = lw.moe_expert_gate_stride.unwrap_or(gate_stride);
                let layer_up_stride = lw.moe_expert_up_stride.unwrap_or(up_stride);
                let layer_down_stride = lw.moe_expert_down_stride.unwrap_or(down_stride);
                let moe_t = OpTimer::start();
                // Resident MoE manages its own internal hazards, but the outer
                // decode barrier still needs to see the overall hidden->hidden
                // dependency so the residual write before it and the next layer
                // after it are synchronized correctly.
                barrier.pre_dispatch(&[hidden_buf], &[hidden_buf]);
                metal_ops.encode_moe_ffn_gpu_resident_cached_with_policy(
                    encoder,
                    hidden_buf,
                    ffn_norm_w,
                    router_buf,
                    lw.moe_router_dtype.unwrap(),
                    moe_gate_buf,
                    layer_gate_dtype,
                    moe_up_buf,
                    layer_up_dtype,
                    moe_down_buf,
                    layer_down_dtype,
                    1,
                    n_expert,
                    n_expert_used,
                    dim,
                    expert_inter_dim,
                    layer_gate_stride,
                    layer_up_stride,
                    layer_down_stride,
                    eps,
                    None,
                    exec_plan.barriers
                        != crate::model::execution_plan::DecodeBarrierPlan::Implicit,
                    Self::qwen3moe_blocked_q6q8_down_enabled(),
                )?;
                barrier.post_dispatch(&[hidden_buf], &[hidden_buf]);
                barrier.step(encoder);
                if let Some(ops_ref) = ops.as_deref_mut() {
                    ops_ref.gpu_encode_layer_ffn += moe_t.elapsed();
                }
            }
        }

        Ok(())
    }

    fn encode_qwen3moe_pending_gpu_unified_step(
        metal_ops: &MetalOps,
        cfg: &ModelConfig,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        position: usize,
        gpu_kv: &mut crate::kv::GpuKv,
        weights: &WeightStore,
        fuse_argmax: bool,
    ) -> anyhow::Result<ax_engine_metal::PendingFrame> {
        let dim = cfg.embedding_dim as usize;
        let vocab_size = cfg.vocab_size as usize;

        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, 1);
        gpu_kv.ensure_capacity(&metal_ops.device, position.saturating_add(1))?;

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys(metal_ops, weights, cfg)?;
        }

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();

        let (gate_dtype, up_dtype, down_dtype) =
            crate::model::shared::routed_moe_expert_dtypes(weights, "blk.0")?;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
        let gate_stride =
            crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
        let up_stride =
            crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
        let exec_plan = Self::qwen3moe_decode_plan(
            metal_ops,
            gpu_kv,
            cfg.embedding_dim,
            cfg.head_dim,
            position.saturating_add(1),
            true,
            gate_dtype,
            up_dtype,
            down_dtype,
        );

        let encode_body = |encoder: &ax_engine_metal::MetalEncoder| -> anyhow::Result<()> {
            let barrier =
                crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
            barrier.pre_dispatch(&[hidden_buf], &[&s.hidden]);
            metal_ops.elementwise.encode_buffer_copy(
                encoder,
                hidden_buf,
                0,
                &s.hidden,
                0,
                dim as u32,
            );
            barrier.post_dispatch(&[hidden_buf], &[&s.hidden]);
            barrier.step(encoder);

            Self::encode_qwen3moe_gpu_layer_range(
                encoder,
                &barrier,
                metal_ops,
                cfg,
                &s.hidden,
                s,
                gpu_kv,
                cached,
                &weight_cache,
                &moe_weight_cache,
                &exec_plan,
                position,
                0,
                cfg.n_layers as usize,
                gate_dtype,
                up_dtype,
                down_dtype,
                gate_stride,
                up_stride,
                down_stride,
                &mut None,
            )?;
            crate::model::shared::encode_gpu_output_head(
                encoder,
                metal_ops,
                s,
                &s.hidden,
                &exec_plan,
                cached,
                &weight_cache,
                &barrier,
                dim as u32,
                vocab_size as u32,
                cfg.rms_norm_eps,
            );
            barrier.flush();
            if fuse_argmax {
                metal_ops.elementwise.encode_argmax_f32(
                    encoder,
                    &s.logits_buf,
                    &s.argmax_idx,
                    &s.argmax_val,
                    cfg.vocab_size,
                );
            }
            Ok(())
        };

        if exec_plan.encoder == crate::model::execution_plan::DecodeEncoderPlan::Concurrent {
            metal_ops.device.encode_frame_concurrent(encode_body)
        } else {
            metal_ops.device.encode_frame(encode_body)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_single_gpu_unified(
        &self, ctx: &ForwardContext, metal_ops: &MetalOps, token_id: u32, position: usize,
        gpu_kv: &mut crate::kv::GpuKv, weights: &WeightStore, logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let total_t = OpTimer::start();
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
        let eps = cfg.rms_norm_eps;
        anyhow::ensure!(logits.len() >= vocab_size, "logits buffer too small");

        let setup_t = OpTimer::start();
        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, 1); // MoE FFN needs batch scratch even for n=1
        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        gpu_kv.ensure_capacity(&metal_ops.device, position.saturating_add(1))?;

        {
            let hidden_cpu = unsafe { std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim) };
            weights.dequantize_row("token_embd.weight", token_id as usize, hidden_cpu)?;
        }

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys(metal_ops, weights, cfg)?;
        }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();

        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();

        let (gate_dtype, up_dtype, down_dtype) =
            crate::model::shared::routed_moe_expert_dtypes(weights, "blk.0")?;
        let gate_stride =
            crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
        let up_stride =
            crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
        let exec_plan = Self::qwen3moe_decode_plan(
            metal_ops,
            gpu_kv,
            cfg.embedding_dim,
            cfg.head_dim,
            position.saturating_add(1),
            false,
            gate_dtype,
            up_dtype,
            down_dtype,
        );
        let layers_per_command_buffer =
            Self::qwen3moe_decode_layers_per_command_buffer(
                cfg.n_layers as usize,
                gate_dtype,
                up_dtype,
                down_dtype,
            );
        let split_moe_command_buffers =
            Self::qwen3moe_split_moe_decode_command_buffers(
                gate_dtype,
                up_dtype,
                down_dtype,
            );
        if let Some(ops_ref) = ops.as_deref_mut() {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        // Keep routed-MoE decode coalesced to minimize submission overhead.
        // The policy above uses the full layer stack for supported quantized
        // layouts, while the env override can still split ranges for A/B runs
        // or fallback if a model/device combination needs it.
        macro_rules! exec_sync {
            ($body:expr) => {
                if exec_plan.encoder == crate::model::execution_plan::DecodeEncoderPlan::Concurrent
                {
                    metal_ops.device.execute_sync_concurrent($body)
                } else {
                    metal_ops.device.execute_sync($body)
                }
            };
        }
        if split_moe_command_buffers {
            for layer in 0..cfg.n_layers as usize {
                let exec_t = OpTimer::start();
                exec_sync!(|encoder| {
                    let barrier =
                        crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
                    Self::encode_qwen3moe_gpu_layer_range_internal(
                        encoder,
                        &barrier,
                        metal_ops,
                        cfg,
                        &s.hidden,
                        s,
                        gpu_kv,
                        cached,
                        &weight_cache,
                        &moe_weight_cache,
                        &exec_plan,
                        position,
                        layer,
                        layer + 1,
                        gate_dtype,
                        up_dtype,
                        down_dtype,
                        gate_stride,
                        up_stride,
                        down_stride,
                        &mut ops,
                        false,
                    )?;
                    barrier.flush();
                    Ok(())
                })?;
                if let Some(ops_ref) = ops.as_deref_mut() {
                    let elapsed = exec_t.elapsed();
                    ops_ref.gpu_execute += elapsed;
                    ops_ref.gpu_execute_layers += elapsed;
                }

                let lw = &cached.layers[layer];
                let ffn_norm_w = weight_cache.get(&lw.ffn_norm).unwrap();
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
                let layer_gate_dtype = lw.moe_expert_gate_dtype.unwrap_or(gate_dtype);
                let layer_up_dtype = lw.moe_expert_up_dtype.unwrap_or(up_dtype);
                let layer_down_dtype = lw.moe_expert_down_dtype.unwrap_or(down_dtype);
                let layer_gate_stride = lw.moe_expert_gate_stride.unwrap_or(gate_stride);
                let layer_up_stride = lw.moe_expert_up_stride.unwrap_or(up_stride);
                let layer_down_stride = lw.moe_expert_down_stride.unwrap_or(down_stride);
                let moe_exec_t = OpTimer::start();
                exec_sync!(|encoder| {
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_policy(
                        encoder,
                        &s.hidden,
                        ffn_norm_w,
                        router_buf,
                        lw.moe_router_dtype.unwrap(),
                        moe_gate_buf,
                        layer_gate_dtype,
                        moe_up_buf,
                        layer_up_dtype,
                        moe_down_buf,
                        layer_down_dtype,
                        1,
                        cfg.n_expert.unwrap_or(0) as usize,
                        cfg.n_expert_used.unwrap_or(0) as usize,
                        dim,
                        expert_inter_dim,
                        layer_gate_stride,
                        layer_up_stride,
                        layer_down_stride,
                        eps,
                        None,
                        true,
                        Self::qwen3moe_blocked_q6q8_down_enabled(),
                    )
                })?;
                if let Some(ops_ref) = ops.as_deref_mut() {
                    let elapsed = moe_exec_t.elapsed();
                    ops_ref.gpu_execute += elapsed;
                    ops_ref.gpu_execute_layers += elapsed;
                    ops_ref.gpu_encode_layer_ffn += elapsed;
                }
                let hidden_gpu = unsafe {
                    std::slice::from_raw_parts(s.hidden.contents().as_ptr() as *const f32, dim)
                };
                if std::env::var("AX_DEBUG_QWEN3_Q5_RUNTIME_MOE").is_ok()
                    && hidden_gpu.iter().any(|value| !value.is_finite())
                    && let Some(summary) = Self::qwen3moe_single_token_moe_scratch_summary(
                        metal_ops,
                        dim,
                        cfg.n_expert_used.unwrap_or(0) as usize,
                        expert_inter_dim,
                    )
                {
                    eprintln!(
                        "[QWEN3MOE-RUNTIME] layer={layer} position={position} {summary}"
                    );
                }
                Self::assert_finite_if_enabled("layer_hidden", hidden_gpu, layer, position)?;
            }
        } else {
            for layer_start in (0..cfg.n_layers as usize).step_by(layers_per_command_buffer) {
                let layer_end =
                    (layer_start + layers_per_command_buffer).min(cfg.n_layers as usize);
                let exec_t = OpTimer::start();
                exec_sync!(|encoder| {
                    let barrier =
                        crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
                    Self::encode_qwen3moe_gpu_layer_range(
                        encoder,
                        &barrier,
                        metal_ops,
                        cfg,
                        &s.hidden,
                        s,
                        gpu_kv,
                        cached,
                        &weight_cache,
                        &moe_weight_cache,
                        &exec_plan,
                        position,
                        layer_start,
                        layer_end,
                        gate_dtype,
                        up_dtype,
                        down_dtype,
                        gate_stride,
                        up_stride,
                        down_stride,
                        &mut ops,
                    )?;
                    barrier.flush();
                    Ok(())
                })?;
                if let Some(ops_ref) = ops.as_deref_mut() {
                    let elapsed = exec_t.elapsed();
                    ops_ref.gpu_execute += elapsed;
                    ops_ref.gpu_execute_layers += elapsed;
                }
                let hidden_gpu = unsafe {
                    std::slice::from_raw_parts(s.hidden.contents().as_ptr() as *const f32, dim)
                };
                Self::assert_finite_if_enabled(
                    "layer_hidden",
                    hidden_gpu,
                    layer_end.saturating_sub(1),
                    position,
                )?;
            }
        }

        // (debug removed)
        // Output head in its own CB
        let output_exec_t = OpTimer::start();
        exec_sync!(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
            crate::model::shared::encode_gpu_output_head(
                encoder,
                metal_ops,
                s,
                &s.hidden,
                &exec_plan,
                cached,
                &weight_cache,
                &barrier,
                dim as u32,
                vocab_size as u32,
                eps,
            );
            barrier.flush();
            Ok(())
        })?;
        if let Some(ops_ref) = ops.as_deref_mut() {
            let elapsed = output_exec_t.elapsed();
            ops_ref.gpu_execute += elapsed;
            ops_ref.gpu_execute_output += elapsed;
        }

        let readback_t = OpTimer::start();
        let logits_gpu = unsafe { std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab_size) };
        Self::assert_finite_if_enabled("logits", logits_gpu, cfg.n_layers as usize, position)?;
        logits[..vocab_size].copy_from_slice(logits_gpu);
        gpu_kv.finalize_batch(1);
        if let Some(ops_ref) = ops {
            ops_ref.gpu_readback += readback_t.elapsed();
            ops_ref.gpu += total_t.elapsed();
        }
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
        let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

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
        let concurrent_prefill =
            Self::qwen3moe_prefill_concurrent_enabled(gate_dtype, up_dtype, down_dtype);

        let encode_layer_range = |
            encoder: &ax_engine_metal::MetalEncoder,
            layer_range: std::ops::Range<usize>,
            include_output_head: bool,
        | -> anyhow::Result<()> {
            let mut sb = ax_engine_metal::SmartBarrier::new(encoder);
            macro_rules! sb_pre {
                ($reads:expr, $writes:expr) => {
                    sb.pre_dispatch($reads, $writes);
                };
            }
            macro_rules! sb_post {
                ($reads:expr, $writes:expr) => {
                    sb.post_dispatch($reads, $writes);
                };
            }
            macro_rules! sb_flush {
                () => {
                    sb.flush();
                };
            }

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
                let fused_qkv_m = q_dim + 2 * kv_dim;
                let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
                let fused_qkv_buf = if prefill_plan.use_fused_qkv
                    && lw.wq_dtype == lw.wk_dtype
                    && lw.wq_dtype == lw.wv_dtype
                    && matches!(
                        lw.wq_dtype,
                        crate::gguf::tensor::GgmlType::Q4K
                            | crate::gguf::tensor::GgmlType::Q6K
                    ) {
                    fused_qkv_cache.get(&fused_qkv_key)
                } else {
                    None
                };

                // ── Attention norm (batch) ──
                sb_pre!(&[&bs.hidden], &[&bs.norm_buf]);
                metal_ops.elementwise.encode_rms_norm_out_batch(
                    encoder, &bs.hidden, attn_norm_w, &bs.norm_buf,
                    dim as u32, nt, eps,
                );
                sb_post!(&[&bs.hidden], &[&bs.norm_buf]);

                // ── Q/K/V batch projections ──
                let use_f16 = prefill_plan.use_f16_batch_io;
                if let Some(fused_w) = fused_qkv_buf {
                    if use_f16 {
                        sb_pre!(&[&bs.norm_buf], &[&bs.matmul_in_f16]);
                        metal_ops.elementwise.encode_cast_f32_to_f16(
                            encoder, &bs.norm_buf, &bs.matmul_in_f16, nt * dim as u32,
                        );
                        sb_post!(&[&bs.norm_buf], &[&bs.matmul_in_f16]);
                        sb_pre!(&[&bs.matmul_in_f16], &[&bs.qkv_buf]);
                        crate::model::shared::encode_dequant_batch_f16in(
                            metal_ops, encoder, fused_w, &bs.matmul_in_f16, &bs.qkv_buf,
                            fused_qkv_m as u32, nt, dim as u32, lw.wq_dtype,
                        );
                        sb_post!(&[&bs.matmul_in_f16], &[&bs.qkv_buf]);
                    } else {
                        sb_pre!(&[&bs.norm_buf], &[&bs.qkv_buf]);
                        crate::model::shared::encode_dequant_batch(
                            &metal_ops.dequant, &metal_ops.elementwise,
                            encoder, fused_w, &bs.norm_buf, &bs.qkv_buf, &bs.matmul_in_f16,
                            fused_qkv_m as u32, nt, dim as u32, lw.wq_dtype,
                            false, prefill_plan.use_batch_simd, prefill_plan.q5k_prefill_small_n,
                        );
                        sb_post!(&[&bs.norm_buf], &[&bs.qkv_buf]);
                    }
                } else if use_f16 {
                    sb_pre!(&[&bs.norm_buf], &[&bs.matmul_in_f16]);
                    metal_ops.elementwise.encode_cast_f32_to_f16(
                        encoder, &bs.norm_buf, &bs.matmul_in_f16, nt * dim as u32,
                    );
                    sb_post!(&[&bs.norm_buf], &[&bs.matmul_in_f16]);
                    sb_pre!(&[&bs.matmul_in_f16], &[&bs.q_buf]);
                    crate::model::shared::encode_dequant_batch_f16in(
                        metal_ops, encoder, wq_buf, &bs.matmul_in_f16, &bs.q_buf,
                        q_dim as u32, nt, dim as u32, lw.wq_dtype,
                    );
                    sb_post!(&[&bs.matmul_in_f16], &[&bs.q_buf]);
                    sb_pre!(&[&bs.matmul_in_f16], &[&bs.k_buf]);
                    crate::model::shared::encode_dequant_batch_f16in(
                        metal_ops, encoder, wk_buf, &bs.matmul_in_f16, &bs.k_buf,
                        kv_dim as u32, nt, dim as u32, lw.wk_dtype,
                    );
                    sb_post!(&[&bs.matmul_in_f16], &[&bs.k_buf]);
                    sb_pre!(&[&bs.matmul_in_f16], &[&bs.v_buf]);
                    crate::model::shared::encode_dequant_batch_f16in(
                        metal_ops, encoder, wv_buf, &bs.matmul_in_f16, &bs.v_buf,
                        kv_dim as u32, nt, dim as u32, lw.wv_dtype,
                    );
                    sb_post!(&[&bs.matmul_in_f16], &[&bs.v_buf]);
                } else {
                    sb_pre!(&[&bs.norm_buf], &[&bs.q_buf]);
                    crate::model::shared::encode_dequant_batch(
                        &metal_ops.dequant, &metal_ops.elementwise,
                        encoder, wq_buf, &bs.norm_buf, &bs.q_buf, &bs.matmul_in_f16,
                        q_dim as u32, nt, dim as u32, lw.wq_dtype,
                        false, prefill_plan.use_batch_simd, prefill_plan.q5k_prefill_small_n,
                    );
                    sb_post!(&[&bs.norm_buf], &[&bs.q_buf]);
                    sb_pre!(&[&bs.norm_buf], &[&bs.k_buf]);
                    crate::model::shared::encode_dequant_batch(
                        &metal_ops.dequant, &metal_ops.elementwise,
                        encoder, wk_buf, &bs.norm_buf, &bs.k_buf, &bs.matmul_in_f16,
                        kv_dim as u32, nt, dim as u32, lw.wk_dtype,
                        false, prefill_plan.use_batch_simd, prefill_plan.q5k_prefill_small_n,
                    );
                    sb_post!(&[&bs.norm_buf], &[&bs.k_buf]);
                    sb_pre!(&[&bs.norm_buf], &[&bs.v_buf]);
                    crate::model::shared::encode_dequant_batch(
                        &metal_ops.dequant, &metal_ops.elementwise,
                        encoder, wv_buf, &bs.norm_buf, &bs.v_buf, &bs.matmul_in_f16,
                        kv_dim as u32, nt, dim as u32, lw.wv_dtype,
                        false, prefill_plan.use_batch_simd, prefill_plan.q5k_prefill_small_n,
                    );
                    sb_post!(&[&bs.norm_buf], &[&bs.v_buf]);
                }

                let fused_qkv_post = fused_qkv_buf.is_some()
                    && lw.attn_q_norm.is_some()
                    && lw.attn_k_norm.is_some()
                    && !prefill_plan.kv_q8;

                // ── QK norm (batch) + RoPE + KV append ──
                if fused_qkv_post {
                    let qn = weight_cache.get(&lw.attn_q_norm.unwrap()).unwrap();
                    let kn = weight_cache.get(&lw.attn_k_norm.unwrap()).unwrap();
                    sb_pre!(
                        &[&bs.qkv_buf],
                        &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v]
                    );
                    metal_ops
                        .elementwise
                        .encode_qkv_split_qk_norm_rope_append_kv_batch(
                            encoder,
                            &bs.qkv_buf,
                            &bs.q_buf,
                            &bs.k_buf,
                            &bs.v_buf,
                            qn,
                            kn,
                            kv_k,
                            kv_v,
                            prefill_plan.kv_f16,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            eps,
                            base_seq_len as f32,
                            1.0,
                            cfg.rope_freq_base,
                            kv_offset,
                            kv_stride,
                        );
                    sb_post!(
                        &[&bs.qkv_buf],
                        &[&bs.q_buf, &bs.k_buf, &bs.v_buf, kv_k, kv_v]
                    );
                } else {
                    if fused_qkv_buf.is_some() {
                        sb_pre!(&[&bs.qkv_buf], &[&bs.q_buf, &bs.k_buf, &bs.v_buf]);
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
                        sb_post!(&[&bs.qkv_buf], &[&bs.q_buf, &bs.k_buf, &bs.v_buf]);
                    }

                    if let (Some(qn_key), Some(kn_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
                        let qn = weight_cache.get(&qn_key).unwrap();
                        let kn = weight_cache.get(&kn_key).unwrap();
                        sb_pre!(&[&bs.q_buf], &[&bs.q_buf]);
                        metal_ops.elementwise.encode_per_head_rms_norm_batch(
                            encoder,
                            &bs.q_buf,
                            qn,
                            nt,
                            n_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        sb_post!(&[&bs.q_buf], &[&bs.q_buf]);
                        sb_pre!(&[&bs.k_buf], &[&bs.k_buf]);
                        metal_ops.elementwise.encode_per_head_rms_norm_batch(
                            encoder,
                            &bs.k_buf,
                            kn,
                            nt,
                            n_kv_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        sb_post!(&[&bs.k_buf], &[&bs.k_buf]);
                    }

                    sb_pre!(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);
                    metal_ops.elementwise.encode_rope_batch_neox_partial(
                        encoder, &bs.q_buf, &bs.k_buf,
                        nt, n_heads as u32, n_kv_heads as u32,
                        head_dim as u32, head_dim as u32,
                        base_seq_len as f32, 1.0, cfg.rope_freq_base,
                    );
                    sb_post!(&[&bs.q_buf, &bs.k_buf], &[&bs.q_buf, &bs.k_buf]);

                    sb_pre!(&[&bs.k_buf, &bs.v_buf], &[kv_k, kv_v]);
                    metal_ops.elementwise.encode_kv_append_batch_pair(
                        encoder, &bs.k_buf, &bs.v_buf, kv_k, kv_v,
                        prefill_plan.kv_f16, kv_offset, kv_stride,
                        kv_dim as u32, nt,
                    );
                    sb_post!(&[&bs.k_buf, &bs.v_buf], &[kv_k, kv_v]);
                }

                // ── Prefill attention ──
                sb_pre!(&[&bs.q_buf, kv_k, kv_v], &[&bs.attn_out]);
                metal_ops.attention.encode_attention_prefill_cached_with_config(
                    encoder, &bs.q_buf, kv_k, kv_v, &bs.attn_out,
                    prefill_plan.kv_f16, nt,
                    n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    base_seq_len as u32, 0,
                    prefill_plan.attention_dispatch,
                );
                sb_post!(&[&bs.q_buf, kv_k, kv_v], &[&bs.attn_out]);

                // ── Output projection (batch) ──
                sb_pre!(&[&bs.attn_out], &[&bs.proj_buf]);
                crate::model::shared::encode_dequant_batch(
                    &metal_ops.dequant, &metal_ops.elementwise,
                    encoder, wo_buf, &bs.attn_out, &bs.proj_buf, &bs.matmul_in_f16,
                    dim as u32, nt, q_dim as u32, lw.wo_dtype,
                    prefill_plan.use_f16_batch_io, prefill_plan.use_batch_simd,
                    prefill_plan.q5k_prefill_small_n,
                );
                sb_post!(&[&bs.attn_out], &[&bs.proj_buf]);

                // ── Residual add (batch) ──
                sb_pre!(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);
                metal_ops.elementwise.encode_elementwise_add_batch(
                    encoder, &bs.hidden, &bs.proj_buf, dim as u32, nt,
                );
                sb_post!(&[&bs.hidden, &bs.proj_buf], &[&bs.hidden]);

                // ── MoE FFN (batch, GPU resident) ──
                let router_buf = moe_weight_cache.get(&lw.moe_router.unwrap()).unwrap();
                let moe_gate_buf = moe_weight_cache
                    .get(&lw.moe_expert_gate.as_ref().unwrap()[0]).unwrap();
                let moe_up_buf = moe_weight_cache
                    .get(&lw.moe_expert_up.as_ref().unwrap()[0]).unwrap();
                let moe_down_buf = moe_weight_cache
                    .get(&lw.moe_expert_down.as_ref().unwrap()[0]).unwrap();
                let layer_gate_dtype = lw.moe_expert_gate_dtype.unwrap_or(gate_dtype);
                let layer_up_dtype = lw.moe_expert_up_dtype.unwrap_or(up_dtype);
                let layer_down_dtype = lw.moe_expert_down_dtype.unwrap_or(down_dtype);
                let layer_gate_stride = lw.moe_expert_gate_stride.unwrap_or(gate_stride);
                let layer_up_stride = lw.moe_expert_up_stride.unwrap_or(up_stride);
                let layer_down_stride = lw.moe_expert_down_stride.unwrap_or(down_stride);

                {
                    let moe_scratch = crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs)?;
                    // Resident MoE uses its own SmartBarrier internally.
                    // Record the hidden->hidden boundary in the outer tracker
                    // so subsequent layer dispatches see the writeback.
                    sb_pre!(&[&bs.hidden], &[&bs.hidden]);
                    ax_engine_metal::barrier_buffers(encoder);
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
                        encoder, moe_scratch, &bs.hidden, ffn_norm_w,
                        router_buf, lw.moe_router_dtype.unwrap(),
                        moe_gate_buf, layer_gate_dtype,
                        moe_up_buf, layer_up_dtype,
                        moe_down_buf, layer_down_dtype,
                        n_tokens, n_expert, n_expert_used,
                        dim, expert_inter_dim,
                        layer_gate_stride, layer_up_stride, layer_down_stride,
                        eps, None, true,
                        Self::qwen3moe_blocked_q6q8_down_enabled(),
                    )?;
                    ax_engine_metal::barrier_buffers(encoder);
                    sb_post!(&[&bs.hidden], &[&bs.hidden]);
                }
            }

            if include_output_head {
                // ── Output head (last token) ──
                let last_off = (n_tokens - 1) * dim * std::mem::size_of::<f32>();
                sb_pre!(&[&bs.hidden], &[&s.hidden]);
                metal_ops.elementwise.encode_buffer_copy(
                    encoder, &bs.hidden, last_off, &s.hidden, 0, dim as u32,
                );
                sb_post!(&[&bs.hidden], &[&s.hidden]);

                let output_norm_w = weight_cache.get(&cached.output_norm).unwrap();
                sb_pre!(&[&s.hidden], &[&s.hidden]);
                metal_ops.elementwise.encode_rms_norm(
                    encoder, &s.hidden, output_norm_w, dim as u32, eps,
                );
                sb_post!(&[&s.hidden], &[&s.hidden]);

                let lm_head_buf = weight_cache.get(&cached.lm_head).unwrap();
                sb_pre!(&[&s.hidden], &[&s.logits_buf]);
                crate::model::shared::encode_dequant_matvec(
                    metal_ops, encoder, lm_head_buf, &s.hidden, &s.logits_buf,
                    vocab_size as u32, dim as u32, cached.lm_head_dtype,
                );
                sb_post!(&[&s.hidden], &[&s.logits_buf]);
            }
            sb_flush!();
            Ok(())
        };

        if split_layer > 0 && split_layer < n_layers {
            let _cb1 = if concurrent_prefill {
                metal_ops.device.execute_async_concurrent(|encoder| {
                    encode_layer_range(encoder, 0..split_layer, false)
                })?
            } else {
                metal_ops
                    .device
                    .execute_async(|encoder| encode_layer_range(encoder, 0..split_layer, false))?
            };
            if concurrent_prefill {
                metal_ops.device.execute_sync_concurrent(|encoder| {
                    encode_layer_range(encoder, split_layer..n_layers, true)
                })?;
            } else {
                metal_ops.device.execute_sync(|encoder| {
                    encode_layer_range(encoder, split_layer..n_layers, true)
                })?;
            }
        } else {
            if concurrent_prefill {
                metal_ops.device.execute_sync_concurrent(|encoder| {
                    encode_layer_range(encoder, 0..n_layers, true)
                })?;
            } else {
                metal_ops
                    .device
                    .execute_sync(|encoder| encode_layer_range(encoder, 0..n_layers, true))?;
            }
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
