#[derive(Clone, Copy)]
pub(crate) struct Qwen3_5NativeRecurrentProjectionScratch<'a> {
    pub(crate) qkv: &'a MetalBuffer,
    pub(crate) z: &'a MetalBuffer,
    pub(crate) beta: &'a MetalBuffer,
    pub(crate) alpha: &'a MetalBuffer,
}

#[derive(Clone, Copy)]
pub(crate) struct Qwen3_5NativeRecurrentCachedWeights<'a> {
    pub(crate) wqkv: &'a MetalBuffer,
    pub(crate) wgate: &'a MetalBuffer,
    pub(crate) wbeta: &'a MetalBuffer,
    pub(crate) walpha: &'a MetalBuffer,
    pub(crate) conv_kernel: &'a MetalBuffer,
    pub(crate) ssm_norm: &'a MetalBuffer,
    pub(crate) dt_bias: &'a MetalBuffer,
    pub(crate) ssm_a: &'a MetalBuffer,
    pub(crate) wssm_out: &'a MetalBuffer,
}

#[derive(Clone, Copy)]
pub(crate) struct Qwen3_5NativeRecurrentDtypes {
    pub(crate) wqkv: crate::gguf::tensor::GgmlType,
    pub(crate) wgate: crate::gguf::tensor::GgmlType,
    pub(crate) wbeta: crate::gguf::tensor::GgmlType,
    pub(crate) walpha: crate::gguf::tensor::GgmlType,
    pub(crate) wssm_out: crate::gguf::tensor::GgmlType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3_5NativeDecodeLayerRange {
    pub(crate) start: usize,
    pub(crate) end_exclusive: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen3_5NativeDecodeDispatchPlan {
    pub(crate) layer_ranges: Vec<Qwen3_5NativeDecodeLayerRange>,
    pub(crate) encoder: crate::model::execution_plan::DecodeEncoderPlan,
    pub(crate) barriers: crate::model::execution_plan::DecodeBarrierPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3_5NativeDecodeLayerRangeStrategy {
    AllLayers,
    RecurrentRuns,
    PerLayer,
}

impl Qwen3_5Forward {
    fn qwen35_should_flush_between_coalesced_native_decode_layers(
        cfg: &ModelConfig,
        current_layer: usize,
        next_layer: usize,
    ) -> bool {
        cfg.qwen35_is_recurrent_layer(current_layer) && cfg.qwen35_is_recurrent_layer(next_layer)
    }

    fn qwen35_native_decode_layer_range_strategy(
        _cfg: &ModelConfig,
    ) -> Qwen3_5NativeDecodeLayerRangeStrategy {
        if let Ok(value) = std::env::var("AX_QWEN35_GPU_DECODE_LAYER_RANGES") {
            match value.trim().to_ascii_lowercase().as_str() {
                "all" | "all_layers" | "single" | "single_range" => {
                    return Qwen3_5NativeDecodeLayerRangeStrategy::AllLayers;
                }
                "recurrent" | "recurrent_runs" | "coalesced" => {
                    return Qwen3_5NativeDecodeLayerRangeStrategy::RecurrentRuns;
                }
                "per_layer" | "layer" | "layers" => {
                    return Qwen3_5NativeDecodeLayerRangeStrategy::PerLayer;
                }
                _ => {}
            }
        }
        if let Some(enabled) = env_flag_override("AX_QWEN35_GPU_DECODE_COALESCE_RECURRENT") {
            return if enabled {
                Qwen3_5NativeDecodeLayerRangeStrategy::RecurrentRuns
            } else {
                Qwen3_5NativeDecodeLayerRangeStrategy::PerLayer
            };
        }
        Qwen3_5NativeDecodeLayerRangeStrategy::AllLayers
    }

    fn qwen35_native_decode_recurrent_run_coalescing_max_layers() -> usize {
        std::env::var("AX_QWEN35_GPU_DECODE_COALESCE_RECURRENT_MAX_LAYERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(usize::MAX)
    }

    pub(crate) fn qwen35_native_decode_layer_ranges(
        cfg: &ModelConfig,
        layer_limit: usize,
    ) -> Vec<Qwen3_5NativeDecodeLayerRange> {
        if layer_limit == 0 {
            return Vec::new();
        }
        match Self::qwen35_native_decode_layer_range_strategy(cfg) {
            Qwen3_5NativeDecodeLayerRangeStrategy::AllLayers => {
                return vec![Qwen3_5NativeDecodeLayerRange {
                    start: 0,
                    end_exclusive: layer_limit,
                }];
            }
            Qwen3_5NativeDecodeLayerRangeStrategy::PerLayer => {
                return (0..layer_limit)
                    .map(|layer| Qwen3_5NativeDecodeLayerRange {
                        start: layer,
                        end_exclusive: layer + 1,
                    })
                    .collect();
            }
            Qwen3_5NativeDecodeLayerRangeStrategy::RecurrentRuns => {}
        }

        if !Self::qwen35_is_moe(cfg) {
            return vec![Qwen3_5NativeDecodeLayerRange {
                start: 0,
                end_exclusive: layer_limit,
            }];
        }

        let mut ranges = Vec::new();
        let max_recurrent_layers_per_range =
            Self::qwen35_native_decode_recurrent_run_coalescing_max_layers();
        let mut layer = 0usize;
        while layer < layer_limit {
            let end_exclusive = if cfg.qwen35_is_recurrent_layer(layer) {
                let mut end = layer + 1;
                while end < layer_limit
                    && cfg.qwen35_is_recurrent_layer(end)
                    && end - layer < max_recurrent_layers_per_range
                {
                    end += 1;
                }
                end
            } else {
                layer + 1
            };
            ranges.push(Qwen3_5NativeDecodeLayerRange {
                start: layer,
                end_exclusive,
            });
            layer = end_exclusive;
        }
        ranges
    }

    pub(crate) fn qwen35_native_decode_dispatch_plan(
        cfg: &ModelConfig,
        exec_plan: &crate::model::execution_plan::GpuDecodeExecutionPlan,
        layer_limit: usize,
    ) -> Qwen3_5NativeDecodeDispatchPlan {
        let layer_ranges = Self::qwen35_native_decode_layer_ranges(cfg, layer_limit);
        let has_coalesced_moe_recurrent_run = Self::qwen35_is_moe(cfg)
            && layer_ranges
                .iter()
                .any(|range| range.end_exclusive > range.start.saturating_add(1));
        let (encoder, barriers) = if has_coalesced_moe_recurrent_run
            && exec_plan.encoder == crate::model::execution_plan::DecodeEncoderPlan::Concurrent
        {
            (
                crate::model::execution_plan::DecodeEncoderPlan::Serial,
                crate::model::execution_plan::DecodeBarrierPlan::Explicit,
            )
        } else {
            (exec_plan.encoder, exec_plan.barriers)
        };
        Qwen3_5NativeDecodeDispatchPlan {
            layer_ranges,
            encoder,
            barriers,
        }
    }

    fn qwen35_single_token_recurrent_config(
        conv_cache_len: usize,
        dims: Qwen3_5RecurrentDims,
        eps: f32,
    ) -> crate::compute::gdn::Qwen35RecurrentConfig {
        crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len,
            conv_dim: dims.conv_dim(),
            group_count: dims.group_count,
            state_size: dims.state_size,
            time_step_rank: dims.time_step_rank,
            rms_norm_eps: eps,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_qwen35_single_token_recurrent_core(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        qkv_projected: &MetalBuffer,
        alpha_prepared: &MetalBuffer,
        beta_prepared: &MetalBuffer,
        conv_kernel: &MetalBuffer,
        slot_buffers: &crate::backend::metal::Qwen35MetalSlotBuffers,
        output: &MetalBuffer,
        conv_cache_len: usize,
        dims: Qwen3_5RecurrentDims,
        eps: f32,
    ) -> anyhow::Result<()> {
        let recurrent_cfg = Self::qwen35_single_token_recurrent_config(conv_cache_len, dims, eps);
        // Qwen3.5 recurrent decode tensors can be larger than the generic
        // attention scratch buffers (for example 35B-A3B), so keep the whole
        // conv -> QKV pack -> gated-delta sequence on the dedicated recurrent
        // scratch path instead of aliasing the decode scratch buffers.
        barrier.pre_dispatch(
            &[
                qkv_projected,
                alpha_prepared,
                beta_prepared,
                conv_kernel,
                &slot_buffers.conv_state,
                &slot_buffers.recurrent_state,
            ],
            &[
                output,
                &slot_buffers.conv_state,
                &slot_buffers.recurrent_state,
            ],
        );
        metal_ops.encode_qwen35_single_token_recurrent_projected(
            encoder,
            qkv_projected,
            alpha_prepared,
            beta_prepared,
            conv_kernel,
            slot_buffers,
            output,
            recurrent_cfg,
        )?;
        barrier.post_dispatch(
            &[
                qkv_projected,
                alpha_prepared,
                beta_prepared,
                conv_kernel,
                &slot_buffers.conv_state,
                &slot_buffers.recurrent_state,
            ],
            &[
                output,
                &slot_buffers.conv_state,
                &slot_buffers.recurrent_state,
            ],
        );
        barrier.step(encoder);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_qwen35_native_recurrent_inputs(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        weights: Qwen3_5NativeRecurrentCachedWeights<'_>,
        scratch: Qwen3_5NativeRecurrentProjectionScratch<'_>,
        norm_buf: &MetalBuffer,
        dims: Qwen3_5RecurrentDims,
        dim: usize,
        dtypes: Qwen3_5NativeRecurrentDtypes,
        dequant_dispatch: ax_engine_metal::DequantDispatchConfig,
    ) {
        crate::model::shared::encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            weights.wqkv,
            norm_buf,
            scratch.qkv,
            dims.conv_dim() as u32,
            dim as u32,
            dtypes.wqkv,
            dequant_dispatch,
        );
        crate::model::shared::encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            weights.wgate,
            norm_buf,
            scratch.z,
            dims.inner_size as u32,
            dim as u32,
            dtypes.wgate,
            dequant_dispatch,
        );
        if !Self::encode_recurrent_pair_matvec_with_config(
            metal_ops,
            encoder,
            weights.wbeta,
            weights.walpha,
            norm_buf,
            scratch.beta,
            scratch.alpha,
            dims.time_step_rank as u32,
            dim as u32,
            dtypes.wbeta,
            dtypes.walpha,
        ) {
            crate::model::shared::encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                weights.wbeta,
                norm_buf,
                scratch.beta,
                dims.time_step_rank as u32,
                dim as u32,
                dtypes.wbeta,
                dequant_dispatch,
            );
            crate::model::shared::encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                weights.walpha,
                norm_buf,
                scratch.alpha,
                dims.time_step_rank as u32,
                dim as u32,
                dtypes.walpha,
                dequant_dispatch,
            );
        }
        barrier.step(encoder);

        metal_ops.elementwise.encode_softplus_bias_mul_sigmoid_pair(
            encoder,
            scratch.alpha,
            scratch.beta,
            weights.dt_bias,
            weights.ssm_a,
            dims.time_step_rank as u32,
        );
        barrier.step(encoder);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_qwen35_native_recurrent_output(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        weights: Qwen3_5NativeRecurrentCachedWeights<'_>,
        scratch: Qwen3_5NativeRecurrentProjectionScratch<'_>,
        slot_buffers: &crate::backend::metal::Qwen35MetalSlotBuffers,
        recurrent_out: &MetalBuffer,
        conv_cache_len: usize,
        dims: Qwen3_5RecurrentDims,
        eps: f32,
    ) -> anyhow::Result<()> {
        Self::encode_qwen35_single_token_recurrent_core(
            metal_ops,
            encoder,
            barrier,
            scratch.qkv,
            scratch.alpha,
            scratch.beta,
            weights.conv_kernel,
            slot_buffers,
            recurrent_out,
            conv_cache_len,
            dims,
            eps,
        )?;

        metal_ops
            .elementwise
            .encode_per_head_rms_norm_silu_mul_batch(
                encoder,
                scratch.z,
                recurrent_out,
                weights.ssm_norm,
                1,
                dims.time_step_rank as u32,
                dims.state_size as u32,
                eps,
            );
        barrier.step(encoder);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_qwen35_native_recurrent_out_proj(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        weights: Qwen3_5NativeRecurrentCachedWeights<'_>,
        gated_recurrent_out: &MetalBuffer,
        proj_out: &MetalBuffer,
        dim: usize,
        dims: Qwen3_5RecurrentDims,
        dtypes: Qwen3_5NativeRecurrentDtypes,
        dequant_dispatch: ax_engine_metal::DequantDispatchConfig,
    ) {
        crate::model::shared::encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            weights.wssm_out,
            gated_recurrent_out,
            proj_out,
            dim as u32,
            dims.inner_size as u32,
            dtypes.wssm_out,
            dequant_dispatch,
        );
        barrier.step(encoder);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_qwen35_native_recurrent_layer(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        weights: Qwen3_5NativeRecurrentCachedWeights<'_>,
        scratch: Qwen3_5NativeRecurrentProjectionScratch<'_>,
        slot_buffers: &crate::backend::metal::Qwen35MetalSlotBuffers,
        norm_buf: &MetalBuffer,
        recurrent_out: &MetalBuffer,
        proj_out: &MetalBuffer,
        conv_cache_len: usize,
        dims: Qwen3_5RecurrentDims,
        dim: usize,
        eps: f32,
        dtypes: Qwen3_5NativeRecurrentDtypes,
        dequant_dispatch: ax_engine_metal::DequantDispatchConfig,
    ) -> anyhow::Result<()> {
        barrier.pre_dispatch(
            &[norm_buf],
            &[scratch.qkv, scratch.z, scratch.beta, scratch.alpha],
        );
        Self::encode_qwen35_native_recurrent_inputs(
            metal_ops,
            encoder,
            barrier,
            weights,
            scratch,
            norm_buf,
            dims,
            dim,
            dtypes,
            dequant_dispatch,
        );
        barrier.post_dispatch(
            &[norm_buf],
            &[scratch.qkv, scratch.z, scratch.beta, scratch.alpha],
        );

        Self::encode_qwen35_native_recurrent_output(
            metal_ops,
            encoder,
            barrier,
            weights,
            scratch,
            slot_buffers,
            recurrent_out,
            conv_cache_len,
            dims,
            eps,
        )?;

        // The recurrent finalize kernel writes SiLU(z) * recurrent_out back into
        // `scratch.z` because the Metal elementwise kernel stores results in its
        // first buffer argument. Keep the out-projection source explicit here so
        // callers do not need to re-encode that contract at each call site.
        barrier.pre_dispatch(&[scratch.z], &[proj_out]);
        Self::encode_qwen35_native_recurrent_out_proj(
            metal_ops,
            encoder,
            barrier,
            weights,
            scratch.z,
            proj_out,
            dim,
            dims,
            dtypes,
            dequant_dispatch,
        );
        barrier.post_dispatch(&[scratch.z], &[proj_out]);
        Ok(())
    }

    fn encode_qwen35_pending_step(
        metal_ops: &crate::backend::metal::MetalOps,
        cfg: &ModelConfig,
        hidden_buf: &MetalBuffer,
        position: usize,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        weights: &WeightStore,
    ) -> anyhow::Result<ax_engine_metal::PendingFrame> {
        Self::encode_qwen35_pending_step_inner(
            metal_ops, cfg, hidden_buf, position, qwen_kv, weights, false,
        )
    }

    fn encode_qwen35_pending_step_with_argmax(
        metal_ops: &crate::backend::metal::MetalOps,
        cfg: &ModelConfig,
        hidden_buf: &MetalBuffer,
        position: usize,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        weights: &WeightStore,
    ) -> anyhow::Result<ax_engine_metal::PendingFrame> {
        Self::encode_qwen35_pending_step_inner(
            metal_ops, cfg, hidden_buf, position, qwen_kv, weights, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_qwen35_single_token_kv_append(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        gpu_attn: &crate::kv::GpuKv,
        layer: usize,
        token_index: usize,
        kv_dim: usize,
        k_src: &MetalBuffer,
        v_src: &MetalBuffer,
    ) -> anyhow::Result<()> {
        let kv_k = gpu_attn.k_buffer(layer);
        let kv_v = gpu_attn.v_buffer(layer);
        if gpu_attn.is_q8() {
            anyhow::ensure!(
                kv_dim.is_multiple_of(crate::kv::gpu_kv::Q8_0_BLOCK_VALUES),
                "qwen35 q8 kv append requires kv_dim multiple of {} (got {kv_dim})",
                crate::kv::gpu_kv::Q8_0_BLOCK_VALUES
            );
            let blocks_per_row = kv_dim / crate::kv::gpu_kv::Q8_0_BLOCK_VALUES;
            let row_offset = token_index
                .checked_mul(blocks_per_row)
                .ok_or_else(|| anyhow::anyhow!("qwen35 q8 kv append row offset overflow"))?;
            let row_offset = u32::try_from(row_offset)
                .map_err(|_| anyhow::anyhow!("qwen35 q8 kv append row offset exceeds u32"))?;
            let blocks_per_row = u32::try_from(blocks_per_row)
                .map_err(|_| anyhow::anyhow!("qwen35 q8 kv append blocks_per_row exceeds u32"))?;
            let kv_dim = u32::try_from(kv_dim)
                .map_err(|_| anyhow::anyhow!("qwen35 q8 kv append kv_dim exceeds u32"))?;
            barrier.pre_dispatch(&[k_src, v_src], &[kv_k, kv_v]);
            metal_ops.elementwise.encode_kv_append_batch_pair_q8(
                encoder,
                k_src,
                v_src,
                kv_k,
                kv_v,
                row_offset,
                blocks_per_row,
                kv_dim,
                1,
            );
            barrier.post_dispatch(&[k_src, v_src], &[kv_k, kv_v]);
        } else {
            let kv_offset = token_index
                .checked_mul(kv_dim)
                .ok_or_else(|| anyhow::anyhow!("qwen35 kv append offset overflow"))?;
            let kv_offset = u32::try_from(kv_offset)
                .map_err(|_| anyhow::anyhow!("qwen35 kv append offset exceeds u32"))?;
            let kv_dim = u32::try_from(kv_dim)
                .map_err(|_| anyhow::anyhow!("qwen35 kv append kv_dim exceeds u32"))?;
            barrier.pre_dispatch(&[k_src], &[kv_k]);
            metal_ops.elementwise.encode_kv_append(
                encoder,
                k_src,
                kv_k,
                gpu_attn.is_f16(),
                kv_offset,
                kv_dim,
            );
            barrier.post_dispatch(&[k_src], &[kv_k]);
            barrier.pre_dispatch(&[v_src], &[kv_v]);
            metal_ops.elementwise.encode_kv_append(
                encoder,
                v_src,
                kv_v,
                gpu_attn.is_f16(),
                kv_offset,
                kv_dim,
            );
            barrier.post_dispatch(&[v_src], &[kv_v]);
        }
        barrier.step(encoder);
        Ok(())
    }

    fn qwen35_native_cached_shared_expert<'a>(
        moe_keys: &Qwen3_5MoeResidentLayerKeys,
        moe_weight_cache: &'a rustc_hash::FxHashMap<usize, MetalBuffer>,
    ) -> Option<crate::backend::metal::SharedExpertCachedBuffers<'a>> {
        moe_keys
            .shared_expert
            .map(|shared| crate::backend::metal::SharedExpertCachedBuffers {
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
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_qwen35_native_post_attention_ffn(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        scratch: &crate::backend::metal::GpuScratchBuffers,
        layer_keys: &crate::backend::metal::CachedLayerKeys,
        moe_keys: Option<&Qwen3_5MoeResidentLayerKeys>,
        weight_cache: &rustc_hash::FxHashMap<usize, MetalBuffer>,
        moe_weight_cache: Option<&rustc_hash::FxHashMap<usize, MetalBuffer>>,
        moe_scratch: Option<crate::backend::metal::MoeBatchScratchView>,
        exec_plan: &crate::model::execution_plan::GpuDecodeExecutionPlan,
        explicit_barriers: bool,
        dim: usize,
        inter_dim: usize,
        eps: f32,
        ops: &mut Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        use crate::model::shared::encode_dequant_matvec_with_config;

        let residual_norm_t = OpTimer::start();
        if Self::debug_skip_ffn() {
            metal_ops.elementwise.encode_elementwise_add_batch(
                encoder,
                &scratch.hidden,
                &scratch.proj_buf,
                dim as u32,
                1,
            );
            barrier.step(encoder);
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_residual += residual_norm_t.elapsed();
            }
            return Ok(());
        }

        if let Some(moe_keys) = moe_keys {
            metal_ops.elementwise.encode_elementwise_add_batch(
                encoder,
                &scratch.hidden,
                &scratch.proj_buf,
                dim as u32,
                1,
            );
            barrier.step(encoder);
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_residual += residual_norm_t.elapsed();
            }

            let moe_t = OpTimer::start();
            let moe_cache = moe_weight_cache
                .ok_or_else(|| anyhow::anyhow!("missing qwen35 native decode MoE weight cache"))?;
            let moe_scratch = moe_scratch
                .ok_or_else(|| anyhow::anyhow!("missing qwen35 native decode MoE scratch"))?;
            let shared_expert = Self::qwen35_native_cached_shared_expert(moe_keys, moe_cache);
            barrier.pre_dispatch(&[&scratch.hidden], &[&scratch.hidden]);
            metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                encoder,
                moe_scratch,
                &scratch.hidden,
                weight_cache.get(&layer_keys.ffn_norm).unwrap(),
                moe_cache.get(&moe_keys.router).unwrap(),
                moe_keys.router_dtype,
                moe_cache.get(&moe_keys.gate).unwrap(),
                moe_keys.gate_dtype,
                moe_cache.get(&moe_keys.up).unwrap(),
                moe_keys.up_dtype,
                moe_cache.get(&moe_keys.down).unwrap(),
                moe_keys.down_dtype,
                1,
                moe_keys.n_expert,
                moe_keys.n_expert_used,
                dim,
                moe_keys.expert_inter_dim,
                moe_keys.gate_stride,
                moe_keys.up_stride,
                moe_keys.down_stride,
                eps,
                shared_expert.as_ref(),
                explicit_barriers,
            )?;
            barrier.post_dispatch(&[&scratch.hidden], &[&scratch.hidden]);
            barrier.step(encoder);
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_ffn += moe_t.elapsed();
            }
            return Ok(());
        }

        let ffn_nw = weight_cache.get(&layer_keys.ffn_norm).unwrap();
        metal_ops
            .elementwise
            .encode_residual_add_rms_norm_out_batch(
                encoder,
                &scratch.hidden,
                &scratch.proj_buf,
                ffn_nw,
                &scratch.norm_buf,
                dim as u32,
                1,
                eps,
            );
        barrier.step(encoder);
        if let Some(ops_ref) = ops.as_deref_mut() {
            let elapsed = residual_norm_t.elapsed();
            ops_ref.gpu_encode_layer_residual += elapsed / 2;
            ops_ref.gpu_encode_layer_norm += elapsed / 2;
        }

        let gate_up_t = OpTimer::start();
        let wg = weight_cache.get(&layer_keys.wg).unwrap();
        let wu = weight_cache.get(&layer_keys.wu).unwrap();
        if !crate::model::shared::encode_dequant_matvec_pair_with_config(
            metal_ops,
            encoder,
            wg,
            wu,
            &scratch.norm_buf,
            &scratch.gate_buf,
            &scratch.up_buf,
            inter_dim as u32,
            dim as u32,
            layer_keys.wg_dtype,
            layer_keys.wu_dtype,
            exec_plan.dequant_dispatch,
            false,
        ) {
            encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                wg,
                &scratch.norm_buf,
                &scratch.gate_buf,
                inter_dim as u32,
                dim as u32,
                layer_keys.wg_dtype,
                exec_plan.dequant_dispatch,
            );
            encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                wu,
                &scratch.norm_buf,
                &scratch.up_buf,
                inter_dim as u32,
                dim as u32,
                layer_keys.wu_dtype,
                exec_plan.dequant_dispatch,
            );
        }
        barrier.step(encoder);
        if let Some(ops_ref) = ops.as_deref_mut() {
            ops_ref.gpu_encode_layer_ffn += gate_up_t.elapsed();
        }

        let ffn_tail_t = OpTimer::start();
        crate::model::shared::encode_gpu_ffn_decode_tail(
            metal_ops,
            encoder,
            scratch,
            &scratch.hidden,
            weight_cache.get(&layer_keys.wd).unwrap(),
            layer_keys.wd_dtype,
            dim as u32,
            inter_dim as u32,
            eps,
            exec_plan.dequant_dispatch,
            exec_plan.use_fused_silu_down,
            crate::model::layer_ops::FfnActivation::SiLU,
            None,
            None,
            barrier,
        );
        if let Some(ops_ref) = ops.as_deref_mut() {
            ops_ref.gpu_encode_layer_ffn += ffn_tail_t.elapsed();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_qwen35_native_decode_layers(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        barrier: &crate::model::shared::DecodeBarrierCtx<'_>,
        cfg: &ModelConfig,
        qwen_kv: &crate::kv::Qwen3_5Kv,
        cached: &crate::backend::metal::CachedModelKeys,
        gpu_layer_keys: &[Qwen3_5GpuLayerKeys],
        moe_layer_keys: Option<&[Option<Qwen3_5MoeResidentLayerKeys>]>,
        weight_cache: &rustc_hash::FxHashMap<usize, MetalBuffer>,
        moe_weight_cache: Option<&rustc_hash::FxHashMap<usize, MetalBuffer>>,
        scratch: &crate::backend::metal::GpuScratchBuffers,
        moe_scratch: Option<crate::backend::metal::MoeBatchScratchView>,
        exec_plan: &crate::model::execution_plan::GpuDecodeExecutionPlan,
        explicit_barriers: bool,
        layer_start: usize,
        layer_end_exclusive: usize,
        seq_len: usize,
        full_seq_len: usize,
        rope_position: f32,
        recurrent_slot: usize,
        dims: Qwen3_5RecurrentDims,
        dim: usize,
        inter_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        conv_cache_len: usize,
        eps: f32,
        ops: &mut Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        use crate::model::shared::encode_dequant_matvec_with_config;

        let max_layer = Self::debug_max_layer_inclusive();
        for layer in layer_start..layer_end_exclusive.min(cached.layers.len()) {
            if max_layer.is_some_and(|max_layer| layer > max_layer) {
                break;
            }
            let layer_keys = &cached.layers[layer];
            let moe_keys = moe_layer_keys.and_then(|layer_keys| layer_keys[layer].as_ref());

            let layer_norm_t = OpTimer::start();
            let norm_w = weight_cache.get(&layer_keys.attn_norm).unwrap();
            metal_ops.elementwise.encode_rms_norm_out(
                encoder,
                &scratch.hidden,
                norm_w,
                &scratch.norm_buf,
                dim as u32,
                eps,
            );
            barrier.step(encoder);
            if let Some(ops_ref) = ops.as_deref_mut() {
                ops_ref.gpu_encode_layer_norm += layer_norm_t.elapsed();
            }

            if !cfg.qwen35_is_recurrent_layer(layer) {
                let q_norm_key = layer_keys.attn_q_norm;
                let k_norm_key = layer_keys.attn_k_norm;
                let gpu_attn = qwen_kv.gpu_attention().unwrap();

                let qkv_t = OpTimer::start();
                let wq = weight_cache.get(&layer_keys.wq).unwrap();
                let wk = weight_cache.get(&layer_keys.wk).unwrap();
                let wv = weight_cache.get(&layer_keys.wv).unwrap();
                encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wq,
                    &scratch.norm_buf,
                    &scratch.gate_buf,
                    (q_dim * 2) as u32,
                    dim as u32,
                    layer_keys.wq_dtype,
                    exec_plan.dequant_dispatch,
                );
                encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wk,
                    &scratch.norm_buf,
                    &scratch.k_buf,
                    kv_dim as u32,
                    dim as u32,
                    layer_keys.wk_dtype,
                    exec_plan.dequant_dispatch,
                );
                encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wv,
                    &scratch.norm_buf,
                    &scratch.v_buf,
                    kv_dim as u32,
                    dim as u32,
                    layer_keys.wv_dtype,
                    exec_plan.dequant_dispatch,
                );
                metal_ops.elementwise.encode_split_qgate_batch(
                    encoder,
                    &scratch.gate_buf,
                    &scratch.q_buf,
                    &scratch.up_buf,
                    1,
                    q_dim as u32,
                    head_dim as u32,
                );
                barrier.step(encoder);
                if let Some(ops_ref) = ops.as_deref_mut() {
                    ops_ref.gpu_encode_layer_qkv += qkv_t.elapsed();
                }

                if let (Some(q_key), Some(k_key)) = (q_norm_key, k_norm_key) {
                    let qk_norm_t = OpTimer::start();
                    let q_nw = weight_cache.get(&q_key).unwrap();
                    let k_nw = weight_cache.get(&k_key).unwrap();
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder,
                        &scratch.q_buf,
                        q_nw,
                        1,
                        n_heads as u32,
                        head_dim as u32,
                        eps,
                    );
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder,
                        &scratch.k_buf,
                        k_nw,
                        1,
                        n_kv_heads as u32,
                        head_dim as u32,
                        eps,
                    );
                    barrier.step(encoder);
                    if let Some(ops_ref) = ops.as_deref_mut() {
                        ops_ref.gpu_encode_layer_norm += qk_norm_t.elapsed();
                    }
                }

                let rope_t = OpTimer::start();
                metal_ops.elementwise.encode_rope_batch_neox_partial(
                    encoder,
                    &scratch.q_buf,
                    &scratch.k_buf,
                    1,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    (head_dim as u32).min(64),
                    rope_position,
                    0.0,
                    cfg.rope_freq_base,
                );
                barrier.step(encoder);
                if let Some(ops_ref) = ops.as_deref_mut() {
                    ops_ref.gpu_encode_layer_rope += rope_t.elapsed();
                }

                let kv_append_t = OpTimer::start();
                Self::encode_qwen35_single_token_kv_append(
                    metal_ops,
                    encoder,
                    barrier,
                    gpu_attn,
                    layer,
                    seq_len,
                    kv_dim,
                    &scratch.k_buf,
                    &scratch.v_buf,
                )?;
                if let Some(ops_ref) = ops.as_deref_mut() {
                    ops_ref.gpu_encode_layer_kv_append += kv_append_t.elapsed();
                }

                let attn_t = OpTimer::start();
                if gpu_attn.is_q8() {
                    if head_dim == 128 {
                        metal_ops.attention.encode_attention_decode_q8kv(
                            encoder,
                            &scratch.q_buf,
                            gpu_attn.k_buffer(layer),
                            gpu_attn.v_buffer(layer),
                            &scratch.attn_out,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            0,
                            full_seq_len as u32,
                        );
                    } else if head_dim == 256 {
                        metal_ops.attention.encode_attention_decode_q8kv_hd256(
                            encoder,
                            &scratch.q_buf,
                            gpu_attn.k_buffer(layer),
                            gpu_attn.v_buffer(layer),
                            &scratch.attn_out,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            0,
                            full_seq_len as u32,
                        );
                    } else {
                        anyhow::bail!(
                            "qwen35 native decode q8 attention requires head_dim 128 or 256, got {head_dim}"
                        );
                    }
                } else {
                    metal_ops
                        .attention
                        .encode_attention_decode_with_scratch_and_config(
                            encoder,
                            &scratch.q_buf,
                            gpu_attn.k_buffer(layer),
                            gpu_attn.v_buffer(layer),
                            &scratch.attn_out,
                            &scratch.splitk_partial_out,
                            &scratch.splitk_partial_lse,
                            gpu_attn.is_f16(),
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            0,
                            full_seq_len as u32,
                            exec_plan.attention_dispatch,
                        );
                }
                barrier.step(encoder);

                metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                    encoder,
                    &scratch.up_buf,
                    &scratch.attn_out,
                    q_dim as u32,
                );
                barrier.step(encoder);
                if let Some(ops_ref) = ops.as_deref_mut() {
                    ops_ref.gpu_encode_layer_attention += attn_t.elapsed();
                }

                let out_proj_t = OpTimer::start();
                encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    weight_cache.get(&layer_keys.wo).unwrap(),
                    &scratch.attn_out,
                    &scratch.proj_buf,
                    dim as u32,
                    q_dim as u32,
                    layer_keys.wo_dtype,
                    exec_plan.dequant_dispatch,
                );
                barrier.step(encoder);
                if let Some(ops_ref) = ops.as_deref_mut() {
                    ops_ref.gpu_encode_layer_out_proj += out_proj_t.elapsed();
                }

                Self::encode_qwen35_native_post_attention_ffn(
                    metal_ops,
                    encoder,
                    barrier,
                    scratch,
                    layer_keys,
                    moe_keys,
                    weight_cache,
                    moe_weight_cache,
                    moe_scratch,
                    exec_plan,
                    explicit_barriers,
                    dim,
                    inter_dim,
                    eps,
                    ops,
                )?;
                continue;
            }

            let recurrent_keys = match &gpu_layer_keys[layer] {
                Qwen3_5GpuLayerKeys::Recurrent(keys) => keys,
                Qwen3_5GpuLayerKeys::FullAttention => {
                    anyhow::bail!("expected recurrent qwen35 GPU keys for layer {layer}")
                }
            };
            debug_assert!(kv_dim >= dims.time_step_rank);
            let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
            let recurrent_state_stride = qwen_kv.recurrent_state_len();
            metal_ops.with_qwen35_recurrent_slot_buffer_for_kv(
                qwen_kv,
                layer,
                recurrent_slot,
                conv_state_stride,
                recurrent_state_stride,
                |slot_buffers| -> anyhow::Result<()> {
                    let recurrent_weights = Qwen3_5NativeRecurrentCachedWeights {
                        wqkv: weight_cache.get(&recurrent_keys.wqkv).unwrap(),
                        wgate: weight_cache.get(&recurrent_keys.wgate).unwrap(),
                        wbeta: weight_cache.get(&recurrent_keys.wbeta).unwrap(),
                        walpha: weight_cache.get(&recurrent_keys.walpha).unwrap(),
                        conv_kernel: weight_cache.get(&recurrent_keys.conv_kernel).unwrap(),
                        ssm_norm: weight_cache.get(&recurrent_keys.ssm_norm).unwrap(),
                        dt_bias: weight_cache.get(&recurrent_keys.dt_bias).unwrap(),
                        ssm_a: weight_cache.get(&recurrent_keys.ssm_a).unwrap(),
                        wssm_out: weight_cache.get(&recurrent_keys.wssm_out).unwrap(),
                    };
                    let recurrent_dtypes = Qwen3_5NativeRecurrentDtypes {
                        wqkv: recurrent_keys.wqkv_dtype,
                        wgate: recurrent_keys.wgate_dtype,
                        wbeta: recurrent_keys.wbeta_dtype,
                        walpha: recurrent_keys.walpha_dtype,
                        wssm_out: recurrent_keys.wssm_out_dtype,
                    };
                    metal_ops.with_qwen35_recurrent_projection_scratch(
                        1,
                        dims.conv_dim(),
                        dims.inner_size,
                        dims.time_step_rank,
                        |temp_scratch| -> anyhow::Result<()> {
                            let recurrent_scratch = Qwen3_5NativeRecurrentProjectionScratch {
                                qkv: &temp_scratch.qkv,
                                z: &temp_scratch.z,
                                beta: &temp_scratch.beta,
                                alpha: &temp_scratch.alpha,
                            };

                            let recurrent_t = OpTimer::start();
                            Self::encode_qwen35_native_recurrent_layer(
                                metal_ops,
                                encoder,
                                barrier,
                                recurrent_weights,
                                recurrent_scratch,
                                slot_buffers,
                                &scratch.norm_buf,
                                &scratch.up_buf,
                                &scratch.proj_buf,
                                conv_cache_len,
                                dims,
                                dim,
                                eps,
                                recurrent_dtypes,
                                exec_plan.dequant_dispatch,
                            )?;
                            if let Some(ops_ref) = ops.as_deref_mut() {
                                let elapsed = recurrent_t.elapsed();
                                ops_ref.gpu_encode_layer_qkv += elapsed / 3;
                                ops_ref.gpu_encode_layer_attention += elapsed / 3;
                                ops_ref.gpu_encode_layer_out_proj += elapsed / 3;
                            }
                            Ok(())
                        },
                    )
                },
            )?;

            Self::encode_qwen35_native_post_attention_ffn(
                metal_ops,
                encoder,
                barrier,
                scratch,
                layer_keys,
                moe_keys,
                weight_cache,
                moe_weight_cache,
                moe_scratch,
                exec_plan,
                explicit_barriers,
                dim,
                inter_dim,
                eps,
                ops,
            )?;

            if layer + 1 < layer_end_exclusive
                && Self::qwen35_should_flush_between_coalesced_native_decode_layers(
                    cfg,
                    layer,
                    layer + 1,
                )
            {
                barrier.flush();
            }
        }
        Ok(())
    }

    fn encode_qwen35_pending_step_inner(
        metal_ops: &crate::backend::metal::MetalOps,
        cfg: &ModelConfig,
        hidden_buf: &MetalBuffer,
        position: usize,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        weights: &WeightStore,
        fuse_argmax: bool,
    ) -> anyhow::Result<ax_engine_metal::PendingFrame> {
        let dims = Self::recurrent_dims(cfg)?;
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let recurrent_slot = qwen_kv.active_slot();
        let eps = cfg.rms_norm_eps;
        let conv_cache_len = qwen_kv.conv_cache_len();
        let full_seq_len = position + 1;

        anyhow::ensure!(
            qwen_kv.ensure_gpu_attention_capacity_for(full_seq_len),
            "qwen35 pending decode requires GPU attention KV capacity"
        );

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_qwen35(metal_ops, weights, cfg)?;
        }
        metal_ops.init_scratches(cfg);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let gpu_layer_keys = Self::cached_gpu_layer_keys(cached.lm_head)
            .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 gpu layer keys"))?;
        let moe_layer_keys = if Self::qwen35_is_moe(cfg) {
            let layer_keys = Self::cached_moe_layer_keys(cached.lm_head)
                .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 moe layer keys"))?;
            if cached
                .layers
                .iter()
                .zip(layer_keys.iter())
                .any(|(layer, moe)| layer.wg == 0 && moe.is_none())
            {
                anyhow::bail!("missing cached qwen35 moe layer keys");
            }
            Some(layer_keys)
        } else {
            None
        };
        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = moe_layer_keys
            .as_ref()
            .map(|_| metal_ops.lock_moe_weight_cache());
        let moe_scratch = if moe_layer_keys.is_some() {
            metal_ops.init_batch_scratches(cfg, 1);
            Some(metal_ops.moe_batch_scratch_view()?)
        } else {
            None
        };

        let exec_plan = Self::qwen35_decode_plan(
            metal_ops,
            qwen_kv.gpu_attention().ok_or_else(|| {
                anyhow::anyhow!("qwen35 pending decode requires GPU attention KV")
            })?,
            cfg.embedding_dim,
            cfg.head_dim,
            full_seq_len,
            true,
        );
        let dispatch_plan = Self::qwen35_native_decode_dispatch_plan(cfg, &exec_plan, n_layers);
        let rope_position = Self::rope_position(cfg, position);
        let use_concurrent =
            dispatch_plan.encoder == crate::model::execution_plan::DecodeEncoderPlan::Concurrent;
        // Macro to avoid duplicating the entire closure for serial vs concurrent.
        macro_rules! with_encoder {
            ($device:expr, $body:expr) => {
                if use_concurrent {
                    $device.encode_frame_concurrent($body)
                } else {
                    $device.encode_frame($body)
                }
            };
        }
        with_encoder!(metal_ops.device, |encoder| {
            let barrier =
                crate::model::shared::DecodeBarrierCtx::new(encoder, dispatch_plan.barriers);
            barrier.pre_dispatch(&[hidden_buf], &[&s.hidden]);
            metal_ops
                .elementwise
                .encode_buffer_copy(encoder, hidden_buf, 0, &s.hidden, 0, dim as u32);
            barrier.post_dispatch(&[hidden_buf], &[&s.hidden]);
            barrier.step(encoder);

            let mut no_ops = None;
            for (range_idx, layer_range) in dispatch_plan.layer_ranges.iter().copied().enumerate() {
                Self::encode_qwen35_native_decode_layers(
                    metal_ops,
                    encoder,
                    &barrier,
                    cfg,
                    qwen_kv,
                    cached,
                    &gpu_layer_keys,
                    moe_layer_keys.as_deref(),
                    &weight_cache,
                    moe_weight_cache.as_deref(),
                    s,
                    moe_scratch,
                    &exec_plan,
                    dispatch_plan.barriers
                        == crate::model::execution_plan::DecodeBarrierPlan::Explicit,
                    layer_range.start,
                    layer_range.end_exclusive,
                    position,
                    full_seq_len,
                    rope_position,
                    recurrent_slot,
                    dims,
                    dim,
                    inter_dim,
                    q_dim,
                    kv_dim,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    conv_cache_len,
                    eps,
                    &mut no_ops,
                )?;
                if range_idx + 1 < dispatch_plan.layer_ranges.len() {
                    barrier.flush();
                }
            }

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
        })
    }

    /// GPU-accelerated single-token decode for Qwen3.5.
    ///
    /// Encodes full-attention layers using the shared GPU layer encoder (1 CB
    /// per layer) and recurrent layers with minimized CPU round-trips. This is
    /// the default single-token decode path when Metal recurrent support and
    /// GPU attention KV are available.
    ///
    /// Returns `Ok(true)` if GPU path was used, `Ok(false)` to fall back.
    #[allow(clippy::too_many_arguments)]
    fn try_forward_single_gpu(
        ctx: &ForwardContext,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<bool> {
        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(false);
        };
        let cfg = ctx.config;
        // MoE single-token GPU decode: uses mul_mat_id for expert dispatch
        // (same as batch prefill) and GPU-resident recurrent buffers.
        let qwen_kv = kv
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3_5Forward requires ModelKv::Qwen35"))?;
        if qwen_kv.gpu_attention().is_none() {
            return Ok(false);
        }

        let dims = Self::recurrent_dims(cfg)?;
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let recurrent_slot = qwen_kv.active_slot();
        let eps = cfg.rms_norm_eps;
        let conv_cache_len = qwen_kv.conv_cache_len();

        // Cache weights on GPU.
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_qwen35(metal_ops, weights, cfg)?;
        }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let gpu_layer_keys = Self::cached_gpu_layer_keys(cached.lm_head)
            .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 gpu layer keys"))?;
        let moe_layer_keys = if Self::qwen35_is_moe(cfg) {
            let layer_keys = Self::cached_moe_layer_keys(cached.lm_head)
                .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 moe layer keys"))?;
            if cached
                .layers
                .iter()
                .zip(layer_keys.iter())
                .any(|(layer, moe)| layer.wg == 0 && moe.is_none())
            {
                return Ok(false);
            }
            Some(layer_keys)
        } else {
            None
        };
        metal_ops.init_scratches(cfg);
        let moe_scratch = if moe_layer_keys.is_some() {
            metal_ops.init_batch_scratches(cfg, 1);
            Some(metal_ops.moe_batch_scratch_view()?)
        } else {
            None
        };
        let mut scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_mut().unwrap();
        let weight_cache = metal_ops.lock_weight_cache();
        let moe_weight_cache = moe_layer_keys
            .as_ref()
            .map(|_| metal_ops.lock_moe_weight_cache());

        let seq_len = qwen_kv.seq_len();
        let full_seq_len = seq_len + 1;
        if !qwen_kv.ensure_gpu_attention_capacity_for(full_seq_len) {
            return Ok(false);
        }

        // Build execution plan.
        let exec_plan = Self::qwen35_decode_plan(
            metal_ops,
            qwen_kv.gpu_attention().unwrap(),
            cfg.embedding_dim,
            cfg.head_dim,
            full_seq_len,
            false,
        );
        let setup_t = OpTimer::start();

        // Embed token.
        {
            let h = unsafe {
                std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim)
            };
            weights.dequantize_row("token_embd.weight", token_id as usize, h)?;
        }

        let rope_position = Self::rope_position(cfg, position);

        if !qwen_kv.has_gpu_recurrent_state() {
            for layer in 0..n_layers {
                if cfg.qwen35_is_recurrent_layer(layer) {
                    metal_ops.sync_qwen35_slot_buffers_from_kv(qwen_kv, layer, recurrent_slot);
                }
            }
        }

        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        let exec_t = OpTimer::start();
        let layer_limit = Self::debug_max_layer_inclusive()
            .map(|layer| layer.saturating_add(1))
            .unwrap_or(n_layers)
            .min(n_layers);
        let dispatch_plan = Self::qwen35_native_decode_dispatch_plan(cfg, &exec_plan, layer_limit);
        let use_concurrent_sc =
            dispatch_plan.encoder == crate::model::execution_plan::DecodeEncoderPlan::Concurrent;
        macro_rules! exec_sync {
            ($device:expr, $body:expr) => {
                if use_concurrent_sc {
                    $device.execute_sync_concurrent($body)
                } else {
                    $device.execute_sync($body)
                }
            };
        }
        if dispatch_plan.layer_ranges.len() == 1
            && dispatch_plan.layer_ranges[0]
                == (Qwen3_5NativeDecodeLayerRange {
                    start: 0,
                    end_exclusive: layer_limit,
                })
        {
            exec_sync!(metal_ops.device, |encoder| {
                let barrier =
                    crate::model::shared::DecodeBarrierCtx::new(encoder, dispatch_plan.barriers);
                Self::encode_qwen35_native_decode_layers(
                    metal_ops,
                    encoder,
                    &barrier,
                    cfg,
                    qwen_kv,
                    cached,
                    &gpu_layer_keys,
                    moe_layer_keys.as_deref(),
                    &weight_cache,
                    moe_weight_cache.as_deref(),
                    s,
                    moe_scratch,
                    &exec_plan,
                    dispatch_plan.barriers
                        == crate::model::execution_plan::DecodeBarrierPlan::Explicit,
                    dispatch_plan.layer_ranges[0].start,
                    dispatch_plan.layer_ranges[0].end_exclusive,
                    seq_len,
                    full_seq_len,
                    rope_position,
                    recurrent_slot,
                    dims,
                    dim,
                    inter_dim,
                    q_dim,
                    kv_dim,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    conv_cache_len,
                    eps,
                    &mut ops,
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
                    eps,
                );
                barrier.flush();
                Ok(())
            })?;
        } else {
            let last_range_idx = dispatch_plan.layer_ranges.len().saturating_sub(1);
            for (range_idx, layer_range) in dispatch_plan.layer_ranges.iter().copied().enumerate() {
                exec_sync!(metal_ops.device, |encoder| {
                    let barrier = crate::model::shared::DecodeBarrierCtx::new(
                        encoder,
                        dispatch_plan.barriers,
                    );
                    Self::encode_qwen35_native_decode_layers(
                        metal_ops,
                        encoder,
                        &barrier,
                        cfg,
                        qwen_kv,
                        cached,
                        &gpu_layer_keys,
                        moe_layer_keys.as_deref(),
                        &weight_cache,
                        moe_weight_cache.as_deref(),
                        s,
                        moe_scratch,
                        &exec_plan,
                        dispatch_plan.barriers
                            == crate::model::execution_plan::DecodeBarrierPlan::Explicit,
                        layer_range.start,
                        layer_range.end_exclusive,
                        seq_len,
                        full_seq_len,
                        rope_position,
                        recurrent_slot,
                        dims,
                        dim,
                        inter_dim,
                        q_dim,
                        kv_dim,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        conv_cache_len,
                        eps,
                        &mut ops,
                    )?;
                    if range_idx == last_range_idx {
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
                    }
                    barrier.flush();
                    Ok(())
                })?;
            }
        }
        if std::env::var("AX_DEBUG_MOE_NATIVE").is_ok()
            && let Some(scratch) = moe_scratch
        {
            let ids =
                unsafe { &(*scratch.ids).as_slice::<i32>()[..cfg.n_expert_used.unwrap() as usize] };
            let weights = unsafe {
                &(*scratch.weights).as_slice::<f32>()[..cfg.n_expert_used.unwrap() as usize]
            };
            let norm = unsafe { &(*scratch.norm).as_slice::<f32>()[..4.min(dim)] };
            let accum = unsafe { &(*scratch.accum).as_slice::<f32>()[..4.min(dim)] };
            let hidden = unsafe { &s.hidden.as_slice::<f32>()[..4.min(dim)] };
            eprintln!(
                "[MOE NATIVE DEBUG] ids={ids:?} weights={weights:?} norm[0..4]={norm:?} accum[0..4]={accum:?} hidden[0..4]={hidden:?}"
            );
        }
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_execute += exec_t.elapsed();
        }

        drop(weight_cache);
        drop(cached_guard);

        let rb_t = OpTimer::start();
        for layer in 0..n_layers {
            if cfg.qwen35_is_recurrent_layer(layer) {
                let _ = qwen_kv.note_backend_conv_state_update(recurrent_slot, layer);
                let _ = qwen_kv.note_backend_recurrent_state_update(recurrent_slot, layer);
                if !qwen_kv.has_gpu_recurrent_state() {
                    let conv_generation = qwen_kv.conv_state_generation(recurrent_slot, layer);
                    let recurrent_generation =
                        qwen_kv.recurrent_state_generation(recurrent_slot, layer);
                    let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
                    let recurrent_state_stride = qwen_kv.recurrent_state_len();
                    metal_ops.with_qwen35_recurrent_slot_buffer(
                        layer,
                        recurrent_slot,
                        conv_state_stride,
                        recurrent_state_stride,
                        |slot_buffers| {
                            slot_buffers.conv_synced_generation = Some(conv_generation);
                            slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
                        },
                    );
                }
            }
        }
        qwen_kv.mark_attention_cpu_dirty();

        qwen_kv.finalize_token();

        let logits_gpu = unsafe { &s.logits_buf.as_slice::<f32>()[..vocab_size] };
        logits[..vocab_size].copy_from_slice(logits_gpu);
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }
        Ok(true)
    }
}
