impl Qwen35Forward {
    fn recurrent_batch_from_gpu_qkv_single_slot_check(
        metal_ops: &crate::backend::metal::MetalOps,
        n_tokens: usize,
        dims: Qwen35RecurrentDims,
    ) -> Qwen35RecurrentQkvFastPathCheck {
        let mut check = Qwen35RecurrentQkvFastPathCheck {
            state_size_too_large: dims.state_size > 256,
            group_divisibility_invalid: !dims.time_step_rank.is_multiple_of(dims.group_count),
            ..Default::default()
        };
        if check.state_size_too_large || check.group_divisibility_invalid {
            return check;
        }
        let bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_ref() else {
            check.missing_batch_scratches = true;
            return check;
        };
        let total_inner = n_tokens * dims.inner_size;
        let total_heads = n_tokens * dims.time_step_rank;
        let f32_capacity = |buf: &MetalBuffer| buf.len() / std::mem::size_of::<f32>();
        check.q_capacity_too_small = f32_capacity(&bs.q_buf) < total_inner;
        check.gate_capacity_too_small = f32_capacity(&bs.gate_buf) < total_heads;
        check.up_capacity_too_small = f32_capacity(&bs.up_buf) < total_heads;
        check
    }

    #[allow(clippy::too_many_arguments)]
    fn try_run_recurrent_batch_from_gpu_qkv_single_slot(
        metal_ops: &crate::backend::metal::MetalOps,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        recurrent_slot: usize,
        qkv_gpu: &MetalBuffer,
        beta_gpu: Option<&MetalBuffer>,
        alpha_gpu: Option<&MetalBuffer>,
        recurrent_keys: &Qwen35RecurrentLayerKeys,
        rec_beta_batch: &mut [f32],
        rec_alpha_batch: &mut [f32],
        rec_out_batch: &mut [f32],
        n_tokens: usize,
        dims: Qwen35RecurrentDims,
        rms_norm_eps: f32,
        keep_output_on_gpu: bool,
        force_slot_buffer_state: bool,
        record_backend_native_batch: bool,
    ) -> anyhow::Result<Option<Qwen35RecurrentGpuPrefillStats>> {
        if n_tokens <= 1
            || qwen_kv.conv_cache_len() > 8
            || !Self::recurrent_batch_from_gpu_qkv_single_slot_check(metal_ops, n_tokens, dims)
                .is_eligible()
        {
            return Ok(None);
        }

        let total_inner = n_tokens * dims.inner_size;
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let total_heads = n_tokens * dims.time_step_rank;
        let conv_cache_len_u32 = qwen_kv.conv_cache_len() as u32;
        let mut stats = Qwen35RecurrentGpuPrefillStats::default();

        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(None);
        };
        let weight_cache = metal_ops.lock_weight_cache();
        let conv_kernel_buf = weight_cache
            .get(&recurrent_keys.conv_kernel)
            .expect("qwen35 conv kernel buffer missing after cache");
        let dt_bias_buf = weight_cache
            .get(&recurrent_keys.dt_bias)
            .expect("qwen35 dt_bias buffer missing after cache");
        let ssm_a_buf = weight_cache
            .get(&recurrent_keys.ssm_a)
            .expect("qwen35 ssm_a buffer missing after cache");
        let rec_out_alias = if !keep_output_on_gpu && n_tokens > 1 {
            let active_rec_out = &mut rec_out_batch[..total_inner];
            unsafe {
                MetalBuffer::from_mut_slice_no_copy(metal_ops.device.device(), active_rec_out)
            }
            .ok()
        } else {
            None
        };

        let layer_state_owner = qwen_kv.layer_state_owner(recurrent_slot, layer);
        let cpu_state_fresh =
            layer_state_owner == crate::kv::Qwen35LayerStateOwner::CpuMaterialized;
        let recurrent_state_mode =
            Self::resolve_qwen35_prefill_recurrent_state_mode(n_tokens, layer_state_owner);
        let alpha_beta_storage_mode =
            Self::qwen35_prefill_alpha_beta_storage_mode_for_tokens(n_tokens);
        let handoff_elapsed = metal_ops.with_qwen35_qkv_handoff_scratch(
            n_tokens,
            dims.conv_dim(),
            dims.inner_size,
            dims.time_step_rank,
            |scratch| -> anyhow::Result<(
                Duration,
                bool,
                crate::backend::metal::Qwen35SlotBufferSyncOutcome,
            )> {
                let alpha_slice = unsafe {
                    &mut scratch.alpha.as_mut_slice::<f32>()[..n_tokens * dims.time_step_rank]
                };
                let beta_slice = unsafe {
                    &mut scratch.beta.as_mut_slice::<f32>()[..n_tokens * dims.time_step_rank]
                };
                // When alpha/beta GPU pointers aren't available, copy the
                // CPU-projected values into UMA scratch buffers. The actual
                // softplus/sigmoid/f16-cast is done on GPU inside run_handoff
                // (same kernels as the GPU-primary path) instead of CPU.
                let needs_gpu_alpha_beta_prep =
                    beta_gpu.is_none() || alpha_gpu.is_none();
                if needs_gpu_alpha_beta_prep {
                    alpha_slice.copy_from_slice(&rec_alpha_batch[..total_heads]);
                    beta_slice.copy_from_slice(&rec_beta_batch[..total_heads]);
                }
                // Scratch buffers from chunked GDN must outlive the command
                // buffer. Store them here so they survive past execute_sync.
                let gdn_scratch_hold: std::cell::RefCell<Vec<ax_engine_metal::MetalBuffer>> =
                    std::cell::RefCell::new(Vec::new());

                let run_handoff = |conv_state: &MetalBuffer,
                                   recurrent_state: &MetalBuffer|
                 -> anyhow::Result<Duration> {
                    let t = OpTimer::start();
                    metal_ops.device.execute_sync(|encoder| {
                        metal_ops.gdn.encode_causal_conv_sequence(
                            encoder,
                            qkv_gpu,
                            conv_kernel_buf,
                            conv_state,
                            &scratch.conv_out,
                            n_tokens as u32,
                            conv_cache_len_u32,
                            dims.conv_dim() as u32,
                        );
                        let encoded =
                            if let (Some(beta_gpu), Some(alpha_gpu)) = (beta_gpu, alpha_gpu) {
                                metal_ops.elementwise.encode_softplus_bias_mul_batch(
                                    encoder,
                                    alpha_gpu,
                                    dt_bias_buf,
                                    ssm_a_buf,
                                    total_heads as u32,
                                    dims.time_step_rank as u32,
                                );
                                metal_ops.elementwise.encode_sigmoid_inplace(
                                    encoder,
                                    beta_gpu,
                                    total_heads as u32,
                                );
                                match alpha_beta_storage_mode {
                                    Qwen35PrefillAlphaBetaStorageMode::F16 => {
                                        metal_ops.elementwise.encode_cast_f32_to_f16(
                                            encoder,
                                            alpha_gpu,
                                            &scratch.alpha_f16,
                                            total_heads as u32,
                                        );
                                        metal_ops.elementwise.encode_cast_f32_to_f16(
                                            encoder,
                                            beta_gpu,
                                            &scratch.beta_f16,
                                            total_heads as u32,
                                        );
                                        metal_ops.gdn.encode_prepare_multi_token_qkv_alpha_beta_f16(
                                            encoder,
                                            &scratch.conv_out,
                                            &scratch.alpha_f16,
                                            &scratch.beta_f16,
                                            &bs.q_buf,
                                            &scratch.k,
                                            &scratch.v,
                                            &bs.gate_buf,
                                            &bs.up_buf,
                                            n_tokens as u32,
                                            dims.group_count as u32,
                                            dims.time_step_rank as u32,
                                            dims.state_size as u32,
                                            rms_norm_eps,
                                        )
                                    }
                                    _ => metal_ops.gdn.encode_prepare_multi_token_qkv(
                                        encoder,
                                        &scratch.conv_out,
                                        alpha_gpu,
                                        beta_gpu,
                                        &bs.q_buf,
                                        &scratch.k,
                                        &scratch.v,
                                        &bs.gate_buf,
                                        &bs.up_buf,
                                        n_tokens as u32,
                                        dims.group_count as u32,
                                        dims.time_step_rank as u32,
                                        dims.state_size as u32,
                                        rms_norm_eps,
                                    ),
                                }
                            } else {
                                // Alpha/beta are in scratch (CPU-copied). Run
                                // softplus/sigmoid/f16-cast on GPU instead of
                                // CPU — same kernels as the GPU-primary path.
                                if needs_gpu_alpha_beta_prep {
                                    metal_ops.elementwise.encode_softplus_bias_mul_batch(
                                        encoder,
                                        &scratch.alpha,
                                        dt_bias_buf,
                                        ssm_a_buf,
                                        total_heads as u32,
                                        dims.time_step_rank as u32,
                                    );
                                    metal_ops.elementwise.encode_sigmoid_inplace(
                                        encoder,
                                        &scratch.beta,
                                        total_heads as u32,
                                    );
                                }
                                match alpha_beta_storage_mode {
                                    Qwen35PrefillAlphaBetaStorageMode::F16 => {
                                        if needs_gpu_alpha_beta_prep {
                                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                                encoder,
                                                &scratch.alpha,
                                                &scratch.alpha_f16,
                                                total_heads as u32,
                                            );
                                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                                encoder,
                                                &scratch.beta,
                                                &scratch.beta_f16,
                                                total_heads as u32,
                                            );
                                        }
                                        metal_ops.gdn.encode_prepare_multi_token_qkv_alpha_beta_f16(
                                            encoder,
                                            &scratch.conv_out,
                                            &scratch.alpha_f16,
                                            &scratch.beta_f16,
                                            &bs.q_buf,
                                            &scratch.k,
                                            &scratch.v,
                                            &bs.gate_buf,
                                            &bs.up_buf,
                                            n_tokens as u32,
                                            dims.group_count as u32,
                                            dims.time_step_rank as u32,
                                            dims.state_size as u32,
                                            rms_norm_eps,
                                        )
                                    }
                                    _ => metal_ops.gdn.encode_prepare_multi_token_qkv(
                                        encoder,
                                        &scratch.conv_out,
                                        &scratch.alpha,
                                        &scratch.beta,
                                        &bs.q_buf,
                                        &scratch.k,
                                        &scratch.v,
                                        &bs.gate_buf,
                                        &bs.up_buf,
                                        n_tokens as u32,
                                        dims.group_count as u32,
                                        dims.time_step_rank as u32,
                                        dims.state_size as u32,
                                        rms_norm_eps,
                                    ),
                                }
                            };
                        anyhow::ensure!(
                            encoded,
                            "qwen35 recurrent batch Metal pack kernel does not support this shape"
                        );
                        *gdn_scratch_hold.borrow_mut() = metal_ops.gdn.encode_gated_delta_sequence_auto(
                            encoder,
                            &metal_ops.elementwise,
                            &metal_ops.device,
                            &bs.q_buf,
                            &scratch.k,
                            &scratch.v,
                            &bs.gate_buf,
                            &bs.up_buf,
                            recurrent_state,
                            qkv_gpu,
                            n_tokens as u32,
                            dims.time_step_rank as u32,
                            dims.state_size as u32,
                        );
                        if let Some(output_buf) = rec_out_alias.as_ref() {
                            metal_ops.gdn.encode_unpack_bhsk_to_token_major(
                                encoder,
                                qkv_gpu,
                                output_buf,
                                n_tokens as u32,
                                dims.time_step_rank as u32,
                                dims.state_size as u32,
                            );
                        }
                        Ok(())
                    })?;
                    Ok(t.elapsed())
                };

                let mut used_cpu_alias_state = false;
                let mut slot_buffer_sync =
                    crate::backend::metal::Qwen35SlotBufferSyncOutcome::default();
                let elapsed = if let Some((conv_buf, rec_buf)) =
                    qwen_kv.gpu_recurrent_buffers(recurrent_slot, layer)
                {
                    // GPU-resident path: buffers owned by Qwen35Kv, no sync needed.
                    slot_buffer_sync.note_backend_carryover();
                    run_handoff(conv_buf, rec_buf)?
                } else {
                    let force_slot_buffer = force_slot_buffer_state
                        || matches!(
                            recurrent_state_mode,
                            Qwen35PrefillRecurrentStateMode::SlotBuffer
                                | Qwen35PrefillRecurrentStateMode::BackendOwned
                        );
                    if cpu_state_fresh && !force_slot_buffer {
                        let (conv_state_cpu, recurrent_state_cpu) =
                            qwen_kv.recurrent_buffers_for_slot_mut(recurrent_slot, layer);
                        let conv_state_alias = unsafe {
                            MetalBuffer::from_mut_slice_no_copy(
                                metal_ops.device.device(),
                                conv_state_cpu,
                            )
                        };
                        let recurrent_state_alias = unsafe {
                            MetalBuffer::from_mut_slice_no_copy(
                                metal_ops.device.device(),
                                recurrent_state_cpu,
                            )
                        };
                        if let (Ok(conv_state_alias), Ok(recurrent_state_alias)) =
                            (conv_state_alias, recurrent_state_alias)
                        {
                            used_cpu_alias_state = true;
                            run_handoff(&conv_state_alias, &recurrent_state_alias)?
                        } else {
                            slot_buffer_sync = metal_ops.sync_qwen35_slot_buffers_from_kv(
                                qwen_kv,
                                layer,
                                recurrent_slot,
                            );
                            metal_ops.with_qwen35_recurrent_slot_buffer(
                                layer,
                                recurrent_slot,
                                conv_state_stride,
                                recurrent_state_stride,
                                |slot_buffers| {
                                    run_handoff(
                                        &slot_buffers.conv_state,
                                        &slot_buffers.recurrent_state,
                                    )
                                },
                            )?
                        }
                    } else {
                        slot_buffer_sync = metal_ops.sync_qwen35_slot_buffers_from_kv(
                            qwen_kv,
                            layer,
                            recurrent_slot,
                        );
                        metal_ops.with_qwen35_recurrent_slot_buffer(
                            layer,
                            recurrent_slot,
                            conv_state_stride,
                            recurrent_state_stride,
                            |slot_buffers| {
                                run_handoff(&slot_buffers.conv_state, &slot_buffers.recurrent_state)
                            },
                        )?
                    }
                };
                Ok((elapsed, used_cpu_alias_state, slot_buffer_sync))
            },
        )?;
        let (handoff_elapsed, used_cpu_alias_state, slot_buffer_sync) = handoff_elapsed;
        stats.gpu_execute += handoff_elapsed;
        metal_ops.record_qwen35_recurrent_batch_qkv_handoff_gpu(handoff_elapsed);
        if record_backend_native_batch {
            metal_ops.record_qwen35_recurrent_batch_state_batch_backend_native();
        }
        if used_cpu_alias_state {
            metal_ops.record_qwen35_recurrent_batch_qkv_handoff_cpu_alias();
        } else {
            metal_ops.record_qwen35_recurrent_batch_qkv_handoff_slot_buffer();
            if slot_buffer_sync.used_backend_carryover {
                metal_ops.record_qwen35_recurrent_batch_qkv_handoff_backend_carryover();
            }
            if slot_buffer_sync.used_backend_zero_init {
                metal_ops.record_qwen35_recurrent_batch_qkv_handoff_backend_zero_init();
            }
            if slot_buffer_sync.used_cpu_materialization {
                metal_ops.record_qwen35_recurrent_batch_qkv_handoff_cpu_materialization();
            }
        }

        if !keep_output_on_gpu {
            let unpack_t = OpTimer::start();
            let used_gpu_unpack = rec_out_alias.is_some();
            if !used_gpu_unpack {
                unsafe {
                    crate::backend::metal::unpack_bhsk_to_token_major(
                        &qkv_gpu.as_slice::<f32>()[..total_inner],
                        &mut rec_out_batch[..total_inner],
                        n_tokens,
                        dims.time_step_rank,
                        dims.state_size,
                    );
                }
            }
            let unpack_elapsed = unpack_t.elapsed();
            stats.gpu_readback += unpack_elapsed;
            metal_ops.record_qwen35_recurrent_batch_unpack(unpack_elapsed);
        }

        drop(weight_cache);
        drop(bs_guard);

        if qwen_kv.has_gpu_recurrent_state() {
            // GPU-resident path: buffers live in Qwen35Kv, just mark backend ownership.
            let _ = qwen_kv.note_backend_conv_state_update(recurrent_slot, layer);
            let _ = qwen_kv.note_backend_recurrent_state_update(recurrent_slot, layer);
        } else if used_cpu_alias_state {
            let _ = qwen_kv.note_cpu_visible_layer_state_update(recurrent_slot, layer);
            if matches!(
                recurrent_state_mode,
                Qwen35PrefillRecurrentStateMode::BackendOwned
            ) {
                let _ = metal_ops.sync_qwen35_slot_buffers_from_kv(qwen_kv, layer, recurrent_slot);
                qwen_kv.mark_layer_state_backend_owned(recurrent_slot, layer);
            }
        } else {
            let conv_generation = qwen_kv.note_backend_conv_state_update(recurrent_slot, layer);
            let recurrent_generation =
                qwen_kv.note_backend_recurrent_state_update(recurrent_slot, layer);
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

        Ok(Some(stats))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn try_run_recurrent_batch_layer_fused_gpu_single_slot(
        metal_ops: &crate::backend::metal::MetalOps,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        recurrent_slot: usize,
        temp_qkv: &MetalBuffer,
        temp_z: &MetalBuffer,
        temp_beta: Option<&MetalBuffer>,
        temp_alpha: Option<&MetalBuffer>,
        recurrent_keys: &Qwen35RecurrentLayerKeys,
        rec_beta_batch: &[f32],
        rec_alpha_batch: &[f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        dims: Qwen35RecurrentDims,
        rms_norm_eps: f32,
        ssm_key: usize,
        ssm_out_dtype: crate::gguf::tensor::GgmlType,
        ffn_nw_key: usize,
        moe_layer: Option<&Qwen35MoeResidentLayerKeys>,
        wg_key: usize,
        wg_dtype: crate::gguf::tensor::GgmlType,
        wu_key: usize,
        wu_dtype: crate::gguf::tensor::GgmlType,
        wd_key: usize,
        wd_dtype: crate::gguf::tensor::GgmlType,
        merged_projection: Option<&FusedProjectionParams<'_>>,
        record_backend_native_batch: bool,
    ) -> anyhow::Result<Option<Qwen35RecurrentGpuPrefillStats>> {
        if n_tokens <= 1
            || qwen_kv.conv_cache_len() > 8
            || !Self::recurrent_batch_from_gpu_qkv_single_slot_check(metal_ops, n_tokens, dims)
                .is_eligible()
        {
            return Ok(None);
        }

        let layer_state_owner = qwen_kv.layer_state_owner(recurrent_slot, layer);
        let recurrent_state_mode =
            Self::resolve_qwen35_prefill_recurrent_state_mode(n_tokens, layer_state_owner);
        if recurrent_state_mode == Qwen35PrefillRecurrentStateMode::CpuAlias {
            return Ok(None);
        }

        let alpha_beta_storage_mode =
            Self::qwen35_prefill_alpha_beta_storage_mode_for_tokens(n_tokens);
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let conv_cache_len_u32 = qwen_kv.conv_cache_len() as u32;
        let nt = n_tokens as u32;
        let total_heads = n_tokens * dims.time_step_rank;
        let total_inner = n_tokens * dims.inner_size;
        let mut stats = Qwen35RecurrentGpuPrefillStats::default();

        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(None);
        };
        let weight_cache = metal_ops.lock_weight_cache();
        let conv_kernel_buf = weight_cache
            .get(&recurrent_keys.conv_kernel)
            .expect("qwen35 conv kernel buffer missing after cache");
        let dt_bias_buf = weight_cache
            .get(&recurrent_keys.dt_bias)
            .expect("qwen35 dt_bias buffer missing after cache");
        let ssm_a_buf = weight_cache
            .get(&recurrent_keys.ssm_a)
            .expect("qwen35 ssm_a buffer missing after cache");
        let ssm_norm_buf = weight_cache
            .get(&recurrent_keys.ssm_norm)
            .expect("qwen35 ssm_norm buffer missing after cache");
        let dt_bias = unsafe { dt_bias_buf.as_slice::<f32>() };
        let ssm_a = unsafe { ssm_a_buf.as_slice::<f32>() };
        let ssm_buf = weight_cache
            .get(&ssm_key)
            .expect("qwen35 ssm_out buffer missing after cache");
        let ffn_nw_buf = weight_cache
            .get(&ffn_nw_key)
            .expect("qwen35 post-attention norm buffer missing after cache");
        let moe_weight_cache = moe_layer.map(|_| metal_ops.lock_moe_weight_cache());
        let wg_buf = moe_layer.is_none().then(|| {
            weight_cache
                .get(&wg_key)
                .expect("qwen35 ffn_gate buffer missing after cache")
        });
        let wu_buf = moe_layer.is_none().then(|| {
            weight_cache
                .get(&wu_key)
                .expect("qwen35 ffn_up buffer missing after cache")
        });
        let wd_buf = moe_layer.is_none().then(|| {
            weight_cache
                .get(&wd_key)
                .expect("qwen35 ffn_down buffer missing after cache")
        });

        let slot_buffer_sync =
            metal_ops.sync_qwen35_slot_buffers_from_kv(qwen_kv, layer, recurrent_slot);
        let elapsed = metal_ops.with_qwen35_qkv_handoff_scratch(
            n_tokens,
            dims.conv_dim(),
            dims.inner_size,
            dims.time_step_rank,
            |scratch| -> anyhow::Result<Duration> {
                metal_ops.with_qwen35_recurrent_slot_buffer_for_kv(
                    qwen_kv,
                    layer,
                    recurrent_slot,
                    conv_state_stride,
                    recurrent_state_stride,
                    |slot_buffers| -> anyhow::Result<Duration> {
                        let t = OpTimer::start();
                        let gdn_scratch_hold2: std::cell::RefCell<Vec<ax_engine_metal::MetalBuffer>> =
                            std::cell::RefCell::new(Vec::new());
                        metal_ops.device.execute_sync(|encoder| {
                            // When merged_projection is provided, encode norm +
                            // input projections into this same CB before the
                            // recurrent body. Eliminates one CB submission.
                            if let Some(proj) = merged_projection {
                                let nw_buf = weight_cache.get(&proj.nw_key).unwrap();
                                metal_ops.elementwise.encode_rms_norm_out_batch(
                                    encoder,
                                    &bs.hidden,
                                    nw_buf,
                                    &bs.norm_buf,
                                    proj.dim as u32,
                                    nt,
                                    proj.eps,
                                );
                                let needs_f16 = proj.gpu_proj_indices.iter().any(|&i| {
                                    Self::qwen35_batch_projection_needs_f16_input(
                                        proj.proj_dtypes[i],
                                    )
                                });
                                if needs_f16 {
                                    metal_ops.elementwise.encode_cast_f32_to_f16(
                                        encoder,
                                        &bs.norm_buf,
                                        &bs.matmul_in_f16,
                                        nt * proj.dim as u32,
                                    );
                                }
                                for &i in proj.gpu_proj_indices {
                                    let w_buf = weight_cache.get(&proj.proj_keys[i]).unwrap();
                                    let out_buf = match i {
                                        0 => temp_qkv,
                                        1 => temp_z,
                                        2 => temp_beta.unwrap(),
                                        3 => temp_alpha.unwrap(),
                                        _ => unreachable!(),
                                    };
                                    Self::encode_qwen35_batch_projection(
                                        metal_ops,
                                        encoder,
                                        w_buf,
                                        &bs.norm_buf,
                                        &bs.matmul_in_f16,
                                        out_buf,
                                        proj.proj_dims[i] as u32,
                                        nt,
                                        proj.dim as u32,
                                        proj.proj_dtypes[i],
                                    );
                                }
                            }
                            metal_ops.gdn.encode_causal_conv_sequence(
                                encoder,
                                temp_qkv,
                                conv_kernel_buf,
                                &slot_buffers.conv_state,
                                &scratch.conv_out,
                                n_tokens as u32,
                                conv_cache_len_u32,
                                dims.conv_dim() as u32,
                            );

                            let encoded = match (temp_alpha, temp_beta, alpha_beta_storage_mode) {
                                (Some(temp_alpha), Some(temp_beta), Qwen35PrefillAlphaBetaStorageMode::F16) => {
                                    metal_ops.elementwise.encode_softplus_bias_mul_batch(
                                        encoder,
                                        temp_alpha,
                                        dt_bias_buf,
                                        ssm_a_buf,
                                        total_heads as u32,
                                        dims.time_step_rank as u32,
                                    );
                                    metal_ops.elementwise.encode_sigmoid_inplace(
                                        encoder,
                                        temp_beta,
                                        total_heads as u32,
                                    );
                                    metal_ops.elementwise.encode_cast_f32_to_f16(
                                        encoder,
                                        temp_alpha,
                                        &scratch.alpha_f16,
                                        total_heads as u32,
                                    );
                                    metal_ops.elementwise.encode_cast_f32_to_f16(
                                        encoder,
                                        temp_beta,
                                        &scratch.beta_f16,
                                        total_heads as u32,
                                    );
                                    metal_ops.gdn.encode_prepare_multi_token_qkv_alpha_beta_f16(
                                        encoder,
                                        &scratch.conv_out,
                                        &scratch.alpha_f16,
                                        &scratch.beta_f16,
                                        &bs.q_buf,
                                        &scratch.k,
                                        &scratch.v,
                                        &bs.gate_buf,
                                        &bs.up_buf,
                                        n_tokens as u32,
                                        dims.group_count as u32,
                                        dims.time_step_rank as u32,
                                        dims.state_size as u32,
                                        rms_norm_eps,
                                    )
                                }
                                (Some(temp_alpha), Some(temp_beta), _) => {
                                    metal_ops.elementwise.encode_softplus_bias_mul_batch(
                                        encoder,
                                        temp_alpha,
                                        dt_bias_buf,
                                        ssm_a_buf,
                                        total_heads as u32,
                                        dims.time_step_rank as u32,
                                    );
                                    metal_ops.elementwise.encode_sigmoid_inplace(
                                        encoder,
                                        temp_beta,
                                        total_heads as u32,
                                    );
                                    metal_ops.gdn.encode_prepare_multi_token_qkv(
                                        encoder,
                                        &scratch.conv_out,
                                        temp_alpha,
                                        temp_beta,
                                        &bs.q_buf,
                                        &scratch.k,
                                        &scratch.v,
                                        &bs.gate_buf,
                                        &bs.up_buf,
                                        n_tokens as u32,
                                        dims.group_count as u32,
                                        dims.time_step_rank as u32,
                                        dims.state_size as u32,
                                        rms_norm_eps,
                                    )
                                }
                                (_, _, Qwen35PrefillAlphaBetaStorageMode::F16) => {
                                    let alpha_slice = unsafe {
                                        &mut scratch.alpha.as_mut_slice::<f32>()[..total_heads]
                                    };
                                    let beta_slice = unsafe {
                                        &mut scratch.beta.as_mut_slice::<f32>()[..total_heads]
                                    };
                                    Self::prepare_qwen35_handoff_alpha_beta(
                                        alpha_slice,
                                        beta_slice,
                                        &rec_alpha_batch[..total_heads],
                                        &rec_beta_batch[..total_heads],
                                        dt_bias,
                                        ssm_a,
                                    );
                                    metal_ops.elementwise.encode_cast_f32_to_f16(
                                        encoder,
                                        &scratch.alpha,
                                        &scratch.alpha_f16,
                                        total_heads as u32,
                                    );
                                    metal_ops.elementwise.encode_cast_f32_to_f16(
                                        encoder,
                                        &scratch.beta,
                                        &scratch.beta_f16,
                                        total_heads as u32,
                                    );
                                    metal_ops.gdn.encode_prepare_multi_token_qkv_alpha_beta_f16(
                                        encoder,
                                        &scratch.conv_out,
                                        &scratch.alpha_f16,
                                        &scratch.beta_f16,
                                        &bs.q_buf,
                                        &scratch.k,
                                        &scratch.v,
                                        &bs.gate_buf,
                                        &bs.up_buf,
                                        n_tokens as u32,
                                        dims.group_count as u32,
                                        dims.time_step_rank as u32,
                                        dims.state_size as u32,
                                        rms_norm_eps,
                                    )
                                }
                                _ => {
                                    let alpha_slice = unsafe {
                                        &mut scratch.alpha.as_mut_slice::<f32>()[..total_heads]
                                    };
                                    let beta_slice = unsafe {
                                        &mut scratch.beta.as_mut_slice::<f32>()[..total_heads]
                                    };
                                    Self::prepare_qwen35_handoff_alpha_beta(
                                        alpha_slice,
                                        beta_slice,
                                        &rec_alpha_batch[..total_heads],
                                        &rec_beta_batch[..total_heads],
                                        dt_bias,
                                        ssm_a,
                                    );
                                    metal_ops.gdn.encode_prepare_multi_token_qkv(
                                        encoder,
                                        &scratch.conv_out,
                                        &scratch.alpha,
                                        &scratch.beta,
                                        &bs.q_buf,
                                        &scratch.k,
                                        &scratch.v,
                                        &bs.gate_buf,
                                        &bs.up_buf,
                                        n_tokens as u32,
                                        dims.group_count as u32,
                                        dims.time_step_rank as u32,
                                        dims.state_size as u32,
                                        rms_norm_eps,
                                    )
                                }
                            };
                            anyhow::ensure!(
                                encoded,
                                "qwen35 fused recurrent layer Metal pack kernel does not support this shape"
                            );
                            *gdn_scratch_hold2.borrow_mut() = metal_ops.gdn.encode_gated_delta_sequence_auto(
                                encoder,
                                &metal_ops.elementwise,
                                &metal_ops.device,
                                &bs.q_buf,
                                &scratch.k,
                                &scratch.v,
                                &bs.gate_buf,
                                &bs.up_buf,
                                &slot_buffers.recurrent_state,
                                temp_qkv,
                                n_tokens as u32,
                                dims.time_step_rank as u32,
                                dims.state_size as u32,
                            );
                            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                                encoder,
                                temp_qkv,
                                ssm_norm_buf,
                                nt,
                                dims.time_step_rank as u32,
                                dims.state_size as u32,
                                rms_norm_eps,
                            );
                            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                                encoder,
                                temp_z,
                                temp_qkv,
                                dims.inner_size as u32,
                                nt,
                            );
                            if Self::qwen35_batch_projection_needs_f16_input(ssm_out_dtype) {
                                metal_ops.elementwise.encode_cast_f32_to_f16(
                                    encoder,
                                    temp_z,
                                    &bs.matmul_in_f16,
                                    total_inner as u32,
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
                            if let Some(moe_layer) = moe_layer {
                                let moe_cache = moe_weight_cache.as_ref().unwrap();
                                let moe_scratch =
                                    crate::backend::metal::MoeBatchScratchView::from_batch_scratches(
                                        bs,
                                    )?;
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
                                metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                                    encoder,
                                    moe_scratch,
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
                                    rms_norm_eps,
                                    shared_expert.as_ref(),
                                    false,
                                )?;
                            } else {
                                metal_ops
                                    .elementwise
                                    .encode_residual_add_rms_norm_out_batch(
                                        encoder,
                                        &bs.hidden,
                                        &bs.attn_out,
                                        ffn_nw_buf,
                                        &bs.norm_buf,
                                        dim as u32,
                                        nt,
                                        rms_norm_eps,
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
                        })?;
                        Ok(t.elapsed())
                    },
                )
            },
        )?;
        stats.gpu_execute += elapsed;
        metal_ops.record_qwen35_recurrent_batch_qkv_handoff_gpu(elapsed);
        metal_ops.record_qwen35_recurrent_batch_qkv_handoff();
        metal_ops.record_qwen35_recurrent_batch_qkv_handoff_fused_tail();
        if record_backend_native_batch {
            metal_ops.record_qwen35_recurrent_batch_state_batch_backend_native();
        }
        metal_ops.record_qwen35_recurrent_batch_qkv_handoff_slot_buffer();
        if slot_buffer_sync.used_backend_carryover {
            metal_ops.record_qwen35_recurrent_batch_qkv_handoff_backend_carryover();
        }
        if slot_buffer_sync.used_backend_zero_init {
            metal_ops.record_qwen35_recurrent_batch_qkv_handoff_backend_zero_init();
        }
        if slot_buffer_sync.used_cpu_materialization {
            metal_ops.record_qwen35_recurrent_batch_qkv_handoff_cpu_materialization();
        }

        let _ = qwen_kv.note_backend_conv_state_update(recurrent_slot, layer);
        let _ = qwen_kv.note_backend_recurrent_state_update(recurrent_slot, layer);
        if !qwen_kv.has_gpu_recurrent_state() {
            let conv_generation = qwen_kv.conv_state_generation(recurrent_slot, layer);
            let recurrent_generation = qwen_kv.recurrent_state_generation(recurrent_slot, layer);
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

        Ok(Some(stats))
    }
    #[allow(clippy::too_many_arguments)]
    fn run_recurrent_batch_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        recurrent_slot: usize,
        layer: usize,
        batch_position: usize,
        dims: Qwen35RecurrentDims,
        recurrent_slot_indices: &[usize],
        norm_buf: &[f32],
        rec_qkv_batch: &mut [f32],
        rec_z_batch: &mut [f32],
        rec_beta_batch: &mut [f32],
        rec_alpha_batch: &mut [f32],
        rec_out_batch: &mut [f32],
        proj_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        skip_input_projections: bool,
    ) -> anyhow::Result<()> {
        Self::validate_recurrent_layer_state(qwen_kv, recurrent_slot, layer, "prefill")?;
        let slot_count = recurrent_slot_indices.len();
        let total_tokens = n_tokens * slot_count;
        let total_conv = total_tokens * dims.conv_dim();
        let total_inner = total_tokens * dims.inner_size;
        let total_rank = total_tokens * dims.time_step_rank;
        if !skip_input_projections {
            let input_ops = Self::recurrent_input_ops(weights, prefix, dims)?;
            Self::project_recurrent_inputs(
                input_ops,
                [rec_qkv_batch, rec_z_batch, rec_beta_batch, rec_alpha_batch],
                |raw, dtype, rows, out| {
                    Self::batched_dequant_matmul_token_major(
                        backend, raw, dtype, norm_buf, out, n_tokens, rows, dim,
                    );
                },
            );
        }
        Self::repeat_token_major_batch_in_place(
            rec_qkv_batch,
            n_tokens,
            dims.conv_dim(),
            slot_count,
        );
        Self::repeat_token_major_batch_in_place(rec_z_batch, n_tokens, dims.inner_size, slot_count);
        Self::repeat_token_major_batch_in_place(
            rec_beta_batch,
            n_tokens,
            dims.time_step_rank,
            slot_count,
        );
        Self::repeat_token_major_batch_in_place(
            rec_alpha_batch,
            n_tokens,
            dims.time_step_rank,
            slot_count,
        );
        Self::assert_finite_if_enabled(
            "recurrent_qkv_input_batch",
            &rec_qkv_batch[..total_conv],
            layer,
            batch_position,
        )?;

        rec_out_batch[..total_inner].fill(0.0);
        let runtime = Self::run_recurrent_sequence(
            backend,
            weights,
            prefix,
            qwen_kv,
            layer,
            recurrent_slot_indices,
            dims,
            cfg.rms_norm_eps,
            &rec_qkv_batch[..total_conv],
            &mut rec_beta_batch[..total_rank],
            &mut rec_alpha_batch[..total_rank],
            &mut rec_out_batch[..total_inner],
            n_tokens,
        )?;
        Self::assert_finite_if_enabled(
            "recurrent_kernel_output_batch",
            &rec_out_batch[..total_inner],
            layer,
            batch_position,
        )?;

        let used_gpu_finalize = Self::try_finalize_recurrent_output_batch_gpu(
            cfg,
            backend,
            &mut rec_out_batch[..total_inner],
            &rec_z_batch[..total_inner],
            total_tokens,
            dims,
            runtime.ssm_norm,
            cfg.rms_norm_eps,
        )?;
        if !used_gpu_finalize {
            Self::finalize_recurrent_output_batch(
                &mut rec_out_batch[..total_inner],
                &rec_z_batch[..total_inner],
                total_tokens,
                dims,
                runtime.ssm_norm,
                cfg.rms_norm_eps,
            );
        }
        Self::assert_finite_if_enabled(
            "recurrent_output_batch",
            &rec_out_batch[..total_inner],
            layer,
            batch_position,
        )?;

        let (ssm_out_raw, ssm_out_dtype, _) = Self::recurrent_output_op(weights, prefix, dim)?;
        let active_rec_out = Self::slot_batch_slice(
            &rec_out_batch[..total_inner],
            recurrent_slot_indices,
            recurrent_slot,
            n_tokens,
            dims.inner_size,
        );
        Self::batched_dequant_matmul_token_major(
            backend,
            ssm_out_raw,
            ssm_out_dtype,
            active_rec_out,
            proj_buf,
            n_tokens,
            dim,
            dims.inner_size,
        );
        Self::assert_finite_if_enabled("recurrent_proj_batch", proj_buf, layer, batch_position)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_recurrent_single_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        recurrent_slot: usize,
        layer: usize,
        position: usize,
        dims: Qwen35RecurrentDims,
        recurrent_slot_indices: &[usize],
        norm_buf: &[f32],
        rec_qkv: &mut [f32],
        rec_z: &mut [f32],
        rec_beta: &mut [f32],
        rec_alpha: &mut [f32],
        rec_out: &mut [f32],
        proj_buf: &mut [f32],
        dim: usize,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        Self::validate_recurrent_layer_state(qwen_kv, recurrent_slot, layer, "decode")?;
        let input_ops = Self::recurrent_input_ops(weights, prefix, dims)?;
        timed_matmul_bucket!(ops, matmul_input_proj, {
            let mut outputs = [&mut *rec_qkv, &mut *rec_z, &mut *rec_beta, &mut *rec_alpha];
            Self::decode_project_ops_gpu_safe(backend, &input_ops, norm_buf, dim, &mut outputs);
        });
        Self::assert_finite_if_enabled("recurrent_qkv_input", rec_qkv, layer, position)?;
        Self::assert_finite_if_enabled(
            "recurrent_state_before_decode",
            qwen_kv.recurrent_state_for_slot(recurrent_slot, layer),
            layer,
            position,
        )?;

        // Recurrent state staging is backend-owned now so the model does not
        // need to know how state is mirrored.
        let runtime = timed!(
            ops,
            recurrent,
            Self::run_recurrent_sequence(
                backend,
                weights,
                prefix,
                qwen_kv,
                layer,
                recurrent_slot_indices,
                dims,
                cfg.rms_norm_eps,
                rec_qkv,
                rec_beta,
                rec_alpha,
                rec_out,
                1,
            )
        )?;
        Self::assert_finite_if_enabled("recurrent_kernel_output", rec_out, layer, position)?;

        Self::finalize_recurrent_output(rec_out, rec_z, dims, runtime.ssm_norm, cfg.rms_norm_eps);
        Self::assert_finite_if_enabled("recurrent_output", rec_out, layer, position)?;

        let (ssm_out_raw, ssm_out_dtype, _) = timed!(
            ops,
            dequant,
            Self::recurrent_output_op(weights, prefix, dim)?
        );
        timed_matmul_bucket!(
            ops,
            matmul_output_proj,
            Self::decode_dequant_matmul_gpu_safe(
                backend,
                ssm_out_raw,
                ssm_out_dtype,
                rec_out,
                proj_buf,
                dim,
                dims.inner_size,
            )
        );
        Self::assert_finite_if_enabled("recurrent_proj", proj_buf, layer, position)?;
        Ok(())
    }

    /// GPU-encoded layer tail: residual + FFN norm + gate/up matmul + SiLU +
    /// down matmul + final residual, all in one Metal command buffer.
    ///
    /// Returns `true` if the GPU path was used, `false` to fall back to CPU.
    #[allow(clippy::too_many_arguments)]
    fn try_apply_layer_tail_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        proj_buf: &[f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<bool> {
        if Self::qwen35_is_moe(cfg) && Self::qwen35_layer_uses_moe(weights, prefix) {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };
        let input_ops = Self::ffn_input_ops(weights, prefix, inter_dim)?;
        let (wd_raw, wd_dtype, _) = Self::ffn_down_op(weights, prefix, dim)?;
        let post_attn_norm_w =
            weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;

        // Check all dtypes are supported by fused batch kernels.
        let (wg_raw, wg_dtype, _) = input_ops[0];
        let (wu_raw, wu_dtype, _) = input_ops[1];
        let supported = |dt: crate::gguf::tensor::GgmlType| {
            matches!(
                dt,
                crate::gguf::tensor::GgmlType::Q4K
                    | crate::gguf::tensor::GgmlType::Q5K
                    | crate::gguf::tensor::GgmlType::Q6K
            )
        };
        if !supported(wg_dtype) || !supported(wu_dtype) || !supported(wd_dtype) {
            return Ok(false);
        }

        // Cache weight buffers.
        let wg_key = metal_ops.ensure_quant_cached(wg_raw);
        let wu_key = metal_ops.ensure_quant_cached(wu_raw);
        let wd_key = metal_ops.ensure_quant_cached(wd_raw);
        let nw_key = metal_ops.ensure_f32_cached(post_attn_norm_w);

        metal_ops.init_batch_scratches(cfg, n_tokens);

        let nt = n_tokens as u32;
        let eps = rms_norm_eps;

        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        // Copy hidden + proj to GPU.
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(hidden);
            bs.proj_buf.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(proj_buf);
        }

        let weight_cache = metal_ops.lock_weight_cache();
        let nw_buf = weight_cache.get(&nw_key).unwrap();
        let wg_buf = weight_cache.get(&wg_key).unwrap();
        let wu_buf = weight_cache.get(&wu_key).unwrap();
        let wd_buf = weight_cache.get(&wd_key).unwrap();

        metal_ops.device.execute_sync(|encoder| {
            // 1. Residual add: hidden += proj_buf
            metal_ops.elementwise.encode_elementwise_add_batch(
                encoder,
                &bs.hidden,
                &bs.proj_buf,
                dim as u32,
                nt,
            );
            // 2. RMSNorm: norm_buf = RMSNorm(hidden)
            metal_ops.elementwise.encode_rms_norm_out_batch(
                encoder,
                &bs.hidden,
                nw_buf,
                &bs.norm_buf,
                dim as u32,
                nt,
                eps,
            );
            // 3. Gate matmul: gate_buf = dequant(Wg) × norm_buf^T
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
                wg_dtype,
                false,
                false,
                false,
            );
            // 4. Up matmul: up_buf = dequant(Wu) × norm_buf^T
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
                wu_dtype,
                false,
                false,
                false,
            );
            // 5. SiLU: gate_buf = SiLU(gate_buf) * up_buf
            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                encoder,
                &bs.gate_buf,
                &bs.up_buf,
                inter_dim as u32,
                nt,
            );
            // 6. Down matmul: proj_buf = dequant(Wd) × gate_buf^T
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
                wd_dtype,
                false,
                false,
                false,
            );
            // 7. Final residual: hidden += proj_buf (down output)
            metal_ops.elementwise.encode_elementwise_add_batch(
                encoder,
                &bs.hidden,
                &bs.proj_buf,
                dim as u32,
                nt,
            );
            Ok(())
        })?;
        drop(weight_cache);

        // Read back updated hidden.
        let result = unsafe { &bs.hidden.as_slice::<f32>()[..n_tokens * dim] };
        hidden.copy_from_slice(result);
        drop(bs_guard);

        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_layer_tail_batch(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        proj_buf: &[f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        layer: usize,
        batch_position: usize,
    ) -> anyhow::Result<()> {
        // Try GPU-encoded path first (single command buffer for entire FFN).
        if n_tokens > 1
            && let Ok(true) = Self::try_apply_layer_tail_batch_gpu(
                cfg,
                backend,
                weights,
                prefix,
                hidden,
                proj_buf,
                n_tokens,
                dim,
                inter_dim,
                rms_norm_eps,
            )
        {
            Self::assert_finite_if_enabled("post_ffn_hidden_batch", hidden, layer, batch_position)?;
            return Ok(());
        }
        // CPU fallback.
        silu::elementwise_add(hidden, proj_buf);
        Self::assert_finite_if_enabled("layer_hidden_batch", hidden, layer, batch_position)?;
        Self::apply_post_attention_ffn_batch(
            cfg,
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
            rms_norm_eps,
        )?;
        Self::assert_finite_if_enabled("post_ffn_hidden_batch", hidden, layer, batch_position)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_layer_tail_single(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        proj_buf: &[f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        layer: usize,
        position: usize,
        ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        silu::elementwise_add(hidden, proj_buf);
        Self::assert_finite_if_enabled("layer_hidden", hidden, layer, position)?;
        Self::apply_post_attention_ffn_single(
            cfg,
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
            rms_norm_eps,
            ops,
        )
    }
}
