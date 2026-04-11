use super::*;

pub struct GdnKernels {
    causal_conv_sequence_f32: ComputePipeline,
    causal_conv_sequence_parallel_f32: ComputePipeline,
    prepare_single_token_qkv_f32: ComputePipeline,
    prepare_multi_token_qk_f32: ComputePipeline,
    prepare_multi_token_vgb_f32: ComputePipeline,
    prepare_multi_token_vgb_ab_f16: ComputePipeline,
    unpack_bhsk_to_token_major_f32: ComputePipeline,
    single_token_gated_delta_fused_128_64: ComputePipeline,
    single_token_gated_delta_fused_64_64: ComputePipeline,
    gated_delta_128_64: ComputePipeline,
    #[allow(dead_code)]
    gated_delta_128_128: ComputePipeline,
    gated_delta_64_64: ComputePipeline,
    gated_delta_fallback: ComputePipeline,
    #[allow(dead_code)]
    chunked_gated_delta_32_128_64: ComputePipeline,
    chunked_gated_delta_32_128_128: ComputePipeline,
    chunked_gated_delta_32_64_64: ComputePipeline,
    simd_gated_delta_4: ComputePipeline,
    // Chunked GDN graph building blocks (llama.cpp-style decomposition).
    cumsum_f32: ComputePipeline,
    solve_tri_lower_f32_1: ComputePipeline,
    solve_tri_lower_f32_4: ComputePipeline,
    tri_lower_diag_f32: ComputePipeline,
    tri_lower_strict_f32: ComputePipeline,
    diag_identity_f32: ComputePipeline,
    batched_matmul_f32: ComputePipeline,
    batched_matmul_atrans_f32: ComputePipeline,
    batched_matmul_btrans_f32: ComputePipeline,
    broadcast_mul_f32: ComputePipeline,
    build_decay_mask_f32: ComputePipeline,
    extract_last_cumsum_f32: ComputePipeline,
    build_g_diff_exp_f32: ComputePipeline,
    gather_stride_f32: ComputePipeline,
    strided_sub_f32: ComputePipeline,
    strided_copy_f32: ComputePipeline,
}

impl GdnKernels {
    pub fn new(device: &MetalDevice) -> anyhow::Result<Self> {
        let causal_conv_sequence_f32 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "qwen35_causal_conv_sequence_f32",
        )
        .context("Failed to compile qwen35_causal_conv_sequence_f32 kernel")?;
        let causal_conv_sequence_parallel_f32 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "qwen35_causal_conv_sequence_parallel_f32",
        )
        .context("Failed to compile qwen35_causal_conv_sequence_parallel_f32 kernel")?;
        let prepare_single_token_qkv_f32 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "qwen35_prepare_single_token_gdn_qkv_f32",
        )
        .context("Failed to compile qwen35_prepare_single_token_gdn_qkv_f32 kernel")?;
        let prepare_multi_token_qk_f32 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "qwen35_prepare_multi_token_gdn_qk_f32",
        )
        .context("Failed to compile qwen35_prepare_multi_token_gdn_qk_f32 kernel")?;
        let prepare_multi_token_vgb_f32 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "qwen35_prepare_multi_token_gdn_vgb_f32",
        )
        .context("Failed to compile qwen35_prepare_multi_token_gdn_vgb_f32 kernel")?;
        let prepare_multi_token_vgb_ab_f16 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "qwen35_prepare_multi_token_gdn_vgb_ab_f16",
        )
        .context("Failed to compile qwen35_prepare_multi_token_gdn_vgb_ab_f16 kernel")?;
        let unpack_bhsk_to_token_major_f32 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "qwen35_unpack_bhsk_to_token_major_f32",
        )
        .context("Failed to compile qwen35_unpack_bhsk_to_token_major_f32 kernel")?;
        let single_token_gated_delta_fused_128_64 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "qwen35_single_token_gated_delta_fused_128_64",
        )
        .context("Failed to compile qwen35_single_token_gated_delta_fused_128_64 kernel")?;
        let single_token_gated_delta_fused_64_64 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "qwen35_single_token_gated_delta_fused_64_64",
        )
        .context("Failed to compile qwen35_single_token_gated_delta_fused_64_64 kernel")?;
        let gated_delta_128_64 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "gated_delta_rule_128_64",
        )
        .context("Failed to compile gated_delta_rule_128_64 kernel")?;
        let gated_delta_128_128 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "gated_delta_rule_128_128",
        )
        .context("Failed to compile gated_delta_rule_128_128 kernel")?;
        let gated_delta_64_64 =
            ComputePipeline::from_source(device.device(), GDN_SHADER_SRC, "gated_delta_rule_64_64")
                .context("Failed to compile gated_delta_rule_64_64 kernel")?;
        let gated_delta_fallback = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "gated_delta_rule_fallback",
        )
        .context("Failed to compile gated_delta_rule_fallback kernel")?;
        let chunked_gated_delta_32_128_64 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "chunked_gated_delta_rule_32_128_64",
        )
        .context("Failed to compile chunked_gated_delta_rule_32_128_64 kernel")?;
        let chunked_gated_delta_32_128_128 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "chunked_gated_delta_rule_32_128_128",
        )
        .context("Failed to compile chunked_gated_delta_rule_32_128_128 kernel")?;
        let chunked_gated_delta_32_64_64 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "chunked_gated_delta_rule_32_64_64",
        )
        .context("Failed to compile chunked_gated_delta_rule_32_64_64 kernel")?;
        let simd_gated_delta_4 = ComputePipeline::from_source(
            device.device(),
            GDN_SHADER_SRC,
            "gated_delta_rule_simd_4",
        )
        .context("Failed to compile gated_delta_rule_simd_4 kernel")?;
        Ok(Self {
            causal_conv_sequence_f32,
            causal_conv_sequence_parallel_f32,
            prepare_single_token_qkv_f32,
            prepare_multi_token_qk_f32,
            prepare_multi_token_vgb_f32,
            prepare_multi_token_vgb_ab_f16,
            unpack_bhsk_to_token_major_f32,
            single_token_gated_delta_fused_128_64,
            single_token_gated_delta_fused_64_64,
            gated_delta_128_64,
            gated_delta_128_128,
            gated_delta_64_64,
            gated_delta_fallback,
            chunked_gated_delta_32_128_64,
            chunked_gated_delta_32_128_128,
            chunked_gated_delta_32_64_64,
            simd_gated_delta_4,
            cumsum_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_cumsum_f32",
            )
            .context("Failed to compile gdn_cumsum_f32 kernel")?,
            solve_tri_lower_f32_1: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_solve_tri_lower_f32_1",
            )
            .context("Failed to compile gdn_solve_tri_lower_f32_1 kernel")?,
            solve_tri_lower_f32_4: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_solve_tri_lower_f32_4",
            )
            .context("Failed to compile gdn_solve_tri_lower_f32_4 kernel")?,
            tri_lower_diag_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_tri_lower_diag_f32",
            )
            .context("Failed to compile gdn_tri_lower_diag_f32 kernel")?,
            tri_lower_strict_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_tri_lower_strict_f32",
            )
            .context("Failed to compile gdn_tri_lower_strict_f32 kernel")?,
            diag_identity_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_diag_identity_f32",
            )
            .context("Failed to compile gdn_diag_identity_f32 kernel")?,
            batched_matmul_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_batched_matmul_f32",
            )
            .context("Failed to compile gdn_batched_matmul_f32 kernel")?,
            batched_matmul_atrans_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_batched_matmul_atrans_f32",
            )
            .context("Failed to compile gdn_batched_matmul_atrans_f32 kernel")?,
            batched_matmul_btrans_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_batched_matmul_btrans_f32",
            )
            .context("Failed to compile gdn_batched_matmul_btrans_f32 kernel")?,
            broadcast_mul_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_broadcast_mul_f32",
            )
            .context("Failed to compile gdn_broadcast_mul_f32 kernel")?,
            build_decay_mask_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_build_decay_mask_f32",
            )
            .context("Failed to compile gdn_build_decay_mask_f32 kernel")?,
            extract_last_cumsum_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_extract_last_cumsum_f32",
            )
            .context("Failed to compile gdn_extract_last_cumsum_f32 kernel")?,
            build_g_diff_exp_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_build_g_diff_exp_f32",
            )
            .context("Failed to compile gdn_build_g_diff_exp_f32 kernel")?,
            gather_stride_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_gather_stride_f32",
            )
            .context("Failed to compile gdn_gather_stride_f32 kernel")?,
            strided_sub_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_strided_sub_f32",
            )
            .context("Failed to compile gdn_strided_sub_f32 kernel")?,
            strided_copy_f32: ComputePipeline::from_source(
                device.device(),
                GDN_SHADER_SRC,
                "gdn_strided_copy_f32",
            )
            .context("Failed to compile gdn_strided_copy_f32 kernel")?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Encode causal conv into an existing command encoder (no execute_sync).
    pub fn encode_causal_conv_sequence(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &MetalBuffer,
        kernel: &MetalBuffer,
        conv_state: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        conv_cache_len: u32,
        conv_dim: u32,
    ) {
        if seq_len > 1 && seq_len >= conv_cache_len {
            // Parallel kernel: one thread per (channel, token). No sequential
            // dependency — each output token reads conv_cache_len prior inputs
            // directly from the input buffer (or from conv_state for the first
            // few tokens). Requires seq_len >= conv_cache_len for state writeback.
            let tg_x = 256.min(conv_dim as usize);
            let groups_x = (conv_dim as usize).div_ceil(tg_x);
            crate::set_pipeline_cached(encoder, self.causal_conv_sequence_parallel_f32.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(kernel.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(conv_state.mtl_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 3);
            }
            bind_u32(encoder, 4, seq_len);
            bind_u32(encoder, 5, conv_cache_len);
            bind_u32(encoder, 6, conv_dim);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: seq_len as _,
                    depth: 1,
                },
                MTLSize {
                    width: tg_x,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            // Sequential kernel for decode (seq_len=1): one thread per channel.
            let groups_x = (conv_dim as usize).div_ceil(256);
            crate::set_pipeline_cached(encoder, self.causal_conv_sequence_f32.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(kernel.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(conv_state.mtl_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 3);
            }
            bind_u32(encoder, 4, seq_len);
            bind_u32(encoder, 5, conv_cache_len);
            bind_u32(encoder, 6, conv_dim);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }

    /// Dispatch causal conv in its own command buffer (legacy wrapper).
    #[deprecated(note = "Use encode_causal_conv_sequence with a caller-managed command buffer.")]
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv_sequence(
        &self,
        device: &MetalDevice,
        input: &MetalBuffer,
        kernel: &MetalBuffer,
        conv_state: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        conv_cache_len: u32,
        conv_dim: u32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            conv_cache_len <= 8,
            "qwen35 causal conv Metal kernel supports conv_cache_len <= 8, got {conv_cache_len}",
        );
        device.execute_sync(|encoder| {
            self.encode_causal_conv_sequence(
                encoder,
                input,
                kernel,
                conv_state,
                output,
                seq_len,
                conv_cache_len,
                conv_dim,
            );
            Ok(())
        })
    }

    /// Encode dst[token, i] = src[token, i] * scale[token].
    pub fn encode_broadcast_mul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        scale: &MetalBuffer,
        dst: &MetalBuffer,
        dim: u32,
        total: u32,
    ) {
        let dims = DispatchDims::d1(total as usize, 256);
        crate::set_pipeline_cached(encoder, self.broadcast_mul_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scale.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, dim);
        bind_u32(encoder, 4, total);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_prepare_single_token_qkv(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        conv_out: &MetalBuffer,
        q_out: &MetalBuffer,
        k_out: &MetalBuffer,
        v_out: &MetalBuffer,
        group_count: u32,
        time_step_rank: u32,
        state_size: u32,
        eps: f32,
    ) {
        let dims = DispatchDims::d1((time_step_rank * state_size) as usize, 64);
        crate::set_pipeline_cached(encoder, self.prepare_single_token_qkv_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(conv_out.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q_out.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k_out.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v_out.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, group_count);
        bind_u32(encoder, 5, time_step_rank);
        bind_u32(encoder, 6, state_size);
        bind_f32(encoder, 7, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_prepare_multi_token_qkv(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        conv_out: &MetalBuffer,
        alpha_in: &MetalBuffer,
        beta_in: &MetalBuffer,
        q_out: &MetalBuffer,
        k_out: &MetalBuffer,
        v_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        beta_out: &MetalBuffer,
        n_tokens: u32,
        group_count: u32,
        time_step_rank: u32,
        state_size: u32,
        eps: f32,
    ) -> bool {
        if state_size == 0 || state_size > 256 || group_count == 0 {
            return false;
        }
        if !time_step_rank.is_multiple_of(group_count) {
            return false;
        }

        crate::set_pipeline_cached(encoder, self.prepare_multi_token_qk_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(conv_out.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q_out.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k_out.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n_tokens);
        bind_u32(encoder, 4, group_count);
        bind_u32(encoder, 5, time_step_rank);
        bind_u32(encoder, 6, state_size);
        bind_f32(encoder, 7, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: (n_tokens * group_count) as usize,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        let vgb_dims = DispatchDims::d2(
            state_size as usize,
            (n_tokens * time_step_rank) as usize,
            128,
            1,
        );
        crate::set_pipeline_cached(encoder, self.prepare_multi_token_vgb_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(conv_out.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(alpha_in.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(beta_in.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v_out.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(gate_out.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(beta_out.mtl_buffer()), 0, 5);
        }
        bind_u32(encoder, 6, n_tokens);
        bind_u32(encoder, 7, group_count);
        bind_u32(encoder, 8, time_step_rank);
        bind_u32(encoder, 9, state_size);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            vgb_dims.threadgroups,
            vgb_dims.threads_per_threadgroup,
        );
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_prepare_multi_token_qkv_alpha_beta_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        conv_out: &MetalBuffer,
        alpha_in: &MetalBuffer,
        beta_in: &MetalBuffer,
        q_out: &MetalBuffer,
        k_out: &MetalBuffer,
        v_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        beta_out: &MetalBuffer,
        n_tokens: u32,
        group_count: u32,
        time_step_rank: u32,
        state_size: u32,
        eps: f32,
    ) -> bool {
        if state_size == 0 || state_size > 256 || group_count == 0 {
            return false;
        }
        if !time_step_rank.is_multiple_of(group_count) {
            return false;
        }

        crate::set_pipeline_cached(encoder, self.prepare_multi_token_qk_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(conv_out.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q_out.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k_out.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n_tokens);
        bind_u32(encoder, 4, group_count);
        bind_u32(encoder, 5, time_step_rank);
        bind_u32(encoder, 6, state_size);
        bind_f32(encoder, 7, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: (n_tokens * group_count) as usize,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        let vgb_dims = DispatchDims::d2(
            state_size as usize,
            (n_tokens * time_step_rank) as usize,
            128,
            1,
        );
        crate::set_pipeline_cached(encoder, self.prepare_multi_token_vgb_ab_f16.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(conv_out.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(alpha_in.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(beta_in.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v_out.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(gate_out.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(beta_out.mtl_buffer()), 0, 5);
        }
        bind_u32(encoder, 6, n_tokens);
        bind_u32(encoder, 7, group_count);
        bind_u32(encoder, 8, time_step_rank);
        bind_u32(encoder, 9, state_size);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            vgb_dims.threadgroups,
            vgb_dims.threads_per_threadgroup,
        );
        true
    }

    /// Dispatch multi-token QKV preparation in its own command buffer (legacy wrapper).
    #[deprecated(note = "Use encode_prepare_multi_token_qkv with a caller-managed command buffer.")]
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_multi_token_qkv(
        &self,
        device: &MetalDevice,
        conv_out: &MetalBuffer,
        alpha_in: &MetalBuffer,
        beta_in: &MetalBuffer,
        q_out: &MetalBuffer,
        k_out: &MetalBuffer,
        v_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        beta_out: &MetalBuffer,
        n_tokens: u32,
        group_count: u32,
        time_step_rank: u32,
        state_size: u32,
        eps: f32,
    ) -> anyhow::Result<bool> {
        let mut encoded = false;
        device.execute_sync(|encoder| {
            encoded = self.encode_prepare_multi_token_qkv(
                encoder,
                conv_out,
                alpha_in,
                beta_in,
                q_out,
                k_out,
                v_out,
                gate_out,
                beta_out,
                n_tokens,
                group_count,
                time_step_rank,
                state_size,
                eps,
            );
            Ok(())
        })?;
        Ok(encoded)
    }

    pub fn encode_unpack_bhsk_to_token_major(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        head_dim: u32,
    ) {
        let dims = DispatchDims::d2(head_dim as usize, (n_tokens * n_heads) as usize, 128, 1);
        crate::set_pipeline_cached(encoder, self.unpack_bhsk_to_token_major_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n_tokens);
        bind_u32(encoder, 3, n_heads);
        bind_u32(encoder, 4, head_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    pub fn unpack_bhsk_to_token_major(
        &self,
        device: &MetalDevice,
        input: &MetalBuffer,
        output: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_unpack_bhsk_to_token_major(
                encoder, input, output, n_tokens, n_heads, head_dim,
            );
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_single_token_gated_delta_fused(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        conv_out: &MetalBuffer,
        gate: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        output: &MetalBuffer,
        group_count: u32,
        time_step_rank: u32,
        head_dim: u32,
        eps: f32,
    ) -> bool {
        let pipeline = match head_dim {
            128 => &self.single_token_gated_delta_fused_128_64,
            64 => &self.single_token_gated_delta_fused_64_64,
            _ => return false,
        };
        let bv = 64usize;
        let grid_x = (head_dim as usize).div_ceil(bv);
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(conv_out.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(beta.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(state.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, group_count);
        bind_u32(encoder, 6, time_step_rank);
        bind_f32(encoder, 7, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: grid_x,
                height: time_step_rank as usize,
                depth: 1,
            },
            MTLSize {
                width: bv,
                height: 1,
                depth: 1,
            },
        );
        true
    }

    /// Encode gated delta net into an existing command encoder (no execute_sync).
    ///
    /// When `device` is `Some` and `AX_METAL_GDN_CHUNKED_GRAPH=1`, uses the
    /// graph-based chunked decomposition (cumsum + matmul + solve_tri) instead
    /// of the fused kernel. The device is needed to allocate scratch buffers.
    #[allow(clippy::too_many_arguments)]
    /// Encode gated delta sequence, optionally using the graph-based chunked
    /// decomposition when `AX_METAL_GDN_CHUNKED_GRAPH=1`.
    #[allow(clippy::too_many_arguments)]
    /// Encode GDN with auto-selection between chunked graph (parallel intra-chunk
    /// matmuls, O(n_chunks) sequential steps) and sequential kernel (O(seq_len) steps).
    ///
    /// Returns scratch buffers that MUST be kept alive until the command buffer
    /// completes execution. Dropping them early causes a Metal validation error
    /// (`command buffer references deallocated object`).
    pub fn encode_gated_delta_sequence_auto(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        elementwise: &ElementwiseKernels,
        device: &MetalDevice,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        g: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
    ) -> Vec<MetalBuffer> {
        if Self::chunked_graph_enabled() && seq_len >= 64 && head_dim >= 64 {
            return self.encode_chunked_gdn_graph(
                encoder,
                elementwise,
                device,
                q,
                k,
                v,
                g,
                beta,
                state,
                output,
                seq_len,
                n_heads,
                head_dim,
            );
        }
        self.encode_gated_delta_sequence(
            encoder, q, k, v, g, beta, state, output, seq_len, n_heads, head_dim,
        );
        Vec::new()
    }

    /// Encode gated delta net into an existing command encoder (fused kernels).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_gated_delta_sequence(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        g: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
    ) {
        let v_dim = head_dim;
        let use_simd = seq_len >= GDN_CHUNK_THRESHOLD && head_dim == 128;

        if use_simd {
            // llama.cpp-style fused GDN: distribute v-dim across TGs, simd_sum for reductions.
            // Grid: (S_v/NSG, n_heads, 1) = (32, 32, 1) = 1024 TGs — 32x more than old kernel.
            // TG: (32, NSG, 1) = (32, 4, 1) = 128 threads.
            const NSG: usize = 4;
            let grid_x = (v_dim as usize).div_ceil(NSG);
            crate::set_pipeline_cached(encoder, self.simd_gated_delta_4.state());
            bind_buffers7(encoder, q, k, v, g, beta, state, output);
            bind_u32(encoder, 7, seq_len);
            bind_u32(encoder, 8, v_dim);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: grid_x,
                    height: n_heads as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: NSG,
                    depth: 1,
                },
            );
            return;
        }

        let use_chunked = seq_len >= GDN_CHUNK_THRESHOLD;
        let (pipeline, use_fallback, bv) = match (head_dim, use_chunked) {
            (128, true) => (&self.chunked_gated_delta_32_128_128, false, 128u32),
            (64, true) => (&self.chunked_gated_delta_32_64_64, false, 64u32),
            (128, false) => (&self.gated_delta_128_64, false, 64u32),
            (64, false) => (&self.gated_delta_64_64, false, 64u32),
            _ => (&self.gated_delta_fallback, true, 64u32),
        };
        let grid_x = (v_dim as usize).div_ceil(bv as usize);

        crate::set_pipeline_cached(encoder, pipeline.state());
        bind_buffers7(encoder, q, k, v, g, beta, state, output);
        bind_u32(encoder, 7, seq_len);
        if use_fallback {
            bind_u32(encoder, 8, head_dim);
            bind_u32(encoder, 9, v_dim);
            let smem = 2 * head_dim as usize * std::mem::size_of::<f32>();
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(smem, 0);
            }
        } else {
            bind_u32(encoder, 8, v_dim);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: grid_x,
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: bv as usize,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Dispatch gated delta net in its own command buffer (legacy wrapper).
    #[deprecated(note = "Use encode_gated_delta_sequence with a caller-managed command buffer.")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_sequence(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        g: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_gated_delta_sequence(
                encoder, q, k, v, g, beta, state, output, seq_len, n_heads, head_dim,
            );
            Ok(())
        })
    }

    /// Encode inclusive prefix sum along the fastest dimension.
    ///
    /// `src` and `dst` are `[ne2, ne1, ne0]` row-major. Cumsum is computed
    /// along `ne0`. Requires `ne0 <= 1024`.
    /// Grid: `(ne1, ne2, 1)`.  TG: `(nth, 1, 1)` where `nth = next_pow2(ne0)`.
    pub fn encode_cumsum(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        ne0: u32,
        ne1: u32,
        ne2: u32,
    ) {
        let nth = (ne0 as usize).next_power_of_two().min(1024);
        crate::set_pipeline_cached(encoder, self.cumsum_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, ne0);
        bind_u32(encoder, 3, ne1);
        // Shared memory: 32 floats for simdgroup sums.
        let smem = 32 * std::mem::size_of::<f32>();
        unsafe { encoder.setThreadgroupMemoryLength_atIndex(smem, 0) };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: ne1 as _,
                height: ne2 as _,
                depth: 1,
            },
            MTLSize {
                width: nth,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode lower-triangular solve: L × X = B, L unit-diagonal.
    ///
    /// `L` is `[n_slices, N, N]`, `B`/`X` are `[n_slices, N, K]`.
    /// `X` is written to `dst` (may alias `B` for in-place).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_solve_tri_lower(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        l_buf: &MetalBuffer,
        b_buf: &MetalBuffer,
        x_buf: &MetalBuffer,
        n: u32,
        k: u32,
        n_slices: u32,
    ) {
        let nsg: usize = if k >= 4 { 4 } else { 1 };
        let pipeline = if nsg == 4 {
            &self.solve_tri_lower_f32_4
        } else {
            &self.solve_tri_lower_f32_1
        };
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(l_buf.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b_buf.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(x_buf.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_u32(encoder, 4, k);
        let smem = nsg * n as usize * std::mem::size_of::<f32>();
        unsafe { encoder.setThreadgroupMemoryLength_atIndex(smem, 0) };
        let grid_x = (k as usize).div_ceil(nsg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: grid_x,
                height: n_slices as _,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: nsg,
                depth: 1,
            },
        );
    }

    /// Encode lower-triangular mask (with diagonal): dst[i,j] = (j<=i) ? src[i,j] : 0.
    pub fn encode_tri_lower_diag(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n: u32,
        n_slices: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.tri_lower_diag_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_u32(encoder, 3, n_slices);
        let total = (n * n) as usize;
        let groups = total.div_ceil(256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: n_slices as _,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode strict lower-triangular mask: dst[i,j] = (j<i) ? src[i,j] : 0.
    pub fn encode_tri_lower_strict(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n: u32,
        n_slices: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.tri_lower_strict_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_u32(encoder, 3, n_slices);
        let total = (n * n) as usize;
        let groups = total.div_ceil(256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: n_slices as _,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode identity matrix: dst[i,j] = (i==j) ? 1 : 0.
    pub fn encode_diag_identity(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        dst: &MetalBuffer,
        n: u32,
        n_slices: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.diag_identity_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 0);
        }
        bind_u32(encoder, 1, n);
        bind_u32(encoder, 2, n_slices);
        let total = (n * n) as usize;
        let groups = total.div_ceil(256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: n_slices as _,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched C[b] = A[b] × B[b] for b = 0..n_batch-1.
    /// A[b] is [M, K], B[b] is [K, N], C[b] is [M, N] (row-major f32).
    /// Strides are in float elements.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_batched_matmul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        n_batch: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.batched_matmul_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m * k); // stride_a
        bind_u32(encoder, 7, k * n); // stride_b
        bind_u32(encoder, 8, m * n); // stride_c
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(8),
                height: (m as usize).div_ceil(8),
                depth: n_batch as _,
            },
            MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        );
    }

    /// Encode batched C[b] = A[b]^T × B[b] for b = 0..n_batch-1.
    /// A[b] is [K, M] (transposed), B[b] is [K, N], C[b] is [M, N].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_batched_matmul_atrans(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        n_batch: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.batched_matmul_atrans_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, k * m); // stride_a (A is [K, M])
        bind_u32(encoder, 7, k * n); // stride_b
        bind_u32(encoder, 8, m * n); // stride_c
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(8),
                height: (m as usize).div_ceil(8),
                depth: n_batch as _,
            },
            MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        );
    }

    /// Encode batched C[b] = A[b] × B[b]^T for b = 0..n_batch-1.
    /// A[b] is [M, K], B[b] is [N, K] (both row-major), C[b] is [M, N].
    /// result[i,j] = dot(A[i,:], B[j,:]) — inner product of rows.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_batched_matmul_btrans(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        n_batch: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.batched_matmul_btrans_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m * k); // stride_a
        bind_u32(encoder, 7, n * k); // stride_b (B is [N, K])
        bind_u32(encoder, 8, m * n); // stride_c
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(8),
                height: (m as usize).div_ceil(8),
                depth: n_batch as _,
            },
            MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        );
    }

    /// Encode C[b] = A[b] × B[b] with custom strides and byte offsets.
    /// Strides are in f32 elements. Offsets are in bytes.
    #[allow(clippy::too_many_arguments)]
    fn encode_matmul_strided(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ComputePipeline,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        a_off: usize,
        b_off: usize,
        c_off: usize,
        m: u32,
        n: u32,
        k: u32,
        stride_a: u32,
        stride_b: u32,
        stride_c: u32,
        n_batch: u32,
    ) {
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), a_off, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
        }
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, stride_a);
        bind_u32(encoder, 7, stride_b);
        bind_u32(encoder, 8, stride_c);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(8),
                height: (m as usize).div_ceil(8),
                depth: n_batch as _,
            },
            MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        );
    }

    /// Check if the env-gated chunked GDN graph decomposition is enabled.
    /// Enable with `AX_METAL_GDN_CHUNKED_GRAPH=1`. Default OFF.
    pub fn chunked_graph_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("AX_METAL_GDN_CHUNKED_GRAPH")
                .ok()
                .and_then(|v| parse_bool_env_flag(&v))
                .unwrap_or(false)
        })
    }

    /// Encode the graph-based chunked GDN decomposition (GDA path, CS=64).
    ///
    /// All operations encoded into one command encoder — no sync points.
    /// Scratch buffers allocated from `device` and released on return.
    /// Input layout: `[H, T, D]` for Q/K/V, `[H, T]` for gate/beta,
    /// `[H, D, D]` for state, `[H, T, D]` for output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_chunked_gdn_graph(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        elementwise: &ElementwiseKernels,
        device: &MetalDevice,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        g: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
    ) -> Vec<MetalBuffer> {
        let cs: u32 = 64;
        let h = n_heads;
        let d = head_dim;
        let t = seq_len;
        let scale = 1.0 / (d as f32).sqrt();

        // Guard: seq_len must be a multiple of CS for correct layout.
        // Non-aligned lengths fall back to the fused kernel.
        if !t.is_multiple_of(cs) || t == 0 {
            self.encode_gated_delta_sequence(
                encoder, q, k, v, g, beta, state, output, seq_len, n_heads, head_dim,
            );
            return Vec::new();
        }

        let n_chunks = t / cs;
        let hc = h * n_chunks; // independent (head, chunk) slices

        let f4 = std::mem::size_of::<f32>();
        // Scratch buffers must outlive the command buffer — collected into
        // `scratches` and returned to the caller for lifetime management.
        let mut scratches: Vec<MetalBuffer> = Vec::with_capacity(32);
        let alloc = |n: usize| -> MetalBuffer {
            let mut buf =
                MetalBuffer::new(device.device(), (n * f4).max(f4)).expect("GDN graph alloc");
            unsafe { buf.as_mut_slice::<f32>().fill(0.0) };
            buf
        };

        let q_s = alloc((h * t * d) as usize);
        let kb = alloc((h * t * d) as usize);
        let vb = alloc((h * t * d) as usize);
        let g_cs = alloc((hc * cs) as usize);
        let decay = alloc((hc * cs * cs) as usize);
        let kk = alloc((hc * cs * cs) as usize);
        let kq_m = alloc((hc * cs * cs) as usize);
        let g_last = alloc(hc as usize);
        let g_last_exp = alloc(hc as usize);
        let g_diff_exp = alloc((hc * cs) as usize);

        let htd = h * t * d;

        // ── 1. q_scaled = q * scale ──
        elementwise.encode_gen_scale(encoder, q, &q_s, scale, htd);

        // ── 2. kb = k * beta, vb = v * beta (broadcast beta across D) ──
        let encode_bcast = |enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
                            src: &MetalBuffer,
                            sc: &MetalBuffer,
                            dst: &MetalBuffer| {
            crate::set_pipeline_cached(enc, self.broadcast_mul_f32.state());
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(sc.mtl_buffer()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 2);
            }
            bind_u32(enc, 3, d);
            bind_u32(enc, 4, htd);
            let dims = DispatchDims::d1(htd as usize, 256);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        };
        encode_bcast(encoder, k, beta, &kb);
        encode_bcast(encoder, v, beta, &vb);

        // ── 3. cumsum(gate) per chunk ──
        self.encode_cumsum(encoder, g, &g_cs, cs, hc, 1);

        // ── 4. Build decay mask: decay[s,i,j] = exp(g_cs[j]-g_cs[i]) for j<=i ──
        {
            let cs2 = (cs * cs) as usize;
            crate::set_pipeline_cached(encoder, self.build_decay_mask_f32.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(g_cs.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(decay.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, cs);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: cs2.div_ceil(256),
                    height: hc as _,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }

        // ── 5. Intra-chunk correlations (parallel across all H*n_chunks) ──
        // kk[i,j] = sum_d k[i,d]*kb[j,d] = (k × kb^T)[i,j]  →  [CS, CS]
        // kq[i,j] = sum_d k[i,d]*q_s[j,d] = (k × q_s^T)[i,j] → [CS, CS]
        self.encode_batched_matmul_btrans(encoder, k, &kb, &kk, cs, cs, d, hc);
        self.encode_batched_matmul_btrans(encoder, k, &q_s, &kq_m, cs, cs, d, hc);

        // ── 6. Apply decay mask: kk *= decay, kq *= decay ──
        let hcc = hc * cs * cs;
        elementwise.encode_gen_mul(encoder, &kk, &decay, &kk, hcc);
        elementwise.encode_gen_mul(encoder, &kq_m, &decay, &kq_m, hcc);

        // ── 7. Triangular masks ──
        // kq = tri_lower_diag(kq)
        self.encode_tri_lower_diag(encoder, &kq_m, &kq_m, cs, hc);
        // attn_a = tri_lower_strict(kk) (= A, strict lower without diagonal)
        let attn_a = alloc((hc * cs * cs) as usize);
        self.encode_tri_lower_strict(encoder, &kk, &attn_a, cs, hc);

        // ── 8. Triangular solve: correction = (I + A)^{-1} × (-A) + I ──
        // Build lhs = I + A using diag_identity for all slices at once.
        let lhs = alloc((hc * cs * cs) as usize);
        self.encode_diag_identity(encoder, &lhs, cs, hc); // lhs = I (per slice)
        elementwise.encode_gen_add(encoder, &lhs, &attn_a, &lhs, hcc); // lhs = I + A
        // neg_a = -A
        let neg_a = alloc((hc * cs * cs) as usize);
        elementwise.encode_gen_neg(encoder, &attn_a, &neg_a, hcc);
        // solve: (I + A) × X = -A → X
        let solved = alloc((hc * cs * cs) as usize);
        self.encode_solve_tri_lower(encoder, &lhs, &neg_a, &solved, cs, cs, hc);
        // attn_corr = X + I
        let ident_all = alloc((hc * cs * cs) as usize);
        self.encode_diag_identity(encoder, &ident_all, cs, hc);
        let attn_corr = alloc((hc * cs * cs) as usize);
        elementwise.encode_gen_add(encoder, &solved, &ident_all, &attn_corr, hcc);

        // ── 9. Pre-compute inter-chunk decay factors ──
        // g_last[s] = g_cs[s * CS + CS - 1]
        {
            let dims = DispatchDims::d1(hc as usize, 256);
            crate::set_pipeline_cached(encoder, self.extract_last_cumsum_f32.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(g_cs.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(g_last.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, cs);
            bind_u32(encoder, 3, hc);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        }
        elementwise.encode_gen_exp(encoder, &g_last, &g_last_exp, hc);
        // g_diff_exp[s, t] = exp(g_last[s] - g_cs[s*CS + t])
        {
            let total = hc * cs;
            let dims = DispatchDims::d1(total as usize, 256);
            crate::set_pipeline_cached(encoder, self.build_g_diff_exp_f32.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(g_cs.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(g_last.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(g_diff_exp.mtl_buffer()), 0, 2);
            }
            bind_u32(encoder, 3, cs);
            bind_u32(encoder, 4, total);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        }

        // ── 10. Pre-compute g_exp-scaled tensors for per-chunk loop ──
        let g_exp = alloc((hc * cs) as usize);
        elementwise.encode_gen_exp(encoder, &g_cs, &g_exp, hc * cs);
        // q_gexp = q_s * g_exp (broadcast g_exp across D)
        let q_gexp = alloc((h * t * d) as usize);
        encode_bcast(encoder, &q_s, &g_exp, &q_gexp);
        // kg = k * g_diff_exp (broadcast g_diff_exp across D)
        let kg = alloc((h * t * d) as usize);
        encode_bcast(encoder, k, &g_diff_exp, &kg);

        // ── 11. Corrected v: vb_corr = attn_corr^T × vb ──
        // llama.cpp: v = mul_mat(transpose(v_b), attn) = v_b × attn in col-major.
        // In row-major: vb_corr[t, d] = sum_j attn_corr[j, t] * vb[j, d].
        // This is attn_corr^T × vb: use batched_matmul_atrans.
        let vb_corr = alloc((hc * cs * d) as usize);
        self.encode_batched_matmul_atrans(encoder, &attn_corr, &vb, &vb_corr, cs, d, cs, hc);

        // ── 12. k_cd = attn_corr^T × kbg ──
        // kbg = kb * g_exp (broadcast). k_cd follows same transpose rule as vb_corr.
        let kbg = alloc((hc * cs * d) as usize);
        encode_bcast(encoder, &kb, &g_exp, &kbg);
        let k_cd = alloc((hc * cs * d) as usize);
        self.encode_batched_matmul_atrans(encoder, &attn_corr, &kbg, &k_cd, cs, d, cs, hc);

        // ── 13. Per-chunk sequential loop ──
        // Layout [H*n_chunks, ...]: head h chunk c is at batch index h*n_chunks + c.
        // For chunk c across all H heads: batch stride = n_chunks, offset = c.
        // Use encode_matmul_strided with stride = n_chunks * slice_size.
        let f4 = std::mem::size_of::<f32>();
        let stride_cd = n_chunks * cs * d; // stride between heads in [H*nc, CS, D]
        let stride_cc = n_chunks * cs * cs; // stride between heads in [H*nc, CS, CS]
        let stride_dd = d * d; // stride between heads in [H, D, D]
        let off_cd = |c: u32| (c * cs * d) as usize * f4; // byte offset for chunk c in [H*nc, CS, D]
        let off_cc = |c: u32| (c * cs * cs) as usize * f4; // byte offset for chunk c in [H*nc, CS, CS]

        let ch_v_prime = alloc((h * cs * d) as usize);
        let ch_v_new = alloc((h * cs * d) as usize);
        let ch_v_attn = alloc((h * cs * d) as usize);
        let ch_attn_inter = alloc((h * cs * d) as usize);
        let ch_kgv = alloc((h * d * d) as usize);
        let ch_out = alloc((h * cs * d) as usize);

        for c in 0..n_chunks {
            // 13a. v_prime = k_cd[c] × state: [CS, D] × [D, D] → [CS, D]
            self.encode_matmul_strided(
                encoder,
                &self.batched_matmul_f32,
                &k_cd,
                state,
                &ch_v_prime,
                off_cd(c),
                0,
                0,
                cs,
                d,
                d,
                stride_cd,
                stride_dd,
                cs * d,
                h,
            );

            // 13b. v_new[h] = vb_corr[h*nc+c] - v_prime[h]
            // vb_corr is [H*nc, CS, D], ch_v_prime is [H, CS, D].
            // For head h, chunk c: vb_corr offset = (h*nc + c) * CS*D.
            // Use strided sub: src_a stride = nc*CS*D, src_b stride = CS*D,
            //                  dst stride = CS*D, count = CS*D, n_batch = H.
            {
                let slice = cs * d;
                let a_off = (c * slice) as usize * f4;
                let a_stride = (n_chunks * slice) as usize * f4;
                let b_stride = slice as usize * f4;
                crate::set_pipeline_cached(encoder, self.strided_sub_f32.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(vb_corr.mtl_buffer()), a_off, 0);
                    encoder.setBuffer_offset_atIndex(Some(ch_v_prime.mtl_buffer()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(ch_v_new.mtl_buffer()), 0, 2);
                }
                bind_u32(encoder, 3, slice);
                bind_u32(encoder, 4, a_stride as u32 / 4); // stride in elements
                bind_u32(encoder, 5, b_stride as u32 / 4);
                bind_u32(encoder, 6, b_stride as u32 / 4); // dst stride = same as b
                bind_u32(encoder, 7, h);
                let dims = DispatchDims::d1((h * slice) as usize, 256);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    dims.threadgroups,
                    dims.threads_per_threadgroup,
                );
            }

            // 13c. v_attn[t, d] = sum_j kq[j, t] * v_new[j, d] = (kq^T × v_new)[t, d]
            // Use atrans variant for kq^T × v_new.
            self.encode_matmul_strided(
                encoder,
                &self.batched_matmul_atrans_f32,
                &kq_m,
                &ch_v_new,
                &ch_v_attn,
                off_cc(c),
                0,
                0,
                cs,
                d,
                cs,
                stride_cc,
                cs * d,
                cs * d,
                h,
            );

            // 13d. attn_inter[t, d] = sum_d2 state[d, d2] * q_gexp[t, d2]
            //     = sum_d2 q_gexp[t, d2] * state[d, d2] = (q_gexp × state^T)[t, d]
            // Use btrans: C = A × B^T where A=q_gexp[CS,D], B=state[D,D].
            self.encode_matmul_strided(
                encoder,
                &self.batched_matmul_btrans_f32,
                &q_gexp,
                state,
                &ch_attn_inter,
                off_cd(c),
                0,
                0,
                cs,
                d,
                d,
                stride_cd,
                stride_dd,
                cs * d,
                h,
            );

            // 13e. output[c] = attn_inter + v_attn → write directly to output[c]
            // output is [H, T, D]. Chunk c starts at (h*T + c*CS) * D per head.
            // ch_attn_inter/ch_v_attn are [H, CS, D] contiguous (stride = CS*D).
            // output stride between heads = T*D.
            // Use gen_add with custom offsets: can't easily do strided output.
            // Instead write to ch_out, then use strided matmul as identity copy.
            elementwise.encode_gen_add(encoder, &ch_attn_inter, &ch_v_attn, &ch_out, h * cs * d);

            // Copy ch_out → output[chunk c] using strided copy.
            // ch_out is [H, CS, D] contiguous (stride = CS*D).
            // output is [H, T, D] (stride = T*D).
            // For head h: src = ch_out + h*CS*D, dst = output + h*T*D + c*CS*D.
            {
                let slice = cs * d;
                let dst_off = (c * cs * d) as usize * f4;
                crate::set_pipeline_cached(encoder, self.strided_copy_f32.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(ch_out.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), dst_off, 1);
                }
                bind_u32(encoder, 2, slice); // elements per slice
                bind_u32(encoder, 3, slice); // src stride (CS*D, contiguous)
                bind_u32(encoder, 4, t * d); // dst stride (T*D)
                bind_u32(encoder, 5, h); // n_batch
                let dims = DispatchDims::d1((h * slice) as usize, 256);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    dims.threadgroups,
                    dims.threads_per_threadgroup,
                );
            }

            // 13f. kgv = kg[c]^T × v_new: [D, CS]^T × [CS, D] → ...
            // kg[c] is [CS, D] per head. kg^T is [D, CS]. kg^T × v_new = [D, CS] × [CS, D] = [D, D].
            // Using atrans: A=kg[c]=[CS, D], A^T=[D, CS]. B=v_new=[CS, D].
            // C = A^T × B = [D, CS] × [CS, D] = [D, D]. ✓
            self.encode_matmul_strided(
                encoder,
                &self.batched_matmul_atrans_f32,
                &kg,
                &ch_v_new,
                &ch_kgv,
                off_cd(c),
                0,
                0,
                d,
                d,
                cs,
                stride_cd,
                cs * d,
                stride_dd,
                h,
            );

            // 13g. state = state * g_last_exp[c] + kgv
            // g_last_exp is [H*n_chunks]. For chunk c of head h: g_last_exp[h*nc + c].
            // Gather these H values into ch_g_scale via GPU kernel.
            let ch_g_scale = alloc(h as usize);
            {
                let dims = DispatchDims::d1(h as usize, 256);
                crate::set_pipeline_cached(encoder, self.gather_stride_f32.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(g_last_exp.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(ch_g_scale.mtl_buffer()), 0, 1);
                }
                bind_u32(encoder, 2, n_chunks); // stride
                bind_u32(encoder, 3, c); // offset
                bind_u32(encoder, 4, h); // count
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    dims.threadgroups,
                    dims.threads_per_threadgroup,
                );
            }
            // state *= g_scale (broadcast scalar per head across D*D)
            crate::set_pipeline_cached(encoder, self.broadcast_mul_f32.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(state.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(ch_g_scale.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(state.mtl_buffer()), 0, 2);
            }
            bind_u32(encoder, 3, d * d);
            bind_u32(encoder, 4, h * d * d);
            let dims = DispatchDims::d1((h * d * d) as usize, 256);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );

            // state += kgv
            elementwise.encode_gen_add(encoder, state, &ch_kgv, state, h * d * d);

            // Keep ch_g_scale alive until CB completes.
            scratches.push(ch_g_scale);
        }

        // Output and state are now fully computed by the graph.
        // Output was written per-chunk in step 13e via strided scatter.
        // State was updated in-place in steps 13f-13g.

        // Return all scratch buffers so caller keeps them alive until
        // the command buffer completes.
        scratches.push(q_s);
        scratches.push(kb);
        scratches.push(vb);
        scratches.push(g_cs);
        scratches.push(decay);
        scratches.push(kk);
        scratches.push(kq_m);
        scratches.push(g_last);
        scratches.push(g_last_exp);
        scratches.push(g_diff_exp);
        scratches.push(g_exp);
        scratches.push(q_gexp);
        scratches.push(kg);
        scratches.push(vb_corr);
        scratches.push(kbg);
        scratches.push(k_cd);
        scratches.push(attn_a);
        scratches.push(lhs);
        scratches.push(neg_a);
        scratches.push(solved);
        scratches.push(ident_all);
        scratches.push(attn_corr);
        scratches.push(ch_v_prime);
        scratches.push(ch_v_new);
        scratches.push(ch_v_attn);
        scratches.push(ch_attn_inter);
        scratches.push(ch_kgv);
        scratches.push(ch_out);
        scratches
    }
}
