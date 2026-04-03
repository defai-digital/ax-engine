impl Qwen35Forward {
    fn layer_type(cfg: &ModelConfig, layer: usize) -> Qwen35LayerType {
        if cfg.qwen35_is_recurrent_layer(layer) {
            Qwen35LayerType::RecurrentGdn
        } else {
            Qwen35LayerType::FullAttention
        }
    }

    fn rope_position(cfg: &ModelConfig, position: usize) -> f32 {
        cfg.rope_scaling.scaled_position(position)
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_dequant_matmul_token_major(
        backend: &dyn crate::backend::Backend,
        a_quant: &[u8],
        dtype: crate::gguf::tensor::GgmlType,
        input_token_major: &[f32],
        output_token_major: &mut [f32],
        n_tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) {
        backend.dequant_matmul_token_major(
            a_quant,
            dtype,
            input_token_major,
            output_token_major,
            n_tokens,
            out_dim,
            in_dim,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_dequant_matmul_gpu_safe(
        backend: &dyn crate::backend::Backend,
        a_quant: &[u8],
        dtype: crate::gguf::tensor::GgmlType,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        k: usize,
    ) {
        debug_assert_eq!(input.len(), k);
        debug_assert!(output.len() >= m);
        let ops = [(a_quant, dtype, m)];
        let mut outputs = [output];
        backend.safe_batch_dequant_matvec(&ops, input, k, &mut outputs);
    }

    fn decode_project_ops_gpu_safe(
        backend: &dyn crate::backend::Backend,
        input_ops: &[QuantOp<'_>],
        input: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        backend.safe_batch_dequant_matvec(input_ops, input, k, outputs);
    }

    fn qwen35_recurrent_config(
        qwen_kv: &crate::kv::Qwen35Kv,
        dims: Qwen35RecurrentDims,
        rms_norm_eps: f32,
    ) -> gdn::Qwen35RecurrentConfig {
        gdn::Qwen35RecurrentConfig {
            conv_cache_len: qwen_kv.conv_cache_len(),
            conv_dim: qwen_kv.conv_dim(),
            group_count: dims.group_count,
            state_size: dims.state_size,
            time_step_rank: dims.time_step_rank,
            rms_norm_eps,
        }
    }

    fn validate_recurrent_layer_state(
        qwen_kv: &crate::kv::Qwen35Kv,
        recurrent_slot: usize,
        layer: usize,
        stage: &str,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            qwen_kv.is_recurrent_layer(layer),
            "qwen35 KV/state layer mapping mismatch at layer {layer}"
        );
        debug_assert_eq!(
            qwen_kv.recurrent_seqlen_offset(recurrent_slot),
            qwen_kv.seq_len(),
            "qwen35 recurrent slot {recurrent_slot} drifted from seq_len before {stage} layer {layer}"
        );
        Ok(())
    }

    fn maybe_fused_full_attention_input_plan<'a>(
        input_ops: [QuantOp<'a>; 3],
    ) -> Qwen35FullAttentionInputPlan<'a> {
        let [
            (wq_raw, wq_dtype, q_rows),
            (wk_raw, wk_dtype, k_rows),
            (wv_raw, wv_dtype, v_rows),
        ] = input_ops;
        if wq_dtype == wk_dtype && wq_dtype == wv_dtype {
            Qwen35FullAttentionInputPlan::Fused {
                raw: Self::fused_quant_rows_cached(wq_raw, wk_raw, wv_raw),
                dtype: wq_dtype,
                rows: q_rows + k_rows + v_rows,
            }
        } else {
            Qwen35FullAttentionInputPlan::Split([
                (wq_raw, wq_dtype, q_rows),
                (wk_raw, wk_dtype, k_rows),
                (wv_raw, wv_dtype, v_rows),
            ])
        }
    }

    fn full_attention_input_plan<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        q_dim: usize,
        kv_dim: usize,
    ) -> anyhow::Result<Qwen35FullAttentionInputPlan<'a>> {
        let split_ops = Self::full_attention_input_ops(weights, prefix, q_dim, kv_dim)?;
        if !env_flag_enabled("AX_QWEN35_FUSED_FULL_ATTN_INPUT") {
            return Ok(Qwen35FullAttentionInputPlan::Split(split_ops));
        }
        Ok(Self::maybe_fused_full_attention_input_plan(split_ops))
    }

    fn full_attention_input_ops<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        q_dim: usize,
        kv_dim: usize,
    ) -> anyhow::Result<[QuantOp<'a>; 3]> {
        let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
        let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
        let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
        Ok([
            (wq_raw, wq_dtype, q_dim * 2),
            (wk_raw, wk_dtype, kv_dim),
            (wv_raw, wv_dtype, kv_dim),
        ])
    }

    fn full_attention_output_op<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        dim: usize,
    ) -> anyhow::Result<QuantOp<'a>> {
        let (raw, dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
        Ok((raw, dtype, dim))
    }

    fn project_full_attention_inputs<F>(
        input_ops: [QuantOp<'_>; 3],
        outputs: [&mut [f32]; 3],
        mut project: F,
    ) where
        F: FnMut(&[u8], crate::gguf::tensor::GgmlType, usize, &mut [f32]),
    {
        for ((raw, dtype, rows), out) in input_ops.into_iter().zip(outputs.into_iter()) {
            project(raw, dtype, rows, out);
        }
    }

    fn split_full_attention_fused_output(
        fused_output: &[f32],
        q_gate: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) {
        debug_assert_eq!(k.len(), v.len());
        debug_assert_eq!(fused_output.len(), q_gate.len() + k.len() + v.len());
        let q_gate_end = q_gate.len();
        let k_end = q_gate_end + k.len();
        q_gate.copy_from_slice(&fused_output[..q_gate_end]);
        k.copy_from_slice(&fused_output[q_gate_end..k_end]);
        v.copy_from_slice(&fused_output[k_end..]);
    }

    fn split_full_attention_fused_output_batch(
        fused_output_batch: &[f32],
        q_gate_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
        n_tokens: usize,
    ) {
        let q_gate_dim = q_gate_batch.len() / n_tokens;
        let kv_dim = k_batch.len() / n_tokens;
        debug_assert_eq!(k_batch.len(), v_batch.len());
        debug_assert_eq!(
            fused_output_batch.len(),
            n_tokens * (q_gate_dim + 2 * kv_dim)
        );
        fused_output_batch
            .par_chunks(q_gate_dim + 2 * kv_dim)
            .zip(q_gate_batch.par_chunks_mut(q_gate_dim))
            .zip(k_batch.par_chunks_mut(kv_dim))
            .zip(v_batch.par_chunks_mut(kv_dim))
            .take(n_tokens)
            .for_each(|(((fused_output, q_gate), k), v)| {
                Self::split_full_attention_fused_output(fused_output, q_gate, k, v);
            });
    }

    fn project_full_attention_inputs_batch<F>(
        input_plan: Qwen35FullAttentionInputPlan<'_>,
        q_gate_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
        fused_output_batch: &mut [f32],
        n_tokens: usize,
        mut project: F,
    ) where
        F: FnMut(&[u8], crate::gguf::tensor::GgmlType, usize, &mut [f32]),
    {
        match input_plan {
            Qwen35FullAttentionInputPlan::Split(input_ops) => {
                Self::project_full_attention_inputs(
                    input_ops,
                    [q_gate_batch, k_batch, v_batch],
                    project,
                );
            }
            Qwen35FullAttentionInputPlan::Fused { raw, dtype, rows } => {
                debug_assert_eq!(fused_output_batch.len(), n_tokens * rows);
                project(raw.as_ref(), dtype, rows, fused_output_batch);
                Self::split_full_attention_fused_output_batch(
                    fused_output_batch,
                    q_gate_batch,
                    k_batch,
                    v_batch,
                    n_tokens,
                );
            }
        }
    }

    fn maybe_attention_qk_norm<'a>(
        weights: &'a WeightStore,
        prefix: &str,
    ) -> anyhow::Result<Option<Qwen35AttentionNormWeights<'a>>> {
        crate::model::shared::maybe_attention_qk_norm_weights(weights, prefix)
    }

    fn recurrent_input_ops<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        dims: Qwen35RecurrentDims,
    ) -> anyhow::Result<[QuantOp<'a>; 4]> {
        let (wqkv_raw, wqkv_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.attn_qkv.weight"))?;
        let (wgate_raw, wgate_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.attn_gate.weight"))?;
        let (wbeta_raw, wbeta_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ssm_beta.weight"))?;
        let (walpha_raw, walpha_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ssm_alpha.weight"))?;
        Ok([
            (wqkv_raw, wqkv_dtype, dims.conv_dim()),
            (wgate_raw, wgate_dtype, dims.inner_size),
            (wbeta_raw, wbeta_dtype, dims.time_step_rank),
            (walpha_raw, walpha_dtype, dims.time_step_rank),
        ])
    }

    fn recurrent_runtime_tensors<'a>(
        weights: &'a WeightStore,
        prefix: &str,
    ) -> anyhow::Result<Qwen35RecurrentRuntimeTensors<'a>> {
        Ok(Qwen35RecurrentRuntimeTensors {
            dt_bias: weights.f32_slice(&format!("{prefix}.ssm_dt.bias"))?,
            a: weights.f32_slice(&format!("{prefix}.ssm_a"))?,
            conv_kernel: weights.f32_slice(&format!("{prefix}.ssm_conv1d.weight"))?,
            ssm_norm: weights.f32_slice(&format!("{prefix}.ssm_norm.weight"))?,
        })
    }

    fn recurrent_output_op<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        dim: usize,
    ) -> anyhow::Result<QuantOp<'a>> {
        let (raw, dtype) = weights.raw_with_dtype(&format!("{prefix}.ssm_out.weight"))?;
        Ok((raw, dtype, dim))
    }

    fn project_recurrent_inputs<F>(
        input_ops: [QuantOp<'_>; 4],
        outputs: [&mut [f32]; 4],
        mut project: F,
    ) where
        F: FnMut(&[u8], crate::gguf::tensor::GgmlType, usize, &mut [f32]),
    {
        for ((raw, dtype, rows), out) in input_ops.into_iter().zip(outputs.into_iter()) {
            project(raw, dtype, rows, out);
        }
    }

    fn repeat_token_major_batch_in_place(
        batch: &mut [f32],
        n_tokens: usize,
        width: usize,
        slot_count: usize,
    ) {
        if slot_count <= 1 {
            return;
        }
        let single_len = n_tokens * width;
        let total_len = single_len * slot_count;
        assert!(
            batch.len() >= total_len,
            "slot batch buffer is too small for repeated token-major data"
        );
        for slot_idx in 1..slot_count {
            batch.copy_within(0..single_len, slot_idx * single_len);
        }
    }

    fn slot_batch_slice<'a>(
        batch: &'a [f32],
        slot_indices: &[usize],
        target_slot: usize,
        n_tokens: usize,
        width: usize,
    ) -> &'a [f32] {
        let slot_batch_idx = slot_indices
            .iter()
            .position(|&slot_idx| slot_idx == target_slot)
            .expect("target slot missing from recurrent slot batch");
        let single_len = n_tokens * width;
        let start = slot_batch_idx * single_len;
        &batch[start..start + single_len]
    }

    fn slot_batch_slice_mut<'a>(
        batch: &'a mut [f32],
        slot_indices: &[usize],
        target_slot: usize,
        n_tokens: usize,
        width: usize,
    ) -> &'a mut [f32] {
        let slot_batch_idx = slot_indices
            .iter()
            .position(|&slot_idx| slot_idx == target_slot)
            .expect("target slot missing from recurrent slot batch");
        let single_len = n_tokens * width;
        let start = slot_batch_idx * single_len;
        &mut batch[start..start + single_len]
    }

    #[allow(clippy::too_many_arguments)]
    fn run_recurrent_sequence<'a>(
        backend: &dyn crate::backend::Backend,
        weights: &'a WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        recurrent_slot_indices: &[usize],
        dims: Qwen35RecurrentDims,
        rms_norm_eps: f32,
        rec_qkv: &[f32],
        rec_beta: &mut [f32],
        rec_alpha: &mut [f32],
        rec_out: &mut [f32],
        n_tokens: usize,
    ) -> anyhow::Result<Qwen35RecurrentRuntimeTensors<'a>> {
        let runtime = Self::recurrent_runtime_tensors(weights, prefix)?;
        let qwen35_cfg = Self::qwen35_recurrent_config(qwen_kv, dims, rms_norm_eps);
        backend.qwen35_recurrent_sequence_for_kv(
            rec_qkv,
            rec_beta,
            rec_alpha,
            runtime.dt_bias,
            runtime.a,
            runtime.conv_kernel,
            qwen_kv,
            layer,
            recurrent_slot_indices,
            rec_out,
            n_tokens,
            qwen35_cfg,
        );
        Ok(runtime)
    }

    fn ffn_input_ops<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        inter_dim: usize,
    ) -> anyhow::Result<[QuantOp<'a>; 2]> {
        let (wg_raw, wg_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
        let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
        Ok([(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)])
    }

    fn ffn_down_op<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        dim: usize,
    ) -> anyhow::Result<QuantOp<'a>> {
        let (raw, dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
        Ok((raw, dtype, dim))
    }

    fn qwen35_is_moe(cfg: &ModelConfig) -> bool {
        matches!(cfg.architecture.as_str(), "qwen35moe") || cfg.n_expert.is_some_and(|n| n > 0)
    }

    fn qwen35_layer_uses_moe(weights: &WeightStore, prefix: &str) -> bool {
        weights.has(&format!("{prefix}.ffn_gate_inp.weight"))
    }

    fn qwen35_moe_batch_dtype_supported(dtype: crate::gguf::tensor::GgmlType) -> bool {
        matches!(
            dtype,
            crate::gguf::tensor::GgmlType::F32
                | crate::gguf::tensor::GgmlType::Q4K
                | crate::gguf::tensor::GgmlType::Q5K
                | crate::gguf::tensor::GgmlType::Q6K
                | crate::gguf::tensor::GgmlType::Q8_0
        )
    }

    fn qwen35_moe_routed_expert_dtype_supported(dtype: crate::gguf::tensor::GgmlType) -> bool {
        matches!(
            dtype,
            crate::gguf::tensor::GgmlType::Q4K
                | crate::gguf::tensor::GgmlType::Q5K
        )
    }

    fn qwen35_shared_expert_gate_width_supported(rows: usize, dim: usize) -> bool {
        rows == 1 || rows == dim
    }

    fn qwen35_moe_resident_layer_keys(
        metal_ops: &crate::backend::metal::MetalOps,
        cfg: &ModelConfig,
        weights: &WeightStore,
        prefix: &str,
        dim: usize,
    ) -> anyhow::Result<Option<Qwen35MoeResidentLayerKeys>> {
        if !Self::qwen35_layer_uses_moe(weights, prefix) {
            return Ok(None);
        }

        let router_name = format!("{prefix}.ffn_gate_inp.weight");
        let (router_raw, router_dtype) = weights.raw_with_dtype(&router_name)?;
        if !Self::qwen35_moe_batch_dtype_supported(router_dtype) {
            return Ok(None);
        }
        let router_key = metal_ops.ensure_moe_quant_cached(router_raw);

        let n_expert =
            cfg.n_expert.unwrap_or(Self::tensor_output_rows(weights, &router_name)? as u32) as usize;
        let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
        anyhow::ensure!(n_expert > 0, "qwen35moe requires n_expert > 0");
        anyhow::ensure!(n_expert_used > 0, "qwen35moe requires n_expert_used > 0");
        anyhow::ensure!(
            n_expert_used <= n_expert,
            "qwen35moe n_expert_used ({n_expert_used}) > n_expert ({n_expert})"
        );

        let gate_name = format!("{prefix}.ffn_gate_exps.weight");
        let up_name = format!("{prefix}.ffn_up_exps.weight");
        let down_name = format!("{prefix}.ffn_down_exps.weight");
        // GPU mul_mat_id kernel uses the GGUF tensor's ne0 as the matmul
        // output dimension (it transposes internally like ggml_mul_mat).
        // For gate_exps [ne0=2048, ne1=512, ne2=256], output_rows=2048.
        // This is correct for the GPU path — do NOT use cfg.expert_intermediate_dim here.
        let expert_inter_dim = Self::tensor_output_rows(weights, &gate_name)?;
        let (gate_raw, gate_dtype) = weights.raw_with_dtype(&gate_name)?;
        let (up_raw, up_dtype) = weights.raw_with_dtype(&up_name)?;
        let (down_raw, down_dtype) = weights.raw_with_dtype(&down_name)?;
        if !Self::qwen35_moe_routed_expert_dtype_supported(gate_dtype)
            || !Self::qwen35_moe_routed_expert_dtype_supported(up_dtype)
            || !Self::qwen35_moe_routed_expert_dtype_supported(down_dtype)
        {
            return Ok(None);
        }
        let gate_key = metal_ops.ensure_moe_quant_cached(gate_raw);
        let up_key = metal_ops.ensure_moe_quant_cached(up_raw);
        let down_key = metal_ops.ensure_moe_quant_cached(down_raw);

        let gate_stride =
            crate::model::qwen3_moe::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
        let up_stride = crate::model::qwen3_moe::expert_byte_stride(up_dtype, expert_inter_dim * dim);
        let down_stride =
            crate::model::qwen3_moe::expert_byte_stride(down_dtype, dim * expert_inter_dim);

        let shared_gate_name = format!("{prefix}.ffn_gate_shexp.weight");
        let shared_expert = if weights.has(&shared_gate_name) {
            let shared_up_name = format!("{prefix}.ffn_up_shexp.weight");
            let shared_down_name = format!("{prefix}.ffn_down_shexp.weight");
            let shared_gate_inp_name = format!("{prefix}.ffn_gate_inp_shexp.weight");
            let (shared_gate_raw, shared_gate_dtype) = weights.raw_with_dtype(&shared_gate_name)?;
            let (shared_up_raw, shared_up_dtype) = weights.raw_with_dtype(&shared_up_name)?;
            let (shared_down_raw, shared_down_dtype) =
                weights.raw_with_dtype(&shared_down_name)?;
            if !(shared_gate_dtype == shared_up_dtype
                && shared_gate_dtype == shared_down_dtype
                && Self::qwen35_moe_batch_dtype_supported(shared_gate_dtype))
            {
                return Ok(None);
            }
            let shared_inter_dim = Self::tensor_output_rows(weights, &shared_gate_name)?;
            let shared_gate_key = metal_ops.ensure_moe_quant_cached(shared_gate_raw);
            let shared_up_key = metal_ops.ensure_moe_quant_cached(shared_up_raw);
            let shared_down_key = metal_ops.ensure_moe_quant_cached(shared_down_raw);
            let (gate_inp_key, gate_inp_dtype, gate_inp_rows) =
                if weights.has(&shared_gate_inp_name) {
                    let (raw, dtype) = weights.raw_with_dtype(&shared_gate_inp_name)?;
                    let rows = Self::tensor_output_rows(weights, &shared_gate_inp_name)?;
                    if !Self::qwen35_shared_expert_gate_width_supported(rows, dim)
                        || !Self::qwen35_moe_batch_dtype_supported(dtype)
                    {
                        return Ok(None);
                    }
                    (Some(metal_ops.ensure_moe_quant_cached(raw)), Some(dtype), rows)
                } else {
                    (None, None, 0)
                };

            Some(Qwen35SharedExpertResidentKeys {
                gate: shared_gate_key,
                up: shared_up_key,
                down: shared_down_key,
                gate_inp: gate_inp_key,
                gate_inp_dtype,
                dtype: shared_gate_dtype,
                inter_dim: shared_inter_dim,
                gate_inp_rows,
            })
        } else {
            None
        };

        Ok(Some(Qwen35MoeResidentLayerKeys {
            router: router_key,
            router_dtype,
            gate: gate_key,
            gate_dtype,
            up: up_key,
            up_dtype,
            down: down_key,
            down_dtype,
            n_expert,
            n_expert_used,
            expert_inter_dim,
            gate_stride,
            up_stride,
            down_stride,
            shared_expert,
        }))
    }

    fn tensor_output_rows(weights: &WeightStore, name: &str) -> anyhow::Result<usize> {
        let info = weights.info(name)?;
        match info.shape.as_slice() {
            [_input_dim] => Ok(1),
            [_input_dim, output_dim, ..] => Ok(*output_dim as usize),
            [] => anyhow::bail!("{name} has empty shape"),
        }
    }

    fn expert_quant_slice<'a>(
        full: &'a [u8],
        stride: usize,
        eid: usize,
        name: &str,
    ) -> anyhow::Result<&'a [u8]> {
        let start = eid
            .checked_mul(stride)
            .ok_or_else(|| anyhow::anyhow!("expert slice overflow for {name}"))?;
        let end = start
            .checked_add(stride)
            .ok_or_else(|| anyhow::anyhow!("expert slice overflow for {name}"))?;
        anyhow::ensure!(
            end <= full.len(),
            "expert slice out of bounds for {name}: expert={eid}, end={end}, len={}",
            full.len()
        );
        Ok(&full[start..end])
    }

    fn finalize_recurrent_output(
        rec_out: &mut [f32],
        rec_z: &[f32],
        dims: Qwen35RecurrentDims,
        ssm_norm_w: &[f32],
        rms_norm_eps: f32,
    ) {
        for head in 0..dims.time_step_rank {
            let start = head * dims.state_size;
            let end = start + dims.state_size;
            rms_norm::rms_norm(&mut rec_out[start..end], ssm_norm_w, rms_norm_eps);
        }
        let mut z_gate = rec_z.to_vec();
        silu::silu(&mut z_gate);
        silu::elementwise_mul(rec_out, &z_gate);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_recurrent_pair_matvec_with_config(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        wbeta: &MetalBuffer,
        walpha: &MetalBuffer,
        input: &MetalBuffer,
        beta_out: &MetalBuffer,
        alpha_out: &MetalBuffer,
        m: u32,
        k: u32,
        beta_dtype: crate::gguf::tensor::GgmlType,
        alpha_dtype: crate::gguf::tensor::GgmlType,
    ) -> bool {
        if beta_dtype != alpha_dtype {
            return false;
        }

        // Keep the precomputed-f16 route authoritative when either weight was
        // densified ahead of time; the paired quant kernels are only for the
        // raw quantized path.
        if metal_ops.has_precomputed_weight(wbeta) || metal_ops.has_precomputed_weight(walpha) {
            return false;
        }

        match beta_dtype {
            crate::gguf::tensor::GgmlType::Q4K => {
                metal_ops.dequant.encode_fused_matvec_pair_q4_k(
                    encoder, wbeta, walpha, input, beta_out, alpha_out, m, k,
                );
                true
            }
            crate::gguf::tensor::GgmlType::Q5K => {
                metal_ops.dequant.encode_fused_matvec_pair_q5_k(
                    encoder, wbeta, walpha, input, beta_out, alpha_out, m, k,
                );
                true
            }
            crate::gguf::tensor::GgmlType::Q6K => {
                metal_ops.dequant.encode_fused_matvec_pair_q6_k(
                    encoder, wbeta, walpha, input, beta_out, alpha_out, m, k,
                );
                true
            }
            crate::gguf::tensor::GgmlType::Q8_0 => {
                metal_ops.dequant.encode_fused_matvec_pair_q8_0(
                    encoder, wbeta, walpha, input, beta_out, alpha_out, m, k,
                );
                true
            }
            _ => false,
        }
    }

    fn rms_norm_token_major(
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        n_tokens: usize,
        dim: usize,
        rms_norm_eps: f32,
    ) {
        input
            .par_chunks(dim)
            .zip(output.par_chunks_mut(dim))
            .take(n_tokens)
            .for_each(|(input_token, output_token)| {
                rms_norm::rms_norm_out(input_token, weight, output_token, rms_norm_eps);
            });
    }

    /// Extract Q from Qwen3.5 full-attention `wq` output.
    ///
    /// Upstream layout is interleaved per head: `[q_h0, g_h0, q_h1, g_h1, ...]`.
    /// This is not equivalent to a flat split `[all_q, all_gate]`.
    fn extract_q_from_q_gate(q_gate: &[f32], q: &mut [f32], n_heads: usize, head_dim: usize) {
        let q_dim = n_heads * head_dim;
        debug_assert_eq!(q.len(), q_dim);
        debug_assert_eq!(q_gate.len(), q_dim * 2);
        for head in 0..n_heads {
            let src_start = head * head_dim * 2;
            let dst_start = head * head_dim;
            q[dst_start..dst_start + head_dim]
                .copy_from_slice(&q_gate[src_start..src_start + head_dim]);
        }
    }

    #[cfg(test)]
    fn extract_q_from_q_gate_batch(
        q_gate_batch: &[f32],
        q_batch: &mut [f32],
        n_tokens: usize,
        q_dim: usize,
        n_heads: usize,
        head_dim: usize,
    ) {
        for token_idx in 0..n_tokens {
            let src_start = token_idx * q_dim * 2;
            let q_start = token_idx * q_dim;
            Self::extract_q_from_q_gate(
                &q_gate_batch[src_start..src_start + q_dim * 2],
                &mut q_batch[q_start..q_start + q_dim],
                n_heads,
                head_dim,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_qk_norm(
        q: &mut [f32],
        k: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        norm_weights: Qwen35AttentionNormWeights<'_>,
        rms_norm_eps: f32,
    ) {
        crate::model::shared::apply_attention_qk_norm(
            q,
            k,
            n_heads,
            n_kv_heads,
            head_dim,
            norm_weights,
            rms_norm_eps,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn apply_attention_qk_norm_batch(
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        n_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        norm_weights: Qwen35AttentionNormWeights<'_>,
        rms_norm_eps: f32,
    ) {
        for token_idx in 0..n_tokens {
            let q_start = token_idx * q_dim;
            let k_start = token_idx * kv_dim;
            Self::apply_attention_qk_norm(
                &mut q_batch[q_start..q_start + q_dim],
                &mut k_batch[k_start..k_start + kv_dim],
                n_heads,
                n_kv_heads,
                head_dim,
                norm_weights,
                rms_norm_eps,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_rope(
        cfg: &ModelConfig,
        q: &mut [f32],
        k: &mut [f32],
        position: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        let rope_position = Self::rope_position(cfg, position);
        // Qwen3.5 uses NeoX-style RoPE and rotates a prefix (n_rot=64 in GGUF).
        rope::apply_rope_multi_head_neox_partial_scaled(
            q,
            k,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim.min(64),
            rope_position,
            cfg.rope_freq_base,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn apply_rope_batch(
        cfg: &ModelConfig,
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        n_tokens: usize,
        start_position: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for token_idx in 0..n_tokens {
            let q_start = token_idx * q_dim;
            let k_start = token_idx * kv_dim;
            Self::apply_rope(
                cfg,
                &mut q_batch[q_start..q_start + q_dim],
                &mut k_batch[k_start..k_start + kv_dim],
                start_position + token_idx,
                n_heads,
                n_kv_heads,
                head_dim,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_full_attention_qk_batch(
        cfg: &ModelConfig,
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
        norm_weights: Option<Qwen35AttentionNormWeights<'_>>,
        rms_norm_eps: f32,
    ) {
        q_gate_batch
            .par_chunks(q_dim * 2)
            .zip(q_batch.par_chunks_mut(q_dim))
            .zip(k_batch.par_chunks_mut(kv_dim))
            .enumerate()
            .take(n_tokens)
            .for_each(|(token_idx, ((q_gate, q), k))| {
                Self::extract_q_from_q_gate(q_gate, q, n_heads, head_dim);
                if let Some(norm_weights) = norm_weights {
                    Self::apply_attention_qk_norm(
                        q,
                        k,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        norm_weights,
                        rms_norm_eps,
                    );
                }
                Self::apply_rope(
                    cfg,
                    q,
                    k,
                    start_position + token_idx,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                );
            });
    }

    /// Apply sigmoid(gate) to attention output, reading gate from the
    /// interleaved Q|gate layout `[q_h0, g_h0, q_h1, g_h1, ...]`.
    fn apply_attention_gate(q_gate: &[f32], attn_out: &mut [f32], n_heads: usize, head_dim: usize) {
        let q_dim = n_heads * head_dim;
        debug_assert_eq!(q_gate.len(), q_dim * 2);
        debug_assert_eq!(attn_out.len(), q_dim);
        for head in 0..n_heads {
            let src_gate_start = head * head_dim * 2 + head_dim;
            let dst_start = head * head_dim;
            for i in 0..head_dim {
                let gate = 1.0 / (1.0 + (-q_gate[src_gate_start + i]).exp());
                attn_out[dst_start + i] *= gate;
            }
        }
    }

    fn apply_attention_gate_batch(
        q_gate_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) {
        let q_dim = n_heads * head_dim;
        q_gate_batch
            .par_chunks(q_dim * 2)
            .zip(attn_out_batch.par_chunks_mut(q_dim))
            .take(n_tokens)
            .for_each(|(q_gate, attn_out)| {
                Self::apply_attention_gate(q_gate, attn_out, n_heads, head_dim);
            });
    }

    /// GPU-accelerated recurrent finalize: per-head RMS norm + SiLU gate.
    ///
    /// Replaces per-token CPU loops in `finalize_recurrent_output_batch` with
    /// GPU batch dispatches. Falls back to CPU when Metal is unavailable.
    #[allow(clippy::too_many_arguments)]
    fn try_finalize_recurrent_output_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        rec_out_batch: &mut [f32],
        rec_z_batch: &[f32],
        n_tokens: usize,
        dims: Qwen35RecurrentDims,
        ssm_norm_w: &[f32],
        rms_norm_eps: f32,
    ) -> anyhow::Result<bool> {
        if n_tokens <= 1 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };

        let total = n_tokens * dims.inner_size;
        let nw_key = metal_ops.ensure_f32_cached(ssm_norm_w);

        metal_ops.init_batch_scratches(cfg, n_tokens);
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        // Upload rec_out → gate_buf, rec_z → up_buf.
        // gate_buf is [N × inter_dim], up_buf is [N × inter_dim]; inner_size < inter_dim.
        unsafe {
            bs.gate_buf.as_mut_slice::<f32>()[..total].copy_from_slice(&rec_out_batch[..total]);
            bs.up_buf.as_mut_slice::<f32>()[..total].copy_from_slice(&rec_z_batch[..total]);
        }

        let weight_cache = metal_ops.lock_weight_cache();
        let nw_buf = weight_cache.get(&nw_key).unwrap();

        metal_ops.device.execute_sync(|encoder| {
            // 1. Per-head RMS norm on rec_out (in gate_buf).
            // inner_size = time_step_rank × state_size, treat as n_heads=time_step_rank, head_dim=state_size.
            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                encoder,
                &bs.gate_buf,
                nw_buf,
                n_tokens as u32,
                dims.time_step_rank as u32,
                dims.state_size as u32,
                rms_norm_eps,
            );

            // 2. SiLU(rec_z) × rec_out: up_buf = silu(up_buf) * gate_buf.
            // After this, up_buf holds the final result.
            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                encoder,
                &bs.up_buf,
                &bs.gate_buf,
                dims.inner_size as u32,
                n_tokens as u32,
            );
            Ok(())
        })?;
        drop(weight_cache);

        // Read result from up_buf back to rec_out_batch.
        unsafe {
            rec_out_batch[..total].copy_from_slice(&bs.up_buf.as_slice::<f32>()[..total]);
        }
        drop(bs_guard);
        Ok(true)
    }

    fn finalize_recurrent_output_batch(
        rec_out_batch: &mut [f32],
        rec_z_batch: &[f32],
        n_tokens: usize,
        dims: Qwen35RecurrentDims,
        ssm_norm_w: &[f32],
        rms_norm_eps: f32,
    ) {
        rec_out_batch
            .par_chunks_mut(dims.inner_size)
            .zip(rec_z_batch.par_chunks(dims.inner_size))
            .take(n_tokens)
            .for_each(|(rec_out, rec_z)| {
                Self::finalize_recurrent_output(rec_out, rec_z, dims, ssm_norm_w, rms_norm_eps);
            });
    }
}
