impl Qwen35Forward {
    #[allow(clippy::too_many_arguments)]
    fn write_single_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &mut [f32],
        dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        crate::model::shared::apply_output_norm_single(weights, hidden, rms_norm_eps, None)?;
        crate::model::shared::write_normalized_single_logits(
            backend, hidden, dim, vocab_size, weights, logits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn write_normalized_batch_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &[f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        weights: &WeightStore,
        logits_all: &mut [f32],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            hidden.len() >= n_tokens * dim,
            "normalized hidden buffer too small for {n_tokens} tokens"
        );
        anyhow::ensure!(
            logits_all.len() >= n_tokens * vocab_size,
            "all-logits buffer too small for {n_tokens} tokens"
        );
        for token_idx in 0..n_tokens {
            let hidden_start = token_idx * dim;
            let logits_start = token_idx * vocab_size;
            crate::model::shared::write_normalized_single_logits(
                backend,
                &hidden[hidden_start..hidden_start + dim],
                dim,
                vocab_size,
                weights,
                &mut logits_all[logits_start..logits_start + vocab_size],
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_write_normalized_batch_logits_gpu(
        backend: &dyn crate::backend::Backend,
        hidden: &[f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        weights: &WeightStore,
        logits_all: &mut [f32],
    ) -> anyhow::Result<bool> {
        if !Self::gpu_batch_logits_enabled() {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };
        let (lm_raw, lm_dtype) = crate::model::shared::lm_head_raw_with_dtype(weights)?;
        if !gpu_batch_logits_supported(lm_dtype) {
            return Ok(false);
        }

        let vocab = vocab_size as u32;
        let n_rows = n_tokens as u32;
        let hidden_dim = dim as u32;
        Self::prepare_qwen35_batch_projection_weight(
            metal_ops, lm_raw, lm_dtype, vocab, hidden_dim,
        )?;
        let lm_key = metal_ops.ensure_quant_cached(lm_raw);
        metal_ops.with_qwen35_batch_logits_scratch(hidden.len(), logits_all.len(), |scratch| {
            unsafe {
                scratch.hidden.as_mut_slice::<f32>()[..hidden.len()].copy_from_slice(hidden);
            }
            let weight_cache = metal_ops.lock_weight_cache();
            let lm_buf = weight_cache.get(&lm_key).ok_or_else(|| {
                anyhow::anyhow!("missing Metal LM-head buffer for cached key {lm_key}")
            })?;
            metal_ops.device.execute_sync(|encoder| {
                encode_batch_logits(
                    metal_ops,
                    encoder,
                    lm_buf,
                    &scratch.hidden,
                    &scratch.hidden_f16,
                    &scratch.logits,
                    vocab,
                    n_rows,
                    hidden_dim,
                    lm_dtype,
                    metal_ops.metal_batch_f16_io_enabled(),
                    metal_ops.metal_batch_simd_enabled(),
                );
                Ok(())
            })?;

            let logits_gpu = unsafe {
                std::slice::from_raw_parts(
                    scratch.logits.contents().as_ptr() as *const f32,
                    n_tokens * vocab_size,
                )
            };
            logits_all.copy_from_slice(logits_gpu);
            anyhow::Ok(())
        })?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn write_all_batch_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &[f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        let mut final_hidden = vec![0.0f32; hidden.len()];
        Self::write_all_batch_logits_with_scratch(
            backend,
            hidden,
            n_tokens,
            dim,
            vocab_size,
            rms_norm_eps,
            weights,
            logits_all,
            &mut final_hidden,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn write_all_batch_logits_with_scratch(
        backend: &dyn crate::backend::Backend,
        hidden: &[f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
        final_hidden: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        final_hidden.resize(hidden.len(), 0.0);
        Self::rms_norm_token_major(
            hidden,
            final_norm_w,
            final_hidden.as_mut_slice(),
            n_tokens,
            dim,
            rms_norm_eps,
        );
        Self::assert_finite_if_enabled("final_norm_batch", final_hidden.as_slice(), 0, 0)?;
        logits_all.resize(n_tokens * vocab_size, 0.0);
        if !Self::try_write_normalized_batch_logits_gpu(
            backend,
            final_hidden.as_slice(),
            n_tokens,
            dim,
            vocab_size,
            weights,
            logits_all.as_mut_slice(),
        )? {
            Self::write_normalized_batch_logits(
                backend,
                final_hidden.as_slice(),
                n_tokens,
                dim,
                vocab_size,
                weights,
                logits_all.as_mut_slice(),
            )?;
        }
        Self::assert_finite_if_enabled("logits_all_batch", logits_all.as_slice(), 0, 0)?;
        Ok(())
    }

}
