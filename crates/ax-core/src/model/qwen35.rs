//! Qwen3.5 hybrid forward pass.
//!
//! The architecture alternates recurrent GDN layers with periodic full
//! attention layers. This implementation follows the upstream structure from
//! `llama.cpp` and `mistral.rs`:
//! - every layer uses `attn_norm`
//! - recurrent layers use fused `attn_qkv` + `attn_gate` + `ssm_*`
//! - full-attention layers use doubled Q projection (`q + gate`)
//! - every layer uses `post_attention_norm` before the FFN
//!
//! AX now routes the recurrent projections and recurrent primitives through the
//! active backend, including Metal kernels for the causal-conv and gated-delta
//! steps. The hybrid `ModelKv::Qwen35` state remains CPU-owned today, so Metal
//! recurrent execution still copies state through backend-owned kernels rather
//! than keeping it GPU-resident end to end.

use crate::compute::attention::{self, AttentionParams};
use crate::compute::gdn;
use crate::compute::rms_norm;
use crate::compute::rope;
use crate::compute::silu;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{env_flag_enabled, per_head_rms_norm};
use crate::model::weights::WeightStore;

macro_rules! timed {
    ($ops:expr, $field:ident, $body:expr) => {{
        if let Some(ref mut ops) = $ops {
            let _t = OpTimer::start();
            let _r = $body;
            ops.$field += _t.elapsed();
            _r
        } else {
            $body
        }
    }};
}

macro_rules! timed_matmul_bucket {
    ($ops:expr, $field:ident, $body:expr) => {{
        if let Some(ref mut ops) = $ops {
            let _t = OpTimer::start();
            let _r = $body;
            let _elapsed = _t.elapsed();
            ops.matmul += _elapsed;
            ops.$field += _elapsed;
            _r
        } else {
            $body
        }
    }};
}

#[derive(Debug)]
pub struct Qwen35Forward;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35LayerType {
    FullAttention,
    RecurrentGdn,
}

#[derive(Debug, Clone, Copy)]
struct Qwen35RecurrentDims {
    conv_kernel: usize,
    inner_size: usize,
    state_size: usize,
    time_step_rank: usize,
    group_count: usize,
}

impl Qwen35RecurrentDims {
    fn key_dim(self) -> usize {
        self.group_count * self.state_size
    }

    fn value_dim(self) -> usize {
        self.inner_size
    }

    fn conv_dim(self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

type QuantOp<'a> = (&'a [u8], crate::gguf::tensor::GgmlType, usize);

#[derive(Clone, Copy)]
struct Qwen35AttentionNormWeights<'a> {
    q: &'a [f32],
    k: &'a [f32],
}

#[derive(Clone, Copy)]
struct Qwen35RecurrentRuntimeTensors<'a> {
    dt_bias: &'a [f32],
    a: &'a [f32],
    conv_kernel: &'a [f32],
    ssm_norm: &'a [f32],
}

impl Qwen35Forward {
    fn assert_finite_if_enabled(
        label: &str,
        values: &[f32],
        layer: usize,
        position: usize,
    ) -> anyhow::Result<()> {
        if !env_flag_enabled("AX_QWEN35_ASSERT_FINITE") {
            return Ok(());
        }
        anyhow::ensure!(
            values.iter().all(|value| value.is_finite()),
            "qwen35 non-finite values at {label} (layer={layer}, position={position})"
        );
        Ok(())
    }

    fn recurrent_dims(cfg: &ModelConfig) -> anyhow::Result<Qwen35RecurrentDims> {
        let dims = Qwen35RecurrentDims {
            conv_kernel: cfg
                .qwen35_ssm_conv_kernel
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.conv_kernel"))?
                as usize,
            inner_size: cfg
                .qwen35_ssm_inner_size
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.inner_size"))?
                as usize,
            state_size: cfg
                .qwen35_ssm_state_size
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.state_size"))?
                as usize,
            time_step_rank: cfg
                .qwen35_ssm_time_step_rank
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.time_step_rank"))?
                as usize,
            group_count: cfg
                .qwen35_ssm_group_count
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.group_count"))?
                as usize,
        };
        anyhow::ensure!(dims.conv_kernel > 0, "qwen35 conv_kernel must be > 0");
        anyhow::ensure!(dims.state_size > 0, "qwen35 state_size must be > 0");
        anyhow::ensure!(dims.time_step_rank > 0, "qwen35 time_step_rank must be > 0");
        anyhow::ensure!(dims.group_count > 0, "qwen35 group_count must be > 0");
        anyhow::ensure!(dims.inner_size > 0, "qwen35 inner_size must be > 0");
        anyhow::ensure!(
            dims.inner_size == dims.state_size * dims.time_step_rank,
            "qwen35 inner_size ({}) must equal state_size ({}) * time_step_rank ({})",
            dims.inner_size,
            dims.state_size,
            dims.time_step_rank
        );
        anyhow::ensure!(
            dims.time_step_rank.is_multiple_of(dims.group_count),
            "qwen35 time_step_rank ({}) must be a multiple of group_count ({})",
            dims.time_step_rank,
            dims.group_count
        );
        Ok(dims)
    }

    fn layer_type(cfg: &ModelConfig, layer: usize) -> Qwen35LayerType {
        if cfg.qwen35_is_recurrent_layer(layer) {
            Qwen35LayerType::RecurrentGdn
        } else {
            Qwen35LayerType::FullAttention
        }
    }

    fn rope_position(cfg: &ModelConfig, position: usize) -> f32 {
        match cfg.rope_scaling {
            crate::model::config::RopeScaling::Linear(factor) => position as f32 / factor,
            crate::model::config::RopeScaling::None => position as f32,
        }
    }

    fn transpose(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert_eq!(input.len(), rows * cols);
        let mut out = vec![0.0f32; input.len()];
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = input[row * cols + col];
            }
        }
        out
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
        let input_t = Self::transpose(input_token_major, n_tokens, in_dim);
        let mut output_mn = vec![0.0f32; out_dim * n_tokens];
        backend.dequant_matmul(
            a_quant,
            dtype,
            &input_t,
            &mut output_mn,
            out_dim,
            n_tokens,
            in_dim,
        );
        let output_nm = Self::transpose(&output_mn, out_dim, n_tokens);
        output_token_major.copy_from_slice(&output_nm);
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

    fn maybe_attention_qk_norm<'a>(
        weights: &'a WeightStore,
        prefix: &str,
    ) -> anyhow::Result<Option<Qwen35AttentionNormWeights<'a>>> {
        if !weights.has(&format!("{prefix}.attn_q_norm.weight")) {
            return Ok(None);
        }
        Ok(Some(Qwen35AttentionNormWeights {
            q: weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?,
            k: weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?,
        }))
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

    fn project_ffn_inputs<F>(input_ops: [QuantOp<'_>; 2], outputs: [&mut [f32]; 2], mut project: F)
    where
        F: FnMut(&[u8], crate::gguf::tensor::GgmlType, usize, &mut [f32]),
    {
        for ((raw, dtype, rows), out) in input_ops.into_iter().zip(outputs.into_iter()) {
            project(raw, dtype, rows, out);
        }
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

    fn rms_norm_token_major(
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        n_tokens: usize,
        dim: usize,
        rms_norm_eps: f32,
    ) {
        for token_idx in 0..n_tokens {
            let start = token_idx * dim;
            let end = start + dim;
            rms_norm::rms_norm_out(
                &input[start..end],
                weight,
                &mut output[start..end],
                rms_norm_eps,
            );
        }
    }

    fn extract_q_from_q_gate(q_gate: &[f32], q: &mut [f32]) {
        debug_assert_eq!(q_gate.len(), q.len() * 2);
        q.copy_from_slice(&q_gate[..q.len()]);
    }

    fn extract_q_from_q_gate_batch(
        q_gate_batch: &[f32],
        q_batch: &mut [f32],
        n_tokens: usize,
        q_dim: usize,
    ) {
        for token_idx in 0..n_tokens {
            let src_start = token_idx * q_dim * 2;
            let q_start = token_idx * q_dim;
            Self::extract_q_from_q_gate(
                &q_gate_batch[src_start..src_start + q_dim * 2],
                &mut q_batch[q_start..q_start + q_dim],
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
        per_head_rms_norm(q, n_heads, head_dim, norm_weights.q, rms_norm_eps);
        per_head_rms_norm(k, n_kv_heads, head_dim, norm_weights.k, rms_norm_eps);
    }

    #[allow(clippy::too_many_arguments)]
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
        rope::apply_rope_multi_head_scaled(
            q,
            k,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_position,
            cfg.rope_freq_base,
        );
    }

    #[allow(clippy::too_many_arguments)]
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

    fn apply_attention_gate(gate: &mut [f32], attn_out: &mut [f32]) {
        debug_assert_eq!(gate.len(), attn_out.len());
        gdn::sigmoid_in_place(gate);
        silu::elementwise_mul(attn_out, gate);
    }

    fn apply_attention_gate_batch(
        q_gate_batch: &mut [f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        q_dim: usize,
    ) {
        for token_idx in 0..n_tokens {
            let gate_start = token_idx * q_dim * 2 + q_dim;
            let attn_start = token_idx * q_dim;
            Self::apply_attention_gate(
                &mut q_gate_batch[gate_start..gate_start + q_dim],
                &mut attn_out_batch[attn_start..attn_start + q_dim],
            );
        }
    }

    fn finalize_recurrent_output_batch(
        rec_out_batch: &mut [f32],
        rec_z_batch: &[f32],
        n_tokens: usize,
        dims: Qwen35RecurrentDims,
        ssm_norm_w: &[f32],
        rms_norm_eps: f32,
    ) {
        for token_idx in 0..n_tokens {
            let rec_out_start = token_idx * dims.inner_size;
            let rec_out_end = rec_out_start + dims.inner_size;
            Self::finalize_recurrent_output(
                &mut rec_out_batch[rec_out_start..rec_out_end],
                &rec_z_batch[rec_out_start..rec_out_end],
                dims,
                ssm_norm_w,
                rms_norm_eps,
            );
        }
    }

    fn lm_head_weight_name(weights: &WeightStore) -> &'static str {
        if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        }
    }

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
        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        rms_norm::rms_norm(hidden, final_norm_w, rms_norm_eps);
        Self::write_normalized_single_logits(backend, hidden, dim, vocab_size, weights, logits)
    }

    fn write_normalized_single_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &[f32],
        dim: usize,
        vocab_size: usize,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let lm_weight_name = Self::lm_head_weight_name(weights);
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        anyhow::ensure!(logits.len() >= vocab_size, "logits buffer too small");
        backend.dequant_matmul(lm_raw, lm_dtype, hidden, logits, vocab_size, 1, dim);
        Ok(())
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
            Self::write_normalized_single_logits(
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
    fn full_attention_prefill_batch(
        backend: &dyn crate::backend::Backend,
        qwen_kv: &crate::kv::Qwen35Kv,
        layer: usize,
        q_batch: &[f32],
        k_batch: &[f32],
        v_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        params: &AttentionParams,
    ) {
        let prefix_len = qwen_kv.seq_len();
        if prefix_len == 0 {
            backend.attention_prefill(
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                params.n_heads,
                params.n_kv_heads,
                params.head_dim,
            );
        } else {
            attention::multi_head_attention_prefill_with_prefix(
                qwen_kv.attention_k_slice_including_current(layer, prefix_len),
                qwen_kv.attention_v_slice_including_current(layer, prefix_len),
                prefix_len,
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                params,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_last_batch_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &mut [f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let last_hidden = &mut hidden[(n_tokens - 1) * dim..n_tokens * dim];
        Self::write_single_logits(
            backend,
            last_hidden,
            dim,
            vocab_size,
            rms_norm_eps,
            weights,
            logits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_post_attention_ffn_batch(
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<()> {
        let post_attn_norm_w =
            weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
        Self::rms_norm_token_major(
            hidden,
            post_attn_norm_w,
            norm_buf,
            n_tokens,
            dim,
            rms_norm_eps,
        );

        let input_ops = Self::ffn_input_ops(weights, prefix, inter_dim)?;
        Self::project_ffn_inputs(input_ops, [gate_buf, up_buf], |raw, dtype, rows, out| {
            Self::batched_dequant_matmul_token_major(
                backend, raw, dtype, norm_buf, out, n_tokens, rows, dim,
            );
        });
        silu::silu_elementwise_mul(gate_buf, up_buf);

        let (wd_raw, wd_dtype, _) = Self::ffn_down_op(weights, prefix, dim)?;
        Self::batched_dequant_matmul_token_major(
            backend, wd_raw, wd_dtype, gate_buf, down_buf, n_tokens, dim, inter_dim,
        );
        silu::elementwise_add(hidden, down_buf);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_post_attention_ffn_single(
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let post_attn_norm_w = timed!(
            ops,
            dequant,
            weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?
        );
        timed!(
            ops,
            norm,
            rms_norm::rms_norm_out(hidden, post_attn_norm_w, norm_buf, rms_norm_eps)
        );

        let input_ops = Self::ffn_input_ops(weights, prefix, inter_dim)?;
        timed_matmul_bucket!(ops, matmul_input_proj, {
            let mut outputs = [&mut *gate_buf, &mut *up_buf];
            Self::decode_project_ops_gpu_safe(backend, &input_ops, norm_buf, dim, &mut outputs);
        });
        silu::silu_elementwise_mul(gate_buf, up_buf);

        let (wd_raw, wd_dtype, _) = timed!(ops, dequant, Self::ffn_down_op(weights, prefix, dim)?);
        timed_matmul_bucket!(
            ops,
            matmul_output_proj,
            Self::decode_dequant_matmul_gpu_safe(
                backend, wd_raw, wd_dtype, gate_buf, down_buf, dim, inter_dim
            )
        );
        silu::elementwise_add(hidden, down_buf);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_norm_batch(
        weights: &WeightStore,
        prefix: &str,
        hidden: &[f32],
        norm_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<()> {
        let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
        Self::rms_norm_token_major(hidden, attn_norm_w, norm_buf, n_tokens, dim, rms_norm_eps);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_norm_single(
        weights: &WeightStore,
        prefix: &str,
        hidden: &[f32],
        norm_buf: &mut [f32],
        rms_norm_eps: f32,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let attn_norm_w = timed!(
            ops,
            dequant,
            weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?
        );
        timed!(
            ops,
            norm,
            rms_norm::rms_norm_out(hidden, attn_norm_w, norm_buf, rms_norm_eps)
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_full_attention_batch_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        batch_position: usize,
        norm_buf: &[f32],
        q_gate_batch: &mut [f32],
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
        attn_out_batch: &mut [f32],
        proj_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        full_attn_params: &AttentionParams,
    ) -> anyhow::Result<()> {
        let input_ops = Self::full_attention_input_ops(weights, prefix, q_dim, kv_dim)?;

        Self::project_full_attention_inputs(
            input_ops,
            [q_gate_batch, k_batch, v_batch],
            |raw, dtype, rows, out| {
                Self::batched_dequant_matmul_token_major(
                    backend, raw, dtype, norm_buf, out, n_tokens, rows, dim,
                );
            },
        );

        Self::extract_q_from_q_gate_batch(q_gate_batch, q_batch, n_tokens, q_dim);

        if let Some(norm_weights) = Self::maybe_attention_qk_norm(weights, prefix)? {
            Self::apply_attention_qk_norm_batch(
                q_batch,
                k_batch,
                n_tokens,
                q_dim,
                kv_dim,
                n_heads,
                n_kv_heads,
                head_dim,
                norm_weights,
                cfg.rms_norm_eps,
            );
        }

        Self::apply_rope_batch(
            cfg,
            q_batch,
            k_batch,
            n_tokens,
            qwen_kv.seq_len(),
            q_dim,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        Self::full_attention_prefill_batch(
            backend,
            qwen_kv,
            layer,
            q_batch,
            k_batch,
            v_batch,
            attn_out_batch,
            n_tokens,
            full_attn_params,
        );
        Self::apply_attention_gate_batch(q_gate_batch, attn_out_batch, n_tokens, q_dim);

        qwen_kv.attention_append_batch(layer, k_batch, v_batch, n_tokens);

        let (wo_raw, wo_dtype, _) = Self::full_attention_output_op(weights, prefix, dim)?;
        Self::batched_dequant_matmul_token_major(
            backend,
            wo_raw,
            wo_dtype,
            attn_out_batch,
            proj_buf,
            n_tokens,
            dim,
            q_dim,
        );
        Self::assert_finite_if_enabled(
            "full_attention_proj_batch",
            proj_buf,
            layer,
            batch_position,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_full_attention_single_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        position: usize,
        norm_buf: &[f32],
        q_gate_buf: &mut [f32],
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        proj_buf: &mut [f32],
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        full_attn_params: &AttentionParams,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let input_ops = Self::full_attention_input_ops(weights, prefix, q_dim, kv_dim)?;

        timed_matmul_bucket!(ops, matmul_input_proj, {
            let mut outputs = [&mut *q_gate_buf, &mut *k_buf, &mut *v_buf];
            Self::decode_project_ops_gpu_safe(backend, &input_ops, norm_buf, dim, &mut outputs);
        });
        Self::extract_q_from_q_gate(q_gate_buf, q_buf);
        let gate_attn = &mut q_gate_buf[q_dim..];

        if let Some(norm_weights) = timed!(
            ops,
            dequant,
            Self::maybe_attention_qk_norm(weights, prefix)?
        ) {
            timed!(ops, norm, {
                Self::apply_attention_qk_norm(
                    q_buf,
                    k_buf,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    norm_weights,
                    cfg.rms_norm_eps,
                );
            });
        }

        timed!(
            ops,
            rope,
            Self::apply_rope(cfg, q_buf, k_buf, position, n_heads, n_kv_heads, head_dim,)
        );

        qwen_kv.attention_append(layer, k_buf, v_buf);
        let seq_len = qwen_kv.seq_len() + 1;
        timed!(
            ops,
            attention,
            attention::multi_head_attention(
                q_buf,
                qwen_kv.attention_k_slice_including_current(layer, seq_len),
                qwen_kv.attention_v_slice_including_current(layer, seq_len),
                attn_out,
                full_attn_params,
                seq_len,
            )
        );

        Self::apply_attention_gate(gate_attn, attn_out);

        let (wo_raw, wo_dtype, _) = timed!(
            ops,
            dequant,
            Self::full_attention_output_op(weights, prefix, dim)?
        );
        timed_matmul_bucket!(
            ops,
            matmul_output_proj,
            Self::decode_dequant_matmul_gpu_safe(
                backend, wo_raw, wo_dtype, attn_out, proj_buf, dim, q_dim
            )
        );
        Self::assert_finite_if_enabled("full_attention_proj", proj_buf, layer, position)?;
        Ok(())
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
    ) -> anyhow::Result<()> {
        Self::validate_recurrent_layer_state(qwen_kv, recurrent_slot, layer, "prefill")?;
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
        Self::assert_finite_if_enabled(
            "recurrent_qkv_input_batch",
            rec_qkv_batch,
            layer,
            batch_position,
        )?;

        rec_out_batch.fill(0.0);
        let runtime = Self::run_recurrent_sequence(
            backend,
            weights,
            prefix,
            qwen_kv,
            layer,
            recurrent_slot_indices,
            dims,
            cfg.rms_norm_eps,
            rec_qkv_batch,
            rec_beta_batch,
            rec_alpha_batch,
            rec_out_batch,
            n_tokens,
        )?;
        Self::assert_finite_if_enabled(
            "recurrent_kernel_output_batch",
            rec_out_batch,
            layer,
            batch_position,
        )?;

        Self::finalize_recurrent_output_batch(
            rec_out_batch,
            rec_z_batch,
            n_tokens,
            dims,
            runtime.ssm_norm,
            cfg.rms_norm_eps,
        );
        Self::assert_finite_if_enabled(
            "recurrent_output_batch",
            rec_out_batch,
            layer,
            batch_position,
        )?;

        let (ssm_out_raw, ssm_out_dtype, _) = Self::recurrent_output_op(weights, prefix, dim)?;
        Self::batched_dequant_matmul_token_major(
            backend,
            ssm_out_raw,
            ssm_out_dtype,
            rec_out_batch,
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

    #[allow(clippy::too_many_arguments)]
    fn apply_layer_tail_batch(
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
        silu::elementwise_add(hidden, proj_buf);
        Self::assert_finite_if_enabled("layer_hidden_batch", hidden, layer, batch_position)?;
        Self::apply_post_attention_ffn_batch(
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

    #[allow(clippy::too_many_arguments)]
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
        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        let mut final_hidden = vec![0.0f32; hidden.len()];
        Self::rms_norm_token_major(
            hidden,
            final_norm_w,
            &mut final_hidden,
            n_tokens,
            dim,
            rms_norm_eps,
        );
        Self::assert_finite_if_enabled("final_norm_batch", &final_hidden, 0, 0)?;
        logits_all.resize(n_tokens * vocab_size, 0.0);
        // Qwen3.5 speculative verify currently stays finite through the
        // batched hybrid forward path and only destabilizes in the final
        // batched LM-head write. Reuse the proven per-token LM-head route for
        // all-logits emission until the batched logits projection is fixed.
        Self::write_normalized_batch_logits(
            backend,
            &final_hidden,
            n_tokens,
            dim,
            vocab_size,
            weights,
            logits_all.as_mut_slice(),
        )?;
        Self::assert_finite_if_enabled("logits_all_batch", logits_all.as_slice(), 0, 0)?;
        Ok(())
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

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_impl(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: Option<&mut [f32]>,
        logits_all: Option<&mut Vec<f32>>,
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
        if force_serial || !ctx.backend.use_gpu_decode() || token_ids.len() <= 1 {
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

        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            anyhow::bail!("Qwen35Forward requires ModelKv::Qwen35");
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

        let mut hidden = vec![0.0f32; n_tokens * dim];
        for (i, &tid) in token_ids.iter().enumerate() {
            weights.dequantize_row(
                "token_embd.weight",
                tid as usize,
                &mut hidden[i * dim..(i + 1) * dim],
            )?;
        }

        let mut norm_buf = vec![0.0f32; n_tokens * dim];
        let mut proj_buf = vec![0.0f32; n_tokens * dim];
        let mut gate_buf = vec![0.0f32; n_tokens * inter_dim];
        let mut up_buf = vec![0.0f32; n_tokens * inter_dim];
        let mut down_buf = vec![0.0f32; n_tokens * dim];
        let mut q_gate_batch = vec![0.0f32; n_tokens * q_dim * 2];
        let mut q_batch = vec![0.0f32; n_tokens * q_dim];
        let mut k_batch = vec![0.0f32; n_tokens * kv_dim];
        let mut v_batch = vec![0.0f32; n_tokens * kv_dim];
        let mut attn_out_batch = vec![0.0f32; n_tokens * q_dim];
        let mut rec_qkv_batch = vec![0.0f32; n_tokens * conv_dim];
        let mut rec_z_batch = vec![0.0f32; n_tokens * dims.inner_size];
        let mut rec_beta_batch = vec![0.0f32; n_tokens * dims.time_step_rank];
        let mut rec_alpha_batch = vec![0.0f32; n_tokens * dims.time_step_rank];
        let mut rec_out_batch = vec![0.0f32; n_tokens * dims.inner_size];
        let recurrent_slot_indices = [recurrent_slot];
        let batch_position = qwen_kv.seq_len();

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            Self::apply_attention_norm_batch(
                weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                n_tokens,
                dim,
                cfg.rms_norm_eps,
            )?;
            Self::assert_finite_if_enabled(
                "attn_norm_output_batch",
                &norm_buf,
                layer,
                batch_position,
            )?;

            match Self::layer_type(cfg, layer) {
                Qwen35LayerType::FullAttention => Self::run_full_attention_batch_layer(
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
                )?,
                Qwen35LayerType::RecurrentGdn => Self::run_recurrent_batch_layer(
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
                    &mut rec_qkv_batch,
                    &mut rec_z_batch,
                    &mut rec_beta_batch,
                    &mut rec_alpha_batch,
                    &mut rec_out_batch,
                    &mut proj_buf,
                    n_tokens,
                    dim,
                )?,
            }

            Self::apply_layer_tail_batch(
                backend,
                weights,
                &prefix,
                &mut hidden,
                &proj_buf,
                &mut norm_buf,
                &mut gate_buf,
                &mut up_buf,
                &mut down_buf,
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
                &mut hidden,
                n_tokens,
                dim,
                vocab_size,
                cfg.rms_norm_eps,
                weights,
                logits,
            ),
            (None, Some(logits_all)) => Self::write_all_batch_logits(
                backend,
                &hidden,
                n_tokens,
                dim,
                vocab_size,
                cfg.rms_norm_eps,
                weights,
                logits_all,
            ),
            _ => unreachable!("validated above"),
        }
    }
}

impl ForwardPass for Qwen35Forward {
    #[allow(clippy::too_many_arguments)]
    fn forward_single(
        &self,
        ctx: &ForwardContext,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let qwen_kv = kv
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen35Forward requires ModelKv::Qwen35"))?;
        let recurrent_slot = qwen_kv.active_slot();

        let dims = Self::recurrent_dims(cfg)?;
        let backend = ctx.backend;
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;

        let mut hidden = vec![0.0f32; dim];
        timed!(
            ops,
            dequant,
            weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?
        );

        let mut norm_buf = vec![0.0f32; dim];
        let mut q_gate_buf = vec![0.0f32; q_dim * 2];
        let mut q_buf = vec![0.0f32; q_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; dim];
        let mut rec_qkv = vec![0.0f32; dims.conv_dim()];
        let mut rec_z = vec![0.0f32; dims.inner_size];
        let mut rec_beta = vec![0.0f32; dims.time_step_rank];
        let mut rec_alpha = vec![0.0f32; dims.time_step_rank];
        let mut rec_out = vec![0.0f32; dims.inner_size];
        let recurrent_slot_indices = [recurrent_slot];
        let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            Self::apply_attention_norm_single(
                weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;
            Self::assert_finite_if_enabled("attn_norm_output", &norm_buf, layer, position)?;

            match Self::layer_type(cfg, layer) {
                Qwen35LayerType::FullAttention => Self::run_full_attention_single_layer(
                    cfg,
                    backend,
                    weights,
                    &prefix,
                    qwen_kv,
                    layer,
                    position,
                    &norm_buf,
                    &mut q_gate_buf,
                    &mut q_buf,
                    &mut k_buf,
                    &mut v_buf,
                    &mut attn_out,
                    &mut proj_buf,
                    dim,
                    q_dim,
                    kv_dim,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    &full_attn_params,
                    ops.as_deref_mut(),
                )?,
                Qwen35LayerType::RecurrentGdn => Self::run_recurrent_single_layer(
                    cfg,
                    backend,
                    weights,
                    &prefix,
                    qwen_kv,
                    recurrent_slot,
                    layer,
                    position,
                    dims,
                    &recurrent_slot_indices,
                    &norm_buf,
                    &mut rec_qkv,
                    &mut rec_z,
                    &mut rec_beta,
                    &mut rec_alpha,
                    &mut rec_out,
                    &mut proj_buf,
                    dim,
                    ops.as_deref_mut(),
                )?,
            }

            Self::apply_layer_tail_single(
                backend,
                weights,
                &prefix,
                &mut hidden,
                &proj_buf,
                &mut norm_buf,
                &mut gate_buf,
                &mut up_buf,
                &mut down_buf,
                dim,
                inter_dim,
                cfg.rms_norm_eps,
                layer,
                position,
                ops.as_deref_mut(),
            )?;
        }

        qwen_kv.finalize_token();

        let final_norm_w = timed!(ops, dequant, weights.f32_slice("output_norm.weight")?);
        timed!(
            ops,
            norm,
            rms_norm::rms_norm(&mut hidden, final_norm_w, cfg.rms_norm_eps)
        );

        timed_matmul_bucket!(
            ops,
            matmul_lm_head,
            Self::write_normalized_single_logits(
                backend, &hidden, dim, vocab_size, weights, logits,
            )?
        );
        Ok(())
    }

    fn forward_batch(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        self.forward_batch_impl(ctx, token_ids, kv, weights, Some(logits), None)
    }

    fn forward_batch_all_logits(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        self.forward_batch_impl(ctx, token_ids, kv, weights, None, Some(logits_all))
    }

    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.architecture == "qwen35",
            "Qwen35Forward only supports qwen35, got {}",
            config.architecture
        );
        let _ = Self::recurrent_dims(config)?;
        anyhow::ensure!(
            config.qwen35_full_attention_interval.unwrap_or(0) > 0,
            "qwen35 full_attention_interval must be > 0"
        );
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "qwen35"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::gguf::MetadataValue;
    use crate::gguf::header::GgufHeader;
    use crate::gguf::mmap::MappedModel;
    use crate::gguf::tensor::GgmlType;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_header(kv: Vec<(&str, MetadataValue)>) -> GgufHeader {
        let mut metadata = HashMap::new();
        for (k, v) in kv {
            metadata.insert(k.to_string(), v);
        }
        GgufHeader {
            version: 3,
            tensor_count: 0,
            metadata,
        }
    }

    fn align_to(offset: usize, alignment: usize) -> usize {
        offset.div_ceil(alignment) * alignment
    }

    fn push_string_metadata(buf: &mut Vec<u8>, key: &str, value: &str) {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value.as_bytes());
    }

    fn push_u32_metadata(buf: &mut Vec<u8>, key: &str, value: u32) {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_tensor_info(
        buf: &mut Vec<u8>,
        name: &str,
        shape: &[u64],
        dtype: GgmlType,
        offset: u64,
    ) {
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dim in shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&(dtype as u32).to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn build_qwen35_logits_test_gguf(
        output_norm: &[f32],
        output_weight: &[f32],
        dim: usize,
        vocab_size: usize,
    ) -> Vec<u8> {
        let alignment = 32usize;
        let output_norm_bytes = f32_bytes(output_norm);
        let output_weight_bytes = f32_bytes(output_weight);
        let output_weight_offset = align_to(output_norm_bytes.len(), alignment);

        let mut buf = Vec::new();
        buf.extend_from_slice(&crate::gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&crate::gguf::GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        push_string_metadata(&mut buf, "general.architecture", "qwen35");
        push_u32_metadata(&mut buf, "general.alignment", alignment as u32);
        push_tensor_info(
            &mut buf,
            "output_norm.weight",
            &[dim as u64],
            GgmlType::F32,
            0,
        );
        push_tensor_info(
            &mut buf,
            "output.weight",
            &[dim as u64, vocab_size as u64],
            GgmlType::F32,
            output_weight_offset as u64,
        );
        let data_start = align_to(buf.len(), alignment);
        buf.resize(data_start, 0);
        buf.extend_from_slice(&output_norm_bytes);
        buf.resize(data_start + output_weight_offset, 0);
        buf.extend_from_slice(&output_weight_bytes);
        buf
    }

    fn write_test_gguf_to_temp(data: &[u8]) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ax-qwen35-logits-{}-{}.gguf",
            std::process::id(),
            unique
        ));
        std::fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn test_qwen35_layer_pattern() {
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("qwen35".into()),
            ),
            ("qwen35.block_count", MetadataValue::Uint32(8)),
            ("qwen35.attention.head_count", MetadataValue::Uint32(16)),
            ("qwen35.attention.head_count_kv", MetadataValue::Uint32(8)),
            ("qwen35.embedding_length", MetadataValue::Uint32(2048)),
            ("qwen35.attention.key_length", MetadataValue::Uint32(128)),
            ("qwen35.feed_forward_length", MetadataValue::Uint32(8192)),
            ("qwen35.context_length", MetadataValue::Uint32(4096)),
            ("qwen35.full_attention_interval", MetadataValue::Uint32(4)),
            ("qwen35.ssm.conv_kernel", MetadataValue::Uint32(4)),
            ("qwen35.ssm.inner_size", MetadataValue::Uint32(1024)),
            ("qwen35.ssm.state_size", MetadataValue::Uint32(128)),
            ("qwen35.ssm.time_step_rank", MetadataValue::Uint32(8)),
            ("qwen35.ssm.group_count", MetadataValue::Uint32(2)),
        ]);
        let cfg = ModelConfig::from_gguf(&header).unwrap();
        assert!(cfg.qwen35_is_recurrent_layer(0));
        assert!(cfg.qwen35_is_recurrent_layer(1));
        assert!(cfg.qwen35_is_recurrent_layer(2));
        assert!(!cfg.qwen35_is_recurrent_layer(3));
    }

    #[test]
    fn test_qwen35_validate_requires_recurrent_dims() {
        let fwd = Qwen35Forward;
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 16,
            n_kv_heads: 8,
            embedding_dim: 2048,
            head_dim: 128,
            intermediate_dim: 8192,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(1024),
            qwen35_ssm_state_size: Some(128),
            qwen35_ssm_time_step_rank: Some(8),
            qwen35_ssm_group_count: Some(2),
        };
        fwd.validate_config(&cfg).unwrap();
    }

    #[test]
    fn test_qwen35_rope_position_honors_linear_scaling() {
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 16,
            n_kv_heads: 8,
            embedding_dim: 2048,
            head_dim: 128,
            intermediate_dim: 8192,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::Linear(8.0),
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(1024),
            qwen35_ssm_state_size: Some(128),
            qwen35_ssm_time_step_rank: Some(8),
            qwen35_ssm_group_count: Some(2),
        };

        assert!((Qwen35Forward::rope_position(&cfg, 16) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_qwen35_apply_rope_batch_uses_absolute_positions() {
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 1,
            n_kv_heads: 1,
            embedding_dim: 8,
            head_dim: 4,
            intermediate_dim: 16,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(16),
            qwen35_ssm_state_size: Some(4),
            qwen35_ssm_time_step_rank: Some(4),
            qwen35_ssm_group_count: Some(1),
        };
        let n_tokens = 2usize;
        let start_position = 7usize;
        let q_dim = 4usize;
        let kv_dim = 4usize;
        let n_heads = 1usize;
        let n_kv_heads = 1usize;
        let head_dim = 4usize;
        let mut actual_q = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5];
        let mut actual_k = vec![4.0f32, 3.0, 2.0, 1.0, 3.5, 2.5, 1.5, 0.5];
        let mut expected_q = actual_q.clone();
        let mut expected_k = actual_k.clone();

        for token_idx in 0..n_tokens {
            let q_start = token_idx * q_dim;
            let k_start = token_idx * kv_dim;
            rope::apply_rope_multi_head_scaled(
                &mut expected_q[q_start..q_start + q_dim],
                &mut expected_k[k_start..k_start + kv_dim],
                n_heads,
                n_kv_heads,
                head_dim,
                Qwen35Forward::rope_position(&cfg, start_position + token_idx),
                cfg.rope_freq_base,
            );
        }

        Qwen35Forward::apply_rope_batch(
            &cfg,
            &mut actual_q,
            &mut actual_k,
            n_tokens,
            start_position,
            q_dim,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_qwen35_validate_rejects_incompatible_head_expansion() {
        let fwd = Qwen35Forward;
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 16,
            n_kv_heads: 8,
            embedding_dim: 2048,
            head_dim: 128,
            intermediate_dim: 8192,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(768),
            qwen35_ssm_state_size: Some(128),
            qwen35_ssm_time_step_rank: Some(6),
            qwen35_ssm_group_count: Some(4),
        };

        let err = fwd.validate_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("multiple of group_count"));
    }

    #[test]
    fn test_qwen35_write_all_batch_logits_matches_per_token_reference() {
        let dim = 4usize;
        let vocab_size = 3usize;
        let n_tokens = 2usize;
        let rms_norm_eps = 1e-6f32;
        let output_norm = [1.0f32, 0.5, 1.5, 0.75];
        let output_weight = [
            0.25f32, -0.5, 1.0, 0.75, -1.0, 0.5, 0.25, -0.75, 0.1, 0.2, -0.3, 0.4,
        ];
        let hidden = vec![1.0f32, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.5];
        let gguf = build_qwen35_logits_test_gguf(&output_norm, &output_weight, dim, vocab_size);
        let path = write_test_gguf_to_temp(&gguf);

        let result = (|| {
            let model = MappedModel::open(&path).unwrap();
            let weights = WeightStore::new(&model);
            let backend = CpuBackend;

            let mut actual = Vec::new();
            Qwen35Forward::write_all_batch_logits(
                &backend,
                &hidden,
                n_tokens,
                dim,
                vocab_size,
                rms_norm_eps,
                &weights,
                &mut actual,
            )
            .unwrap();

            let mut expected = vec![0.0f32; n_tokens * vocab_size];
            for token_idx in 0..n_tokens {
                let hidden_start = token_idx * dim;
                let logits_start = token_idx * vocab_size;
                let mut token_hidden = hidden[hidden_start..hidden_start + dim].to_vec();
                Qwen35Forward::write_single_logits(
                    &backend,
                    &mut token_hidden,
                    dim,
                    vocab_size,
                    rms_norm_eps,
                    &weights,
                    &mut expected[logits_start..logits_start + vocab_size],
                )
                .unwrap();
            }

            assert_eq!(actual.len(), expected.len());
            for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "logit {idx} mismatch: actual={actual}, expected={expected}"
                );
            }
        })();

        std::fs::remove_file(&path).ok();
        result
    }
}
