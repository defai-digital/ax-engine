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
//! AX does not yet have dedicated GPU kernels for the recurrent path, so the
//! hybrid `ModelKv::Qwen35` state remains CPU-resident. Dense projections still
//! route through the active backend, allowing Metal/Hybrid backends to
//! accelerate the heavy dequant+matmul work while recurrent state updates stay
//! on CPU.

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

impl Qwen35Forward {
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
        let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            let attn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?
            );
            timed!(
                ops,
                norm,
                rms_norm::rms_norm_out(&hidden, attn_norm_w, &mut norm_buf, cfg.rms_norm_eps)
            );

            match Self::layer_type(cfg, layer) {
                Qwen35LayerType::FullAttention => {
                    let (wq_raw, wq_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
                    let (wk_raw, wk_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
                    let (wv_raw, wv_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;

                    timed!(ops, matmul, {
                        backend.dequant_matmul(
                            wq_raw,
                            wq_dtype,
                            &norm_buf,
                            &mut q_gate_buf,
                            q_dim * 2,
                            1,
                            dim,
                        );
                        backend.dequant_matmul(
                            wk_raw, wk_dtype, &norm_buf, &mut k_buf, kv_dim, 1, dim,
                        );
                        backend.dequant_matmul(
                            wv_raw, wv_dtype, &norm_buf, &mut v_buf, kv_dim, 1, dim,
                        );
                    });
                    q_buf.copy_from_slice(&q_gate_buf[..q_dim]);
                    let gate_attn = &mut q_gate_buf[q_dim..];

                    if weights.has(&format!("{prefix}.attn_q_norm.weight")) {
                        let q_norm_w = timed!(
                            ops,
                            dequant,
                            weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?
                        );
                        let k_norm_w = timed!(
                            ops,
                            dequant,
                            weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?
                        );
                        timed!(ops, norm, {
                            per_head_rms_norm(
                                &mut q_buf,
                                n_heads,
                                head_dim,
                                q_norm_w,
                                cfg.rms_norm_eps,
                            );
                            per_head_rms_norm(
                                &mut k_buf,
                                n_kv_heads,
                                head_dim,
                                k_norm_w,
                                cfg.rms_norm_eps,
                            );
                        });
                    }

                    let rope_position = Self::rope_position(cfg, position);
                    timed!(
                        ops,
                        rope,
                        rope::apply_rope_multi_head_scaled(
                            &mut q_buf,
                            &mut k_buf,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            rope_position,
                            cfg.rope_freq_base,
                        )
                    );

                    qwen_kv.attention_append(layer, &k_buf, &v_buf);
                    let seq_len = qwen_kv.seq_len() + 1;
                    timed!(
                        ops,
                        attention,
                        attention::multi_head_attention(
                            &q_buf,
                            qwen_kv.attention_k_slice_including_current(layer, seq_len),
                            qwen_kv.attention_v_slice_including_current(layer, seq_len),
                            &mut attn_out,
                            &full_attn_params,
                            seq_len,
                        )
                    );

                    gdn::sigmoid_in_place(gate_attn);
                    silu::elementwise_mul(&mut attn_out, gate_attn);

                    let (wo_raw, wo_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
                    timed!(
                        ops,
                        matmul,
                        backend.dequant_matmul(
                            wo_raw,
                            wo_dtype,
                            &attn_out,
                            &mut proj_buf,
                            dim,
                            1,
                            q_dim
                        )
                    );
                }
                Qwen35LayerType::RecurrentGdn => {
                    anyhow::ensure!(
                        qwen_kv.is_recurrent_layer(layer),
                        "qwen35 KV/state layer mapping mismatch at layer {layer}"
                    );
                    let (wqkv_raw, wqkv_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_qkv.weight"))?;
                    let (wgate_raw, wgate_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_gate.weight"))?;
                    let (wbeta_raw, wbeta_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.ssm_beta.weight"))?;
                    let (walpha_raw, walpha_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.ssm_alpha.weight"))?;
                    timed!(ops, matmul, {
                        backend.dequant_matmul(
                            wqkv_raw,
                            wqkv_dtype,
                            &norm_buf,
                            &mut rec_qkv,
                            dims.conv_dim(),
                            1,
                            dim,
                        );
                        backend.dequant_matmul(
                            wgate_raw,
                            wgate_dtype,
                            &norm_buf,
                            &mut rec_z,
                            dims.inner_size,
                            1,
                            dim,
                        );
                        backend.dequant_matmul(
                            wbeta_raw,
                            wbeta_dtype,
                            &norm_buf,
                            &mut rec_beta,
                            dims.time_step_rank,
                            1,
                            dim,
                        );
                        backend.dequant_matmul(
                            walpha_raw,
                            walpha_dtype,
                            &norm_buf,
                            &mut rec_alpha,
                            dims.time_step_rank,
                            1,
                            dim,
                        );
                    });

                    let dt_bias = timed!(
                        ops,
                        dequant,
                        weights.f32_slice(&format!("{prefix}.ssm_dt.bias"))?
                    );
                    let a = timed!(ops, dequant, weights.f32_slice(&format!("{prefix}.ssm_a"))?);

                    let conv_kernel = timed!(
                        ops,
                        dequant,
                        weights.f32_slice(&format!("{prefix}.ssm_conv1d.weight"))?
                    );
                    gdn::prepare_alpha_beta(&mut rec_alpha, &mut rec_beta, dt_bias, a);
                    let conv_cache_len = qwen_kv.conv_cache_len();
                    let conv_dim = qwen_kv.conv_dim();
                    let mut conv_out = vec![0.0f32; dims.conv_dim()];
                    let (conv_state, _) = qwen_kv.recurrent_buffers_mut(layer);
                    timed!(
                        ops,
                        attention,
                        backend.qwen35_causal_conv_sequence(
                            &rec_qkv,
                            conv_kernel,
                            conv_state,
                            &mut conv_out,
                            1,
                            conv_cache_len,
                            conv_dim,
                        )
                    );

                    let key_dim = dims.key_dim();
                    let value_dim = dims.value_dim();
                    let mut q_lin = conv_out[..key_dim].to_vec();
                    let mut k_lin = conv_out[key_dim..2 * key_dim].to_vec();
                    let v_lin = conv_out[2 * key_dim..2 * key_dim + value_dim].to_vec();
                    gdn::l2_norm_heads(
                        &mut q_lin,
                        dims.group_count,
                        dims.state_size,
                        cfg.rms_norm_eps,
                    );
                    gdn::l2_norm_heads(
                        &mut k_lin,
                        dims.group_count,
                        dims.state_size,
                        cfg.rms_norm_eps,
                    );
                    let q_rep = gdn::repeat_heads(
                        &q_lin,
                        dims.group_count,
                        dims.time_step_rank,
                        dims.state_size,
                    );
                    let k_rep = gdn::repeat_heads(
                        &k_lin,
                        dims.group_count,
                        dims.time_step_rank,
                        dims.state_size,
                    );
                    let (_, recurrent_state) = qwen_kv.recurrent_buffers_mut(layer);
                    timed!(
                        ops,
                        attention,
                        backend.qwen35_gated_delta_sequence(
                            &q_rep,
                            &k_rep,
                            &v_lin,
                            &rec_alpha,
                            &rec_beta,
                            recurrent_state,
                            &mut rec_out,
                            1,
                            dims.time_step_rank,
                            dims.state_size,
                        )
                    );

                    let ssm_norm_w = timed!(
                        ops,
                        dequant,
                        weights.f32_slice(&format!("{prefix}.ssm_norm.weight"))?
                    );
                    for head in 0..dims.time_step_rank {
                        let start = head * dims.state_size;
                        let end = start + dims.state_size;
                        rms_norm::rms_norm(&mut rec_out[start..end], ssm_norm_w, cfg.rms_norm_eps);
                    }
                    let mut z_gate = rec_z.clone();
                    silu::silu(&mut z_gate);
                    silu::elementwise_mul(&mut rec_out, &z_gate);

                    let (ssm_out_raw, ssm_out_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.ssm_out.weight"))?;
                    timed!(
                        ops,
                        matmul,
                        backend.dequant_matmul(
                            ssm_out_raw,
                            ssm_out_dtype,
                            &rec_out,
                            &mut proj_buf,
                            dim,
                            1,
                            dims.inner_size,
                        )
                    );
                }
            }

            silu::elementwise_add(&mut hidden, &proj_buf);

            let post_attn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?
            );
            timed!(
                ops,
                norm,
                rms_norm::rms_norm_out(&hidden, post_attn_norm_w, &mut norm_buf, cfg.rms_norm_eps)
            );

            let (wg_raw, wg_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            timed!(ops, matmul, {
                backend.batch_dequant_matvec(
                    &[(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)],
                    &norm_buf,
                    dim,
                    &mut [&mut gate_buf, &mut up_buf],
                );
            });
            silu::silu_elementwise_mul(&mut gate_buf, &up_buf);

            let (wd_raw, wd_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            timed!(
                ops,
                matmul,
                backend.dequant_matmul(
                    wd_raw,
                    wd_dtype,
                    &gate_buf,
                    &mut down_buf,
                    dim,
                    1,
                    inter_dim
                )
            );
            silu::elementwise_add(&mut hidden, &down_buf);
        }

        qwen_kv.finalize_token();

        let final_norm_w = timed!(ops, dequant, weights.f32_slice("output_norm.weight")?);
        timed!(
            ops,
            norm,
            rms_norm::rms_norm(&mut hidden, final_norm_w, cfg.rms_norm_eps)
        );

        let lm_weight_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        anyhow::ensure!(logits.len() >= vocab_size, "logits buffer too small");
        timed!(
            ops,
            matmul,
            backend.dequant_matmul(lm_raw, lm_dtype, &hidden, logits, vocab_size, 1, dim)
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
        let force_serial = env_flag_enabled("AX_SERIAL_PREFILL");
        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            anyhow::bail!("Qwen35Forward requires ModelKv::Qwen35");
        };
        if force_serial || token_ids.len() <= 1 || qwen_kv.seq_len() != 0 {
            let start_pos = kv.seq_len();
            for (i, &tid) in token_ids.iter().enumerate() {
                logits.fill(0.0);
                self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, None)?;
            }
            return Ok(());
        }

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

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            for token_idx in 0..n_tokens {
                let start = token_idx * dim;
                let end = start + dim;
                rms_norm::rms_norm_out(
                    &hidden[start..end],
                    attn_norm_w,
                    &mut norm_buf[start..end],
                    cfg.rms_norm_eps,
                );
            }

            match Self::layer_type(cfg, layer) {
                Qwen35LayerType::FullAttention => {
                    let (wq_raw, wq_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
                    let (wk_raw, wk_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
                    let (wv_raw, wv_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;

                    Self::batched_dequant_matmul_token_major(
                        backend,
                        wq_raw,
                        wq_dtype,
                        &norm_buf,
                        &mut q_gate_batch,
                        n_tokens,
                        q_dim * 2,
                        dim,
                    );
                    Self::batched_dequant_matmul_token_major(
                        backend,
                        wk_raw,
                        wk_dtype,
                        &norm_buf,
                        &mut k_batch,
                        n_tokens,
                        kv_dim,
                        dim,
                    );
                    Self::batched_dequant_matmul_token_major(
                        backend,
                        wv_raw,
                        wv_dtype,
                        &norm_buf,
                        &mut v_batch,
                        n_tokens,
                        kv_dim,
                        dim,
                    );

                    for token_idx in 0..n_tokens {
                        let src_start = token_idx * q_dim * 2;
                        let q_start = token_idx * q_dim;
                        q_batch[q_start..q_start + q_dim]
                            .copy_from_slice(&q_gate_batch[src_start..src_start + q_dim]);
                    }

                    if weights.has(&format!("{prefix}.attn_q_norm.weight")) {
                        let q_norm_w =
                            weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?;
                        let k_norm_w =
                            weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?;
                        for token_idx in 0..n_tokens {
                            let q_start = token_idx * q_dim;
                            let k_start = token_idx * kv_dim;
                            per_head_rms_norm(
                                &mut q_batch[q_start..q_start + q_dim],
                                n_heads,
                                head_dim,
                                q_norm_w,
                                cfg.rms_norm_eps,
                            );
                            per_head_rms_norm(
                                &mut k_batch[k_start..k_start + kv_dim],
                                n_kv_heads,
                                head_dim,
                                k_norm_w,
                                cfg.rms_norm_eps,
                            );
                        }
                    }

                    for token_idx in 0..n_tokens {
                        let q_start = token_idx * q_dim;
                        let k_start = token_idx * kv_dim;
                        let rope_position = Self::rope_position(cfg, token_idx);
                        rope::apply_rope_multi_head_scaled(
                            &mut q_batch[q_start..q_start + q_dim],
                            &mut k_batch[k_start..k_start + kv_dim],
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            rope_position,
                            cfg.rope_freq_base,
                        );
                    }

                    backend.attention_prefill(
                        &q_batch,
                        &k_batch,
                        &v_batch,
                        &mut attn_out_batch,
                        n_tokens,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                    );
                    for token_idx in 0..n_tokens {
                        let gate_start = token_idx * q_dim * 2 + q_dim;
                        let attn_start = token_idx * q_dim;
                        let gate = &mut q_gate_batch[gate_start..gate_start + q_dim];
                        gdn::sigmoid_in_place(gate);
                        silu::elementwise_mul(
                            &mut attn_out_batch[attn_start..attn_start + q_dim],
                            gate,
                        );
                    }

                    qwen_kv.attention_append_batch(layer, &k_batch, &v_batch, n_tokens);

                    let (wo_raw, wo_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
                    Self::batched_dequant_matmul_token_major(
                        backend,
                        wo_raw,
                        wo_dtype,
                        &attn_out_batch,
                        &mut proj_buf,
                        n_tokens,
                        dim,
                        q_dim,
                    );
                }
                Qwen35LayerType::RecurrentGdn => {
                    anyhow::ensure!(
                        qwen_kv.is_recurrent_layer(layer),
                        "qwen35 KV/state layer mapping mismatch at layer {layer}"
                    );
                    let (wqkv_raw, wqkv_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_qkv.weight"))?;
                    let (wgate_raw, wgate_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_gate.weight"))?;
                    let (wbeta_raw, wbeta_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.ssm_beta.weight"))?;
                    let (walpha_raw, walpha_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.ssm_alpha.weight"))?;

                    Self::batched_dequant_matmul_token_major(
                        backend,
                        wqkv_raw,
                        wqkv_dtype,
                        &norm_buf,
                        &mut rec_qkv_batch,
                        n_tokens,
                        conv_dim,
                        dim,
                    );
                    Self::batched_dequant_matmul_token_major(
                        backend,
                        wgate_raw,
                        wgate_dtype,
                        &norm_buf,
                        &mut rec_z_batch,
                        n_tokens,
                        dims.inner_size,
                        dim,
                    );
                    Self::batched_dequant_matmul_token_major(
                        backend,
                        wbeta_raw,
                        wbeta_dtype,
                        &norm_buf,
                        &mut rec_beta_batch,
                        n_tokens,
                        dims.time_step_rank,
                        dim,
                    );
                    Self::batched_dequant_matmul_token_major(
                        backend,
                        walpha_raw,
                        walpha_dtype,
                        &norm_buf,
                        &mut rec_alpha_batch,
                        n_tokens,
                        dims.time_step_rank,
                        dim,
                    );

                    let dt_bias = weights.f32_slice(&format!("{prefix}.ssm_dt.bias"))?;
                    let a = weights.f32_slice(&format!("{prefix}.ssm_a"))?;
                    let conv_kernel = weights.f32_slice(&format!("{prefix}.ssm_conv1d.weight"))?;
                    let ssm_norm_w = weights.f32_slice(&format!("{prefix}.ssm_norm.weight"))?;

                    rec_out_batch.fill(0.0);
                    gdn::prepare_alpha_beta(&mut rec_alpha_batch, &mut rec_beta_batch, dt_bias, a);
                    let conv_cache_len = qwen_kv.conv_cache_len();
                    let mut conv_out_batch = vec![0.0f32; n_tokens * conv_dim];
                    let (conv_state, _) = qwen_kv.recurrent_buffers_mut(layer);
                    backend.qwen35_causal_conv_sequence(
                        &rec_qkv_batch,
                        conv_kernel,
                        conv_state,
                        &mut conv_out_batch,
                        n_tokens,
                        conv_cache_len,
                        conv_dim,
                    );

                    let value_dim = dims.value_dim();
                    let mut q_batch = vec![0.0f32; n_tokens * value_dim];
                    let mut k_batch = vec![0.0f32; n_tokens * value_dim];
                    let mut v_batch = vec![0.0f32; n_tokens * value_dim];
                    for token_idx in 0..n_tokens {
                        let conv_start = token_idx * conv_dim;
                        let conv_end = conv_start + conv_dim;
                        let conv_out = &conv_out_batch[conv_start..conv_end];
                        let mut q_lin = conv_out[..dims.key_dim()].to_vec();
                        let mut k_lin = conv_out[dims.key_dim()..2 * dims.key_dim()].to_vec();
                        let v_lin = &conv_out[2 * dims.key_dim()..2 * dims.key_dim() + value_dim];
                        gdn::l2_norm_heads(
                            &mut q_lin,
                            dims.group_count,
                            dims.state_size,
                            cfg.rms_norm_eps,
                        );
                        gdn::l2_norm_heads(
                            &mut k_lin,
                            dims.group_count,
                            dims.state_size,
                            cfg.rms_norm_eps,
                        );
                        let q_rep = gdn::repeat_heads(
                            &q_lin,
                            dims.group_count,
                            dims.time_step_rank,
                            dims.state_size,
                        );
                        let k_rep = gdn::repeat_heads(
                            &k_lin,
                            dims.group_count,
                            dims.time_step_rank,
                            dims.state_size,
                        );
                        let out_start = token_idx * value_dim;
                        let out_end = out_start + value_dim;
                        q_batch[out_start..out_end].copy_from_slice(&q_rep);
                        k_batch[out_start..out_end].copy_from_slice(&k_rep);
                        v_batch[out_start..out_end].copy_from_slice(v_lin);
                    }

                    let (_, recurrent_state) = qwen_kv.recurrent_buffers_mut(layer);
                    backend.qwen35_gated_delta_sequence(
                        &q_batch,
                        &k_batch,
                        &v_batch,
                        &rec_alpha_batch,
                        &rec_beta_batch,
                        recurrent_state,
                        &mut rec_out_batch,
                        n_tokens,
                        dims.time_step_rank,
                        dims.state_size,
                    );

                    for token_idx in 0..n_tokens {
                        let rec_out_start = token_idx * dims.inner_size;
                        let rec_out_end = rec_out_start + dims.inner_size;
                        for head in 0..dims.time_step_rank {
                            let head_start = rec_out_start + head * dims.state_size;
                            let head_end = head_start + dims.state_size;
                            rms_norm::rms_norm(
                                &mut rec_out_batch[head_start..head_end],
                                ssm_norm_w,
                                cfg.rms_norm_eps,
                            );
                        }

                        let mut z_gate = rec_z_batch
                            [token_idx * dims.inner_size..(token_idx + 1) * dims.inner_size]
                            .to_vec();
                        silu::silu(&mut z_gate);
                        silu::elementwise_mul(
                            &mut rec_out_batch[rec_out_start..rec_out_end],
                            &z_gate,
                        );
                    }

                    let (ssm_out_raw, ssm_out_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.ssm_out.weight"))?;
                    Self::batched_dequant_matmul_token_major(
                        backend,
                        ssm_out_raw,
                        ssm_out_dtype,
                        &rec_out_batch,
                        &mut proj_buf,
                        n_tokens,
                        dim,
                        dims.inner_size,
                    );
                }
            }

            silu::elementwise_add(&mut hidden, &proj_buf);

            let post_attn_norm_w =
                weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
            for token_idx in 0..n_tokens {
                let start = token_idx * dim;
                let end = start + dim;
                rms_norm::rms_norm_out(
                    &hidden[start..end],
                    post_attn_norm_w,
                    &mut norm_buf[start..end],
                    cfg.rms_norm_eps,
                );
            }

            let (wg_raw, wg_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            Self::batched_dequant_matmul_token_major(
                backend,
                wg_raw,
                wg_dtype,
                &norm_buf,
                &mut gate_buf,
                n_tokens,
                inter_dim,
                dim,
            );
            Self::batched_dequant_matmul_token_major(
                backend,
                wu_raw,
                wu_dtype,
                &norm_buf,
                &mut up_buf,
                n_tokens,
                inter_dim,
                dim,
            );
            silu::silu_elementwise_mul(&mut gate_buf, &up_buf);

            let (wd_raw, wd_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            Self::batched_dequant_matmul_token_major(
                backend,
                wd_raw,
                wd_dtype,
                &gate_buf,
                &mut down_buf,
                n_tokens,
                dim,
                inter_dim,
            );
            silu::elementwise_add(&mut hidden, &down_buf);
        }

        qwen_kv.finalize_batch(n_tokens);

        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        let last_hidden = &mut hidden[(n_tokens - 1) * dim..n_tokens * dim];
        rms_norm::rms_norm(last_hidden, final_norm_w, cfg.rms_norm_eps);
        let lm_weight_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        anyhow::ensure!(logits.len() >= vocab_size, "logits buffer too small");
        backend.dequant_matmul(lm_raw, lm_dtype, last_hidden, logits, vocab_size, 1, dim);
        Ok(())
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
    use crate::gguf::MetadataValue;
    use crate::gguf::header::GgufHeader;
    use std::collections::HashMap;

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
}
