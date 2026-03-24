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
//! AX does not yet have GPU kernels for the recurrent path, so the first
//! implementation is CPU-only and uses the hybrid `ModelKv::Qwen35` state.

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::compute::attention::{self, AttentionParams};
use crate::compute::rms_norm;
use crate::compute::rope;
use crate::compute::silu;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::per_head_rms_norm;
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
        Ok(dims)
    }

    fn layer_type(cfg: &ModelConfig, layer: usize) -> Qwen35LayerType {
        if cfg.qwen35_is_recurrent_layer(layer) {
            Qwen35LayerType::RecurrentGdn
        } else {
            Qwen35LayerType::FullAttention
        }
    }

    fn sigmoid_in_place(buf: &mut [f32]) {
        for v in buf {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }

    fn l2_norm_heads(buf: &mut [f32], n_heads: usize, head_dim: usize, eps: f32) {
        for head in buf.chunks_mut(head_dim).take(n_heads) {
            let sum_sq = head.iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0 / (sum_sq + eps).sqrt();
            for v in head {
                *v *= inv;
            }
        }
    }

    fn repeat_heads(input: &[f32], n_src_heads: usize, n_dst_heads: usize, head_dim: usize) -> Vec<f32> {
        if n_src_heads == n_dst_heads {
            return input.to_vec();
        }
        assert!(n_dst_heads.is_multiple_of(n_src_heads));
        let repeat = n_dst_heads / n_src_heads;
        let mut out = vec![0.0f32; n_dst_heads * head_dim];
        for src in 0..n_src_heads {
            let src_slice = &input[src * head_dim..(src + 1) * head_dim];
            for rep in 0..repeat {
                let dst = src * repeat + rep;
                out[dst * head_dim..(dst + 1) * head_dim].copy_from_slice(src_slice);
            }
        }
        out
    }

    fn depthwise_conv1d_step(
        conv_state: &mut [f32],
        input: &[f32],
        kernel: &[f32],
        conv_cache_len: usize,
        conv_dim: usize,
        out: &mut [f32],
    ) {
        assert_eq!(conv_state.len(), conv_cache_len * conv_dim);
        assert_eq!(input.len(), conv_dim);
        assert_eq!(kernel.len(), (conv_cache_len + 1) * conv_dim);
        assert_eq!(out.len(), conv_dim);

        for c in 0..conv_dim {
            let mut acc = input[c] * kernel[conv_cache_len * conv_dim + c];
            for t in 0..conv_cache_len {
                acc += conv_state[t * conv_dim + c] * kernel[t * conv_dim + c];
            }
            out[c] = acc / (1.0 + (-acc).exp());
        }

        if conv_cache_len > 0 {
            if conv_cache_len > 1 {
                conv_state.copy_within(conv_dim.., 0);
            }
            let start = (conv_cache_len - 1) * conv_dim;
            conv_state[start..start + conv_dim].copy_from_slice(input);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_delta_rule_step(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        assert_eq!(q.len(), n_heads * head_dim);
        assert_eq!(k.len(), n_heads * head_dim);
        assert_eq!(v.len(), n_heads * head_dim);
        assert_eq!(gate.len(), n_heads);
        assert_eq!(beta.len(), n_heads);
        assert_eq!(state.len(), n_heads * head_dim * head_dim);
        assert_eq!(out.len(), n_heads * head_dim);

        let scale = 1.0 / (head_dim as f32).sqrt();
        for h in 0..n_heads {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let kh = &k[h * head_dim..(h + 1) * head_dim];
            let vh = &v[h * head_dim..(h + 1) * head_dim];
            let state_h = &mut state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];
            let decay = gate[h].exp();
            for s in state_h.iter_mut() {
                *s *= decay;
            }

            let mut delta = vec![0.0f32; head_dim];
            for value_idx in 0..head_dim {
                let mut kv_mem = 0.0f32;
                for key_idx in 0..head_dim {
                    kv_mem += state_h[key_idx * head_dim + value_idx] * kh[key_idx];
                }
                delta[value_idx] = (vh[value_idx] - kv_mem) * beta[h];
            }

            for key_idx in 0..head_dim {
                let k_val = kh[key_idx];
                for value_idx in 0..head_dim {
                    state_h[key_idx * head_dim + value_idx] += k_val * delta[value_idx];
                }
            }

            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            for value_idx in 0..head_dim {
                let mut sum = 0.0f32;
                for key_idx in 0..head_dim {
                    sum += state_h[key_idx * head_dim + value_idx] * qh[key_idx];
                }
                out_h[value_idx] = sum * scale;
            }
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

        let dims = Self::recurrent_dims(cfg)?;
        let cpu = CpuBackend;
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
        let mut conv_out = vec![0.0f32; dims.conv_dim()];
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
                    let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
                    let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
                    let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;

                    timed!(ops, matmul, {
                        cpu.dequant_matmul(wq_raw, wq_dtype, &norm_buf, &mut q_gate_buf, q_dim * 2, 1, dim);
                        cpu.dequant_matmul(wk_raw, wk_dtype, &norm_buf, &mut k_buf, kv_dim, 1, dim);
                        cpu.dequant_matmul(wv_raw, wv_dtype, &norm_buf, &mut v_buf, kv_dim, 1, dim);
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
                            per_head_rms_norm(&mut q_buf, n_heads, head_dim, q_norm_w, cfg.rms_norm_eps);
                            per_head_rms_norm(&mut k_buf, n_kv_heads, head_dim, k_norm_w, cfg.rms_norm_eps);
                        });
                    }

                    let rope_position = match cfg.rope_scaling {
                        crate::model::config::RopeScaling::Linear(factor) => position as f32 / factor,
                        crate::model::config::RopeScaling::None => position as f32,
                    };
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

                    Self::sigmoid_in_place(gate_attn);
                    silu::elementwise_mul(&mut attn_out, gate_attn);

                    let (wo_raw, wo_dtype) =
                        weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
                    timed!(
                        ops,
                        matmul,
                        cpu.dequant_matmul(wo_raw, wo_dtype, &attn_out, &mut proj_buf, dim, 1, q_dim)
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
                        cpu.dequant_matmul(
                            wqkv_raw,
                            wqkv_dtype,
                            &norm_buf,
                            &mut rec_qkv,
                            dims.conv_dim(),
                            1,
                            dim,
                        );
                        cpu.dequant_matmul(
                            wgate_raw,
                            wgate_dtype,
                            &norm_buf,
                            &mut rec_z,
                            dims.inner_size,
                            1,
                            dim,
                        );
                        cpu.dequant_matmul(
                            wbeta_raw,
                            wbeta_dtype,
                            &norm_buf,
                            &mut rec_beta,
                            dims.time_step_rank,
                            1,
                            dim,
                        );
                        cpu.dequant_matmul(
                            walpha_raw,
                            walpha_dtype,
                            &norm_buf,
                            &mut rec_alpha,
                            dims.time_step_rank,
                            1,
                            dim,
                        );
                    });

                    Self::sigmoid_in_place(&mut rec_beta);
                    let dt_bias = timed!(
                        ops,
                        dequant,
                        weights.f32_slice(&format!("{prefix}.ssm_dt.bias"))?
                    );
                    let a = timed!(ops, dequant, weights.f32_slice(&format!("{prefix}.ssm_a"))?);
                    for ((alpha, &bias), &a_val) in rec_alpha.iter_mut().zip(dt_bias.iter()).zip(a.iter()) {
                        *alpha = (1.0 + (*alpha + bias).exp()).ln() * a_val;
                    }

                    let conv_kernel = timed!(
                        ops,
                        dequant,
                        weights.f32_slice(&format!("{prefix}.ssm_conv1d.weight"))?
                    );
                    let conv_cache_len = qwen_kv.conv_cache_len();
                    let conv_dim = qwen_kv.conv_dim();
                    Self::depthwise_conv1d_step(
                        qwen_kv.conv_state_mut(layer),
                        &rec_qkv,
                        conv_kernel,
                        conv_cache_len,
                        conv_dim,
                        &mut conv_out,
                    );

                    let key_dim = dims.key_dim();
                    let value_dim = dims.value_dim();
                    let mut q_lin = conv_out[..key_dim].to_vec();
                    let mut k_lin = conv_out[key_dim..2 * key_dim].to_vec();
                    let v_lin = conv_out[2 * key_dim..2 * key_dim + value_dim].to_vec();

                    Self::l2_norm_heads(&mut q_lin, dims.group_count, dims.state_size, cfg.rms_norm_eps);
                    Self::l2_norm_heads(&mut k_lin, dims.group_count, dims.state_size, cfg.rms_norm_eps);
                    let q_rep =
                        Self::repeat_heads(&q_lin, dims.group_count, dims.time_step_rank, dims.state_size);
                    let k_rep =
                        Self::repeat_heads(&k_lin, dims.group_count, dims.time_step_rank, dims.state_size);

                    Self::gated_delta_rule_step(
                        &q_rep,
                        &k_rep,
                        &v_lin,
                        &rec_alpha,
                        &rec_beta,
                        qwen_kv.recurrent_state_mut(layer),
                        dims.time_step_rank,
                        dims.state_size,
                        &mut rec_out,
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
                        cpu.dequant_matmul(
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

            let (wg_raw, wg_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            timed!(ops, matmul, {
                cpu.batch_dequant_matvec(
                    &[(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)],
                    &norm_buf,
                    dim,
                    &mut [&mut gate_buf, &mut up_buf],
                );
            });
            silu::silu_elementwise_mul(&mut gate_buf, &up_buf);

            let (wd_raw, wd_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            timed!(
                ops,
                matmul,
                cpu.dequant_matmul(wd_raw, wd_dtype, &gate_buf, &mut down_buf, dim, 1, inter_dim)
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
            cpu.dequant_matmul(lm_raw, lm_dtype, &hidden, logits, vocab_size, 1, dim)
        );
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
            ("general.architecture", MetadataValue::String("qwen35".into())),
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
}
