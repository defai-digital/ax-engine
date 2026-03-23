//! Mixtral (Mixture of Experts) transformer forward pass.
//!
//! Identical to LLaMA/Mistral except the dense FFN is replaced with a sparse MoE layer:
//!   1. Router: matmul(hidden, router_weight) → softmax → top-K expert selection
//!   2. For each selected expert: run that expert's FFN (gate → SiLU*up → down)
//!   3. Weighted sum of expert outputs
//!
//! Weight naming in GGUF:
//!   - Router:  `blk.{layer}.ffn_gate_inp.weight`   [n_expert × dim]
//!   - Gate:    `blk.{layer}.ffn_gate.{expert}.weight`
//!   - Up:      `blk.{layer}.ffn_up.{expert}.weight`
//!   - Down:    `blk.{layer}.ffn_down.{expert}.weight`
//!
//! GPU strategy:
//!   - Decode (single token): router on CPU (tiny matmul), run top-K expert FFNs on GPU
//!   - Batch prefill: falls back to serial forward_single (MoE routing is per-token)

use crate::compute::attention;
use crate::compute::rms_norm;
use crate::compute::rope;
use crate::compute::silu;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::weights::WeightStore;

/// Timing helper.
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

/// Softmax top-K: returns (indices, weights) for the top-K experts.
fn softmax_top_k(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    // Find top-K indices by value
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);

    // Softmax over only the selected experts
    let max_val = indexed
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = indexed.iter().map(|(_, v)| (*v - max_val).exp()).sum();
    let inv_sum = 1.0 / exp_sum;

    let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = indexed
        .iter()
        .map(|(_, v)| (*v - max_val).exp() * inv_sum)
        .collect();

    (indices, weights)
}

/// Mixtral (MoE) forward pass implementation.
#[derive(Debug)]
pub struct MixtralForward;

impl MixtralForward {}

impl ForwardPass for MixtralForward {
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
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let n_expert = cfg.n_expert.unwrap_or(8) as usize;
        let n_expert_used = cfg.n_expert_used.unwrap_or(2) as usize;

        assert!(logits.len() >= vocab_size);

        let cpu_kv = kv
            .as_cpu_mut()
            .expect("MixtralForward requires ModelKv::Cpu (MoE GPU not yet implemented)");

        // Step 1: Token embedding
        let mut hidden = vec![0.0f32; dim];
        timed!(
            ops,
            dequant,
            weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?
        );

        // Scratch buffers
        let mut norm_buf = vec![0.0f32; dim];
        let mut q_buf = vec![0.0f32; n_heads * head_dim];
        let mut k_buf = vec![0.0f32; n_kv_heads * head_dim];
        let mut v_buf = vec![0.0f32; n_kv_heads * head_dim];
        let mut attn_out = vec![0.0f32; n_heads * head_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut expert_out = vec![0.0f32; dim];
        let mut moe_out = vec![0.0f32; dim];

        // Step 2: Transformer layers
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            // 2a. Attention norm
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

            // 2b. Q/K/V projections
            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;

            timed!(ops, matmul, {
                ctx.backend.batch_dequant_matvec(
                    &[
                        (wq_raw, wq_dtype, n_heads * head_dim),
                        (wk_raw, wk_dtype, n_kv_heads * head_dim),
                        (wv_raw, wv_dtype, n_kv_heads * head_dim),
                    ],
                    &norm_buf,
                    dim,
                    &mut [&mut q_buf, &mut k_buf, &mut v_buf],
                );
            });

            // 2c. RoPE
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

            // 2d. KV cache
            cpu_kv.append_and_advance(layer, &k_buf, &v_buf);

            // 2e. Attention
            let seq_len = cpu_kv.seq_len() + 1;
            timed!(
                ops,
                attention,
                attention::multi_head_attention(
                    &q_buf,
                    cpu_kv.k_slice_including_current(layer, seq_len),
                    cpu_kv.v_slice_including_current(layer, seq_len),
                    &mut attn_out,
                    ctx.attn_params,
                    seq_len,
                )
            );

            // 2f. Output projection + residual
            let (wo_raw, wo_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            timed!(
                ops,
                matmul,
                ctx.backend.dequant_matmul(
                    wo_raw,
                    wo_dtype,
                    &attn_out,
                    &mut proj_buf,
                    dim,
                    1,
                    n_heads * head_dim,
                )
            );
            silu::elementwise_add(&mut hidden, &proj_buf);

            // 2g. FFN norm
            let ffn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?
            );
            timed!(
                ops,
                norm,
                rms_norm::rms_norm_out(&hidden, ffn_norm_w, &mut norm_buf, cfg.rms_norm_eps)
            );

            // 2h. MoE routing
            let router_name = format!("{prefix}.ffn_gate_inp.weight");
            let router_w = timed!(ops, dequant, weights.dequantize(&router_name)?);
            let mut router_logits = vec![0.0f32; n_expert];
            for e in 0..n_expert {
                let row = &router_w[e * dim..(e + 1) * dim];
                router_logits[e] = row.iter().zip(norm_buf.iter()).map(|(&w, &x)| w * x).sum();
            }
            let (expert_ids, expert_weights) = softmax_top_k(&router_logits, n_expert_used);

            // 2i. Run selected experts and accumulate weighted output
            moe_out.fill(0.0);
            for (&eid, &ew) in expert_ids.iter().zip(expert_weights.iter()) {
                let (wg_raw, wg_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_gate.{eid}.weight"))?;
                let (wu_raw, wu_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_up.{eid}.weight"))?;

                timed!(ops, matmul, {
                    ctx.backend.batch_dequant_matvec(
                        &[(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)],
                        &norm_buf,
                        dim,
                        &mut [&mut gate_buf, &mut up_buf],
                    );
                });

                // SiLU(gate) * up
                silu::silu_elementwise_mul(&mut gate_buf, &up_buf);

                // Down projection
                let (wd_raw, wd_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_down.{eid}.weight"))?;
                timed!(
                    ops,
                    matmul,
                    ctx.backend.dequant_matmul(
                        wd_raw,
                        wd_dtype,
                        &gate_buf,
                        &mut expert_out,
                        dim,
                        1,
                        inter_dim,
                    )
                );

                // Accumulate: moe_out += weight * expert_out
                for (o, &e) in moe_out.iter_mut().zip(expert_out.iter()) {
                    *o += ew * e;
                }
            }

            // 2j. Residual add
            silu::elementwise_add(&mut hidden, &moe_out);
        }

        cpu_kv.finalize_token();

        // Step 3: Final RMSNorm
        let final_norm_w = timed!(ops, dequant, weights.f32_slice("output_norm.weight")?);
        timed!(
            ops,
            norm,
            rms_norm::rms_norm(&mut hidden, final_norm_w, cfg.rms_norm_eps)
        );

        // Step 4: LM head
        let lm_weight_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        timed!(
            ops,
            matmul,
            ctx.backend
                .dequant_matmul(lm_raw, lm_dtype, &hidden, logits, vocab_size, 1, dim)
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
        // MoE batch prefill: fall back to serial (routing is per-token)
        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, None)?;
        }
        Ok(())
    }

    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.n_expert.is_some(),
            "MixtralForward requires n_expert in model config (GGUF: {arch}.expert_count)",
            arch = config.architecture,
        );
        anyhow::ensure!(
            config.n_expert_used.is_some(),
            "MixtralForward requires n_expert_used in model config (GGUF: {arch}.expert_used_count)",
            arch = config.architecture,
        );
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "mixtral"
    }

    fn supports_pipelined_decode(&self, _ctx: &ForwardContext) -> bool {
        false // MoE routing requires CPU-GPU sync per layer
    }

    fn embed_pipelined_token(
        &self,
        _ctx: &ForwardContext,
        _token_id: u32,
        _hidden_buf: &ax_metal::MetalBuffer,
        _weights: &WeightStore,
    ) -> anyhow::Result<()> {
        anyhow::bail!("MixtralForward does not support pipelined decode")
    }

    fn encode_pending_decode_step(
        &self,
        _ctx: &ForwardContext,
        _hidden_buf: &ax_metal::MetalBuffer,
        _position: usize,
        _kv: &mut ModelKv,
        _weights: &WeightStore,
    ) -> anyhow::Result<Option<ax_metal::PendingFrame>> {
        Ok(None) // Not supported
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixtral_forward_arch_name() {
        let fwd = MixtralForward;
        assert_eq!(fwd.arch_name(), "mixtral");
    }

    #[test]
    fn test_softmax_top_k_basic() {
        let logits = vec![1.0, 3.0, 2.0, 0.5, 4.0, 0.1, 0.2, 0.3];
        let (indices, weights) = softmax_top_k(&logits, 2);
        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        // Top-2 should be expert 4 (4.0) and expert 1 (3.0)
        assert_eq!(indices[0], 4);
        assert_eq!(indices[1], 1);
        // Weights should sum to ~1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights sum = {sum}");
        // Expert 4 should have higher weight
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_softmax_top_k_single() {
        let logits = vec![1.0, 5.0, 2.0];
        let (indices, weights) = softmax_top_k(&logits, 1);
        assert_eq!(indices, vec![1]);
        assert!((weights[0] - 1.0).abs() < 1e-5);
    }
}
