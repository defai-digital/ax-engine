//! Qwen3 MoE transformer forward pass.
//!
//! Pure transformer architecture with Mixture-of-Experts FFN on every layer.
//! Key characteristics:
//!   1. Separate Q/K/V projections (no fused QKV)
//!   2. Per-head QK RMSNorm before RoPE
//!   3. All layers use MoE FFN (no dense FFN layers)
//!   4. Top-k expert routing with softmax + re-normalization
//!   5. SiLU activation in expert FFN
//!   6. No shared experts, no SSM/recurrent layers
//!   7. Standard NeoX RoPE

use crate::backend::Backend;
use crate::backend::metal::MetalOps;
use crate::compute::{attention, rms_norm, rope, silu};
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::execution_plan::DecodeExecutionPlan;
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    apply_attention_norm_single, apply_optional_attention_qk_norm_single, apply_output_norm_single,
    cache_attention_qk_norm_keys, cache_output_head_keys, encode_dequant_matvec_with_config,
    write_normalized_single_logits_with_breakdown,
};
use crate::model::weights::WeightStore;

use rayon::prelude::*;

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

/// Qwen3 MoE forward pass implementation.
#[derive(Debug)]
pub struct Qwen3MoeForward;

/// Scratch buffers for MoE expert computation (single token).
struct MoeSingleScratch {
    gate_buf: Vec<f32>,
    up_buf: Vec<f32>,
    down_buf: Vec<f32>,
    accum_buf: Vec<f32>,
    router_logits: Vec<f32>,
}

/// Scratch buffers for CPU batch path.
struct Qwen3MoeBatchScratch {
    hidden: Vec<f32>,
    norm_buf: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    attn_out: Vec<f32>,
    proj_buf: Vec<f32>,
    moe_norm_buf: Vec<f32>,
    moe_accum_buf: Vec<f32>,
    moe_scratch: MoeSingleScratch,
}

impl Qwen3MoeBatchScratch {
    fn new(config: &ModelConfig, chunk_len: usize) -> Self {
        let dim = config.embedding_dim as usize;
        let n_heads = config.n_heads as usize;
        let head_dim = config.head_dim as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let n_expert = config.n_expert.unwrap_or(0) as usize;
        let expert_inter_dim = config.expert_intermediate_dim.unwrap_or(0) as usize;
        let hidden_len = chunk_len * dim;

        Self {
            hidden: vec![0.0; hidden_len],
            norm_buf: vec![0.0; hidden_len],
            q_buf: vec![0.0; chunk_len * q_dim],
            k_buf: vec![0.0; chunk_len * kv_dim],
            v_buf: vec![0.0; chunk_len * kv_dim],
            attn_out: vec![0.0; chunk_len * q_dim],
            proj_buf: vec![0.0; hidden_len],
            moe_norm_buf: vec![0.0; hidden_len],
            moe_accum_buf: vec![0.0; hidden_len],
            moe_scratch: MoeSingleScratch {
                gate_buf: vec![0.0; expert_inter_dim],
                up_buf: vec![0.0; expert_inter_dim],
                down_buf: vec![0.0; dim],
                accum_buf: vec![0.0; dim],
                router_logits: vec![0.0; n_expert],
            },
        }
    }
}

impl Qwen3MoeForward {
    const PARALLEL_BATCH_MIN_TOKENS: usize = 64;
    const PARALLEL_FLOAT_CHUNK: usize = 16 * 1024;

    fn assert_finite_if_enabled(
        label: &str,
        values: &[f32],
        layer: usize,
        position: usize,
    ) -> anyhow::Result<()> {
        if !crate::model::shared::env_flag_enabled("AX_QWEN3MOE_ASSERT_FINITE") {
            return Ok(());
        }
        anyhow::ensure!(
            values.iter().all(|value| value.is_finite()),
            "qwen3moe non-finite values at {label} (layer={layer}, position={position})"
        );
        Ok(())
    }

    fn parallel_elementwise_add(dst: &mut [f32], src: &[f32]) {
        if dst.len() >= Self::PARALLEL_FLOAT_CHUNK {
            dst.par_chunks_mut(Self::PARALLEL_FLOAT_CHUNK)
                .zip(src.par_chunks(Self::PARALLEL_FLOAT_CHUNK))
                .for_each(|(dst_chunk, src_chunk)| {
                    silu::elementwise_add(dst_chunk, src_chunk);
                });
        } else {
            silu::elementwise_add(dst, src);
        }
    }

    /// Apply the MoE FFN block for a single token on CPU.
    ///
    /// Sequence: RMSNorm → router → top-k softmax → per-expert gate/up/SiLU/down → weighted accum → residual.
    #[allow(clippy::too_many_arguments)]
    fn apply_moe_ffn_single(
        backend: &dyn Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        norm_buf: &mut [f32],
        moe_scratch: &mut MoeSingleScratch,
        dim: usize,
        n_expert: usize,
        n_expert_used: usize,
        expert_inter_dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<()> {
        // 1. FFN norm: hidden → norm_buf
        let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
        rms_norm::rms_norm_out(hidden, ffn_norm_w, norm_buf, rms_norm_eps);

        // 2. Router: norm_buf → router_logits (dim → n_expert)
        let (router_raw, router_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_gate_inp.weight"))?;
        let router_logits = &mut moe_scratch.router_logits[..n_expert];
        backend.dequant_matmul(
            router_raw,
            router_dtype,
            norm_buf,
            router_logits,
            n_expert,
            1,
            dim,
        );

        // 3. Top-k softmax selection with re-normalization
        let (top_indices, top_weights) =
            crate::model::moe_utils::top_k_softmax(router_logits, n_expert_used);

        // 4. Per-expert computation
        let (gate_exps_raw, gate_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_gate_exps.weight"))?;
        let (up_exps_raw, up_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_up_exps.weight"))?;
        let (down_exps_raw, down_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_down_exps.weight"))?;

        let gate_stride =
            crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
        let up_stride =
            crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

        let cpu = crate::backend::cpu::CpuBackend;

        let accum = &mut moe_scratch.accum_buf[..dim];
        accum.fill(0.0);

        for (i, &expert_idx) in top_indices.iter().enumerate() {
            let weight = top_weights[i];

            let expert_gate = crate::model::moe_utils::expert_quant_slice(
                gate_exps_raw,
                gate_stride,
                expert_idx,
                &format!("{prefix}.ffn_gate_exps.weight"),
            )?;
            let gate_buf = &mut moe_scratch.gate_buf[..expert_inter_dim];
            cpu.dequant_matmul(
                expert_gate,
                gate_dtype,
                norm_buf,
                gate_buf,
                expert_inter_dim,
                1,
                dim,
            );

            let expert_up = crate::model::moe_utils::expert_quant_slice(
                up_exps_raw,
                up_stride,
                expert_idx,
                &format!("{prefix}.ffn_up_exps.weight"),
            )?;
            let up_buf = &mut moe_scratch.up_buf[..expert_inter_dim];
            cpu.dequant_matmul(
                expert_up,
                up_dtype,
                norm_buf,
                up_buf,
                expert_inter_dim,
                1,
                dim,
            );

            silu::silu_elementwise_mul(gate_buf, up_buf);

            let expert_down = crate::model::moe_utils::expert_quant_slice(
                down_exps_raw,
                down_stride,
                expert_idx,
                &format!("{prefix}.ffn_down_exps.weight"),
            )?;
            let down_buf = &mut moe_scratch.down_buf[..dim];
            cpu.dequant_matmul(
                expert_down,
                down_dtype,
                gate_buf,
                down_buf,
                dim,
                1,
                expert_inter_dim,
            );

            for (a, &val) in accum.iter_mut().zip(down_buf.iter()) {
                *a += weight * val;
            }
        }

        // 5. Residual add
        silu::elementwise_add(hidden, accum);
        Ok(())
    }

    /// Apply the MoE FFN block for a batch of tokens on CPU.
    #[allow(clippy::too_many_arguments)]
    fn apply_moe_ffn_batch(
        backend: &dyn Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        moe_norm_buf: &mut [f32],
        moe_accum_buf: &mut [f32],
        moe_scratch: &mut MoeSingleScratch,
        n_tokens: usize,
        dim: usize,
        n_expert: usize,
        n_expert_used: usize,
        expert_inter_dim: usize,
        rms_norm_eps: f32,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        // 1. FFN norm: per-token RMSNorm
        let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
        let hidden_len = n_tokens * dim;
        timed!(ops, norm, {
            if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
                hidden[..hidden_len]
                    .par_chunks(dim)
                    .zip(moe_norm_buf[..hidden_len].par_chunks_mut(dim))
                    .for_each(|(h, n)| {
                        rms_norm::rms_norm_out(h, ffn_norm_w, n, rms_norm_eps);
                    });
            } else {
                for t in 0..n_tokens {
                    let start = t * dim;
                    rms_norm::rms_norm_out(
                        &hidden[start..start + dim],
                        ffn_norm_w,
                        &mut moe_norm_buf[start..start + dim],
                        rms_norm_eps,
                    );
                }
            }
        });

        // Preload expert weight metadata
        let (router_raw, router_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_gate_inp.weight"))?;
        let (gate_exps_raw, gate_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_gate_exps.weight"))?;
        let (up_exps_raw, up_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_up_exps.weight"))?;
        let (down_exps_raw, down_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_down_exps.weight"))?;

        let gate_stride =
            crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
        let up_stride =
            crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

        let cpu = crate::backend::cpu::CpuBackend;
        moe_accum_buf[..hidden_len].fill(0.0);

        // 2. Per-token MoE routing and computation
        for t in 0..n_tokens {
            let token_norm = &moe_norm_buf[t * dim..(t + 1) * dim];
            let token_accum = &mut moe_accum_buf[t * dim..(t + 1) * dim];

            // Router matmul
            let router_logits = &mut moe_scratch.router_logits[..n_expert];
            timed!(
                ops,
                matmul,
                backend.dequant_matmul(
                    router_raw,
                    router_dtype,
                    token_norm,
                    router_logits,
                    n_expert,
                    1,
                    dim,
                )
            );

            let (top_indices, top_weights) =
                crate::model::moe_utils::top_k_softmax(router_logits, n_expert_used);

            for (i, &expert_idx) in top_indices.iter().enumerate() {
                let weight = top_weights[i];

                let expert_gate = crate::model::moe_utils::expert_quant_slice(
                    gate_exps_raw,
                    gate_stride,
                    expert_idx,
                    &format!("{prefix}.ffn_gate_exps.weight"),
                )?;
                let gate_buf = &mut moe_scratch.gate_buf[..expert_inter_dim];
                timed!(
                    ops,
                    matmul,
                    cpu.dequant_matmul(
                        expert_gate,
                        gate_dtype,
                        token_norm,
                        gate_buf,
                        expert_inter_dim,
                        1,
                        dim,
                    )
                );

                let expert_up = crate::model::moe_utils::expert_quant_slice(
                    up_exps_raw,
                    up_stride,
                    expert_idx,
                    &format!("{prefix}.ffn_up_exps.weight"),
                )?;
                let up_buf = &mut moe_scratch.up_buf[..expert_inter_dim];
                timed!(
                    ops,
                    matmul,
                    cpu.dequant_matmul(
                        expert_up,
                        up_dtype,
                        token_norm,
                        up_buf,
                        expert_inter_dim,
                        1,
                        dim,
                    )
                );

                silu::silu_elementwise_mul(gate_buf, up_buf);

                let expert_down = crate::model::moe_utils::expert_quant_slice(
                    down_exps_raw,
                    down_stride,
                    expert_idx,
                    &format!("{prefix}.ffn_down_exps.weight"),
                )?;
                let down_buf = &mut moe_scratch.down_buf[..dim];
                timed!(
                    ops,
                    matmul,
                    cpu.dequant_matmul(
                        expert_down,
                        down_dtype,
                        gate_buf,
                        down_buf,
                        dim,
                        1,
                        expert_inter_dim,
                    )
                );

                for (a, &val) in token_accum.iter_mut().zip(down_buf.iter()) {
                    *a += weight * val;
                }
            }
        }

        // 3. Residual add
        Self::parallel_elementwise_add(hidden, &moe_accum_buf[..hidden_len]);
        Ok(())
    }
}

include!("core.rs");
include!("forward.rs");

#[cfg(test)]
mod tests;
