//! Qwen3 MoE (Mixture of Experts) forward pass.
//!
//! Reuses the Qwen3 dense attention path and replaces the FFN block with
//! router → top-k expert selection → per-expert FFN → weighted combine.
//!
//! GGUF tensor naming:
//! - Router:       `blk.{layer}.ffn_gate_inp.weight`   [n_expert, dim]
//! - Expert gate:  `blk.{layer}.ffn_gate_exps.weight`  [n_expert * intermediate_dim, dim]
//! - Expert up:    `blk.{layer}.ffn_up_exps.weight`    [n_expert * intermediate_dim, dim]
//! - Expert down:  `blk.{layer}.ffn_down_exps.weight`  [n_expert * dim, intermediate_dim]
//! - Shared gate:  `blk.{layer}.ffn_gate_shexp.weight` [intermediate_dim, dim]
//! - Shared up:    `blk.{layer}.ffn_up_shexp.weight`   [intermediate_dim, dim]
//! - Shared down:  `blk.{layer}.ffn_down_shexp.weight` [dim, intermediate_dim]

use crate::compute::{attention, rms_norm, rope, silu};
use crate::gguf::tensor::GgmlType;
use crate::kv::ModelKv;
use crate::model::config::ModelConfig;
use crate::model::execution_plan::{
    DecodeExecutionPlan, PrefillAttentionPlan, PrefillExecutionPlan, PrefillMode,
};
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    encode_dequant_batch, encode_dequant_matvec_pair_with_config,
    encode_dequant_matvec_with_config, encode_dequant_silu_down_matvec_with_config,
    gpu_decode_quant_supported, gpu_prefill_q5k_small_n_auto_eligible, gpu_prefill_uses_q5k,
    per_head_rms_norm,
};
use crate::model::weights::WeightStore;

#[cfg(target_os = "macos")]
use crate::backend::metal::MetalOps;

#[derive(Debug)]
pub struct Qwen3MoeForward;

/// Compute byte size of `n_elements` in the given quantization type.
pub(crate) fn expert_byte_stride(dtype: GgmlType, n_elements: usize) -> usize {
    let bs = dtype.block_size();
    // Quantized GGUF tensors still allocate a full trailing block for
    // partially filled experts, so the per-expert byte stride must round up.
    n_elements.div_ceil(bs) * dtype.bytes_per_block()
}

fn ensure_matching_moe_dtypes(
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
    role: &str,
) -> anyhow::Result<GgmlType> {
    anyhow::ensure!(
        gate_dtype == up_dtype && gate_dtype == down_dtype,
        "AX Metal {role} weights must use one quant dtype; got gate={gate_dtype:?}, up={up_dtype:?}, down={down_dtype:?}",
    );
    Ok(gate_dtype)
}

fn qwen3_moe_has_shared_expert(
    gate: Option<usize>,
    up: Option<usize>,
    down: Option<usize>,
) -> bool {
    gate.is_some() || up.is_some() || down.is_some()
}

fn qwen3_moe_gpu_resident_supported(
    router_dtype: GgmlType,
    expert_dtype: GgmlType,
    has_shared_expert: bool,
) -> bool {
    matches!(
        router_dtype,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
    ) && matches!(expert_dtype, GgmlType::Q4K)
        && !has_shared_expert
}

/// Softmax over all experts, then select top-k.
/// Matches llama.cpp: softmax(all logits) → argsort_top_k → extract weights.
pub(crate) fn top_k_softmax(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    if logits.is_empty() || k == 0 {
        return (Vec::new(), Vec::new());
    }

    // Step 1: Softmax over ALL expert logits
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs = if !max.is_finite() {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        let mut probs: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            probs.fill(1.0 / logits.len() as f32);
        }
        probs
    };

    // Step 2: Select top-k from softmax probabilities
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    indexed.truncate(k.min(indexed.len()));

    // Step 3: Re-normalize selected weights to sum to 1
    let sel_sum: f32 = indexed.iter().map(|x| x.1).sum();
    let weights: Vec<f32> = if sel_sum > 0.0 {
        indexed.iter().map(|x| x.1 / sel_sum).collect()
    } else {
        indexed.iter().map(|_| 1.0 / indexed.len() as f32).collect()
    };

    (indexed.iter().map(|x| x.0).collect(), weights)
}

impl ForwardPass for Qwen3MoeForward {
    fn forward_single(
        &self,
        ctx: &ForwardContext,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        _ops: Option<&mut crate::metrics::OpBreakdown>,
    ) -> anyhow::Result<()> {
        // GPU path: use Metal when available
        #[cfg(target_os = "macos")]
        if ctx.backend.use_gpu_decode()
            && let Some(metal_ops) = ctx.backend.metal_ops()
            && let Some(gpu_kv) = kv.as_gpu_mut()
            && gpu_decode_quant_supported(weights)
        {
            return self
                .forward_single_gpu(ctx, metal_ops, token_id, position, gpu_kv, weights, logits);
        }

        // CPU fallback
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let shared_inter_dim = cfg.intermediate_dim as usize;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(cfg.intermediate_dim) as usize;
        let eps = cfg.rms_norm_eps;
        let n_expert = cfg.n_expert.unwrap() as usize;
        let n_expert_used = cfg.n_expert_used.unwrap() as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        // Use the larger of shared/expert inter_dim for buffer allocation
        let max_inter_dim = shared_inter_dim.max(expert_inter_dim);

        let mut hidden = vec![0.0f32; dim];
        let mut norm_buf = vec![0.0f32; dim];
        let mut q_buf = vec![0.0f32; q_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; max_inter_dim];
        let mut up_buf = vec![0.0f32; max_inter_dim];
        let mut down_buf = vec![0.0f32; dim];
        let mut router_logits = vec![0.0f32; n_expert];
        let mut expert_accum = vec![0.0f32; dim];

        // Token embedding
        weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?;

        let rope_position = cfg.rope_scaling.scaled_position(position);
        let cpu_kv = kv.as_cpu_mut().expect("MoE CPU path requires CpuKv");

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            // ── Attention norm ──
            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            rms_norm::rms_norm_out(&hidden, attn_norm_w, &mut norm_buf, eps);

            // ── Q/K/V projections ──
            let (wq, wq_dt) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk, wk_dt) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv, wv_dt) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
            ctx.backend
                .dequant_matmul(wq, wq_dt, &norm_buf, &mut q_buf, q_dim, 1, dim);
            ctx.backend
                .dequant_matmul(wk, wk_dt, &norm_buf, &mut k_buf, kv_dim, 1, dim);
            ctx.backend
                .dequant_matmul(wv, wv_dt, &norm_buf, &mut v_buf, kv_dim, 1, dim);

            // ── QKV bias ──
            if weights.has(&format!("{prefix}.attn_q.bias")) {
                let qb = weights.dequantize(&format!("{prefix}.attn_q.bias"))?;
                silu::elementwise_add(&mut q_buf, &qb);
            }
            if weights.has(&format!("{prefix}.attn_k.bias")) {
                let kb = weights.dequantize(&format!("{prefix}.attn_k.bias"))?;
                silu::elementwise_add(&mut k_buf, &kb);
            }
            if weights.has(&format!("{prefix}.attn_v.bias")) {
                let vb = weights.dequantize(&format!("{prefix}.attn_v.bias"))?;
                silu::elementwise_add(&mut v_buf, &vb);
            }

            // ── Per-head QK norm ──
            if weights.has(&format!("{prefix}.attn_q_norm.weight")) {
                let qnw = weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?;
                let knw = weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?;
                per_head_rms_norm(&mut q_buf, n_heads, head_dim, qnw, eps);
                per_head_rms_norm(&mut k_buf, n_kv_heads, head_dim, knw, eps);
            }

            // ── RoPE ──
            rope::apply_rope_multi_head_scaled(
                &mut q_buf,
                &mut k_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                rope_position,
                cfg.rope_freq_base,
            );

            // ── KV cache + attention ──
            cpu_kv.append_and_advance(layer, &k_buf, &v_buf);
            let seq_len = cpu_kv.seq_len() + 1;
            attention::multi_head_attention(
                &q_buf,
                cpu_kv.k_slice_including_current(layer, seq_len),
                cpu_kv.v_slice_including_current(layer, seq_len),
                &mut attn_out,
                ctx.attn_params,
                seq_len,
            );

            // ── Output projection ──
            let (wo, wo_dt) = weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            ctx.backend
                .dequant_matmul(wo, wo_dt, &attn_out, &mut proj_buf, dim, 1, q_dim);

            // ── Residual + FFN norm ──
            for (h, p) in hidden.iter_mut().zip(proj_buf.iter()) {
                *h += p;
            }
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
            rms_norm::rms_norm_out(&hidden, ffn_norm_w, &mut norm_buf, eps);

            // ══════════════════════════════════════════════════════════════
            // MoE FFN
            // ══════════════════════════════════════════════════════════════

            // Router
            let (rw, rw_dt) = weights.raw_with_dtype(&format!("{prefix}.ffn_gate_inp.weight"))?;
            ctx.backend
                .dequant_matmul(rw, rw_dt, &norm_buf, &mut router_logits, n_expert, 1, dim);

            let (expert_ids, expert_weights) = top_k_softmax(&router_logits, n_expert_used);

            // Expert weight tensors: GGUF 3D layout [ne00, ne01, ne02]
            // where ne00=K (input dim), ne01=M (output dim), ne02=n_expert.
            // Elements stored row-major: data[m * ne00 * ne02 + k * ne02 + e]
            //
            // gate_exps: [dim, expert_ff, n_expert] → per expert e: W[expert_ff, dim]
            // up_exps:   [dim, expert_ff, n_expert] → per expert e: W[expert_ff, dim]
            // down_exps: [expert_ff, dim, n_expert] → per expert e: W[dim, expert_ff]

            let gate_exps_f32 = weights.dequantize(&format!("{prefix}.ffn_gate_exps.weight"))?;
            let up_exps_f32 = weights.dequantize(&format!("{prefix}.ffn_up_exps.weight"))?;
            let down_exps_f32 = weights.dequantize(&format!("{prefix}.ffn_down_exps.weight"))?;

            // GGUF 3D tensor [ne00, ne01, ne02] stores element [i0, i1, i2]
            // at index i2 * ne01 * ne00 + i1 * ne00 + i0. Expert `e`
            // occupies a contiguous block of ne01 * ne00 elements starting
            // at offset e * ne01 * ne00. (Verified against ggml's nb2 stride.)
            fn expert_slice(full: &[f32], ne00: usize, ne01: usize, eid: usize) -> &[f32] {
                let start = eid * ne01 * ne00;
                &full[start..start + ne01 * ne00]
            }

            expert_accum.fill(0.0);

            for (i, &eid) in expert_ids.iter().enumerate() {
                let ew = expert_weights[i];

                // gate_exps shape [ne00=dim, ne01=expert_ff, ne02=n_expert]
                // Expert slice: contiguous [expert_ff, dim] = [M, K]
                let expert_gate = expert_slice(&gate_exps_f32, dim, expert_inter_dim, eid);
                let expert_up = expert_slice(&up_exps_f32, dim, expert_inter_dim, eid);
                // down_exps shape [ne00=expert_ff, ne01=dim, ne02=n_expert]
                let expert_down = expert_slice(&down_exps_f32, expert_inter_dim, dim, eid);

                // matmul(A[M*K], B[K*N], C[M*N], M, N, K)
                ctx.backend.matmul(
                    expert_gate,
                    &norm_buf,
                    &mut gate_buf[..expert_inter_dim],
                    expert_inter_dim,
                    1,
                    dim,
                );
                ctx.backend.matmul(
                    expert_up,
                    &norm_buf,
                    &mut up_buf[..expert_inter_dim],
                    expert_inter_dim,
                    1,
                    dim,
                );
                silu::silu_elementwise_mul(
                    &mut gate_buf[..expert_inter_dim],
                    &up_buf[..expert_inter_dim],
                );
                ctx.backend.matmul(
                    expert_down,
                    &gate_buf[..expert_inter_dim],
                    &mut down_buf,
                    dim,
                    1,
                    expert_inter_dim,
                );

                for (a, d) in expert_accum.iter_mut().zip(down_buf.iter()) {
                    *a += ew * d;
                }
            }

            // Shared expert (if present)
            if weights.has(&format!("{prefix}.ffn_gate_shexp.weight")) {
                let (sg, sg_dt) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_gate_shexp.weight"))?;
                let (su, su_dt) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_up_shexp.weight"))?;
                let (sd, sd_dt) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_down_shexp.weight"))?;
                ctx.backend.dequant_matmul(
                    sg,
                    sg_dt,
                    &norm_buf,
                    &mut gate_buf,
                    shared_inter_dim,
                    1,
                    dim,
                );
                ctx.backend.dequant_matmul(
                    su,
                    su_dt,
                    &norm_buf,
                    &mut up_buf,
                    shared_inter_dim,
                    1,
                    dim,
                );
                silu::silu_elementwise_mul(
                    &mut gate_buf[..shared_inter_dim],
                    &up_buf[..shared_inter_dim],
                );
                ctx.backend.dequant_matmul(
                    sd,
                    sd_dt,
                    &gate_buf[..shared_inter_dim],
                    &mut down_buf,
                    dim,
                    1,
                    shared_inter_dim,
                );
                for (a, d) in expert_accum.iter_mut().zip(down_buf.iter()) {
                    *a += d;
                }
            }

            // Residual
            for (h, e) in hidden.iter_mut().zip(expert_accum.iter()) {
                *h += e;
            }
        }

        // ── Final norm + LM head ──
        let out_norm_w = weights.f32_slice("output_norm.weight")?;
        rms_norm::rms_norm_out(&hidden, out_norm_w, &mut norm_buf, eps);

        let lm_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (lm, lm_dt) = weights.raw_with_dtype(lm_name)?;
        ctx.backend.dequant_matmul(
            lm,
            lm_dt,
            &norm_buf,
            logits,
            cfg.vocab_size as usize,
            1,
            dim,
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
        let prefill_plan =
            PrefillExecutionPlan::for_forward_batch(ctx, kv, weights, token_ids.len(), false)?;

        if prefill_plan.mode == PrefillMode::GpuBatch
            && let Some(metal_ops) = ctx.backend.metal_ops()
            && let Some(gpu_kv) = kv.as_gpu_mut()
        {
            match self.forward_batch_gpu_moe(ctx, metal_ops, token_ids, gpu_kv, weights, logits) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!("MoE GPU batch prefill failed, falling back to serial: {e}");
                }
            }
        }

        // Fallback: serial forward_single calls
        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, None)?;
        }
        Ok(())
    }

    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.n_expert.is_some_and(|n| n > 0),
            "Qwen3MoeForward requires n_expert > 0"
        );
        anyhow::ensure!(
            config.n_expert_used.is_some_and(|n| n > 0),
            "Qwen3MoeForward requires n_expert_used > 0"
        );
        if let (Some(n_exp), Some(n_used)) = (config.n_expert, config.n_expert_used) {
            anyhow::ensure!(
                n_used <= n_exp,
                "n_expert_used ({n_used}) > n_expert ({n_exp})"
            );
        }
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "qwen3_moe"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU decode path
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
impl Qwen3MoeForward {
    #[allow(clippy::too_many_arguments)]
    fn run_moe_ffn_gpu_resident(
        metal_ops: &MetalOps,
        weights: &WeightStore,
        cfg: &ModelConfig,
        prefix: &str,
        hidden_gpu: &ax_engine_metal::MetalBuffer,
        n_tokens: usize,
        dim: usize,
        expert_inter_dim: usize,
        eps: f32,
    ) -> anyhow::Result<()> {
        let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
        let (router_raw, router_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_gate_inp.weight"))?;
        let (gate_raw, gate_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_gate_exps.weight"))?;
        let (up_raw, up_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up_exps.weight"))?;
        let (down_raw, down_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_down_exps.weight"))?;
        let expert_dtype = ensure_matching_moe_dtypes(gate_dtype, up_dtype, down_dtype, "expert")?;
        let has_shared_expert = weights.has(&format!("{prefix}.ffn_gate_shexp.weight"))
            || weights.has(&format!("{prefix}.ffn_up_shexp.weight"))
            || weights.has(&format!("{prefix}.ffn_down_shexp.weight"));
        anyhow::ensure!(
            qwen3_moe_gpu_resident_supported(router_dtype, expert_dtype, has_shared_expert),
            "qwen3 resident MoE fast path requires a supported router dtype, Q4_K experts, and no shared expert tensors",
        );

        let n_expert = cfg.n_expert.unwrap() as usize;
        let n_expert_used = cfg.n_expert_used.unwrap() as usize;
        let gate_stride = expert_byte_stride(expert_dtype, expert_inter_dim * dim);
        let up_stride = expert_byte_stride(expert_dtype, expert_inter_dim * dim);
        let down_stride = expert_byte_stride(expert_dtype, dim * expert_inter_dim);

        metal_ops.moe_ffn_gpu_resident(
            hidden_gpu,
            ffn_norm_w,
            router_raw,
            router_dtype,
            gate_raw,
            up_raw,
            down_raw,
            expert_dtype,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            gate_stride,
            up_stride,
            down_stride,
            eps,
        )
    }

    /// Build and cache expert weight MetalBuffers (first call only).
    fn build_cached_model_keys_moe(
        metal_ops: &MetalOps,
        weights: &WeightStore,
        cfg: &ModelConfig,
    ) -> anyhow::Result<()> {
        use crate::backend::metal::CachedLayerKeys;

        let n_layers = cfg.n_layers as usize;
        let n_expert = cfg.n_expert.unwrap() as usize;
        let dim = cfg.embedding_dim as usize;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(cfg.intermediate_dim) as usize;

        let mut layers = Vec::with_capacity(n_layers);

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            // Attention weights (same as Qwen3 dense)
            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            let attn_norm_key = metal_ops.ensure_f32_cached(attn_norm_w);

            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
            let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
            let (wo_raw, wo_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;

            let wq_key = metal_ops.ensure_quant_cached(wq_raw);
            let wk_key = metal_ops.ensure_quant_cached(wk_raw);
            let wv_key = metal_ops.ensure_quant_cached(wv_raw);
            let wo_key = metal_ops.ensure_quant_cached(wo_raw);

            // Biases
            let q_bias_key = if weights.has(&format!("{prefix}.attn_q.bias")) {
                Some(
                    metal_ops
                        .ensure_f32_cached(weights.f32_slice(&format!("{prefix}.attn_q.bias"))?),
                )
            } else {
                None
            };
            let k_bias_key = if weights.has(&format!("{prefix}.attn_k.bias")) {
                Some(
                    metal_ops
                        .ensure_f32_cached(weights.f32_slice(&format!("{prefix}.attn_k.bias"))?),
                )
            } else {
                None
            };
            let v_bias_key = if weights.has(&format!("{prefix}.attn_v.bias")) {
                Some(
                    metal_ops
                        .ensure_f32_cached(weights.f32_slice(&format!("{prefix}.attn_v.bias"))?),
                )
            } else {
                None
            };

            // QK norms
            let attn_q_norm_key =
                if weights.has(&format!("{prefix}.attn_q_norm.weight")) {
                    Some(metal_ops.ensure_f32_cached(
                        weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?,
                    ))
                } else {
                    None
                };
            let attn_k_norm_key =
                if weights.has(&format!("{prefix}.attn_k_norm.weight")) {
                    Some(metal_ops.ensure_f32_cached(
                        weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?,
                    ))
                } else {
                    None
                };

            // FFN norm
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);

            // Router weight
            let (router_raw, router_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_gate_inp.weight"))?;
            let router_key = metal_ops.ensure_quant_cached(router_raw);

            // Per-expert weights (slice stacked tensors)
            let (ge_raw, ge_dt) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_gate_exps.weight"))?;
            let (ue_raw, ue_dt) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_up_exps.weight"))?;
            let (de_raw, de_dt) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_down_exps.weight"))?;
            let expert_dtype = ensure_matching_moe_dtypes(ge_dt, ue_dt, de_dt, "MoE expert")?;

            let ge_stride = expert_byte_stride(ge_dt, expert_inter_dim * dim);
            let ue_stride = expert_byte_stride(ue_dt, expert_inter_dim * dim);
            let de_stride = expert_byte_stride(de_dt, dim * expert_inter_dim);

            let mut expert_gate_keys = Vec::with_capacity(n_expert);
            let mut expert_up_keys = Vec::with_capacity(n_expert);
            let mut expert_down_keys = Vec::with_capacity(n_expert);

            for eid in 0..n_expert {
                let ge_slice = &ge_raw[eid * ge_stride..(eid + 1) * ge_stride];
                let ue_slice = &ue_raw[eid * ue_stride..(eid + 1) * ue_stride];
                let de_slice = &de_raw[eid * de_stride..(eid + 1) * de_stride];
                expert_gate_keys.push(metal_ops.ensure_quant_cached(ge_slice));
                expert_up_keys.push(metal_ops.ensure_quant_cached(ue_slice));
                expert_down_keys.push(metal_ops.ensure_quant_cached(de_slice));
            }

            // Shared expert
            let (shared_gate_key, shared_up_key, shared_down_key, shared_dtype) =
                if weights.has(&format!("{prefix}.ffn_gate_shexp.weight")) {
                    let (sg, sg_dt) =
                        weights.raw_with_dtype(&format!("{prefix}.ffn_gate_shexp.weight"))?;
                    let (su, su_dt) =
                        weights.raw_with_dtype(&format!("{prefix}.ffn_up_shexp.weight"))?;
                    let (sd, sd_dt) =
                        weights.raw_with_dtype(&format!("{prefix}.ffn_down_shexp.weight"))?;
                    let shared_dtype =
                        ensure_matching_moe_dtypes(sg_dt, su_dt, sd_dt, "shared expert")?;
                    (
                        Some(metal_ops.ensure_quant_cached(sg)),
                        Some(metal_ops.ensure_quant_cached(su)),
                        Some(metal_ops.ensure_quant_cached(sd)),
                        Some(shared_dtype),
                    )
                } else {
                    (None, None, None, None)
                };

            layers.push(CachedLayerKeys {
                attn_norm: attn_norm_key,
                wq: wq_key,
                wq_dtype,
                wk: wk_key,
                wk_dtype,
                wv: wv_key,
                wv_dtype,
                wo: wo_key,
                wo_dtype,
                ffn_norm: ffn_norm_key,
                // Dense FFN fields unused for MoE (set to router as placeholder)
                wg: router_key,
                wg_dtype: router_dtype,
                wu: router_key,
                wu_dtype: router_dtype,
                wd: router_key,
                wd_dtype: router_dtype,
                attn_q_norm: attn_q_norm_key,
                attn_k_norm: attn_k_norm_key,
                post_attn_norm: None,
                post_ffn_norm: None,
                q_bias: q_bias_key,
                k_bias: k_bias_key,
                v_bias: v_bias_key,
                wo_bias: None,
                gate_bias: None,
                up_bias: None,
                down_bias: None,
                moe_router: Some(router_key),
                moe_router_dtype: Some(router_dtype),
                moe_expert_gate: Some(expert_gate_keys),
                moe_expert_up: Some(expert_up_keys),
                moe_expert_down: Some(expert_down_keys),
                moe_expert_dtype: Some(expert_dtype),
                moe_shared_gate: shared_gate_key,
                moe_shared_up: shared_up_key,
                moe_shared_down: shared_down_key,
                moe_shared_dtype: shared_dtype,
            });
        }

        // Output head
        let out_norm_w = weights.f32_slice("output_norm.weight")?;
        let out_norm_key = metal_ops.ensure_f32_cached(out_norm_w);
        let lm_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_name)?;
        let lm_key = metal_ops.ensure_quant_cached(lm_raw);

        metal_ops.set_cached_model_keys(crate::backend::metal::CachedModelKeys {
            layers,
            output_norm: out_norm_key,
            lm_head: lm_key,
            lm_head_dtype: lm_dtype,
        });

        tracing::info!(
            n_layers,
            n_expert,
            "MoE model keys cached ({} expert weight buffers per layer)",
            n_expert * 3
        );
        Ok(())
    }

    /// GPU decode path.
    ///
    /// Supported Q4_K expert layouts without shared experts use the resident
    /// MoE fast path after attention. Other layouts fall back to the older
    /// router-readback path.
    #[allow(clippy::too_many_arguments)]
    fn forward_single_gpu(
        &self,
        ctx: &ForwardContext,
        metal_ops: &MetalOps,
        token_id: u32,
        position: usize,
        gpu_kv: &mut crate::kv::GpuKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let shared_inter_dim = cfg.intermediate_dim as usize;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(cfg.intermediate_dim) as usize;
        let vocab_size = cfg.vocab_size as usize;
        let n_layers = cfg.n_layers as usize;
        let n_expert = cfg.n_expert.unwrap() as usize;
        let n_expert_used = cfg.n_expert_used.unwrap() as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let eps = cfg.rms_norm_eps;

        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, 1);
        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();

        let next_seq = gpu_kv.seq_len() + 1;
        gpu_kv.ensure_capacity(&metal_ops.device, next_seq)?;

        // Token embedding → hidden
        {
            let hidden_cpu = unsafe {
                std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim)
            };
            weights.dequantize_row("token_embd.weight", token_id as usize, hidden_cpu)?;
        }

        let rope_position = cfg.rope_scaling.scaled_position(position);

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_moe(metal_ops, weights, cfg)?;
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let cur_seq_len = gpu_kv.seq_len();
        let kv_offset = (cur_seq_len * kv_dim) as u32;

        let exec_plan = crate::model::execution_plan::DecodeExecutionPlan::qwen3_single_cb(
            metal_ops,
            gpu_kv,
            cfg.embedding_dim,
            cfg.head_dim,
            cur_seq_len + 1,
        );

        let weight_cache = metal_ops.lock_weight_cache();

        // Allocate router logits buffer on stack (small: n_expert floats)
        let mut router_logits_cpu = vec![0.0f32; n_expert];

        for layer in 0..n_layers {
            let lw = &cached.layers[layer];
            let prefix = format!("blk.{layer}");
            let use_gpu_resident_moe = qwen3_moe_gpu_resident_supported(
                lw.moe_router_dtype.unwrap(),
                lw.moe_expert_dtype.unwrap(),
                qwen3_moe_has_shared_expert(
                    lw.moe_shared_gate,
                    lw.moe_shared_up,
                    lw.moe_shared_down,
                ),
            );

            // ═══════════════════════════════════════════════════════════════
            // CB1: Attention + FFN norm + router matvec
            // ═══════════════════════════════════════════════════════════════
            metal_ops.device.execute_sync(|encoder| {
                let decode_barrier = |enc: &ax_engine_metal::MetalEncoder| {
                    if exec_plan.barriers
                        == crate::model::execution_plan::DecodeBarrierPlan::Explicit
                    {
                        ax_engine_metal::barrier_buffers(enc);
                    }
                };

                // Attention norm
                let norm_w = weight_cache.get(&lw.attn_norm).unwrap();
                metal_ops.elementwise.encode_rms_norm_out(
                    encoder,
                    &s.hidden,
                    norm_w,
                    &s.norm_buf,
                    dim as u32,
                    eps,
                );
                decode_barrier(encoder);

                // QKV matvec (separate for simplicity; fused QKV can be added later)
                let wq = weight_cache.get(&lw.wq).unwrap();
                let wk = weight_cache.get(&lw.wk).unwrap();
                let wv = weight_cache.get(&lw.wv).unwrap();
                encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wq,
                    &s.norm_buf,
                    &s.q_buf,
                    q_dim as u32,
                    dim as u32,
                    lw.wq_dtype,
                    exec_plan.dequant_dispatch,
                );
                encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wk,
                    &s.norm_buf,
                    &s.k_buf,
                    kv_dim as u32,
                    dim as u32,
                    lw.wk_dtype,
                    exec_plan.dequant_dispatch,
                );
                encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wv,
                    &s.norm_buf,
                    &s.v_buf,
                    kv_dim as u32,
                    dim as u32,
                    lw.wv_dtype,
                    exec_plan.dequant_dispatch,
                );
                decode_barrier(encoder);

                // Bias (if present)
                if let Some(qb_key) = lw.q_bias {
                    let qb = weight_cache.get(&qb_key).unwrap();
                    metal_ops.elementwise.encode_elementwise_add(
                        encoder,
                        &s.q_buf,
                        qb,
                        q_dim as u32,
                    );
                }
                if let Some(kb_key) = lw.k_bias {
                    let kb = weight_cache.get(&kb_key).unwrap();
                    metal_ops.elementwise.encode_elementwise_add(
                        encoder,
                        &s.k_buf,
                        kb,
                        kv_dim as u32,
                    );
                }
                if let Some(vb_key) = lw.v_bias {
                    let vb = weight_cache.get(&vb_key).unwrap();
                    metal_ops.elementwise.encode_elementwise_add(
                        encoder,
                        &s.v_buf,
                        vb,
                        kv_dim as u32,
                    );
                }
                decode_barrier(encoder);

                // Per-head QK norm
                if let (Some(qn_key), Some(kn_key)) = (lw.attn_q_norm, lw.attn_k_norm) {
                    let qnw = weight_cache.get(&qn_key).unwrap();
                    let knw = weight_cache.get(&kn_key).unwrap();
                    metal_ops.elementwise.encode_per_head_rms_norm(
                        encoder,
                        &s.q_buf,
                        qnw,
                        n_heads as u32,
                        head_dim as u32,
                        eps,
                    );
                    metal_ops.elementwise.encode_per_head_rms_norm(
                        encoder,
                        &s.k_buf,
                        knw,
                        n_kv_heads as u32,
                        head_dim as u32,
                        eps,
                    );
                    decode_barrier(encoder);
                }

                // RoPE
                metal_ops.elementwise.encode_rope(
                    encoder,
                    &s.q_buf,
                    &s.k_buf,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    rope_position,
                    cfg.rope_freq_base,
                );
                decode_barrier(encoder);

                // KV cache append
                metal_ops.elementwise.encode_kv_append(
                    encoder,
                    &s.k_buf,
                    gpu_kv.k_buffer(layer),
                    exec_plan.kv_f16,
                    kv_offset,
                    kv_dim as u32,
                );
                metal_ops.elementwise.encode_kv_append(
                    encoder,
                    &s.v_buf,
                    gpu_kv.v_buffer(layer),
                    exec_plan.kv_f16,
                    kv_offset,
                    kv_dim as u32,
                );
                decode_barrier(encoder);

                // Attention decode
                metal_ops
                    .attention
                    .encode_attention_decode_with_scratch_and_config(
                        encoder,
                        &s.q_buf,
                        gpu_kv.k_buffer(layer),
                        gpu_kv.v_buffer(layer),
                        &s.attn_out,
                        &s.splitk_partial_out,
                        &s.splitk_partial_lse,
                        exec_plan.kv_f16,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        0,
                        (cur_seq_len + 1) as u32,
                        exec_plan.attention_dispatch,
                    );
                decode_barrier(encoder);

                // Output projection
                let wo = weight_cache.get(&lw.wo).unwrap();
                encode_dequant_matvec_with_config(
                    metal_ops,
                    encoder,
                    wo,
                    &s.attn_out,
                    &s.proj_buf,
                    dim as u32,
                    q_dim as u32,
                    lw.wo_dtype,
                    exec_plan.dequant_dispatch,
                );
                decode_barrier(encoder);

                // Residual + FFN norm
                metal_ops.elementwise.encode_elementwise_add(
                    encoder,
                    &s.hidden,
                    &s.proj_buf,
                    dim as u32,
                );
                decode_barrier(encoder);
                if !use_gpu_resident_moe {
                    let ffn_nw = weight_cache.get(&lw.ffn_norm).unwrap();
                    metal_ops.elementwise.encode_rms_norm_out(
                        encoder,
                        &s.hidden,
                        ffn_nw,
                        &s.norm_buf,
                        dim as u32,
                        eps,
                    );
                    decode_barrier(encoder);

                    // Router matvec
                    let router_w = weight_cache.get(&lw.moe_router.unwrap()).unwrap();
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        router_w,
                        &s.norm_buf,
                        &s.down_buf,
                        n_expert as u32,
                        dim as u32,
                        lw.moe_router_dtype.unwrap(),
                        exec_plan.dequant_dispatch,
                    );
                }

                Ok(())
            })?;

            if use_gpu_resident_moe {
                Self::run_moe_ffn_gpu_resident(
                    metal_ops,
                    weights,
                    cfg,
                    &prefix,
                    &s.hidden,
                    1,
                    dim,
                    expert_inter_dim,
                    eps,
                )?;
                continue;
            }

            // ═══════════════════════════════════════════════════════════════
            // CPU: Read router logits, top-k selection
            // ═══════════════════════════════════════════════════════════════
            {
                let router_gpu = unsafe {
                    std::slice::from_raw_parts(
                        s.down_buf.contents().as_ptr() as *const f32,
                        n_expert,
                    )
                };
                router_logits_cpu.copy_from_slice(router_gpu);
            }
            let (expert_ids, expert_weights) = top_k_softmax(&router_logits_cpu, n_expert_used);

            // ═══════════════════════════════════════════════════════════════
            // CB2: Per-expert FFN (fused) + shared expert + residual
            // ═══════════════════════════════════════════════════════════════
            metal_ops.device.execute_sync(|encoder| {
                let decode_barrier = |enc: &ax_engine_metal::MetalEncoder| {
                    if exec_plan.barriers
                        == crate::model::execution_plan::DecodeBarrierPlan::Explicit
                    {
                        ax_engine_metal::barrier_buffers(enc);
                    }
                };

                let expert_gate_keys = lw.moe_expert_gate.as_ref().unwrap();
                let expert_up_keys = lw.moe_expert_up.as_ref().unwrap();
                let expert_down_keys = lw.moe_expert_down.as_ref().unwrap();
                let expert_dtype = lw.moe_expert_dtype.unwrap();

                // Zero the accumulator (reuse proj_buf as expert_accum)
                // Write zeros via CPU UMA
                unsafe {
                    let accum_ptr = s.proj_buf.contents().as_ptr() as *mut f32;
                    std::ptr::write_bytes(accum_ptr, 0, dim);
                }

                for (i, &eid) in expert_ids.iter().enumerate() {
                    let ew = expert_weights[i];
                    let eg = weight_cache.get(&expert_gate_keys[eid]).unwrap();
                    let eu = weight_cache.get(&expert_up_keys[eid]).unwrap();
                    let ed = weight_cache.get(&expert_down_keys[eid]).unwrap();

                    // Fused pair gate+up
                    if !encode_dequant_matvec_pair_with_config(
                        metal_ops,
                        encoder,
                        eg,
                        eu,
                        &s.norm_buf,
                        &s.gate_buf,
                        &s.up_buf,
                        expert_inter_dim as u32,
                        dim as u32,
                        expert_dtype,
                        expert_dtype,
                        exec_plan.dequant_dispatch,
                        exec_plan.use_pair_matvec,
                    ) {
                        encode_dequant_matvec_with_config(
                            metal_ops,
                            encoder,
                            eg,
                            &s.norm_buf,
                            &s.gate_buf,
                            expert_inter_dim as u32,
                            dim as u32,
                            expert_dtype,
                            exec_plan.dequant_dispatch,
                        );
                        encode_dequant_matvec_with_config(
                            metal_ops,
                            encoder,
                            eu,
                            &s.norm_buf,
                            &s.up_buf,
                            expert_inter_dim as u32,
                            dim as u32,
                            expert_dtype,
                            exec_plan.dequant_dispatch,
                        );
                    }
                    decode_barrier(encoder);

                    // Fused SiLU+down
                    if !encode_dequant_silu_down_matvec_with_config(
                        metal_ops,
                        encoder,
                        ed,
                        &s.gate_buf,
                        &s.up_buf,
                        &s.down_buf,
                        dim as u32,
                        expert_inter_dim as u32,
                        expert_dtype,
                        exec_plan.dequant_dispatch,
                        exec_plan.use_fused_silu_down,
                    ) {
                        metal_ops.elementwise.encode_silu_elementwise_mul(
                            encoder,
                            &s.gate_buf,
                            &s.up_buf,
                            expert_inter_dim as u32,
                        );
                        decode_barrier(encoder);
                        encode_dequant_matvec_with_config(
                            metal_ops,
                            encoder,
                            ed,
                            &s.gate_buf,
                            &s.down_buf,
                            dim as u32,
                            expert_inter_dim as u32,
                            expert_dtype,
                            exec_plan.dequant_dispatch,
                        );
                    }
                    decode_barrier(encoder);

                    // Weighted accumulate: proj_buf += ew * down_buf
                    metal_ops.elementwise.encode_elementwise_weighted_add(
                        encoder,
                        &s.proj_buf,
                        &s.down_buf,
                        ew,
                        dim as u32,
                    );
                    decode_barrier(encoder);
                }

                // Shared expert (if present)
                if let (Some(sg_key), Some(su_key), Some(sd_key)) =
                    (lw.moe_shared_gate, lw.moe_shared_up, lw.moe_shared_down)
                {
                    let sg = weight_cache.get(&sg_key).unwrap();
                    let su = weight_cache.get(&su_key).unwrap();
                    let sd = weight_cache.get(&sd_key).unwrap();
                    let sd_dtype = lw.moe_shared_dtype.unwrap();

                    if !encode_dequant_matvec_pair_with_config(
                        metal_ops,
                        encoder,
                        sg,
                        su,
                        &s.norm_buf,
                        &s.gate_buf,
                        &s.up_buf,
                        shared_inter_dim as u32,
                        dim as u32,
                        sd_dtype,
                        sd_dtype,
                        exec_plan.dequant_dispatch,
                        exec_plan.use_pair_matvec,
                    ) {
                        encode_dequant_matvec_with_config(
                            metal_ops,
                            encoder,
                            sg,
                            &s.norm_buf,
                            &s.gate_buf,
                            shared_inter_dim as u32,
                            dim as u32,
                            sd_dtype,
                            exec_plan.dequant_dispatch,
                        );
                        encode_dequant_matvec_with_config(
                            metal_ops,
                            encoder,
                            su,
                            &s.norm_buf,
                            &s.up_buf,
                            shared_inter_dim as u32,
                            dim as u32,
                            sd_dtype,
                            exec_plan.dequant_dispatch,
                        );
                    }
                    decode_barrier(encoder);

                    if !encode_dequant_silu_down_matvec_with_config(
                        metal_ops,
                        encoder,
                        sd,
                        &s.gate_buf,
                        &s.up_buf,
                        &s.down_buf,
                        dim as u32,
                        shared_inter_dim as u32,
                        sd_dtype,
                        exec_plan.dequant_dispatch,
                        exec_plan.use_fused_silu_down,
                    ) {
                        metal_ops.elementwise.encode_silu_elementwise_mul(
                            encoder,
                            &s.gate_buf,
                            &s.up_buf,
                            shared_inter_dim as u32,
                        );
                        decode_barrier(encoder);
                        encode_dequant_matvec_with_config(
                            metal_ops,
                            encoder,
                            sd,
                            &s.gate_buf,
                            &s.down_buf,
                            dim as u32,
                            shared_inter_dim as u32,
                            sd_dtype,
                            exec_plan.dequant_dispatch,
                        );
                    }
                    decode_barrier(encoder);

                    // Shared expert accumulate (unweighted)
                    metal_ops.elementwise.encode_elementwise_add(
                        encoder,
                        &s.proj_buf,
                        &s.down_buf,
                        dim as u32,
                    );
                    decode_barrier(encoder);
                }

                // Final residual: hidden += expert_accum (proj_buf)
                metal_ops.elementwise.encode_elementwise_add(
                    encoder,
                    &s.hidden,
                    &s.proj_buf,
                    dim as u32,
                );

                Ok(())
            })?;
        }

        metal_ops.device.execute_sync(|encoder| {
            let out_norm = weight_cache.get(&cached.output_norm).unwrap();
            metal_ops.elementwise.encode_rms_norm_out(
                encoder,
                &s.hidden,
                out_norm,
                &s.norm_buf,
                dim as u32,
                eps,
            );
            if exec_plan.barriers == crate::model::execution_plan::DecodeBarrierPlan::Explicit {
                ax_engine_metal::barrier_buffers(encoder);
            }
            let lm_head = weight_cache.get(&cached.lm_head).unwrap();
            encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                lm_head,
                &s.norm_buf,
                &s.logits_buf,
                vocab_size as u32,
                dim as u32,
                cached.lm_head_dtype,
                exec_plan.dequant_dispatch,
            );
            Ok(())
        })?;

        gpu_kv.finalize_token();

        // Read logits from GPU
        let logits_gpu = unsafe {
            std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab_size)
        };
        logits[..vocab_size].copy_from_slice(logits_gpu);

        Ok(())
    }

    /// GPU batched prefill for MoE.
    ///
    /// Per-layer two command buffers:
    /// - CB1: batched attention + FFN norm + router matmul (all N tokens)
    /// - CPU: read router logits, per-token top-k softmax
    /// - CB2: per-token expert FFN + shared expert + residual
    ///
    /// Attention is fully batched (same throughput as dense Qwen3 prefill).
    /// Expert FFN is serial per-token (routing differs per token).
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_gpu_moe(
        &self,
        ctx: &ForwardContext,
        metal_ops: &MetalOps,
        token_ids: &[u32],
        gpu_kv: &mut crate::kv::GpuKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let cfg = ctx.config;
        let n_tokens = token_ids.len();
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let shared_inter_dim = cfg.intermediate_dim as usize;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(cfg.intermediate_dim) as usize;
        let vocab_size = cfg.vocab_size as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let eps = cfg.rms_norm_eps;
        let n_expert = cfg.n_expert.unwrap() as usize;
        let n_expert_used = cfg.n_expert_used.unwrap() as usize;

        metal_ops.init_scratches(cfg);
        metal_ops.init_batch_scratches(cfg, n_tokens);

        let scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_ref().unwrap();
        let batch_guard = metal_ops.batch_scratches();
        let bs = batch_guard.as_ref().unwrap();

        gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + n_tokens)?;

        // Embed all N tokens into batch hidden buffer
        {
            let batch_hidden_cpu = unsafe {
                std::slice::from_raw_parts_mut(
                    bs.hidden.contents().as_ptr() as *mut f32,
                    n_tokens * dim,
                )
            };
            for (i, &tid) in token_ids.iter().enumerate() {
                weights.dequantize_row(
                    "token_embd.weight",
                    tid as usize,
                    &mut batch_hidden_cpu[i * dim..(i + 1) * dim],
                )?;
            }
        }

        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_moe(metal_ops, weights, cfg)?;
        }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let base_seq_len = gpu_kv.seq_len();
        let weight_cache = metal_ops.lock_weight_cache();

        // Reuse Qwen3 dense prefill plan — attention architecture is identical
        let has_q5k = gpu_prefill_uses_q5k(weights);
        let q5k_auto = gpu_prefill_q5k_small_n_auto_eligible(weights);
        let prefill_plan = DecodeExecutionPlan::qwen3_prefill(
            metal_ops,
            gpu_kv,
            base_seq_len,
            n_tokens as u32,
            cfg.head_dim,
            cfg.sliding_window_size.unwrap_or(0),
            has_q5k,
            q5k_auto,
        );

        // Decode plan for expert FFN dispatch config
        let exec_plan = DecodeExecutionPlan::qwen3_single_cb(
            metal_ops,
            gpu_kv,
            cfg.embedding_dim,
            cfg.head_dim,
            base_seq_len + n_tokens,
        );

        // Pre-allocated zero buffer for GPU-side accumulator clearing
        let zero_buf = ax_engine_metal::MetalBuffer::new(
            metal_ops.device.device(),
            dim * std::mem::size_of::<f32>(),
        )?;
        unsafe {
            std::ptr::write_bytes(
                zero_buf.contents().as_ptr() as *mut u8,
                0,
                dim * std::mem::size_of::<f32>(),
            );
        }

        let mut router_logits_cpu = vec![0.0f32; n_tokens * n_expert];

        for layer in 0..n_layers {
            let lw = &cached.layers[layer];
            let prefix = format!("blk.{layer}");
            let use_gpu_resident_moe = qwen3_moe_gpu_resident_supported(
                lw.moe_router_dtype.unwrap(),
                lw.moe_expert_dtype.unwrap(),
                qwen3_moe_has_shared_expert(
                    lw.moe_shared_gate,
                    lw.moe_shared_up,
                    lw.moe_shared_down,
                ),
            );
            let (rope_start, rope_step) = cfg.rope_scaling.scaled_start_step(base_seq_len);
            let cache_offset = (base_seq_len * kv_dim) as u32;

            // ══════════════════════════════════════════════════════════
            // CB1: Batched attention + residual, with router prep only on fallback
            // ══════════════════════════════════════════════════════════
            metal_ops.device.execute_sync(|encoder| {
                let nt = n_tokens as u32;

                // Attention norm (layer 0 only; subsequent layers pre-computed
                // at end of previous CB2)
                if layer == 0 {
                    let norm_w = weight_cache.get(&lw.attn_norm).unwrap();
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        norm_w,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        eps,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                }

                // Batched QKV matmul (separate Q, K, V)
                let wq = weight_cache.get(&lw.wq).unwrap();
                let wk = weight_cache.get(&lw.wk).unwrap();
                let wv = weight_cache.get(&lw.wv).unwrap();
                encode_dequant_batch(
                    &metal_ops.dequant,
                    &metal_ops.elementwise,
                    encoder,
                    wq,
                    &bs.norm_buf,
                    &bs.q_buf,
                    &bs.matmul_in_f16,
                    q_dim as u32,
                    nt,
                    dim as u32,
                    lw.wq_dtype,
                    false,
                    prefill_plan.use_batch_simd,
                    prefill_plan.q5k_prefill_small_n,
                );
                encode_dequant_batch(
                    &metal_ops.dequant,
                    &metal_ops.elementwise,
                    encoder,
                    wk,
                    &bs.norm_buf,
                    &bs.k_buf,
                    &bs.matmul_in_f16,
                    kv_dim as u32,
                    nt,
                    dim as u32,
                    lw.wk_dtype,
                    false,
                    prefill_plan.use_batch_simd,
                    prefill_plan.q5k_prefill_small_n,
                );
                encode_dequant_batch(
                    &metal_ops.dequant,
                    &metal_ops.elementwise,
                    encoder,
                    wv,
                    &bs.norm_buf,
                    &bs.v_buf,
                    &bs.matmul_in_f16,
                    kv_dim as u32,
                    nt,
                    dim as u32,
                    lw.wv_dtype,
                    false,
                    prefill_plan.use_batch_simd,
                    prefill_plan.q5k_prefill_small_n,
                );
                ax_engine_metal::barrier_buffers(encoder);

                // QKV bias (optional)
                if let Some(qb_key) = lw.q_bias {
                    let qb = weight_cache.get(&qb_key).unwrap();
                    let kb = weight_cache.get(&lw.k_bias.unwrap()).unwrap();
                    let vb = weight_cache.get(&lw.v_bias.unwrap()).unwrap();
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.q_buf,
                        qb,
                        q_dim as u32,
                        nt,
                    );
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.k_buf,
                        kb,
                        kv_dim as u32,
                        nt,
                    );
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.v_buf,
                        vb,
                        kv_dim as u32,
                        nt,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                }

                // QK norm + RoPE, or just RoPE
                if let Some(q_norm_key) = lw.attn_q_norm {
                    let q_nw = weight_cache.get(&q_norm_key).unwrap();
                    let k_nw = weight_cache.get(&lw.attn_k_norm.unwrap()).unwrap();
                    metal_ops.elementwise.encode_qk_norm_rope_batch(
                        encoder,
                        &bs.q_buf,
                        &bs.k_buf,
                        q_nw,
                        k_nw,
                        nt,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        eps,
                        rope_start,
                        rope_step,
                        cfg.rope_freq_base,
                    );
                } else {
                    metal_ops.elementwise.encode_rope_batch(
                        encoder,
                        &bs.q_buf,
                        &bs.k_buf,
                        nt,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        rope_start,
                        rope_step,
                        cfg.rope_freq_base,
                    );
                }
                ax_engine_metal::barrier_buffers(encoder);

                // KV append (batched)
                let kv_k = gpu_kv.k_buffer(layer);
                let kv_v = gpu_kv.v_buffer(layer);
                metal_ops.elementwise.encode_kv_append_batch(
                    encoder,
                    &bs.k_buf,
                    kv_k,
                    prefill_plan.kv_f16,
                    cache_offset,
                    kv_dim as u32,
                    kv_dim as u32,
                    nt,
                );
                metal_ops.elementwise.encode_kv_append_batch(
                    encoder,
                    &bs.v_buf,
                    kv_v,
                    prefill_plan.kv_f16,
                    cache_offset,
                    kv_dim as u32,
                    kv_dim as u32,
                    nt,
                );
                ax_engine_metal::barrier_buffers(encoder);

                // Batched attention
                if prefill_plan.attention == PrefillAttentionPlan::Cached {
                    metal_ops
                        .attention
                        .encode_attention_prefill_cached_with_config(
                            encoder,
                            &bs.q_buf,
                            gpu_kv.k_buffer(layer),
                            gpu_kv.v_buffer(layer),
                            &bs.attn_out,
                            prefill_plan.kv_f16,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            base_seq_len as u32,
                            prefill_plan.attention_sliding_window,
                            prefill_plan.attention_dispatch,
                        );
                } else {
                    metal_ops.attention.encode_attention_prefill_with_config(
                        encoder,
                        &bs.q_buf,
                        &bs.k_buf,
                        &bs.v_buf,
                        &bs.attn_out,
                        nt,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        prefill_plan.attention_dispatch,
                    );
                }
                ax_engine_metal::barrier_buffers(encoder);

                // Output projection (batched)
                let wo = weight_cache.get(&lw.wo).unwrap();
                encode_dequant_batch(
                    &metal_ops.dequant,
                    &metal_ops.elementwise,
                    encoder,
                    wo,
                    &bs.attn_out,
                    &bs.proj_buf,
                    &bs.matmul_in_f16,
                    dim as u32,
                    nt,
                    q_dim as u32,
                    lw.wo_dtype,
                    prefill_plan.use_f16_batch_io,
                    prefill_plan.use_batch_simd,
                    prefill_plan.q5k_prefill_small_n,
                );
                ax_engine_metal::barrier_buffers(encoder);

                if use_gpu_resident_moe {
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                    );
                } else {
                    // Residual + FFN norm: hidden += proj_buf; norm_buf = RMSNorm(hidden)
                    let ffn_nw = weight_cache.get(&lw.ffn_norm).unwrap();
                    metal_ops
                        .elementwise
                        .encode_residual_add_rms_norm_out_batch(
                            encoder,
                            &bs.hidden,
                            &bs.proj_buf,
                            ffn_nw,
                            &bs.norm_buf,
                            dim as u32,
                            nt,
                            eps,
                        );
                    ax_engine_metal::barrier_buffers(encoder);

                    // Router matmul: [n_tokens × n_expert] → reuse bs.proj_buf
                    let router_w = weight_cache.get(&lw.moe_router.unwrap()).unwrap();
                    encode_dequant_batch(
                        &metal_ops.dequant,
                        &metal_ops.elementwise,
                        encoder,
                        router_w,
                        &bs.norm_buf,
                        &bs.proj_buf,
                        &bs.matmul_in_f16,
                        n_expert as u32,
                        nt,
                        dim as u32,
                        lw.moe_router_dtype.unwrap(),
                        false,
                        prefill_plan.use_batch_simd,
                        false,
                    );
                }

                Ok(())
            })?;

            if use_gpu_resident_moe {
                Self::run_moe_ffn_gpu_resident(
                    metal_ops,
                    weights,
                    cfg,
                    &prefix,
                    &bs.hidden,
                    n_tokens,
                    dim,
                    expert_inter_dim,
                    eps,
                )?;

                if layer + 1 < n_layers {
                    let next_norm_w = weight_cache
                        .get(&cached.layers[layer + 1].attn_norm)
                        .unwrap();
                    metal_ops.device.execute_sync(|encoder| {
                        metal_ops.elementwise.encode_rms_norm_out_batch(
                            encoder,
                            &bs.hidden,
                            next_norm_w,
                            &bs.norm_buf,
                            dim as u32,
                            n_tokens as u32,
                            eps,
                        );
                        Ok(())
                    })?;
                }

                continue;
            }

            // ─── CPU: Read router logits + per-token top-k ───
            {
                let router_gpu = unsafe {
                    std::slice::from_raw_parts(
                        bs.proj_buf.contents().as_ptr() as *const f32,
                        n_tokens * n_expert,
                    )
                };
                router_logits_cpu.copy_from_slice(router_gpu);
            }
            let routing: Vec<(Vec<usize>, Vec<f32>)> = (0..n_tokens)
                .map(|t| {
                    top_k_softmax(
                        &router_logits_cpu[t * n_expert..(t + 1) * n_expert],
                        n_expert_used,
                    )
                })
                .collect();

            // ══════════════════════════════════════════════════════════
            // CB2: Fallback per-token expert FFN + shared expert + residual
            // ══════════════════════════════════════════════════════════
            metal_ops.device.execute_sync(|encoder| {
                let expert_gate_keys = lw.moe_expert_gate.as_ref().unwrap();
                let expert_up_keys = lw.moe_expert_up.as_ref().unwrap();
                let expert_down_keys = lw.moe_expert_down.as_ref().unwrap();
                let expert_dtype = lw.moe_expert_dtype.unwrap();

                for (t, (expert_ids, expert_weights)) in routing.iter().enumerate() {
                    let tok_byte_off = t * dim * std::mem::size_of::<f32>();

                    // Copy FFN input for this token: batch norm_buf[t] → s.norm_buf
                    metal_ops.elementwise.encode_buffer_copy(
                        encoder,
                        &bs.norm_buf,
                        tok_byte_off,
                        &s.norm_buf,
                        0,
                        dim as u32,
                    );
                    ax_engine_metal::barrier_buffers(encoder);

                    // Zero expert accumulator via GPU copy from pre-zeroed buffer
                    metal_ops.elementwise.encode_buffer_copy(
                        encoder,
                        &zero_buf,
                        0,
                        &s.proj_buf,
                        0,
                        dim as u32,
                    );
                    ax_engine_metal::barrier_buffers(encoder);

                    for (i, &eid) in expert_ids.iter().enumerate() {
                        let ew = expert_weights[i];
                        let eg = weight_cache.get(&expert_gate_keys[eid]).unwrap();
                        let eu = weight_cache.get(&expert_up_keys[eid]).unwrap();
                        let ed = weight_cache.get(&expert_down_keys[eid]).unwrap();

                        // Fused pair gate+up
                        if !encode_dequant_matvec_pair_with_config(
                            metal_ops,
                            encoder,
                            eg,
                            eu,
                            &s.norm_buf,
                            &s.gate_buf,
                            &s.up_buf,
                            expert_inter_dim as u32,
                            dim as u32,
                            expert_dtype,
                            expert_dtype,
                            exec_plan.dequant_dispatch,
                            exec_plan.use_pair_matvec,
                        ) {
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                eg,
                                &s.norm_buf,
                                &s.gate_buf,
                                expert_inter_dim as u32,
                                dim as u32,
                                expert_dtype,
                                exec_plan.dequant_dispatch,
                            );
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                eu,
                                &s.norm_buf,
                                &s.up_buf,
                                expert_inter_dim as u32,
                                dim as u32,
                                expert_dtype,
                                exec_plan.dequant_dispatch,
                            );
                        }
                        ax_engine_metal::barrier_buffers(encoder);

                        // Fused SiLU+down
                        if !encode_dequant_silu_down_matvec_with_config(
                            metal_ops,
                            encoder,
                            ed,
                            &s.gate_buf,
                            &s.up_buf,
                            &s.down_buf,
                            dim as u32,
                            expert_inter_dim as u32,
                            expert_dtype,
                            exec_plan.dequant_dispatch,
                            exec_plan.use_fused_silu_down,
                        ) {
                            metal_ops.elementwise.encode_silu_elementwise_mul(
                                encoder,
                                &s.gate_buf,
                                &s.up_buf,
                                expert_inter_dim as u32,
                            );
                            ax_engine_metal::barrier_buffers(encoder);
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                ed,
                                &s.gate_buf,
                                &s.down_buf,
                                dim as u32,
                                expert_inter_dim as u32,
                                expert_dtype,
                                exec_plan.dequant_dispatch,
                            );
                        }
                        ax_engine_metal::barrier_buffers(encoder);

                        // Weighted accumulate: proj_buf += ew * down_buf
                        metal_ops.elementwise.encode_elementwise_weighted_add(
                            encoder,
                            &s.proj_buf,
                            &s.down_buf,
                            ew,
                            dim as u32,
                        );
                        ax_engine_metal::barrier_buffers(encoder);
                    }

                    // Shared expert (if present)
                    if let (Some(sg_key), Some(su_key), Some(sd_key)) =
                        (lw.moe_shared_gate, lw.moe_shared_up, lw.moe_shared_down)
                    {
                        let sg = weight_cache.get(&sg_key).unwrap();
                        let su = weight_cache.get(&su_key).unwrap();
                        let sd = weight_cache.get(&sd_key).unwrap();
                        let sd_dtype = lw.moe_shared_dtype.unwrap();

                        if !encode_dequant_matvec_pair_with_config(
                            metal_ops,
                            encoder,
                            sg,
                            su,
                            &s.norm_buf,
                            &s.gate_buf,
                            &s.up_buf,
                            shared_inter_dim as u32,
                            dim as u32,
                            sd_dtype,
                            sd_dtype,
                            exec_plan.dequant_dispatch,
                            exec_plan.use_pair_matvec,
                        ) {
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                sg,
                                &s.norm_buf,
                                &s.gate_buf,
                                shared_inter_dim as u32,
                                dim as u32,
                                sd_dtype,
                                exec_plan.dequant_dispatch,
                            );
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                su,
                                &s.norm_buf,
                                &s.up_buf,
                                shared_inter_dim as u32,
                                dim as u32,
                                sd_dtype,
                                exec_plan.dequant_dispatch,
                            );
                        }
                        ax_engine_metal::barrier_buffers(encoder);

                        if !encode_dequant_silu_down_matvec_with_config(
                            metal_ops,
                            encoder,
                            sd,
                            &s.gate_buf,
                            &s.up_buf,
                            &s.down_buf,
                            dim as u32,
                            shared_inter_dim as u32,
                            sd_dtype,
                            exec_plan.dequant_dispatch,
                            exec_plan.use_fused_silu_down,
                        ) {
                            metal_ops.elementwise.encode_silu_elementwise_mul(
                                encoder,
                                &s.gate_buf,
                                &s.up_buf,
                                shared_inter_dim as u32,
                            );
                            ax_engine_metal::barrier_buffers(encoder);
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                sd,
                                &s.gate_buf,
                                &s.down_buf,
                                dim as u32,
                                shared_inter_dim as u32,
                                sd_dtype,
                                exec_plan.dequant_dispatch,
                            );
                        }
                        ax_engine_metal::barrier_buffers(encoder);

                        // Shared expert: unweighted add
                        metal_ops.elementwise.encode_elementwise_add(
                            encoder,
                            &s.proj_buf,
                            &s.down_buf,
                            dim as u32,
                        );
                        ax_engine_metal::barrier_buffers(encoder);
                    }

                    // Residual: hidden[t] += expert output (proj_buf)
                    metal_ops.elementwise.encode_buffer_copy(
                        encoder,
                        &bs.hidden,
                        tok_byte_off,
                        &s.hidden,
                        0,
                        dim as u32,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                    metal_ops.elementwise.encode_elementwise_add(
                        encoder,
                        &s.hidden,
                        &s.proj_buf,
                        dim as u32,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                    metal_ops.elementwise.encode_buffer_copy(
                        encoder,
                        &s.hidden,
                        0,
                        &bs.hidden,
                        (t * dim) as u32,
                        dim as u32,
                    );
                    ax_engine_metal::barrier_buffers(encoder);
                }

                // Pre-compute next layer's attention norm
                if layer + 1 < n_layers {
                    let next_norm_w = weight_cache
                        .get(&cached.layers[layer + 1].attn_norm)
                        .unwrap();
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        next_norm_w,
                        &bs.norm_buf,
                        dim as u32,
                        n_tokens as u32,
                        eps,
                    );
                }

                Ok(())
            })?;
        }

        // ══════════════════════════════════════════════════════════
        // Output head: last token norm + LM head matmul
        // ══════════════════════════════════════════════════════════
        metal_ops.device.execute_sync(|encoder| {
            let last_off = (n_tokens - 1) * dim * std::mem::size_of::<f32>();
            metal_ops
                .elementwise
                .encode_buffer_copy(encoder, &bs.hidden, last_off, &s.hidden, 0, dim as u32);
            ax_engine_metal::barrier_buffers(encoder);

            let fnw = weight_cache.get(&cached.output_norm).unwrap();
            metal_ops.elementwise.encode_rms_norm_out(
                encoder,
                &s.hidden,
                fnw,
                &s.norm_buf,
                dim as u32,
                eps,
            );
            ax_engine_metal::barrier_buffers(encoder);

            let lm = weight_cache.get(&cached.lm_head).unwrap();
            encode_dequant_matvec_with_config(
                metal_ops,
                encoder,
                lm,
                &s.norm_buf,
                &s.logits_buf,
                vocab_size as u32,
                dim as u32,
                cached.lm_head_dtype,
                exec_plan.dequant_dispatch,
            );

            Ok(())
        })?;

        gpu_kv.finalize_batch(n_tokens);

        let logits_gpu = unsafe {
            std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab_size)
        };
        logits[..vocab_size].copy_from_slice(logits_gpu);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── top_k_softmax tests ──────────────────────────────────────────

    #[test]
    fn test_ensure_matching_moe_dtypes_accepts_uniform_quant_family() {
        assert_eq!(
            ensure_matching_moe_dtypes(GgmlType::Q4K, GgmlType::Q4K, GgmlType::Q4K, "expert")
                .unwrap(),
            GgmlType::Q4K
        );
    }

    #[test]
    fn test_ensure_matching_moe_dtypes_rejects_mixed_quant_families() {
        let err = ensure_matching_moe_dtypes(GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q4K, "expert")
            .expect_err("mixed MoE dtypes should be rejected");
        assert!(err.to_string().contains("gate=Q4K"));
        assert!(err.to_string().contains("up=Q5K"));
    }

    #[test]
    fn test_qwen3_moe_gpu_resident_supported_requires_q4k_experts_and_no_shared_expert() {
        assert!(qwen3_moe_gpu_resident_supported(
            GgmlType::Q4K,
            GgmlType::Q4K,
            false,
        ));
        assert!(qwen3_moe_gpu_resident_supported(
            GgmlType::Q8_0,
            GgmlType::Q4K,
            false,
        ));
        assert!(!qwen3_moe_gpu_resident_supported(
            GgmlType::Q4K,
            GgmlType::Q5K,
            false,
        ));
        assert!(!qwen3_moe_gpu_resident_supported(
            GgmlType::Q4K,
            GgmlType::Q4K,
            true,
        ));
    }

    #[test]
    fn test_top_k_selects_highest_values() {
        let logits = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 0.0];
        let (ids, _weights) = top_k_softmax(&logits, 2);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 3); // 7.0
        assert_eq!(ids[1], 5); // 6.0
    }

    #[test]
    fn test_top_k_weights_sum_to_one() {
        let logits = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 0.0];
        let (_ids, weights) = top_k_softmax(&logits, 3);
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights sum to {sum}, expected ~1.0"
        );
    }

    #[test]
    fn test_top_k_with_k_equals_n() {
        let logits = vec![3.0, 1.0, 2.0];
        let (ids, weights) = top_k_softmax(&logits, 3);
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], 0); // 3.0 is highest
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_single_expert() {
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let (ids, weights) = top_k_softmax(&logits, 1);
        assert_eq!(ids, vec![0]);
        assert!(
            (weights[0] - 1.0).abs() < 1e-5,
            "single expert weight should be ~1.0"
        );
    }

    #[test]
    fn test_top_k_equal_logits() {
        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let (ids, weights) = top_k_softmax(&logits, 2);
        assert_eq!(ids, vec![0, 1]);
        // Equal logits → equal softmax weights
        assert!((weights[0] - weights[1]).abs() < 1e-5);
        assert!((weights[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_negative_logits() {
        let logits = vec![-1.0, -5.0, -3.0, -0.5];
        let (ids, _weights) = top_k_softmax(&logits, 2);
        assert_eq!(ids[0], 3); // -0.5 is highest
        assert_eq!(ids[1], 0); // -1.0 is second
    }

    #[test]
    fn test_top_k_zero_returns_empty() {
        let logits = vec![1.0, 2.0, 3.0];
        let (ids, weights) = top_k_softmax(&logits, 0);
        assert!(ids.is_empty());
        assert!(weights.is_empty());
    }

    #[test]
    fn test_top_k_all_non_finite_logits_is_deterministic_and_normalized() {
        let logits = vec![f32::NEG_INFINITY; 4];
        let (ids, weights) = top_k_softmax(&logits, 2);
        assert_eq!(ids, vec![0, 1]);
        assert!((weights[0] - 0.5).abs() < 1e-5);
        assert!((weights[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_k_larger_than_logits_len_still_normalizes_weights() {
        let logits = vec![f32::NEG_INFINITY; 2];
        let (ids, weights) = top_k_softmax(&logits, 4);
        assert_eq!(ids, vec![0, 1]);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights sum to {sum}");
    }

    // ── expert_byte_stride tests ─────────────────────────────────────

    #[test]
    fn test_expert_byte_stride_q4k() {
        // Q4_K: block_size=256, bytes_per_block=144
        // For inter_dim=14336, dim=4096: n_elements = 14336*4096 = 58720256
        // n_blocks = 58720256 / 256 = 229376
        // stride = 229376 * 144 = 33030144
        let stride = expert_byte_stride(GgmlType::Q4K, 14336 * 4096);
        assert_eq!(stride, 229376 * 144);
    }

    #[test]
    fn test_expert_byte_stride_q8_0() {
        // Q8_0: block_size=32, bytes_per_block=34
        let stride = expert_byte_stride(GgmlType::Q8_0, 4096 * 4096);
        let n_blocks = (4096 * 4096) / 32;
        assert_eq!(stride, n_blocks * 34);
    }

    #[test]
    fn test_expert_byte_stride_rounds_up_partial_quant_block() {
        let stride = expert_byte_stride(GgmlType::Q8_0, 33);
        assert_eq!(stride, 2 * GgmlType::Q8_0.bytes_per_block());
    }

    #[test]
    fn test_expert_byte_stride_f32() {
        // F32: block_size=1, bytes_per_block=4
        let stride = expert_byte_stride(GgmlType::F32, 1024);
        assert_eq!(stride, 1024 * 4);
    }

    // ── Validate config tests ────────────────────────────────────────

    #[test]
    fn test_validate_config_requires_n_expert() {
        let cfg = ModelConfig {
            architecture: "qwen3".to_string(),
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 2,
            embedding_dim: 256,
            head_dim: 32,
            intermediate_dim: 512,
            context_length: 1024,
            vocab_size: 1000,
            rms_norm_eps: 1e-5,
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
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            expert_intermediate_dim: None,
        };
        let fwd = Qwen3MoeForward;
        assert!(fwd.validate_config(&cfg).is_err());
    }

    #[test]
    fn test_validate_config_rejects_n_expert_used_greater_than_n_expert() {
        let cfg = ModelConfig {
            architecture: "qwen3".to_string(),
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 2,
            embedding_dim: 256,
            head_dim: 32,
            intermediate_dim: 512,
            context_length: 1024,
            vocab_size: 1000,
            rms_norm_eps: 1e-5,
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
            n_expert: Some(4),
            n_expert_used: Some(8),
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            expert_intermediate_dim: None,
        };
        let fwd = Qwen3MoeForward;
        let err = fwd.validate_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("n_expert_used"));
    }

    #[test]
    fn test_validate_config_accepts_valid_moe() {
        let cfg = ModelConfig {
            architecture: "qwen3".to_string(),
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 2,
            embedding_dim: 256,
            head_dim: 32,
            intermediate_dim: 512,
            context_length: 1024,
            vocab_size: 1000,
            rms_norm_eps: 1e-5,
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
            n_expert: Some(8),
            n_expert_used: Some(2),
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            expert_intermediate_dim: None,
        };
        let fwd = Qwen3MoeForward;
        assert!(fwd.validate_config(&cfg).is_ok());
    }

    // ── arch_name test ───────────────────────────────────────────────

    #[test]
    fn test_arch_name() {
        let fwd = Qwen3MoeForward;
        assert_eq!(fwd.arch_name(), "qwen3_moe");
    }

    // ── arch_registry MoE detection tests ────────────────────────────

    #[test]
    fn test_arch_registry_detects_moe() {
        let cfg = ModelConfig {
            architecture: "qwen3".to_string(),
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 2,
            embedding_dim: 256,
            head_dim: 32,
            intermediate_dim: 512,
            context_length: 1024,
            vocab_size: 1000,
            rms_norm_eps: 1e-5,
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
            n_expert: Some(8),
            n_expert_used: Some(2),
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            expert_intermediate_dim: None,
        };
        let fwd = crate::model::arch_registry::forward_for_arch_with_config("qwen3", &cfg).unwrap();
        assert_eq!(fwd.arch_name(), "qwen3_moe");
    }

    #[test]
    fn test_arch_registry_dense_qwen3_unchanged() {
        let cfg = ModelConfig {
            architecture: "qwen3".to_string(),
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 2,
            embedding_dim: 256,
            head_dim: 32,
            intermediate_dim: 512,
            context_length: 1024,
            vocab_size: 1000,
            rms_norm_eps: 1e-5,
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
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            expert_intermediate_dim: None,
        };
        let fwd = crate::model::arch_registry::forward_for_arch_with_config("qwen3", &cfg).unwrap();
        assert_eq!(fwd.arch_name(), "qwen3");
    }
}
