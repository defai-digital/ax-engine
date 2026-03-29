
/// Encode a fused dequant+matvec dispatch for the appropriate quant type.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_matvec(
    metal_ops: &MetalOps,
    encoder: &ax_engine_metal::MetalEncoder,
    weight: &ax_engine_metal::MetalBuffer,
    input: &ax_engine_metal::MetalBuffer,
    output: &ax_engine_metal::MetalBuffer,
    m: u32,
    k: u32,
    dtype: GgmlType,
) {
    encode_dequant_matvec_with_config(
        metal_ops,
        encoder,
        weight,
        input,
        output,
        m,
        k,
        dtype,
        metal_ops.dequant_dispatch_config(),
    )
}

/// Encode a fused dequant+matvec dispatch using an explicit dispatch config.
///
/// Decode execution planning uses this to keep kernel routing aligned with the
/// typed plan rather than re-reading backend state at every call site.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_matvec_with_config(
    metal_ops: &MetalOps,
    encoder: &ax_engine_metal::MetalEncoder,
    weight: &ax_engine_metal::MetalBuffer,
    input: &ax_engine_metal::MetalBuffer,
    output: &ax_engine_metal::MetalBuffer,
    m: u32,
    k: u32,
    dtype: GgmlType,
    dispatch_config: ax_engine_metal::DequantDispatchConfig,
) {
    match dtype {
        GgmlType::Q4_0 => metal_ops
            .dequant
            .encode_fused_matvec_q4_0(encoder, weight, input, output, m, k),
        GgmlType::Q8_0 => {
            if metal_ops
                .encode_precomputed_q4k_matvec_if_available(encoder, weight, input, output, m, k)
            {
                return;
            }
            metal_ops
                .dequant
                .encode_fused_matvec_q8_0(encoder, weight, input, output, m, k)
        }
        GgmlType::Q4K => {
            if metal_ops
                .encode_precomputed_q4k_matvec_if_available(encoder, weight, input, output, m, k)
            {
                return;
            }
            metal_ops.dequant.encode_fused_matvec_q4_k_with_config(
                encoder,
                weight,
                input,
                output,
                m,
                k,
                dispatch_config,
            )
        }
        GgmlType::Q5K => metal_ops.dequant.encode_fused_matvec_q5_k_with_config(
            encoder,
            weight,
            input,
            output,
            m,
            k,
            dispatch_config,
        ),
        GgmlType::Q6K => {
            if metal_ops
                .encode_precomputed_q4k_matvec_if_available(encoder, weight, input, output, m, k)
            {
                return;
            }
            metal_ops.dequant.encode_fused_matvec_q6_k_with_config(
                encoder,
                weight,
                input,
                output,
                m,
                k,
                dispatch_config,
            )
        }
        GgmlType::F32 => {
            // F32 weights: use the standard f32 matmul kernel (no dequant needed).
            metal_ops
                .matmul
                .encode_matvec(encoder, weight, input, output, m, k);
        }
        _ => panic!(
            "GPU phased dispatch only supports F32, Q4_0, Q8_0, Q4_K, Q5_K, and Q6_K, got {:?}",
            dtype
        ),
    }
}

/// Encode a fused pair of decode-side dequant+matvec dispatches when supported.
///
/// Returns `true` if a paired kernel was encoded. Callers should fall back to
/// separate `encode_dequant_matvec_with_config(...)` calls when this returns `false`.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_matvec_pair_with_config(
    metal_ops: &MetalOps,
    encoder: &ax_engine_metal::MetalEncoder,
    w0: &ax_engine_metal::MetalBuffer,
    w1: &ax_engine_metal::MetalBuffer,
    input: &ax_engine_metal::MetalBuffer,
    out0: &ax_engine_metal::MetalBuffer,
    out1: &ax_engine_metal::MetalBuffer,
    m: u32,
    k: u32,
    dtype0: GgmlType,
    dtype1: GgmlType,
    _dispatch_config: ax_engine_metal::DequantDispatchConfig,
    allow_pair_matvec: bool,
) -> bool {
    if !allow_pair_matvec {
        return false;
    }

    if dtype0 != dtype1 {
        return false;
    }

    // If either weight has a precomputed (dequantized) f16 version in the
    // cache, bail out and let the caller dispatch them individually via the
    // precomputed path.  We must NOT call encode_precomputed_q4k_matvec_if_available
    // here because it dispatches as a side effect — calling it for w0 and
    // then returning false would cause the caller to dispatch w0 a second time.
    if metal_ops.has_precomputed_weight(w0) || metal_ops.has_precomputed_weight(w1) {
        return false;
    }

    match dtype0 {
        GgmlType::Q4K => {
            metal_ops
                .dequant
                .encode_fused_matvec_pair_q4_k(encoder, w0, w1, input, out0, out1, m, k);
            true
        }
        GgmlType::Q5K => {
            metal_ops
                .dequant
                .encode_fused_matvec_pair_q5_k(encoder, w0, w1, input, out0, out1, m, k);
            true
        }
        GgmlType::Q6K => {
            metal_ops
                .dequant
                .encode_fused_matvec_pair_q6_k(encoder, w0, w1, input, out0, out1, m, k);
            true
        }
        GgmlType::Q8_0 => {
            metal_ops
                .dequant
                .encode_fused_matvec_pair_q8_0(encoder, w0, w1, input, out0, out1, m, k);
            true
        }
        _ => false,
    }
}

/// Encode a fused decode-side `SiLU(gate) * up` + down projection matvec when supported.
///
/// Returns `true` if a fused kernel was encoded. Callers should fall back to the
/// separate elementwise + matvec path when this returns `false`.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_silu_down_matvec_with_config(
    metal_ops: &MetalOps,
    encoder: &ax_engine_metal::MetalEncoder,
    weight: &ax_engine_metal::MetalBuffer,
    gate: &ax_engine_metal::MetalBuffer,
    up: &ax_engine_metal::MetalBuffer,
    output: &ax_engine_metal::MetalBuffer,
    m: u32,
    k: u32,
    dtype: GgmlType,
    _dispatch_config: ax_engine_metal::DequantDispatchConfig,
    allow_fused_silu_down: bool,
) -> bool {
    if !allow_fused_silu_down {
        return false;
    }

    if metal_ops.has_precomputed_weight(weight) {
        return false;
    }

    match dtype {
        GgmlType::Q4K => {
            metal_ops
                .dequant
                .encode_fused_silu_down_matvec_q4_k(encoder, weight, gate, up, output, m, k);
            true
        }
        GgmlType::Q5K => {
            metal_ops
                .dequant
                .encode_fused_silu_down_matvec_q5_k(encoder, weight, gate, up, output, m, k);
            true
        }
        GgmlType::Q6K => {
            metal_ops
                .dequant
                .encode_fused_silu_down_matvec_q6_k(encoder, weight, gate, up, output, m, k);
            true
        }
        GgmlType::Q8_0 => {
            metal_ops
                .dequant
                .encode_fused_silu_down_matvec_q8_0(encoder, weight, gate, up, output, m, k);
            true
        }
        _ => false,
    }
}

/// Encode a fused decode-side `GELU(gate) * up` + down projection matvec when supported.
///
/// Returns `true` if a fused kernel was encoded. Callers should fall back to the
/// separate elementwise + matvec path when this returns `false`.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_gelu_down_matvec_with_config(
    metal_ops: &MetalOps,
    encoder: &ax_engine_metal::MetalEncoder,
    weight: &ax_engine_metal::MetalBuffer,
    gate: &ax_engine_metal::MetalBuffer,
    up: &ax_engine_metal::MetalBuffer,
    output: &ax_engine_metal::MetalBuffer,
    m: u32,
    k: u32,
    dtype: GgmlType,
    _dispatch_config: ax_engine_metal::DequantDispatchConfig,
) -> bool {
    if !decode_fused_gelu_down_enabled() {
        return false;
    }

    if metal_ops.has_precomputed_weight(weight) {
        return false;
    }

    match dtype {
        GgmlType::Q4K => {
            metal_ops
                .dequant
                .encode_fused_gelu_down_matvec_q4_k(encoder, weight, gate, up, output, m, k);
            true
        }
        GgmlType::Q5K => {
            metal_ops
                .dequant
                .encode_fused_gelu_down_matvec_q5_k(encoder, weight, gate, up, output, m, k);
            true
        }
        GgmlType::Q6K => {
            metal_ops
                .dequant
                .encode_fused_gelu_down_matvec_q6_k(encoder, weight, gate, up, output, m, k);
            true
        }
        GgmlType::Q8_0 => {
            metal_ops
                .dequant
                .encode_fused_gelu_down_matvec_q8_0(encoder, weight, gate, up, output, m, k);
            true
        }
        _ => false,
    }
}

/// Strategy trait for the model-specific part of GPU decode layer encoding.
///
/// Steps 1 (QKV matmul), 6 (WO projection), and 8-10 (FFN tail) are shared
/// across all models. This trait covers the model-specific middle: QKV
/// post-processing (bias, norm, RoPE), KV cache append, attention dispatch,
/// and the first residual + FFN-norm handoff.
pub(super) trait GpuDecodeLayerStrategy {
    /// Encode the model-specific part of the GPU decode layer:
    /// QKV post-processing → KV append → attention → WO projection →
    /// post-attention residual + FFN norm.
    ///
    /// Inputs: `s.q_buf`, `s.k_buf`, `s.v_buf` (or `s.qkv_buf` if fused) are
    /// already populated by the shared QKV matmul. This method must produce:
    /// - `s.norm_buf` (FFN-norm of hidden after attention residual add)
    /// - Updated `hidden_buf` (with attention residual + WO projection added)
    #[allow(clippy::too_many_arguments)]
    fn encode_qkv_post_attend_residual(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        metal_ops: &MetalOps,
        s: &crate::backend::metal::GpuScratchBuffers,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        lw: &crate::backend::metal::CachedLayerKeys,
        weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
        gpu_kv: &crate::kv::GpuKv,
        layer: usize,
        exec_plan: &super::execution_plan::GpuDecodeExecutionPlan,
        barrier_fn: &dyn Fn(&ax_engine_metal::MetalEncoder),
        used_fused_qkv: bool,
    );
}

/// Dimensions needed by the shared GPU layer encoder.
pub(super) struct GpuLayerDims {
    pub dim: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub inter_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub eps: f32,
}

/// Encode a full GPU decode layer using the generic infrastructure.
///
/// Handles: QKV matmul → (strategy: QKV post + attend + residual) → WO →
/// gate/up pair → (shared: FFN tail).
///
/// The caller provides a `GpuDecodeLayerStrategy` for the model-specific
/// QKV post-processing, attention dispatch, and first residual handoff.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_gpu_decode_layer(
    encoder: &ax_engine_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_engine_metal::MetalBuffer,
    lw: &crate::backend::metal::CachedLayerKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
    fused_qkv_cache: &rustc_hash::FxHashMap<(usize, usize, usize), ax_engine_metal::MetalBuffer>,
    gpu_kv: &crate::kv::GpuKv,
    layer: usize,
    _n_layers: usize,
    exec_plan: &super::execution_plan::GpuDecodeExecutionPlan,
    next_attn_norm_key: Option<usize>,
    dims: &GpuLayerDims,
    strategy: &dyn GpuDecodeLayerStrategy,
    activation: super::layer_ops::FfnActivation,
    post_ffn_norm_w: Option<&ax_engine_metal::MetalBuffer>,
    barrier_fn: &dyn Fn(&ax_engine_metal::MetalEncoder),
) {
    let GpuLayerDims {
        dim,
        q_dim,
        kv_dim,
        inter_dim,
        eps,
        ..
    } = *dims;

    // --- Step 1: QKV matmul (shared) ---
    let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
    let fused_qkv_buf = if exec_plan.qkv == super::execution_plan::DecodeQkvPlan::Fused {
        fused_qkv_cache.get(&fused_qkv_key)
    } else {
        None
    };

    let used_fused = if let Some(fused_w) = fused_qkv_buf {
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            fused_w,
            &s.norm_buf,
            &s.qkv_buf,
            q_dim + 2 * kv_dim,
            dim,
            lw.wq_dtype,
            exec_plan.dequant_dispatch,
        );
        true
    } else {
        let wq_buf = weight_cache.get(&lw.wq).unwrap();
        let wk_buf = weight_cache.get(&lw.wk).unwrap();
        let wv_buf = weight_cache.get(&lw.wv).unwrap();
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wq_buf,
            &s.norm_buf,
            &s.q_buf,
            q_dim,
            dim,
            lw.wq_dtype,
            exec_plan.dequant_dispatch,
        );
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wk_buf,
            &s.norm_buf,
            &s.k_buf,
            kv_dim,
            dim,
            lw.wk_dtype,
            exec_plan.dequant_dispatch,
        );
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wv_buf,
            &s.norm_buf,
            &s.v_buf,
            kv_dim,
            dim,
            lw.wv_dtype,
            exec_plan.dequant_dispatch,
        );
        false
    };
    barrier_fn(encoder);

    // --- Steps 2-7: Model-specific QKV post + attention + WO + residual + FFN norm ---
    // Strategy handles: QKV post-processing, KV append, attention dispatch,
    // WO output projection, and residual + FFN norm handoff. This covers all
    // model-specific parts (bias, QK norm, RoPE base, sliding window, post-attn norm).
    strategy.encode_qkv_post_attend_residual(
        encoder,
        metal_ops,
        s,
        hidden_buf,
        lw,
        weight_cache,
        gpu_kv,
        layer,
        exec_plan,
        barrier_fn,
        used_fused,
    );

    let wg_buf = weight_cache.get(&lw.wg).unwrap();
    let wu_buf = weight_cache.get(&lw.wu).unwrap();
    if !encode_dequant_matvec_pair_with_config(
        metal_ops,
        encoder,
        wg_buf,
        wu_buf,
        &s.norm_buf,
        &s.gate_buf,
        &s.up_buf,
        inter_dim,
        dim,
        lw.wg_dtype,
        lw.wu_dtype,
        exec_plan.dequant_dispatch,
        exec_plan.use_pair_matvec,
    ) {
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wg_buf,
            &s.norm_buf,
            &s.gate_buf,
            inter_dim,
            dim,
            lw.wg_dtype,
            exec_plan.dequant_dispatch,
        );
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wu_buf,
            &s.norm_buf,
            &s.up_buf,
            inter_dim,
            dim,
            lw.wu_dtype,
            exec_plan.dequant_dispatch,
        );
    }
    barrier_fn(encoder);

    let wd_buf = weight_cache.get(&lw.wd).unwrap();
    let next_norm_w = next_attn_norm_key.and_then(|k| weight_cache.get(&k));
    encode_gpu_ffn_decode_tail(
        metal_ops,
        encoder,
        s,
        hidden_buf,
        wd_buf,
        lw.wd_dtype,
        dim,
        inter_dim,
        eps,
        exec_plan.dequant_dispatch,
        exec_plan.use_fused_silu_down,
        activation,
        post_ffn_norm_w,
        next_norm_w,
        barrier_fn,
    );
}

/// Encode the FFN activation+down+residual section of a GPU decode layer.
///
/// Handles: fused activation+down (when available) or separate activation + down matvec,
/// then the residual handoff to the next layer (fused residual+norm or plain add).
///
/// This is the shared tail of the GPU decode layer loop, identical across
/// LLaMA (SiLU), Qwen3 (SiLU), and Gemma3 (GELU) except for the activation
/// function and the optional post-FFN norm (Gemma3).
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_gpu_ffn_decode_tail(
    metal_ops: &MetalOps,
    encoder: &ax_engine_metal::MetalEncoder,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_engine_metal::MetalBuffer,
    wd_buf: &ax_engine_metal::MetalBuffer,
    wd_dtype: GgmlType,
    dim: u32,
    inter_dim: u32,
    eps: f32,
    exec_plan_dequant: ax_engine_metal::DequantDispatchConfig,
    allow_fused_silu_down: bool,
    activation: super::layer_ops::FfnActivation,
    post_ffn_norm_w: Option<&ax_engine_metal::MetalBuffer>,
    next_norm_w: Option<&ax_engine_metal::MetalBuffer>,
    barrier_fn: &dyn Fn(&ax_engine_metal::MetalEncoder),
) {
    use super::layer_ops::FfnActivation;

    // 1. Fused activation+down or separate activation + down matvec.
    let fused = match activation {
        FfnActivation::SiLU => encode_dequant_silu_down_matvec_with_config(
            metal_ops,
            encoder,
            wd_buf,
            &s.gate_buf,
            &s.up_buf,
            &s.down_buf,
            dim,
            inter_dim,
            wd_dtype,
            exec_plan_dequant,
            allow_fused_silu_down,
        ),
        FfnActivation::GELU => encode_dequant_gelu_down_matvec_with_config(
            metal_ops,
            encoder,
            wd_buf,
            &s.gate_buf,
            &s.up_buf,
            &s.down_buf,
            dim,
            inter_dim,
            wd_dtype,
            exec_plan_dequant,
        ),
    };
    if !fused {
        match activation {
            FfnActivation::SiLU => {
                metal_ops.elementwise.encode_silu_elementwise_mul(
                    encoder,
                    &s.gate_buf,
                    &s.up_buf,
                    inter_dim,
                );
            }
            FfnActivation::GELU => {
                metal_ops.elementwise.encode_gelu_elementwise_mul(
                    encoder,
                    &s.gate_buf,
                    &s.up_buf,
                    inter_dim,
                );
            }
        }
        barrier_fn(encoder);
        encode_dequant_matvec_with_config(
            metal_ops,
            encoder,
            wd_buf,
            &s.gate_buf,
            &s.down_buf,
            dim,
            inter_dim,
            wd_dtype,
            exec_plan_dequant,
        );
    }
    barrier_fn(encoder);

    // 2. Residual handoff: fused residual+norm for next layer, or plain add for last.
    if let Some(next_nw) = next_norm_w {
        if let Some(post_nw) = post_ffn_norm_w {
            metal_ops
                .elementwise
                .encode_post_ffn_norm_residual_add_rms_norm_out_batch(
                    encoder,
                    hidden_buf,
                    &s.down_buf,
                    post_nw,
                    next_nw,
                    &s.norm_buf,
                    dim,
                    1,
                    eps,
                );
        } else {
            metal_ops
                .elementwise
                .encode_residual_add_rms_norm_out_batch(
                    encoder,
                    hidden_buf,
                    &s.down_buf,
                    next_nw,
                    &s.norm_buf,
                    dim,
                    1,
                    eps,
                );
        }
        barrier_fn(encoder);
    } else {
        // Last layer: optional post-FFN norm then plain residual add.
        if let Some(post_nw) = post_ffn_norm_w {
            metal_ops
                .elementwise
                .encode_rms_norm(encoder, &s.down_buf, post_nw, dim, eps);
            barrier_fn(encoder);
        }
        metal_ops
            .elementwise
            .encode_elementwise_add(encoder, hidden_buf, &s.down_buf, dim);
    }
}

/// Encode the GPU output head: final RMSNorm + LM-head matvec → logits.
///
/// Shared across LLaMA, Qwen3, and Gemma3 (identical logic).
pub(super) fn encode_gpu_output_head(
    encoder: &ax_engine_metal::MetalEncoder,
    metal_ops: &MetalOps,
    s: &crate::backend::metal::GpuScratchBuffers,
    hidden_buf: &ax_engine_metal::MetalBuffer,
    exec_plan: &super::execution_plan::GpuDecodeExecutionPlan,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
    dim: u32,
    vocab_size: u32,
    eps: f32,
) {
    let decode_barrier = |encoder: &ax_engine_metal::MetalEncoder| {
        if exec_plan.barriers == super::execution_plan::DecodeBarrierPlan::Explicit {
            ax_engine_metal::barrier_buffers(encoder);
        }
    };

    let fnw_buf = weight_cache.get(&cached.output_norm).unwrap();
    decode_barrier(encoder);
    metal_ops
        .elementwise
        .encode_rms_norm(encoder, hidden_buf, fnw_buf, dim, eps);

    decode_barrier(encoder);
    let lm_buf = weight_cache.get(&cached.lm_head).unwrap();
    encode_dequant_matvec_with_config(
        metal_ops,
        encoder,
        lm_buf,
        hidden_buf,
        &s.logits_buf,
        vocab_size,
        dim,
        cached.lm_head_dtype,
        exec_plan.dequant_dispatch,
    );
}

