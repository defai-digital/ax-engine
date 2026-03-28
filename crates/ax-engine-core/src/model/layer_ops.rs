//! Shared layer operations used by multiple model architectures.
//!
//! These functions extract the common patterns from the per-model forward
//! passes (LLaMA, Qwen3, Gemma3, Qwen3.5) to eliminate duplication. Each
//! model's `forward_single` / `forward_batch` delegates to these helpers
//! for the structurally identical parts of the transformer layer.

use crate::backend::Backend;
use crate::compute::{gelu, rms_norm, silu};
use crate::model::weights::WeightStore;

/// FFN activation function selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SiLU,
    GELU,
}

/// Apply the post-attention FFN block for a single decode token.
///
/// Sequence: RMSNorm → gate/up projections → activation → down projection → residual.
/// This is identical across all supported architectures except for the activation
/// function (SiLU for LLaMA/Qwen3/Qwen3.5, GELU for Gemma3) and the norm weight
/// name, which the caller resolves before passing `ffn_norm_w`.
#[allow(clippy::too_many_arguments)]
pub fn apply_ffn_single(
    backend: &dyn Backend,
    weights: &WeightStore,
    prefix: &str,
    hidden: &mut [f32],
    norm_buf: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    down_buf: &mut [f32],
    dim: usize,
    inter_dim: usize,
    ffn_norm_w: &[f32],
    rms_norm_eps: f32,
    activation: FfnActivation,
) {
    // 1. FFN norm: hidden → norm_buf
    rms_norm::rms_norm_out(hidden, ffn_norm_w, norm_buf, rms_norm_eps);

    // 2. Gate + Up projections (batched matvec on shared input)
    let (wg_raw, wg_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))
        .expect("missing ffn_gate weight");
    let (wu_raw, wu_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_up.weight"))
        .expect("missing ffn_up weight");
    backend.batch_dequant_matvec(
        &[(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)],
        norm_buf,
        dim,
        &mut [gate_buf, up_buf],
    );

    // 3. Activation: SiLU(gate) * up  or  GELU(gate) * up
    match activation {
        FfnActivation::SiLU => silu::silu_elementwise_mul(gate_buf, up_buf),
        FfnActivation::GELU => gelu::gelu_elementwise_mul(gate_buf, up_buf),
    }

    // 4. Down projection
    let (wd_raw, wd_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_down.weight"))
        .expect("missing ffn_down weight");
    backend.dequant_matmul(wd_raw, wd_dtype, gate_buf, down_buf, dim, 1, inter_dim);

    // 5. Optional post-FFN norm (Gemma3: `post_ffw_norm.weight`)
    if let Ok(post_ffw_norm_w) = weights.f32_slice(&format!("{prefix}.post_ffw_norm.weight")) {
        rms_norm::rms_norm(down_buf, post_ffw_norm_w, rms_norm_eps);
    }

    // 6. Residual add
    silu::elementwise_add(hidden, down_buf);
}

/// Apply the post-attention FFN block for a batch of tokens (prefill).
///
/// Same operation sequence as `apply_ffn_single` but operates on token-major
/// buffers `[n_tokens × dim]`. Uses `dequant_matmul_token_major` for GPU-
/// accelerated fused dequant batch matmul when available.
#[allow(clippy::too_many_arguments)]
pub fn apply_ffn_batch(
    backend: &dyn Backend,
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
    ffn_norm_w: &[f32],
    rms_norm_eps: f32,
    activation: FfnActivation,
) {
    // 1. FFN norm: per-token RMSNorm
    for t in 0..n_tokens {
        let start = t * dim;
        rms_norm::rms_norm_out(
            &hidden[start..start + dim],
            ffn_norm_w,
            &mut norm_buf[start..start + dim],
            rms_norm_eps,
        );
    }

    // 2. Gate + Up projections (token-major batch matmul)
    let (wg_raw, wg_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))
        .expect("missing ffn_gate weight");
    let (wu_raw, wu_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_up.weight"))
        .expect("missing ffn_up weight");
    backend.dequant_matmul_token_major(
        wg_raw, wg_dtype, norm_buf, gate_buf, n_tokens, inter_dim, dim,
    );
    backend
        .dequant_matmul_token_major(wu_raw, wu_dtype, norm_buf, up_buf, n_tokens, inter_dim, dim);

    // 3. Activation
    match activation {
        FfnActivation::SiLU => silu::silu_elementwise_mul(gate_buf, up_buf),
        FfnActivation::GELU => gelu::gelu_elementwise_mul(gate_buf, up_buf),
    }

    // 4. Down projection
    let (wd_raw, wd_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_down.weight"))
        .expect("missing ffn_down weight");
    backend.dequant_matmul_token_major(
        wd_raw, wd_dtype, gate_buf, down_buf, n_tokens, dim, inter_dim,
    );

    // 5. Residual add
    silu::elementwise_add(hidden, down_buf);
}
