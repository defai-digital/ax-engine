//! Shared helpers used by all three model implementations (LLaMA, Gemma3, Qwen3).

use crate::backend::metal::MetalOps;
use crate::compute::rms_norm;
use crate::gguf::tensor::GgmlType;
use crate::model::weights::WeightStore;

/// Check if all layer-0 weight tensors use GPU-supported quant types.
/// Falls back to CPU path for models with unsupported types.
pub(super) fn gpu_quant_supported(weights: &WeightStore) -> bool {
    const LAYER_SUFFIXES: &[&str] = &[
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];
    let is_supported = |dtype: GgmlType| {
        matches!(
            dtype,
            GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q6K | GgmlType::F32
        )
    };
    all_layers_match(weights, LAYER_SUFFIXES, is_supported)
}

/// Check if all layer-0 weight tensors use quant types supported by decode-only GPU path.
///
/// Decode can use additional fused matvec kernels (such as Q8_0) that are not yet
/// available for batch-prefill kernels.
pub(super) fn gpu_decode_quant_supported(weights: &WeightStore) -> bool {
    const LAYER_SUFFIXES: &[&str] = &[
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];
    let is_supported = |dtype: GgmlType| {
        matches!(
            dtype,
            GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0 | GgmlType::F32
        )
    };
    all_layers_match(weights, LAYER_SUFFIXES, is_supported)
}

/// Return true when a quantized LM head can use the existing batched GPU matmul path.
pub(super) fn gpu_batch_logits_supported(dtype: GgmlType) -> bool {
    matches!(dtype, GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0)
}

fn all_layers_match(
    weights: &WeightStore,
    layer_suffixes: &[&str],
    is_supported: impl Fn(GgmlType) -> bool,
) -> bool {
    for layer in 0usize.. {
        let probe = format!("blk.{layer}.{}", layer_suffixes[0]);
        if !weights.has(&probe) {
            break;
        }

        for suffix in layer_suffixes {
            let name = format!("blk.{layer}.{suffix}");
            match weights.raw_with_dtype(&name) {
                Ok((_, dtype)) if is_supported(dtype) => {}
                Ok((_, dtype)) => {
                    tracing::warn!(%name, ?dtype, "unsupported quant dtype for GPU path");
                    return false;
                }
                Err(e) => {
                    tracing::warn!(%name, error = %e, "missing or unreadable tensor for GPU path");
                    return false;
                }
            }
        }
    }
    true
}

/// Apply per-head RMSNorm in-place.
///
/// `buf` contains `n_heads` concatenated vectors of size `head_dim`.
/// `weight` has length `head_dim` and is shared across all heads.
pub(super) fn per_head_rms_norm(
    buf: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) {
    debug_assert_eq!(buf.len(), n_heads * head_dim);
    debug_assert_eq!(weight.len(), head_dim);
    for head in buf.chunks_mut(head_dim) {
        rms_norm::rms_norm(head, weight, eps);
    }
}

/// Encode a fused dequant+matvec dispatch for the appropriate quant type.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_matvec(
    metal_ops: &MetalOps,
    encoder: &ax_metal::MetalEncoder,
    weight: &ax_metal::MetalBuffer,
    input: &ax_metal::MetalBuffer,
    output: &ax_metal::MetalBuffer,
    m: u32,
    k: u32,
    dtype: GgmlType,
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
            metal_ops
                .dequant
                .encode_fused_matvec_q4_k(encoder, weight, input, output, m, k)
        }
        GgmlType::Q6K => {
            if metal_ops
                .encode_precomputed_q4k_matvec_if_available(encoder, weight, input, output, m, k)
            {
                return;
            }
            metal_ops
                .dequant
                .encode_fused_matvec_q6_k(encoder, weight, input, output, m, k)
        }
        _ => panic!(
            "GPU phased dispatch only supports Q4_0, Q8_0, Q4_K, and Q6_K, got {:?}",
            dtype
        ),
    }
}

/// Encode a batched dequant+matmul: C[N×M] = B[N×K] × dequant(A[M×K])^T.
///
/// If `use_f16_io` is true, casts `input` to f16 into `input_f16` before dispatch
/// (avoids per-matmul output cast while keeping downstream buffers f32).
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_batch(
    dequant: &ax_metal::DequantKernels,
    elementwise: &ax_metal::ElementwiseKernels,
    encoder: &ax_metal::MetalEncoder,
    weight: &ax_metal::MetalBuffer,
    input: &ax_metal::MetalBuffer,
    output: &ax_metal::MetalBuffer,
    input_f16: &ax_metal::MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: GgmlType,
    use_f16_io: bool,
    use_batch_simd: bool,
) {
    if use_batch_simd {
        match dtype {
            GgmlType::Q4K => {
                dequant.encode_batch_simd_q4k(encoder, weight, input, output, m, n, k, m);
                return;
            }
            GgmlType::Q6K => {
                dequant.encode_batch_simd_q6k(encoder, weight, input, output, m, n, k, m);
                return;
            }
            _ => {}
        }
    }

    if use_f16_io {
        elementwise.encode_cast_f32_to_f16(encoder, input, input_f16, n * k);
        match dtype {
            GgmlType::Q4K => {
                dequant.encode_fused_batch_q4_k_f16in(encoder, weight, input_f16, output, m, n, k)
            }
            GgmlType::Q6K => {
                dequant.encode_fused_batch_q6_k_f16in(encoder, weight, input_f16, output, m, n, k)
            }
            _ => panic!(
                "GPU batch matmul only supports Q4_K and Q6_K, got {:?}",
                dtype
            ),
        }
    } else {
        match dtype {
            GgmlType::Q4K => {
                dequant.encode_fused_batch_q4_k(encoder, weight, input, output, m, n, k)
            }
            GgmlType::Q6K => {
                dequant.encode_fused_batch_q6_k(encoder, weight, input, output, m, n, k)
            }
            _ => panic!(
                "GPU batch matmul only supports Q4_K and Q6_K, got {:?}",
                dtype
            ),
        }
    }
}

/// Encode a batched dequant+matmul with pre-cast f16 input.
///
/// Caller is responsible for casting input to f16 in `input_f16` before calling.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_batch_f16in(
    metal_ops: &MetalOps,
    encoder: &ax_metal::MetalEncoder,
    weight: &ax_metal::MetalBuffer,
    input_f16: &ax_metal::MetalBuffer,
    output: &ax_metal::MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: GgmlType,
) {
    match dtype {
        GgmlType::Q8_0 => {
            if crate::backend::metal::metal_q8_batch_native_shape_enabled(m, n, k) {
                metal_ops
                    .dequant
                    .encode_fused_batch_q8_0_f16in(encoder, weight, input_f16, output, m, n, k);
                return;
            }
            if metal_ops
                .encode_precomputed_batch_if_available(encoder, weight, input_f16, output, m, n, k)
            {
                return;
            }
            panic!(
                "GPU batch matmul for Q8_0 requires precomputed dense f16 weight, got {:?}",
                dtype
            );
        }
        GgmlType::Q4K => {
            if metal_ops.encode_precomputed_q4k_batch_if_available(
                encoder, weight, input_f16, output, m, n, k,
            ) {
                return;
            }
            metal_ops
                .dequant
                .encode_fused_batch_q4_k_f16in(encoder, weight, input_f16, output, m, n, k)
        }
        GgmlType::Q6K => {
            if metal_ops.encode_precomputed_q4k_batch_if_available(
                encoder, weight, input_f16, output, m, n, k,
            ) {
                return;
            }
            metal_ops
                .dequant
                .encode_fused_batch_q6_k_f16in(encoder, weight, input_f16, output, m, n, k)
        }
        _ => panic!(
            "GPU batch matmul only supports Q4_K and Q6_K, got {:?}",
            dtype
        ),
    }
}

/// Encode a batched LM-head projection from `[n × k]` hidden states to
/// `[n × vocab]` logits.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_batch_logits(
    metal_ops: &MetalOps,
    encoder: &ax_metal::MetalEncoder,
    weight: &ax_metal::MetalBuffer,
    hidden: &ax_metal::MetalBuffer,
    hidden_f16: &ax_metal::MetalBuffer,
    logits: &ax_metal::MetalBuffer,
    vocab: u32,
    n_rows: u32,
    hidden_dim: u32,
    dtype: GgmlType,
    prefer_f16_io: bool,
    use_batch_simd: bool,
) {
    match dtype {
        GgmlType::Q8_0 => {
            metal_ops.elementwise.encode_cast_f32_to_f16(
                encoder,
                hidden,
                hidden_f16,
                n_rows * hidden_dim,
            );
            encode_dequant_batch_f16in(
                metal_ops, encoder, weight, hidden_f16, logits, vocab, n_rows, hidden_dim, dtype,
            );
        }
        GgmlType::Q4K | GgmlType::Q6K => {
            if prefer_f16_io {
                metal_ops.elementwise.encode_cast_f32_to_f16(
                    encoder,
                    hidden,
                    hidden_f16,
                    n_rows * hidden_dim,
                );
                encode_dequant_batch_f16in(
                    metal_ops, encoder, weight, hidden_f16, logits, vocab, n_rows, hidden_dim,
                    dtype,
                );
            } else {
                encode_dequant_batch(
                    &metal_ops.dequant,
                    &metal_ops.elementwise,
                    encoder,
                    weight,
                    hidden,
                    logits,
                    hidden_f16,
                    vocab,
                    n_rows,
                    hidden_dim,
                    dtype,
                    false,
                    use_batch_simd,
                );
            }
        }
        _ => panic!("GPU batch logits path does not support {:?}", dtype),
    }
}

/// Encode a fused pair of batched dequant+matmuls with pre-cast f16 input.
///
/// Dispatches gate and up projections in a single paired kernel.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_batch_pair_f16in(
    dequant: &ax_metal::DequantKernels,
    encoder: &ax_metal::MetalEncoder,
    w0: &ax_metal::MetalBuffer,
    w1: &ax_metal::MetalBuffer,
    input_f16: &ax_metal::MetalBuffer,
    out0: &ax_metal::MetalBuffer,
    out1: &ax_metal::MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: GgmlType,
) {
    match dtype {
        GgmlType::Q4K => dequant
            .encode_fused_batch_pair_q4_k_f16in(encoder, w0, w1, input_f16, out0, out1, m, n, k),
        GgmlType::Q6K => dequant
            .encode_fused_batch_pair_q6_k_f16in(encoder, w0, w1, input_f16, out0, out1, m, n, k),
        GgmlType::Q8_0 => dequant
            .encode_fused_batch_pair_q8_0_f16in(encoder, w0, w1, input_f16, out0, out1, m, n, k),
        _ => panic!(
            "GPU batch pair matmul only supports Q4_K, Q6_K, and Q8_0, got {:?}",
            dtype
        ),
    }
}
