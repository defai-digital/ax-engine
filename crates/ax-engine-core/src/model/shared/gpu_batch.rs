#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_batch(
    dequant: &ax_engine_metal::DequantKernels,
    elementwise: &ax_engine_metal::ElementwiseKernels,
    encoder: &ax_engine_metal::MetalEncoder,
    weight: &ax_engine_metal::MetalBuffer,
    input: &ax_engine_metal::MetalBuffer,
    output: &ax_engine_metal::MetalBuffer,
    input_f16: &ax_engine_metal::MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: GgmlType,
    use_f16_io: bool,
    use_batch_simd: bool,
    use_q5k_small_n: bool,
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

    if use_f16_io || dtype == GgmlType::Q4_0 {
        elementwise.encode_cast_f32_to_f16(encoder, input, input_f16, n * k);
        match dtype {
            GgmlType::Q4_0 => {
                dequant.encode_fused_batch_q4_0_f16in(encoder, weight, input_f16, output, m, n, k)
            }
            GgmlType::Q4K => dequant.encode_fused_batch_q4_k_f16in_with_config(
                encoder,
                weight,
                input_f16,
                output,
                m,
                n,
                k,
                ax_engine_metal::DequantDispatchConfig::default(),
            ),
            GgmlType::Q6K => dequant.encode_fused_batch_q6_k_f16in_with_config(
                encoder,
                weight,
                input_f16,
                output,
                m,
                n,
                k,
                ax_engine_metal::DequantDispatchConfig::default(),
            ),
            _ => gpu_batch_prefill_panic(dtype),
        }
    } else {
        match dtype {
            GgmlType::Q4K => {
                dequant.encode_fused_batch_q4_k(encoder, weight, input, output, m, n, k)
            }
            GgmlType::Q5K => {
                if use_q5k_small_n {
                    dequant.encode_fused_batch_q5_k_small(encoder, weight, input, output, m, n, k)
                } else {
                    dequant.encode_fused_batch_q5_k(encoder, weight, input, output, m, n, k)
                }
            }
            GgmlType::Q6K => {
                dequant.encode_fused_batch_q6_k(encoder, weight, input, output, m, n, k)
            }
            // F32 not handled here — use encode_qwen35_batch_projection which
            // routes F32 directly to MatmulKernels::encode_matmul.
            _ => gpu_batch_prefill_panic(dtype),
        }
    }
}

/// Encode a batched dequant+matmul with pre-cast f16 input.
///
/// Caller is responsible for casting input to f16 in `input_f16` before calling.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_batch_f16in(
    metal_ops: &MetalOps,
    encoder: &ax_engine_metal::MetalEncoder,
    weight: &ax_engine_metal::MetalBuffer,
    input_f16: &ax_engine_metal::MetalBuffer,
    output: &ax_engine_metal::MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: GgmlType,
) {
    match dtype {
        GgmlType::Q4_0 => {
            metal_ops
                .dequant
                .encode_fused_batch_q4_0_f16in(encoder, weight, input_f16, output, m, n, k);
        }
        GgmlType::Q8_0 => {
            if metal_ops.metal_q8_batch_native_shape_enabled(m, n, k) {
                metal_ops.dequant.encode_fused_batch_q8_0_f16in_with_config(
                    encoder,
                    weight,
                    input_f16,
                    output,
                    m,
                    n,
                    k,
                    metal_ops.dequant_dispatch_config(),
                );
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
            metal_ops.dequant.encode_fused_batch_q4_k_f16in_with_config(
                encoder,
                weight,
                input_f16,
                output,
                m,
                n,
                k,
                metal_ops.dequant_dispatch_config(),
            )
        }
        GgmlType::Q6K => {
            if metal_ops.encode_precomputed_q4k_batch_if_available(
                encoder, weight, input_f16, output, m, n, k,
            ) {
                return;
            }
            metal_ops.dequant.encode_fused_batch_q6_k_f16in_with_config(
                encoder,
                weight,
                input_f16,
                output,
                m,
                n,
                k,
                metal_ops.dequant_dispatch_config(),
            )
        }
        GgmlType::Q5K => metal_ops
            .dequant
            .encode_fused_batch_q5_k_f16in(encoder, weight, input_f16, output, m, n, k),
        _ => gpu_batch_prefill_panic(dtype),
    }
}

/// Encode a batched LM-head projection from `[n × k]` hidden states to
/// `[n × vocab]` logits.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_batch_logits(
    metal_ops: &MetalOps,
    encoder: &ax_engine_metal::MetalEncoder,
    weight: &ax_engine_metal::MetalBuffer,
    hidden: &ax_engine_metal::MetalBuffer,
    hidden_f16: &ax_engine_metal::MetalBuffer,
    logits: &ax_engine_metal::MetalBuffer,
    vocab: u32,
    n_rows: u32,
    hidden_dim: u32,
    dtype: GgmlType,
    prefer_f16_io: bool,
    use_batch_simd: bool,
) {
    match dtype {
        GgmlType::Q4_0 | GgmlType::Q8_0 => {
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
                    false,
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
    dequant: &ax_engine_metal::DequantKernels,
    encoder: &ax_engine_metal::MetalEncoder,
    w0: &ax_engine_metal::MetalBuffer,
    w1: &ax_engine_metal::MetalBuffer,
    input_f16: &ax_engine_metal::MetalBuffer,
    out0: &ax_engine_metal::MetalBuffer,
    out1: &ax_engine_metal::MetalBuffer,
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
        ),
    }
}
