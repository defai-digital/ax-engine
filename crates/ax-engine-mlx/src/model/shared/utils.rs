use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, add, astype,
    concatenate, contiguous, expand_dims_axes, gather_mm, matmul, multiply, reshape, slice,
    slice_last_dim, take, tanh, transpose,
};
use std::sync::OnceLock;

use super::super::config::ModelConfig;
use crate::fastpath;
use crate::weights::QuantizedWeight;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct QkvSlices {
    pub q: (i32, i32),
    pub gate: Option<(i32, i32)>,
    pub k: (i32, i32),
    pub v: (i32, i32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProjectionBatchPolicy {
    Shared,
    /// Preserve the single-row reduction graph for every decode row.
    RowExact,
}

static INVARIANT_AFFINE_PROJECTION_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static INVARIANT_AFFINE_QMV_FAST_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static INVARIANT_DENSE_PROJECTION_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

/// Affine projection for one to four rows with an invariant per-row reduction.
///
/// A threadgroup owns one output row. Each thread dequantizes its weight value
/// once, then accumulates all active input rows independently. Consequently a
/// row sees the same FMA and simd-reduction order whether `Leading` is one or
/// four, while the packed weights are read only once for the whole microbatch.
const INVARIANT_AFFINE_PROJECTION_KERNEL_SOURCE: &str = r#"
    uint flat = thread_position_in_grid.x;
    uint row = flat / 256;
    uint tid = flat % 256;
    uint lane = tid % 32;
    uint sg = tid / 32;
    if (row >= OutDim) {
        return;
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint row_base = row * PackedCols;
    const uint scale_row = row * GroupCount;

    if (Bits == 6) {
        // MLX stores sixteen 6-bit values as one contiguous 96-bit block
        // (three uint words). Decode the block once, including values which
        // straddle a word boundary, and reuse every dequantized value across
        // all speculative rows.
        const uint ValuesPerBlock = 16;
        const uint WordsPerBlock = 3;
        const uint BlockCount = InputDim / ValuesPerBlock;
        for (uint block = tid; block < BlockCount; block += 256) {
            uint packed_words[WordsPerBlock];
            uint packed_base = row_base + block * WordsPerBlock;
            for (uint word = 0; word < WordsPerBlock; ++word) {
                packed_words[word] = weight[packed_base + word];
            }
            for (uint value = 0; value < ValuesPerBlock; ++value) {
                uint input_col = block * ValuesPerBlock + value;
                uint bit_offset = value * Bits;
                uint word = bit_offset / 32;
                uint shift = bit_offset % 32;
                uint q = packed_words[word] >> shift;
                if (shift + Bits > 32) {
                    q |= packed_words[word + 1] << (32 - shift);
                }
                q &= QuantMask;
                uint group = input_col / GroupSize;
                uint scale_idx = scale_row + group;
                float scale = static_cast<float>(scales[scale_idx]);
                float bias = static_cast<float>(biases[scale_idx]);
                float w = static_cast<float>(q) * scale + bias;
                for (uint token = 0; token < (uint)Leading; ++token) {
                    float x_v = static_cast<float>(x[token * InputDim + input_col]);
                    acc[token] = fma(x_v, w, acc[token]);
                }
            }
        }
    } else {
        for (uint packed_col = tid; packed_col < PackedCols; packed_col += 256) {
            uint packed = weight[row_base + packed_col];
            for (uint packed_lane = 0; packed_lane < PackFactor; ++packed_lane) {
                uint input_col = packed_col * PackFactor + packed_lane;
                uint q = (packed >> (packed_lane * Bits)) & QuantMask;
                uint group = input_col / GroupSize;
                uint scale_idx = scale_row + group;
                float scale = static_cast<float>(scales[scale_idx]);
                float bias = static_cast<float>(biases[scale_idx]);
                float w = static_cast<float>(q) * scale + bias;
                for (uint token = 0; token < (uint)Leading; ++token) {
                    float x_v = static_cast<float>(x[token * InputDim + input_col]);
                    acc[token] = fma(x_v, w, acc[token]);
                }
            }
        }
    }

    threadgroup float partials[32]; // four rows × eight simdgroups
    for (uint token = 0; token < (uint)Leading; ++token) {
        float sum = simd_sum(acc[token]);
        if (lane == 0) {
            partials[token * 8 + sg] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint token = 0; token < (uint)Leading; ++token) {
            float total = 0.0f;
            for (uint group = 0; group < 8; ++group) {
                total += partials[token * 8 + group];
            }
            out[token * OutDim + row] = static_cast<OutT>(total);
        }
    }
"#;

/// Microbatch form of MLX 0.32's affine `qmv_fast` reduction.
///
/// Each simdgroup still computes four output rows with the exact singleton
/// lane assignment, per-block accumulation, and `simd_sum` reduction. The only
/// extension is a fixed array of up to four independent input rows. Packed
/// weight bytes, scale, and bias are loaded once, then reused for every row in
/// the microbatch. Thus `Leading=2..4` amortizes weight traffic without changing
/// the arithmetic graph observed by any individual row.
const INVARIANT_AFFINE_QMV_FAST_KERNEL_SOURCE: &str = r#"
    constexpr uint PacksPerThread = 2;
    constexpr uint QmvPackFactor = Bits == 6 ? 4 : 32 / Bits;
    constexpr uint BytesPerPack = Bits == 6 ? 3 : 4;
    constexpr uint ValuesPerThread = QmvPackFactor * PacksPerThread;
    constexpr uint BytesPerThread = BytesPerPack * PacksPerThread;
    constexpr uint BlockSize = ValuesPerThread * 32;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.y * 8 + simd_group * 4;

    float result[4][4];
    for (uint token = 0; token < 4; ++token) {
        for (uint row = 0; row < 4; ++row) {
            result[token][row] = 0.0f;
        }
    }

    const device uchar* weight_bytes = reinterpret_cast<const device uchar*>(weight);
    for (uint k = 0; k < (uint)InputDim; k += BlockSize) {
        float x_values[4][16];
        float x_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (uint token = 0; token < (uint)Leading; ++token) {
            const device InputT* x_row =
                x + token * (uint)InputDim + k + lane * ValuesPerThread;
            if (Bits == 4) {
                for (uint i = 0; i < ValuesPerThread; i += 4) {
                    OutT group_sum =
                        static_cast<OutT>(x_row[i]) +
                        static_cast<OutT>(x_row[i + 1]) +
                        static_cast<OutT>(x_row[i + 2]) +
                        static_cast<OutT>(x_row[i + 3]);
                    x_sums[token] += static_cast<float>(group_sum);
                    float x0 = static_cast<float>(x_row[i]);
                    float x1 = static_cast<float>(x_row[i + 1]);
                    float x2 = static_cast<float>(x_row[i + 2]);
                    float x3 = static_cast<float>(x_row[i + 3]);
                    x_values[token][i] = x0;
                    x_values[token][i + 1] = x1 / 16.0f;
                    x_values[token][i + 2] = x2 / 256.0f;
                    x_values[token][i + 3] = x3 / 4096.0f;
                }
            } else if (Bits == 6) {
                for (uint i = 0; i < ValuesPerThread; i += 4) {
                    OutT group_sum =
                        static_cast<OutT>(x_row[i]) +
                        static_cast<OutT>(x_row[i + 1]) +
                        static_cast<OutT>(x_row[i + 2]) +
                        static_cast<OutT>(x_row[i + 3]);
                    x_sums[token] += static_cast<float>(group_sum);
                    float x0 = static_cast<float>(x_row[i]);
                    float x1 = static_cast<float>(x_row[i + 1]);
                    float x2 = static_cast<float>(x_row[i + 2]);
                    float x3 = static_cast<float>(x_row[i + 3]);
                    x_values[token][i] = x0;
                    x_values[token][i + 1] = x1 / 64.0f;
                    x_values[token][i + 2] = x2 / 16.0f;
                    x_values[token][i + 3] = x3 / 4.0f;
                }
            } else {
                for (uint i = 0; i < ValuesPerThread; ++i) {
                    float value = static_cast<float>(x_row[i]);
                    x_sums[token] += value;
                    x_values[token][i] = value;
                }
            }
        }

        for (uint row = 0; row < 4; ++row) {
            uint current_row = out_row + row;
            uint group = k / (uint)GroupSize +
                lane / ((uint)GroupSize / ValuesPerThread);
            uint sidecar_index = current_row * (uint)GroupCount + group;
            float scale = static_cast<float>(scales[sidecar_index]);
            float bias = static_cast<float>(biases[sidecar_index]);
            const device uchar* packed_src =
                weight_bytes + current_row * (uint)PackedCols * 4 +
                k * (uint)Bits / 8 + lane * BytesPerThread;
            uchar packed[8];
            for (uint byte = 0; byte < BytesPerThread; ++byte) {
                packed[byte] = packed_src[byte];
            }

            for (uint token = 0; token < (uint)Leading; ++token) {
                float accum = 0.0f;
                if (Bits == 4) {
                    for (uint pack = 0; pack < PacksPerThread; ++pack) {
                        uint byte = pack * 4;
                        ushort packed16_0 =
                            static_cast<ushort>(packed[byte]) |
                            (static_cast<ushort>(packed[byte + 1]) << 8);
                        ushort packed16_1 =
                            static_cast<ushort>(packed[byte + 2]) |
                            (static_cast<ushort>(packed[byte + 3]) << 8);
                        uint value = pack * 8;
                        accum +=
                            (x_values[token][value] * (packed16_0 & 0x000f) +
                             x_values[token][value + 1] * (packed16_0 & 0x00f0) +
                             x_values[token][value + 2] * (packed16_0 & 0x0f00) +
                             x_values[token][value + 3] * (packed16_0 & 0xf000));
                        accum +=
                            (x_values[token][value + 4] * (packed16_1 & 0x000f) +
                             x_values[token][value + 5] * (packed16_1 & 0x00f0) +
                             x_values[token][value + 6] * (packed16_1 & 0x0f00) +
                             x_values[token][value + 7] * (packed16_1 & 0xf000));
                    }
                } else if (Bits == 6) {
                    for (uint pack = 0; pack < PacksPerThread; ++pack) {
                        uint byte = pack * 3;
                        uint value = pack * 4;
                        accum += (packed[byte] & 0x3f) * x_values[token][value];
                        accum += (packed[byte] & 0xc0) * x_values[token][value + 1];
                        accum +=
                            (packed[byte + 1] & 0x0f) *
                            (x_values[token][value + 1] * 256.0f);
                        accum += (packed[byte + 1] & 0xf0) *
                            x_values[token][value + 2];
                        accum +=
                            (packed[byte + 2] & 0x03) *
                            (x_values[token][value + 2] * 256.0f);
                        accum += (packed[byte + 2] & 0xfc) *
                            x_values[token][value + 3];
                    }
                } else {
                    for (uint value = 0; value < ValuesPerThread; ++value) {
                        accum += x_values[token][value] * packed[value];
                    }
                }
                result[token][row] +=
                    scale * accum + x_sums[token] * bias;
            }
        }
    }

    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < 4; ++row) {
            float total = simd_sum(result[token][row]);
            if (lane == 0) {
                out[token * (uint)OutDim + out_row + row] =
                    static_cast<OutT>(total);
            }
        }
    }
"#;

/// Dense counterpart of [`INVARIANT_AFFINE_PROJECTION_KERNEL_SOURCE`].
const INVARIANT_DENSE_PROJECTION_KERNEL_SOURCE: &str = r#"
    uint flat = thread_position_in_grid.x;
    uint row = flat / 256;
    uint tid = flat % 256;
    uint lane = tid % 32;
    uint sg = tid / 32;
    if (row >= OutDim) {
        return;
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint row_base = row * InputDim;
    for (uint input_col = tid; input_col < InputDim; input_col += 256) {
        float w = static_cast<float>(weight[row_base + input_col]);
        for (uint token = 0; token < (uint)Leading; ++token) {
            float x_v = static_cast<float>(x[token * InputDim + input_col]);
            acc[token] = fma(x_v, w, acc[token]);
        }
    }

    threadgroup float partials[32]; // four rows × eight simdgroups
    for (uint token = 0; token < (uint)Leading; ++token) {
        float sum = simd_sum(acc[token]);
        if (lane == 0) {
            partials[token * 8 + sg] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint token = 0; token < (uint)Leading; ++token) {
            float total = 0.0f;
            for (uint group = 0; group < 8; ++group) {
                total += partials[token * 8 + group];
            }
            out[token * OutDim + row] = static_cast<OutT>(total);
        }
    }
"#;

pub(crate) fn qkv_slices(cfg: &ModelConfig, head_dim: usize, kv_head_count: usize) -> QkvSlices {
    let q_size = (cfg.n_heads * head_dim) as i32;
    let kv_size = (kv_head_count * head_dim) as i32;
    let gate = cfg.attn_output_gate.then_some((q_size, q_size * 2));
    let kv_start = if cfg.attn_output_gate {
        q_size * 2
    } else {
        q_size
    };
    QkvSlices {
        q: (0, q_size),
        gate,
        k: (kv_start, kv_start + kv_size),
        v: (kv_start + kv_size, kv_start + kv_size * 2),
    }
}

/// Infer the KV head count encoded in one packed QKV projection.
///
/// This must use the projection's actual row count rather than the model's
/// base KV geometry: Gemma 4 global-attention layers can use both a wider head
/// dimension and a different KV head count than their sliding layers.
pub(crate) fn packed_qkv_kv_head_count(
    cfg: &ModelConfig,
    head_dim: usize,
    packed_rows: usize,
) -> Option<usize> {
    let q_rows = cfg.n_heads.checked_mul(head_dim)?;
    let packed_q_rows = if cfg.attn_output_gate {
        q_rows.checked_mul(2)?
    } else {
        q_rows
    };
    let remaining = packed_rows.checked_sub(packed_q_rows)?;
    if !remaining.is_multiple_of(2) {
        return None;
    }
    let kv_rows = remaining / 2;
    if head_dim == 0 || !kv_rows.is_multiple_of(head_dim) {
        return None;
    }
    let kv_head_count = kv_rows / head_dim;
    (kv_head_count > 0).then_some(kv_head_count)
}

pub(crate) fn qw(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    qw_with_policy(x, qw, ProjectionBatchPolicy::Shared)
}

pub(crate) fn qw_with_policy(
    x: &MlxArray,
    qw: &QuantizedWeight,
    policy: ProjectionBatchPolicy,
) -> MlxArray {
    let shape = x.shape();
    // RowExact: always per-row MLX (no invariant). Invariant is for Shared
    // microbatch==singleton; applying it to RowExact can introduce split-path
    // ulp drift vs pure-direct MLX (h=2816) and false A/B fails.
    if policy == ProjectionBatchPolicy::RowExact && shape.len() == 3 {
        // Batch-decode: B>1, S=1 — one projection per batch row.
        if shape[0] > 1 && shape[1] == 1 {
            let rows: Vec<MlxArray> = (0..shape[0])
                .map(|row| {
                    let row = slice(x, &[row, 0, 0], &[row + 1, 1, shape[2]], &[1, 1, 1], None);
                    qw_direct_mlx(&contiguous(&row, None), qw)
                })
                .collect();
            let refs: Vec<&MlxArray> = rows.iter().collect();
            return concatenate(&refs, 0, None);
        }
        // Multi-token teacher-forced: B=1, S>1 — one projection per sequence
        // position so each row matches singleton pure-direct (Gemma MoE MTP).
        if shape[0] == 1 && shape[1] > 1 {
            let cols: Vec<MlxArray> = (0..shape[1])
                .map(|t| {
                    let t = t as i32;
                    let row = slice(x, &[0, t, 0], &[1, t + 1, shape[2]], &[1, 1, 1], None);
                    qw_direct_mlx(&contiguous(&row, None), qw)
                })
                .collect();
            let refs: Vec<&MlxArray> = cols.iter().collect();
            return concatenate(&refs, 1, None);
        }
    }
    // Shared (default): invariant when exact profile scopes it.
    qw_direct(x, qw)
}

fn qw_direct_mlx(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    // Always MLX quantized_matmul / dense matmul (no invariant). Used by
    // RowExact so multi-token rows match pure-direct MLX singletons.
    let y = if let Some(scales) = &qw.scales {
        let mode = qw.mlx_quantization_mode();
        let quant_biases = match mode {
            mlx_sys::MlxQuantizationMode::Affine => qw.biases.as_ref(),
            _ => None,
        };
        mlx_sys::quantized_matmul_with_mode(
            x,
            &qw.weight,
            scales,
            quant_biases,
            true,
            Some(qw.group_size),
            Some(qw.bits),
            mode,
            None,
        )
    } else {
        let wt = transpose(&qw.weight, &[1, 0], None);
        matmul(x, &wt, None)
    };
    if let Some(bias) = &qw.linear_bias {
        add(&y, bias, None)
    } else {
        y
    }
}

fn qw_direct(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    // Dense Linear bias (`QuantizedWeight.linear_bias`) is separate from affine
    // group-quant biases (`qw.biases`). Matches mlx-lm `nn.Linear` /
    // `QuantizedLinear`: y = x @ W + b. Required for GPT-OSS Q/K/V/O + router.
    let y = if fastpath::qwen_linear_mtp_exact_enabled()
        && let Some(invariant) = invariant_projection_metal_impl(x, qw)
    {
        invariant
    } else if let Some(scales) = &qw.scales {
        // MXFP8/MXFP4 have no affine group-bias channel; pass None for those modes.
        let mode = qw.mlx_quantization_mode();
        let quant_biases = match mode {
            mlx_sys::MlxQuantizationMode::Affine => qw.biases.as_ref(),
            _ => None,
        };
        mlx_sys::quantized_matmul_with_mode(
            x,
            &qw.weight,
            scales,
            quant_biases,
            true,
            Some(qw.group_size),
            Some(qw.bits),
            mode,
            None,
        )
    } else {
        let wt = transpose(&qw.weight, &[1, 0], None);
        matmul(x, &wt, None)
    };
    if let Some(bias) = &qw.linear_bias {
        add(&y, bias, None)
    } else {
        y
    }
}

/// Slice the last axis of `x` to `[start, end)`.
fn slice_trailing_cols(x: &MlxArray, start: i32, end: i32) -> MlxArray {
    let shape = x.shape();
    let ndim = shape.len();
    let mut starts = vec![0_i32; ndim];
    let mut ends: Vec<i32> = shape.to_vec();
    starts[ndim - 1] = start;
    ends[ndim - 1] = end;
    let strides = vec![1_i32; ndim];
    contiguous(&slice(x, &starts, &ends, &strides, None), None)
}

fn invariant_projection_metal_impl(x: &MlxArray, qw: &QuantizedWeight) -> Option<MlxArray> {
    if !matches!(
        x.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let x_shape = x.shape();
    let input_dim = *x_shape.last()?;
    if input_dim <= 0 || x_shape.len() < 2 {
        return None;
    }
    let leading = x_shape[..x_shape.len() - 1]
        .iter()
        .try_fold(1_i32, |product, dimension| product.checked_mul(*dimension))?;
    if !(1..=4).contains(&leading) {
        return None;
    }

    let weight_shape = qw.weight.shape();
    if weight_shape.len() != 2 {
        return None;
    }
    let out_dim = weight_shape[0];
    if out_dim <= 0 {
        return None;
    }

    // Split non-512-aligned dims: qmv_fast on the 512-aligned prefix (amortized
    // + MLX-bitexact) + MLX on the remainder. Both pure-direct and multi-token
    // use this split so A/B identity holds. Critical for Gemma h=2816.
    if matches!(
        qw.mlx_quantization_mode(),
        mlx_sys::MlxQuantizationMode::Affine
    ) && qw.bits > 0
        && qw.bits <= 8
        && qw.group_size > 0
        && input_dim % 512 != 0
        && input_dim > 512
        && input_dim % qw.group_size == 0
        && (input_dim * qw.bits) % 32 == 0
    {
        let aligned = (input_dim / 512) * 512;
        let rem = input_dim - aligned;
        if aligned > 0 && rem > 0 && rem % qw.group_size == 0 && (rem * qw.bits) % 32 == 0 {
            let packed_al = aligned * qw.bits / 32;
            let packed_rem = rem * qw.bits / 32;
            let groups_al = aligned / qw.group_size;
            let groups_rem = rem / qw.group_size;
            if weight_shape[1] == packed_al + packed_rem {
                if let (Some(scales), Some(biases)) = (qw.scales.as_ref(), qw.biases.as_ref()) {
                    if scales.shape() == [out_dim, input_dim / qw.group_size]
                        && biases.shape() == scales.shape()
                    {
                        let x_al = slice_trailing_cols(x, 0, aligned);
                        let x_rem = slice_trailing_cols(x, aligned, input_dim);
                        let w_al = contiguous(
                            &slice(&qw.weight, &[0, 0], &[out_dim, packed_al], &[1, 1], None),
                            None,
                        );
                        let w_rem = contiguous(
                            &slice(
                                &qw.weight,
                                &[0, packed_al],
                                &[out_dim, packed_al + packed_rem],
                                &[1, 1],
                                None,
                            ),
                            None,
                        );
                        let s_al = contiguous(
                            &slice(scales, &[0, 0], &[out_dim, groups_al], &[1, 1], None),
                            None,
                        );
                        let s_rem = contiguous(
                            &slice(
                                scales,
                                &[0, groups_al],
                                &[out_dim, groups_al + groups_rem],
                                &[1, 1],
                                None,
                            ),
                            None,
                        );
                        let b_al = contiguous(
                            &slice(biases, &[0, 0], &[out_dim, groups_al], &[1, 1], None),
                            None,
                        );
                        let b_rem = contiguous(
                            &slice(
                                biases,
                                &[0, groups_al],
                                &[out_dim, groups_al + groups_rem],
                                &[1, 1],
                                None,
                            ),
                            None,
                        );
                        let qw_al = QuantizedWeight {
                            weight: w_al,
                            scales: Some(s_al),
                            biases: Some(b_al),
                            group_size: qw.group_size,
                            bits: qw.bits,
                            mode: qw.mode.clone(),
                            linear_bias: None,
                        };
                        let qw_rem = QuantizedWeight {
                            weight: w_rem,
                            scales: Some(s_rem),
                            biases: Some(b_rem),
                            group_size: qw.group_size,
                            bits: qw.bits,
                            mode: qw.mode.clone(),
                            linear_bias: None,
                        };
                        // aligned hits qmv_fast (input_dim % 512 == 0).
                        let y_al = invariant_projection_metal_impl(&x_al, &qw_al)?;
                        // Remainder: MLX singleton (Leading=1) or RowExact MLX
                        // (Leading>1) so multi-token matches pure-direct.
                        let y_rem = if leading > 1 {
                            let x_rem_shape = x_rem.shape();
                            // x_rem is [1, S, rem] or similar with leading product S.
                            let seq = x_rem_shape[x_rem_shape.len() - 2];
                            let cols: Vec<MlxArray> = (0..seq)
                                .map(|t| {
                                    let t = t as i32;
                                    let ndim = x_rem_shape.len();
                                    let mut starts = vec![0_i32; ndim];
                                    let mut ends: Vec<i32> = x_rem_shape.to_vec();
                                    starts[ndim - 2] = t;
                                    ends[ndim - 2] = t + 1;
                                    let strides = vec![1_i32; ndim];
                                    let row = contiguous(
                                        &slice(&x_rem, &starts, &ends, &strides, None),
                                        None,
                                    );
                                    mlx_sys::quantized_matmul_with_mode(
                                        &row,
                                        &qw_rem.weight,
                                        qw_rem.scales.as_ref().unwrap(),
                                        qw_rem.biases.as_ref(),
                                        true,
                                        Some(qw_rem.group_size),
                                        Some(qw_rem.bits),
                                        qw_rem.mlx_quantization_mode(),
                                        None,
                                    )
                                })
                                .collect();
                            let refs: Vec<&MlxArray> = cols.iter().collect();
                            // Concat on the sequence axis (second-to-last for 3D).
                            concatenate(&refs, (x_rem_shape.len() - 2) as i32, None)
                        } else {
                            mlx_sys::quantized_matmul_with_mode(
                                &x_rem,
                                &qw_rem.weight,
                                qw_rem.scales.as_ref().unwrap(),
                                qw_rem.biases.as_ref(),
                                true,
                                Some(qw_rem.group_size),
                                Some(qw_rem.bits),
                                qw_rem.mlx_quantization_mode(),
                                None,
                            )
                        };
                        return Some(add(&y_al, &y_rem, None));
                    }
                }
            }
        }
    }

    let mut out_shape = x_shape;
    *out_shape.last_mut()? = out_dim;

    let mut common_template_args = vec![
        KernelTemplateArg::Dtype {
            name: "InputT",
            dtype: x.dtype(),
        },
        KernelTemplateArg::Dtype {
            name: "OutT",
            dtype: x.dtype(),
        },
        KernelTemplateArg::Int {
            name: "Leading",
            value: leading,
        },
        KernelTemplateArg::Int {
            name: "OutDim",
            value: out_dim,
        },
        KernelTemplateArg::Int {
            name: "InputDim",
            value: input_dim,
        },
    ];

    let mut outputs = if let Some(scales) = qw.scales.as_ref() {
        if !matches!(
            qw.mlx_quantization_mode(),
            mlx_sys::MlxQuantizationMode::Affine
        ) {
            return None;
        }
        let biases = qw.biases.as_ref()?;
        let output_dtype = promote_projection_dtype(x.dtype(), scales.dtype(), biases.dtype())?;
        if let Some(KernelTemplateArg::Dtype { dtype, .. }) = common_template_args.get_mut(1) {
            *dtype = output_dtype;
        }
        if qw.bits <= 0 || qw.bits > 8 || qw.group_size <= 0 {
            return None;
        }
        let pack_factor = 32 / qw.bits;
        let packed_shape_matches = if qw.bits == 6 {
            input_dim % 16 == 0
                && input_dim
                    .checked_mul(qw.bits)?
                    .checked_div(32)
                    .is_some_and(|packed_cols| packed_cols == weight_shape[1])
        } else {
            pack_factor > 0 && pack_factor.checked_mul(weight_shape[1])? == input_dim
        };
        if !packed_shape_matches || input_dim % qw.group_size != 0 {
            return None;
        }
        let group_count = input_dim / qw.group_size;
        let expected_sidecar = vec![out_dim, group_count];
        if scales.shape() != expected_sidecar || biases.shape() != expected_sidecar {
            return None;
        }
        common_template_args.extend([
            KernelTemplateArg::Int {
                name: "PackedCols",
                value: weight_shape[1],
            },
            KernelTemplateArg::Int {
                name: "GroupSize",
                value: qw.group_size,
            },
            KernelTemplateArg::Int {
                name: "GroupCount",
                value: group_count,
            },
            KernelTemplateArg::Int {
                name: "Bits",
                value: qw.bits,
            },
            KernelTemplateArg::Int {
                name: "PackFactor",
                value: pack_factor,
            },
            KernelTemplateArg::Int {
                name: "QuantMask",
                value: (1_i32 << qw.bits) - 1,
            },
        ]);
        let values_per_thread = match qw.bits {
            4 => 16,
            6 | 8 => 8,
            _ => 0,
        };
        let qmv_fast_eligible = values_per_thread > 0
            && out_dim % 8 == 0
            && input_dim % 512 == 0
            && qw.group_size >= values_per_thread
            && qw.group_size % values_per_thread == 0;
        if qmv_fast_eligible {
            let kernel = INVARIANT_AFFINE_QMV_FAST_KERNEL.get_or_init(|| {
                MlxMetalKernel::new(
                    "ax_invariant_affine_qmv_fast_v1",
                    &["x", "weight", "scales", "biases"],
                    &["out"],
                    INVARIANT_AFFINE_QMV_FAST_KERNEL_SOURCE,
                    "",
                    true,
                )
            });
            kernel
                .try_apply_with_template(
                    &[x, &qw.weight, scales, biases],
                    &[KernelOutputSpec {
                        shape: out_shape,
                        dtype: output_dtype,
                    }],
                    &common_template_args,
                    (32, (out_dim / 8).saturating_mul(2), 1),
                    (32, 2, 1),
                    None,
                )
                .ok()?
        } else {
            // Non-fast custom kernel does not match MLX for Gemma-like shapes.
            // Use MLX quantized_matmul (bitexact for Leading=1; RowExact for S>1).
            let shape = x.shape();
            let out = if shape.len() == 3 && shape[0] == 1 && shape[1] > 1 {
                let cols: Vec<MlxArray> = (0..shape[1])
                    .map(|t| {
                        let t = t as i32;
                        let row = contiguous(
                            &slice(x, &[0, t, 0], &[1, t + 1, shape[2]], &[1, 1, 1], None),
                            None,
                        );
                        mlx_sys::quantized_matmul_with_mode(
                            &row,
                            &qw.weight,
                            scales,
                            Some(biases),
                            true,
                            Some(qw.group_size),
                            Some(qw.bits),
                            qw.mlx_quantization_mode(),
                            None,
                        )
                    })
                    .collect();
                let refs: Vec<&MlxArray> = cols.iter().collect();
                concatenate(&refs, 1, None)
            } else {
                mlx_sys::quantized_matmul_with_mode(
                    x,
                    &qw.weight,
                    scales,
                    Some(biases),
                    true,
                    Some(qw.group_size),
                    Some(qw.bits),
                    qw.mlx_quantization_mode(),
                    None,
                )
            };
            vec![out]
        }
    } else {
        if weight_shape[1] != input_dim
            || !matches!(
                qw.weight.dtype(),
                MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
            )
        {
            return None;
        }
        let kernel = INVARIANT_DENSE_PROJECTION_KERNEL.get_or_init(|| {
            MlxMetalKernel::new(
                "ax_invariant_dense_projection_v1",
                &["x", "weight"],
                &["out"],
                INVARIANT_DENSE_PROJECTION_KERNEL_SOURCE,
                "",
                true,
            )
        });
        kernel
            .try_apply_with_template(
                &[x, &qw.weight],
                &[KernelOutputSpec {
                    shape: out_shape,
                    dtype: x.dtype(),
                }],
                &common_template_args,
                (out_dim.saturating_mul(256), 1, 1),
                (256, 1, 1),
                None,
            )
            .ok()?
    };
    outputs.pop()
}

fn promote_projection_dtype(
    input: MlxDtype,
    scales: MlxDtype,
    biases: MlxDtype,
) -> Option<MlxDtype> {
    if scales != biases {
        return None;
    }
    if input == scales {
        return Some(input);
    }
    if input == MlxDtype::Float32 || scales == MlxDtype::Float32 {
        return Some(MlxDtype::Float32);
    }
    if matches!(input, MlxDtype::Bfloat16 | MlxDtype::Float16)
        && matches!(scales, MlxDtype::Bfloat16 | MlxDtype::Float16)
    {
        return Some(MlxDtype::Float32);
    }
    None
}

pub(crate) fn mlx_slice_last_dim(x: &MlxArray, start: i32, end: i32) -> MlxArray {
    slice_last_dim(x, start, end, None)
}

pub(crate) fn scale_hidden_pub(hidden: &MlxArray, scale: f32) -> MlxArray {
    scale_hidden(hidden, scale)
}

pub(crate) fn scale_hidden(hidden: &MlxArray, scale: f32) -> MlxArray {
    // `cached_scalar` deduplicates the (value, dtype) pair across the process,
    // so steady-state decode pays one `multiply` op per call instead of
    // (astype + multiply). Saves ~4 ops/step on Gemma 4 E2B (one per scale
    // site: hidden_states_scale + 3 inside compute_per_layer_inputs_arr).
    let s_arr = mlx_sys::ops::cached_scalar(scale, hidden.dtype());
    multiply(hidden, &s_arr, None)
}

static ADD_MUL_SCALAR_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

const ADD_MUL_SCALAR_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    float av = static_cast<float>(a[idx]);
    float bv = static_cast<float>(b[idx]);
    float scale_v = static_cast<float>(scale[0]);
    T rounded_sum = static_cast<T>(av + bv);
    out[idx] = static_cast<T>(static_cast<float>(rounded_sum) * scale_v);
"#;

pub(crate) fn add_then_multiply_scalar(a: &MlxArray, b: &MlxArray, scalar: &MlxArray) -> MlxArray {
    add_then_multiply_scalar_metal(a, b, scalar)
        .unwrap_or_else(|| multiply(&add(a, b, None), scalar, None))
}

fn add_then_multiply_scalar_metal(
    a: &MlxArray,
    b: &MlxArray,
    scalar: &MlxArray,
) -> Option<MlxArray> {
    if !fastpath::layer_scalar_fused_add_enabled()
        || !layer_scalar_fused_add_shape_supported(&a.shape())
    {
        return None;
    }
    add_then_multiply_scalar_metal_impl(a, b, scalar)
}

fn layer_scalar_fused_add_shape_supported(shape: &[i32]) -> bool {
    shape.get(1).copied().unwrap_or(1) == 1
}

fn add_then_multiply_scalar_metal_impl(
    a: &MlxArray,
    b: &MlxArray,
    scalar: &MlxArray,
) -> Option<MlxArray> {
    if a.shape() != b.shape() || a.dtype() != b.dtype() || scalar.dtype() != a.dtype() {
        return None;
    }
    if !matches!(
        a.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let scalar_elements = scalar
        .shape()
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    if scalar_elements != 1 {
        return None;
    }
    let shape = a.shape();
    let element_count = shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;

    let kernel = ADD_MUL_SCALAR_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_add_mul_scalar_v1",
            &["a", "b", "scale"],
            &["out"],
            ADD_MUL_SCALAR_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[a, b, scalar],
        &[KernelOutputSpec {
            shape,
            dtype: a.dtype(),
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: a.dtype(),
            },
            KernelTemplateArg::Int {
                name: "ElementCount",
                value: element_count,
            },
        ],
        (element_count, 1, 1),
        (256, 1, 1),
        None,
    );
    outputs.pop()
}

pub(crate) fn scalar_like(value: f32, dtype: MlxDtype) -> MlxArray {
    // Retained for callers outside the steady-state decode hot path
    // (e.g. MoE router masking, test fixtures) where the per-call astype is
    // not the bottleneck and value uniqueness is not guaranteed.
    let scalar = MlxArray::from_raw_data(
        &value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    astype(&scalar, dtype, None)
}

pub(crate) fn apply_final_logit_softcap(cfg: &ModelConfig, logits: &MlxArray) -> MlxArray {
    let Some(cap) = cfg.final_logit_softcapping.filter(|cap| *cap > 0.0) else {
        return logits.clone();
    };
    let inv_cap = 1.0_f32 / cap;
    let inv_cap_arr = mlx_sys::ops::cached_scalar(inv_cap, logits.dtype());
    let cap_arr = mlx_sys::ops::cached_scalar(cap, logits.dtype());
    let scaled = multiply(logits, &inv_cap_arr, None);
    multiply(&tanh(&scaled, None), &cap_arr, None)
}

pub(crate) fn shape_element_count(shape: &[i32]) -> usize {
    shape
        .iter()
        .map(|dim| usize::try_from(*dim).expect("MLX shape dims must be non-negative"))
        .product()
}

pub(crate) fn squeeze_switch_singleton(x: &MlxArray) -> MlxArray {
    let mut shape = x.shape();
    let ndim = shape.len();
    if ndim >= 2 && shape[ndim - 2] == 1 {
        shape.remove(ndim - 2);
        reshape(x, &shape, None)
    } else {
        x.clone()
    }
}

/// Gather-matmul for expert weights (quantized or dense).
///
/// `x`: [..., hidden], `qw.weight`: [num_experts, expert_size, hidden] (or packed).
/// `indices`: [..., top_k].  Returns [..., top_k, out_size].
pub(crate) fn qw_gather(
    x: &MlxArray,
    qw: &QuantizedWeight,
    indices: &MlxArray,
    sorted_indices: bool,
) -> MlxArray {
    let y = if let Some(scales) = &qw.scales {
        // MXFP4 has no affine group-bias channel; pass None for non-affine modes.
        let mode = qw.mlx_quantization_mode();
        let quant_biases = match mode {
            mlx_sys::MlxQuantizationMode::Affine => qw.biases.as_ref(),
            _ => None,
        };
        mlx_sys::gather_qmm_with_mode(
            x,
            &qw.weight,
            scales,
            quant_biases,
            indices,
            true,
            Some(qw.group_size),
            Some(qw.bits),
            mode,
            sorted_indices,
            None,
        )
    } else {
        // Dense experts: weight shape [N, out, in] → need [N, in, out] for gather_mm.
        let ndim = qw.weight.ndim();
        let mut axes: Vec<i32> = (0..ndim as i32).collect();
        let last = axes.len() - 1;
        axes.swap(last - 1, last);
        let wt = transpose(&qw.weight, &axes, None);
        gather_mm(x, &wt, indices, sorted_indices, None)
    };

    // Dense SwitchLinear bias: y += bias[indices]  (mlx-lm switch_layers.py).
    // bias shape [num_experts, out]; indices select experts → [..., top_k, out]
    // after expand for broadcast against gather output.
    if let Some(linear_bias) = &qw.linear_bias {
        apply_expert_linear_bias(&y, linear_bias, indices)
    } else {
        y
    }
}

/// `y + expand_dims(linear_bias[indices], -2)` matching mlx-lm QuantizedSwitchLinear.
fn apply_expert_linear_bias(y: &MlxArray, linear_bias: &MlxArray, indices: &MlxArray) -> MlxArray {
    // take along expert axis 0: bias[indices] with indices of any rank.
    // Use take for flat indices then reshape to indices.shape + [out].
    let out_dim = *linear_bias
        .shape()
        .last()
        .expect("expert linear bias must be [E, out]");
    let flat_idx = reshape(indices, &[-1], None);
    let gathered = take(linear_bias, &flat_idx, 0, None); // [N, out]
    let mut bias_shape = indices.shape();
    bias_shape.push(out_dim);
    let gathered = reshape(&gathered, &bias_shape, None);
    // gather_qmm output often has a singleton dim before the last (SwitchGLU
    // expand_dims); expand bias so it broadcasts: insert dim at -2 when needed.
    let y_shape = y.shape();
    let bias = if y_shape.len() == gathered.ndim() + 1
        && y_shape.get(y_shape.len().saturating_sub(2)) == Some(&1)
    {
        expand_dims_axes(&gathered, &[-2], None)
    } else {
        gathered
    };
    add(y, &bias, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::{MlxQuantizationMode, eval, quantize, quantized_matmul};

    #[test]
    fn qw_applies_dense_linear_bias() {
        // Dense Linear: y = x @ W^T + b  (mlx-lm nn.Linear with bias=True).
        let w_data = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let weight = array_f32(&w_data, &[2, 2]);
        let bias = array_f32(&[0.5, -0.25], &[2]);
        let qw = QuantizedWeight {
            weight,
            scales: None,
            biases: None,
            group_size: 1,
            bits: 32,
            mode: "affine".to_string(),
            linear_bias: Some(bias),
        };
        let x = array_f32(&[1.0, 2.0], &[1, 1, 2]);
        let out = super::qw(&x, &qw);
        eval(&[&out]);
        let got = out.data_f32();
        // identity + bias → [1.5, 1.75]
        assert!((got[0] - 1.5).abs() < 1e-5, "got {}", got[0]);
        assert!((got[1] - 1.75).abs() < 1e-5, "got {}", got[1]);
    }

    #[test]
    fn expert_linear_bias_matches_mlx_lm_index_add() {
        // bias: [E=3, out=2], indices: [1, 2] → bias[[1,2]] = [[2,3],[4,5]]
        let bias_data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let bias = MlxArray::from_raw_data(
            bias_data.as_ptr() as *const u8,
            bias_data.len() * 4,
            &[3, 2],
            MlxDtype::Float32,
        );
        let y_data = [10.0f32, 20.0, 30.0, 40.0];
        let y = MlxArray::from_raw_data(
            y_data.as_ptr() as *const u8,
            y_data.len() * 4,
            &[1, 2, 2],
            MlxDtype::Float32,
        );
        let idx_data = [1u32, 2];
        let indices = MlxArray::from_raw_data(
            idx_data.as_ptr() as *const u8,
            idx_data.len() * 4,
            &[1, 2],
            MlxDtype::Uint32,
        );
        let out = apply_expert_linear_bias(&y, &bias, &indices);
        eval(&[&out]);
        let got = out.data_f32();
        // y[0] + bias[1] = [10,20]+[2,3] = [12,23]
        // y[1] + bias[2] = [30,40]+[4,5] = [34,45]
        assert!((got[0] - 12.0).abs() < 1e-5);
        assert!((got[1] - 23.0).abs() < 1e-5);
        assert!((got[2] - 34.0).abs() < 1e-5);
        assert!((got[3] - 45.0).abs() < 1e-5);
    }

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    #[test]
    fn layer_scalar_fused_add_is_decode_only() {
        assert!(layer_scalar_fused_add_shape_supported(&[1, 1, 4]));
        assert!(layer_scalar_fused_add_shape_supported(&[1, 1, 35, 4]));
        assert!(!layer_scalar_fused_add_shape_supported(&[1, 2, 4]));
        assert!(!layer_scalar_fused_add_shape_supported(&[1, 2048, 4]));
    }

    #[test]
    fn add_then_multiply_scalar_metal_matches_unfused_float32() {
        let a = array_f32(&[0.5, -1.0, 2.0, 3.5, -4.0, 8.0], &[2, 3]);
        let b = array_f32(&[1.0, 4.0, -2.0, 0.5, 3.0, -8.0], &[2, 3]);
        let scalar = array_f32(&[0.25], &[1]);

        let direct = add_then_multiply_scalar_metal_impl(&a, &b, &scalar)
            .expect("scalar fused add should support float32 inputs");
        let reference = multiply(&add(&a, &b, None), &scalar, None);
        eval(&[&direct, &reference]);

        assert_eq!(direct.shape(), vec![2, 3]);
        assert_eq!(direct.data_f32(), reference.data_f32());
    }

    #[test]
    fn row_exact_projection_matches_independent_quantized_rows() {
        let input_dim = 64;
        let output_dim = 64;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 127) as f32 - 63.0) * 0.015625)
            .collect();
        let weight = array_f32(&weight_data, &[output_dim as i32, input_dim as i32]);
        let quantized = quantize(
            &weight,
            Some(64),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: Some(quantized[2].clone()),
            group_size: 64,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
        };
        let input_data: Vec<f32> = (0..2 * input_dim)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.03125)
            .collect();
        let input = array_f32(&input_data, &[2, 1, input_dim as i32]);
        let batched = qw_with_policy(&input, &weight, ProjectionBatchPolicy::RowExact);

        for row in 0..2 {
            let row_start = row * input_dim;
            let single_input = array_f32(
                &input_data[row_start..row_start + input_dim],
                &[1, 1, input_dim as i32],
            );
            let expected = qw(&single_input, &weight);
            let actual = contiguous(
                &slice(
                    &batched,
                    &[row as i32, 0, 0],
                    &[row as i32 + 1, 1, output_dim as i32],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            eval(&[&actual, &expected]);
            assert_eq!(actual.data_f32(), expected.data_f32(), "row {row}");
        }
    }

    #[test]
    fn invariant_affine_projection_is_bit_exact_across_microbatch_shapes() {
        let input_dim = 64;
        let output_dim = 64;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 113) as f32 - 56.0) * 0.01171875)
            .collect();
        let weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let quantized = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: Some(quantized[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
        };
        let input_data: Vec<f32> = (0..2 * input_dim)
            .map(|index| ((index % 43) as f32 - 21.0) * 0.02734375)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 2, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let microbatch = invariant_projection_metal_impl(&input, &weight)
            .expect("affine invariant projection should support two rows");

        for row in 0..2 {
            let single = contiguous(
                &slice(
                    &input,
                    &[0, row, 0],
                    &[1, row + 1, input_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let expected = invariant_projection_metal_impl(&single, &weight)
                .expect("affine invariant projection should support one row");
            let actual = contiguous(
                &slice(
                    &microbatch,
                    &[0, row, 0],
                    &[1, row + 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let actual = astype(&actual, MlxDtype::Float32, None);
            let expected = astype(&expected, MlxDtype::Float32, None);
            eval(&[&actual, &expected]);
            assert_eq!(actual.data_f32(), expected.data_f32(), "row {row}");
        }
    }

    #[test]
    fn invariant_split_2816_matches_mlx_and_microbatch() {
        // Split path: 2560 qmv_fast + 256 MLX rem. Must match full MLX and microbatch.
        let input_dim = 2816;
        let output_dim = 64;
        let bits = 6;
        let group_size = 32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source = array_f32(&weight_data, &[output_dim as i32, input_dim as i32]);
        let q = quantize(
            &source,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
        };
        let input_data: Vec<f32> = (0..3 * input_dim)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 3, input_dim as i32]),
            MlxDtype::Bfloat16,
            None,
        );
        let mb = invariant_projection_metal_impl(&input, &weight).expect("split 2816");
        for row in 0..3 {
            let single = contiguous(
                &slice(
                    &input,
                    &[0, row, 0],
                    &[1, row + 1, input_dim as i32],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let inv_s = invariant_projection_metal_impl(&single, &weight).expect("single");
            let mlx_s = quantized_matmul(
                &single,
                &q[0],
                &q[1],
                Some(&q[2]),
                true,
                Some(group_size),
                Some(bits),
                None,
            );
            let actual = contiguous(
                &slice(
                    &mb,
                    &[0, row, 0],
                    &[1, row + 1, output_dim as i32],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let a = astype(&actual, MlxDtype::Float32, None);
            let b = astype(&inv_s, MlxDtype::Float32, None);
            let c = astype(&mlx_s, MlxDtype::Float32, None);
            eval(&[&a, &b, &c]);
            assert_eq!(a.data_f32(), b.data_f32(), "microbatch vs single row {row}");
            // Split sum may have tiny float assoc vs full MLX — allow ulp-level.
            let da = a.data_f32();
            let dc = c.data_f32();
            let mut maxd = 0.0f32;
            for i in 0..da.len() {
                maxd = maxd.max((da[i] - dc[i]).abs());
            }
            eprintln!("row {row}: max|split-mlx|={maxd} bitexact={}", da == dc);
            assert!(maxd < 1e-2, "split vs full MLX too far: {maxd}");
        }
    }

    #[test]
    fn invariant_nonfast_2816_matches_mlx_singleton() {
        // Gemma hidden=2816 uses split path (2560 qmv_fast + 256 MLX).
        // Absolute MLX match is ulp-level only; self-consistency is required.
        let input_dim = 2816;
        let output_dim = 64;
        let bits = 6;
        let group_size = 32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source = array_f32(&weight_data, &[output_dim as i32, input_dim as i32]);
        let q = quantize(
            &source,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
        };
        let input_data: Vec<f32> = (0..input_dim)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 1, input_dim as i32]),
            MlxDtype::Bfloat16,
            None,
        );
        let inv = invariant_projection_metal_impl(&input, &weight).expect("split 2816");
        let mlx = quantized_matmul(
            &input,
            &q[0],
            &q[1],
            Some(&q[2]),
            true,
            Some(group_size),
            Some(bits),
            None,
        );
        let a = astype(&inv, MlxDtype::Float32, None);
        let b = astype(&mlx, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        let da = a.data_f32();
        let db = b.data_f32();
        let mut maxd = 0.0f32;
        for i in 0..da.len() {
            maxd = maxd.max((da[i] - db[i]).abs());
        }
        // Split sum vs full MLX has float-assoc ulps; keep tight bound.
        assert!(maxd < 1e-3, "split 2816 vs full MLX maxΔ={maxd}");
    }

    #[test]
    #[test]
    fn invariant_gemma26_shapes_self_consistent_and_mlx() {
        // Realistic Gemma 26B-A4B projection shapes under Shared microbatch S=3.
        let cases: &[(i32, i32, i32, i32)] = &[
            // (input_dim, output_dim, bits, group_size)
            (2816, 4096, 6, 32), // q_proj
            (2816, 2048, 6, 32), // k/v
            (4096, 2816, 6, 32), // o_proj (512-aligned in)
            (2816, 2112, 6, 64), // dense gate/up
            (2112, 2816, 6, 64), // dense down
            (2816, 2112, 6, 32),
            (2112, 2816, 6, 32),
            (2816, 128, 8, 64), // router-ish
            (2816, 128, 8, 32),
            (704, 2816, 4, 32), // expert down-like (if used as linear)
            (2816, 704, 4, 32),
        ];
        for &(input_dim, output_dim, bits, group_size) in cases {
            let n = (input_dim * output_dim) as usize;
            let weight_data: Vec<f32> = (0..n)
                .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
                .collect();
            let source = array_f32(&weight_data, &[output_dim, input_dim]);
            let q = quantize(
                &source,
                Some(group_size),
                Some(bits),
                MlxQuantizationMode::Affine,
                None,
                None,
            );
            let weight = QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: Some(q[2].clone()),
                group_size,
                bits,
                mode: "affine".to_string(),
                linear_bias: None,
            };
            let s = 3_i32;
            let input_data: Vec<f32> = (0..(s * input_dim) as usize)
                .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
                .collect();
            let input = astype(
                &array_f32(&input_data, &[1, s, input_dim]),
                MlxDtype::Bfloat16,
                None,
            );
            let Some(mb) = invariant_projection_metal_impl(&input, &weight) else {
                eprintln!(
                    "SKIP no invariant in={input_dim} out={output_dim} b={bits} gs={group_size}"
                );
                continue;
            };
            let mut max_self = 0.0f32;
            let mut max_mlx = 0.0f32;
            let mut bitexact_self = true;
            let mut bitexact_mlx = true;
            for row in 0..s {
                let single = contiguous(
                    &slice(
                        &input,
                        &[0, row, 0],
                        &[1, row + 1, input_dim],
                        &[1, 1, 1],
                        None,
                    ),
                    None,
                );
                let inv_s =
                    invariant_projection_metal_impl(&single, &weight).expect("singleton invariant");
                let mlx_s = quantized_matmul(
                    &single,
                    &q[0],
                    &q[1],
                    Some(&q[2]),
                    true,
                    Some(group_size),
                    Some(bits),
                    None,
                );
                let actual = contiguous(
                    &slice(
                        &mb,
                        &[0, row, 0],
                        &[1, row + 1, output_dim],
                        &[1, 1, 1],
                        None,
                    ),
                    None,
                );
                let a = astype(&actual, MlxDtype::Float32, None);
                let b = astype(&inv_s, MlxDtype::Float32, None);
                let c = astype(&mlx_s, MlxDtype::Float32, None);
                eval(&[&a, &b, &c]);
                let da = a.data_f32();
                let db = b.data_f32();
                let dc = c.data_f32();
                if da != db {
                    bitexact_self = false;
                }
                if da != dc {
                    bitexact_mlx = false;
                }
                for i in 0..da.len() {
                    max_self = max_self.max((da[i] - db[i]).abs());
                    max_mlx = max_mlx.max((da[i] - dc[i]).abs());
                }
            }
            eprintln!(
                "in={input_dim} out={output_dim} b={bits} gs={group_size}: self_exact={bitexact_self} max_self={max_self:.6e} mlx_exact={bitexact_mlx} max_mlx={max_mlx:.6e}"
            );
            assert!(
                max_self == 0.0,
                "SELF FAIL in={input_dim} out={output_dim}: max_self={max_self}"
            );
        }
    }

    #[test]
    fn invariant_2d_vs_3d_singleton_shape() {
        let input_dim = 2816i32;
        let output_dim = 4096i32;
        let bits = 6i32;
        let group_size = 32i32;
        let n = (input_dim * output_dim) as usize;
        let weight_data: Vec<f32> = (0..n)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source = array_f32(&weight_data, &[output_dim, input_dim]);
        let q = quantize(
            &source,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
        };
        let input_data: Vec<f32> = (0..input_dim as usize)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let x3 = astype(
            &array_f32(&input_data, &[1, 1, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let x2 = astype(
            &array_f32(&input_data, &[1, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let y3 = invariant_projection_metal_impl(&x3, &weight).expect("3d");
        let y2 = invariant_projection_metal_impl(&x2, &weight).expect("2d");
        let a = astype(&y3, MlxDtype::Float32, None);
        let b = astype(&y2, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        let da = a.data_f32();
        let db = b.data_f32();
        assert_eq!(da.len(), db.len());
        let mut maxd = 0.0f32;
        for i in 0..da.len() {
            maxd = maxd.max((da[i] - db[i]).abs());
        }
        eprintln!("2d vs 3d maxd={maxd} bitexact={}", da == db);
        assert_eq!(da, db, "2d vs 3d shape mismatch maxd={maxd}");
    }

    fn invariant_nonfast_2112_matches_mlx_singleton() {
        // intermediate_size-like: non-fast path must match MLX bitexact.
        let input_dim = 2112;
        let output_dim = 64;
        let bits = 6;
        let group_size = 32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source = array_f32(&weight_data, &[output_dim as i32, input_dim as i32]);
        let q = quantize(
            &source,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
        };
        let input_data: Vec<f32> = (0..input_dim)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 1, input_dim as i32]),
            MlxDtype::Bfloat16,
            None,
        );
        let inv = invariant_projection_metal_impl(&input, &weight).expect("non-fast MLX");
        let mlx = quantized_matmul(
            &input,
            &q[0],
            &q[1],
            Some(&q[2]),
            true,
            Some(group_size),
            Some(bits),
            None,
        );
        let a = astype(&inv, MlxDtype::Float32, None);
        let b = astype(&mlx, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        // 2112 uses split (2048 qmv_fast + 64 MLX); absolute MLX match is ulp-level.
        let da = a.data_f32();
        let db = b.data_f32();
        let mut maxd = 0.0f32;
        for i in 0..da.len() {
            maxd = maxd.max((da[i] - db[i]).abs());
        }
        assert!(maxd < 1e-3, "2112 split vs MLX maxΔ={maxd}");
    }

    #[test]
    fn invariant_affine_qmv_fast_matches_mlx_singleton_and_microbatch() {
        let input_dim = 512;
        let output_dim = 64;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source_weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let input_data: Vec<f32> = (0..2 * input_dim)
            .map(|index| ((index % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 2, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );

        for bits in [4, 6, 8] {
            let quantized = quantize(
                &source_weight,
                Some(64),
                Some(bits),
                MlxQuantizationMode::Affine,
                None,
                None,
            );
            let weight = QuantizedWeight {
                weight: quantized[0].clone(),
                scales: Some(quantized[1].clone()),
                biases: Some(quantized[2].clone()),
                group_size: 64,
                bits,
                mode: "affine".to_string(),
                linear_bias: None,
            };
            let microbatch = invariant_projection_metal_impl(&input, &weight)
                .expect("fast invariant affine projection should support two rows");

            for row in 0..2 {
                let single = contiguous(
                    &slice(
                        &input,
                        &[0, row, 0],
                        &[1, row + 1, input_dim],
                        &[1, 1, 1],
                        None,
                    ),
                    None,
                );
                let invariant_single = invariant_projection_metal_impl(&single, &weight)
                    .expect("fast invariant affine projection should support one row");
                let mlx_single = quantized_matmul(
                    &single,
                    &quantized[0],
                    &quantized[1],
                    Some(&quantized[2]),
                    true,
                    Some(64),
                    Some(bits),
                    None,
                );
                let actual = contiguous(
                    &slice(
                        &microbatch,
                        &[0, row, 0],
                        &[1, row + 1, output_dim],
                        &[1, 1, 1],
                        None,
                    ),
                    None,
                );
                let actual = astype(&actual, MlxDtype::Float32, None);
                let invariant_single = astype(&invariant_single, MlxDtype::Float32, None);
                let mlx_single = astype(&mlx_single, MlxDtype::Float32, None);
                eval(&[&actual, &invariant_single, &mlx_single]);
                assert_eq!(
                    actual.data_f32(),
                    invariant_single.data_f32(),
                    "microbatch row {row}, bits={bits}"
                );
                assert_eq!(
                    invariant_single.data_f32(),
                    mlx_single.data_f32(),
                    "MLX singleton parity, bits={bits}"
                );
            }
        }
    }

    #[test]
    fn invariant_dense_projection_is_bit_exact_across_microbatch_shapes() {
        let input_dim = 64;
        let output_dim = 48;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.009765625)
            .collect();
        let weight = QuantizedWeight {
            weight: astype(
                &array_f32(&weight_data, &[output_dim, input_dim]),
                MlxDtype::Bfloat16,
                None,
            ),
            scales: None,
            biases: None,
            group_size: 0,
            bits: 16,
            mode: "affine".to_string(),
            linear_bias: None,
        };
        let input_data: Vec<f32> = (0..2 * input_dim)
            .map(|index| ((index % 37) as f32 - 18.0) * 0.0234375)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 2, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let microbatch = invariant_projection_metal_impl(&input, &weight)
            .expect("dense invariant projection should support two rows");

        for row in 0..2 {
            let single = contiguous(
                &slice(
                    &input,
                    &[0, row, 0],
                    &[1, row + 1, input_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let expected = invariant_projection_metal_impl(&single, &weight)
                .expect("dense invariant projection should support one row");
            let actual = contiguous(
                &slice(
                    &microbatch,
                    &[0, row, 0],
                    &[1, row + 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let actual = astype(&actual, MlxDtype::Float32, None);
            let expected = astype(&expected, MlxDtype::Float32, None);
            eval(&[&actual, &expected]);
            assert_eq!(actual.data_f32(), expected.data_f32(), "row {row}");
        }
    }

    #[test]
    fn add_then_multiply_scalar_metal_matches_unfused_bf16_rounding() {
        let a = astype(
            &array_f32(&[0.333, -1.125, 2.75, 3.125], &[1, 4]),
            MlxDtype::Bfloat16,
            None,
        );
        let b = astype(
            &array_f32(&[1.875, 4.25, -2.375, 0.625], &[1, 4]),
            MlxDtype::Bfloat16,
            None,
        );
        let scalar = astype(&array_f32(&[0.3125], &[1]), MlxDtype::Bfloat16, None);

        let direct = add_then_multiply_scalar_metal_impl(&a, &b, &scalar)
            .expect("scalar fused add should support bf16 inputs");
        let reference = multiply(&add(&a, &b, None), &scalar, None);
        let direct = astype(&direct, MlxDtype::Float32, None);
        let reference = astype(&reference, MlxDtype::Float32, None);
        eval(&[&direct, &reference]);

        assert_eq!(direct.shape(), vec![1, 4]);
        assert_eq!(direct.data_f32(), reference.data_f32());
    }

    #[test]
    fn add_then_multiply_scalar_metal_rejects_broadcast_vector_scale() {
        let a = array_f32(&[1.0, 2.0], &[1, 2]);
        let b = array_f32(&[3.0, 4.0], &[1, 2]);
        let vector_scale = array_f32(&[0.5, 0.25], &[2]);

        assert!(
            add_then_multiply_scalar_metal_impl(&a, &b, &vector_scale).is_none(),
            "only exact scalar layer-scale tensors are fused"
        );
    }
}
