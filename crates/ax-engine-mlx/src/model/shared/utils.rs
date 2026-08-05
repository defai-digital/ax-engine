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

pub(crate) fn qkv_slices(cfg: &ModelConfig, head_dim: usize) -> QkvSlices {
    let q_size = (cfg.n_heads * head_dim) as i32;
    // The KV section width is fixed by the base geometry (`kv_head_count` ×
    // base head dim) even on layers whose per-layer `head_dim` is wider; the
    // manifest validator sizes packed tensors with the same rule.
    let kv_size = (cfg.n_kv_heads * cfg.head_dim) as i32;
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

pub(crate) fn qw(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    qw_with_policy(x, qw, ProjectionBatchPolicy::Shared)
}

pub(crate) fn qw_with_policy(
    x: &MlxArray,
    qw: &QuantizedWeight,
    policy: ProjectionBatchPolicy,
) -> MlxArray {
    let shape = x.shape();
    if policy == ProjectionBatchPolicy::RowExact
        && shape.len() == 3
        && shape[0] > 1
        && shape[1] == 1
    {
        let rows: Vec<MlxArray> = (0..shape[0])
            .map(|row| {
                let row = slice(x, &[row, 0, 0], &[row + 1, 1, shape[2]], &[1, 1, 1], None);
                qw_direct(&contiguous(&row, None), qw)
            })
            .collect();
        let refs: Vec<&MlxArray> = rows.iter().collect();
        return concatenate(&refs, 0, None);
    }
    qw_direct(x, qw)
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
            let kernel = INVARIANT_AFFINE_PROJECTION_KERNEL.get_or_init(|| {
                MlxMetalKernel::new(
                    "ax_invariant_affine_projection_v1",
                    &["x", "weight", "scales", "biases"],
                    &["out"],
                    INVARIANT_AFFINE_PROJECTION_KERNEL_SOURCE,
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
                    (out_dim.saturating_mul(256), 1, 1),
                    (256, 1, 1),
                    None,
                )
                .ok()?
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
