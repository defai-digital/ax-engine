use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, add, astype,
    async_eval, concatenate, contiguous, dequantize_with_mode, eval, expand_dims_axes, gather_mm,
    matmul, multiply, reshape, slice, slice_last_dim, take, tanh, transpose,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::super::config::ModelConfig;
use crate::fastpath;
use crate::weights::{DECODE_LM_HEAD_QUANT_BITS, DECODE_LM_HEAD_QUANT_GROUP_SIZE, QuantizedWeight};

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

thread_local! {
    static QWEN_PREFILL_DEQUANT_DENSE_FAMILY: Cell<bool> = const { Cell::new(false) };
    static PREFILL_DEQUANT_DENSE_T: RefCell<HashMap<usize, MlxArray>> =
        RefCell::new(HashMap::new());
    static QWEN_PREFILL_SKIP_EMBED_CLIP: Cell<bool> = const { Cell::new(false) };
    static QWEN_PREFILL_SKIP_F32_SDPA: Cell<bool> = const { Cell::new(false) };
    static QWEN_PREFILL_BF16_EMBED_DEQUANT: Cell<bool> = const { Cell::new(false) };
    static QWEN_PREFILL_NATIVE_OFFSET_CAUSAL: Cell<bool> = const { Cell::new(false) };
    static QWEN_PREFILL_SKIP_SWIGLU_COMPILE: Cell<bool> = const { Cell::new(false) };
    static GEMMA4_PREFILL_SKIP_LAST_FFN_PACKED: Cell<bool> = const { Cell::new(false) };
}

/// Mark the current Qwen3.5 / Qwen3-Next layer so `qw` may dequant+dense.
pub(crate) fn set_qwen_prefill_dequant_dense_family(active: bool) {
    QWEN_PREFILL_DEQUANT_DENSE_FAMILY.set(active);
}

/// Enable skipping the unused embed-id `clip` for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_skip_embed_clip(active: bool) {
    QWEN_PREFILL_SKIP_EMBED_CLIP.set(active);
}

/// Whether [`set_qwen_prefill_skip_embed_clip`] is set for this forward.
pub(crate) fn qwen_prefill_skip_embed_clip_active() -> bool {
    QWEN_PREFILL_SKIP_EMBED_CLIP.get()
}

/// Enable skipping the unused f32 SDPA upcast for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_skip_f32_sdpa(active: bool) {
    QWEN_PREFILL_SKIP_F32_SDPA.set(active);
}

/// Whether [`set_qwen_prefill_skip_f32_sdpa`] is set for this forward.
pub(crate) fn qwen_prefill_skip_f32_sdpa_active() -> bool {
    QWEN_PREFILL_SKIP_F32_SDPA.get()
}

/// Arms [`set_qwen_prefill_skip_f32_sdpa`] for the rest of this forward.
pub(crate) struct QwenPrefillSkipF32SdpaGuard;

impl QwenPrefillSkipF32SdpaGuard {
    pub(crate) fn arm(active: bool) -> Self {
        set_qwen_prefill_skip_f32_sdpa(active);
        Self
    }
}

impl Drop for QwenPrefillSkipF32SdpaGuard {
    fn drop(&mut self) {
        set_qwen_prefill_skip_f32_sdpa(false);
    }
}

/// Enable BF16 embedding dequant for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_bf16_embed_dequant(active: bool) {
    QWEN_PREFILL_BF16_EMBED_DEQUANT.set(active);
}

/// Whether [`set_qwen_prefill_bf16_embed_dequant`] is set for this forward.
pub(crate) fn qwen_prefill_bf16_embed_dequant_active() -> bool {
    QWEN_PREFILL_BF16_EMBED_DEQUANT.get()
}

/// Enable native offset-causal SDPA for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_native_offset_causal(active: bool) {
    QWEN_PREFILL_NATIVE_OFFSET_CAUSAL.set(active);
}

/// Whether [`set_qwen_prefill_native_offset_causal`] is set for this forward.
pub(crate) fn qwen_prefill_native_offset_causal_active() -> bool {
    QWEN_PREFILL_NATIVE_OFFSET_CAUSAL.get()
}

/// Arms [`set_qwen_prefill_native_offset_causal`] for the rest of this forward.
pub(crate) struct QwenPrefillNativeOffsetCausalGuard;

impl QwenPrefillNativeOffsetCausalGuard {
    pub(crate) fn arm(active: bool) -> Self {
        set_qwen_prefill_native_offset_causal(active);
        Self
    }
}

impl Drop for QwenPrefillNativeOffsetCausalGuard {
    fn drop(&mut self) {
        set_qwen_prefill_native_offset_causal(false);
    }
}

/// Enable skipping unused SwiGLU compile for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_skip_swiglu_compile(active: bool) {
    QWEN_PREFILL_SKIP_SWIGLU_COMPILE.set(active);
}

/// Whether [`set_qwen_prefill_skip_swiglu_compile`] is set for this forward.
pub(crate) fn qwen_prefill_skip_swiglu_compile_active() -> bool {
    QWEN_PREFILL_SKIP_SWIGLU_COMPILE.get()
}

/// Arms [`set_qwen_prefill_skip_swiglu_compile`] for the rest of this forward.
pub(crate) struct QwenPrefillSkipSwigluCompileGuard;

impl QwenPrefillSkipSwigluCompileGuard {
    pub(crate) fn arm(active: bool) -> Self {
        set_qwen_prefill_skip_swiglu_compile(active);
        Self
    }
}

impl Drop for QwenPrefillSkipSwigluCompileGuard {
    fn drop(&mut self) {
        set_qwen_prefill_skip_swiglu_compile(false);
    }
}

pub(crate) fn set_gemma4_prefill_skip_last_ffn_packed(active: bool) {
    GEMMA4_PREFILL_SKIP_LAST_FFN_PACKED.set(active);
}

pub(crate) fn gemma4_prefill_skip_last_ffn_packed_active() -> bool {
    GEMMA4_PREFILL_SKIP_LAST_FFN_PACKED.get()
}

pub(crate) struct Gemma4PrefillSkipLastFfnPackedGuard;

impl Gemma4PrefillSkipLastFfnPackedGuard {
    pub(crate) fn arm(active: bool) -> Self {
        set_gemma4_prefill_skip_last_ffn_packed(active);
        Self
    }
}

impl Drop for Gemma4PrefillSkipLastFfnPackedGuard {
    fn drop(&mut self) {
        set_gemma4_prefill_skip_last_ffn_packed(false);
    }
}

/// Skip a no-op BF16 astype when the gather is already BF16.
pub(crate) fn qwen_prefill_maybe_skip_bf16_astype(
    x: &MlxArray,
    model_family: &str,
    seq: i32,
) -> MlxArray {
    if (fastpath::should_qwen_prefill_skip_bf16_astype(model_family, seq)
        || fastpath::should_gemma4_prefill_bf16_embed(model_family, seq))
        && x.dtype() == MlxDtype::Bfloat16
    {
        x.clone()
    } else {
        astype(x, MlxDtype::Bfloat16, None)
    }
}

/// Submit the embedding gather so GPU starts while the first layer is built.
pub(crate) fn qwen_prefill_maybe_async_embed(hidden: &MlxArray, model_family: &str, seq: i32) {
    qwen_prefill_maybe_async_embed_for(
        hidden,
        fastpath::qwen_prefill_async_embed_enabled(),
        model_family,
        seq,
    );
}

/// Pure helper for [`qwen_prefill_maybe_async_embed`].
pub(crate) fn qwen_prefill_maybe_async_embed_for(
    hidden: &MlxArray,
    enabled: bool,
    model_family: &str,
    seq: i32,
) {
    if fastpath::should_qwen_prefill_async_embed_for(enabled, model_family, seq) {
        async_eval(&[hidden]);
    }
}

fn activation_seq_len(x: &MlxArray) -> i32 {
    let shape = x.shape();
    match shape.len() {
        0 => 0,
        1 => shape[0],
        _ => shape[shape.len() - 2],
    }
}

fn qwen_prefill_dequant_dense_applies(x: &MlxArray) -> bool {
    QWEN_PREFILL_DEQUANT_DENSE_FAMILY.get()
        && fastpath::should_qwen_prefill_dequant_dense_for(
            fastpath::qwen_prefill_dequant_dense_enabled(),
            "qwen3_5",
            activation_seq_len(x),
        )
}

fn cached_prefill_dequant_weight_t(qw: &QuantizedWeight) -> Option<MlxArray> {
    let scales = qw.scales.as_ref()?;
    let key = qw as *const QuantizedWeight as usize;
    PREFILL_DEQUANT_DENSE_T.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return Some(existing.clone());
        }
        let mode = qw.mlx_quantization_mode();
        let quant_biases = match mode {
            mlx_sys::MlxQuantizationMode::Affine => qw.biases.as_ref(),
            _ => None,
        };
        let dense = dequantize_with_mode(
            &qw.weight,
            scales,
            quant_biases,
            Some(qw.group_size),
            Some(qw.bits),
            mode,
            None,
            Some(MlxDtype::Bfloat16),
            None,
        );
        let weight_t = transpose(&dense, &[1, 0], None);
        eval(&[&weight_t]);
        cache.borrow_mut().insert(key, weight_t.clone());
        Some(weight_t)
    })
}

static INVARIANT_AFFINE_QMV_FAST_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static INVARIANT_AFFINE_QMV_FAST_BF16_Q4_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static INVARIANT_MXFP4_QMV_FAST_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static INVARIANT_DENSE_PROJECTION_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

/// Microbatch form of MLX 0.32's MXFP4 `fp_qmv_fast`.
///
/// Same lane assignment, 512-wide K blocks, `qdot` nibble order, e8m0 scale,
/// and `simd_sum` as the singleton kernel. Packed weights and scales load once
/// and reuse across `Leading` input rows so S=2 verify amortizes the weight
/// read without changing any row's arithmetic versus S=1 `fp_qmv_fast`.
const INVARIANT_MXFP4_QMV_FAST_KERNEL_SOURCE: &str = r#"
    constexpr uint PacksPerThread = 2;
    constexpr uint PackFactor = 8;
    constexpr uint BytesPerPack = 4;
    constexpr uint ValuesPerThread = PackFactor * PacksPerThread;
    constexpr uint BytesPerThread = BytesPerPack * PacksPerThread;
    constexpr uint BlockSize = ValuesPerThread * 32;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.y * 8 + simd_group * 4;

    float result[Leading][4];
    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < 4; ++row) {
            result[token][row] = 0.0f;
        }
    }

    const device uchar* weight_bytes = reinterpret_cast<const device uchar*>(weight);
    for (uint k = 0; k < (uint)InputDim; k += BlockSize) {
        float x_values[Leading][16];
        for (uint token = 0; token < (uint)Leading; ++token) {
            const device InputT* x_row =
                x + token * (uint)InputDim + k + lane * ValuesPerThread;
            for (uint i = 0; i < ValuesPerThread; ++i) {
                x_values[token][i] = static_cast<float>(x_row[i]);
            }
        }

        for (uint row = 0; row < 4; ++row) {
            uint current_row = out_row + row;
            uint group = k / (uint)GroupSize +
                lane / ((uint)GroupSize / ValuesPerThread);
            uint sidecar_index = current_row * (uint)GroupCount + group;
            uchar scale_bits = scales[sidecar_index];
            uint scale_u32 = scale_bits == 0 ? 0x400000u : (uint(scale_bits) << 23);
            float scale = as_type<float>(scale_u32);
            const device uchar* packed_src =
                weight_bytes + current_row * (uint)PackedCols * 4 +
                k / 2 + lane * BytesPerThread;
            const device ushort* packed16 =
                reinterpret_cast<const device ushort*>(packed_src);

            for (uint token = 0; token < (uint)Leading; ++token) {
                float accum = 0.0f;
                for (uint i = 0; i < (ValuesPerThread / 4); ++i) {
                    ushort pack = packed16[i];
                    for (uint n = 0; n < 4; ++n) {
                        uint nibble = (pack >> (n * 4)) & 0xfu;
                        half mag = as_type<half>(ushort((nibble & 7u) << 9));
                        mag *= 16384.0h;
                        float deq = static_cast<float>(mag);
                        if (nibble & 8u) {
                            deq = -deq;
                        }
                        accum += x_values[token][4 * i + n] * deq;
                    }
                }
                result[token][row] += scale * accum;
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

/// `ax_invariant_affine_qmv_fast_v1` specialised for bf16 inputs and 4-bit
/// affine weights.
///
/// Same lane assignment, 512-wide K blocks, per-lane accumulation order,
/// per-block `scale * accum + x_sum * bias` fold and `simd_sum` as the
/// generic kernel, so every output row is bit-identical to it (pinned by
/// `invariant_affine_bf16_q4_kernel_matches_generic_bit_exact`). Two changes
/// keep the multi-row verify shape (`Leading` 3..4) off the register cliff
/// the generic kernel hits when it holds four f32 copies of every x value:
///
/// - x stays in registers as raw bf16 bit pairs (half the registers) and is
///   widened to f32 at use, which is exact;
/// - weights are unpacked to unscaled nibbles instead of pre-scaling x by
///   `1/16^i`: `x * n` and `(x / 16^i) * (16^i * n)` are the same real
///   number and each is rounded once, so the products are identical.
///
/// Measured on the M5 Max verify shapes (2026-09-22): neutral at `Leading`
/// 1..2, 4-11% faster at 3..4; end-to-end neutral on the 6bit-MTP pack
/// because the default throughput profile verifies with the relaxed stock
/// MLX arithmetic and only S=1 singleton steps reach this route (evidence in
/// `benchmarks/results/inference/mlx-inference/2026-09-22-m5-invariant-qmv-bf16-q4/`).
const INVARIANT_AFFINE_QMV_FAST_BF16_Q4_KERNEL_SOURCE: &str = r#"
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

    // bf16 x kept as raw bit pairs (half the registers of the f32 copy);
    // each value is widened to f32 at use (exact), and 4-bit weights are
    // unpacked to unscaled nibbles: x * n equals (x / 16) * (16 n) exactly,
    // so the products and their summation order match the singleton kernel.
    const device uchar* weight_bytes = reinterpret_cast<const device uchar*>(weight);
    for (uint k = 0; k < (uint)InputDim; k += BlockSize) {
        uint x_bits[4][8];
        float x_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (uint token = 0; token < (uint)Leading; ++token) {
            const device uint4* x_row = reinterpret_cast<const device uint4*>(
                x + token * (uint)InputDim + k + lane * ValuesPerThread);
            uint4 xl = x_row[0];
            uint4 xh = x_row[1];
            x_bits[token][0] = xl.x; x_bits[token][1] = xl.y;
            x_bits[token][2] = xl.z; x_bits[token][3] = xl.w;
            x_bits[token][4] = xh.x; x_bits[token][5] = xh.y;
            x_bits[token][6] = xh.z; x_bits[token][7] = xh.w;
            for (uint i = 0; i < ValuesPerThread; i += 4) {
                uint p0 = x_bits[token][i / 2];
                uint p1 = x_bits[token][i / 2 + 1];
                // Same OutT-typed 4-wide group sum as the generic kernel's
                // static_cast<OutT>(x_row[i]) chain. `bfloat` is the native
                // Metal 3.1+ type MLX uses for bf16 template arguments on the
                // supported macOS, so the reinterpret is free; widening
                // through f32 and converting back costs four extra
                // conversions per group and measured slower than the generic
                // kernel at S=4 (M5 Max, 2026-09-22).
                OutT group_sum =
                    static_cast<OutT>(as_type<bfloat>(static_cast<ushort>(p0 & 0xffff))) +
                    static_cast<OutT>(as_type<bfloat>(static_cast<ushort>(p0 >> 16))) +
                    static_cast<OutT>(as_type<bfloat>(static_cast<ushort>(p1 & 0xffff))) +
                    static_cast<OutT>(as_type<bfloat>(static_cast<ushort>(p1 >> 16)));
                x_sums[token] += static_cast<float>(group_sum);
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
            uint2 raw = *reinterpret_cast<const device uint2*>(packed_src);

            for (uint token = 0; token < (uint)Leading; ++token) {
                float accum = 0.0f;
                uint w0 = raw.x;
                uint w1 = raw.y;
                for (uint pack = 0; pack < PacksPerThread; ++pack) {
                    uint w = pack == 0 ? w0 : w1;
                    uint value = pack * 8;
                    uint xa = x_bits[token][value / 2];
                    uint xb = x_bits[token][value / 2 + 1];
                    uint xc = x_bits[token][value / 2 + 2];
                    uint xd = x_bits[token][value / 2 + 3];
                    accum +=
                        (as_type<float>(xa << 16) * (w & 0x0000000f) +
                         as_type<float>(xa & 0xffff0000) * ((w >> 4) & 0x0000000f) +
                         as_type<float>(xb << 16) * ((w >> 8) & 0x0000000f) +
                         as_type<float>(xb & 0xffff0000) * ((w >> 12) & 0x0000000f));
                    accum +=
                        (as_type<float>(xc << 16) * ((w >> 16) & 0x0000000f) +
                         as_type<float>(xc & 0xffff0000) * ((w >> 20) & 0x0000000f) +
                         as_type<float>(xd << 16) * ((w >> 24) & 0x0000000f) +
                         as_type<float>(xd & 0xffff0000) * (w >> 28));
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

/// Dense (non-quantized) microbatch projection with invariant per-row reduction.
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

/// Flatten `[B,S,H]` to `[B*S,H]` for steel qmm when the flat-qmm flag is on.
fn qwen_prefill_maybe_flat_qmm(x: &MlxArray, qmm: impl FnOnce(&MlxArray) -> MlxArray) -> MlxArray {
    qwen_prefill_maybe_flat_qmm_for(x, fastpath::qwen_prefill_flat_qmm_enabled(), qmm)
}

/// Pure helper for [`qwen_prefill_maybe_flat_qmm`].
pub(crate) fn qwen_prefill_maybe_flat_qmm_for(
    x: &MlxArray,
    enabled: bool,
    qmm: impl FnOnce(&MlxArray) -> MlxArray,
) -> MlxArray {
    let shape = x.shape();
    let seq = match shape.len() {
        3 => shape[1],
        _ => 0,
    };
    if !fastpath::should_qwen_prefill_flat_qmm_for(enabled, seq, shape.len()) {
        return qmm(x);
    }
    let batch = shape[0];
    let hidden = shape[2];
    let flat = reshape(x, &[batch * seq, hidden], None);
    let out = qmm(&flat);
    let out_last = *out.shape().last().unwrap_or(&hidden);
    reshape(&out, &[batch, seq, out_last], None)
}

/// Short verify is S=2..=4 `[1,S,H]`. Flatten to `[S,H]` so MLX qmm uses
/// the 2-D M=S path instead of a 3-D leading dim of 1. Exact verification
/// always keeps its historical route; relaxed verification is separately
/// admitted. Singleton decode stays `[1,1,H]`.
fn exact_mxfp4_short_qmm(
    x: &MlxArray,
    _mode: mlx_sys::MlxQuantizationMode,
    qmm: impl FnOnce(&MlxArray) -> MlxArray,
) -> MlxArray {
    let shape = x.shape();
    if !fastpath::qwen_linear_mtp_exact_enabled()
        || shape.len() != 3
        || shape[0] != 1
        || !fastpath::qwen_linear_mtp_verify_seq_contains(i64::from(shape[1]))
    {
        return qmm(x);
    }
    let seq = shape[1];
    let hidden = shape[2];
    let flat = reshape(x, &[seq, hidden], None);
    let out = qmm(&flat);
    let out_last = *out.shape().last().unwrap_or(&hidden);
    reshape(&out, &[1, seq, out_last], None)
}

/// Tile `[B,S,H]` into 512-token slices for steel qmm, then concatenate.
fn qwen_prefill_maybe_tile_qmm(x: &MlxArray, qmm: impl Fn(&MlxArray) -> MlxArray) -> MlxArray {
    qwen_prefill_maybe_tile_qmm_for(x, fastpath::qwen_prefill_tile_qmm_enabled(), "qwen3_5", qmm)
}

/// Pure helper for [`qwen_prefill_maybe_tile_qmm`].
pub(crate) fn qwen_prefill_maybe_tile_qmm_for(
    x: &MlxArray,
    enabled: bool,
    model_family: &str,
    qmm: impl Fn(&MlxArray) -> MlxArray,
) -> MlxArray {
    let shape = x.shape();
    if shape.len() != 3 {
        return qmm(x);
    }
    let seq = shape[1];
    if !fastpath::should_qwen_prefill_tile_qmm_for(enabled, model_family, seq) {
        return qmm(x);
    }
    let tile = fastpath::QWEN_PREFILL_QMM_TILE;
    if seq <= tile {
        return qmm(x);
    }
    let batch = shape[0];
    let hidden = shape[2];
    let mut parts: Vec<MlxArray> = Vec::new();
    let mut start = 0i32;
    while start < seq {
        let end = (start + tile).min(seq);
        let chunk = contiguous(
            &slice(x, &[0, start, 0], &[batch, end, hidden], &[1, 1, 1], None),
            None,
        );
        parts.push(qmm(&chunk));
        start = end;
    }
    let refs: Vec<&MlxArray> = parts.iter().collect();
    concatenate(&refs, 1, None)
}

pub(crate) fn qw(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    qw_with_policy(x, qw, ProjectionBatchPolicy::Shared)
}

pub(crate) fn qw_with_policy(
    x: &MlxArray,
    qw: &QuantizedWeight,
    policy: ProjectionBatchPolicy,
) -> MlxArray {
    // Under an invariant projection scope, the S=1 baseline uses this same
    // kernel through Shared and S>1 RowExact must use it too. The scope owner
    // is responsible for applying the arithmetic contract symmetrically.
    if policy == ProjectionBatchPolicy::RowExact
        && fastpath::qwen_linear_mtp_exact_enabled()
        && let Some(invariant) = invariant_projection_metal_impl(x, qw)
    {
        return invariant;
    }
    // Outside an invariant scope, RowExact stays on per-row MLX so it matches
    // an ordinary pure-direct singleton.
    if policy == ProjectionBatchPolicy::RowExact
        && let Some(row_exact) = qw_row_exact_mlx(x, qw)
    {
        return row_exact;
    }
    // Shared preserves the backend batch schedule unless an exact scope applies.
    // MXFP4 BF16 projections can differ from singleton rows even at S=2;
    // callers that retain singleton-compatible state must select RowExact.
    qw_direct(x, qw)
}

/// Per-row MLX projection so S>1 / B>1 matches the corresponding singleton.
fn qw_row_exact_mlx(x: &MlxArray, qw: &QuantizedWeight) -> Option<MlxArray> {
    let shape = x.shape();
    if shape.len() != 3 {
        return None;
    }
    // Batch-decode: B>1, S=1 — one projection per batch row.
    if shape[0] > 1 && shape[1] == 1 {
        let rows: Vec<MlxArray> = (0..shape[0])
            .map(|row| {
                let row = slice(x, &[row, 0, 0], &[row + 1, 1, shape[2]], &[1, 1, 1], None);
                qw_direct_mlx(&contiguous(&row, None), qw)
            })
            .collect();
        let refs: Vec<&MlxArray> = rows.iter().collect();
        return Some(concatenate(&refs, 0, None));
    }
    // Multi-token teacher-forced / MTP verify: B=1, S>1.
    if shape[0] == 1 && shape[1] > 1 {
        let cols: Vec<MlxArray> = (0..shape[1])
            .map(|t| {
                let row = slice(x, &[0, t, 0], &[1, t + 1, shape[2]], &[1, 1, 1], None);
                qw_direct_mlx(&contiguous(&row, None), qw)
            })
            .collect();
        let refs: Vec<&MlxArray> = cols.iter().collect();
        return Some(concatenate(&refs, 1, None));
    }
    None
}

/// Runtime 2-bit `lm_head` is a decode GEMV. Prefill (S>1) stays on the
/// BF16 `W_t` GEMM — q4 qmm was a wash at p2048 and 2-bit is worse there.
fn decode_lm_head_quant_cache_eligible(x: &MlxArray) -> bool {
    let shape = x.shape();
    if shape.len() < 2 {
        return shape.first().copied() == Some(1);
    }
    shape[..shape.len() - 1]
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))
        == Some(1)
}

fn qw_direct_mlx(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    // Always MLX quantized_matmul / dense matmul (no invariant). Used by
    // RowExact so multi-token rows match pure-direct MLX singletons.
    let y = if qwen_prefill_dequant_dense_applies(x)
        && let Some(weight_t) = cached_prefill_dequant_weight_t(qw)
    {
        matmul(x, &weight_t, None)
    } else if let Some(scales) = &qw.scales {
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
    } else if decode_lm_head_quant_cache_eligible(x)
        && let (Some(q_w), Some(q_s), Some(q_b)) = (
            qw.decode_q2_weight.as_ref(),
            qw.decode_q2_scales.as_ref(),
            qw.decode_q2_biases.as_ref(),
        )
    {
        mlx_sys::quantized_matmul_with_mode(
            x,
            q_w,
            q_s,
            Some(q_b),
            true,
            Some(DECODE_LM_HEAD_QUANT_GROUP_SIZE),
            Some(DECODE_LM_HEAD_QUANT_BITS),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
        )
    } else if let Some(weight_t) = &qw.decode_weight_t {
        matmul(x, weight_t, None)
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

static DENSE_WIDE_GEMV_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static DENSE_WIDE_GEMV_BF16_VEC4_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static DENSE_WIDE_GEMV_BF16_VEC8_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

/// Multi-row dense GEMV over a contiguous `[in, out]` weight: one thread per
/// output column, `Leading` (1..=8) f32 accumulators in registers, so each
/// weight element is read exactly once and FMA'd against every row. Adjacent
/// threads read adjacent columns (coalesced); `x` reads are warp-broadcast.
///
/// Scalar reference form. One 2-byte load per thread per `k` leaves the
/// memory system under-subscribed: ~260-290 GB/s on the 2.54 GB
/// Qwen3.8-27B bf16 head (M5 Max, 2026-09-21) against ~530-550 GB/s for
/// the vector forms below. Kept for non-bf16 weights and column counts
/// the vector forms cannot tile.
const DENSE_WIDE_GEMV_SOURCE: &str = r#"
    uint n = thread_position_in_grid.x;
    if (n >= (uint)OutDim) {
        return;
    }
    float acc[Leading];
    for (uint s = 0; s < (uint)Leading; ++s) {
        acc[s] = 0.0f;
    }
    for (uint k = 0; k < (uint)InputDim; ++k) {
        float w = static_cast<float>(weight_t[k * (uint)OutDim + n]);
        for (uint s = 0; s < (uint)Leading; ++s) {
            acc[s] = fma(w, static_cast<float>(x[s * (uint)InputDim + k]), acc[s]);
        }
    }
    for (uint s = 0; s < (uint)Leading; ++s) {
        out[s * (uint)OutDim + n] = static_cast<OutT>(acc[s]);
    }
"#;

/// bf16 weight, `Cols` adjacent output columns per thread, loaded as one
/// 8-byte (`uint2`, Cols=4) or 16-byte (`uint4`, Cols=8) word per `k`.
/// Each column keeps the scalar form's arithmetic exactly: sequential `k`,
/// one f32 `fma` per (row, column), bf16 widened by bit extension (which is
/// what `static_cast<float>(bfloat16_t)` performs), so the output is
/// bit-identical to `DENSE_WIDE_GEMV_SOURCE` and rows stay
/// `Leading`-invariant. Requires `OutDim % Cols == 0` and a word-aligned
/// weight buffer. The kernel wrapper row-contiguates inputs, so the only
/// views that reach the kernel uncopied are row-contiguous ones; with
/// `OutDim % Cols == 0` a row offset is a whole number of words, and the
/// production `decode_weight_t` is a fresh allocation. A row-contiguous
/// view carved out of a flattened buffer at an odd element offset is the
/// one shape this cannot see (no caller builds one; documented rather
/// than probed, since the C API exposes no device-address query).
fn dense_wide_gemv_bf16_vec_source(cols: usize) -> String {
    debug_assert!(matches!(cols, 4 | 8), "vector tile must be 4 or 8 columns");
    let (word, lanes) = match cols {
        4 => ("uint2", ["x", "y", "", ""]),
        _ => ("uint4", ["x", "y", "z", "w"]),
    };
    let mut unpack = String::new();
    for c in 0..cols {
        let lane = lanes[c / 2];
        if c % 2 == 0 {
            unpack.push_str(&format!(
                "        w[{c}] = as_type<float>(raw.{lane} << 16);\n"
            ));
        } else {
            unpack.push_str(&format!(
                "        w[{c}] = as_type<float>(raw.{lane} & 0xffff0000u);\n"
            ));
        }
    }
    format!(
        r#"
    uint n0 = thread_position_in_grid.x * {cols}u;
    if (n0 >= (uint)OutDim) {{
        return;
    }}
    float acc[Leading][{cols}];
    for (uint s = 0; s < (uint)Leading; ++s) {{
        for (uint c = 0; c < {cols}u; ++c) {{
            acc[s][c] = 0.0f;
        }}
    }}
    device const {word}* wv = (device const {word}*)(weight_t);
    const uint words_per_row = (uint)OutDim / {cols}u;
    const uint word_col = n0 / {cols}u;
    for (uint k = 0; k < (uint)InputDim; ++k) {{
        {word} raw = wv[k * words_per_row + word_col];
        float w[{cols}];
{unpack}        for (uint s = 0; s < (uint)Leading; ++s) {{
            float xv = static_cast<float>(x[s * (uint)InputDim + k]);
            for (uint c = 0; c < {cols}u; ++c) {{
                acc[s][c] = fma(w[c], xv, acc[s][c]);
            }}
        }}
    }}
    for (uint s = 0; s < (uint)Leading; ++s) {{
        for (uint c = 0; c < {cols}u; ++c) {{
            out[s * (uint)OutDim + n0 + c] = static_cast<OutT>(acc[s][c]);
        }}
    }}
"#
    )
}

/// Widest row count the 8-column tile serves: `Leading * 8` f32
/// accumulators live in registers, and the kernel spills past 48 of them
/// (M5 Max, 2026-09-21, 2.54 GB head: S=6 4.7 ms, S=7 17.1 ms, S=8
/// 17.4 ms with 8 columns, against 5.2-5.7 ms for 4 columns and ~8 ms
/// scalar). Wider rows take the 4-column tile.
const DENSE_WIDE_GEMV_VEC8_MAX_LEADING: i32 = 6;

/// `AX_MLX_DENSE_WIDE_GEMV_VEC8_MAX_LEADING` — override the 8-column row
/// ceiling for a device whose register budget differs from the M5 Max
/// measurement (0 disables the 8-column tile). Read once.
fn dense_wide_gemv_vec8_max_leading() -> i32 {
    static CEILING: OnceLock<i32> = OnceLock::new();
    *CEILING.get_or_init(|| {
        std::env::var("AX_MLX_DENSE_WIDE_GEMV_VEC8_MAX_LEADING")
            .ok()
            .and_then(|value| value.trim().parse::<i32>().ok())
            .filter(|value| (0..=8).contains(value))
            .unwrap_or(DENSE_WIDE_GEMV_VEC8_MAX_LEADING)
    })
}

/// Columns per thread the dense wide GEMV tiles for this weight and row
/// count: 8 or 4 for bf16 weights whose column count divides evenly (8 only
/// while the accumulator set fits in registers), otherwise the scalar form.
fn dense_wide_gemv_cols(weight_dtype: MlxDtype, out_dim: i32, leading: i32) -> i32 {
    if weight_dtype != MlxDtype::Bfloat16 {
        return 1;
    }
    if out_dim % 8 == 0 && leading <= dense_wide_gemv_vec8_max_leading() {
        8
    } else if out_dim % 4 == 0 {
        4
    } else {
        1
    }
}

/// Product of every leading dimension of `x` (the row count `Leading`).
fn dense_wide_gemv_leading(x_shape: &[i32]) -> Option<i32> {
    x_shape[..x_shape.len().saturating_sub(1)]
        .iter()
        .try_fold(1_i32, |product, dimension| product.checked_mul(*dimension))
}

/// `x [.., S, in] @ weight_t [in, out]` for `S ∈ 1..=8` without the steel
/// GEMM's per-row weight re-read — and, crucially, without materializing a
/// contiguous `[out, in]` copy of a `decode_weight_t`-prepared head whose
/// `weight` field is only a lazy transpose view. One arithmetic for every
/// `S`, so exact-profile MTP-off singles and S=2..4 verify stay mutually
/// consistent. See `AX_MLX_DENSE_WIDE_GEMV` (non-exact opt-in hook) and
/// `AX_MLX_EXACT_DENSE_WEIGHT_T_GEMV` (exact-profile routing).
pub(crate) fn dense_wide_gemv_weight_t(x: &MlxArray, weight_t: &MlxArray) -> Option<MlxArray> {
    let leading = dense_wide_gemv_leading(&x.shape())?;
    let cols = dense_wide_gemv_cols(weight_t.dtype(), weight_t.shape().get(1).copied()?, leading);
    dense_wide_gemv_weight_t_with_cols(x, weight_t, cols)
}

/// `dense_wide_gemv_weight_t` with an explicit column tile (1, 4 or 8). The
/// public entry picks the widest tile the weight admits; tests pin the
/// vector tiles against the scalar form.
pub(crate) fn dense_wide_gemv_weight_t_with_cols(
    x: &MlxArray,
    weight_t: &MlxArray,
    cols: i32,
) -> Option<MlxArray> {
    if !matches!(
        x.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        weight_t.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let weight_shape = weight_t.shape();
    if weight_shape.len() != 2 {
        return None;
    }
    let input_dim = weight_shape[0];
    let out_dim = weight_shape[1];
    if input_dim <= 0 || out_dim <= 0 {
        return None;
    }
    let x_shape = x.shape();
    if x_shape.last().copied() != Some(input_dim) {
        return None;
    }
    let leading = dense_wide_gemv_leading(&x_shape)?;
    if !(1..=8).contains(&leading) {
        return None;
    }
    let kernel = match cols {
        8 if weight_t.dtype() == MlxDtype::Bfloat16 && out_dim % 8 == 0 => {
            DENSE_WIDE_GEMV_BF16_VEC8_KERNEL.get_or_init(|| {
                MlxMetalKernel::new(
                    "ax_dense_wide_gemv_wt_bf16_c8_v1",
                    &["x", "weight_t"],
                    &["out"],
                    &dense_wide_gemv_bf16_vec_source(8),
                    "",
                    true,
                )
            })
        }
        4 if weight_t.dtype() == MlxDtype::Bfloat16 && out_dim % 4 == 0 => {
            DENSE_WIDE_GEMV_BF16_VEC4_KERNEL.get_or_init(|| {
                MlxMetalKernel::new(
                    "ax_dense_wide_gemv_wt_bf16_c4_v1",
                    &["x", "weight_t"],
                    &["out"],
                    &dense_wide_gemv_bf16_vec_source(4),
                    "",
                    true,
                )
            })
        }
        1 => DENSE_WIDE_GEMV_KERNEL.get_or_init(|| {
            MlxMetalKernel::new(
                "ax_dense_wide_gemv_wt_v1",
                &["x", "weight_t"],
                &["out"],
                DENSE_WIDE_GEMV_SOURCE,
                "",
                true,
            )
        }),
        _ => return None,
    };
    let x_flat = reshape(x, &[leading, input_dim], None);
    let threads = out_dim.checked_add(cols - 1)?.checked_div(cols)?;
    let grid_x = threads
        .checked_add(255)?
        .checked_div(256)?
        .checked_mul(256)?;
    let mut outputs = kernel
        .try_apply_with_template(
            &[&x_flat, weight_t],
            &[KernelOutputSpec {
                shape: vec![leading, out_dim],
                dtype: x.dtype(),
            }],
            &[
                KernelTemplateArg::Dtype {
                    name: "OutT",
                    dtype: x.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "OutDim",
                    value: out_dim,
                },
                KernelTemplateArg::Int {
                    name: "InputDim",
                    value: input_dim,
                },
                KernelTemplateArg::Int {
                    name: "Leading",
                    value: leading,
                },
            ],
            (grid_x, 1, 1),
            (256, 1, 1),
            None,
        )
        .ok()?;
    let flat = outputs.pop()?;
    let mut out_shape = x_shape;
    *out_shape.last_mut()? = out_dim;
    Some(reshape(&flat, &out_shape, None))
}

fn qw_direct(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    // Dense Linear bias (`QuantizedWeight.linear_bias`) is separate from affine
    // group-quant biases (`qw.biases`). Matches mlx-lm `nn.Linear` /
    // `QuantizedLinear`: y = x @ W + b. Required for GPT-OSS Q/K/V/O + router.
    let y = if fastpath::qwen_linear_mtp_exact_enabled()
        && let Some(invariant) = invariant_projection_metal_impl(x, qw)
    {
        invariant
    } else if let Some(verify) = super::verify_qmm::try_qwen_mtp_verify_qmm(x, qw) {
        verify
    } else if qwen_prefill_dequant_dense_applies(x)
        && let Some(weight_t) = cached_prefill_dequant_weight_t(qw)
    {
        matmul(x, &weight_t, None)
    } else if let Some(scales) = &qw.scales {
        // MXFP8/MXFP4 have no affine group-bias channel; pass None for those modes.
        let mode = qw.mlx_quantization_mode();
        let quant_biases = match mode {
            mlx_sys::MlxQuantizationMode::Affine => qw.biases.as_ref(),
            _ => None,
        };
        qwen_prefill_maybe_tile_qmm(x, |tiled| {
            qwen_prefill_maybe_flat_qmm(tiled, |flat| {
                exact_mxfp4_short_qmm(flat, mode, |row_major| {
                    mlx_sys::quantized_matmul_with_mode(
                        row_major,
                        &qw.weight,
                        scales,
                        quant_biases,
                        true,
                        Some(qw.group_size),
                        Some(qw.bits),
                        mode,
                        None,
                    )
                })
            })
        })
    } else if decode_lm_head_quant_cache_eligible(x)
        && let (Some(q_w), Some(q_s), Some(q_b)) = (
            qw.decode_q2_weight.as_ref(),
            qw.decode_q2_scales.as_ref(),
            qw.decode_q2_biases.as_ref(),
        )
    {
        mlx_sys::quantized_matmul_with_mode(
            x,
            q_w,
            q_s,
            Some(q_b),
            true,
            Some(DECODE_LM_HEAD_QUANT_GROUP_SIZE),
            Some(DECODE_LM_HEAD_QUANT_BITS),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
        )
    } else if let Some(weight_t) = &qw.decode_weight_t {
        if fastpath::dense_wide_gemv_enabled()
            && let Some(wide) = dense_wide_gemv_weight_t(x, weight_t)
        {
            wide
        } else {
            matmul(x, weight_t, None)
        }
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

static UNQUANTIZED_DECODE_PROJECTION_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static UNQUANTIZED_DECODE_PROJECTION_HITS: AtomicU64 = AtomicU64::new(0);

/// Metal GEMV for a single-token dense projection. Reads `weight` as
/// `[out, in]` in place — no `[out, in] → [in, out]` transpose buffer.
/// Eight output rows per threadgroup via simdgroup_matrix (phase1 sg_bf16).
const UNQUANTIZED_DECODE_PROJECTION_SOURCE: &str = r#"
    constexpr uint Tile = 8;
    uint tg = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint row_base = tg * Tile;
    if (row_base >= (uint)OutDim) {
        return;
    }
    const uint N = (uint)InputDim;
    if (row_base + Tile > (uint)OutDim) {
        if (lane < Tile) {
            uint row = row_base + lane;
            if (row < (uint)OutDim) {
                float val = 0.0f;
                uint row_kbase = row * N;
                for (uint c = 0; c < N; ++c) {
                    val = fma(static_cast<float>(weight[row_kbase + c]),
                              static_cast<float>(x[c]), val);
                }
                out[row] = static_cast<OutT>(val);
            }
        }
        return;
    }
    simdgroup_matrix<float, 8, 8> acc;
    acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint k = 0;
    for (; k + Tile <= N; k += Tile) {
        simdgroup_matrix<OutT, 8, 8> w_tile;
        simdgroup_matrix<float, 8, 8> h_tile;
        simdgroup_load(w_tile, weight + row_base * N + k, (ulong)N, ulong2(0, 0), false);
        simdgroup_load(h_tile, x + k, 0, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc, w_tile, h_tile, acc);
    }
    threadgroup float out_buf[Tile * Tile];
    simdgroup_store(acc, out_buf, Tile, ulong2(0, 0));
    if (lane < Tile) {
        uint row = row_base + lane;
        float val = out_buf[lane * Tile];
        uint row_kbase = row * N;
        for (uint c = k; c < N; ++c) {
            val = fma(static_cast<float>(weight[row_kbase + c]),
                      static_cast<float>(x[c]), val);
        }
        out[row] = static_cast<OutT>(val);
    }
"#;

/// Decode-only unquantized projection: `logits = x @ weight.T` without
/// materializing `weight.T`. `weight` is `[out, in]`, `x` is rank ≥ 1
/// with last dim `in` and all other dims product 1.
///
/// Wired decode remasure on AXQ 27B (df-macbookpro-m5): 29.47 vs 28.78
/// (1.024×), slower than `qw` 30.20. Production stays on `qw`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn project_unquantized_decode(x: &MlxArray, weight: &MlxArray) -> Option<MlxArray> {
    if !matches!(
        x.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || x.dtype() != weight.dtype()
    {
        return None;
    }
    let weight_shape = weight.shape();
    if weight_shape.len() != 2 {
        return None;
    }
    let out_dim = weight_shape[0];
    let input_dim = weight_shape[1];
    if out_dim <= 0 || input_dim <= 0 {
        return None;
    }
    let x_shape = x.shape();
    if x_shape.last().copied() != Some(input_dim) {
        return None;
    }
    let leading = x_shape[..x_shape.len() - 1]
        .iter()
        .try_fold(1_i32, |product, dimension| product.checked_mul(*dimension))?;
    if leading != 1 {
        return None;
    }
    let tiles = out_dim.saturating_add(7) / 8;
    let grid_x = tiles.checked_mul(32)?;
    let x_flat = reshape(x, &[input_dim], None);
    // simdgroup_load into a float tile requires a float source (phase1
    // decode_logits_projection_sg_*). Hidden is 5k elements — not a
    // 2.54 GB weight transpose.
    let x_f32 = if x.dtype() == MlxDtype::Float32 {
        x_flat
    } else {
        astype(&x_flat, MlxDtype::Float32, None)
    };
    let kernel = UNQUANTIZED_DECODE_PROJECTION_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_unquantized_decode_projection_sg_v2",
            &["x", "weight"],
            &["out"],
            UNQUANTIZED_DECODE_PROJECTION_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel
        .try_apply_with_template(
            &[&x_f32, weight],
            &[KernelOutputSpec {
                shape: vec![out_dim],
                dtype: x.dtype(),
            }],
            &[
                KernelTemplateArg::Dtype {
                    name: "OutT",
                    dtype: x.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "OutDim",
                    value: out_dim,
                },
                KernelTemplateArg::Int {
                    name: "InputDim",
                    value: input_dim,
                },
            ],
            (grid_x, 1, 1),
            (32, 1, 1),
            None,
        )
        .ok()?;
    let flat = outputs.pop()?;
    let mut out_shape = x_shape;
    *out_shape.last_mut()? = out_dim;
    Some(reshape(&flat, &out_shape, None))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn unquantized_decode_projection_hits() -> u64 {
    UNQUANTIZED_DECODE_PROJECTION_HITS.load(Ordering::Relaxed)
}

/// Unquantized decode `lm_head` entry: no-copy GEMV, else `qw`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn project_lm_head(x: &MlxArray, lm_head: &QuantizedWeight) -> MlxArray {
    if lm_head.scales.is_none()
        && let Some(y) = project_unquantized_decode(x, &lm_head.weight)
    {
        UNQUANTIZED_DECODE_PROJECTION_HITS.fetch_add(1, Ordering::Relaxed);
        return if let Some(bias) = &lm_head.linear_bias {
            add(&y, bias, None)
        } else {
            y
        };
    }
    qw(x, lm_head)
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

fn invariant_mxfp4_qmv_fast_impl(x: &MlxArray, qw: &QuantizedWeight) -> Option<MlxArray> {
    if !matches!(
        qw.mlx_quantization_mode(),
        mlx_sys::MlxQuantizationMode::Mxfp4
    ) || qw.bits != 4
        || qw.group_size != 32
        || qw.biases.is_some()
    {
        return None;
    }
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
    if out_dim <= 0 || out_dim % 8 != 0 || input_dim % 512 != 0 {
        return None;
    }
    let packed_cols = input_dim / 8;
    if weight_shape[1] != packed_cols {
        return None;
    }
    let scales = qw.scales.as_ref()?;
    let group_count = input_dim / 32;
    if scales.shape() != vec![out_dim, group_count] {
        return None;
    }

    // Production activations after residual/concat are often non-contiguous.
    // The kernel addresses x as packed [Leading, InputDim].
    let x_contig = contiguous(x, None);
    let mut out_shape = x_shape;
    *out_shape.last_mut()? = out_dim;
    let kernel = INVARIANT_MXFP4_QMV_FAST_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_invariant_mxfp4_qmv_fast_v1",
            &["x", "weight", "scales"],
            &["out"],
            INVARIANT_MXFP4_QMV_FAST_KERNEL_SOURCE,
            "",
            true,
        )
    });
    kernel
        .try_apply_with_template(
            &[&x_contig, &qw.weight, scales],
            &[KernelOutputSpec {
                shape: out_shape,
                dtype: x.dtype(),
            }],
            &[
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
                KernelTemplateArg::Int {
                    name: "GroupSize",
                    value: 32,
                },
                KernelTemplateArg::Int {
                    name: "GroupCount",
                    value: group_count,
                },
                KernelTemplateArg::Int {
                    name: "PackedCols",
                    value: packed_cols,
                },
            ],
            (32, (out_dim / 8).saturating_mul(2), 1),
            (32, 2, 1),
            None,
        )
        .ok()?
        .pop()
}

/// `AX_MLX_INVARIANT_QMV_TRACE=1`: log each distinct (leading, K, N, bits,
/// group size, dtype) shape the invariant affine qmv_fast route launches, once,
/// so a profile can tell which projections actually reach this kernel.
fn trace_invariant_qmv_shape(
    leading: i32,
    input_dim: i32,
    out_dim: i32,
    bits: i32,
    group_size: i32,
    dtype: MlxDtype,
) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| crate::fastpath::env_flag("AX_MLX_INVARIANT_QMV_TRACE")) {
        return;
    }
    type ShapeKey = (i32, i32, i32, i32, i32, MlxDtype);
    static SEEN: std::sync::Mutex<Vec<ShapeKey>> = std::sync::Mutex::new(Vec::new());
    let key: ShapeKey = (leading, input_dim, out_dim, bits, group_size, dtype);
    if let Ok(mut seen) = SEEN.lock()
        && !seen.contains(&key)
    {
        seen.push(key);
        tracing::info!(
            target: "ax_engine_mlx::invariant_qmv",
            leading,
            input_dim,
            out_dim,
            bits,
            group_size,
            ?dtype,
            "invariant affine qmv_fast shape"
        );
    }
}

/// The bf16 / 4-bit specialisation of the invariant affine qmv_fast kernel
/// is bit-identical to the generic kernel; `AX_MLX_INVARIANT_QMV_BF16_Q4=0`
/// is the kill switch back to the generic form.
fn invariant_affine_bf16_q4_kernel_selected(input_dtype: MlxDtype, bits: i32) -> bool {
    input_dtype == MlxDtype::Bfloat16 && bits == 4 && fastpath::invariant_qmv_bf16_q4_enabled()
}

fn invariant_projection_metal_impl(x: &MlxArray, qw: &QuantizedWeight) -> Option<MlxArray> {
    invariant_projection_metal_impl_with_kernel(
        x,
        qw,
        invariant_affine_bf16_q4_kernel_selected(x.dtype(), qw.bits),
    )
}

/// `invariant_projection_metal_impl` with the affine qmv_fast kernel choice
/// made explicit (`bf16_q4`: the bf16 / 4-bit specialisation, which is only
/// valid for bf16 inputs and 4-bit weights). Tests pin both kernels against
/// each other through this entry.
fn invariant_projection_metal_impl_with_kernel(
    x: &MlxArray,
    qw: &QuantizedWeight,
    bf16_q4: bool,
) -> Option<MlxArray> {
    // S=1 stays on MLX `fp_qmv_fast` so MTP-off and exact singleton steps
    // share one kernel. S=2 verify uses the microbatch that matches that
    // singleton arithmetic while reading weights once.
    let verify_leading = x.shape()[..x.shape().len().saturating_sub(1)]
        .iter()
        .try_fold(1_i32, |product, dimension| product.checked_mul(*dimension))
        .unwrap_or(0);
    if verify_leading == 2
        && fastpath::invariant_mxfp4_qmv_fast_enabled()
        && let Some(out) = invariant_mxfp4_qmv_fast_impl(x, qw)
    {
        return Some(out);
    }
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

    // Split dimensions that are not aligned to one complete qmv_fast lane
    // block: 512 values for 4-bit, 256 for 6/8-bit. Both pure-direct and
    // multi-token use this split so A/B identity holds. Other bit widths
    // never reach the custom qmv (`values_per_thread == 0` below), so the
    // split would only trade one MLX qmm for two plus an add — skip it.
    let qmv_block_size = match qw.bits {
        4 => 512,
        6 | 8 => 256,
        _ => 0,
    };
    if qmv_block_size > 0
        && matches!(
            qw.mlx_quantization_mode(),
            mlx_sys::MlxQuantizationMode::Affine
        )
        && qw.bits > 0
        && qw.bits <= 8
        && qw.group_size > 0
        && input_dim % qmv_block_size != 0
        && input_dim > qmv_block_size
        && input_dim % qw.group_size == 0
        && (input_dim * qw.bits) % 32 == 0
    {
        let aligned = (input_dim / qmv_block_size) * qmv_block_size;
        let rem = input_dim - aligned;
        if aligned > 0 && rem > 0 && rem % qw.group_size == 0 && (rem * qw.bits) % 32 == 0 {
            let packed_al = aligned * qw.bits / 32;
            let packed_rem = rem * qw.bits / 32;
            let groups_al = aligned / qw.group_size;
            let groups_rem = rem / qw.group_size;
            if weight_shape[1] == packed_al + packed_rem
                && let (Some(scales), Some(biases)) = (qw.scales.as_ref(), qw.biases.as_ref())
                && scales.shape() == [out_dim, input_dim / qw.group_size]
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
                    decode_weight_t: None,
                    decode_q2_weight: None,
                    decode_q2_scales: None,
                    decode_q2_biases: None,
                };
                let qw_rem = QuantizedWeight {
                    weight: w_rem,
                    scales: Some(s_rem),
                    biases: Some(b_rem),
                    group_size: qw.group_size,
                    bits: qw.bits,
                    mode: qw.mode.clone(),
                    linear_bias: None,
                    decode_weight_t: None,
                    decode_q2_weight: None,
                    decode_q2_scales: None,
                    decode_q2_biases: None,
                };
                // The aligned prefix hits qmv_fast.
                let y_al = invariant_projection_metal_impl_with_kernel(&x_al, &qw_al, bf16_q4)?;
                // Remainder: MLX singleton (Leading=1) or RowExact MLX
                // (Leading>1) so multi-token matches pure-direct.
                let y_rem = if leading > 1 {
                    let x_rem_shape = x_rem.shape();
                    // x_rem is [1, S, rem] or similar with leading product S.
                    let seq = x_rem_shape[x_rem_shape.len() - 2];
                    let cols: Vec<MlxArray> = (0..seq)
                        .map(|t| {
                            let ndim = x_rem_shape.len();
                            let mut starts = vec![0_i32; ndim];
                            let mut ends: Vec<i32> = x_rem_shape.to_vec();
                            starts[ndim - 2] = t;
                            ends[ndim - 2] = t + 1;
                            let strides = vec![1_i32; ndim];
                            let row =
                                contiguous(&slice(&x_rem, &starts, &ends, &strides, None), None);
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
        let block_size = values_per_thread * 32;
        let qmv_fast_eligible = values_per_thread > 0
            && out_dim % 8 == 0
            && input_dim % block_size == 0
            && qw.group_size >= values_per_thread
            && qw.group_size % values_per_thread == 0;
        if qmv_fast_eligible {
            trace_invariant_qmv_shape(
                leading,
                input_dim,
                out_dim,
                qw.bits,
                qw.group_size,
                x.dtype(),
            );
            let launch = |kernel: &MlxMetalKernel| {
                kernel
                    .try_apply_with_template(
                        &[x, &qw.weight, scales, biases],
                        &[KernelOutputSpec {
                            shape: out_shape.clone(),
                            dtype: output_dtype,
                        }],
                        &common_template_args,
                        (32, (out_dim / 8).saturating_mul(2), 1),
                        (32, 2, 1),
                        None,
                    )
                    .ok()
            };
            // The specialisation is bit-identical to the generic kernel, so a
            // launch failure (e.g. a Metal toolchain without native `bfloat`)
            // falls back to the generic kernel rather than leaving the
            // invariant route.
            let specialised = (bf16_q4 && x.dtype() == MlxDtype::Bfloat16 && qw.bits == 4)
                .then(|| {
                    launch(INVARIANT_AFFINE_QMV_FAST_BF16_Q4_KERNEL.get_or_init(|| {
                        MlxMetalKernel::new(
                            "ax_invariant_affine_qmv_fast_bf16_q4_v1",
                            &["x", "weight", "scales", "biases"],
                            &["out"],
                            INVARIANT_AFFINE_QMV_FAST_BF16_Q4_KERNEL_SOURCE,
                            "",
                            true,
                        )
                    }))
                })
                .flatten();
            match specialised {
                Some(out) => out,
                None => launch(INVARIANT_AFFINE_QMV_FAST_KERNEL.get_or_init(|| {
                    MlxMetalKernel::new(
                        "ax_invariant_affine_qmv_fast_v1",
                        &["x", "weight", "scales", "biases"],
                        &["out"],
                        INVARIANT_AFFINE_QMV_FAST_KERNEL_SOURCE,
                        "",
                        true,
                    )
                }))?,
            }
        } else {
            // Non-fast custom kernel does not match MLX for Gemma-like shapes.
            // Use MLX quantized_matmul (bitexact for Leading=1; RowExact for S>1).
            let shape = x.shape();
            let out = if shape.len() == 3 && shape[0] == 1 && shape[1] > 1 {
                let cols: Vec<MlxArray> = (0..shape[1])
                    .map(|t| {
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
        // A `decode_weight_t`-prepared head only exposes `weight` as a lazy
        // transpose view; feeding that to the kernel below re-materializes
        // the full contiguous head every call (2.54 GB/step on Qwen3.8-27B
        // — the 6bit-MTP 0.80× regression). Read the `[in, out]` buffer
        // directly instead; one arithmetic covers S=1..8 so MTP-off and
        // verify stay mutually consistent.
        if fastpath::exact_dense_weight_t_gemv_enabled()
            && let Some(weight_t) = qw.decode_weight_t.as_ref()
            && let Some(out) = dense_wide_gemv_weight_t(x, weight_t)
        {
            return Some(out);
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
    // Muse Glimmer `output_multiplier`: scale logits BEFORE the softcap
    // (reference: `logits * output_multiplier`, then `tanh(x * (1/cap)) * cap`).
    let scaled_logits;
    let logits = if let Some(scale) = cfg.final_logits_scale.filter(|s| *s != 1.0) {
        let scale_arr = mlx_sys::ops::cached_scalar(scale, logits.dtype());
        scaled_logits = multiply(logits, &scale_arr, None);
        &scaled_logits
    } else {
        logits
    };
    let Some(cap) = cfg.final_logit_softcapping.filter(|cap| *cap > 0.0) else {
        return logits.clone();
    };
    let inv_cap = 1.0_f32 / cap;
    let inv_cap_arr = mlx_sys::ops::cached_scalar(inv_cap, logits.dtype());
    let cap_arr = mlx_sys::ops::cached_scalar(cap, logits.dtype());
    let scaled = multiply(logits, &inv_cap_arr, None);
    multiply(&tanh(&scaled, None), &cap_arr, None)
}

/// Weightless RMSNorm on token embeddings (Muse Glimmer `embed_norm`).
/// No-op for families without `embed_norm_no_weight`.
pub(crate) fn maybe_weightless_embed_norm(cfg: &ModelConfig, hidden: MlxArray) -> MlxArray {
    if cfg.embed_norm_no_weight {
        mlx_sys::rms_norm(&hidden, None, cfg.rms_norm_eps, None)
    } else {
        hidden
    }
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

static GATHER_QMV_WIDE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

/// Segmented multi-row gather matvec over sorted expert selections.
///
/// MLX has no `gather_qmv_wide`: at small M each selection row re-reads its
/// expert's full weight matrix, so multi-token MoE verify scales linearly
/// with stacked rows. This kernel walks host-computed segments of sorted
/// selections (rows sharing one expert, capped at `MaxRows = 8` like
/// mlxcel's CUDA multirow qmv), dequantizes each weight pack once, and
/// FMAs it against every row in the segment. One simdgroup per
/// (segment, output row); f32 accumulators + `simd_sum` per row.
///
/// Spike only — measured and NOT wired (`project_unquantized_decode`
/// precedent); 4-bit affine, `x` pre-sorted `[rows, in]`.
///
/// Verdict (M3 Max, release, 32 rows over 8 of 64 experts, 2048→768):
/// kernel 498µs vs `gather_qmm` 230–249µs — **0.46–0.50×**. The naive
/// one-simdgroup-per-output-row loop loses more to MLX's tiled
/// `qmv_fast` (per-thread value tiling, vectorized loads) than the ≤8-row
/// weight reuse wins back. A competitive version must first match the
/// `qmv_fast` structure per row — same class of effort that left the
/// invariant MXFP4 microbatch clone at 0.84×. Parity holds (1e-4).
const GATHER_QMV_WIDE_SOURCE: &str = r#"
    uint flat = thread_position_in_grid.x;
    uint seg = thread_position_in_grid.y;
    uint row = flat / 32;
    uint lane = flat % 32;
    if (row >= (uint)OutDim || seg >= (uint)SegCount) {
        return;
    }
    uint expert = seg_expert[seg];
    uint start = seg_start[seg];
    uint cnt = seg_count[seg];
    float acc[8];
    for (uint r = 0; r < 8u; ++r) {
        acc[r] = 0.0f;
    }
    const uint row_base = (expert * (uint)OutDim + row) * (uint)PackedCols;
    const uint scale_row = (expert * (uint)OutDim + row) * (uint)GroupCount;
    for (uint packed_col = lane; packed_col < (uint)PackedCols; packed_col += 32) {
        uint packed = weight[row_base + packed_col];
        for (uint packed_lane = 0; packed_lane < (uint)PackFactor; ++packed_lane) {
            uint input_col = packed_col * (uint)PackFactor + packed_lane;
            uint q = (packed >> (packed_lane * (uint)Bits)) & (uint)QuantMask;
            uint scale_idx = scale_row + input_col / (uint)GroupSize;
            float w = static_cast<float>(q) * static_cast<float>(scales[scale_idx])
                + static_cast<float>(biases[scale_idx]);
            for (uint r = 0; r < cnt; ++r) {
                acc[r] = fma(
                    w,
                    static_cast<float>(x[(start + r) * (uint)InputDim + input_col]),
                    acc[r]);
            }
        }
    }
    for (uint r = 0; r < cnt; ++r) {
        float total = simd_sum(acc[r]);
        if (lane == 0) {
            out[(start + r) * (uint)OutDim + row] = static_cast<OutT>(total);
        }
    }
"#;

/// Build ≤8-row segments from sorted flat expert ids.
#[cfg_attr(not(test), allow(dead_code))]
fn gather_wide_segments(sorted_expert_ids: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut expert = Vec::new();
    let mut start = Vec::new();
    let mut count = Vec::new();
    let mut i = 0usize;
    while i < sorted_expert_ids.len() {
        let e = sorted_expert_ids[i];
        let mut j = i + 1;
        while j < sorted_expert_ids.len() && sorted_expert_ids[j] == e && j - i < 8 {
            j += 1;
        }
        expert.push(e);
        start.push(i as u32);
        count.push((j - i) as u32);
        i = j;
    }
    (expert, start, count)
}

/// `x_sorted [rows, in]` against `exps [E, out, packed]` (4-bit affine),
/// selections pre-sorted by expert id. Returns `[rows, out]`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn gather_qmv_wide_sorted(
    x_sorted: &MlxArray,
    exps: &QuantizedWeight,
    sorted_expert_ids: &[u32],
) -> Option<MlxArray> {
    if !matches!(
        x_sorted.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let (Some(scales), Some(biases)) = (exps.scales.as_ref(), exps.biases.as_ref()) else {
        return None;
    };
    if exps.bits != 4 || exps.group_size <= 0 {
        return None;
    }
    let weight_shape = exps.weight.shape();
    if weight_shape.len() != 3 {
        return None;
    }
    let (num_experts, out_dim, packed_cols) = (weight_shape[0], weight_shape[1], weight_shape[2]);
    let pack_factor = 32 / exps.bits;
    let input_dim = packed_cols.checked_mul(pack_factor)?;
    if input_dim % exps.group_size != 0 {
        return None;
    }
    let group_count = input_dim / exps.group_size;
    let x_shape = x_sorted.shape();
    if x_shape != vec![sorted_expert_ids.len() as i32, input_dim] {
        return None;
    }
    if sorted_expert_ids.iter().any(|&e| e as i32 >= num_experts) {
        return None;
    }
    let (seg_expert, seg_start, seg_count) = gather_wide_segments(sorted_expert_ids);
    let seg_len = seg_expert.len() as i32;
    if seg_len == 0 {
        return None;
    }
    let u32_arr = |data: &[u32]| {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            &[data.len() as i32],
            MlxDtype::Uint32,
        )
    };
    let seg_expert_arr = u32_arr(&seg_expert);
    let seg_start_arr = u32_arr(&seg_start);
    let seg_count_arr = u32_arr(&seg_count);
    let kernel = GATHER_QMV_WIDE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_gather_qmv_wide_v1",
            &[
                "x",
                "weight",
                "scales",
                "biases",
                "seg_expert",
                "seg_start",
                "seg_count",
            ],
            &["out"],
            GATHER_QMV_WIDE_SOURCE,
            "",
            true,
        )
    });
    let rows = sorted_expert_ids.len() as i32;
    let mut outputs = kernel
        .try_apply_with_template(
            &[
                x_sorted,
                &exps.weight,
                scales,
                biases,
                &seg_expert_arr,
                &seg_start_arr,
                &seg_count_arr,
            ],
            &[KernelOutputSpec {
                shape: vec![rows, out_dim],
                dtype: x_sorted.dtype(),
            }],
            &[
                KernelTemplateArg::Dtype {
                    name: "OutT",
                    dtype: x_sorted.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "OutDim",
                    value: out_dim,
                },
                KernelTemplateArg::Int {
                    name: "InputDim",
                    value: input_dim,
                },
                KernelTemplateArg::Int {
                    name: "PackedCols",
                    value: packed_cols,
                },
                KernelTemplateArg::Int {
                    name: "GroupSize",
                    value: exps.group_size,
                },
                KernelTemplateArg::Int {
                    name: "GroupCount",
                    value: group_count,
                },
                KernelTemplateArg::Int {
                    name: "Bits",
                    value: exps.bits,
                },
                KernelTemplateArg::Int {
                    name: "PackFactor",
                    value: pack_factor,
                },
                KernelTemplateArg::Int {
                    name: "QuantMask",
                    value: (1 << exps.bits) - 1,
                },
                KernelTemplateArg::Int {
                    name: "SegCount",
                    value: seg_len,
                },
            ],
            (out_dim.saturating_mul(32), seg_len, 1),
            (32, 1, 1),
            None,
        )
        .ok()?;
    outputs.pop()
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
mod tests;
