// SPDX-License-Identifier: Apache-2.0
//
// The split-K and multi-simdgroup morphologies are adapted from MTPLX's
// verify kernels via oMLX 0.6.2 (`qwen35_verify_qmm.py`). MTPLX and oMLX
// publish that implementation under Apache-2.0. AX's integration, template
// specialization, fail-closed routing, and runner-scoped guard are Rust-side.

//! Verify-shape affine QMM for Qwen Lightning MTP.
//!
//! MLX's general QMM is excellent at M=1 and large prefill matrices, but a
//! speculative verifier repeatedly presents M=3..6. This route reads each
//! large projection once for the skinny row batch. It is armed only around a
//! target-model verify forward and remains opt-in until model-level output and
//! throughput admission pass on Apple Silicon.

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, add, concatenate,
    contiguous, reshape, slice, zeros,
};
use std::cell::Cell;
use std::sync::OnceLock;

use crate::weights::QuantizedWeight;

// Match oMLX's measured dispatch floor: below 16K outputs the extra custom
// dispatches cost more than they save, while Qwen's wide gate/up projections
// have enough K-reduction work to benefit from split-K occupancy.
const MIN_ROUTE_N: i32 = 16_384;
#[cfg(test)]
const SPLIT_K_TEST_N: i32 = 16_384;
const MSG_ROUTE_N: i32 = 100_000;
const DEFAULT_MSG_SIMDGROUPS: i32 = 8;

/// Minimum output width routed through the verify QMM.
///
/// oMLX keeps a 16K floor because its process-wide Python method patch pays a
/// call/dispatch tax on every projection. AX calls the kernel directly from
/// Rust, so M5 admission can safely sweep the layer-projection range without
/// recompiling. Invalid values retain the conservative oMLX-derived default.
fn min_route_n() -> i32 {
    static CACHED: OnceLock<i32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_VERIFY_QMM_MIN_N")
            .ok()
            .and_then(|raw| raw.trim().parse::<i32>().ok())
            .filter(|value| *value >= 0)
            .unwrap_or(MIN_ROUTE_N)
    })
}

fn msg_pad_m4_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_VERIFY_QMM_PAD_M4")
            .ok()
            .is_some_and(|raw| matches!(raw.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
    })
}

fn msg_simdgroups() -> i32 {
    static CACHED: OnceLock<i32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_VERIFY_QMM_MSG_SIMDGROUPS")
            .ok()
            .and_then(|raw| raw.trim().parse::<i32>().ok())
            .filter(|value| matches!(value, 2 | 4 | 8 | 16))
            .unwrap_or(DEFAULT_MSG_SIMDGROUPS)
    })
}

fn split_k_huge_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_VERIFY_QMM_SPLIT_K_HUGE")
            .ok()
            .is_some_and(|raw| matches!(raw.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
    })
}

fn split_k_parts(n: i32) -> i32 {
    if n >= 4096 { 2 } else { 4 }
}

thread_local! {
    static VERIFY_QMM_ARMED: Cell<bool> = const { Cell::new(false) };
}

/// Restores the previous verify-QMM routing state when one target forward ends.
#[must_use]
pub(crate) struct QwenMtpVerifyQmmGuard {
    previous: bool,
}

impl QwenMtpVerifyQmmGuard {
    pub(crate) fn arm(enabled: bool) -> Self {
        let previous = VERIFY_QMM_ARMED.replace(enabled);
        Self { previous }
    }
}

impl Drop for QwenMtpVerifyQmmGuard {
    fn drop(&mut self) {
        VERIFY_QMM_ARMED.set(self.previous);
    }
}

fn verify_qmm_armed() -> bool {
    VERIFY_QMM_ARMED.get()
}

static VERIFY_QMM_SPLIT_K: OnceLock<MlxMetalKernel> = OnceLock::new();
static VERIFY_QMM_MSG: OnceLock<MlxMetalKernel> = OnceLock::new();

const VERIFY_QMM_SPLIT_K_SOURCE: &str = r#"
    uint part = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint tile = threadgroup_position_in_grid.y;

    constexpr int AccCount = 4 * MRows;
    constexpr int K = KDim;
    constexpr int N = NDim;
    const int PairCount = K / 8;
    const int WordsPerRow = K * Bits / 32;
    const int GroupCount = K / GroupSize;
    const int PairsPerPart = PairCount / KParts;

    int pair_begin = int(part) * PairsPerPart;
    int pair_end = int(part) == KParts - 1 ? PairCount : pair_begin + PairsPerPart;
    int n0 = int(tile) * 4;

    float accum[AccCount];
    _Pragma("unroll")
    for (int i = 0; i < AccCount; ++i) {
        accum[i] = 0.0f;
    }

    using Vec8 = vec<InputT, 8>;
    const device Vec8* x8 = reinterpret_cast<const device Vec8*>(x);

    for (int pair = pair_begin + int(lane); pair < pair_end; pair += 32) {
        int k_base = pair * 8;
        int group = k_base / GroupSize;
        Vec8 values[MRows];
        _Pragma("unroll")
        for (int row = 0; row < MRows; ++row) {
            values[row] = x8[(row * K + k_base) / 8];
        }

        // Hoist all four columns' independent global loads ahead of the FMA
        // chain. Interleaving one column's loads with its dependent math makes
        // this skinny projection latency-bound on Apple GPUs.
        float s0 = static_cast<float>(scales[(n0 + 0) * GroupCount + group]);
        float s1 = static_cast<float>(scales[(n0 + 1) * GroupCount + group]);
        float s2 = static_cast<float>(scales[(n0 + 2) * GroupCount + group]);
        float s3 = static_cast<float>(scales[(n0 + 3) * GroupCount + group]);
        float b0 = static_cast<float>(biases[(n0 + 0) * GroupCount + group]);
        float b1 = static_cast<float>(biases[(n0 + 1) * GroupCount + group]);
        float b2 = static_cast<float>(biases[(n0 + 2) * GroupCount + group]);
        float b3 = static_cast<float>(biases[(n0 + 3) * GroupCount + group]);
        if (Bits == 4) {
            uint p0 = weight[(n0 + 0) * WordsPerRow + pair];
            uint p1 = weight[(n0 + 1) * WordsPerRow + pair];
            uint p2 = weight[(n0 + 2) * WordsPerRow + pair];
            uint p3 = weight[(n0 + 3) * WordsPerRow + pair];
            {
                uint packed = p0;
                float scale = s0;
                float bias = b0;
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    float wv = float((packed >> (ki * 4)) & 0xFu) * scale + bias;
                    _Pragma("unroll")
                    for (int row = 0; row < MRows; ++row) {
                        accum[0 * MRows + row] += float(values[row][ki]) * wv;
                    }
                }
            }
            {
                uint packed = p1;
                float scale = s1;
                float bias = b1;
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    float wv = float((packed >> (ki * 4)) & 0xFu) * scale + bias;
                    _Pragma("unroll")
                    for (int row = 0; row < MRows; ++row) {
                        accum[1 * MRows + row] += float(values[row][ki]) * wv;
                    }
                }
            }
            {
                uint packed = p2;
                float scale = s2;
                float bias = b2;
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    float wv = float((packed >> (ki * 4)) & 0xFu) * scale + bias;
                    _Pragma("unroll")
                    for (int row = 0; row < MRows; ++row) {
                        accum[2 * MRows + row] += float(values[row][ki]) * wv;
                    }
                }
            }
            {
                uint packed = p3;
                float scale = s3;
                float bias = b3;
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    float wv = float((packed >> (ki * 4)) & 0xFu) * scale + bias;
                    _Pragma("unroll")
                    for (int row = 0; row < MRows; ++row) {
                        accum[3 * MRows + row] += float(values[row][ki]) * wv;
                    }
                }
            }
        } else {
            uint pa0 = weight[(n0 + 0) * WordsPerRow + pair * 2];
            uint pa1 = weight[(n0 + 1) * WordsPerRow + pair * 2];
            uint pa2 = weight[(n0 + 2) * WordsPerRow + pair * 2];
            uint pa3 = weight[(n0 + 3) * WordsPerRow + pair * 2];
            uint pb0 = weight[(n0 + 0) * WordsPerRow + pair * 2 + 1];
            uint pb1 = weight[(n0 + 1) * WordsPerRow + pair * 2 + 1];
            uint pb2 = weight[(n0 + 2) * WordsPerRow + pair * 2 + 1];
            uint pb3 = weight[(n0 + 3) * WordsPerRow + pair * 2 + 1];
            {
                uint pa = pa0;
                uint pb = pb0;
                float scale = s0;
                float bias = b0;
                _Pragma("unroll")
                for (int ki = 0; ki < 4; ++ki) {
                    float wa = float((pa >> (ki * 8)) & 0xFFu) * scale + bias;
                    float wb = float((pb >> (ki * 8)) & 0xFFu) * scale + bias;
                    _Pragma("unroll")
                    for (int row = 0; row < MRows; ++row) {
                        accum[0 * MRows + row] += float(values[row][ki]) * wa;
                        accum[0 * MRows + row] += float(values[row][ki + 4]) * wb;
                    }
                }
            }
            {
                uint pa = pa1;
                uint pb = pb1;
                float scale = s1;
                float bias = b1;
                _Pragma("unroll")
                for (int ki = 0; ki < 4; ++ki) {
                    float wa = float((pa >> (ki * 8)) & 0xFFu) * scale + bias;
                    float wb = float((pb >> (ki * 8)) & 0xFFu) * scale + bias;
                    _Pragma("unroll")
                    for (int row = 0; row < MRows; ++row) {
                        accum[1 * MRows + row] += float(values[row][ki]) * wa;
                        accum[1 * MRows + row] += float(values[row][ki + 4]) * wb;
                    }
                }
            }
            {
                uint pa = pa2;
                uint pb = pb2;
                float scale = s2;
                float bias = b2;
                _Pragma("unroll")
                for (int ki = 0; ki < 4; ++ki) {
                    float wa = float((pa >> (ki * 8)) & 0xFFu) * scale + bias;
                    float wb = float((pb >> (ki * 8)) & 0xFFu) * scale + bias;
                    _Pragma("unroll")
                    for (int row = 0; row < MRows; ++row) {
                        accum[2 * MRows + row] += float(values[row][ki]) * wa;
                        accum[2 * MRows + row] += float(values[row][ki + 4]) * wb;
                    }
                }
            }
            {
                uint pa = pa3;
                uint pb = pb3;
                float scale = s3;
                float bias = b3;
                _Pragma("unroll")
                for (int ki = 0; ki < 4; ++ki) {
                    float wa = float((pa >> (ki * 8)) & 0xFFu) * scale + bias;
                    float wb = float((pb >> (ki * 8)) & 0xFFu) * scale + bias;
                    _Pragma("unroll")
                    for (int row = 0; row < MRows; ++row) {
                        accum[3 * MRows + row] += float(values[row][ki]) * wa;
                        accum[3 * MRows + row] += float(values[row][ki + 4]) * wb;
                    }
                }
            }
        }
    }

    _Pragma("unroll")
    for (int i = 0; i < AccCount; ++i) {
        accum[i] = simd_sum(accum[i]);
    }

    threadgroup float partials[KParts * AccCount];
    if (lane == 0) {
        _Pragma("unroll")
        for (int i = 0; i < AccCount; ++i) {
            partials[int(part) * AccCount + i] = accum[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (part == 0 && int(lane) < AccCount) {
        float total = 0.0f;
        _Pragma("unroll")
        for (int p = 0; p < KParts; ++p) {
            total += partials[p * AccCount + int(lane)];
        }
        int column = int(lane) / MRows;
        int row = int(lane) - column * MRows;
        out[row * N + n0 + column] = static_cast<InputT>(total);
    }
"#;

const VERIFY_QMM_MSG_SOURCE: &str = r#"
    uint simdgroup = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint tile = threadgroup_position_in_grid.y;

    constexpr int AccCount = 4 * MRows;
    constexpr int K = KDim;
    constexpr int N = NDim;
    const int PairCount = K / 8;
    const int WordsPerRow = K * Bits / 32;
    const int GroupCount = K / GroupSize;
    int n0 = (int(tile) * NumSimdgroups + int(simdgroup)) * 4;
    if (n0 + 3 >= N) {
        return;
    }

    float accum[AccCount];
    _Pragma("unroll")
    for (int i = 0; i < AccCount; ++i) {
        accum[i] = 0.0f;
    }

    using Vec8 = vec<InputT, 8>;
    const device Vec8* x8 = reinterpret_cast<const device Vec8*>(x);
    for (int pair = int(lane); pair < PairCount; pair += 32) {
        int k_base = pair * 8;
        int group = k_base / GroupSize;
        Vec8 values[MRows];
        _Pragma("unroll")
        for (int row = 0; row < MRows; ++row) {
            values[row] = x8[(row * K + k_base) / 8];
        }
        float s0 = static_cast<float>(scales[(n0 + 0) * GroupCount + group]);
        float s1 = static_cast<float>(scales[(n0 + 1) * GroupCount + group]);
        float s2 = static_cast<float>(scales[(n0 + 2) * GroupCount + group]);
        float s3 = static_cast<float>(scales[(n0 + 3) * GroupCount + group]);
        float b0 = static_cast<float>(biases[(n0 + 0) * GroupCount + group]);
        float b1 = static_cast<float>(biases[(n0 + 1) * GroupCount + group]);
        float b2 = static_cast<float>(biases[(n0 + 2) * GroupCount + group]);
        float b3 = static_cast<float>(biases[(n0 + 3) * GroupCount + group]);
        if (Bits == 4) {
            uint p0 = weight[(n0 + 0) * WordsPerRow + pair];
            uint p1 = weight[(n0 + 1) * WordsPerRow + pair];
            uint p2 = weight[(n0 + 2) * WordsPerRow + pair];
            uint p3 = weight[(n0 + 3) * WordsPerRow + pair];
            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {
                float w0 = float((p0 >> (ki * 4)) & 0xFu) * s0 + b0;
                float w1 = float((p1 >> (ki * 4)) & 0xFu) * s1 + b1;
                float w2 = float((p2 >> (ki * 4)) & 0xFu) * s2 + b2;
                float w3 = float((p3 >> (ki * 4)) & 0xFu) * s3 + b3;
                _Pragma("unroll")
                for (int row = 0; row < MRows; ++row) {
                    float xv = static_cast<float>(values[row][ki]);
                    accum[0 * MRows + row] += xv * w0;
                    accum[1 * MRows + row] += xv * w1;
                    accum[2 * MRows + row] += xv * w2;
                    accum[3 * MRows + row] += xv * w3;
                }
            }
        } else {
            uint pa0 = weight[(n0 + 0) * WordsPerRow + pair * 2];
            uint pa1 = weight[(n0 + 1) * WordsPerRow + pair * 2];
            uint pa2 = weight[(n0 + 2) * WordsPerRow + pair * 2];
            uint pa3 = weight[(n0 + 3) * WordsPerRow + pair * 2];
            uint pb0 = weight[(n0 + 0) * WordsPerRow + pair * 2 + 1];
            uint pb1 = weight[(n0 + 1) * WordsPerRow + pair * 2 + 1];
            uint pb2 = weight[(n0 + 2) * WordsPerRow + pair * 2 + 1];
            uint pb3 = weight[(n0 + 3) * WordsPerRow + pair * 2 + 1];
            _Pragma("unroll")
            for (int ki = 0; ki < 4; ++ki) {
                float wa0 = float((pa0 >> (ki * 8)) & 0xFFu) * s0 + b0;
                float wa1 = float((pa1 >> (ki * 8)) & 0xFFu) * s1 + b1;
                float wa2 = float((pa2 >> (ki * 8)) & 0xFFu) * s2 + b2;
                float wa3 = float((pa3 >> (ki * 8)) & 0xFFu) * s3 + b3;
                float wb0 = float((pb0 >> (ki * 8)) & 0xFFu) * s0 + b0;
                float wb1 = float((pb1 >> (ki * 8)) & 0xFFu) * s1 + b1;
                float wb2 = float((pb2 >> (ki * 8)) & 0xFFu) * s2 + b2;
                float wb3 = float((pb3 >> (ki * 8)) & 0xFFu) * s3 + b3;
                _Pragma("unroll")
                for (int row = 0; row < MRows; ++row) {
                    float xa = static_cast<float>(values[row][ki]);
                    float xb = static_cast<float>(values[row][ki + 4]);
                    accum[0 * MRows + row] += xa * wa0 + xb * wb0;
                    accum[1 * MRows + row] += xa * wa1 + xb * wb1;
                    accum[2 * MRows + row] += xa * wa2 + xb * wb2;
                    accum[3 * MRows + row] += xa * wa3 + xb * wb3;
                }
            }
        }
    }

    _Pragma("unroll")
    for (int i = 0; i < AccCount; ++i) {
        accum[i] = simd_sum(accum[i]);
    }
    if (int(lane) < AccCount) {
        int column = int(lane) / MRows;
        int row = int(lane) - column * MRows;
        out[row * N + n0 + column] = static_cast<InputT>(accum[int(lane)]);
    }
"#;

fn eligible(x: &MlxArray, weight: &QuantizedWeight, min_route_n: i32) -> Option<(i32, i32, i32)> {
    if !verify_qmm_armed()
        || !matches!(x.dtype(), MlxDtype::Bfloat16 | MlxDtype::Float16)
        || !matches!(
            weight.mlx_quantization_mode(),
            mlx_sys::MlxQuantizationMode::Affine
        )
        || !matches!(weight.bits, 4 | 8)
        || !matches!(weight.group_size, 32 | 64 | 128)
    {
        return None;
    }
    let shape = x.shape();
    if shape.len() != 3 || shape[0] != 1 || !(3..=6).contains(&shape[1]) {
        return None;
    }
    let m = shape[1];
    let k = shape[2];
    let weight_shape = weight.weight.shape();
    if weight_shape.len() != 2 {
        return None;
    }
    let n = weight_shape[0];
    if k <= 0
        || n < min_route_n
        || k % 64 != 0
        || n % 4 != 0
        || weight_shape[1] != k * weight.bits / 32
        || k % weight.group_size != 0
    {
        return None;
    }
    let groups = k / weight.group_size;
    if weight.scales.as_ref()?.shape() != [n, groups]
        || weight.biases.as_ref()?.shape() != [n, groups]
    {
        return None;
    }
    Some((m, k, n))
}

pub(crate) fn try_qwen_mtp_verify_qmm(x: &MlxArray, weight: &QuantizedWeight) -> Option<MlxArray> {
    try_qwen_mtp_verify_qmm_for_min_n(x, weight, min_route_n())
}

fn try_qwen_mtp_verify_qmm_for_min_n(
    x: &MlxArray,
    weight: &QuantizedWeight,
    min_route_n: i32,
) -> Option<MlxArray> {
    let (m, k, n) = eligible(x, weight, min_route_n)?;
    let scales = weight.scales.as_ref()?;
    let biases = weight.biases.as_ref()?;
    // Inputs to individual projections can be reshaped residual/norm views.
    // The Metal kernels index row-major storage directly and therefore require
    // the same explicit contiguous boundary used by the reference runtimes.
    let x_flat = contiguous(&reshape(x, &[m, k], None), None);
    let common_args = |m_rows| {
        vec![
            KernelTemplateArg::Dtype {
                name: "InputT",
                dtype: x.dtype(),
            },
            KernelTemplateArg::Int {
                name: "MRows",
                value: m_rows,
            },
            KernelTemplateArg::Int {
                name: "Bits",
                value: weight.bits,
            },
            KernelTemplateArg::Int {
                name: "GroupSize",
                value: weight.group_size,
            },
            KernelTemplateArg::Int {
                name: "KDim",
                value: k,
            },
            KernelTemplateArg::Int {
                name: "NDim",
                value: n,
            },
        ]
    };
    let msg_simdgroups = msg_simdgroups();
    // MTPLX 2.9.0's admitted M4 profile keeps its constant-K split reduction
    // for the vocabulary head (`vk_k`). AX historically selected the oMLX
    // multi-simdgroup morphology at this width; retain that default while the
    // matched M5 trial can select split-K explicitly.
    let use_msg = !split_k_huge_enabled() && n >= MSG_ROUTE_N && n % (4 * msg_simdgroups) == 0;
    let (mut outputs, output_rows) = if use_msg {
        // oMLX templates the huge-vocabulary MSG kernel at four rows for
        // M=3, padding one zero row. Weight traffic dominates this projection;
        // the fixed M4 morphology can schedule better than a 12-accumulator
        // M3 specialization. Keep it independently gated until M5 admission.
        let output_rows = if msg_pad_m4_enabled() && m < 4 { 4 } else { m };
        let padded;
        let kernel_x = if output_rows == m {
            &x_flat
        } else {
            let pad = zeros(&[output_rows - m, k], x.dtype(), None);
            padded = contiguous(&concatenate(&[&x_flat, &pad], 0, None), None);
            &padded
        };
        let kernel = VERIFY_QMM_MSG.get_or_init(|| {
            MlxMetalKernel::new(
                "ax_qwen_mtp_verify_qmm_msg_v4",
                &["x", "weight", "scales", "biases"],
                &["out"],
                VERIFY_QMM_MSG_SOURCE,
                "",
                true,
            )
        });
        let mut args = common_args(output_rows);
        args.push(KernelTemplateArg::Int {
            name: "NumSimdgroups",
            value: msg_simdgroups,
        });
        let outputs = kernel
            .try_apply_with_template(
                &[kernel_x, &weight.weight, scales, biases],
                &[KernelOutputSpec {
                    shape: vec![output_rows, n],
                    dtype: x.dtype(),
                }],
                &args,
                (32 * msg_simdgroups, n / (4 * msg_simdgroups), 1),
                (32 * msg_simdgroups, 1, 1),
                None,
            )
            .ok()?;
        (outputs, output_rows)
    } else {
        let kernel = VERIFY_QMM_SPLIT_K.get_or_init(|| {
            MlxMetalKernel::new(
                "ax_qwen_mtp_verify_qmm_split_k_v4",
                &["x", "weight", "scales", "biases"],
                &["out"],
                VERIFY_QMM_SPLIT_K_SOURCE,
                "",
                true,
            )
        });
        // Small output projections expose too few column tiles to saturate an
        // M5 GPU with only two K partitions. Four partitions restore occupancy;
        // wider projections keep the lower-reduction-overhead two-way split.
        let k_parts = split_k_parts(n);
        let mut args = common_args(m);
        args.push(KernelTemplateArg::Int {
            name: "KParts",
            value: k_parts,
        });
        let outputs = kernel
            .try_apply_with_template(
                &[&x_flat, &weight.weight, scales, biases],
                &[KernelOutputSpec {
                    shape: vec![m, n],
                    dtype: x.dtype(),
                }],
                &args,
                (32 * k_parts, n / 4, 1),
                (32 * k_parts, 1, 1),
                None,
            )
            .ok()?;
        (outputs, m)
    };
    let flat = outputs.pop()?;
    let flat = if output_rows == m {
        flat
    } else {
        slice(&flat, &[0, 0], &[m, n], &[1, 1], None)
    };
    let projected = reshape(&flat, &[1, m, n], None);
    Some(if let Some(linear_bias) = weight.linear_bias.as_ref() {
        add(&projected, linear_bias, None)
    } else {
        projected
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::{MlxQuantizationMode, astype, eval, quantize, quantized_matmul_with_mode};

    #[test]
    fn guard_restores_nested_state() {
        assert!(!verify_qmm_armed());
        {
            let _outer = QwenMtpVerifyQmmGuard::arm(true);
            assert!(verify_qmm_armed());
            {
                let _inner = QwenMtpVerifyQmmGuard::arm(false);
                assert!(!verify_qmm_armed());
            }
            assert!(verify_qmm_armed());
        }
        assert!(!verify_qmm_armed());
    }

    fn assert_route_matches_affine_qmm(n: i32, min_route_n: i32, m: i32, route: &str) {
        let k = 64_i32;
        let dense_data: Vec<f32> = (0..(n * k) as usize)
            .map(|index| ((index % 127) as f32 - 63.0) / 512.0)
            .collect();
        let dense = reshape(&MlxArray::from_f32_slice(&dense_data), &[n, k], None);
        let quantized = quantize(
            &dense,
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
            mode: "affine".to_owned(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        let input_data: Vec<f32> = (0..(m * k) as usize)
            .map(|index| ((index % 29) as f32 - 14.0) / 128.0)
            .collect();
        let input_f32 = reshape(&MlxArray::from_f32_slice(&input_data), &[1, m, k], None);
        let input = astype(&input_f32, MlxDtype::Bfloat16, None);

        let _guard = QwenMtpVerifyQmmGuard::arm(true);
        let routed = try_qwen_mtp_verify_qmm_for_min_n(&input, &weight, min_route_n)
            .unwrap_or_else(|| panic!("eligible {route} route"));
        let reference = quantized_matmul_with_mode(
            &input,
            &weight.weight,
            weight.scales.as_ref().unwrap(),
            weight.biases.as_ref(),
            true,
            Some(weight.group_size),
            Some(weight.bits),
            MlxQuantizationMode::Affine,
            None,
        );
        let routed_f32 = astype(&routed, MlxDtype::Float32, None);
        let reference_f32 = astype(&reference, MlxDtype::Float32, None);
        eval(&[&routed_f32, &reference_f32]);
        assert_eq!(routed.shape(), vec![1, m, n]);
        let max_abs = routed_f32
            .data_f32()
            .iter()
            .zip(reference_f32.data_f32())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs <= 0.03125, "{route} max abs diff was {max_abs}");
    }

    #[test]
    fn split_k_matches_affine_qmm_on_minimum_routed_shape() {
        assert_route_matches_affine_qmm(SPLIT_K_TEST_N, SPLIT_K_TEST_N, 3, "split-K");
    }

    #[test]
    fn four_row_split_k_matches_affine_qmm_on_minimum_routed_shape() {
        assert_route_matches_affine_qmm(SPLIT_K_TEST_N, SPLIT_K_TEST_N, 4, "four-row split-K");
    }

    #[test]
    fn four_partition_split_k_matches_small_projection() {
        assert_route_matches_affine_qmm(2048, 0, 3, "small split-K");
    }

    #[test]
    fn multi_simdgroup_matches_affine_qmm_on_large_n_shape() {
        assert_route_matches_affine_qmm(MSG_ROUTE_N, MIN_ROUTE_N, 3, "multi-simdgroup");
    }
}
