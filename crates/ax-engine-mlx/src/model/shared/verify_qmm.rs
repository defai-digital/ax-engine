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
    KernelOutputSpec, KernelTemplateArg, LazyKernel, MlxArray, MlxDtype, MlxMetalKernel,
    concatenate, contiguous, reshape, slice, zeros,
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
/// Widest verify window the affine kernels serve. The accumulators are
/// `4 * rows` floats per simdgroup, which fits the 32-lane reduction up to 8
/// rows; 7 and 8 rows serve DFlash2's 8-row windows.
const AFFINE_MAX_ROWS: i32 = 8;
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
    *CACHED.get_or_init(|| crate::fastpath::env_flag("AX_MLX_MTP_VERIFY_QMM_PAD_M4"))
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

/// `AX_MLX_MTP_VERIFY_QMM_MXFP4` (default on; `0` disables): route MXFP4 verify
/// matmuls of 3..=8 rows through the fused window kernel. On the M5 Max with
/// Qwen 3.8 27B MXFP4 MTP over 17 prompts it decodes 1.138x faster than the
/// stock MLX matmul (every prompt faster, run-to-run noise under 0.2%) and
/// stays as close to direct decode as the stock route (12/17 vs 11/17 prompts
/// token-identical).
fn mxfp4_window_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| crate::fastpath::env_flag_default_on("AX_MLX_MTP_VERIFY_QMM_MXFP4"))
}

/// Minimum output width routed through the MXFP4 window kernel. Zero routes
/// every eligible projection: the M5 FFN-chain probe gained on the narrow
/// down projection (N = hidden size) as much as on the wide gate/up one.
fn mxfp4_window_min_n() -> i32 {
    static CACHED: OnceLock<i32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_VERIFY_QMM_MXFP4_MIN_N")
            .ok()
            .and_then(|raw| raw.trim().parse::<i32>().ok())
            .filter(|value| *value >= 0)
            .unwrap_or(0)
    })
}

fn split_k_huge_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| crate::fastpath::env_flag("AX_MLX_MTP_VERIFY_QMM_SPLIT_K_HUGE"))
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

static VERIFY_QMM_SPLIT_K: LazyKernel = LazyKernel::new(
    "ax_qwen_mtp_verify_qmm_split_k_v5",
    &["x", "weight", "scales", "biases"],
    &["out"],
    VERIFY_QMM_SPLIT_K_SOURCE,
    "",
    true,
);
static VERIFY_QMM_MSG: LazyKernel = LazyKernel::new(
    "ax_qwen_mtp_verify_qmm_msg_v5",
    &["x", "weight", "scales", "biases"],
    &["out"],
    VERIFY_QMM_MSG_SOURCE,
    "",
    true,
);

static VERIFY_QMM_MXFP4_WINDOW: LazyKernel = LazyKernel::new(
    "ax_qwen_mtp_verify_qmm_mxfp4_window_v1",
    &["x", "weight", "scales"],
    &["out"],
    VERIFY_QMM_MXFP4_WINDOW_SOURCE,
    VERIFY_QMM_MXFP4_WINDOW_HEADER,
    true,
);

/// Test-only switch that forces the lazy verify-QMM kernels to report as
/// unavailable, so the fail-closed fallback route can be exercised without a
/// genuine Metal compile failure.
#[cfg(test)]
static FORCE_VERIFY_QMM_KERNEL_UNAVAILABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fail-closed lookup for the split-K verify kernel.
fn verify_qmm_split_k_kernel() -> Option<&'static MlxMetalKernel> {
    #[cfg(test)]
    if FORCE_VERIFY_QMM_KERNEL_UNAVAILABLE.load(std::sync::atomic::Ordering::Acquire) {
        return None;
    }
    VERIFY_QMM_SPLIT_K.get()
}

/// Fail-closed lookup for the MXFP4 window verify kernel.
fn verify_qmm_mxfp4_window_kernel() -> Option<&'static MlxMetalKernel> {
    #[cfg(test)]
    if FORCE_VERIFY_QMM_KERNEL_UNAVAILABLE.load(std::sync::atomic::Ordering::Acquire) {
        return None;
    }
    VERIFY_QMM_MXFP4_WINDOW.get()
}

/// Fail-closed lookup for the multi-simdgroup verify kernel.
fn verify_qmm_msg_kernel() -> Option<&'static MlxMetalKernel> {
    #[cfg(test)]
    if FORCE_VERIFY_QMM_KERNEL_UNAVAILABLE.load(std::sync::atomic::Ordering::Acquire) {
        return None;
    }
    VERIFY_QMM_MSG.get()
}

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
            // `pa` holds values 0..3 and `pb` values 4..7 of this pair. 8-bit:
            // one word each, value i at bit 8*i. 6-bit (MLX packs 4 values per
            // 3 bytes): the pair is 6 bytes at a 2-byte-aligned offset, read
            // as three halves into two 24-bit packs, value i at bit 6*i.
            uint pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3;
            if (Bits == 6) {
                const device ushort* w16 = reinterpret_cast<const device ushort*>(weight);
                const int HalvesPerRow = WordsPerRow * 2;
                int h0 = (n0 + 0) * HalvesPerRow + pair * 3;
                int h1 = (n0 + 1) * HalvesPerRow + pair * 3;
                int h2 = (n0 + 2) * HalvesPerRow + pair * 3;
                int h3 = (n0 + 3) * HalvesPerRow + pair * 3;
                uint u00 = w16[h0], u01 = w16[h0 + 1], u02 = w16[h0 + 2];
                uint u10 = w16[h1], u11 = w16[h1 + 1], u12 = w16[h1 + 2];
                uint u20 = w16[h2], u21 = w16[h2 + 1], u22 = w16[h2 + 2];
                uint u30 = w16[h3], u31 = w16[h3 + 1], u32 = w16[h3 + 2];
                pa0 = u00 | ((u01 & 0xFFu) << 16);
                pa1 = u10 | ((u11 & 0xFFu) << 16);
                pa2 = u20 | ((u21 & 0xFFu) << 16);
                pa3 = u30 | ((u31 & 0xFFu) << 16);
                pb0 = (u01 >> 8) | (u02 << 8);
                pb1 = (u11 >> 8) | (u12 << 8);
                pb2 = (u21 >> 8) | (u22 << 8);
                pb3 = (u31 >> 8) | (u32 << 8);
            } else {
                pa0 = weight[(n0 + 0) * WordsPerRow + pair * 2];
                pa1 = weight[(n0 + 1) * WordsPerRow + pair * 2];
                pa2 = weight[(n0 + 2) * WordsPerRow + pair * 2];
                pa3 = weight[(n0 + 3) * WordsPerRow + pair * 2];
                pb0 = weight[(n0 + 0) * WordsPerRow + pair * 2 + 1];
                pb1 = weight[(n0 + 1) * WordsPerRow + pair * 2 + 1];
                pb2 = weight[(n0 + 2) * WordsPerRow + pair * 2 + 1];
                pb3 = weight[(n0 + 3) * WordsPerRow + pair * 2 + 1];
            }
            const uint VShift = Bits == 6 ? 6u : 8u;
            const uint VMask = Bits == 6 ? 0x3Fu : 0xFFu;
            {
                uint pa = pa0;
                uint pb = pb0;
                float scale = s0;
                float bias = b0;
                _Pragma("unroll")
                for (int ki = 0; ki < 4; ++ki) {
                    float wa = float((pa >> (ki * VShift)) & VMask) * scale + bias;
                    float wb = float((pb >> (ki * VShift)) & VMask) * scale + bias;
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
                    float wa = float((pa >> (ki * VShift)) & VMask) * scale + bias;
                    float wb = float((pb >> (ki * VShift)) & VMask) * scale + bias;
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
                    float wa = float((pa >> (ki * VShift)) & VMask) * scale + bias;
                    float wb = float((pb >> (ki * VShift)) & VMask) * scale + bias;
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
                    float wa = float((pa >> (ki * VShift)) & VMask) * scale + bias;
                    float wb = float((pb >> (ki * VShift)) & VMask) * scale + bias;
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
            // `pa` holds values 0..3 and `pb` values 4..7 of this pair. 8-bit:
            // one word each, value i at bit 8*i. 6-bit (MLX packs 4 values per
            // 3 bytes): the pair is 6 bytes at a 2-byte-aligned offset, read
            // as three halves into two 24-bit packs, value i at bit 6*i.
            uint pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3;
            if (Bits == 6) {
                const device ushort* w16 = reinterpret_cast<const device ushort*>(weight);
                const int HalvesPerRow = WordsPerRow * 2;
                int h0 = (n0 + 0) * HalvesPerRow + pair * 3;
                int h1 = (n0 + 1) * HalvesPerRow + pair * 3;
                int h2 = (n0 + 2) * HalvesPerRow + pair * 3;
                int h3 = (n0 + 3) * HalvesPerRow + pair * 3;
                uint u00 = w16[h0], u01 = w16[h0 + 1], u02 = w16[h0 + 2];
                uint u10 = w16[h1], u11 = w16[h1 + 1], u12 = w16[h1 + 2];
                uint u20 = w16[h2], u21 = w16[h2 + 1], u22 = w16[h2 + 2];
                uint u30 = w16[h3], u31 = w16[h3 + 1], u32 = w16[h3 + 2];
                pa0 = u00 | ((u01 & 0xFFu) << 16);
                pa1 = u10 | ((u11 & 0xFFu) << 16);
                pa2 = u20 | ((u21 & 0xFFu) << 16);
                pa3 = u30 | ((u31 & 0xFFu) << 16);
                pb0 = (u01 >> 8) | (u02 << 8);
                pb1 = (u11 >> 8) | (u12 << 8);
                pb2 = (u21 >> 8) | (u22 << 8);
                pb3 = (u31 >> 8) | (u32 << 8);
            } else {
                pa0 = weight[(n0 + 0) * WordsPerRow + pair * 2];
                pa1 = weight[(n0 + 1) * WordsPerRow + pair * 2];
                pa2 = weight[(n0 + 2) * WordsPerRow + pair * 2];
                pa3 = weight[(n0 + 3) * WordsPerRow + pair * 2];
                pb0 = weight[(n0 + 0) * WordsPerRow + pair * 2 + 1];
                pb1 = weight[(n0 + 1) * WordsPerRow + pair * 2 + 1];
                pb2 = weight[(n0 + 2) * WordsPerRow + pair * 2 + 1];
                pb3 = weight[(n0 + 3) * WordsPerRow + pair * 2 + 1];
            }
            const uint VShift = Bits == 6 ? 6u : 8u;
            const uint VMask = Bits == 6 ? 0x3Fu : 0xFFu;
            _Pragma("unroll")
            for (int ki = 0; ki < 4; ++ki) {
                float wa0 = float((pa0 >> (ki * VShift)) & VMask) * s0 + b0;
                float wa1 = float((pa1 >> (ki * VShift)) & VMask) * s1 + b1;
                float wa2 = float((pa2 >> (ki * VShift)) & VMask) * s2 + b2;
                float wa3 = float((pa3 >> (ki * VShift)) & VMask) * s3 + b3;
                float wb0 = float((pb0 >> (ki * VShift)) & VMask) * s0 + b0;
                float wb1 = float((pb1 >> (ki * VShift)) & VMask) * s1 + b1;
                float wb2 = float((pb2 >> (ki * VShift)) & VMask) * s2 + b2;
                float wb3 = float((pb3 >> (ki * VShift)) & VMask) * s3 + b3;
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

/// FP4 (e2m1) decode by bit placement: the three magnitude bits go to the top
/// of a half's exponent/mantissa field and the sign bit to bit 15, which reads
/// back as the FP4 value times 2^14 (exact, including the subnormal case). The
/// caller folds 2^14 and the e8m0 group scale into one multiply per group.
const VERIFY_QMM_MXFP4_WINDOW_HEADER: &str = r#"
inline float fp4_scaled(uint nib) {
    ushort bits = ushort(((nib & 7u) << 9) | ((nib & 8u) << 12));
    return float(as_type<half>(bits));
}
"#;

/// Fused multi-row MXFP4 matmul for speculative verify windows: every weight
/// word is read and decoded once and applied to all `Rows` activation rows
/// held in registers (qmv generalised to 3..=8 rows). Two simdgroups per
/// threadgroup, two output columns per simdgroup, eight K values per lane per
/// step.
const VERIFY_QMM_MXFP4_WINDOW_SOURCE: &str = r#"
    constexpr int SimdGroups = 2;
    constexpr int Results = 2;
    constexpr int Lanes = 32;
    constexpr int ValuesPerLane = 8;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint col0 = (threadgroup_position_in_grid.x * SimdGroups + sg) * Results;
    if (col0 >= NDim) {
        return;
    }

    float acc[Results][Rows];
    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < Results; c++) {
        _Pragma("clang loop unroll(full)")
        for (int m = 0; m < Rows; m++) {
            acc[c][m] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < KDim; k0 += Lanes * ValuesPerLane) {
        int kk = k0 + int(lane) * ValuesPerLane;
        float xv[Rows][ValuesPerLane];
        _Pragma("clang loop unroll(full)")
        for (int m = 0; m < Rows; m++) {
            _Pragma("clang loop unroll(full)")
            for (int i = 0; i < ValuesPerLane; i++) {
                xv[m][i] = float(x[m * KDim + kk + i]);
            }
        }
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < Results; c++) {
            uint col = col0 + uint(c);
            uint word = weight[col * uint(KDim / 8) + uint(kk / 8)];
            float group_scale =
                as_type<float>(uint(scales[col * uint(KDim / 32) + uint(kk / 32)]) << 23) * 16384.0f;
            float part[Rows];
            _Pragma("clang loop unroll(full)")
            for (int m = 0; m < Rows; m++) {
                part[m] = 0.0f;
            }
            _Pragma("clang loop unroll(full)")
            for (int i = 0; i < ValuesPerLane; i++) {
                float v = fp4_scaled((word >> (4 * i)) & 0xFu);
                _Pragma("clang loop unroll(full)")
                for (int m = 0; m < Rows; m++) {
                    part[m] += v * xv[m][i];
                }
            }
            _Pragma("clang loop unroll(full)")
            for (int m = 0; m < Rows; m++) {
                acc[c][m] += part[m] * group_scale;
            }
        }
    }

    _Pragma("clang loop unroll(full)")
    for (int c = 0; c < Results; c++) {
        _Pragma("clang loop unroll(full)")
        for (int m = 0; m < Rows; m++) {
            float r = simd_sum(acc[c][m]);
            if (lane == 0) {
                out[m * NDim + col0 + uint(c)] = static_cast<InputT>(r);
            }
        }
    }
"#;

const MXFP4_WINDOW_MIN_ROWS: i32 = 3;
const MXFP4_WINDOW_MAX_ROWS: i32 = 8;
/// Output columns per threadgroup: 2 simdgroups x 2 results.
const MXFP4_WINDOW_COLUMNS_PER_GROUP: i32 = 4;
/// K values consumed per loop step: 32 lanes x 8 values.
const MXFP4_WINDOW_K_STEP: i32 = 256;

fn mxfp4_window_eligible(
    x: &MlxArray,
    weight: &QuantizedWeight,
    min_n: i32,
) -> Option<(i32, i32, i32)> {
    if !verify_qmm_armed()
        || !matches!(x.dtype(), MlxDtype::Bfloat16 | MlxDtype::Float16)
        || !matches!(
            weight.mlx_quantization_mode(),
            mlx_sys::MlxQuantizationMode::Mxfp4
        )
        || weight.bits != 4
        || weight.group_size != 32
        || weight.biases.is_some()
        || weight.weight.dtype() != MlxDtype::Uint32
    {
        return None;
    }
    let shape = x.shape();
    if shape.len() != 3
        || shape[0] != 1
        || !(MXFP4_WINDOW_MIN_ROWS..=MXFP4_WINDOW_MAX_ROWS).contains(&shape[1])
    {
        return None;
    }
    let (m, k) = (shape[1], shape[2]);
    let weight_shape = weight.weight.shape();
    if weight_shape.len() != 2 {
        return None;
    }
    let n = weight_shape[0];
    if k <= 0
        || n < min_n
        || k % MXFP4_WINDOW_K_STEP != 0
        || n % MXFP4_WINDOW_COLUMNS_PER_GROUP != 0
        || weight_shape[1] != k / 8
    {
        return None;
    }
    let scales = weight.scales.as_ref()?;
    if scales.dtype() != MlxDtype::Uint8 || scales.shape() != [n, k / 32] {
        return None;
    }
    Some((m, k, n))
}

fn try_mxfp4_window_qmm(x: &MlxArray, weight: &QuantizedWeight) -> Option<MlxArray> {
    if !mxfp4_window_enabled() {
        return None;
    }
    mxfp4_window_qmm(x, weight, mxfp4_window_min_n())
}

fn mxfp4_window_qmm(x: &MlxArray, weight: &QuantizedWeight, min_n: i32) -> Option<MlxArray> {
    let (m, k, n) = mxfp4_window_eligible(x, weight, min_n)?;
    let scales = weight.scales.as_ref()?;
    let kernel = verify_qmm_mxfp4_window_kernel()?;
    let x_flat = contiguous(&reshape(x, &[m, k], None), None);
    let mut outputs = kernel
        .try_apply_with_template(
            &[&x_flat, &weight.weight, scales],
            &[KernelOutputSpec {
                shape: vec![m, n],
                dtype: x.dtype(),
            }],
            &[
                KernelTemplateArg::Dtype {
                    name: "InputT",
                    dtype: x.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "Rows",
                    value: m,
                },
                KernelTemplateArg::Int {
                    name: "KDim",
                    value: k,
                },
                KernelTemplateArg::Int {
                    name: "NDim",
                    value: n,
                },
            ],
            (n / MXFP4_WINDOW_COLUMNS_PER_GROUP * 64, 1, 1),
            (64, 1, 1),
            None,
        )
        .ok()?;
    let flat = outputs.pop()?;
    // The projection caller adds the separate dense Linear bias once.
    Some(reshape(&flat, &[1, m, n], None))
}

fn eligible(x: &MlxArray, weight: &QuantizedWeight, min_route_n: i32) -> Option<(i32, i32, i32)> {
    if !verify_qmm_armed()
        || !matches!(x.dtype(), MlxDtype::Bfloat16 | MlxDtype::Float16)
        || !matches!(
            weight.mlx_quantization_mode(),
            mlx_sys::MlxQuantizationMode::Affine
        )
        || !matches!(weight.bits, 4 | 6 | 8)
        || !matches!(weight.group_size, 32 | 64 | 128)
    {
        return None;
    }
    let shape = x.shape();
    if shape.len() != 3 || shape[0] != 1 || !(3..=AFFINE_MAX_ROWS).contains(&shape[1]) {
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
    let scales = weight.scales.as_ref()?;
    let biases = weight.biases.as_ref()?;
    // Mixed affine dtypes need MLX's common-type promotion before projection.
    if scales.dtype() != x.dtype()
        || biases.dtype() != x.dtype()
        || scales.shape() != [n, groups]
        || biases.shape() != [n, groups]
    {
        return None;
    }
    Some((m, k, n))
}

/// Relaxed verify-shape projection: the kernels here fold the group bias into
/// every product (`sum(x * (q * scale + bias))`) instead of MLX's
/// `scale * sum(q * x) + bias * sum(x)`, so they are **not** bit-exact with
/// MLX and are admitted by tolerance-only tests. They may only be armed for a
/// relaxed target-verify profile; arming this route under the exact profile
/// would silently give up the invariant that route is built to preserve.
pub(crate) fn try_qwen_mtp_verify_qmm(x: &MlxArray, weight: &QuantizedWeight) -> Option<MlxArray> {
    if matches!(
        weight.mlx_quantization_mode(),
        mlx_sys::MlxQuantizationMode::Mxfp4
    ) {
        return try_mxfp4_window_qmm(x, weight);
    }
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
        let kernel = verify_qmm_msg_kernel()?;
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
        let kernel = verify_qmm_split_k_kernel()?;
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
    // The projection caller adds the separate dense Linear bias once.
    Some(reshape(&flat, &[1, m, n], None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::{
        MlxQuantizationMode, arange, astype, eval, quantize, quantized_matmul_with_mode,
    };

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

    #[test]
    fn kernel_unavailable_falls_back_to_reference_qmm() {
        let k = 64_i32;
        let n = SPLIT_K_TEST_N;
        let m = 3;
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
            scales: Some(astype(&quantized[1], MlxDtype::Bfloat16, None)),
            biases: Some(astype(&quantized[2], MlxDtype::Bfloat16, None)),
            group_size: 32,
            bits: 4,
            mode: "affine".to_owned(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
            weight_t_view: Default::default(),
        };
        let input_data: Vec<f32> = (0..(m * k) as usize)
            .map(|index| ((index % 29) as f32 - 14.0) / 128.0)
            .collect();
        let input = astype(
            &reshape(&MlxArray::from_f32_slice(&input_data), &[1, m, k], None),
            MlxDtype::Bfloat16,
            None,
        );

        let _guard = QwenMtpVerifyQmmGuard::arm(true);
        // Mock the kernel as unavailable: the routed path must signal fallback
        // (None) rather than panicking, and the caller's MLX reference QMM must
        // still produce the projection.
        FORCE_VERIFY_QMM_KERNEL_UNAVAILABLE.store(true, std::sync::atomic::Ordering::Release);
        assert!(
            try_qwen_mtp_verify_qmm(&input, &weight).is_none(),
            "unavailable kernel must fall back to None (reference route)"
        );
        FORCE_VERIFY_QMM_KERNEL_UNAVAILABLE.store(false, std::sync::atomic::Ordering::Release);

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
        let reference_f32 = astype(&reference, MlxDtype::Float32, None);
        eval(&[&reference_f32]);
        assert_eq!(reference.shape(), vec![1, m, n]);
        assert!(
            reference_f32
                .data_f32()
                .iter()
                .all(|value| value.is_finite()),
            "reference QMM must remain usable when the kernel is unavailable"
        );
    }

    fn assert_route_matches_affine_qmm(n: i32, min_route_n: i32, m: i32, route: &str) {
        assert_route_matches_affine_qmm_with(n, min_route_n, m, 4, 64, 32, route);
    }

    fn assert_route_matches_affine_qmm_with(
        n: i32,
        min_route_n: i32,
        m: i32,
        bits: i32,
        k: i32,
        group_size: i32,
        route: &str,
    ) {
        let route = format!("{route} bits={bits} k={k} gs={group_size} m={m}");
        let dense_data: Vec<f32> = (0..(n * k) as usize)
            .map(|index| ((index % 127) as f32 - 63.0) / 512.0)
            .collect();
        let dense = reshape(&MlxArray::from_f32_slice(&dense_data), &[n, k], None);
        let quantized = quantize(
            &dense,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(astype(&quantized[1], MlxDtype::Bfloat16, None)),
            biases: Some(astype(&quantized[2], MlxDtype::Bfloat16, None)),
            group_size,
            bits,
            mode: "affine".to_owned(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
            weight_t_view: Default::default(),
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
        assert_eq!(routed.dtype(), MlxDtype::Bfloat16);
        assert_eq!(reference.dtype(), routed.dtype());
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
    fn wide_windows_of_seven_and_eight_rows_match_affine_qmm() {
        for m in [7, 8] {
            assert_route_matches_affine_qmm(SPLIT_K_TEST_N, SPLIT_K_TEST_N, m, "wide split-K");
            assert_route_matches_affine_qmm(MSG_ROUTE_N, MIN_ROUTE_N, m, "wide multi-simdgroup");
        }
    }

    #[test]
    fn six_and_eight_bit_windows_match_affine_qmm() {
        // K spans several 8-value pairs per lane and both multi-group and
        // single-group rows, so a misplaced 6-bit field or half-word cannot
        // cancel out.
        for bits in [6, 8] {
            for (k, group_size) in [(64, 32), (256, 64), (512, 128)] {
                for m in [3, 4, 8] {
                    assert_route_matches_affine_qmm_with(
                        SPLIT_K_TEST_N,
                        SPLIT_K_TEST_N,
                        m,
                        bits,
                        k,
                        group_size,
                        "split-K",
                    );
                }
            }
            for m in [3, 8] {
                assert_route_matches_affine_qmm_with(
                    MSG_ROUTE_N,
                    MIN_ROUTE_N,
                    m,
                    bits,
                    256,
                    64,
                    "multi-simdgroup",
                );
            }
            assert_route_matches_affine_qmm_with(2048, 0, 3, bits, 256, 64, "small split-K");
        }
    }

    #[test]
    fn six_bit_one_hot_rows_read_every_field_of_a_pair() {
        // Row r is one-hot at column 33 * r, which lands on position r of an
        // 8-value pair, so each output is one dequantized weight value and a
        // misplaced 6-bit field is off by at least one quantization step.
        let (k, group_size, m) = (256_i32, 64_i32, 8_i32);
        for (n, min_route_n) in [(SPLIT_K_TEST_N, SPLIT_K_TEST_N), (MSG_ROUTE_N, MIN_ROUTE_N)] {
            let dense_data: Vec<f32> = (0..(n * k) as usize)
                .map(|index| ((index * 37 % 1009) as f32 - 504.0) / 504.0)
                .collect();
            let dense = reshape(&MlxArray::from_f32_slice(&dense_data), &[n, k], None);
            let quantized = quantize(
                &dense,
                Some(group_size),
                Some(6),
                MlxQuantizationMode::Affine,
                None,
                None,
            );
            let scales = astype(&quantized[1], MlxDtype::Bfloat16, None);
            let biases = astype(&quantized[2], MlxDtype::Bfloat16, None);
            let weight = QuantizedWeight {
                weight: quantized[0].clone(),
                scales: Some(scales.clone()),
                biases: Some(biases.clone()),
                group_size,
                bits: 6,
                mode: "affine".to_owned(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
                weight_t_view: Default::default(),
            };
            let mut input_data = vec![0.0_f32; (m * k) as usize];
            for row in 0..m {
                input_data[(row * k + 33 * row) as usize] = 1.0;
            }
            let input = astype(
                &reshape(&MlxArray::from_f32_slice(&input_data), &[1, m, k], None),
                MlxDtype::Bfloat16,
                None,
            );

            let _guard = QwenMtpVerifyQmmGuard::arm(true);
            let routed = try_qwen_mtp_verify_qmm_for_min_n(&input, &weight, min_route_n)
                .unwrap_or_else(|| panic!("eligible 6-bit route n={n}"));
            let dequantized = mlx_sys::dequantize(
                &weight.weight,
                &scales,
                Some(&biases),
                Some(group_size),
                Some(6),
                None,
            );
            let routed_f32 = astype(&routed, MlxDtype::Float32, None);
            let expected_f32 = astype(&dequantized, MlxDtype::Float32, None);
            eval(&[&routed_f32, &expected_f32]);
            let routed = routed_f32.data_f32();
            let expected = expected_f32.data_f32();
            for row in 0..m {
                let column = (33 * row) as usize;
                for out in 0..n as usize {
                    let got = routed[row as usize * n as usize + out];
                    let want = expected[out * k as usize + column];
                    // One bf16 rounding of the same float on each side.
                    assert!(
                        (got - want).abs() <= want.abs() / 128.0 + 1e-6,
                        "n={n} row={row} out={out}: got {got}, want {want}"
                    );
                }
            }
        }
    }

    /// Per-call cost of the verify route vs MLX's stock QMM at Qwen 3.8 27B
    /// projection shapes (`m = 1` is the stock single-row decode baseline).
    /// Values are arbitrary bit patterns: neither path branches on data.
    /// Run: `cargo test -p ax-engine-mlx --release --lib bench_affine_verify_qmm
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe; run on Apple Silicon with --ignored --nocapture"]
    fn bench_affine_verify_qmm_shapes() {
        const CALLS: usize = 16;
        const REPEATS: usize = 7;
        let time_us = |build: &dyn Fn(&MlxArray) -> MlxArray, inputs: &[MlxArray]| {
            let mut best = f64::INFINITY;
            for _ in 0..REPEATS {
                let start = std::time::Instant::now();
                let outputs: Vec<MlxArray> = inputs.iter().map(build).collect();
                eval(&outputs.iter().collect::<Vec<_>>());
                best = best.min(start.elapsed().as_secs_f64() * 1e6 / CALLS as f64);
            }
            best
        };
        println!("bits      n      k  m   stock_us  route_us  route/stock  stock_m/stock_1");
        for bits in [4, 6, 8] {
            for (n, k) in [(17_408, 5_120), (16_384, 5_120), (248_320, 5_120)] {
                let group_size = 32;
                let words = k * bits / 32;
                let packed = reshape(
                    &arange(
                        0.0,
                        f64::from(n) * f64::from(words),
                        1.0,
                        MlxDtype::Uint32,
                        None,
                    ),
                    &[n, words],
                    None,
                );
                let scales = zeros(&[n, k / group_size], MlxDtype::Bfloat16, None);
                let biases = zeros(&[n, k / group_size], MlxDtype::Bfloat16, None);
                eval(&[&packed, &scales, &biases]);
                let weight = QuantizedWeight {
                    weight: packed,
                    scales: Some(scales),
                    biases: Some(biases),
                    group_size,
                    bits,
                    mode: "affine".to_owned(),
                    linear_bias: None,
                    decode_weight_t: None,
                    decode_q2_weight: None,
                    decode_q2_scales: None,
                    decode_q2_biases: None,
                    weight_t_view: Default::default(),
                };
                let stock = |x: &MlxArray| {
                    quantized_matmul_with_mode(
                        x,
                        &weight.weight,
                        weight.scales.as_ref().unwrap(),
                        weight.biases.as_ref(),
                        true,
                        Some(group_size),
                        Some(bits),
                        MlxQuantizationMode::Affine,
                        None,
                    )
                };
                let inputs_for = |m: i32| -> Vec<MlxArray> {
                    let inputs: Vec<MlxArray> = (0..CALLS)
                        .map(|call| {
                            let start = call as f64;
                            astype(
                                &reshape(
                                    &arange(
                                        start,
                                        start + f64::from(m * k),
                                        1.0,
                                        MlxDtype::Float32,
                                        None,
                                    ),
                                    &[1, m, k],
                                    None,
                                ),
                                MlxDtype::Bfloat16,
                                None,
                            )
                        })
                        .collect();
                    eval(&inputs.iter().collect::<Vec<_>>());
                    inputs
                };
                let single = inputs_for(1);
                let _ = time_us(&stock, &single);
                let stock_1 = time_us(&stock, &single);
                for m in [3, 4, 8] {
                    let inputs = inputs_for(m);
                    let _guard = QwenMtpVerifyQmmGuard::arm(true);
                    let route = |x: &MlxArray| {
                        try_qwen_mtp_verify_qmm_for_min_n(x, &weight, 0)
                            .expect("benchmark shape must be eligible")
                    };
                    let _ = time_us(&stock, &inputs);
                    let _ = time_us(&route, &inputs);
                    let stock_m = time_us(&stock, &inputs);
                    let route_m = time_us(&route, &inputs);
                    println!(
                        "{bits:>4} {n:>6} {k:>6} {m:>2} {stock_m:>10.1} {route_m:>9.1} {:>12.3} {:>16.2}",
                        route_m / stock_m,
                        stock_m / stock_1
                    );
                }
            }
        }
    }

    #[test]
    fn four_partition_split_k_matches_small_projection() {
        assert_route_matches_affine_qmm(2048, 0, 3, "small split-K");
    }

    #[test]
    fn multi_simdgroup_matches_affine_qmm_on_large_n_shape() {
        assert_route_matches_affine_qmm(MSG_ROUTE_N, MIN_ROUTE_N, 3, "multi-simdgroup");
    }

    fn mxfp4_test_weight(n: i32, k: i32) -> QuantizedWeight {
        let dense_data: Vec<f32> = (0..(n * k) as usize)
            .map(|index| ((index * 7 % 251) as f32 - 125.0) / 512.0)
            .collect();
        let dense = astype(
            &reshape(&MlxArray::from_f32_slice(&dense_data), &[n, k], None),
            MlxDtype::Bfloat16,
            None,
        );
        let quantized = quantize(
            &dense,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
        QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_owned(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
            weight_t_view: Default::default(),
        }
    }

    fn mxfp4_test_input(m: i32, k: i32) -> MlxArray {
        let data: Vec<f32> = (0..(m * k) as usize)
            .map(|index| ((index * 13 % 61) as f32 - 30.0) / 64.0)
            .collect();
        astype(
            &reshape(&MlxArray::from_f32_slice(&data), &[1, m, k], None),
            MlxDtype::Bfloat16,
            None,
        )
    }

    fn mxfp4_reference(x: &MlxArray, weight: &QuantizedWeight) -> MlxArray {
        quantized_matmul_with_mode(
            x,
            &weight.weight,
            weight.scales.as_ref().unwrap(),
            None,
            true,
            Some(weight.group_size),
            Some(weight.bits),
            MlxQuantizationMode::Mxfp4,
            None,
        )
    }

    #[test]
    fn mxfp4_window_matches_mxfp4_qmm_for_every_supported_row_count() {
        // K covers one and several 256-value steps; N is not a multiple of the
        // tile in the second case beyond the 4-column requirement.
        for (n, k) in [(64, 256), (132, 768)] {
            let weight = mxfp4_test_weight(n, k);
            let _guard = QwenMtpVerifyQmmGuard::arm(true);
            for m in MXFP4_WINDOW_MIN_ROWS..=MXFP4_WINDOW_MAX_ROWS {
                let input = mxfp4_test_input(m, k);
                let routed = mxfp4_window_qmm(&input, &weight, 0)
                    .unwrap_or_else(|| panic!("eligible MXFP4 window route m={m} n={n} k={k}"));
                let reference = mxfp4_reference(&input, &weight);
                assert_eq!(routed.dtype(), MlxDtype::Bfloat16);
                assert_eq!(routed.shape(), vec![1, m, n]);
                let routed_f32 = astype(&routed, MlxDtype::Float32, None);
                let reference_f32 = astype(&reference, MlxDtype::Float32, None);
                eval(&[&routed_f32, &reference_f32]);
                let max_abs = routed_f32
                    .data_f32()
                    .iter()
                    .zip(reference_f32.data_f32())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_abs <= 0.03125,
                    "m={m} n={n} k={k}: max abs diff {max_abs}"
                );
            }
        }
    }

    #[test]
    fn mxfp4_window_declines_what_it_cannot_serve() {
        let weight = mxfp4_test_weight(64, 256);
        let input = mxfp4_test_input(4, 256);
        // Not armed: no route.
        assert!(mxfp4_window_qmm(&input, &weight, 0).is_none());
        let _guard = QwenMtpVerifyQmmGuard::arm(true);
        assert!(mxfp4_window_qmm(&input, &weight, 0).is_some());
        // One and two rows stay on MLX's own qmv path; nine rows exceed the window.
        for m in [1, 2, 9] {
            assert!(
                mxfp4_window_qmm(&mxfp4_test_input(m, 256), &weight, 0).is_none(),
                "m={m}"
            );
        }
        // K not a multiple of the 256-value step.
        let narrow = mxfp4_test_weight(64, 160);
        assert!(mxfp4_window_qmm(&mxfp4_test_input(4, 160), &narrow, 0).is_none());
        // Output narrower than the configured floor.
        assert!(mxfp4_window_qmm(&input, &weight, 65).is_none());
        // Affine weights (they carry group biases) never take the MXFP4 route.
        // A 4-bit/group-32 weight without biases is MXFP4 whatever its label,
        // which is how `mlx_quantization_mode` treats real packs.
        let mut affine = mxfp4_test_weight(64, 256);
        affine.mode = "affine".to_owned();
        affine.biases = affine.scales.clone();
        assert!(mxfp4_window_qmm(&input, &affine, 0).is_none());
        let mut mislabeled = mxfp4_test_weight(64, 256);
        mislabeled.mode = "affine".to_owned();
        assert!(mxfp4_window_qmm(&input, &mislabeled, 0).is_some());
        // An unavailable kernel falls back instead of panicking.
        FORCE_VERIFY_QMM_KERNEL_UNAVAILABLE.store(true, std::sync::atomic::Ordering::Release);
        let unavailable = mxfp4_window_qmm(&input, &weight, 0);
        FORCE_VERIFY_QMM_KERNEL_UNAVAILABLE.store(false, std::sync::atomic::Ordering::Release);
        assert!(unavailable.is_none());
    }
}
