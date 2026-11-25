// AX port of MLX's skinny-row NAX MX-format quantized GEMV (ADR-053).
//
// Ported from MLX (https://github.com/ml-explore/mlx) at commit
// 914732419f255f8b45ff1ee253ad33ab0a3ce623:
//   mlx/backend/metal/kernels/fp_quantized_nax.h: fp_qmv_nax (lines 1485-1628)
//     and dequantize_fp_x4 (lines 342-374), group_size 32 only
//   mlx/backend/metal/kernels/steel/gemm/nax.h: BaseNAXFrag::get_coord
//     (lines 45-51), BaseNAXFrag::mma 16x32x16 (lines 387-455), and the
//     NAXTile threadgroup store/load and store_rows paths it uses
//   mlx/backend/metal/kernels/fp8.h: fp8_e8m0 to float (lines 75-80)
// Copyright © 2023-2026 Apple Inc. Licensed under the MIT License; the full
// notice is in THIRD_PARTY_NOTICES at the repository root.
#include <metal_simdgroup>
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace ax_nax {

template <typename U>
using frag_t = vec<U, 8>;

METAL_FUNC short2 frag_coord() {
  const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2{fn, fm};
}

// C[16x32] += A[16x16] * B[32x16]^T on the neural accelerators.
template <typename CType, typename AType, typename BType>
METAL_FUNC void mma_16x32(
    thread frag_t<CType>& c0,
    thread frag_t<CType>& c1,
    const thread frag_t<AType>& a,
    const thread frag_t<BType>& b0,
    const thread frag_t<BType>& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b = op.template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      CType>();
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; i++) {
    ct_a[i] = a[i];
    ct_b[i] = b0[i];
    ct_b[8 + i] = b1[i];
    ct_c[i] = c0[i];
    ct_c[8 + i] = c1[i];
  }
  op.run(ct_a, ct_b, ct_c);
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; i++) {
    c0[i] = ct_c[i];
    c1[i] = ct_c[8 + i];
  }
}

METAL_FUNC float e8m0_to_float(uint8_t bits) {
  const uint out = (bits == 0)
      ? 0x400000u
      : (bits == 0xFF ? 0x7FC00000u : (uint(bits) << 23));
  return as_type<float>(out);
}

// Four MX elements (FP4 E2M1 or FP8 E4M3) times their E8M0 group scale.
template <typename Wtype, int bits>
METAL_FUNC vec<Wtype, 4> dequantize_mx_x4(uint q, uint8_t s) {
  half2 a;
  half2 b;
  if constexpr (bits == 4) {
    const uint lo = (q & 0x000f) | ((q & 0x00f0) << 12);
    const uint hi = ((q >> 8) & 0x000f) | ((q & 0xf000) << 4);
    a = as_type<half2>(((lo & 0x00070007) << 9) | ((lo & 0x00080008) << 12));
    b = as_type<half2>(((hi & 0x00070007) << 9) | ((hi & 0x00080008) << 12));
  } else {
    const uint lo = (q & 0x000000ff) | ((q & 0x0000ff00) << 8);
    const uint hi = ((q >> 16) & 0x000000ff) | ((q >> 8) & 0x00ff0000);
    a = as_type<half2>(((lo & 0x00800080) << 8) | ((lo & 0x007f007f) << 7));
    b = as_type<half2>(((hi & 0x00800080) << 8) | ((hi & 0x007f007f) << 7));
  }
  const float scale = e8m0_to_float(s) * (bits == 4 ? 16384.0f : 256.0f);
  return vec<Wtype, 4>(
      Wtype(float(a.x) * scale),
      Wtype(float(a.y) * scale),
      Wtype(float(b.x) * scale),
      Wtype(float(b.y) * scale));
}

// y[M, N] = x[M, K] * dequant(w[N, K])^T for 1 <= M <= 16 rows. One
// threadgroup of WK simdgroups covers BN output columns; each simdgroup
// strides over K in BK steps and the partial tiles are summed through
// threadgroup memory `ps` of (WK - 1) * 16 * BN floats.
template <typename T, int bits, int WK, int BN, int K, int N>
METAL_FUNC void mx_qmv_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const int M,
    const uint tgx,
    const uint simd_gid,
    threadgroup float* ps) {
  static_assert(BN % 32 == 0, "BN should be divisible by 32");
  static_assert(bits == 4 || bits == 8, "MXFP4 or MXFP8");
  using Wtype = bfloat;
  constexpr int group_size = 32;
  constexpr int BK = 128;
  constexpr short TK = BK / 16;
  constexpr short TN = BN / 16;
  constexpr int pack_factor = 8 / bits;
  constexpr short loads = bits / 4;
  constexpr int row_bytes = K / pack_factor;
  constexpr int row_scales = K / group_size;

  const short2 coord = frag_coord();
  const short fm = coord.y;
  const short fn = coord.x;
  const int k_lane = 8 * fn;

  const int y_col = int(tgx) * BN;
  auto wl = (const device uint8_t*)w +
      (y_col + fm) * static_cast<int64_t>(row_bytes) + k_lane / pack_factor;
  scales += (y_col + fm) * static_cast<int64_t>(row_scales) + k_lane / group_size;
  x += fm * static_cast<int64_t>(K) + k_lane;
  y += y_col;
  const bool x_rows[2] = {fm < M, fm + 8 < M};

  frag_t<float> acc[TN];
  _Pragma("clang loop unroll(full)")
  for (short t = 0; t < TN; t++) {
    acc[t] = frag_t<float>(0);
  }

  for (int k = int(simd_gid) * BK; k < K; k += WK * BK) {
    vec<T, 8> xs[2][4];
    _Pragma("clang loop unroll(full)")
    for (short r = 0; r < 2; r++) {
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 4; i++) {
        xs[r][i] = x_rows[r]
            ? *(const device vec<T, 8>*)(x + r * 8 * static_cast<int64_t>(K) + k + 8 * i)
            : vec<T, 8>(0);
      }
    }
    uint4 ws[TN][2][loads];
    uint8_t ss[TN][2];
    _Pragma("clang loop unroll(full)")
    for (short t = 0; t < TN; t++) {
      _Pragma("clang loop unroll(full)")
      for (short r = 0; r < 2; r++) {
        const int row = 16 * t + 8 * r;
        const device uint8_t* wt =
            wl + row * static_cast<int64_t>(row_bytes) + k / pack_factor;
        const device uint8_t* st =
            scales + row * static_cast<int64_t>(row_scales) + k / group_size;
        _Pragma("clang loop unroll(full)")
        for (short l = 0; l < loads; l++) {
          ws[t][r][l] = *(const device uint4*)(wt + 16 * l);
        }
        ss[t][r] = *st;
      }
    }

    _Pragma("clang loop unroll(full)")
    for (short f = 0; f < TK; f++) {
      frag_t<T> a;
      _Pragma("clang loop unroll(full)")
      for (short r = 0; r < 2; r++) {
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; j++) {
          a[4 * r + j] = xs[r][f / 2][4 * (f % 2) + j];
        }
      }
      frag_t<Wtype> b[TN];
      _Pragma("clang loop unroll(full)")
      for (short t = 0; t < TN; t++) {
        _Pragma("clang loop unroll(full)")
        for (short r = 0; r < 2; r++) {
          uint q;
          if constexpr (bits == 4) {
            q = ws[t][r][0][f / 2] >> (16 * (f % 2));
          } else {
            q = ws[t][r][f / 4][f % 4];
          }
          const vec<Wtype, 4> v = dequantize_mx_x4<Wtype, bits>(q, ss[t][r]);
          _Pragma("clang loop unroll(full)")
          for (short j = 0; j < 4; j++) {
            b[t][4 * r + j] = v[j];
          }
        }
      }
      _Pragma("clang loop unroll(full)")
      for (short t = 0; t < TN; t += 2) {
        mma_16x32<float, T, Wtype>(acc[t], acc[t + 1], a, b[t], b[t + 1]);
      }
    }
  }

  // Each lane holds rows fm and fm + 8, columns fn..fn + 3 of every 16x16 frag.
  if (simd_gid > 0) {
    threadgroup float* dst = ps + (simd_gid - 1) * 16 * BN + fm * BN + fn;
    _Pragma("clang loop unroll(full)")
    for (short t = 0; t < TN; t++) {
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 2; i++) {
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; j++) {
          dst[i * 8 * BN + 16 * t + j] = acc[t][i * 4 + j];
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    for (int p = 0; p < WK - 1; p++) {
      const threadgroup float* src = ps + p * 16 * BN + fm * BN + fn;
      _Pragma("clang loop unroll(full)")
      for (short t = 0; t < TN; t++) {
        _Pragma("clang loop unroll(full)")
        for (short i = 0; i < 2; i++) {
          _Pragma("clang loop unroll(full)")
          for (short j = 0; j < 4; j++) {
            acc[t][i * 4 + j] += src[i * 8 * BN + 16 * t + j];
          }
        }
      }
    }
    device T* dst = y + fm * N + fn;
    const int lx = M - fm;
    _Pragma("clang loop unroll(full)")
    for (short t = 0; t < TN; t++) {
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 2; i++) {
        if (i * 8 < lx) {
          _Pragma("clang loop unroll(full)")
          for (short j = 0; j < 4; j++) {
            dst[i * 8 * N + 16 * t + j] = static_cast<T>(acc[t][i * 4 + j]);
          }
        }
      }
    }
  }
}

}  // namespace ax_nax
