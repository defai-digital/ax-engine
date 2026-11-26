// AX port of MLX's sorted NAX MXFP4 gather quantized matmul (ADR-053), with
// the row tile height as a parameter and empty row fragments skipped.
//
// Ported from MLX (https://github.com/ml-explore/mlx) at commit
// 914732419f255f8b45ff1ee253ad33ab0a3ce623:
//   mlx/backend/metal/kernels/fp_quantized_nax.h: fp_gather_qmm_rhs_nax
//     (lines 1264-1474) for transpose, MXFP4, group_size 32, aligned N and K,
//     and the QuantizedBlockLoader paths it uses (lines 72-190)
//   mlx/backend/metal/kernels/utils.h: schedule_row_tile (lines 501-542)
//   mlx/backend/metal/kernels/steel/gemm/nax.h: the BaseNAXFrag load and store
//     layouts (lines 61-96, 185-222, 308-351) and tile_matmad_nax's loop order
//     (lines 830-895)
// Copyright © 2023-2026 Apple Inc. Licensed under the MIT License; the full
// notice is in THIRD_PARTY_NOTICES at the repository root.
//
// This file follows verify_qmm_mx_nax.metal in one kernel header and uses its
// ax_nax::frag_coord, ax_nax::mma_16x32 and ax_nax::dequantize_mx_x4.

namespace ax_gq {

// The row tile `tile` of a gather sorted by expert: tiles never span two
// experts. `offsets[g]` is the first row of expert g. Each lane scans one
// expert of 32 at a time. Returns false when `tile` is past the last tile.
template <int BM>
METAL_FUNC bool schedule_row_tile(
    const device int* offsets,
    const int num_groups,
    const int M,
    const int tile,
    const uint simd_lane_id,
    thread int& row,
    thread int& group,
    thread short& rows) {
  int tiles_before = 0;
  for (int e = 0; e < num_groups; e += 32) {
    const int g = e + int(simd_lane_id);
    int start = M;
    int end = M;
    if (g < num_groups) {
      start = offsets[g];
      end = g + 1 < num_groups ? offsets[g + 1] : M;
    }
    const int n = (end - start + BM - 1) / BM;
    const int tiles_through = tiles_before + simd_prefix_inclusive_sum(n);
    const ushort owner_lane = ushort(simd_sum(int(tiles_through <= tile)));
    if (owner_lane < 32) {
      group = e + owner_lane;
      row = simd_shuffle(start, owner_lane) +
          (tile - simd_shuffle(tiles_through - n, owner_lane)) * BM;
      rows = short(min(BM, simd_shuffle(end, owner_lane) - row));
      return true;
    }
    tiles_before = simd_shuffle(tiles_through, ushort(31));
  }
  return false;
}

// y[M, N] = x[M, K] * dequant(w[indices[M]])^T for rows sorted by expert.
//
// One threadgroup of WM * 2 simdgroups computes a tile of up to BM rows of
// one expert by 64 output columns. Each simdgroup owns BM / WM rows by 32
// columns. Per 64-deep K step the threadgroup dequantizes the expert's
// 64 x 64 weight tile into `ws` once, and every simdgroup that holds rows
// multiplies its activations with it on the tensor units, 16 K values at a
// time in ascending K, which is the order MLX's kernel accumulates in: the
// outputs are bit-identical to it.
//
// MLX sizes its row tiles by the pairs per expert of the whole stack and
// takes 32 rows for a Flash Next prefill. Real routing concentrates on a
// fraction of the experts, and a 32-row tile dequantizes their weights twice
// as often as a 64-row one. A 16-row fragment past the tile's last row is
// skipped, so a short run costs what it costs in a 32-row tile.
template <typename T, int BM, int WM, int K, int N>
METAL_FUNC void gather_mxfp4_nax(
    const device T* x,
    const device uint32_t* w,
    const device uint8_t* scales,
    const device int* offsets,
    device T* y,
    const int M,
    const int num_groups,
    const uint3 tid,
    const uint simd_gid,
    threadgroup bfloat* ws) {
  constexpr int BN = 64;
  constexpr int BK = 64;
  constexpr int WN = 2;
  constexpr int BKP = BK + 8;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  static_assert(BM % (16 * WM) == 0, "simdgroup rows are whole fragments");
  static_assert(WM == 1 || WM == 2 || WM == 4, "1, 2 or 4 row simdgroups");
  static_assert(K % BK == 0 && N % BN == 0, "aligned K and N only");
  static_assert(TN == 2, "two column fragments per simdgroup");

  const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
  int y_row;
  int group;
  short tile_rows;
  if (!schedule_row_tile<BM>(
          offsets, num_groups, M, int(tid.y), uint(lane), y_row, group, tile_rows)) {
    return;
  }
  const int y_col = int(tid.x) * BN;
  constexpr int row_bytes = K / 2;
  constexpr int row_scales = K / 32;

  const short2 coord = ax_nax::frag_coord();
  const short fm = coord.y;
  const short fn = coord.x;
  const short tm = SM * short(simd_gid / WN);
  const short tn = SN * short(simd_gid % WN);
  // Rows of the tile this simdgroup holds, and its fragments with any.
  const short sg_rows = short(clamp(int(tile_rows) - int(tm), 0, int(SM)));
  const short sg_frags = (sg_rows + 15) / 16;

  // Weight loader: the threadgroup's 64 * WM threads share the 64 rows of 32
  // bytes; a thread dequantizes 32 / WM bytes of one row.
  constexpr short loader_bytes = 32 / WM;
  const short thread_idx = short(simd_gid) * 32 + short(lane);
  const short load_row = thread_idx / WM;
  const short load_byte = (thread_idx % WM) * loader_bytes;
  const device uint8_t* w_src = (const device uint8_t*)w +
      (size_t(group) * N + size_t(y_col + load_row)) * size_t(row_bytes) + load_byte;
  const device uint8_t* s_src = scales +
      (size_t(group) * N + size_t(y_col + load_row)) * size_t(row_scales);
  threadgroup bfloat* w_dst = ws + load_row * BKP + load_byte * 2;

  // Activation rows this lane reads: rows fm and fm + 8 of each fragment. A
  // row past the tile belongs to the next expert, and one past the end of x
  // is clamped to the last row; neither is stored.
  const device T* x_row[TM][2];
  _Pragma("clang loop unroll(full)")
  for (short mm = 0; mm < TM; mm++) {
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; i++) {
      const int row = min(y_row + int(tm) + 16 * mm + fm + 8 * i, M - 1);
      x_row[mm][i] = x + size_t(row) * size_t(K) + fn;
    }
  }

  ax_nax::frag_t<float> acc[TM][TN];
  _Pragma("clang loop unroll(full)")
  for (short mm = 0; mm < TM; mm++) {
    _Pragma("clang loop unroll(full)")
    for (short nn = 0; nn < TN; nn++) {
      acc[mm][nn] = ax_nax::frag_t<float>(0);
    }
  }

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (short q = 0; q < loader_bytes / 2; q++) {
      const short byte = load_byte + 2 * q;
      const uint packed = uint(*(const device ushort*)(w_src + k / 2 + 2 * q));
      const vec<bfloat, 4> v = ax_nax::dequantize_mx_x4<bfloat, 4>(
          packed, s_src[k / 32 + byte / 16]);
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; j++) {
        w_dst[4 * q + j] = v[j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    _Pragma("clang loop unroll(disable)")
    for (short k1 = 0; k1 < BK; k1 += 32) {
      if (sg_frags > 0) {
        ax_nax::frag_t<bfloat> b[2][TN];
        _Pragma("clang loop unroll(full)")
        for (short f = 0; f < 2; f++) {
          _Pragma("clang loop unroll(full)")
          for (short nn = 0; nn < TN; nn++) {
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 2; i++) {
              const threadgroup bfloat* src =
                  ws + (tn + 16 * nn + fm + 8 * i) * BKP + k1 + 16 * f + fn;
              _Pragma("clang loop unroll(full)")
              for (short j = 0; j < 4; j++) {
                b[f][nn][4 * i + j] = src[j];
              }
            }
          }
        }
        _Pragma("clang loop unroll(full)")
        for (short mm = 0; mm < TM; mm++) {
          if (mm < sg_frags) {
            _Pragma("clang loop unroll(full)")
            for (short f = 0; f < 2; f++) {
              ax_nax::frag_t<T> a;
              _Pragma("clang loop unroll(full)")
              for (short i = 0; i < 2; i++) {
                const device T* src = x_row[mm][i] + k + k1 + 16 * f;
                _Pragma("clang loop unroll(full)")
                for (short j = 0; j < 4; j++) {
                  a[4 * i + j] = src[j];
                }
              }
              ax_nax::mma_16x32<float, T, bfloat>(acc[mm][0], acc[mm][1], a, b[f][0], b[f][1]);
            }
          }
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (short mm = 0; mm < TM; mm++) {
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; i++) {
      const short r = 16 * mm + fm + 8 * i;
      if (r < sg_rows) {
        device T* dst = y + size_t(y_row + tm + r) * size_t(N) + y_col + tn + fn;
        _Pragma("clang loop unroll(full)")
        for (short nn = 0; nn < TN; nn++) {
          _Pragma("clang loop unroll(full)")
          for (short j = 0; j < 4; j++) {
            dst[16 * nn + j] = static_cast<T>(acc[mm][nn][4 * i + j]);
          }
        }
      }
    }
  }
}

}  // namespace ax_gq
