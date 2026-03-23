// AX Engine — Matrix multiply compute shaders
//
// Two kernels optimized for LLM inference:
//   1. matmul_tiled_f32  — general tiled matmul for prefill (M > 1)
//   2. matvec_f32        — optimized matrix-vector for decode (N = 1)
//
// All matrices are row-major f32. Computes C = A × B.
//   A: M × K
//   B: K × N
//   C: M × N

#include <metal_stdlib>
using namespace metal;

// ── Tiled General Matmul ────────────────────────────────────────────────
//
// Each threadgroup computes a TILE × TILE output tile using shared memory.
// Threadgroup size: TILE × TILE = 16 × 16 = 256 threads.

constant uint TILE = 16;

kernel void matmul_tiled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid             [[thread_position_in_grid]],
    uint2 lid             [[thread_position_in_threadgroup]]
) {
    uint row = gid.y;
    uint col = gid.x;

    threadgroup float tile_A[TILE][TILE];
    threadgroup float tile_B[TILE][TILE];

    float sum = 0.0f;

    uint n_tiles = (K + TILE - 1) / TILE;

    for (uint t = 0; t < n_tiles; t++) {
        // Load A tile: row from A, column from k-tile
        uint a_col = t * TILE + lid.x;
        tile_A[lid.y][lid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // Load B tile: row from k-tile, column from B
        uint b_row = t * TILE + lid.y;
        tile_B[lid.y][lid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate partial dot product from this tile
        for (uint i = 0; i < TILE; i++) {
            sum += tile_A[lid.y][i] * tile_B[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ── Simdgroup Matrix Matmul ──────────────────────────────────────────
//
// High-performance matmul using Apple Silicon simdgroup_matrix hardware.
// Each threadgroup computes a 32×32 output tile using 128 threads
// (4 simdgroups). Each simdgroup computes an 8×32 strip via
// simdgroup_multiply_accumulate with 8×8 sub-tiles.
//
// This provides ~4x throughput over the 16×16 naive tiled matmul
// for compute-bound cases (prefill).
//
// Threadgroup: 128 threads (4 simdgroups × 32 threads).
// Grid: ceil(N/32) × ceil(M/32) threadgroups.

constant uint SG_BM = 32;  // Output tile rows
constant uint SG_BN = 32;  // Output tile cols
constant uint SG_BK = 32;  // K-dimension tile size
constant uint SG_TG = 128; // Threads per threadgroup

kernel void matmul_simdgroup_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 group_id        [[threadgroup_position_in_grid]],
    uint  tid             [[thread_index_in_threadgroup]],
    uint  simd_id         [[simdgroup_index_in_threadgroup]],
    uint  simd_lane       [[thread_index_in_simdgroup]]
) {
    // Output tile position in global matrix
    uint tile_row = group_id.y * SG_BM;
    uint tile_col = group_id.x * SG_BN;

    // Shared memory for A and B tiles
    threadgroup float tg_A[SG_BM * SG_BK];  // 32×32 = 4 KB
    threadgroup float tg_B[SG_BK * SG_BN];  // 32×32 = 4 KB

    // Each simdgroup computes 8 rows of the 32×32 output tile
    // simd_id 0 → rows 0..7, simd_id 1 → rows 8..15, etc.
    // 4 accumulators per simdgroup: 8×8 sub-tiles covering 8×32
    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint n_k_tiles = (K + SG_BK - 1) / SG_BK;

    for (uint kt = 0; kt < n_k_tiles; kt++) {
        uint k_offset = kt * SG_BK;

        // Cooperative load A tile: 32×32 = 1024 floats, 128 threads → 8 per thread
        for (uint i = tid; i < SG_BM * SG_BK; i += SG_TG) {
            uint r = i / SG_BK;
            uint c = i % SG_BK;
            uint gr = tile_row + r;
            uint gc = k_offset + c;
            tg_A[r * SG_BK + c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        // Cooperative load B tile: 32×32 = 1024 floats, 128 threads → 8 per thread
        for (uint i = tid; i < SG_BK * SG_BN; i += SG_TG) {
            uint r = i / SG_BN;
            uint c = i % SG_BN;
            uint gr = k_offset + r;
            uint gc = tile_col + c;
            tg_B[r * SG_BN + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply-accumulate using simdgroup 8×8 matrix operations
        // For each K sub-block (4 iterations of 8):
        for (uint kk = 0; kk < SG_BK / 8; kk++) {
            // Load 8×8 A fragment for this simdgroup's row range
            simdgroup_float8x8 a_frag;
            simdgroup_load(a_frag, &tg_A[simd_id * 8 * SG_BK + kk * 8], SG_BK);

            // Load 4 B fragments (8×8 each) covering all 32 output columns
            simdgroup_float8x8 b0, b1, b2, b3;
            simdgroup_load(b0, &tg_B[kk * 8 * SG_BN + 0],  SG_BN);
            simdgroup_load(b1, &tg_B[kk * 8 * SG_BN + 8],  SG_BN);
            simdgroup_load(b2, &tg_B[kk * 8 * SG_BN + 16], SG_BN);
            simdgroup_load(b3, &tg_B[kk * 8 * SG_BN + 24], SG_BN);

            simdgroup_multiply_accumulate(acc0, a_frag, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a_frag, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a_frag, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a_frag, b3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulators to threadgroup memory, then cooperative write to global
    threadgroup float out_tile[SG_BM * SG_BN];  // 32×32

    simdgroup_store(acc0, &out_tile[simd_id * 8 * SG_BN + 0],  SG_BN);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * SG_BN + 8],  SG_BN);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * SG_BN + 16], SG_BN);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * SG_BN + 24], SG_BN);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative store to global memory: 1024 floats, 128 threads → 8 per thread
    for (uint i = tid; i < SG_BM * SG_BN; i += SG_TG) {
        uint r = i / SG_BN;
        uint c = i % SG_BN;
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < M && gc < N) {
            C[gr * N + gc] = out_tile[r * SG_BN + c];
        }
    }
}

// ── Optimized Matrix-Vector Multiply ────────────────────────────────────
//
// Specialized for N=1 (decode step). Each threadgroup computes one row's
// dot product using parallel reduction:
//   y[row] = dot(A[row, :], x)
//
// Threadgroup size: 256 threads.
// Grid: M threadgroups × 1.

constant uint MATVEC_TG_SIZE = 256;

kernel void matvec_f32(
    device const float* A    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float* y          [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    uint row                 [[threadgroup_position_in_grid]],
    uint lid                 [[thread_index_in_threadgroup]],
    uint simd_lane           [[thread_index_in_simdgroup]],
    uint simd_id             [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    device const float* a_row = A + row * K;

    // Each thread accumulates a strided partial sum
    float sum = 0.0f;
    for (uint i = lid; i < K; i += MATVEC_TG_SIZE) {
        sum += a_row[i] * x[i];
    }

    // SIMD-level reduction (32 threads → 1 value)
    sum = simd_sum(sum);

    // Cross-SIMD reduction via threadgroup memory
    // MATVEC_TG_SIZE / 32 = 8 SIMD groups
    threadgroup float simd_sums[8];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD group reduces across all groups
    if (simd_id == 0) {
        constexpr uint n_groups = MATVEC_TG_SIZE / 32;
        sum = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) {
            y[row] = sum;
        }
    }
}
