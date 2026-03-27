// AX Engine — Dequantization compute shaders
//
// GPU-accelerated dequantization for Q4_0, Q4_K, Q5_K, and Q6_K formats.
//
// Loop unroll hint — matches llama.cpp's FOR_UNROLL for inner accumulation loops.
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
//
// Standalone dequant (quantized → f32):
//   1. dequant_q4_0          — Q4_0 blocks → f32
//   2. dequant_q4_k          — Q4_K blocks → f32
//
// Fused dequant + matrix-vector (decode, N=1):
//   3. dequant_matvec_q4_0   — y = dequant(A_q4_0) × x
//   4. dequant_matvec_q4_k   — y = dequant(A_q4_k) × x

#include <metal_stdlib>
using namespace metal;

// ── Q4_0 Block ─────────────────────────────────────────────────────────
//
// 18 bytes → 32 f32 values.
//   - 2 bytes:  f16 scale (d)
//   - 16 bytes: 32 × 4-bit quants packed 2 per byte
//
// Byte layout: byte[i] low nibble = element[i], high nibble = element[i+16]
// Dequant: output = (q - 8) * d

struct Q4_0_Block {
    half d;        // scale
    uchar qs[16];  // packed 4-bit quants
};

static_assert(sizeof(Q4_0_Block) == 18, "Q4_0_Block must be exactly 18 bytes");

constant uint Q4_0_BLOCK_VALUES = 32;

// ── Q8_0 Block ─────────────────────────────────────────────────────────
//
// 34 bytes → 32 f32 values.
//   - 2 bytes:  f16 scale (d)
//   - 32 bytes: 32 × i8 quants
//
// Dequant: output = q * d
struct Q8_0_Block {
    half d;
    char qs[32];
};

static_assert(sizeof(Q8_0_Block) == 34, "Q8_0_Block must be exactly 34 bytes");

constant uint Q8_0_BLOCK_VALUES = 32;

// ── Q4_K Block ─────────────────────────────────────────────────────────
//
// 144 bytes → 256 f32 values.
//   - 2 bytes:   f16 d    (super-block scale)
//   - 2 bytes:   f16 dmin (super-block min scale)
//   - 12 bytes:  scales[12] — 8 × 6-bit scales + 8 × 6-bit mins, packed
//   - 128 bytes: qs[128]    — 256 × 4-bit quants (2 per byte)
//
// 8 sub-blocks of 32 values, processed in 4 pairs.
// value = d * sc * nibble - dmin * m

struct Q4_K_Block {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

static_assert(sizeof(Q4_K_Block) == 144, "Q4_K_Block must be exactly 144 bytes");

constant uint Q4_K_BLOCK_VALUES = 256;

// ── Q5_K Block ─────────────────────────────────────────────────────────
//
// 176 bytes → 256 f32 values.
//   - 2 bytes:   f16 d    (super-block scale)
//   - 2 bytes:   f16 dmin (super-block min scale)
//   - 12 bytes:  scales[12] — 8 × 6-bit scales + 8 × 6-bit mins, packed
//   - 32 bytes:  qh[32]     — high-bit plane for the 5th quant bit
//   - 128 bytes: qs[128]    — 256 × low 4-bit quants (2 per byte)
//
// 8 sub-blocks of 32 values, processed in 4 pairs.
// value = d * sc * (ql | (qh_bit << 4)) - dmin * m

struct Q5_K_Block {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
};

static_assert(sizeof(Q5_K_Block) == 176, "Q5_K_Block must be exactly 176 bytes");

constant uint Q5_K_BLOCK_VALUES = 256;

// Extract 6-bit scale and min for sub-block j from packed scales array.
// Returns float2(scale, min).
inline float2 get_scale_min_q4k(uint j, device const uchar* scales) {
    if (j < 4) {
        return float2(float(scales[j] & 63),
                       float(scales[j + 4] & 63));
    } else {
        float sc = float((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
        float m  = float((scales[j + 4] >> 4)    | ((scales[j]     >> 6) << 4));
        return float2(sc, m);
    }
}

// ── Standalone Dequantization ──────────────────────────────────────────

kernel void dequant_q4_0(
    device const Q4_0_Block* blocks [[buffer(0)]],
    device float* output            [[buffer(1)]],
    constant uint& n_blocks         [[buffer(2)]],
    uint idx                        [[thread_position_in_grid]]
) {
    if (idx >= n_blocks) return;

    device const Q4_0_Block& blk = blocks[idx];
    float d = float(blk.d);
    device float* out = output + idx * Q4_0_BLOCK_VALUES;

    for (uint i = 0; i < 16; i++) {
        uchar byte = blk.qs[i];
        out[i]      = d * float(int(byte & 0x0F) - 8);
        out[i + 16] = d * float(int(byte >> 4) - 8);
    }
}

kernel void dequant_q4_k(
    device const Q4_K_Block* blocks [[buffer(0)]],
    device float* output            [[buffer(1)]],
    constant uint& n_blocks         [[buffer(2)]],
    uint idx                        [[thread_position_in_grid]]
) {
    if (idx >= n_blocks) return;

    device const Q4_K_Block& blk = blocks[idx];
    float d    = float(blk.d);
    float dmin = float(blk.dmin);
    device float* out = output + idx * Q4_K_BLOCK_VALUES;

    uint out_idx = 0;
    uint q_idx = 0;
    uint is = 0;

    for (uint pair = 0; pair < 4; pair++) {
        float2 sm1 = get_scale_min_q4k(is, blk.scales);
        float2 sm2 = get_scale_min_q4k(is + 1, blk.scales);

        float d1 = d * sm1.x;
        float m1 = dmin * sm1.y;
        float d2 = d * sm2.x;
        float m2 = dmin * sm2.y;

        for (uint l = 0; l < 32; l++) {
            uchar byte = blk.qs[q_idx + l];
            out[out_idx + l]      = d1 * float(byte & 0x0F) - m1;
            out[out_idx + l + 32] = d2 * float(byte >> 4)   - m2;
        }

        out_idx += 64;
        q_idx += 32;
        is += 2;
    }
}

// ── Q6_K Block ─────────────────────────────────────────────────────────
//
// 210 bytes → 256 f32 values.
//   - 128 bytes: ql[128] — lower 4 bits of each 6-bit quant
//   - 64 bytes:  qh[64]  — upper 2 bits of each 6-bit quant
//   - 16 bytes:  scales[16] — per-sub-block 8-bit signed scales
//   - 2 bytes:   d — f16 super-block scale
//
// 16 sub-blocks of 16 values each.
// q6 = (ql_nibble | (qh_2bits << 4)) - 32
// value = d * scale * q6

struct Q6_K_Block {
    uchar ql[128];    // lower 4 bits
    uchar qh[64];     // upper 2 bits
    char  scales[16]; // signed 8-bit per-sub-block scales
    half  d;          // super-block scale
};

static_assert(sizeof(Q6_K_Block) == 210, "Q6_K_Block must be exactly 210 bytes");

constant uint Q6_K_BLOCK_VALUES = 256;

// ── Standalone Q6_K Dequantization ──────────────────────────────────────

kernel void dequant_q6_k(
    device const Q6_K_Block* blocks [[buffer(0)]],
    device float* output            [[buffer(1)]],
    constant uint& n_blocks         [[buffer(2)]],
    uint idx                        [[thread_position_in_grid]]
) {
    if (idx >= n_blocks) return;

    device const Q6_K_Block& blk = blocks[idx];
    float d = float(blk.d);
    device float* out = output + idx * Q6_K_BLOCK_VALUES;

    uint ql_idx = 0;
    uint qh_idx = 0;
    uint sc_idx = 0;
    uint out_idx = 0;

    // Two groups of 128 values
    for (uint group = 0; group < 2; group++) {
        for (uint l = 0; l < 32; l++) {
            uint is = l / 16;

            int q1 = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
            int q2 = int((blk.ql[ql_idx + l + 32] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
            int q3 = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
            int q4 = int((blk.ql[ql_idx + l + 32] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;

            float sc1 = float(blk.scales[sc_idx + is]);
            float sc2 = float(blk.scales[sc_idx + is + 2]);
            float sc3 = float(blk.scales[sc_idx + is + 4]);
            float sc4 = float(blk.scales[sc_idx + is + 6]);

            out[out_idx + l]      = d * sc1 * float(q1);
            out[out_idx + l + 32] = d * sc2 * float(q2);
            out[out_idx + l + 64] = d * sc3 * float(q3);
            out[out_idx + l + 96] = d * sc4 * float(q4);
        }

        out_idx += 128;
        ql_idx += 64;
        qh_idx += 32;
        sc_idx += 8;
    }
}

// ── Fused Dequant + Simdgroup Matrix Multiply (Q4_K) ──────────────────
//
// C = dequant(A_q4k) × B, where:
//   A: M × (K/256) Q4_K blocks (quantized weight matrix, row-major)
//   B: K × N f32 (input activations, row-major)
//   C: M × N f32 (output, row-major)
//
// Uses simdgroup_matrix_8x8 for high-throughput matmul on Apple Silicon.
// Processes K in chunks of 64 (one Q4_K sub-block pair per K-tile).
//
// Threadgroup: 128 threads (4 simdgroups × 32 threads).
// Grid: ceil(N/32) × ceil(M/32) threadgroups.

constant uint DQ_BM = 32;   // Output tile rows
constant uint DQ_BN = 32;   // Output tile cols
constant uint DQ_BK = 64;   // K-tile size (1 Q4_K sub-block pair = 64 values)
constant uint DQ_TG = 128;  // Threads per threadgroup

kernel void dequant_matmul_simdgroup_q4_k(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_row = group_id.y * DQ_BM;
    uint tile_col = group_id.x * DQ_BN;

    threadgroup float tg_A[DQ_BM * DQ_BK];  // 32×64 = 8 KB
    threadgroup float tg_B[DQ_BK * DQ_BN];  // 64×32 = 8 KB

    // Each simdgroup computes 8 rows of the 32×32 output tile
    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    // Process K dimension in chunks of 64 (one sub-block pair)
    for (uint kt = 0; kt < K; kt += DQ_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DQ_BK;

        // Cooperative dequant A tile: 32 rows × 64 values
        // 128 threads → ~16 values per thread
        for (uint i = tid; i < DQ_BM * DQ_BK; i += DQ_TG) {
            uint r = i / DQ_BK;
            uint c = i % DQ_BK;
            uint global_r = tile_row + r;

            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d    = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);

                uchar byte = blk.qs[pair * 32 + (c < 32 ? c : c - 32)];
                if (c < 32) {
                    tg_A[r * DQ_BK + c] = d * sm1.x * float(byte & 0x0F) - dmin * sm1.y;
                } else {
                    tg_A[r * DQ_BK + c] = d * sm2.x * float(byte >> 4) - dmin * sm2.y;
                }
            } else {
                tg_A[r * DQ_BK + c] = 0.0f;
            }
        }

        // Cooperative load B tile: 64×32 f32
        for (uint i = tid; i < DQ_BK * DQ_BN; i += DQ_TG) {
            uint r = i / DQ_BN;
            uint c = i % DQ_BN;
            uint gr = kt + r;
            uint gc = tile_col + c;
            tg_B[r * DQ_BN + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Simdgroup multiply-accumulate: 8 sub-tiles of K (DQ_BK/8 = 8)
        for (uint kk = 0; kk < DQ_BK / 8; kk++) {
            simdgroup_float8x8 a_frag;
            simdgroup_load(a_frag, &tg_A[simd_id * 8 * DQ_BK + kk * 8], DQ_BK);

            simdgroup_float8x8 b0, b1, b2, b3;
            simdgroup_load(b0, &tg_B[kk * 8 * DQ_BN + 0],  DQ_BN);
            simdgroup_load(b1, &tg_B[kk * 8 * DQ_BN + 8],  DQ_BN);
            simdgroup_load(b2, &tg_B[kk * 8 * DQ_BN + 16], DQ_BN);
            simdgroup_load(b3, &tg_B[kk * 8 * DQ_BN + 24], DQ_BN);

            simdgroup_multiply_accumulate(acc0, a_frag, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a_frag, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a_frag, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a_frag, b3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulators to threadgroup memory, then cooperative write to global
    threadgroup float out_tile[DQ_BM * DQ_BN];

    simdgroup_store(acc0, &out_tile[simd_id * 8 * DQ_BN + 0],  DQ_BN);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DQ_BN + 8],  DQ_BN);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DQ_BN + 16], DQ_BN);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DQ_BN + 24], DQ_BN);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DQ_BM * DQ_BN; i += DQ_TG) {
        uint r = i / DQ_BN;
        uint c = i % DQ_BN;
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < M && gc < N) {
            C[gr * N + gc] = out_tile[r * DQ_BN + c];
        }
    }
}

// Same kernel for Q6_K format.
// Processes K in chunks of 64 values (half a 128-value group in Q6_K).

kernel void dequant_matmul_simdgroup_q6_k(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_row = group_id.y * DQ_BM;
    uint tile_col = group_id.x * DQ_BN;

    threadgroup float tg_A[DQ_BM * DQ_BK];
    threadgroup float tg_B[DQ_BK * DQ_BN];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    // Process K in chunks of 64
    for (uint kt = 0; kt < K; kt += DQ_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        // Q6_K: 256 values per block, split into 2 groups of 128.
        // Each group has 4 sub-groups of 32 values.
        // Our 64-value chunk maps to 2 sub-groups within a group.
        uint in_block = kt % Q6_K_BLOCK_VALUES;  // 0, 64, 128, 192
        uint group = in_block / 128;              // 0 or 1
        uint sub_pair = (in_block % 128) / 64;    // 0 or 1

        for (uint i = tid; i < DQ_BM * DQ_BK; i += DQ_TG) {
            uint r = i / DQ_BK;
            uint c = i % DQ_BK;
            uint global_r = tile_row + r;

            if (global_r < M) {
                device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);

                // Offsets into ql/qh/scales for this group and sub-pair
                uint ql_base = group * 64;
                uint qh_base = group * 32;
                uint sc_base = group * 8;

                // c = 0..63 maps to 2 sub-groups of 32
                uint sub = c / 32;          // 0 or 1: which sub-group within pair
                uint l = c % 32;            // 0..31: position within sub-group
                uint is = l / 16;           // 0 or 1: which half of sub-group

                // Scale index: sub_pair*4 + sub*2 + is... but actually the Q6_K
                // layout maps differently. Let me follow the standalone kernel.
                // The standalone iterates: for l in 0..31, computing 4 values per l.
                // Our 64 values per chunk are: sub_pair selects which pair of sub-groups.
                // sub_pair=0: values 0..63 within the group (offsets 0..31 and 32..63)
                // sub_pair=1: values 64..127 within the group (offsets 64..95 and 96..127)

                uint ql_idx = ql_base + sub_pair * 32;
                uint qh_idx = qh_base;

                int q;
                float sc;
                if (sub == 0) {
                    // First 32 of the 64: ql low or high nibble + qh bits
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is]);
                    } else {
                        q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 4]);
                    }
                } else {
                    // Second 32: ql from +32 offset
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 2]);
                    } else {
                        q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 6]);
                    }
                }

                tg_A[r * DQ_BK + c] = d * sc * float(q);
            } else {
                tg_A[r * DQ_BK + c] = 0.0f;
            }
        }

        // Cooperative load B tile
        for (uint i = tid; i < DQ_BK * DQ_BN; i += DQ_TG) {
            uint r = i / DQ_BN;
            uint c = i % DQ_BN;
            uint gr = kt + r;
            uint gc = tile_col + c;
            tg_B[r * DQ_BN + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DQ_BK / 8; kk++) {
            simdgroup_float8x8 a_frag;
            simdgroup_load(a_frag, &tg_A[simd_id * 8 * DQ_BK + kk * 8], DQ_BK);

            simdgroup_float8x8 b0, b1, b2, b3;
            simdgroup_load(b0, &tg_B[kk * 8 * DQ_BN + 0],  DQ_BN);
            simdgroup_load(b1, &tg_B[kk * 8 * DQ_BN + 8],  DQ_BN);
            simdgroup_load(b2, &tg_B[kk * 8 * DQ_BN + 16], DQ_BN);
            simdgroup_load(b3, &tg_B[kk * 8 * DQ_BN + 24], DQ_BN);

            simdgroup_multiply_accumulate(acc0, a_frag, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a_frag, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a_frag, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a_frag, b3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float out_tile[DQ_BM * DQ_BN];

    simdgroup_store(acc0, &out_tile[simd_id * 8 * DQ_BN + 0],  DQ_BN);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DQ_BN + 8],  DQ_BN);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DQ_BN + 16], DQ_BN);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DQ_BN + 24], DQ_BN);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DQ_BM * DQ_BN; i += DQ_TG) {
        uint r = i / DQ_BN;
        uint c = i % DQ_BN;
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < M && gc < N) {
            C[gr * N + gc] = out_tile[r * DQ_BN + c];
        }
    }
}

// ── Batch Dequant+Matmul (B-transposed) ─────────────────────────────────
//
// C[N × M] = B[N × K] × dequant(A[M × K])^T
//
// This is the "batch" variant for prefill: the input B has N tokens as rows
// (natural batch layout), and the output C also has N tokens as rows.
// Weight matrix A is the same [M × K] quantized format as the existing kernels.
//
// Uses simdgroup_matrix_8x8 with transposed A loads for the multiply.
// Processes K in chunks of 64 (one Q4_K sub-block pair per K-tile).
//
// Threadgroup: 128 threads (4 simdgroups × 32 threads).
// Grid: ceil(M/32) × ceil(N/32) threadgroups.

constant uint DB_BM = 32;   // Output tile M-cols (weight rows)
constant uint DB_BN = 64;   // Output tile N-rows (tokens)
constant uint DB_BK = 64;   // K-tile size
constant uint DB_TG = 256;  // Threads per threadgroup (8 simdgroups)

// Tile constants for small-N batch kernels (BN=32, TG=128). Used by both
// dequant_batch_q4_k_bn32 and dequant_batch_q4_k_small / dequant_batch_q6_k_small.
constant uint SB_BM = 32;
constant uint SB_BN = 32;
constant uint SB_BK = 64;
constant uint SB_TG = 128;

// Tile constants for 64x64 full-tile kernels (dense f16 matmul + f16in dequant).
constant uint D64_BM = 64;
constant uint D64_BN = 64;
constant uint D64_BK = 64;
constant uint D64_TG = 256;  // 8 simdgroups

kernel void dequant_batch_q4_k(
    device const Q4_K_Block* A [[buffer(0)]],   // [M × K/256] quantized weights
    device const float* B      [[buffer(1)]],   // [N × K] batch input (rows = tokens)
    device float* C            [[buffer(2)]],   // [N × M] batch output (rows = tokens)
    constant uint& M           [[buffer(3)]],   // output features (weight rows)
    constant uint& N           [[buffer(4)]],   // number of tokens (batch size)
    constant uint& K           [[buffer(5)]],   // input features
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    // group_id.x → M tiles, group_id.y → N tiles
    uint tile_m = group_id.x * DB_BM;  // which M-cols (weight rows)
    uint tile_n = group_id.y * DB_BN;  // which N-rows (tokens)

    // Half-precision A and B tiles: saves 12 KB vs float (4+8 vs 8+16 KB).
    threadgroup half tg_A[DB_BM * DB_BK];   // [BM × BK half] = 4 KB
    threadgroup half tg_B[DB_BN * DB_BK];   // [BN × BK half] = 8 KB
    // Per-row precomputed dequant params: d*scale and dmin*min for each sub-block pair.
    // Preloaded once per K-tile to eliminate per-element device-memory re-reads.
    threadgroup half row_dsc1[DB_BM];   // d * sm1.x
    threadgroup half row_dmin1[DB_BM];  // dmin * sm1.y
    threadgroup half row_dsc2[DB_BM];   // d * sm2.x
    threadgroup half row_dmin2[DB_BM];  // dmin * sm2.y
    threadgroup float out_tile[DB_BN * DB_BM];

    // Float accumulators for half×half→float matmul.
    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DB_BK;

        // Phase 1: preload per-row dequant params (d*scale, dmin*min) for this pair.
        // Threads 0..DB_BM-1 each load one row's params; rest idle during this phase.
        if (tid < DB_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d    = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2,     blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                row_dsc1[tid]  = half(d * sm1.x);
                row_dmin1[tid] = half(dmin * sm1.y);
                row_dsc2[tid]  = half(d * sm2.x);
                row_dmin2[tid] = half(dmin * sm2.y);
            } else {
                row_dsc1[tid]  = half(0.0f);
                row_dmin1[tid] = half(0.0f);
                row_dsc2[tid]  = half(0.0f);
                row_dmin2[tid] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble extraction — one device byte load, two tg_A writes.
        // Each byte qs[pair*32+b] holds lo nibble (sub-block pair*2, col=b) and
        // hi nibble (sub-block pair*2+1, col=b+32). Halves byte loads vs per-element.
        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB_TG) {
            uint r        = i / (DB_BK / 2);
            uint b        = i % (DB_BK / 2);  // byte offset in 32-byte qs range
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar byte = blk.qs[pair * 32 + b];
                tg_A[r * DB_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
                tg_A[r * DB_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
            } else {
                tg_A[r * DB_BK + b]      = half(0.0f);
                tg_A[r * DB_BK + b + 32] = half(0.0f);
            }
        }

        // Phase 3: load B tile, cast float→half.
        // B is [N × K] row-major: B[n][k] = B[n * K + k]
        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? half(B[gn * K + gk]) : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: half×half simdgroup matmul, float accumulator.
        // out[8n × 8m] += B_tile[8n × 8k] × A_tile[8m × 8k]^T
        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fast path: full tile write directly to global output, no threadgroup staging.
    if (tile_n + DB_BN <= N && tile_m + DB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc0, c_base + 0,  M);
        simdgroup_store(acc1, c_base + 8,  M);
        simdgroup_store(acc2, c_base + 16, M);
        simdgroup_store(acc3, c_base + 24, M);
        return;
    }

    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0],  DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8],  DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
}

kernel void dequant_batch_q5_k(
    device const Q5_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];
    threadgroup half row_dsc1[DB_BM];
    threadgroup half row_dmin1[DB_BM];
    threadgroup half row_dsc2[DB_BM];
    threadgroup half row_dmin2[DB_BM];
    threadgroup float out_tile[DB_BN * DB_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q5_K_BLOCK_VALUES;
        uint pair = (kt % Q5_K_BLOCK_VALUES) / DB_BK;

        if (tid < DB_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                row_dsc1[tid] = half(d * sm1.x);
                row_dmin1[tid] = half(dmin * sm1.y);
                row_dsc2[tid] = half(d * sm2.x);
                row_dmin2[tid] = half(dmin * sm2.y);
            } else {
                row_dsc1[tid] = half(0.0f);
                row_dmin1[tid] = half(0.0f);
                row_dsc2[tid] = half(0.0f);
                row_dmin2[tid] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB_TG) {
            uint r = i / (DB_BK / 2);
            uint b = i % (DB_BK / 2);
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar byte = blk.qs[pair * 32 + b];
                uchar high_bits = blk.qh[b];
                float lo_q =
                    float(byte & 0x0F) + (((high_bits >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
                float hi_q = float(byte >> 4)
                    + (((high_bits >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);
                tg_A[r * DB_BK + b] =
                    half(float(row_dsc1[r]) * lo_q - float(row_dmin1[r]));
                tg_A[r * DB_BK + b + 32] =
                    half(float(row_dsc2[r]) * hi_q - float(row_dmin2[r]));
            } else {
                tg_A[r * DB_BK + b] = half(0.0f);
                tg_A[r * DB_BK + b + 32] = half(0.0f);
            }
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? half(B[gn * K + gk]) : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + DB_BN <= N && tile_m + DB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc0, c_base + 0, M);
        simdgroup_store(acc1, c_base + 8, M);
        simdgroup_store(acc2, c_base + 16, M);
        simdgroup_store(acc3, c_base + 24, M);
        return;
    }

    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0], DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8], DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
}

kernel void dequant_batch_q5_k_f16in(
    device const Q5_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];
    threadgroup half row_dsc1[DB_BM];
    threadgroup half row_dmin1[DB_BM];
    threadgroup half row_dsc2[DB_BM];
    threadgroup half row_dmin2[DB_BM];
    threadgroup float out_tile[DB_BN * DB_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q5_K_BLOCK_VALUES;
        uint pair = (kt % Q5_K_BLOCK_VALUES) / DB_BK;

        if (tid < DB_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                row_dsc1[tid] = half(d * sm1.x);
                row_dmin1[tid] = half(dmin * sm1.y);
                row_dsc2[tid] = half(d * sm2.x);
                row_dmin2[tid] = half(dmin * sm2.y);
            } else {
                row_dsc1[tid] = half(0.0f);
                row_dmin1[tid] = half(0.0f);
                row_dsc2[tid] = half(0.0f);
                row_dmin2[tid] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB_TG) {
            uint r = i / (DB_BK / 2);
            uint b = i % (DB_BK / 2);
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar byte = blk.qs[pair * 32 + b];
                uchar high_bits = blk.qh[b];
                float lo_q =
                    float(byte & 0x0F) + (((high_bits >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
                float hi_q = float(byte >> 4)
                    + (((high_bits >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);
                tg_A[r * DB_BK + b] =
                    half(float(row_dsc1[r]) * lo_q - float(row_dmin1[r]));
                tg_A[r * DB_BK + b + 32] =
                    half(float(row_dsc2[r]) * hi_q - float(row_dmin2[r]));
            } else {
                tg_A[r * DB_BK + b] = half(0.0f);
                tg_A[r * DB_BK + b + 32] = half(0.0f);
            }
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + DB_BN <= N && tile_m + DB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
        simdgroup_store(acc0, c_base + 0, C_STRIDE);
        simdgroup_store(acc1, c_base + 8, C_STRIDE);
        simdgroup_store(acc2, c_base + 16, C_STRIDE);
        simdgroup_store(acc3, c_base + 24, C_STRIDE);
        return;
    }

    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0], DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8], DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * C_STRIDE + gm] = out_tile[r * DB_BM + c];
        }
    }
}

kernel void dequant_batch_q5_k_small(
    device const Q5_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * SB_BM;
    uint tile_n = group_id.y * SB_BN;

    threadgroup half tg_A[SB_BM * SB_BK];
    threadgroup float tg_B[SB_BN * SB_BK];
    threadgroup half row_dsc1[SB_BM];
    threadgroup half row_dmin1[SB_BM];
    threadgroup half row_dsc2[SB_BM];
    threadgroup half row_dmin2[SB_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += SB_BK) {
        uint block_idx = kt / Q5_K_BLOCK_VALUES;
        uint pair = (kt % Q5_K_BLOCK_VALUES) / SB_BK;

        if (tid < SB_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                row_dsc1[tid] = half(d * sm1.x);
                row_dmin1[tid] = half(dmin * sm1.y);
                row_dsc2[tid] = half(d * sm2.x);
                row_dmin2[tid] = half(dmin * sm2.y);
            } else {
                row_dsc1[tid] = half(0.0f);
                row_dmin1[tid] = half(0.0f);
                row_dsc2[tid] = half(0.0f);
                row_dmin2[tid] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < SB_BM * (SB_BK / 2); i += SB_TG) {
            uint r = i / (SB_BK / 2);
            uint b = i % (SB_BK / 2);
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar byte = blk.qs[pair * 32 + b];
                uchar high_bits = blk.qh[b];
                float lo_q =
                    float(byte & 0x0F) + (((high_bits >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
                float hi_q = float(byte >> 4)
                    + (((high_bits >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);
                tg_A[r * SB_BK + b] =
                    half(float(row_dsc1[r]) * lo_q - float(row_dmin1[r]));
                tg_A[r * SB_BK + b + 32] =
                    half(float(row_dsc2[r]) * hi_q - float(row_dmin2[r]));
            } else {
                tg_A[r * SB_BK + b] = half(0.0f);
                tg_A[r * SB_BK + b + 32] = half(0.0f);
            }
        }

        for (uint i = tid; i < SB_BN * SB_BK; i += SB_TG) {
            uint r = i / SB_BK;
            uint c = i % SB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * SB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SB_BK / 8; kk++) {
            simdgroup_float8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * SB_BK + kk * 8], SB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * SB_BK + kk * 8], SB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * SB_BK + kk * 8], SB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * SB_BK + kk * 8], SB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * SB_BK + kk * 8], SB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + SB_BN <= N && tile_m + SB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc0, c_base + 0, M);
        simdgroup_store(acc1, c_base + 8, M);
        simdgroup_store(acc2, c_base + 16, M);
        simdgroup_store(acc3, c_base + 24, M);
        return;
    }

    threadgroup float out_tile[SB_BN * SB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * SB_BM + 0], SB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * SB_BM + 8], SB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * SB_BM + 16], SB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * SB_BM + 24], SB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < SB_BN * SB_BM; i += SB_TG) {
        uint r = i / SB_BM;
        uint c = i % SB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = out_tile[r * SB_BM + c];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// dequant_batch_q4_k_v2 — 2-B-fragment inner loop (llama.cpp pattern)
//
// Same tile dimensions as dequant_batch_q4_k (BM=32, BN=64, BK=64) but
// restructured for 4 simdgroups (TG=128) instead of 8 (TG=256).
//
// Each simdgroup handles 16 N-rows (2 blocks of 8) and all 32 M-cols,
// loading 2 B fragments per K-step and reusing 4 A fragments across both:
//
//   OLD (v1): 1 B × 4 A = 4 MACs  → 5 loads/step → 0.80 MACs/load
//   NEW (v2): 2 B × 4 A = 8 MACs  → 6 loads/step → 1.33 MACs/load  (+66%)
//
// This matches llama.cpp's kernel_mul_mm inner loop structure.
//
// Threadgroup memory:
//   half tg_A[32 × 64]          = 4 KB
//   half tg_B[64 × 64]          = 8 KB
//   half row_dsc/dmin × 4       = 256 B
//   float out_tile[64 × 32]     = 8 KB  (boundary path only)
//   Total ≈ 20.25 KB
//
// TG: 128 threads (4 simdgroups × 32 threads)
// Grid: ceil(M/32) × ceil(N/64) threadgroups
// ═══════════════════════════════════════════════════════════════════════════

constant uint V2_BM = 32;
constant uint V2_BN = 64;
constant uint V2_BK = 64;
constant uint V2_TG = 128;   // 4 simdgroups
constant uint V2_NSG = 4;

kernel void dequant_batch_q4_k_v2(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * V2_BM;
    uint tile_n = group_id.y * V2_BN;

    threadgroup half tg_A[V2_BM * V2_BK];       // 4 KB
    threadgroup half tg_B[V2_BN * V2_BK];       // 8 KB
    threadgroup half row_dsc1[V2_BM];
    threadgroup half row_dmin1[V2_BM];
    threadgroup half row_dsc2[V2_BM];
    threadgroup half row_dmin2[V2_BM];
    threadgroup float out_tile[V2_BN * V2_BM];  // 8 KB (boundary path)

    // 8 half-precision accumulators: 2 B-blocks × 4 A-blocks.
    // Using half instead of float halves register pressure (1 KB vs 2 KB per SG),
    // avoiding register spilling on Apple Silicon. The MAC inputs are already half,
    // so accumulating in half is natural. Precision is sufficient for Q4_K inference.
    simdgroup_half8x8 acc0, acc1, acc2, acc3;  // B0 × A0..A3
    simdgroup_half8x8 acc4, acc5, acc6, acc7;  // B1 × A0..A3
    acc0 = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
    acc1 = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
    acc2 = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
    acc3 = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
    acc4 = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
    acc5 = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
    acc6 = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
    acc7 = make_filled_simdgroup_matrix<half, 8>(half(0.0h));

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += V2_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / V2_BK;

        // Phase 1: preload per-row dequant params.
        if (tid < V2_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d    = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2,     blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                row_dsc1[tid]  = half(d * sm1.x);
                row_dmin1[tid] = half(dmin * sm1.y);
                row_dsc2[tid]  = half(d * sm2.x);
                row_dmin2[tid] = half(dmin * sm2.y);
            } else {
                row_dsc1[tid]  = half(0.0f);
                row_dmin1[tid] = half(0.0f);
                row_dsc2[tid]  = half(0.0f);
                row_dmin2[tid] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble extraction → tg_A[BM × BK].
        // 128 threads, BM*(BK/2) = 32*32 = 1024 elements → 8 per thread.
        for (uint i = tid; i < V2_BM * (V2_BK / 2); i += V2_TG) {
            uint r = i / (V2_BK / 2);
            uint b = i % (V2_BK / 2);
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar byte = blk.qs[pair * 32 + b];
                tg_A[r * V2_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
                tg_A[r * V2_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
            } else {
                tg_A[r * V2_BK + b]      = half(0.0f);
                tg_A[r * V2_BK + b + 32] = half(0.0f);
            }
        }

        // Phase 3: load B tile, cast float→half.
        // 128 threads, BN*BK = 64*64 = 4096 elements → 32 per thread.
        for (uint i = tid; i < V2_BN * V2_BK; i += V2_TG) {
            uint r = i / V2_BK;
            uint c = i % V2_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * V2_BK + c] = (gn < N && gk < K) ? half(B[gn * K + gk]) : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: 2-B-fragment inner loop.
        // Each simdgroup handles N-rows [simd_id*16 .. simd_id*16+15].
        // b0 = first 8 rows, b1 = second 8 rows.
        // A fragments a0..a3 cover all 32 M-rows (shared across b0 and b1).
        for (uint kk = 0; kk < V2_BK / 8; kk++) {
            simdgroup_half8x8 b0, b1;
            simdgroup_load(b0, &tg_B[(simd_id * 16)     * V2_BK + kk * 8], V2_BK);
            simdgroup_load(b1, &tg_B[(simd_id * 16 + 8) * V2_BK + kk * 8], V2_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * V2_BK + kk * 8], V2_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * V2_BK + kk * 8], V2_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * V2_BK + kk * 8], V2_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * V2_BK + kk * 8], V2_BK, ulong2(0,0), true);

            // 8 MACs: b0 × a0..a3, b1 × a0..a3
            simdgroup_multiply_accumulate(acc0, b0, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b0, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b0, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b0, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b1, a0, acc4);
            simdgroup_multiply_accumulate(acc5, b1, a1, acc5);
            simdgroup_multiply_accumulate(acc6, b1, a2, acc6);
            simdgroup_multiply_accumulate(acc7, b1, a3, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output: stage half accumulators to threadgroup, convert to float on write.
    // Reuse tg_B as half staging buffer (8 KB, fits 16×32 per SG = 2 KB, 4 SGs = 8 KB).
    threadgroup half* stg = (threadgroup half*)tg_B;  // reuse B tile (no longer needed)

    simdgroup_store(acc0, &stg[(simd_id * 16)     * V2_BM + 0],  V2_BM);
    simdgroup_store(acc1, &stg[(simd_id * 16)     * V2_BM + 8],  V2_BM);
    simdgroup_store(acc2, &stg[(simd_id * 16)     * V2_BM + 16], V2_BM);
    simdgroup_store(acc3, &stg[(simd_id * 16)     * V2_BM + 24], V2_BM);
    simdgroup_store(acc4, &stg[(simd_id * 16 + 8) * V2_BM + 0],  V2_BM);
    simdgroup_store(acc5, &stg[(simd_id * 16 + 8) * V2_BM + 8],  V2_BM);
    simdgroup_store(acc6, &stg[(simd_id * 16 + 8) * V2_BM + 16], V2_BM);
    simdgroup_store(acc7, &stg[(simd_id * 16 + 8) * V2_BM + 24], V2_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Convert half→float and write to global output (with bounds check).
    for (uint i = tid; i < V2_BN * V2_BM; i += V2_TG) {
        uint r = i / V2_BM;
        uint c = i % V2_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = float(stg[r * V2_BM + c]);
        }
    }
}

// BN=32 variant of dequant_batch_q4_k for small-N prefill.
//
// Uses SB_BN=32 and SB_TG=128 (4 simdgroups × 8 N-rows = 32-row N-tiles).
// Keeps DB_BK=64 and DB_BM=32 so A-tile loading and inner loop are identical
// to the main kernel.
//
// Threadgroup memory:
//   half tg_A[32 × 64]          = 4 KB
//   half tg_B[32 × 64]          = 4 KB
//   half row_dsc/dmin × 4       = 256 B
//   float out_tile[32 × 32]     = 4 KB
//   Total ≈ 12.25 KB  →  2 TGs/SM  (vs 20 KB → 1 TGs/SM for the BN=64 kernel)
//
// At N=39 (typical short-prompt prefill):
//   - BN=64: 128 TGs, ALL boundary (0+64 > 39)  →  100% slow scatter path
//   - BN=32: 256 TGs, 50% fast path (0+32≤39), 50% boundary (32+32>39, 7 rows)
//            plus 2 TGs/SM → all 256 TGs fit in ~1 wave instead of 1.28 waves
constant uint DB32_BN = 32;
constant uint DB32_TG = 128;

kernel void dequant_batch_q4_k_bn32(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB32_BN;  // 32-row N-tile

    threadgroup half tg_A[DB_BM * DB_BK];   // [32 × 64] = 4 KB
    threadgroup half tg_B[DB32_BN * DB_BK];   // [32 × 64] = 4 KB
    threadgroup half row_dsc1[DB_BM];
    threadgroup half row_dmin1[DB_BM];
    threadgroup half row_dsc2[DB_BM];
    threadgroup half row_dmin2[DB_BM];
    threadgroup float out_tile[DB32_BN * DB_BM];  // [32 × 32] = 4 KB

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DB_BK;

        // Phase 1: preload per-row scale params (same as dequant_batch_q4_k).
        if (tid < DB_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d    = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2,     blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                row_dsc1[tid]  = half(d * sm1.x);
                row_dmin1[tid] = half(dmin * sm1.y);
                row_dsc2[tid]  = half(d * sm2.x);
                row_dmin2[tid] = half(dmin * sm2.y);
            } else {
                row_dsc1[tid]  = half(0.0f);
                row_dmin1[tid] = half(0.0f);
                row_dsc2[tid]  = half(0.0f);
                row_dmin2[tid] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble A-tile load (same as dequant_batch_q4_k).
        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB32_TG) {
            uint r        = i / (DB_BK / 2);
            uint b        = i % (DB_BK / 2);
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar byte = blk.qs[pair * 32 + b];
                tg_A[r * DB_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
                tg_A[r * DB_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
            } else {
                tg_A[r * DB_BK + b]      = half(0.0f);
                tg_A[r * DB_BK + b + 32] = half(0.0f);
            }
        }

        // Phase 3: B-tile load (SB_BN=32 rows instead of DB_BN=64).
        for (uint i = tid; i < DB32_BN * DB_BK; i += DB32_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? half(B[gn * K + gk]) : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: half×half simdgroup matmul, float accumulator.
        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fast path: direct store when the full SB_BN×DB_BM output tile is in-bounds.
    if (tile_n + DB32_BN <= N && tile_m + DB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc0, c_base + 0,  M);
        simdgroup_store(acc1, c_base + 8,  M);
        simdgroup_store(acc2, c_base + 16, M);
        simdgroup_store(acc3, c_base + 24, M);
        return;
    }

    // Boundary path: stage through out_tile[SB_BN × DB_BM].
    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0],  DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8],  DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB32_BN * DB_BM; i += DB32_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// dequant_batch_q4_k_blocked — Blocked threadgroup layout (llama.cpp pattern)
//
// Ports llama.cpp's kernel_mul_mm_q4_K_f32 architecture:
//   - Tile: BM=64, BN=32, BK=32
//   - TG: 128 threads (4 simdgroups)
//   - TG memory: 6-8 KB (sa=4KB blocked + sb=2KB blocked)
//   - Layout: BLOCKED — each 8×8 fragment at stride 8 (1 cache line)
//   - Dequant: all 128 threads inline, each producing 16 values
//   - Inner loop: 4A + 2B loads + 8 MACs per K/8 step (1.33 MACs/load)
//
// Why blocked layout is faster:
//   simdgroup_load with stride 8 reads an 8×8 fragment from 1 cache line (128B).
//   Row-major stride 64 reads from 8 cache lines with bank conflicts.
//
// Memory map (dynamic threadgroup via [[threadgroup(0)]]):
//   [0..4095]     sa: 64×32 half in blocked layout (32 blocks of 64 elements)
//   [4096..6143]  sb: 32×32 half in blocked layout (16 blocks of 64 elements)
//   Total: 6144 bytes (6 KB). Output staging reuses sa as float (8192 bytes).
//
// Grid: (ceil(N/32), ceil(M/64)) threadgroups.
// Requires: K % 256 == 0 (Q4_K block size).
// ═══════════════════════════════════════════════════════════════════════════

// Scale extraction matching llama.cpp's get_scale_min_k4_just2.
// Returns uchar2(scale, min) for sub-block (j, k) from 6-bit packed scales.
inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar* q) {
    return j < 4
        ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
        : uchar2{uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)),
                  uchar((q[j + 4 + k] >> 4) | ((q[j + 0 + k] & 0xc0) >> 2))};
}

// Per-thread dequantization matching llama.cpp's dequantize_q4_K.
// Produces 16 half values in a 4×4 thread-local matrix from one Q4_K sub-block.
inline void dequantize_q4k_blocked(
    device const Q4_K_Block* xb,
    short il,
    thread half4x4& reg
) {
    device const uchar* q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float mn  = float(xb->dmin);
    const float dl  = d * float(sc[0]);
    const float ml  = mn * float(sc[1]);

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = half(dl * float(q[i] & mask) - ml);
    }
}

constant bool BLOCKED_BC_OUT [[function_constant(1)]];

kernel void dequant_batch_q4_k_blocked(
    device const Q4_K_Block* A [[buffer(0)]],  // [M × K/256] quantized weights
    device const float* B      [[buffer(1)]],  // [N × K] activations (float32)
    device float* C            [[buffer(2)]],  // [N × M] output (float32)
    constant uint& M           [[buffer(3)]],  // weight rows (output features)
    constant uint& N           [[buffer(4)]],  // tokens (batch size)
    constant uint& K           [[buffer(5)]],  // input features
    threadgroup char* shmem    [[threadgroup(0)]],
    uint2 tgpig                [[threadgroup_position_in_grid]],
    ushort tiitg               [[thread_index_in_threadgroup]],
    ushort sgitg               [[simdgroup_index_in_threadgroup]]
) {
    // Blocked threadgroup buffers.
    // sa: 64×32 half = 4096 bytes. sb: 32×32 half = 2048 bytes.
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;   // M-tile (weight rows)
    constexpr short NR1 = 32;   // N-tile (tokens)
    constexpr short NK  = 32;   // K-tile
    constexpr short NL0 = NK / 16;  // = 2: dequant sub-positions per thread
    constexpr short NL1 = NK / 8;   // = 4: activation sub-positions per thread
    constexpr short QK_NL = 16;     // Q4_K: 256 values / 16 per dequant call

    // Tile position in output matrix.
    const int r0 = tgpig.y * NR0;   // M offset (weight rows)
    const int r1 = tgpig.x * NR1;   // N offset (tokens)

    // Boundary clamping (same as llama.cpp).
    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = short(min(uint(NR1), N - uint(r1)));
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    short il = il0;

    // Weight pointer: each thread reads its own row's Q4_K block.
    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    const short offset1 = il0 / QK_NL;  // = 0 for Q4_K (il0 < 16)
    device const Q4_K_Block* x = A + uint(r0 + lr0) * blocks_per_row + offset1;

    // Activation pointer: each thread reads 8 consecutive K-elements.
    const short iy = short(8 * (tiitg % NL1));
    device const float* y = B + uint(r1 + lr1) * K + iy;

    // 8 accumulators: 4 simdgroups × 2 rows × 4 cols = 8 output tiles of 8×8.
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    // ── K-loop: process NK=32 elements per iteration ──────────────────
    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        // Phase 1: Dequantize A — all 128 threads, each produces 16 values.
        half4x4 temp_a;
        dequantize_q4k_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter to blocked layout: sa[64*ib + 8*ly + lx].
        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        // Phase 2: Load B — all 128 threads, each loads 8 values.
        // For K-quants, K is always 256-aligned, so each thread's 8-value
        // scatter is contiguous and can use the same vectorized float2x4 ->
        // half2x4 path as llama.cpp's aligned kernel_mul_mm fast path.
        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
            (half2x4)(*(device float2x4*)y);

        // Advance pointers for next K-tile (llama.cpp pattern).
        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: Compute — blocked simdgroup matmul.
        // Each simdgroup loads from its quadrant of sa/sb.
        // lsma selects one of 2 halves of 64 M-rows (32 rows each).
        // lsmb selects one of 2 halves of 32 N-rows (16 rows each).
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // ── Output write ──────────────────────────────────────────────────
    if (!BLOCKED_BC_OUT) {
        // Fast path: full tile, direct write to device memory.
        device float* out = C
            + uint(r0 + 32 * (sgitg & 1))
            + uint(r1 + 16 * (sgitg >> 1)) * M;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], out + 8 * (i % 4) + 8 * M * (i / 4), M, ulong2(0, 0), false);
        }
    } else {
        // Slow path: stage through threadgroup memory, bounds-checked write.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, ulong2(0, 0), false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float*  D  = C + uint(r0) + uint(r1 + j) * M;
                device float4* D4 = (device float4*)D;
                threadgroup float*  Cs  = temp_str + j * NR0;
                threadgroup float4* C4 = (threadgroup float4*)Cs;

                int i = 0;
                for (; i < nr0 / 4; i++) {
                    *(D4 + i) = *(C4 + i);
                }
                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(Cs + i);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// dequant_batch_q4_k_blocked_bm32 — BM=32 variant for small M dimensions
//
// Same blocked layout (stride 8) as dequant_batch_q4_k_blocked but with
// BM=32 instead of 64. This doubles the threadgroup count for small M:
//   M=1024: BM=64 → 16 TG rows, BM=32 → 32 TG rows (2× more GPU saturation)
//
// Tile: BM=32, BN=32, BK=32. TG=128 (4 simdgroups).
// sa: 32×32 half = 2 KB (16 blocks of 64 elements, ib = 4*sx + sy)
// sb: 32×32 half = 2 KB (same as BM=64)
// Total TG memory: 4 KB → 8 TGs/SM (vs 6 KB → 5 TGs/SM for BM=64)
// Inner loop: 2A + 2B loads + 4 MACs per K-step (1.0 MACs/load)
//
// Grid: (ceil(N/32), ceil(M/32)) threadgroups.
// ═══════════════════════════════════════════════════════════════════════════

kernel void dequant_batch_q4_k_blocked_bm32(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    threadgroup char* shmem    [[threadgroup(0)]],
    uint2 tgpig                [[threadgroup_position_in_grid]],
    ushort tiitg               [[thread_index_in_threadgroup]],
    ushort sgitg               [[simdgroup_index_in_threadgroup]]
) {
    // sa: 32×32 half = 2 KB. sb: 32×32 half = 2 KB.
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 2048);

    constexpr short NR0 = 32;   // M-tile (half of BM=64)
    constexpr short NR1 = 32;   // N-tile
    constexpr short NK  = 32;   // K-tile
    constexpr short NL0 = NK / 16;  // = 2
    constexpr short NL1 = NK / 8;   // = 4
    constexpr short QK_NL = 16;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = short(min(uint(NR1), N - uint(r1)));
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    short il = il0;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    const short offset1 = il0 / QK_NL;
    device const Q4_K_Block* x = A + uint(r0 + lr0) * blocks_per_row + offset1;

    const short iy = short(8 * (tiitg % NL1));
    device const float* y = B + uint(r1 + lr1) * K + iy;

    // 4 accumulators: 2 M-halves × 2 N-halves per simdgroup.
    simdgroup_half8x8 ma[2];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[4];
    for (short i = 0; i < 4; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q4k_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter to blocked layout: sa[64*ib + 8*ly + lx].
        // ib = 4*sx + sy (not 8*sx + sy) because BM=32 → 16 blocks.
        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 4 * sx + sy;  // 4 for BM=32 (was 8 for BM=64)
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
            (half2x4)(*(device float2x4*)y);

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: each SG handles 16×16 output sub-tile (2 A-blocks × 2 B-blocks).
        // lsma: 2 halves of 32 M-rows (16 each)
        // lsmb: 2 halves of 32 N-rows (16 each)
        threadgroup const half* lsma = sa + 2 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // 4 MACs: mb[i/2] × ma[i%2]
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 2], ma[i % 2], mc[i]);
            }

            lsma += 4 * 64;  // advance past 4 blocks (2 SG halves × 2 A-blocks)
            lsmb += 4 * 64;
        }
    }

    // Output: each SG writes 16×16 sub-tile.
    if (!BLOCKED_BC_OUT) {
        device float* out = C
            + uint(r0 + 16 * (sgitg & 1))
            + uint(r1 + 16 * (sgitg >> 1)) * M;

        for (short i = 0; i < 4; i++) {
            simdgroup_store(mc[i], out + 8 * (i % 2) + 8 * M * (i / 2), M, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 16 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

        for (short i = 0; i < 4; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 2) + 8 * NR0 * (i / 2),
                            NR0, ulong2(0, 0), false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float*  D  = C + uint(r0) + uint(r1 + j) * M;
                device float4* D4 = (device float4*)D;
                threadgroup float*  Cs  = temp_str + j * NR0;
                threadgroup float4* C4 = (threadgroup float4*)Cs;
                int i = 0;
                for (; i < nr0 / 4; i++) { *(D4 + i) = *(C4 + i); }
                i *= 4;
                for (; i < nr0; i++) { *(D + i) = *(Cs + i); }
            }
        }
    }
}

// Per-thread dequantization matching llama.cpp's dequantize_q5_K.
// Produces 16 half values in a 4×4 thread-local matrix from one Q5_K sub-block.
inline void dequantize_q5k_blocked(
    device const Q5_K_Block* xb,
    short il,
    thread half4x4& reg
) {
    device const uchar* q = xb->qs;
    device const uchar* qh = xb->qh;

    short is = (il / 4) * 2;
    q = q + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    const uchar ul = uchar(1u << (il / 2));
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float mn = float(xb->dmin);
    const float dl = d * float(sc[0]);
    const float ml = mn * float(sc[1]);

    const ushort mask = il < 2 ? 0x0Fu : 0xF0u;
    const float qh_val = il < 2 ? 16.0f : 256.0f;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = half(
            dl * (float(q[i] & mask) + ((qh[i] & ul) ? qh_val : 0.0f)) - ml
        );
    }
}

// Q5_K blocked batch kernel (llama.cpp kernel_mul_mm geometry).
kernel void dequant_batch_q5_k_blocked(
    device const Q5_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    threadgroup char* shmem    [[threadgroup(0)]],
    uint2 tgpig                [[threadgroup_position_in_grid]],
    ushort tiitg               [[thread_index_in_threadgroup]],
    ushort sgitg               [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short QK_NL = 16;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = short(min(uint(NR1), N - uint(r1)));
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    short il = il0;

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    const short offset1 = il0 / QK_NL;
    device const Q5_K_Block* x = A + uint(r0 + lr0) * blocks_per_row + offset1;

    const short iy = short(8 * (tiitg % NL1));
    device const float* y = B + uint(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q5k_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
            (half2x4)(*(device float2x4*)y);

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (!BLOCKED_BC_OUT) {
        device float* out = C
            + uint(r0 + 32 * (sgitg & 1))
            + uint(r1 + 16 * (sgitg >> 1)) * M;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], out + 8 * (i % 4) + 8 * M * (i / 4), M, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, ulong2(0, 0), false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float* D = C + uint(r0) + uint(r1 + j) * M;
                device float4* D4 = (device float4*)D;
                threadgroup float* Cs = temp_str + j * NR0;
                threadgroup float4* C4 = (threadgroup float4*)Cs;

                int i = 0;
                for (; i < nr0 / 4; i++) {
                    *(D4 + i) = *(C4 + i);
                }
                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(Cs + i);
                }
            }
        }
    }
}

// Q5_K blocked batch kernel with f16 input and f32 output.
kernel void dequant_batch_q5_k_blocked_f16in(
    device const Q5_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    threadgroup char* shmem    [[threadgroup(0)]],
    uint2 tgpig                [[threadgroup_position_in_grid]],
    ushort tiitg               [[thread_index_in_threadgroup]],
    ushort sgitg               [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short QK_NL = 16;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = short(min(uint(NR1), N - uint(r1)));
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    short il = il0;

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    const short offset1 = il0 / QK_NL;
    device const Q5_K_Block* x = A + uint(r0 + lr0) * blocks_per_row + offset1;

    const short iy = short(8 * (tiitg % NL1));
    device const half* y = B + uint(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q5k_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
            *(device half2x4*)y;

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= int(M) && r1 + NR1 <= int(N)) {
        device float* out = C
            + uint(r0 + 32 * (sgitg & 1))
            + uint(r1 + 16 * (sgitg >> 1)) * C_STRIDE;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(
                mc[i],
                out + 8 * (i % 4) + 8 * C_STRIDE * (i / 4),
                C_STRIDE,
                ulong2(0, 0),
                false
            );
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, ulong2(0, 0), false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float* D = C + uint(r0) + uint(r1 + j) * C_STRIDE;
                device float4* D4 = (device float4*)D;
                threadgroup float* Cs = temp_str + j * NR0;
                threadgroup float4* C4 = (threadgroup float4*)Cs;

                int i = 0;
                for (; i < nr0 / 4; i++) {
                    *(D4 + i) = *(C4 + i);
                }
                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(Cs + i);
                }
            }
        }
    }
}

// Forward declaration (defined later, after Q6_K blocked BM=64).
inline void dequantize_q6k_blocked(device const Q6_K_Block* xb, short il, thread half4x4& reg);

// Same for Q6_K:
kernel void dequant_batch_q6_k_blocked_bm32(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    threadgroup char* shmem    [[threadgroup(0)]],
    uint2 tgpig                [[threadgroup_position_in_grid]],
    ushort tiitg               [[thread_index_in_threadgroup]],
    ushort sgitg               [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 2048);

    constexpr short NR0 = 32;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short QK_NL = 16;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = short(min(uint(NR1), N - uint(r1)));
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    short il = il0;

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    const short offset1 = il0 / QK_NL;
    device const Q6_K_Block* x = A + uint(r0 + lr0) * blocks_per_row + offset1;

    const short iy = short(8 * (tiitg % NL1));
    device const float* y = B + uint(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[2];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[4];
    for (short i = 0; i < 4; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q6k_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 4 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
            (half2x4)(*(device float2x4*)y);

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 2 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 2], ma[i % 2], mc[i]);
            }
            lsma += 4 * 64;
            lsmb += 4 * 64;
        }
    }

    if (!BLOCKED_BC_OUT) {
        device float* out = C
            + uint(r0 + 16 * (sgitg & 1))
            + uint(r1 + 16 * (sgitg >> 1)) * M;
        for (short i = 0; i < 4; i++) {
            simdgroup_store(mc[i], out + 8 * (i % 2) + 8 * M * (i / 2), M, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 16 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;
        for (short i = 0; i < 4; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 2) + 8 * NR0 * (i / 2),
                            NR0, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float*  D  = C + uint(r0) + uint(r1 + j) * M;
                device float4* D4 = (device float4*)D;
                threadgroup float*  Cs  = temp_str + j * NR0;
                threadgroup float4* C4 = (threadgroup float4*)Cs;
                int i = 0;
                for (; i < nr0 / 4; i++) { *(D4 + i) = *(C4 + i); }
                i *= 4;
                for (; i < nr0; i++) { *(D + i) = *(Cs + i); }
            }
        }
    }
}

// ── BN=32 full-tile (no bounds check) ──────────────────────────────────
//
// Same as dequant_batch_q4_k_bn32 but without out_tile allocation or
// boundary checks. Requires N % 32 == 0 and M % 32 == 0.
//
// Threadgroup memory:
//   half tg_A[32 × 64]     = 4 KB
//   half tg_B[32 × 64]     = 4 KB
//   half row_dsc/dmin × 4  = 256 B
//   Total ≈ 8.25 KB  →  3 TGs/SM (vs 12 KB with out_tile → 2 TGs/SM)
//
// TG: 128 threads (4 simdgroups × 32 threads)
// Grid: (M/32) × (N/32) threadgroups
kernel void dequant_batch_q4_k_bn32_full(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB32_BN;

    threadgroup half tg_A[DB_BM * DB_BK];       // 4 KB
    threadgroup half tg_B[DB32_BN * DB_BK];     // 4 KB
    threadgroup half row_dsc1[DB_BM];
    threadgroup half row_dmin1[DB_BM];
    threadgroup half row_dsc2[DB_BM];
    threadgroup half row_dmin2[DB_BM];
    // No out_tile — full tile only, saves 4 KB TG memory.

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DB_BK;

        if (tid < DB_BM) {
            uint global_r = tile_m + tid;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            float2 sm1 = get_scale_min_q4k(pair * 2,     blk.scales);
            float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
            row_dsc1[tid]  = half(d * sm1.x);
            row_dmin1[tid] = half(dmin * sm1.y);
            row_dsc2[tid]  = half(d * sm2.x);
            row_dmin2[tid] = half(dmin * sm2.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB32_TG) {
            uint r = i / (DB_BK / 2);
            uint b = i % (DB_BK / 2);
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            uchar byte = blk.qs[pair * 32 + b];
            tg_A[r * DB_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
            tg_A[r * DB_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
        }

        for (uint i = tid; i < DB32_BN * DB_BK; i += DB32_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            tg_B[r * DB_BK + c] = half(B[(tile_n + r) * K + kt + c]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Direct store — no bounds check needed (caller guarantees alignment).
    device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
    simdgroup_store(acc0, c_base + 0,  M);
    simdgroup_store(acc1, c_base + 8,  M);
    simdgroup_store(acc2, c_base + 16, M);
    simdgroup_store(acc3, c_base + 24, M);
}

// Same batch kernel for Q6_K format.
kernel void dequant_batch_q6_k(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup float tg_A[DB_BM * DB_BK];
    threadgroup float tg_B[DB_BN * DB_BK];
    threadgroup float out_tile[DB_BN * DB_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        // Cooperative dequant A tile
        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint global_r = tile_m + r;

            if (global_r < M) {
                device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);

                uint ql_base = group * 64;
                uint qh_base = group * 32;
                uint sub = c / 32;
                uint l = c % 32;
                uint is = l / 16;

                uint ql_idx = ql_base + sub_pair * 32;
                uint qh_idx = qh_base;

                int q;
                float sc;
                if (sub == 0) {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                        sc = float(blk.scales[group * 8 + is]);
                    } else {
                        q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        sc = float(blk.scales[group * 8 + is + 4]);
                    }
                } else {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        sc = float(blk.scales[group * 8 + is + 2]);
                    } else {
                        q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        sc = float(blk.scales[group * 8 + is + 6]);
                    }
                }

                tg_A[r * DB_BK + c] = d * sc * float(q);
            } else {
                tg_A[r * DB_BK + c] = 0.0f;
            }
        }

        // Cooperative load B tile
        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_float8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_float8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + DB_BN <= N && tile_m + DB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc0, c_base + 0,  M);
        simdgroup_store(acc1, c_base + 8,  M);
        simdgroup_store(acc2, c_base + 16, M);
        simdgroup_store(acc3, c_base + 24, M);
        return;
    }

    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0],  DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8],  DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
}

// Dense f16 batch matmul (B-transposed layout):
//   C[N×M] = B[N×K] × A[M×K]^T
//
// A and B are half, accumulators/output are float.
// Uses the same tile shape as dequant_batch_* for easy routing parity.
kernel void batch_matmul_btrans_f16_f32(
    device const half* A      [[buffer(0)]],  // [M × K]
    device const half* B      [[buffer(1)]],  // [N × K]
    device float* C           [[buffer(2)]],  // [N × M]
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    uint2 group_id            [[threadgroup_position_in_grid]],
    uint  tid                 [[thread_index_in_threadgroup]],
    uint  simd_id             [[simdgroup_index_in_threadgroup]],
    uint  simd_lane           [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    for (uint kt = 0; kt < K; kt += DB_BK) {
        // Cooperative load A tile: [BM × BK] from row-major [M × K]
        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gm = tile_m + r;
            uint gk = kt + c;
            tg_A[r * DB_BK + c] = (gm < M && gk < K) ? A[gm * K + gk] : half(0.0f);
        }

        // Cooperative load B tile: [BN × BK] from row-major [N × K]
        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + DB_BN <= N && tile_m + DB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc0, c_base + 0,  M);
        simdgroup_store(acc1, c_base + 8,  M);
        simdgroup_store(acc2, c_base + 16, M);
        simdgroup_store(acc3, c_base + 24, M);
        return;
    }

    threadgroup float out_tile[DB_BN * DB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0],  DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8],  DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
}

// 64x64 full-tile fast path for dense f16 batch matmul (B-transposed).
// Uses D64_BM/D64_BN/D64_BK/D64_TG constants (defined below near f16in_full64).
// Preconditions enforced by dispatch:
// - M is divisible by 64
// - N is divisible by 64
kernel void batch_matmul_btrans_f16_f32_full64(
    device const half* A      [[buffer(0)]],  // [M × K]
    device const half* B      [[buffer(1)]],  // [N × K]
    device float* C           [[buffer(2)]],  // [N × C_STRIDE]
    constant uint& M          [[buffer(3)]],  // output cols for this dispatch
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& C_STRIDE   [[buffer(6)]],  // destination row stride
    uint2 group_id            [[threadgroup_position_in_grid]],
    uint tid                  [[thread_index_in_threadgroup]],
    uint simd_id              [[simdgroup_index_in_threadgroup]],
    uint simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D64_BM;
    uint tile_n = group_id.y * D64_BN;

    threadgroup half tg_A[D64_BM * D64_BK];
    threadgroup half tg_B[D64_BN * D64_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    for (uint kt = 0; kt < K; kt += D64_BK) {
        for (uint i = tid; i < D64_BM * D64_BK; i += D64_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gm = tile_m + r;
            uint gk = kt + c;
            tg_A[r * D64_BK + c] = A[gm * K + gk];
        }

        for (uint i = tid; i < D64_BN * D64_BK; i += D64_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * D64_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0,  C_STRIDE);
    simdgroup_store(acc1, c_base + 8,  C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
    simdgroup_store(acc4, c_base + 32, C_STRIDE);
    simdgroup_store(acc5, c_base + 40, C_STRIDE);
    simdgroup_store(acc6, c_base + 48, C_STRIDE);
    simdgroup_store(acc7, c_base + 56, C_STRIDE);
}

// Full-tile fast paths for f32-input kernels.
// Preconditions enforced by dispatch:
// - M is divisible by DB_BM
// - N is divisible by DB_BN

kernel void dequant_batch_q4_k_full(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    // Half-precision A and B tiles: 4+8 KB vs 8+16 KB.
    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];
    threadgroup half row_dsc1[DB_BM];
    threadgroup half row_dmin1[DB_BM];
    threadgroup half row_dsc2[DB_BM];
    threadgroup half row_dmin2[DB_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DB_BK;

        // Phase 1: preload per-row dequant params (no boundary check — full tile).
        if (tid < DB_BM) {
            device const Q4_K_Block& blk = A[(tile_m + tid) * blocks_per_row + block_idx];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            float2 sm1 = get_scale_min_q4k(pair * 2,     blk.scales);
            float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
            row_dsc1[tid]  = half(d * sm1.x);
            row_dmin1[tid] = half(dmin * sm1.y);
            row_dsc2[tid]  = half(d * sm2.x);
            row_dmin2[tid] = half(dmin * sm2.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble extraction — one byte load, two tg_A writes.
        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB_TG) {
            uint r = i / (DB_BK / 2);
            uint b = i % (DB_BK / 2);
            device const Q4_K_Block& blk = A[(tile_m + r) * blocks_per_row + block_idx];
            uchar byte = blk.qs[pair * 32 + b];
            tg_A[r * DB_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
            tg_A[r * DB_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
        }

        // Phase 3: load B tile, cast float→half (no boundary check — full tile).
        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            tg_B[r * DB_BK + c] = half(B[(tile_n + r) * K + (kt + c)]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
    simdgroup_store(acc0, c_base + 0,  M);
    simdgroup_store(acc1, c_base + 8,  M);
    simdgroup_store(acc2, c_base + 16, M);
    simdgroup_store(acc3, c_base + 24, M);
}

// ── Inline-dequant full-tile Q4_K batch matmul ─────────────────────────
//
// Fuses Phase 1 (32-thread scale preload) into Phase 2 (256-thread dequant).
// Each thread reads the Q4_K block header (d, dmin, scales) inline during
// nibble extraction. Eliminates one barrier and the 88%-idle Phase 1.
//
// Block headers are tiny (16 bytes) and L1-cached: 32 threads in a simdgroup
// all read the same row's header, so only one L1 miss per row per simdgroup.
//
// Threadgroup memory: same as dequant_batch_q4_k_full minus row_dsc/dmin.
//   tg_A[32×64] half = 4 KB
//   tg_B[64×64] half = 8 KB
//   Total = 12 KB (no scale staging needed)
//
// TG: 256 threads (8 simdgroups). Grid: (M/32) × (N/64).
// Requires M % 32 == 0, N % 64 == 0.
kernel void dequant_batch_q4_k_inline(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];   // 4 KB
    threadgroup half tg_B[DB_BN * DB_BK];   // 8 KB
    // No row_dsc/dmin arrays — scales extracted inline per thread.

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DB_BK;

        // Fused Phase 1+2: inline dequant, all 256 threads active.
        // Each thread reads block header + nibble byte, produces 2 half values.
        // Block headers (16 B) are L1-cached across threads in the same simdgroup.
        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB_TG) {
            uint r = i / (DB_BK / 2);   // M-row within tile (0..31)
            uint b = i % (DB_BK / 2);   // byte index (0..31)
            device const Q4_K_Block& blk = A[(tile_m + r) * blocks_per_row + block_idx];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            float2 sm1 = get_scale_min_q4k(pair * 2,     blk.scales);
            float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
            float dsc1 = d * sm1.x;
            float mn1  = dmin * sm1.y;
            float dsc2 = d * sm2.x;
            float mn2  = dmin * sm2.y;
            uchar byte = blk.qs[pair * 32 + b];
            tg_A[r * DB_BK + b]      = half(dsc1 * float(byte & 0x0F) - mn1);
            tg_A[r * DB_BK + b + 32] = half(dsc2 * float(byte >> 4)   - mn2);
        }

        // Phase 3: B-tile load (unchanged).
        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            tg_B[r * DB_BK + c] = half(B[(tile_n + r) * K + (kt + c)]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: simdgroup matmul (unchanged).
        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
    simdgroup_store(acc0, c_base + 0,  M);
    simdgroup_store(acc1, c_base + 8,  M);
    simdgroup_store(acc2, c_base + 16, M);
    simdgroup_store(acc3, c_base + 24, M);
}

kernel void dequant_batch_q6_k_full(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup float tg_A[DB_BM * DB_BK];
    threadgroup float tg_B[DB_BN * DB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint global_r = tile_m + r;
            device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d = float(blk.d);

            uint ql_base = group * 64;
            uint qh_base = group * 32;
            uint sub = c / 32;
            uint l = c % 32;
            uint is = l / 16;
            uint ql_idx = ql_base + sub_pair * 32;
            uint qh_idx = qh_base;

            int q;
            float sc;
            if (sub == 0) {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                    sc = float(blk.scales[group * 8 + is]);
                } else {
                    q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                    sc = float(blk.scales[group * 8 + is + 4]);
                }
            } else {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                    sc = float(blk.scales[group * 8 + is + 2]);
                } else {
                    q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                    sc = float(blk.scales[group * 8 + is + 6]);
                }
            }

            tg_A[r * DB_BK + c] = d * sc * float(q);
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_float8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_float8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
    simdgroup_store(acc0, c_base + 0,  M);
    simdgroup_store(acc1, c_base + 8,  M);
    simdgroup_store(acc2, c_base + 16, M);
    simdgroup_store(acc3, c_base + 24, M);
}

// ═══════════════════════════════════════════════════════════════════════════
// dequant_batch_q6_k_blocked — Blocked threadgroup layout for Q6_K
//
// Same architecture as dequant_batch_q4_k_blocked (llama.cpp pattern):
//   BM=64, BN=32, BK=32, TG=128, blocked layout (stride 8), 6 KB TG memory.
// Only the dequant function differs (Q6_K has 6-bit values: ql[128]+qh[64]).
//
// Grid: (ceil(N/32), ceil(M/64)) threadgroups.
// ═══════════════════════════════════════════════════════════════════════════

// Port of llama.cpp's dequantize_q6_K for blocked layout.
// Each call with il ∈ [0..15] produces 16 half values in a 4×4 matrix.
inline void dequantize_q6k_blocked(
    device const Q6_K_Block* xb,
    short il,
    thread half4x4& reg
) {
    const half d_all = xb->d;
    device const uint16_t* ql = (device const uint16_t*)xb->ql;
    device const uint16_t* qh = (device const uint16_t*)xb->qh;
    device const char*  scales = (device const char*)xb->scales;

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    float sc = float(scales[(il % 2) + 2 * (il / 2)]);
    il = (il / 2) & 3;

    const uint kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0u : 0x30303030u)
                                : (il > 0 ? 0x0C0C0C0Cu : 0x03030303u);
    const uint kmask2 = il > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
    const float ml  = float(d_all) * sc * 32.0f;
    const float dl0 = float(d_all) * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const uint shr_h = il > 2 ? 2u : 0u;
    const uint shl_h = il > 1 ? 0u : (il > 0 ? 2u : 4u);
    const uint shr_l = il > 1 ? 4u : 0u;

    for (int i = 0; i < 4; ++i) {
        const uint low  = (uint(ql[2 * i]) | (uint(ql[2 * i + 1]) << 16)) & kmask2;
        const uint high = (uint(qh[2 * i]) | (uint(qh[2 * i + 1]) << 16)) & kmask1;
        const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = half(dl0 * float(q & 0xFFu)         - ml);
        reg[i][1] = half(dl1 * float(q & 0xFF00u)        - ml);
        reg[i][2] = half(dl2 * float(q & 0xFF0000u)      - ml);
        reg[i][3] = half(dl3 * float(q & 0xFF000000u)    - ml);
    }
}

kernel void dequant_batch_q6_k_blocked(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    threadgroup char* shmem    [[threadgroup(0)]],
    uint2 tgpig                [[threadgroup_position_in_grid]],
    ushort tiitg               [[thread_index_in_threadgroup]],
    ushort sgitg               [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;  // = 2
    constexpr short NL1 = NK / 8;   // = 4
    constexpr short QK_NL = 16;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = short(min(uint(NR1), N - uint(r1)));
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    short il = il0;

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    const short offset1 = il0 / QK_NL;
    device const Q6_K_Block* x = A + uint(r0 + lr0) * blocks_per_row + offset1;

    const short iy = short(8 * (tiitg % NL1));
    device const float* y = B + uint(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q6k_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
            (half2x4)(*(device float2x4*)y);

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= int(M) && r1 + NR1 <= int(N)) {
        device float* out = C
            + uint(r0 + 32 * (sgitg & 1))
            + uint(r1 + 16 * (sgitg >> 1)) * M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], out + 8 * (i % 4) + 8 * M * (i / 4), M, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float*  D  = C + uint(r0) + uint(r1 + j) * M;
                device float4* D4 = (device float4*)D;
                threadgroup float*  Cs  = temp_str + j * NR0;
                threadgroup float4* C4 = (threadgroup float4*)Cs;
                int i = 0;
                for (; i < nr0 / 4; i++) { *(D4 + i) = *(C4 + i); }
                i *= 4;
                for (; i < nr0; i++) { *(D + i) = *(Cs + i); }
            }
        }
    }
}

// f16-IO variant:
// - B input is f16 (halves activation bandwidth)
// - C output is f16 (f32 accumulators preserved for numeric stability)
kernel void dequant_batch_q4_k_f16io(
    device const Q4_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device half* C             [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup float tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];
    threadgroup float row_d[DB_BM];
    threadgroup float row_sc[DB_BM * 8];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DB_BK;

        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                uchar byte = blk.qs[pair * 32 + (c < 32 ? c : c - 32)];
                if (c < 32) {
                    tg_A[r * DB_BK + c] = d * sm1.x * float(byte & 0x0F) - dmin * sm1.y;
                } else {
                    tg_A[r * DB_BK + c] = d * sm2.x * float(byte >> 4) - dmin * sm2.y;
                }
            } else {
                tg_A[r * DB_BK + c] = 0.0f;
            }
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_float8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float out_tile[DB_BN * DB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0], DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8], DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = half(out_tile[r * DB_BM + c]);
        }
    }
}

kernel void dequant_batch_q6_k_f16io(
    device const Q6_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device half* C             [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup float tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);

                uint ql_base = group * 64;
                uint qh_base = group * 32;
                uint sub = c / 32;
                uint l = c % 32;
                uint is = l / 16;
                uint ql_idx = ql_base + sub_pair * 32;
                uint qh_idx = qh_base;

                int q;
                float sc;
                if (sub == 0) {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                        sc = float(blk.scales[group * 8 + is]);
                    } else {
                        q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        sc = float(blk.scales[group * 8 + is + 4]);
                    }
                } else {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        sc = float(blk.scales[group * 8 + is + 2]);
                    } else {
                        q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        sc = float(blk.scales[group * 8 + is + 6]);
                    }
                }
                tg_A[r * DB_BK + c] = d * sc * float(q);
            } else {
                tg_A[r * DB_BK + c] = 0.0f;
            }
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_float8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float out_tile[DB_BN * DB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0], DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8], DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = half(out_tile[r * DB_BM + c]);
        }
    }
}

// f16-input variant with f32 output:
// - B input is f16 (bandwidth reduction)
// - C output remains f32 (avoids f16->f32 cast after matmul)
kernel void dequant_batch_q4_k_f16in(
    device const Q4_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];
    // Precomputed dequant params (same as f32 kernel's paired-nibble pattern).
    threadgroup half row_dsc1[DB_BM];
    threadgroup half row_dmin1[DB_BM];
    threadgroup half row_dsc2[DB_BM];
    threadgroup half row_dmin2[DB_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DB_BK;

        // Phase 1: preload precomputed d*scale and dmin*min (same as f32 kernel).
        if (tid < DB_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d    = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2,     blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                row_dsc1[tid]  = half(d * sm1.x);
                row_dmin1[tid] = half(dmin * sm1.y);
                row_dsc2[tid]  = half(d * sm2.x);
                row_dmin2[tid] = half(dmin * sm2.y);
            } else {
                row_dsc1[tid]  = half(0.0f);
                row_dmin1[tid] = half(0.0f);
                row_dsc2[tid]  = half(0.0f);
                row_dmin2[tid] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble extraction (same as f32 kernel — 4 iters, no branch).
        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB_TG) {
            uint r = i / (DB_BK / 2);
            uint b = i % (DB_BK / 2);
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar byte = blk.qs[pair * 32 + b];
                tg_A[r * DB_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
                tg_A[r * DB_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
            } else {
                tg_A[r * DB_BK + b]      = half(0.0f);
                tg_A[r * DB_BK + b + 32] = half(0.0f);
            }
        }

        // Phase 3: B load (half input — no cast needed, half bandwidth vs f32 kernel).
        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + DB_BN <= N && tile_m + DB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
        simdgroup_store(acc0, c_base + 0, C_STRIDE);
        simdgroup_store(acc1, c_base + 8, C_STRIDE);
        simdgroup_store(acc2, c_base + 16, C_STRIDE);
        simdgroup_store(acc3, c_base + 24, C_STRIDE);
        return;
    }

    threadgroup float out_tile[DB_BN * DB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0], DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8], DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * C_STRIDE + gm] = out_tile[r * DB_BM + c];
        }
    }
}

kernel void dequant_batch_q6_k_f16in(
    device const Q6_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);

                uint ql_base = group * 64;
                uint qh_base = group * 32;
                uint sc_base = group * 8;
                uint sub = c / 32;
                uint l = c % 32;
                uint is = l / 16;
                uint ql_idx = ql_base + sub_pair * 32;
                uint qh_idx = qh_base;

                int q;
                float sc;
                if (sub == 0) {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is]);
                    } else {
                        q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 4]);
                    }
                } else {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 2]);
                    } else {
                        q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 6]);
                    }
                }
                tg_A[r * DB_BK + c] = half(d * sc * float(q));
            } else {
                tg_A[r * DB_BK + c] = half(0.0f);
            }
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + DB_BN <= N && tile_m + DB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
        simdgroup_store(acc0, c_base + 0, C_STRIDE);
        simdgroup_store(acc1, c_base + 8, C_STRIDE);
        simdgroup_store(acc2, c_base + 16, C_STRIDE);
        simdgroup_store(acc3, c_base + 24, C_STRIDE);
        return;
    }

    threadgroup float out_tile[DB_BN * DB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0], DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8], DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * C_STRIDE + gm] = out_tile[r * DB_BM + c];
        }
    }
}

// Q8_0 batch dequant + matmul with f16 input, f32 output.
//
// C[N×M] = B[N×K] × dequant(A[M×K])^T
// - A is Q8_0 blocks (32 values per block)
// - B is f16 activations
// - C is f32 output
inline void dequantize_q8_0_blocked(
    device const Q8_0_Block* xb,
    short il,
    thread half4x4& reg
) {
    device const char* qs = xb->qs + 16 * il;
    const float d = float(xb->d);
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = half(d * float(int(qs[i])));
    }
}

inline void dequantize_q4_0_blocked(
    device const Q4_0_Block* xb,
    short il,
    thread half4x4& reg
) {
    const float d = float(xb->d);
    for (int i = 0; i < 16; ++i) {
        uchar byte = xb->qs[i];
        uchar q = il == 0 ? (byte & 0x0F) : (byte >> 4);
        reg[i / 4][i % 4] = half(d * float(int(q) - 8));
    }
}

kernel void dequant_batch_q4_0_blocked_f16in(
    device const Q4_0_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    threadgroup char* shmem    [[threadgroup(0)]],
    uint2 tgpig                [[threadgroup_position_in_grid]],
    ushort tiitg               [[thread_index_in_threadgroup]],
    ushort sgitg               [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = short(min(uint(NR1), N - uint(r1)));
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il = short(tiitg % NL0);

    uint blocks_per_row = K / Q4_0_BLOCK_VALUES;
    device const Q4_0_Block* x = A + uint(r0 + lr0) * blocks_per_row;

    const short iy = short(8 * (tiitg % NL1));
    device const half* y = B + uint(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q4_0_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = *(device half2x4*)y;

        x += 1;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (!BLOCKED_BC_OUT) {
        device float* out = C
            + uint(r0 + 32 * (sgitg & 1))
            + uint(r1 + 16 * (sgitg >> 1)) * C_STRIDE;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(
                mc[i],
                out + 8 * (i % 4) + 8 * C_STRIDE * (i / 4),
                C_STRIDE,
                ulong2(0, 0),
                false
            );
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, ulong2(0, 0), false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float* D = C + uint(r0) + uint(r1 + j) * C_STRIDE;
                device float4* D4 = (device float4*)D;
                threadgroup float* Cs = temp_str + j * NR0;
                threadgroup float4* C4 = (threadgroup float4*)Cs;

                int i = 0;
                for (; i < nr0 / 4; i++) {
                    *(D4 + i) = *(C4 + i);
                }
                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(Cs + i);
                }
            }
        }
    }
}

kernel void dequant_batch_q8_0_blocked_f16in(
    device const Q8_0_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    threadgroup char* shmem    [[threadgroup(0)]],
    uint2 tgpig                [[threadgroup_position_in_grid]],
    ushort tiitg               [[thread_index_in_threadgroup]],
    ushort sgitg               [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = short(min(uint(NR1), N - uint(r1)));
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il = short(tiitg % NL0);

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    device const Q8_0_Block* x = A + uint(r0 + lr0) * blocks_per_row;

    const short iy = short(8 * (tiitg % NL1));
    device const half* y = B + uint(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q8_0_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = *(device half2x4*)y;

        x += 1;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (!BLOCKED_BC_OUT) {
        device float* out = C
            + uint(r0 + 32 * (sgitg & 1))
            + uint(r1 + 16 * (sgitg >> 1)) * C_STRIDE;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(
                mc[i],
                out + 8 * (i % 4) + 8 * C_STRIDE * (i / 4),
                C_STRIDE,
                ulong2(0, 0),
                false
            );
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, ulong2(0, 0), false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float* D = C + uint(r0) + uint(r1 + j) * C_STRIDE;
                device float4* D4 = (device float4*)D;
                threadgroup float* Cs = temp_str + j * NR0;
                threadgroup float4* C4 = (threadgroup float4*)Cs;

                int i = 0;
                for (; i < nr0 / 4; i++) {
                    *(D4 + i) = *(C4 + i);
                }
                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(Cs + i);
                }
            }
        }
    }
}

kernel void dequant_batch_q8_0_f16in(
    device const Q8_0_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint global_r = tile_m + r;
            uint global_k = kt + c;
            if (global_r < M && global_k < K) {
                uint block_idx = global_k / Q8_0_BLOCK_VALUES;
                uint in_block = global_k % Q8_0_BLOCK_VALUES;
                device const Q8_0_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);
                int q = int(blk.qs[in_block]);
                tg_A[r * DB_BK + c] = half(d * float(q));
            } else {
                tg_A[r * DB_BK + c] = half(0.0f);
            }
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + DB_BN <= N && tile_m + DB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
        simdgroup_store(acc0, c_base + 0, C_STRIDE);
        simdgroup_store(acc1, c_base + 8, C_STRIDE);
        simdgroup_store(acc2, c_base + 16, C_STRIDE);
        simdgroup_store(acc3, c_base + 24, C_STRIDE);
        return;
    }

    threadgroup float out_tile[DB_BN * DB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * DB_BM + 0], DB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DB_BM + 8], DB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * C_STRIDE + gm] = out_tile[r * DB_BM + c];
        }
    }
}

// Full-tile fast path for Q8_0 f16-input batch kernel.
// Preconditions enforced by dispatch:
// - M is divisible by DB_BM (32)
// - N is divisible by DB_BN (64)
// - K is divisible by DB_BK (64)
kernel void dequant_batch_q8_0_f16in_full(
    device const Q8_0_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint global_r = tile_m + r;
            uint global_k = kt + c;
            uint block_idx = global_k / Q8_0_BLOCK_VALUES;
            uint in_block = global_k % Q8_0_BLOCK_VALUES;
            device const Q8_0_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d = float(blk.d);
            int q = int(blk.qs[in_block]);
            tg_A[r * DB_BK + c] = half(d * float(q));
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0, C_STRIDE);
    simdgroup_store(acc1, c_base + 8, C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
}

// Full-tile fast paths for f16-input kernels.
// Preconditions enforced by dispatch:
// - M is divisible by DB_BM
// - N is divisible by DB_BN

kernel void dequant_batch_q4_k_f16in_full(
    device const Q4_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];
    threadgroup half row_dsc1[DB_BM];
    threadgroup half row_dmin1[DB_BM];
    threadgroup half row_dsc2[DB_BM];
    threadgroup half row_dmin2[DB_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DB_BK;

        // Phase 1: precompute d*scale and dmin*min.
        for (uint r = tid; r < DB_BM; r += DB_TG) {
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
            float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
            row_dsc1[r]  = half(d * sm1.x);
            row_dmin1[r] = half(dmin * sm1.y);
            row_dsc2[r]  = half(d * sm2.x);
            row_dmin2[r] = half(dmin * sm2.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble extraction (no branch).
        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB_TG) {
            uint r = i / (DB_BK / 2);
            uint b = i % (DB_BK / 2);
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            uchar byte = blk.qs[pair * 32 + b];
            tg_A[r * DB_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
            tg_A[r * DB_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
    simdgroup_store(acc0, c_base + 0, M);
    simdgroup_store(acc1, c_base + 8, M);
    simdgroup_store(acc2, c_base + 16, M);
    simdgroup_store(acc3, c_base + 24, M);
}

kernel void dequant_batch_q6_k_f16in_full(
    device const Q6_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint global_r = tile_m + r;
            device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d = float(blk.d);

            uint ql_base = group * 64;
            uint qh_base = group * 32;
            uint sc_base = group * 8;
            uint sub = c / 32;
            uint l = c % 32;
            uint is = l / 16;
            uint ql_idx = ql_base + sub_pair * 32;
            uint qh_idx = qh_base;

            int q;
            float sc;
            if (sub == 0) {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is]);
                } else {
                    q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 4]);
                }
            } else {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 2]);
                } else {
                    q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 6]);
                }
            }
            tg_A[r * DB_BK + c] = half(d * sc * float(q));
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * DB_BK + kk * 8], DB_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
    simdgroup_store(acc0, c_base + 0, M);
    simdgroup_store(acc1, c_base + 8, M);
    simdgroup_store(acc2, c_base + 16, M);
    simdgroup_store(acc3, c_base + 24, M);
}

// 64x64 full-tile fast paths for f16-input kernels.
// Preconditions enforced by dispatch:
// - M is divisible by 64
// - N is divisible by 64
// - no boundary masking required

kernel void dequant_batch_q4_k_f16in_full64(
    device const Q4_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D64_BM;
    uint tile_n = group_id.y * D64_BN;

    threadgroup half tg_A[D64_BM * D64_BK];
    threadgroup half tg_B[D64_BN * D64_BK];
    threadgroup half row_dsc1[D64_BM];
    threadgroup half row_dmin1[D64_BM];
    threadgroup half row_dsc2[D64_BM];
    threadgroup half row_dmin2[D64_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / D64_BK;

        // Phase 1: precompute d*scale and dmin*min.
        for (uint r = tid; r < D64_BM; r += D64_TG) {
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
            float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
            row_dsc1[r]  = half(d * sm1.x);
            row_dmin1[r] = half(dmin * sm1.y);
            row_dsc2[r]  = half(d * sm2.x);
            row_dmin2[r] = half(dmin * sm2.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble extraction (no branch).
        for (uint i = tid; i < D64_BM * (D64_BK / 2); i += D64_TG) {
            uint r = i / (D64_BK / 2);
            uint b = i % (D64_BK / 2);
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            uchar byte = blk.qs[pair * 32 + b];
            tg_A[r * D64_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
            tg_A[r * D64_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
        }

        for (uint i = tid; i < D64_BN * D64_BK; i += D64_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * D64_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0,  C_STRIDE);
    simdgroup_store(acc1, c_base + 8,  C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
    simdgroup_store(acc4, c_base + 32, C_STRIDE);
    simdgroup_store(acc5, c_base + 40, C_STRIDE);
    simdgroup_store(acc6, c_base + 48, C_STRIDE);
    simdgroup_store(acc7, c_base + 56, C_STRIDE);
}

// 64x64 full-tile BK=32 variant for Q4_K f16in.
// This reduces per-iteration working set and increases K-loop iterations.
constant uint D64S_BM = 64;
constant uint D64S_BN = 64;
constant uint D64S_BK = 32;
constant uint D64S_TG = 256;

kernel void dequant_batch_q4_k_f16in_full64_bk32(
    device const Q4_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D64S_BM;
    uint tile_n = group_id.y * D64S_BN;

    threadgroup half tg_A[D64S_BM * D64S_BK];
    threadgroup half tg_B[D64S_BN * D64S_BK];
    threadgroup half row_dsc[D64S_BM];
    threadgroup half row_dmn[D64S_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64S_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair64 = (kt % Q4_K_BLOCK_VALUES) / 64;
        uint half32 = (kt % 64) / 32;

        // Phase 1: precompute d*scale and dmin*min for the active half.
        for (uint r = tid; r < D64S_BM; r += D64S_TG) {
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            float2 sm1 = get_scale_min_q4k(pair64 * 2, blk.scales);
            float2 sm2 = get_scale_min_q4k(pair64 * 2 + 1, blk.scales);
            if (half32 == 0) {
                row_dsc[r] = half(d * sm1.x);
                row_dmn[r] = half(dmin * sm1.y);
            } else {
                row_dsc[r] = half(d * sm2.x);
                row_dmn[r] = half(dmin * sm2.y);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: dequant with precomputed scales (no branch in inner loop).
        for (uint i = tid; i < D64S_BM * D64S_BK; i += D64S_TG) {
            uint r = i / D64S_BK;
            uint c = i % D64S_BK;
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            uchar byte = blk.qs[pair64 * 32 + c];
            float q = (half32 == 0) ? float(byte & 0x0F) : float(byte >> 4);
            tg_A[r * D64S_BK + c] = half(float(row_dsc[r]) * q - float(row_dmn[r]));
        }

        for (uint i = tid; i < D64S_BN * D64S_BK; i += D64S_TG) {
            uint r = i / D64S_BK;
            uint c = i % D64S_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * D64S_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64S_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64S_BK + kk * 8], D64S_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64S_BK + kk * 8], D64S_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64S_BK + kk * 8], D64S_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64S_BK + kk * 8], D64S_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64S_BK + kk * 8], D64S_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64S_BK + kk * 8], D64S_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64S_BK + kk * 8], D64S_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64S_BK + kk * 8], D64S_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64S_BK + kk * 8], D64S_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0,  C_STRIDE);
    simdgroup_store(acc1, c_base + 8,  C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
    simdgroup_store(acc4, c_base + 32, C_STRIDE);
    simdgroup_store(acc5, c_base + 40, C_STRIDE);
    simdgroup_store(acc6, c_base + 48, C_STRIDE);
    simdgroup_store(acc7, c_base + 56, C_STRIDE);
}

kernel void dequant_batch_q6_k_f16in_full64(
    device const Q6_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D64_BM;
    uint tile_n = group_id.y * D64_BN;

    threadgroup half tg_A[D64_BM * D64_BK];
    threadgroup half tg_B[D64_BN * D64_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < D64_BM * D64_BK; i += D64_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint global_r = tile_m + r;
            device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d = float(blk.d);

            uint ql_base = group * 64;
            uint qh_base = group * 32;
            uint sc_base = group * 8;
            uint sub = c / 32;
            uint l = c % 32;
            uint is = l / 16;
            uint ql_idx = ql_base + sub_pair * 32;
            uint qh_idx = qh_base;

            int q;
            float sc;
            if (sub == 0) {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is]);
                } else {
                    q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 4]);
                }
            } else {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 2]);
                } else {
                    q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 6]);
                }
            }
            tg_A[r * D64_BK + c] = half(d * sc * float(q));
        }

        for (uint i = tid; i < D64_BN * D64_BK; i += D64_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * D64_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0,  C_STRIDE);
    simdgroup_store(acc1, c_base + 8,  C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
    simdgroup_store(acc4, c_base + 32, C_STRIDE);
    simdgroup_store(acc5, c_base + 40, C_STRIDE);
    simdgroup_store(acc6, c_base + 48, C_STRIDE);
    simdgroup_store(acc7, c_base + 56, C_STRIDE);
}

kernel void dequant_batch_q8_0_f16in_full64(
    device const Q8_0_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D64_BM;
    uint tile_n = group_id.y * D64_BN;

    threadgroup half tg_A[D64_BM * D64_BK];
    threadgroup half tg_B[D64_BN * D64_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        for (uint i = tid; i < D64_BM * D64_BK; i += D64_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint global_r = tile_m + r;
            uint global_k = kt + c;
            uint block_idx = global_k / Q8_0_BLOCK_VALUES;
            uint in_block = global_k % Q8_0_BLOCK_VALUES;
            device const Q8_0_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d = float(blk.d);
            int q = int(blk.qs[in_block]);
            tg_A[r * D64_BK + c] = half(d * float(q));
        }

        for (uint i = tid; i < D64_BN * D64_BK; i += D64_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * D64_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0,  C_STRIDE);
    simdgroup_store(acc1, c_base + 8,  C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
    simdgroup_store(acc4, c_base + 32, C_STRIDE);
    simdgroup_store(acc5, c_base + 40, C_STRIDE);
    simdgroup_store(acc6, c_base + 48, C_STRIDE);
    simdgroup_store(acc7, c_base + 56, C_STRIDE);
}

kernel void dequant_batch_q8_0_f16in_full32(
    device const Q8_0_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint BM = 64;
    constexpr uint BN = 32;
    constexpr uint BK = 64;
    constexpr uint TG = 128;
    uint tile_m = group_id.x * BM;
    uint tile_n = group_id.y * BN;

    threadgroup half tg_A[BM * BK];
    threadgroup half tg_B[BN * BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        for (uint i = tid; i < BM * BK; i += TG) {
            uint r = i / BK;
            uint c = i % BK;
            uint global_r = tile_m + r;
            uint global_k = kt + c;
            uint block_idx = global_k / Q8_0_BLOCK_VALUES;
            uint in_block = global_k % Q8_0_BLOCK_VALUES;
            device const Q8_0_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d = float(blk.d);
            int q = int(blk.qs[in_block]);
            tg_A[r * BK + c] = half(d * float(q));
        }

        for (uint i = tid; i < BN * BK; i += TG) {
            uint r = i / BK;
            uint c = i % BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * BK + kk * 8], BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * BK + kk * 8], BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0,  C_STRIDE);
    simdgroup_store(acc1, c_base + 8,  C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
    simdgroup_store(acc4, c_base + 32, C_STRIDE);
    simdgroup_store(acc5, c_base + 40, C_STRIDE);
    simdgroup_store(acc6, c_base + 48, C_STRIDE);
    simdgroup_store(acc7, c_base + 56, C_STRIDE);
}

kernel void dequant_batch_q8_0_f16in_tail32(
    device const Q8_0_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint BM = 64;
    constexpr uint BN = 32;
    constexpr uint TG = 128;
    uint tile_m = group_id.x * BM;

    threadgroup half tg_A[BM * D64_BK];
    threadgroup half tg_B[BN * D64_BK];
    threadgroup float out_tile[BN * BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        for (uint i = tid; i < BM * D64_BK; i += TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint global_r = tile_m + r;
            uint global_k = kt + c;
            uint block_idx = global_k / Q8_0_BLOCK_VALUES;
            uint in_block = global_k % Q8_0_BLOCK_VALUES;
            device const Q8_0_Block& blk = A[global_r * blocks_per_row + block_idx];
            tg_A[r * D64_BK + c] = half(float(blk.d) * float(int(blk.qs[in_block])));
        }

        for (uint i = tid; i < BN * D64_BK; i += TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gk = kt + c;
            tg_B[r * D64_BK + c] = (r < N) ? B[r * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* tg_base = out_tile + simd_id * 8 * BM;
    simdgroup_store(acc0, tg_base + 0,  BM);
    simdgroup_store(acc1, tg_base + 8,  BM);
    simdgroup_store(acc2, tg_base + 16, BM);
    simdgroup_store(acc3, tg_base + 24, BM);
    simdgroup_store(acc4, tg_base + 32, BM);
    simdgroup_store(acc5, tg_base + 40, BM);
    simdgroup_store(acc6, tg_base + 48, BM);
    simdgroup_store(acc7, tg_base + 56, BM);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < BN * BM; i += TG) {
        uint n_idx = i / BM;
        uint m_idx = i % BM;
        if (n_idx < N) {
            C[n_idx * C_STRIDE + (tile_m + m_idx)] = out_tile[i];
        }
    }
}

kernel void dequant_batch_q8_0_f16in_full32x32(
    device const Q8_0_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint BM = 32;
    constexpr uint BN = 32;
    constexpr uint BK = 64;
    constexpr uint TG = 128;
    uint tile_m = group_id.x * BM;
    uint tile_n = group_id.y * BN;

    threadgroup half tg_A[BM * BK];
    threadgroup half tg_B[BN * BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += BK) {
        for (uint i = tid; i < BM * BK; i += TG) {
            uint r = i / BK;
            uint c = i % BK;
            uint global_r = tile_m + r;
            uint global_k = kt + c;
            uint block_idx = global_k / Q8_0_BLOCK_VALUES;
            uint in_block = global_k % Q8_0_BLOCK_VALUES;
            device const Q8_0_Block& blk = A[global_r * blocks_per_row + block_idx];
            tg_A[r * BK + c] = half(float(blk.d) * float(int(blk.qs[in_block])));
        }

        for (uint i = tid; i < BN * BK; i += TG) {
            uint r = i / BK;
            uint c = i % BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * BK + kk * 8], BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * BK + kk * 8], BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * BK + kk * 8], BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0,  C_STRIDE);
    simdgroup_store(acc1, c_base + 8,  C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
}

// ── 64×32 full-tile fast paths for f16-input kernels ──────────────────
//
// BM=64, BN=32, BK=64, TG=128 (4 simdgroups).
// TG memory: tg_A(8KB) + tg_B(4KB) + scales(768B) ≈ 13KB
//   → 2 TGs fit per SM within the 32KB threadgroup memory limit
//   vs 1 TG for the BN=64 variant (~17KB)
// Grid: (M/64, N/32) → 2× more TGs than full64 for the same N coverage.
// Preconditions enforced by dispatch:
//   - M is divisible by 64
//   - N is divisible by 32

constant uint D32_BM = 64;
constant uint D32_BN = 32;
constant uint D32_TG = 128;  // 4 simdgroups × 32 threads

kernel void dequant_batch_q4_k_f16in_full32(
    device const Q4_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D32_BM;
    uint tile_n = group_id.y * D32_BN;

    threadgroup half tg_A[D32_BM * D64_BK];
    threadgroup half tg_B[D32_BN * D64_BK];
    threadgroup half row_dsc1[D32_BM];
    threadgroup half row_dmin1[D32_BM];
    threadgroup half row_dsc2[D32_BM];
    threadgroup half row_dmin2[D32_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / D64_BK;

        // Phase 1: precompute d*scale and dmin*min.
        for (uint r = tid; r < D32_BM; r += D32_TG) {
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
            float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
            row_dsc1[r]  = half(d * sm1.x);
            row_dmin1[r] = half(dmin * sm1.y);
            row_dsc2[r]  = half(d * sm2.x);
            row_dmin2[r] = half(dmin * sm2.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble extraction (no branch).
        for (uint i = tid; i < D32_BM * (D64_BK / 2); i += D32_TG) {
            uint r = i / (D64_BK / 2);
            uint b = i % (D64_BK / 2);
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            uchar byte = blk.qs[pair * 32 + b];
            tg_A[r * D64_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
            tg_A[r * D64_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
        }

        for (uint i = tid; i < D32_BN * D64_BK; i += D32_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * D64_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0,  C_STRIDE);
    simdgroup_store(acc1, c_base + 8,  C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
    simdgroup_store(acc4, c_base + 32, C_STRIDE);
    simdgroup_store(acc5, c_base + 40, C_STRIDE);
    simdgroup_store(acc6, c_base + 48, C_STRIDE);
    simdgroup_store(acc7, c_base + 56, C_STRIDE);
}

kernel void dequant_batch_q6_k_f16in_full32(
    device const Q6_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D32_BM;
    uint tile_n = group_id.y * D32_BN;

    threadgroup half tg_A[D32_BM * D64_BK];
    threadgroup half tg_B[D32_BN * D64_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < D32_BM * D64_BK; i += D32_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint global_r = tile_m + r;
            device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d = float(blk.d);

            uint ql_base = group * 64;
            uint qh_base = group * 32;
            uint sc_base = group * 8;
            uint sub = c / 32;
            uint l = c % 32;
            uint is = l / 16;
            uint ql_idx = ql_base + sub_pair * 32;
            uint qh_idx = qh_base;

            int q;
            float sc;
            if (sub == 0) {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is]);
                } else {
                    q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 4]);
                }
            } else {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 2]);
                } else {
                    q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 6]);
                }
            }
            tg_A[r * D64_BK + c] = half(d * sc * float(q));
        }

        for (uint i = tid; i < D32_BN * D64_BK; i += D32_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * D64_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
    simdgroup_store(acc0, c_base + 0,  C_STRIDE);
    simdgroup_store(acc1, c_base + 8,  C_STRIDE);
    simdgroup_store(acc2, c_base + 16, C_STRIDE);
    simdgroup_store(acc3, c_base + 24, C_STRIDE);
    simdgroup_store(acc4, c_base + 32, C_STRIDE);
    simdgroup_store(acc5, c_base + 40, C_STRIDE);
    simdgroup_store(acc6, c_base + 48, C_STRIDE);
    simdgroup_store(acc7, c_base + 56, C_STRIDE);
}

// ── 64×32 N-tail boundary kernels for f16-input batch dequant+matmul ─
//
// Same BM/BK structure as full32 but handles n_tail < 32 N-rows.
// B loading uses a bounds check (gn < N) so callers can pass N=n_tail.
// Output written through a threadgroup staging buffer to avoid
// out-of-bounds simdgroup_store when n_tail is not a multiple of 8.
//
// TG memory: tg_A(8KB) + tg_B(4KB) + scales(768B) + out_tile(8KB) ≈ 21KB
// Grid: (m_full/D32_BM, 1)  [caller offsets B and C to the tail start]
// Preconditions:
//   - M is divisible by D32_BM (64) — caller uses m_full not raw M
//   - N is 1..31 (n_tail)

kernel void dequant_batch_q4_k_f16in_tail32(
    device const Q4_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D32_BM;

    threadgroup half tg_A[D32_BM * D64_BK];
    threadgroup half tg_B[D32_BN * D64_BK];
    threadgroup half row_dsc1[D32_BM];
    threadgroup half row_dmin1[D32_BM];
    threadgroup half row_dsc2[D32_BM];
    threadgroup half row_dmin2[D32_BM];
    threadgroup float out_tile[D32_BN * D32_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / D64_BK;

        // Phase 1: precompute d*scale and dmin*min.
        for (uint r = tid; r < D32_BM; r += D32_TG) {
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
            float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
            row_dsc1[r]  = half(d * sm1.x);
            row_dmin1[r] = half(dmin * sm1.y);
            row_dsc2[r]  = half(d * sm2.x);
            row_dmin2[r] = half(dmin * sm2.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble extraction (no branch).
        for (uint i = tid; i < D32_BM * (D64_BK / 2); i += D32_TG) {
            uint r = i / (D64_BK / 2);
            uint b = i % (D64_BK / 2);
            uint global_r = tile_m + r;
            device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            uchar byte = blk.qs[pair * 32 + b];
            tg_A[r * D64_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
            tg_A[r * D64_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
        }

        // B tile load with N bounds check (handles n_tail < D32_BN)
        for (uint i = tid; i < D32_BN * D64_BK; i += D32_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gk = kt + c;
            tg_B[r * D64_BK + c] = (r < N) ? B[r * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulators into threadgroup staging buffer (layout: [n_row * D32_BM + m_col])
    threadgroup float* tg_base = out_tile + simd_id * 8 * D32_BM;
    simdgroup_store(acc0, tg_base + 0,  D32_BM);
    simdgroup_store(acc1, tg_base + 8,  D32_BM);
    simdgroup_store(acc2, tg_base + 16, D32_BM);
    simdgroup_store(acc3, tg_base + 24, D32_BM);
    simdgroup_store(acc4, tg_base + 32, D32_BM);
    simdgroup_store(acc5, tg_base + 40, D32_BM);
    simdgroup_store(acc6, tg_base + 48, D32_BM);
    simdgroup_store(acc7, tg_base + 56, D32_BM);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter to global C with N bounds check
    for (uint i = tid; i < D32_BN * D32_BM; i += D32_TG) {
        uint n_idx = i / D32_BM;
        uint m_idx = i % D32_BM;
        if (n_idx < N) {
            C[n_idx * C_STRIDE + (tile_m + m_idx)] = out_tile[i];
        }
    }
}

kernel void dequant_batch_q6_k_f16in_tail32(
    device const Q6_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D32_BM;

    threadgroup half tg_A[D32_BM * D64_BK];
    threadgroup half tg_B[D32_BN * D64_BK];
    threadgroup float out_tile[D32_BN * D32_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);
    acc4 = simdgroup_float8x8(0);
    acc5 = simdgroup_float8x8(0);
    acc6 = simdgroup_float8x8(0);
    acc7 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < D32_BM * D64_BK; i += D32_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint global_r = tile_m + r;
            device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            float d = float(blk.d);

            uint ql_base = group * 64;
            uint qh_base = group * 32;
            uint sc_base = group * 8;
            uint sub = c / 32;
            uint l = c % 32;
            uint is = l / 16;
            uint ql_idx = ql_base + sub_pair * 32;
            uint qh_idx = qh_base;

            int q;
            float sc;
            if (sub == 0) {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is]);
                } else {
                    q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 4]);
                }
            } else {
                if (sub_pair == 0) {
                    q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 2]);
                } else {
                    q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                    sc = float(blk.scales[sc_base + is + 6]);
                }
            }
            tg_A[r * D64_BK + c] = half(d * sc * float(q));
        }

        // B tile load with N bounds check (handles n_tail < D32_BN)
        for (uint i = tid; i < D32_BN * D64_BK; i += D32_TG) {
            uint r = i / D64_BK;
            uint c = i % D64_BK;
            uint gk = kt + c;
            tg_B[r * D64_BK + c] = (r < N) ? B[r * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_half8x8 a0, a1, a2, a3, a4, a5, a6, a7;
            simdgroup_load(a0, &tg_A[0  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a1, &tg_A[8  * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a2, &tg_A[16 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a3, &tg_A[24 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a4, &tg_A[32 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a5, &tg_A[40 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a6, &tg_A[48 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);
            simdgroup_load(a7, &tg_A[56 * D64_BK + kk * 8], D64_BK, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
            simdgroup_multiply_accumulate(acc4, b_frag, a4, acc4);
            simdgroup_multiply_accumulate(acc5, b_frag, a5, acc5);
            simdgroup_multiply_accumulate(acc6, b_frag, a6, acc6);
            simdgroup_multiply_accumulate(acc7, b_frag, a7, acc7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulators into threadgroup staging buffer
    threadgroup float* tg_base = out_tile + simd_id * 8 * D32_BM;
    simdgroup_store(acc0, tg_base + 0,  D32_BM);
    simdgroup_store(acc1, tg_base + 8,  D32_BM);
    simdgroup_store(acc2, tg_base + 16, D32_BM);
    simdgroup_store(acc3, tg_base + 24, D32_BM);
    simdgroup_store(acc4, tg_base + 32, D32_BM);
    simdgroup_store(acc5, tg_base + 40, D32_BM);
    simdgroup_store(acc6, tg_base + 48, D32_BM);
    simdgroup_store(acc7, tg_base + 56, D32_BM);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter to global C with N bounds check
    for (uint i = tid; i < D32_BN * D32_BM; i += D32_TG) {
        uint n_idx = i / D32_BM;
        uint m_idx = i % D32_BM;
        if (n_idx < N) {
            C[n_idx * C_STRIDE + (tile_m + m_idx)] = out_tile[i];
        }
    }
}

// ── Batch Dequant+Matmul Pair (B-transposed) ──────────────────────────
//
// Compute two outputs sharing the same input B:
//   C0[N × M] = B[N × K] × dequant(A0[M × K])^T
//   C1[N × M] = B[N × K] × dequant(A1[M × K])^T
//
// Used for FFN gate+up projections to reuse B tile loads.

constant uint PB_BM = 32;
constant uint PB_BN = 32;
constant uint PB_BK = 64;
constant uint PB_TG = 128;

kernel void dequant_batch_pair_q4_k(
    device const Q4_K_Block* A0 [[buffer(0)]],
    device const Q4_K_Block* A1 [[buffer(1)]],
    device const float* B       [[buffer(2)]],
    device float* C0            [[buffer(3)]],
    device float* C1            [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& K            [[buffer(7)]],
    uint2 group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * PB_BM;
    uint tile_n = group_id.y * PB_BN;

    threadgroup float tg_A0[PB_BM * PB_BK];
    threadgroup float tg_A1[PB_BM * PB_BK];
    threadgroup float tg_B[PB_BN * PB_BK];

    simdgroup_float8x8 acc00, acc01, acc02, acc03;
    simdgroup_float8x8 acc10, acc11, acc12, acc13;
    acc00 = simdgroup_float8x8(0);
    acc01 = simdgroup_float8x8(0);
    acc02 = simdgroup_float8x8(0);
    acc03 = simdgroup_float8x8(0);
    acc10 = simdgroup_float8x8(0);
    acc11 = simdgroup_float8x8(0);
    acc12 = simdgroup_float8x8(0);
    acc13 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += PB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / PB_BK;

        for (uint i = tid; i < PB_BM * PB_BK; i += PB_TG) {
            uint r = i / PB_BK;
            uint c = i % PB_BK;
            uint global_r = tile_m + r;

            if (global_r < M) {
                device const Q4_K_Block& blk0 = A0[global_r * blocks_per_row + block_idx];
                device const Q4_K_Block& blk1 = A1[global_r * blocks_per_row + block_idx];

                float2 sm0a = get_scale_min_q4k(pair * 2, blk0.scales);
                float2 sm0b = get_scale_min_q4k(pair * 2 + 1, blk0.scales);
                float2 sm1a = get_scale_min_q4k(pair * 2, blk1.scales);
                float2 sm1b = get_scale_min_q4k(pair * 2 + 1, blk1.scales);

                uchar byte0 = blk0.qs[pair * 32 + (c < 32 ? c : c - 32)];
                uchar byte1 = blk1.qs[pair * 32 + (c < 32 ? c : c - 32)];

                if (c < 32) {
                    tg_A0[r * PB_BK + c] = float(blk0.d) * sm0a.x * float(byte0 & 0x0F) - float(blk0.dmin) * sm0a.y;
                    tg_A1[r * PB_BK + c] = float(blk1.d) * sm1a.x * float(byte1 & 0x0F) - float(blk1.dmin) * sm1a.y;
                } else {
                    tg_A0[r * PB_BK + c] = float(blk0.d) * sm0b.x * float(byte0 >> 4) - float(blk0.dmin) * sm0b.y;
                    tg_A1[r * PB_BK + c] = float(blk1.d) * sm1b.x * float(byte1 >> 4) - float(blk1.dmin) * sm1b.y;
                }
            } else {
                tg_A0[r * PB_BK + c] = 0.0f;
                tg_A1[r * PB_BK + c] = 0.0f;
            }
        }

        for (uint i = tid; i < PB_BN * PB_BK; i += PB_TG) {
            uint r = i / PB_BK;
            uint c = i % PB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * PB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < PB_BK / 8; kk++) {
            simdgroup_float8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * PB_BK + kk * 8], PB_BK);

            simdgroup_float8x8 a00, a01, a02, a03;
            simdgroup_float8x8 a10, a11, a12, a13;
            simdgroup_load(a00, &tg_A0[0  * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a01, &tg_A0[8  * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a02, &tg_A0[16 * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a03, &tg_A0[24 * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a10, &tg_A1[0  * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a11, &tg_A1[8  * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a12, &tg_A1[16 * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a13, &tg_A1[24 * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc00, b_frag, a00, acc00);
            simdgroup_multiply_accumulate(acc01, b_frag, a01, acc01);
            simdgroup_multiply_accumulate(acc02, b_frag, a02, acc02);
            simdgroup_multiply_accumulate(acc03, b_frag, a03, acc03);
            simdgroup_multiply_accumulate(acc10, b_frag, a10, acc10);
            simdgroup_multiply_accumulate(acc11, b_frag, a11, acc11);
            simdgroup_multiply_accumulate(acc12, b_frag, a12, acc12);
            simdgroup_multiply_accumulate(acc13, b_frag, a13, acc13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + PB_BN <= N && tile_m + PB_BM <= M) {
        device float* c0 = C0 + (tile_n + simd_id * 8) * M + tile_m;
        device float* c1 = C1 + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc00, c0 + 0,  M);
        simdgroup_store(acc01, c0 + 8,  M);
        simdgroup_store(acc02, c0 + 16, M);
        simdgroup_store(acc03, c0 + 24, M);
        simdgroup_store(acc10, c1 + 0,  M);
        simdgroup_store(acc11, c1 + 8,  M);
        simdgroup_store(acc12, c1 + 16, M);
        simdgroup_store(acc13, c1 + 24, M);
        return;
    }

    threadgroup float out_tile[PB_BN * PB_BM];
    simdgroup_store(acc00, &out_tile[simd_id * 8 * PB_BM + 0],  PB_BM);
    simdgroup_store(acc01, &out_tile[simd_id * 8 * PB_BM + 8],  PB_BM);
    simdgroup_store(acc02, &out_tile[simd_id * 8 * PB_BM + 16], PB_BM);
    simdgroup_store(acc03, &out_tile[simd_id * 8 * PB_BM + 24], PB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < PB_BN * PB_BM; i += PB_TG) {
        uint r = i / PB_BM;
        uint c = i % PB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C0[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_store(acc10, &out_tile[simd_id * 8 * PB_BM + 0],  PB_BM);
    simdgroup_store(acc11, &out_tile[simd_id * 8 * PB_BM + 8],  PB_BM);
    simdgroup_store(acc12, &out_tile[simd_id * 8 * PB_BM + 16], PB_BM);
    simdgroup_store(acc13, &out_tile[simd_id * 8 * PB_BM + 24], PB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < PB_BN * PB_BM; i += PB_TG) {
        uint r = i / PB_BM;
        uint c = i % PB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C1[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
}

kernel void dequant_batch_pair_q6_k(
    device const Q6_K_Block* A0 [[buffer(0)]],
    device const Q6_K_Block* A1 [[buffer(1)]],
    device const float* B       [[buffer(2)]],
    device float* C0            [[buffer(3)]],
    device float* C1            [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& K            [[buffer(7)]],
    uint2 group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * PB_BM;
    uint tile_n = group_id.y * PB_BN;

    threadgroup float tg_A0[PB_BM * PB_BK];
    threadgroup float tg_A1[PB_BM * PB_BK];
    threadgroup float tg_B[PB_BN * PB_BK];

    simdgroup_float8x8 acc00, acc01, acc02, acc03;
    simdgroup_float8x8 acc10, acc11, acc12, acc13;
    acc00 = simdgroup_float8x8(0);
    acc01 = simdgroup_float8x8(0);
    acc02 = simdgroup_float8x8(0);
    acc03 = simdgroup_float8x8(0);
    acc10 = simdgroup_float8x8(0);
    acc11 = simdgroup_float8x8(0);
    acc12 = simdgroup_float8x8(0);
    acc13 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += PB_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < PB_BM * PB_BK; i += PB_TG) {
            uint r = i / PB_BK;
            uint c = i % PB_BK;
            uint global_r = tile_m + r;

            if (global_r < M) {
                device const Q6_K_Block& blk0 = A0[global_r * blocks_per_row + block_idx];
                device const Q6_K_Block& blk1 = A1[global_r * blocks_per_row + block_idx];

                uint ql_base = group * 64;
                uint qh_base = group * 32;
                uint sc_base = group * 8;
                uint sub = c / 32;
                uint l = c % 32;
                uint is = l / 16;
                uint ql_idx = ql_base + sub_pair * 32;
                uint qh_idx = qh_base;

                int q0, q1;
                float sc0, sc1;
                if (sub == 0) {
                    if (sub_pair == 0) {
                        q0 = int((blk0.ql[ql_idx + l] & 0x0F) | ((blk0.qh[qh_idx + l] & 3) << 4)) - 32;
                        q1 = int((blk1.ql[ql_idx + l] & 0x0F) | ((blk1.qh[qh_idx + l] & 3) << 4)) - 32;
                        sc0 = float(blk0.scales[sc_base + is]);
                        sc1 = float(blk1.scales[sc_base + is]);
                    } else {
                        q0 = int((blk0.ql[ql_idx + l] >> 4) | (((blk0.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        q1 = int((blk1.ql[ql_idx + l] >> 4) | (((blk1.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        sc0 = float(blk0.scales[sc_base + is + 4]);
                        sc1 = float(blk1.scales[sc_base + is + 4]);
                    }
                } else {
                    if (sub_pair == 0) {
                        q0 = int((blk0.ql[ql_idx + 32 + l] & 0x0F) | (((blk0.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        q1 = int((blk1.ql[ql_idx + 32 + l] & 0x0F) | (((blk1.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        sc0 = float(blk0.scales[sc_base + is + 2]);
                        sc1 = float(blk1.scales[sc_base + is + 2]);
                    } else {
                        q0 = int((blk0.ql[ql_idx + 32 + l] >> 4) | (((blk0.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        q1 = int((blk1.ql[ql_idx + 32 + l] >> 4) | (((blk1.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        sc0 = float(blk0.scales[sc_base + is + 6]);
                        sc1 = float(blk1.scales[sc_base + is + 6]);
                    }
                }
                tg_A0[r * PB_BK + c] = float(blk0.d) * sc0 * float(q0);
                tg_A1[r * PB_BK + c] = float(blk1.d) * sc1 * float(q1);
            } else {
                tg_A0[r * PB_BK + c] = 0.0f;
                tg_A1[r * PB_BK + c] = 0.0f;
            }
        }

        for (uint i = tid; i < PB_BN * PB_BK; i += PB_TG) {
            uint r = i / PB_BK;
            uint c = i % PB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * PB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < PB_BK / 8; kk++) {
            simdgroup_float8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * PB_BK + kk * 8], PB_BK);

            simdgroup_float8x8 a00, a01, a02, a03;
            simdgroup_float8x8 a10, a11, a12, a13;
            simdgroup_load(a00, &tg_A0[0  * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a01, &tg_A0[8  * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a02, &tg_A0[16 * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a03, &tg_A0[24 * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a10, &tg_A1[0  * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a11, &tg_A1[8  * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a12, &tg_A1[16 * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);
            simdgroup_load(a13, &tg_A1[24 * PB_BK + kk * 8], PB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc00, b_frag, a00, acc00);
            simdgroup_multiply_accumulate(acc01, b_frag, a01, acc01);
            simdgroup_multiply_accumulate(acc02, b_frag, a02, acc02);
            simdgroup_multiply_accumulate(acc03, b_frag, a03, acc03);
            simdgroup_multiply_accumulate(acc10, b_frag, a10, acc10);
            simdgroup_multiply_accumulate(acc11, b_frag, a11, acc11);
            simdgroup_multiply_accumulate(acc12, b_frag, a12, acc12);
            simdgroup_multiply_accumulate(acc13, b_frag, a13, acc13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + PB_BN <= N && tile_m + PB_BM <= M) {
        device float* c0 = C0 + (tile_n + simd_id * 8) * M + tile_m;
        device float* c1 = C1 + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc00, c0 + 0,  M);
        simdgroup_store(acc01, c0 + 8,  M);
        simdgroup_store(acc02, c0 + 16, M);
        simdgroup_store(acc03, c0 + 24, M);
        simdgroup_store(acc10, c1 + 0,  M);
        simdgroup_store(acc11, c1 + 8,  M);
        simdgroup_store(acc12, c1 + 16, M);
        simdgroup_store(acc13, c1 + 24, M);
        return;
    }

    threadgroup float out_tile[PB_BN * PB_BM];
    simdgroup_store(acc00, &out_tile[simd_id * 8 * PB_BM + 0],  PB_BM);
    simdgroup_store(acc01, &out_tile[simd_id * 8 * PB_BM + 8],  PB_BM);
    simdgroup_store(acc02, &out_tile[simd_id * 8 * PB_BM + 16], PB_BM);
    simdgroup_store(acc03, &out_tile[simd_id * 8 * PB_BM + 24], PB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < PB_BN * PB_BM; i += PB_TG) {
        uint r = i / PB_BM;
        uint c = i % PB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C0[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_store(acc10, &out_tile[simd_id * 8 * PB_BM + 0],  PB_BM);
    simdgroup_store(acc11, &out_tile[simd_id * 8 * PB_BM + 8],  PB_BM);
    simdgroup_store(acc12, &out_tile[simd_id * 8 * PB_BM + 16], PB_BM);
    simdgroup_store(acc13, &out_tile[simd_id * 8 * PB_BM + 24], PB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < PB_BN * PB_BM; i += PB_TG) {
        uint r = i / PB_BM;
        uint c = i % PB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C1[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
}

// f16-input pair kernels tuned for BN=64.
// Reuses one B tile for gate+up projections and writes f32 outputs.
constant uint P16_BM = 32;
constant uint P16_BN = 64;
constant uint P16_BK = 64;
constant uint P16_TG = 256;

kernel void dequant_batch_pair_q4_k_f16in(
    device const Q4_K_Block* A0 [[buffer(0)]],
    device const Q4_K_Block* A1 [[buffer(1)]],
    device const half* B        [[buffer(2)]],
    device float* C0            [[buffer(3)]],
    device float* C1            [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& K            [[buffer(7)]],
    uint2 group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * P16_BM;
    uint tile_n = group_id.y * P16_BN;

    threadgroup float tg_A0[P16_BM * P16_BK];
    threadgroup float tg_A1[P16_BM * P16_BK];
    threadgroup half tg_B[P16_BN * P16_BK];

    simdgroup_float8x8 acc00, acc01, acc02, acc03;
    simdgroup_float8x8 acc10, acc11, acc12, acc13;
    acc00 = simdgroup_float8x8(0);
    acc01 = simdgroup_float8x8(0);
    acc02 = simdgroup_float8x8(0);
    acc03 = simdgroup_float8x8(0);
    acc10 = simdgroup_float8x8(0);
    acc11 = simdgroup_float8x8(0);
    acc12 = simdgroup_float8x8(0);
    acc13 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += P16_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / P16_BK;

        for (uint i = tid; i < P16_BM * P16_BK; i += P16_TG) {
            uint r = i / P16_BK;
            uint c = i % P16_BK;
            uint global_r = tile_m + r;

            if (global_r < M) {
                device const Q4_K_Block& blk0 = A0[global_r * blocks_per_row + block_idx];
                device const Q4_K_Block& blk1 = A1[global_r * blocks_per_row + block_idx];

                float2 sm0a = get_scale_min_q4k(pair * 2, blk0.scales);
                float2 sm0b = get_scale_min_q4k(pair * 2 + 1, blk0.scales);
                float2 sm1a = get_scale_min_q4k(pair * 2, blk1.scales);
                float2 sm1b = get_scale_min_q4k(pair * 2 + 1, blk1.scales);

                uchar byte0 = blk0.qs[pair * 32 + (c < 32 ? c : c - 32)];
                uchar byte1 = blk1.qs[pair * 32 + (c < 32 ? c : c - 32)];

                if (c < 32) {
                    tg_A0[r * P16_BK + c] = float(blk0.d) * sm0a.x * float(byte0 & 0x0F) - float(blk0.dmin) * sm0a.y;
                    tg_A1[r * P16_BK + c] = float(blk1.d) * sm1a.x * float(byte1 & 0x0F) - float(blk1.dmin) * sm1a.y;
                } else {
                    tg_A0[r * P16_BK + c] = float(blk0.d) * sm0b.x * float(byte0 >> 4) - float(blk0.dmin) * sm0b.y;
                    tg_A1[r * P16_BK + c] = float(blk1.d) * sm1b.x * float(byte1 >> 4) - float(blk1.dmin) * sm1b.y;
                }
            } else {
                tg_A0[r * P16_BK + c] = 0.0f;
                tg_A1[r * P16_BK + c] = 0.0f;
            }
        }

        for (uint i = tid; i < P16_BN * P16_BK; i += P16_TG) {
            uint r = i / P16_BK;
            uint c = i % P16_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * P16_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < P16_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * P16_BK + kk * 8], P16_BK);

            simdgroup_float8x8 a00, a01, a02, a03;
            simdgroup_float8x8 a10, a11, a12, a13;
            simdgroup_load(a00, &tg_A0[0  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a01, &tg_A0[8  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a02, &tg_A0[16 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a03, &tg_A0[24 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a10, &tg_A1[0  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a11, &tg_A1[8  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a12, &tg_A1[16 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a13, &tg_A1[24 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc00, b_frag, a00, acc00);
            simdgroup_multiply_accumulate(acc01, b_frag, a01, acc01);
            simdgroup_multiply_accumulate(acc02, b_frag, a02, acc02);
            simdgroup_multiply_accumulate(acc03, b_frag, a03, acc03);
            simdgroup_multiply_accumulate(acc10, b_frag, a10, acc10);
            simdgroup_multiply_accumulate(acc11, b_frag, a11, acc11);
            simdgroup_multiply_accumulate(acc12, b_frag, a12, acc12);
            simdgroup_multiply_accumulate(acc13, b_frag, a13, acc13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + P16_BN <= N && tile_m + P16_BM <= M) {
        device float* c0 = C0 + (tile_n + simd_id * 8) * M + tile_m;
        device float* c1 = C1 + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc00, c0 + 0,  M);
        simdgroup_store(acc01, c0 + 8,  M);
        simdgroup_store(acc02, c0 + 16, M);
        simdgroup_store(acc03, c0 + 24, M);
        simdgroup_store(acc10, c1 + 0,  M);
        simdgroup_store(acc11, c1 + 8,  M);
        simdgroup_store(acc12, c1 + 16, M);
        simdgroup_store(acc13, c1 + 24, M);
        return;
    }

    threadgroup float out_tile[P16_BN * P16_BM];
    simdgroup_store(acc00, &out_tile[simd_id * 8 * P16_BM + 0],  P16_BM);
    simdgroup_store(acc01, &out_tile[simd_id * 8 * P16_BM + 8],  P16_BM);
    simdgroup_store(acc02, &out_tile[simd_id * 8 * P16_BM + 16], P16_BM);
    simdgroup_store(acc03, &out_tile[simd_id * 8 * P16_BM + 24], P16_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < P16_BN * P16_BM; i += P16_TG) {
        uint r = i / P16_BM;
        uint c = i % P16_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C0[gn * M + gm] = out_tile[r * P16_BM + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_store(acc10, &out_tile[simd_id * 8 * P16_BM + 0],  P16_BM);
    simdgroup_store(acc11, &out_tile[simd_id * 8 * P16_BM + 8],  P16_BM);
    simdgroup_store(acc12, &out_tile[simd_id * 8 * P16_BM + 16], P16_BM);
    simdgroup_store(acc13, &out_tile[simd_id * 8 * P16_BM + 24], P16_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < P16_BN * P16_BM; i += P16_TG) {
        uint r = i / P16_BM;
        uint c = i % P16_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C1[gn * M + gm] = out_tile[r * P16_BM + c];
        }
    }
}

kernel void dequant_batch_pair_q6_k_f16in(
    device const Q6_K_Block* A0 [[buffer(0)]],
    device const Q6_K_Block* A1 [[buffer(1)]],
    device const half* B        [[buffer(2)]],
    device float* C0            [[buffer(3)]],
    device float* C1            [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& K            [[buffer(7)]],
    uint2 group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * P16_BM;
    uint tile_n = group_id.y * P16_BN;

    threadgroup float tg_A0[P16_BM * P16_BK];
    threadgroup float tg_A1[P16_BM * P16_BK];
    threadgroup half tg_B[P16_BN * P16_BK];

    simdgroup_float8x8 acc00, acc01, acc02, acc03;
    simdgroup_float8x8 acc10, acc11, acc12, acc13;
    acc00 = simdgroup_float8x8(0);
    acc01 = simdgroup_float8x8(0);
    acc02 = simdgroup_float8x8(0);
    acc03 = simdgroup_float8x8(0);
    acc10 = simdgroup_float8x8(0);
    acc11 = simdgroup_float8x8(0);
    acc12 = simdgroup_float8x8(0);
    acc13 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += P16_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < P16_BM * P16_BK; i += P16_TG) {
            uint r = i / P16_BK;
            uint c = i % P16_BK;
            uint global_r = tile_m + r;

            if (global_r < M) {
                device const Q6_K_Block& blk0 = A0[global_r * blocks_per_row + block_idx];
                device const Q6_K_Block& blk1 = A1[global_r * blocks_per_row + block_idx];

                uint ql_base = group * 64;
                uint qh_base = group * 32;
                uint sc_base = group * 8;
                uint sub = c / 32;
                uint l = c % 32;
                uint is = l / 16;
                uint ql_idx = ql_base + sub_pair * 32;
                uint qh_idx = qh_base;

                int q0, q1;
                float sc0, sc1;
                if (sub == 0) {
                    if (sub_pair == 0) {
                        q0 = int((blk0.ql[ql_idx + l] & 0x0F) | ((blk0.qh[qh_idx + l] & 3) << 4)) - 32;
                        q1 = int((blk1.ql[ql_idx + l] & 0x0F) | ((blk1.qh[qh_idx + l] & 3) << 4)) - 32;
                        sc0 = float(blk0.scales[sc_base + is]);
                        sc1 = float(blk1.scales[sc_base + is]);
                    } else {
                        q0 = int((blk0.ql[ql_idx + l] >> 4) | (((blk0.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        q1 = int((blk1.ql[ql_idx + l] >> 4) | (((blk1.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        sc0 = float(blk0.scales[sc_base + is + 4]);
                        sc1 = float(blk1.scales[sc_base + is + 4]);
                    }
                } else {
                    if (sub_pair == 0) {
                        q0 = int((blk0.ql[ql_idx + 32 + l] & 0x0F) | (((blk0.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        q1 = int((blk1.ql[ql_idx + 32 + l] & 0x0F) | (((blk1.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        sc0 = float(blk0.scales[sc_base + is + 2]);
                        sc1 = float(blk1.scales[sc_base + is + 2]);
                    } else {
                        q0 = int((blk0.ql[ql_idx + 32 + l] >> 4) | (((blk0.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        q1 = int((blk1.ql[ql_idx + 32 + l] >> 4) | (((blk1.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        sc0 = float(blk0.scales[sc_base + is + 6]);
                        sc1 = float(blk1.scales[sc_base + is + 6]);
                    }
                }
                tg_A0[r * P16_BK + c] = float(blk0.d) * sc0 * float(q0);
                tg_A1[r * P16_BK + c] = float(blk1.d) * sc1 * float(q1);
            } else {
                tg_A0[r * P16_BK + c] = 0.0f;
                tg_A1[r * P16_BK + c] = 0.0f;
            }
        }

        for (uint i = tid; i < P16_BN * P16_BK; i += P16_TG) {
            uint r = i / P16_BK;
            uint c = i % P16_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * P16_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < P16_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * P16_BK + kk * 8], P16_BK);

            simdgroup_float8x8 a00, a01, a02, a03;
            simdgroup_float8x8 a10, a11, a12, a13;
            simdgroup_load(a00, &tg_A0[0  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a01, &tg_A0[8  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a02, &tg_A0[16 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a03, &tg_A0[24 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a10, &tg_A1[0  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a11, &tg_A1[8  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a12, &tg_A1[16 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a13, &tg_A1[24 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc00, b_frag, a00, acc00);
            simdgroup_multiply_accumulate(acc01, b_frag, a01, acc01);
            simdgroup_multiply_accumulate(acc02, b_frag, a02, acc02);
            simdgroup_multiply_accumulate(acc03, b_frag, a03, acc03);
            simdgroup_multiply_accumulate(acc10, b_frag, a10, acc10);
            simdgroup_multiply_accumulate(acc11, b_frag, a11, acc11);
            simdgroup_multiply_accumulate(acc12, b_frag, a12, acc12);
            simdgroup_multiply_accumulate(acc13, b_frag, a13, acc13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + P16_BN <= N && tile_m + P16_BM <= M) {
        device float* c0 = C0 + (tile_n + simd_id * 8) * M + tile_m;
        device float* c1 = C1 + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc00, c0 + 0,  M);
        simdgroup_store(acc01, c0 + 8,  M);
        simdgroup_store(acc02, c0 + 16, M);
        simdgroup_store(acc03, c0 + 24, M);
        simdgroup_store(acc10, c1 + 0,  M);
        simdgroup_store(acc11, c1 + 8,  M);
        simdgroup_store(acc12, c1 + 16, M);
        simdgroup_store(acc13, c1 + 24, M);
        return;
    }

    threadgroup float out_tile[P16_BN * P16_BM];
    simdgroup_store(acc00, &out_tile[simd_id * 8 * P16_BM + 0],  P16_BM);
    simdgroup_store(acc01, &out_tile[simd_id * 8 * P16_BM + 8],  P16_BM);
    simdgroup_store(acc02, &out_tile[simd_id * 8 * P16_BM + 16], P16_BM);
    simdgroup_store(acc03, &out_tile[simd_id * 8 * P16_BM + 24], P16_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < P16_BN * P16_BM; i += P16_TG) {
        uint r = i / P16_BM;
        uint c = i % P16_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C0[gn * M + gm] = out_tile[r * P16_BM + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_store(acc10, &out_tile[simd_id * 8 * P16_BM + 0],  P16_BM);
    simdgroup_store(acc11, &out_tile[simd_id * 8 * P16_BM + 8],  P16_BM);
    simdgroup_store(acc12, &out_tile[simd_id * 8 * P16_BM + 16], P16_BM);
    simdgroup_store(acc13, &out_tile[simd_id * 8 * P16_BM + 24], P16_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < P16_BN * P16_BM; i += P16_TG) {
        uint r = i / P16_BM;
        uint c = i % P16_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C1[gn * M + gm] = out_tile[r * P16_BM + c];
        }
    }
}

kernel void dequant_batch_pair_q8_0_f16in(
    device const Q8_0_Block* A0 [[buffer(0)]],
    device const Q8_0_Block* A1 [[buffer(1)]],
    device const half* B        [[buffer(2)]],
    device float* C0            [[buffer(3)]],
    device float* C1            [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& K            [[buffer(7)]],
    uint2 group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * P16_BM;
    uint tile_n = group_id.y * P16_BN;

    threadgroup half tg_A0[P16_BM * P16_BK];
    threadgroup half tg_A1[P16_BM * P16_BK];
    threadgroup half tg_B[P16_BN * P16_BK];

    simdgroup_float8x8 acc00, acc01, acc02, acc03;
    simdgroup_float8x8 acc10, acc11, acc12, acc13;
    acc00 = simdgroup_float8x8(0);
    acc01 = simdgroup_float8x8(0);
    acc02 = simdgroup_float8x8(0);
    acc03 = simdgroup_float8x8(0);
    acc10 = simdgroup_float8x8(0);
    acc11 = simdgroup_float8x8(0);
    acc12 = simdgroup_float8x8(0);
    acc13 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += P16_BK) {
        for (uint i = tid; i < P16_BM * P16_BK; i += P16_TG) {
            uint r = i / P16_BK;
            uint c = i % P16_BK;
            uint global_r = tile_m + r;
            uint global_k = kt + c;

            if (global_r < M && global_k < K) {
                uint block_idx = global_k / Q8_0_BLOCK_VALUES;
                uint in_block = global_k % Q8_0_BLOCK_VALUES;

                device const Q8_0_Block& blk0 = A0[global_r * blocks_per_row + block_idx];
                device const Q8_0_Block& blk1 = A1[global_r * blocks_per_row + block_idx];
                tg_A0[r * P16_BK + c] = half(float(blk0.d) * float(int(blk0.qs[in_block])));
                tg_A1[r * P16_BK + c] = half(float(blk1.d) * float(int(blk1.qs[in_block])));
            } else {
                tg_A0[r * P16_BK + c] = half(0.0f);
                tg_A1[r * P16_BK + c] = half(0.0f);
            }
        }

        for (uint i = tid; i < P16_BN * P16_BK; i += P16_TG) {
            uint r = i / P16_BK;
            uint c = i % P16_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * P16_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < P16_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * P16_BK + kk * 8], P16_BK);

            simdgroup_half8x8 a00, a01, a02, a03;
            simdgroup_half8x8 a10, a11, a12, a13;
            simdgroup_load(a00, &tg_A0[0  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a01, &tg_A0[8  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a02, &tg_A0[16 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a03, &tg_A0[24 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a10, &tg_A1[0  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a11, &tg_A1[8  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a12, &tg_A1[16 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a13, &tg_A1[24 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc00, b_frag, a00, acc00);
            simdgroup_multiply_accumulate(acc01, b_frag, a01, acc01);
            simdgroup_multiply_accumulate(acc02, b_frag, a02, acc02);
            simdgroup_multiply_accumulate(acc03, b_frag, a03, acc03);
            simdgroup_multiply_accumulate(acc10, b_frag, a10, acc10);
            simdgroup_multiply_accumulate(acc11, b_frag, a11, acc11);
            simdgroup_multiply_accumulate(acc12, b_frag, a12, acc12);
            simdgroup_multiply_accumulate(acc13, b_frag, a13, acc13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + P16_BN <= N && tile_m + P16_BM <= M) {
        device float* c0 = C0 + (tile_n + simd_id * 8) * M + tile_m;
        device float* c1 = C1 + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc00, c0 + 0,  M);
        simdgroup_store(acc01, c0 + 8,  M);
        simdgroup_store(acc02, c0 + 16, M);
        simdgroup_store(acc03, c0 + 24, M);
        simdgroup_store(acc10, c1 + 0,  M);
        simdgroup_store(acc11, c1 + 8,  M);
        simdgroup_store(acc12, c1 + 16, M);
        simdgroup_store(acc13, c1 + 24, M);
        return;
    }

    threadgroup float out_tile[P16_BN * P16_BM];
    simdgroup_store(acc00, &out_tile[simd_id * 8 * P16_BM + 0],  P16_BM);
    simdgroup_store(acc01, &out_tile[simd_id * 8 * P16_BM + 8],  P16_BM);
    simdgroup_store(acc02, &out_tile[simd_id * 8 * P16_BM + 16], P16_BM);
    simdgroup_store(acc03, &out_tile[simd_id * 8 * P16_BM + 24], P16_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < P16_BN * P16_BM; i += P16_TG) {
        uint r = i / P16_BM;
        uint c = i % P16_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C0[gn * M + gm] = out_tile[r * P16_BM + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_store(acc10, &out_tile[simd_id * 8 * P16_BM + 0],  P16_BM);
    simdgroup_store(acc11, &out_tile[simd_id * 8 * P16_BM + 8],  P16_BM);
    simdgroup_store(acc12, &out_tile[simd_id * 8 * P16_BM + 16], P16_BM);
    simdgroup_store(acc13, &out_tile[simd_id * 8 * P16_BM + 24], P16_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < P16_BN * P16_BM; i += P16_TG) {
        uint r = i / P16_BM;
        uint c = i % P16_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C1[gn * M + gm] = out_tile[r * P16_BM + c];
        }
    }
}

// Full-tile fast path for Q8_0 f16-input pair kernel.
// Preconditions enforced by dispatch:
// - M is divisible by P16_BM (32)
// - N is divisible by P16_BN (64)
// - K is divisible by P16_BK (64)
kernel void dequant_batch_pair_q8_0_f16in_full(
    device const Q8_0_Block* A0 [[buffer(0)]],
    device const Q8_0_Block* A1 [[buffer(1)]],
    device const half* B        [[buffer(2)]],
    device float* C0            [[buffer(3)]],
    device float* C1            [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& K            [[buffer(7)]],
    uint2 group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * P16_BM;
    uint tile_n = group_id.y * P16_BN;

    threadgroup half tg_A0[P16_BM * P16_BK];
    threadgroup half tg_A1[P16_BM * P16_BK];
    threadgroup half tg_B[P16_BN * P16_BK];

    simdgroup_float8x8 acc00, acc01, acc02, acc03;
    simdgroup_float8x8 acc10, acc11, acc12, acc13;
    acc00 = simdgroup_float8x8(0);
    acc01 = simdgroup_float8x8(0);
    acc02 = simdgroup_float8x8(0);
    acc03 = simdgroup_float8x8(0);
    acc10 = simdgroup_float8x8(0);
    acc11 = simdgroup_float8x8(0);
    acc12 = simdgroup_float8x8(0);
    acc13 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += P16_BK) {
        for (uint i = tid; i < P16_BM * P16_BK; i += P16_TG) {
            uint r = i / P16_BK;
            uint c = i % P16_BK;
            uint global_r = tile_m + r;
            uint global_k = kt + c;
            uint block_idx = global_k / Q8_0_BLOCK_VALUES;
            uint in_block = global_k % Q8_0_BLOCK_VALUES;

            device const Q8_0_Block& blk0 = A0[global_r * blocks_per_row + block_idx];
            device const Q8_0_Block& blk1 = A1[global_r * blocks_per_row + block_idx];
            tg_A0[r * P16_BK + c] = half(float(blk0.d) * float(int(blk0.qs[in_block])));
            tg_A1[r * P16_BK + c] = half(float(blk1.d) * float(int(blk1.qs[in_block])));
        }

        for (uint i = tid; i < P16_BN * P16_BK; i += P16_TG) {
            uint r = i / P16_BK;
            uint c = i % P16_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * P16_BK + c] = B[gn * K + gk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < P16_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * P16_BK + kk * 8], P16_BK);

            simdgroup_half8x8 a00, a01, a02, a03;
            simdgroup_half8x8 a10, a11, a12, a13;
            simdgroup_load(a00, &tg_A0[0  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a01, &tg_A0[8  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a02, &tg_A0[16 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a03, &tg_A0[24 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a10, &tg_A1[0  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a11, &tg_A1[8  * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a12, &tg_A1[16 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);
            simdgroup_load(a13, &tg_A1[24 * P16_BK + kk * 8], P16_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc00, b_frag, a00, acc00);
            simdgroup_multiply_accumulate(acc01, b_frag, a01, acc01);
            simdgroup_multiply_accumulate(acc02, b_frag, a02, acc02);
            simdgroup_multiply_accumulate(acc03, b_frag, a03, acc03);
            simdgroup_multiply_accumulate(acc10, b_frag, a10, acc10);
            simdgroup_multiply_accumulate(acc11, b_frag, a11, acc11);
            simdgroup_multiply_accumulate(acc12, b_frag, a12, acc12);
            simdgroup_multiply_accumulate(acc13, b_frag, a13, acc13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* c0 = C0 + (tile_n + simd_id * 8) * M + tile_m;
    device float* c1 = C1 + (tile_n + simd_id * 8) * M + tile_m;
    simdgroup_store(acc00, c0 + 0,  M);
    simdgroup_store(acc01, c0 + 8,  M);
    simdgroup_store(acc02, c0 + 16, M);
    simdgroup_store(acc03, c0 + 24, M);
    simdgroup_store(acc10, c1 + 0,  M);
    simdgroup_store(acc11, c1 + 8,  M);
    simdgroup_store(acc12, c1 + 16, M);
    simdgroup_store(acc13, c1 + 24, M);
}

// ── Batch Dequant+Matmul (small-N variant) ────────────────────────────
//
// Tuned for smaller prefill batches with lower threadgroup memory pressure.
// Uses BN=32 and TG=128.

kernel void dequant_batch_q4_k_small(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * SB_BM;
    uint tile_n = group_id.y * SB_BN;

    threadgroup float tg_A[SB_BM * SB_BK];
    threadgroup float tg_B[SB_BN * SB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += SB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / SB_BK;

        for (uint i = tid; i < SB_BM * SB_BK; i += SB_TG) {
            uint r = i / SB_BK;
            uint c = i % SB_BK;
            uint global_r = tile_m + r;

            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                uchar byte = blk.qs[pair * 32 + (c < 32 ? c : c - 32)];
                if (c < 32) {
                    tg_A[r * SB_BK + c] = half(d * sm1.x * float(byte & 0x0F) - dmin * sm1.y);
                } else {
                    tg_A[r * SB_BK + c] = half(d * sm2.x * float(byte >> 4) - dmin * sm2.y);
                }
            } else {
                tg_A[r * SB_BK + c] = half(0.0f);
            }
        }

        for (uint i = tid; i < SB_BN * SB_BK; i += SB_TG) {
            uint r = i / SB_BK;
            uint c = i % SB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * SB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SB_BK / 8; kk++) {
            simdgroup_float8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * SB_BK + kk * 8], SB_BK);

            simdgroup_float8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + SB_BN <= N && tile_m + SB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc0, c_base + 0,  M);
        simdgroup_store(acc1, c_base + 8,  M);
        simdgroup_store(acc2, c_base + 16, M);
        simdgroup_store(acc3, c_base + 24, M);
        return;
    }

    threadgroup float out_tile[SB_BN * SB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * SB_BM + 0],  SB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * SB_BM + 8],  SB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * SB_BM + 16], SB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * SB_BM + 24], SB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < SB_BN * SB_BM; i += SB_TG) {
        uint r = i / SB_BM;
        uint c = i % SB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = out_tile[r * SB_BM + c];
        }
    }
}

kernel void dequant_batch_q6_k_small(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * SB_BM;
    uint tile_n = group_id.y * SB_BN;

    threadgroup float tg_A[SB_BM * SB_BK];
    threadgroup float tg_B[SB_BN * SB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += SB_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < SB_BM * SB_BK; i += SB_TG) {
            uint r = i / SB_BK;
            uint c = i % SB_BK;
            uint global_r = tile_m + r;

            if (global_r < M) {
                device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);
                uint ql_base = group * 64;
                uint qh_base = group * 32;
                uint sc_base = group * 8;
                uint sub = c / 32;
                uint l = c % 32;
                uint is = l / 16;
                uint ql_idx = ql_base + sub_pair * 32;
                uint qh_idx = qh_base;

                int q;
                float sc;
                if (sub == 0) {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is]);
                    } else {
                        q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 4]);
                    }
                } else {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 2]);
                    } else {
                        q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 6]);
                    }
                }
                tg_A[r * SB_BK + c] = half(d * sc * float(q));
            } else {
                tg_A[r * SB_BK + c] = half(0.0f);
            }
        }

        for (uint i = tid; i < SB_BN * SB_BK; i += SB_TG) {
            uint r = i / SB_BK;
            uint c = i % SB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * SB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SB_BK / 8; kk++) {
            simdgroup_float8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * SB_BK + kk * 8], SB_BK);

            simdgroup_float8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + SB_BN <= N && tile_m + SB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * M + tile_m;
        simdgroup_store(acc0, c_base + 0,  M);
        simdgroup_store(acc1, c_base + 8,  M);
        simdgroup_store(acc2, c_base + 16, M);
        simdgroup_store(acc3, c_base + 24, M);
        return;
    }

    threadgroup float out_tile[SB_BN * SB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * SB_BM + 0],  SB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * SB_BM + 8],  SB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * SB_BM + 16], SB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * SB_BM + 24], SB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < SB_BN * SB_BM; i += SB_TG) {
        uint r = i / SB_BM;
        uint c = i % SB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * M + gm] = out_tile[r * SB_BM + c];
        }
    }
}

// ── Batch Dequant+Matmul (small-N, f16 input) ────────────────────────
//
// Same tile as small kernels (BN=32, TG=128), but with f16 batch input.

kernel void dequant_batch_q4_k_f16in_small(
    device const Q4_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * SB_BM;
    uint tile_n = group_id.y * SB_BN;

    threadgroup half tg_A[SB_BM * SB_BK];
    threadgroup half tg_B[SB_BN * SB_BK];
    threadgroup half row_dsc1[SB_BM];
    threadgroup half row_dmin1[SB_BM];
    threadgroup half row_dsc2[SB_BM];
    threadgroup half row_dmin2[SB_BM];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += SB_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / SB_BK;

        // Phase 1: precompute d*scale and dmin*min.
        if (tid < SB_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d    = float(blk.d);
                float dmin = float(blk.dmin);
                float2 sm1 = get_scale_min_q4k(pair * 2, blk.scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, blk.scales);
                row_dsc1[tid]  = half(d * sm1.x);
                row_dmin1[tid] = half(dmin * sm1.y);
                row_dsc2[tid]  = half(d * sm2.x);
                row_dmin2[tid] = half(dmin * sm2.y);
            } else {
                row_dsc1[tid]  = half(0.0f);
                row_dmin1[tid] = half(0.0f);
                row_dsc2[tid]  = half(0.0f);
                row_dmin2[tid] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble extraction (no branch).
        for (uint i = tid; i < SB_BM * (SB_BK / 2); i += SB_TG) {
            uint r = i / (SB_BK / 2);
            uint b = i % (SB_BK / 2);
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q4_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar byte = blk.qs[pair * 32 + b];
                tg_A[r * SB_BK + b]      = half(float(row_dsc1[r]) * float(byte & 0x0F) - float(row_dmin1[r]));
                tg_A[r * SB_BK + b + 32] = half(float(row_dsc2[r]) * float(byte >> 4)   - float(row_dmin2[r]));
            } else {
                tg_A[r * SB_BK + b]      = half(0.0f);
                tg_A[r * SB_BK + b + 32] = half(0.0f);
            }
        }

        for (uint i = tid; i < SB_BN * SB_BK; i += SB_TG) {
            uint r = i / SB_BK;
            uint c = i % SB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * SB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * SB_BK + kk * 8], SB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + SB_BN <= N && tile_m + SB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
        simdgroup_store(acc0, c_base + 0,  C_STRIDE);
        simdgroup_store(acc1, c_base + 8,  C_STRIDE);
        simdgroup_store(acc2, c_base + 16, C_STRIDE);
        simdgroup_store(acc3, c_base + 24, C_STRIDE);
        return;
    }

    threadgroup float out_tile[SB_BN * SB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * SB_BM + 0],  SB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * SB_BM + 8],  SB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * SB_BM + 16], SB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * SB_BM + 24], SB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < SB_BN * SB_BM; i += SB_TG) {
        uint r = i / SB_BM;
        uint c = i % SB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * C_STRIDE + gm] = out_tile[r * SB_BM + c];
        }
    }
}

kernel void dequant_batch_q6_k_f16in_small(
    device const Q6_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],   // output cols for this dispatch
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],   // destination row stride
    uint2 group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  simd_id              [[simdgroup_index_in_threadgroup]],
    uint  simd_lane            [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * SB_BM;
    uint tile_n = group_id.y * SB_BN;

    threadgroup half tg_A[SB_BM * SB_BK];
    threadgroup half tg_B[SB_BN * SB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += SB_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        for (uint i = tid; i < SB_BM * SB_BK; i += SB_TG) {
            uint r = i / SB_BK;
            uint c = i % SB_BK;
            uint global_r = tile_m + r;

            if (global_r < M) {
                device const Q6_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);
                uint ql_base = group * 64;
                uint qh_base = group * 32;
                uint sc_base = group * 8;
                uint sub = c / 32;
                uint l = c % 32;
                uint is = l / 16;
                uint ql_idx = ql_base + sub_pair * 32;
                uint qh_idx = qh_base;

                int q;
                float sc;
                if (sub == 0) {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + l] & 0x0F) | ((blk.qh[qh_idx + l] & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is]);
                    } else {
                        q = int((blk.ql[ql_idx + l] >> 4) | (((blk.qh[qh_idx + l] >> 4) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 4]);
                    }
                } else {
                    if (sub_pair == 0) {
                        q = int((blk.ql[ql_idx + 32 + l] & 0x0F) | (((blk.qh[qh_idx + l] >> 2) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 2]);
                    } else {
                        q = int((blk.ql[ql_idx + 32 + l] >> 4) | (((blk.qh[qh_idx + l] >> 6) & 3) << 4)) - 32;
                        sc = float(blk.scales[sc_base + is + 6]);
                    }
                }
                tg_A[r * SB_BK + c] = half(d * sc * float(q));
            } else {
                tg_A[r * SB_BK + c] = half(0.0f);
            }
        }

        for (uint i = tid; i < SB_BN * SB_BK; i += SB_TG) {
            uint r = i / SB_BK;
            uint c = i % SB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * SB_BK + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SB_BK / 8; kk++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * SB_BK + kk * 8], SB_BK);

            simdgroup_half8x8 a0, a1, a2, a3;
            simdgroup_load(a0, &tg_A[0  * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a1, &tg_A[8  * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a2, &tg_A[16 * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);
            simdgroup_load(a3, &tg_A[24 * SB_BK + kk * 8], SB_BK, ulong2(0,0), true);

            simdgroup_multiply_accumulate(acc0, b_frag, a0, acc0);
            simdgroup_multiply_accumulate(acc1, b_frag, a1, acc1);
            simdgroup_multiply_accumulate(acc2, b_frag, a2, acc2);
            simdgroup_multiply_accumulate(acc3, b_frag, a3, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tile_n + SB_BN <= N && tile_m + SB_BM <= M) {
        device float* c_base = C + (tile_n + simd_id * 8) * C_STRIDE + tile_m;
        simdgroup_store(acc0, c_base + 0,  C_STRIDE);
        simdgroup_store(acc1, c_base + 8,  C_STRIDE);
        simdgroup_store(acc2, c_base + 16, C_STRIDE);
        simdgroup_store(acc3, c_base + 24, C_STRIDE);
        return;
    }

    threadgroup float out_tile[SB_BN * SB_BM];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * SB_BM + 0],  SB_BM);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * SB_BM + 8],  SB_BM);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * SB_BM + 16], SB_BM);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * SB_BM + 24], SB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < SB_BN * SB_BM; i += SB_TG) {
        uint r = i / SB_BM;
        uint c = i % SB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C[gn * C_STRIDE + gm] = out_tile[r * SB_BM + c];
        }
    }
}

// ── Fused Dequant + Matrix-Vector Multiply ─────────────────────────────
//
// y[row] = dot(dequant(A_quantized[row, :]), x)
//
// Each threadgroup computes one output row. Threads cooperate via
// strided block processing + two-level SIMD reduction.
//
// Threadgroup size: 128 threads.
// Grid: M threadgroups × 1.

constant uint DEQUANT_MATVEC_TG = 128;

kernel void dequant_matvec_q4_0(
    device const Q4_0_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q4_0_BLOCK_VALUES;
    device const Q4_0_Block* a_row = A + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;

    // SIMD-cooperative: each SIMD group (32 threads) processes one block
    // Q4_0 block = 16 bytes of packed nibbles → 32 values
    // Threads 0-15 each handle 1 byte (2 values), threads 16-31 idle
    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        float d = float(a_row[b].d);
        uint base = b * Q4_0_BLOCK_VALUES;

        if (simd_lane < 16) {
            uchar byte = a_row[b].qs[simd_lane];
            sum += d * float(int(byte & 0x0F) - 8) * x[base + simd_lane];
            sum += d * float(int(byte >> 4) - 8)   * x[base + simd_lane + 16];
        }
    }

    // SIMD-level reduction (32 → 1 per SIMD group)
    sum = simd_sum(sum);

    // Cross-SIMD reduction via threadgroup memory
    threadgroup float simd_sums[num_simd_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) {
            y[row] = sum;
        }
    }
}

kernel void dequant_matvec_q8_0(
    device const Q8_0_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    device const Q8_0_Block* a_row = A + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
        d = simd_broadcast(d, 0);

        uint base = b * Q8_0_BLOCK_VALUES;
        int q = int(a_row[b].qs[simd_lane]);
        sum += d * float(q) * x[base + simd_lane];
    }

    // SIMD-level reduction (32 → 1 per SIMD group)
    sum = simd_sum(sum);

    // Cross-SIMD reduction via threadgroup memory
    threadgroup float simd_sums[num_simd_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) {
            y[row] = sum;
        }
    }
}

// R3: Q4_K decode matvec with register-based half x-caching.
//
// Each simdgroup pre-loads all 8 x-values it needs for a block into
// register-resident half variables (rx0..rx7) before the inner pair loop.
// Benefits vs original:
//   • x bandwidth halved (float → half, 2 bytes instead of 4)
//   • GPU can pipeline the 8 loads before scale computation starts
//   • Inner pair loop reads from registers (no device-memory x reads)
// No extra threadgroup barriers — original stride structure is preserved.
inline float q4_k_block_dot(
    device const Q4_K_Block* a_row,
    uint b,
    device const float* x,
    uint simd_lane
) {
    uint base = b * Q4_K_BLOCK_VALUES;

    half rx0 = half(x[base +   0 + simd_lane]);
    half rx1 = half(x[base +  32 + simd_lane]);
    half rx2 = half(x[base +  64 + simd_lane]);
    half rx3 = half(x[base +  96 + simd_lane]);
    half rx4 = half(x[base + 128 + simd_lane]);
    half rx5 = half(x[base + 160 + simd_lane]);
    half rx6 = half(x[base + 192 + simd_lane]);
    half rx7 = half(x[base + 224 + simd_lane]);

    float d    = (simd_lane == 0) ? float(a_row[b].d)    : 0.0f;
    float dmin = (simd_lane == 0) ? float(a_row[b].dmin) : 0.0f;
    d    = simd_broadcast(d,    0);
    dmin = simd_broadcast(dmin, 0);
    device const uchar* scales = a_row[b].scales;
    device const uchar* qs     = a_row[b].qs;

    float block_sum = 0.0f;
    for (uint pair = 0; pair < 4; pair++) {
        float sm1x = 0.0f, sm1y = 0.0f, sm2x = 0.0f, sm2y = 0.0f;
        if (simd_lane == 0) {
            float2 sm1 = get_scale_min_q4k(pair * 2,     scales);
            float2 sm2 = get_scale_min_q4k(pair * 2 + 1, scales);
            sm1x = sm1.x; sm1y = sm1.y;
            sm2x = sm2.x; sm2y = sm2.y;
        }
        sm1x = simd_broadcast(sm1x, 0); sm1y = simd_broadcast(sm1y, 0);
        sm2x = simd_broadcast(sm2x, 0); sm2y = simd_broadcast(sm2y, 0);

        float d1 = d * sm1x,  m1 = dmin * sm1y;
        float d2 = d * sm2x,  m2 = dmin * sm2y;

        uchar byte = qs[pair * 32 + simd_lane];
        half lo = (pair == 0) ? rx0 : (pair == 1) ? rx2 : (pair == 2) ? rx4 : rx6;
        half hi = (pair == 0) ? rx1 : (pair == 1) ? rx3 : (pair == 2) ? rx5 : rx7;
        block_sum += (d1 * float(byte & 0x0F) - m1) * float(lo);
        block_sum += (d2 * float(byte >> 4)   - m2) * float(hi);
    }

    return block_sum;
}

inline float q5_k_block_dot(
    device const Q5_K_Block* a_row,
    uint b,
    device const float* x,
    uint simd_lane
) {
    uint base = b * Q5_K_BLOCK_VALUES;

    half rx0 = half(x[base +   0 + simd_lane]);
    half rx1 = half(x[base +  32 + simd_lane]);
    half rx2 = half(x[base +  64 + simd_lane]);
    half rx3 = half(x[base +  96 + simd_lane]);
    half rx4 = half(x[base + 128 + simd_lane]);
    half rx5 = half(x[base + 160 + simd_lane]);
    half rx6 = half(x[base + 192 + simd_lane]);
    half rx7 = half(x[base + 224 + simd_lane]);

    float d    = (simd_lane == 0) ? float(a_row[b].d)    : 0.0f;
    float dmin = (simd_lane == 0) ? float(a_row[b].dmin) : 0.0f;
    d    = simd_broadcast(d, 0);
    dmin = simd_broadcast(dmin, 0);
    device const uchar* scales = a_row[b].scales;
    device const uchar* qh     = a_row[b].qh;
    device const uchar* qs     = a_row[b].qs;

    float block_sum = 0.0f;
    uchar high_bits = qh[simd_lane];
    for (uint pair = 0; pair < 4; pair++) {
        float2 sm1 = get_scale_min_q4k(pair * 2, scales);
        float2 sm2 = get_scale_min_q4k(pair * 2 + 1, scales);
        float d1 = d * sm1.x, m1 = dmin * sm1.y;
        float d2 = d * sm2.x, m2 = dmin * sm2.y;

        uchar byte = qs[pair * 32 + simd_lane];
        float lo_q = float(byte & 0x0F) + (((high_bits >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
        float hi_q = float(byte >> 4) + (((high_bits >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);
        half lo = (pair == 0) ? rx0 : (pair == 1) ? rx2 : (pair == 2) ? rx4 : rx6;
        half hi = (pair == 0) ? rx1 : (pair == 1) ? rx3 : (pair == 2) ? rx5 : rx7;
        block_sum += (d1 * lo_q - m1) * float(lo);
        block_sum += (d2 * hi_q - m2) * float(hi);
    }

    return block_sum;
}

// Baseline Q5_K decode matvec imported from llama.cpp's decode-only shape:
// 2 simdgroups per threadgroup, 1 output row per simdgroup (TG=64 total).
kernel void dequant_matvec_q5_k(
    device const Q5_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint first_row = tg_id * 2 + simd_id;
    if (first_row >= M) return;

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    device const Q5_K_Block* a_row = A + first_row * blocks_per_row;

    float sum = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        sum += q5_k_block_dot(a_row, b, x, simd_lane);
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        y[first_row] = sum;
    }
}

// Q5_K decode matvec with llama.cpp-style 4-way interleaved block traversal.
// This keeps TG=64 / 2 simdgroups / 1 row-per-simdgroup, but repartitions each
// simdgroup into 4 block streams so 8 lanes cooperate on one 256-value block.
kernel void dequant_matvec_q5_k_ilp4(
    device const Q5_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint first_row = tg_id * 2 + simd_id;
    if (first_row >= M) return;

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    device const Q5_K_Block* a_row = A + first_row * blocks_per_row;

    uint tid = simd_lane / 4;
    uint ix = simd_lane % 4;
    uint iq = tid / 4;
    uint ir = tid % 4;

    uint l0 = 8 * ir;
    uint q_offset = 32 * iq + l0;
    uint y_offset = 64 * iq + l0;

    uchar hm1 = uchar(1u << (2 * iq));
    uchar hm2 = uchar(hm1 << 1);
    uchar hm3 = uchar(hm1 << 4);
    uchar hm4 = uchar(hm2 << 4);

    float sum = 0.0f;
    for (uint b = ix; b < blocks_per_row; b += 4) {
        device const Q5_K_Block& blk = a_row[b];
        device const float* y1 = x + b * Q5_K_BLOCK_VALUES + y_offset;
        device const float* y2 = y1 + 128;

        float2 sm0 = get_scale_min_q4k(iq * 2, blk.scales);
        float2 sm1 = get_scale_min_q4k(iq * 2 + 1, blk.scales);
        float2 sm2 = get_scale_min_q4k(iq * 2 + 4, blk.scales);
        float2 sm3 = get_scale_min_q4k(iq * 2 + 5, blk.scales);

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        float sumy0 = 0.0f;
        float sumy1 = 0.0f;
        float sumy2 = 0.0f;
        float sumy3 = 0.0f;

        FOR_UNROLL (uint l = 0; l < 8; ++l) {
            float yl0 = y1[l];
            float yl1 = y1[l + 32];
            float yh0 = y2[l];
            float yh1 = y2[l + 32];

            uchar ql_lo_hi = blk.qs[q_offset + l];
            uchar qh_lo_hi = blk.qs[q_offset + 64 + l];
            uchar high_bits = blk.qh[l0 + l];

            acc0 += yl0
                * (float(ql_lo_hi & 0x0F) + ((high_bits & hm1) ? 16.0f : 0.0f));
            acc1 += yl1
                * (float(ql_lo_hi >> 4) + ((high_bits & hm2) ? 16.0f : 0.0f));
            acc2 += yh0
                * (float(qh_lo_hi & 0x0F) + ((high_bits & hm3) ? 16.0f : 0.0f));
            acc3 += yh1
                * (float(qh_lo_hi >> 4) + ((high_bits & hm4) ? 16.0f : 0.0f));

            sumy0 += yl0;
            sumy1 += yl1;
            sumy2 += yh0;
            sumy3 += yh1;
        }

        float d = float(blk.d);
        float dmin = float(blk.dmin);
        sum += d * (sm0.x * acc0 + sm1.x * acc1 + sm2.x * acc2 + sm3.x * acc3)
            - dmin * (sm0.y * sumy0 + sm1.y * sumy1 + sm2.y * sumy2 + sm3.y * sumy3);
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        y[first_row] = sum;
    }
}

constant uint Q5K_NR2_ROWS_PER_SG = 2;
constant uint Q5K_NR2_SG_PER_TG = 2;
constant uint Q5K_NR2_ROWS_PER_TG = Q5K_NR2_ROWS_PER_SG * Q5K_NR2_SG_PER_TG;

// Q5_K decode matvec with 2 rows per simdgroup and 2 simdgroups per
// threadgroup. This keeps the baseline TG=64 launch geometry but reuses the
// loaded x values across two rows inside each simdgroup.
kernel void dequant_matvec_q5_k_nr2(
    device const Q5_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    uint first_row = (tg_id * Q5K_NR2_SG_PER_TG + simd_id) * Q5K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    bool valid1 = (first_row + 1) < M;
    device const Q5_K_Block* row0 = A + first_row * blocks_per_row;
    device const Q5_K_Block* row1 = valid1 ? row0 + blocks_per_row : row0;

    float sumf[Q5K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint base = b * Q5_K_BLOCK_VALUES;

        half rx0 = half(x[base +   0 + simd_lane]);
        half rx1 = half(x[base +  32 + simd_lane]);
        half rx2 = half(x[base +  64 + simd_lane]);
        half rx3 = half(x[base +  96 + simd_lane]);
        half rx4 = half(x[base + 128 + simd_lane]);
        half rx5 = half(x[base + 160 + simd_lane]);
        half rx6 = half(x[base + 192 + simd_lane]);
        half rx7 = half(x[base + 224 + simd_lane]);

        for (ushort row = 0; row < Q5K_NR2_ROWS_PER_SG; ++row) {
            device const Q5_K_Block* row_ptr = (row == 0) ? row0 : row1;
            device const Q5_K_Block& blk = row_ptr[b];
            float d = (simd_lane == 0) ? float(blk.d) : 0.0f;
            float dmin = (simd_lane == 0) ? float(blk.dmin) : 0.0f;
            d = simd_broadcast(d, 0);
            dmin = simd_broadcast(dmin, 0);

            device const uchar* scales = blk.scales;
            device const uchar* qh = blk.qh;
            device const uchar* qs = blk.qs;

            uchar high_bits = qh[simd_lane];
            FOR_UNROLL (uint pair = 0; pair < 4; ++pair) {
                float2 sm1 = get_scale_min_q4k(pair * 2, scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, scales);
                float d1 = d * sm1.x, m1 = dmin * sm1.y;
                float d2 = d * sm2.x, m2 = dmin * sm2.y;

                uchar byte = qs[pair * 32 + simd_lane];
                float lo_q = float(byte & 0x0F)
                    + (((high_bits >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
                float hi_q = float(byte >> 4)
                    + (((high_bits >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);

                half lo = (pair == 0) ? rx0 : (pair == 1) ? rx2 : (pair == 2) ? rx4 : rx6;
                half hi = (pair == 0) ? rx1 : (pair == 1) ? rx3 : (pair == 2) ? rx5 : rx7;
                sumf[row] += (d1 * lo_q - m1) * float(lo);
                sumf[row] += (d2 * hi_q - m2) * float(hi);
            }
        }
    }

    float sum0 = simd_sum(sumf[0]);
    float sum1 = simd_sum(sumf[1]);
    if (simd_lane == 0) {
        y[first_row] = sum0;
        if (valid1) {
            y[first_row + 1] = sum1;
        }
    }
}

kernel void dequant_matvec_q4_k(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block* a_row = A + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        sum += q4_k_block_dot(a_row, b, x, simd_lane);
    }

    // SIMD-level reduction (32 → 1 per SIMD group)
    sum = simd_sum(sum);

    // Cross-SIMD reduction via threadgroup memory
    threadgroup float simd_sums[num_simd_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) {
            y[row] = sum;
        }
    }
}

// Q4_K decode matvec modeled on llama.cpp's multi-row structure:
// 2 rows per simdgroup, 2 simdgroups per threadgroup (TG=64).
// Each simdgroup reuses the loaded x values across both output rows and writes
// them independently, so no cross-simdgroup barrier or reduction is needed.
constant uint Q4K_NR2_ROWS_PER_SG = 2;
constant uint Q4K_NR2_SG_PER_TG = 2;
constant uint Q4K_NR2_ROWS_PER_TG = Q4K_NR2_ROWS_PER_SG * Q4K_NR2_SG_PER_TG;

kernel void dequant_matvec_q4_k_nr2(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    ushort ix = simd_lane / 8;  // 0..3
    ushort it = simd_lane % 8;  // 0..7
    ushort iq = it / 4;         // 0 or 1
    ushort ir = it % 4;         // 0..3

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    uint first_row = (tg_id * Q4K_NR2_SG_PER_TG + simd_id) * Q4K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    bool valid1 = (first_row + 1) < M;
    device const Q4_K_Block* row0 = A + first_row * blocks_per_row;
    device const Q4_K_Block* row1 = valid1 ? row0 + blocks_per_row : row0;
    device const float* y4 = x + ix * Q4_K_BLOCK_VALUES + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[Q4K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*) sc16;

    for (uint ib = ix; ib < blocks_per_row; ib += 4) {
        float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};

        FOR_UNROLL (ushort i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];
            yl[i + 8] = y4[i + 32];
            yh[i + 0] = y4[i + 128];
            yh[i + 8] = y4[i + 160];
            sumy[0] += yl[i + 0];
            sumy[1] += yl[i + 8];
            sumy[2] += yh[i + 0];
            sumy[3] += yh[i + 8];
        }

        for (ushort row = 0; row < Q4K_NR2_ROWS_PER_SG; ++row) {
            device const Q4_K_Block* row_ptr = (row == 0) ? row0 : row1;
            device const Q4_K_Block& blk = row_ptr[ib];
            device const ushort* sc = (device const ushort*) blk.scales + iq;
            device const ushort* q1 = (device const ushort*) blk.qs + 16 * iq + 4 * ir;
            device const half* dh = &blk.d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort* q2 = q1 + 32;

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};

            FOR_UNROLL (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0]) * (
                (acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0]) +
                (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4]) +
                (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5]) * (1.0f / 16.0f)
            ) - float(dh[1]) * (
                sumy[0] * float(sc8[2]) +
                sumy[1] * float(sc8[3]) +
                sumy[2] * float(sc8[6]) +
                sumy[3] * float(sc8[7])
            );
        }

        y4 += 4 * Q4_K_BLOCK_VALUES;
    }

    float sum0 = simd_sum(sumf[0]);
    float sum1 = simd_sum(sumf[1]);
    if (simd_lane == 0) {
        y[first_row] = sum0;
        if (valid1) {
            y[first_row + 1] = sum1;
        }
    }
}

// Q4_K decode matvec with a 2-block unrolled inner loop. This keeps the
// existing TG=128 geometry and output mapping, but increases ILP by letting
// each simdgroup handle two blocks per outer-loop step when available.
kernel void dequant_matvec_q4_k_blk2(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block* a_row = A + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups * 2) {
        sum += q4_k_block_dot(a_row, b, x, simd_lane);
        uint b1 = b + num_simd_groups;
        if (b1 < blocks_per_row) {
            sum += q4_k_block_dot(a_row, b1, x, simd_lane);
        }
    }

    sum = simd_sum(sum);

    threadgroup float simd_sums[num_simd_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) {
            y[row] = sum;
        }
    }
}

// Q4_K decode matvec specialized through a Metal function constant so the same
// kernel body can be measured at a different threadgroup size without adding
// another handwritten variant.
constant ushort Q4K_MATVEC_TG [[function_constant(0)]];
constant uint Q4K_MATVEC_TG256_SIMD_GROUPS = 8;

kernel void dequant_matvec_q4_k_tg(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block* a_row = A + row * blocks_per_row;
    uint num_simd_groups = uint(Q4K_MATVEC_TG) / 32;

    float sum = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        sum += q4_k_block_dot(a_row, b, x, simd_lane);
    }

    sum = simd_sum(sum);

    threadgroup float simd_sums[Q4K_MATVEC_TG256_SIMD_GROUPS];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) {
            y[row] = sum;
        }
    }
}

kernel void dequant_matvec_dense_f16(
    device const half* A [[buffer(0)]],  // [M × K]
    device const float* x [[buffer(1)]], // [K]
    device float* y      [[buffer(2)]],  // [M]
    constant uint& M     [[buffer(3)]],
    constant uint& K     [[buffer(4)]],
    uint row             [[threadgroup_position_in_grid]],
    uint lid             [[thread_index_in_threadgroup]],
    uint simd_lane       [[thread_index_in_simdgroup]],
    uint simd_id         [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;
    device const half* a_row = A + row * K;
    float sum = 0.0f;

    for (uint i = simd_id * 32 + simd_lane; i < K; i += num_simd_groups * 32) {
        sum += float(a_row[i]) * x[i];
    }

    sum = simd_sum(sum);

    threadgroup float simd_sums[num_simd_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) {
            y[row] = sum;
        }
    }
}

// Decode-optimized Q4_K matvec: two output rows per threadgroup with staged x-block.
// Enable via runtime routing for A/B benchmarking.
kernel void dequant_matvec_q4_k_x2(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint row_pair              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint row0 = row_pair * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < M;
    bool valid1 = row1 < M;
    if (!valid0 && !valid1) return;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block* a_row0 = A + row0 * blocks_per_row;
    device const Q4_K_Block* a_row1 = A + row1 * blocks_per_row;

    // 128 threads = 4 simdgroups. Map to 2 rows × 2 partitions.
    uint row_sel = simd_id >> 1;   // 0 or 1
    uint part = simd_id & 1;       // 0 or 1

    threadgroup float x_block[Q4_K_BLOCK_VALUES];
    threadgroup float partial[4];  // [row0_part0, row0_part1, row1_part0, row1_part1]

    float sum = 0.0f;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint base = b * Q4_K_BLOCK_VALUES;

        // Stage one 256-value x block once per threadgroup.
        for (uint i = lid; i < Q4_K_BLOCK_VALUES; i += DEQUANT_MATVEC_TG) {
            x_block[i] = x[base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool active = (row_sel == 0) ? valid0 : valid1;
        if (active) {
            device const Q4_K_Block* row_ptr = (row_sel == 0) ? a_row0 : a_row1;
            float d = float(row_ptr[b].d);
            float dmin = float(row_ptr[b].dmin);
            device const uchar* scales = row_ptr[b].scales;
            device const uchar* qs = row_ptr[b].qs;

            // Split work: part0 handles pairs 0,1; part1 handles pairs 2,3.
            uint pair_start = part * 2;
            for (uint pair = pair_start; pair < pair_start + 2; pair++) {
                float2 sm1 = get_scale_min_q4k(pair * 2, scales);
                float2 sm2 = get_scale_min_q4k(pair * 2 + 1, scales);
                float d1 = d * sm1.x;
                float m1 = dmin * sm1.y;
                float d2 = d * sm2.x;
                float m2 = dmin * sm2.y;

                uchar byte = qs[pair * 32 + simd_lane];
                uint off = pair * 64;
                sum += (d1 * float(byte & 0x0F) - m1) * x_block[off + simd_lane];
                sum += (d2 * float(byte >> 4)   - m2) * x_block[off + simd_lane + 32];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        partial[row_sel * 2 + part] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0 && part == 0) {
        float out = partial[row_sel * 2 + 0] + partial[row_sel * 2 + 1];
        uint out_row = (row_sel == 0) ? row0 : row1;
        if (out_row < M) {
            y[out_row] = out;
        }
    }
}

// N_DST=4 Q4_K decode matvec: four output rows per threadgroup, TG=32 (1 simdgroup).
//
// Design follows the Candle/mistral.rs approach: a single simdgroup of 32 threads
// computes 4 output rows simultaneously. The x vector is loaded once per block and
// reused across all 4 accumulators, amortizing x bandwidth 4×. With TG=32 = 1
// simdgroup, no threadgroup barrier is needed — simd_sum() suffices.
//
// Grid: ceil(M/4) threadgroups × 1.
constant uint NDST4_TG   = 32;  // 1 simdgroup per TG
constant uint NDST4_ROWS = 4;   // output rows per TG

kernel void dequant_matvec_q4_k_n4(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]]   // 0..31, one simdgroup
) {
    uint r0 = tg_id * NDST4_ROWS;
    if (r0 >= M) return;
    bool v1 = (r0 + 1) < M;
    bool v2 = (r0 + 2) < M;
    bool v3 = (r0 + 3) < M;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;

    // Row pointers — clamp out-of-bounds rows to row0 to avoid OOB reads;
    // their outputs are suppressed by the v1/v2/v3 write guards below.
    device const Q4_K_Block* row0 = A + r0 * blocks_per_row;
    device const Q4_K_Block* row1 = v1 ? A + (r0 + 1) * blocks_per_row : row0;
    device const Q4_K_Block* row2 = v2 ? A + (r0 + 2) * blocks_per_row : row0;
    device const Q4_K_Block* row3 = v3 ? A + (r0 + 3) * blocks_per_row : row0;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint base = b * Q4_K_BLOCK_VALUES;

        // Pre-load 8 x values for this block into registers (shared across all 4 rows).
        half rx0 = half(x[base +   0 + lane]);
        half rx1 = half(x[base +  32 + lane]);
        half rx2 = half(x[base +  64 + lane]);
        half rx3 = half(x[base +  96 + lane]);
        half rx4 = half(x[base + 128 + lane]);
        half rx5 = half(x[base + 160 + lane]);
        half rx6 = half(x[base + 192 + lane]);
        half rx7 = half(x[base + 224 + lane]);

        // Uniform reads (same address for all 32 lanes) — hardware broadcasts.
        float d0 = float(row0[b].d),  dm0 = float(row0[b].dmin);
        float d1 = float(row1[b].d),  dm1 = float(row1[b].dmin);
        float d2 = float(row2[b].d),  dm2 = float(row2[b].dmin);
        float d3 = float(row3[b].d),  dm3 = float(row3[b].dmin);

        for (uint pair = 0; pair < 4; pair++) {
            // Uniform scale reads (same address for all 32 lanes — L2 broadcast hits).
            float2 sm0a = get_scale_min_q4k(pair * 2,     row0[b].scales);
            float2 sm0b = get_scale_min_q4k(pair * 2 + 1, row0[b].scales);
            float2 sm1a = get_scale_min_q4k(pair * 2,     row1[b].scales);
            float2 sm1b = get_scale_min_q4k(pair * 2 + 1, row1[b].scales);
            float2 sm2a = get_scale_min_q4k(pair * 2,     row2[b].scales);
            float2 sm2b = get_scale_min_q4k(pair * 2 + 1, row2[b].scales);
            float2 sm3a = get_scale_min_q4k(pair * 2,     row3[b].scales);
            float2 sm3b = get_scale_min_q4k(pair * 2 + 1, row3[b].scales);

            // Lane-dependent qs reads — each lane handles its own position.
            uchar q0b = row0[b].qs[pair * 32 + lane];
            uchar q1b = row1[b].qs[pair * 32 + lane];
            uchar q2b = row2[b].qs[pair * 32 + lane];
            uchar q3b = row3[b].qs[pair * 32 + lane];

            // Select pre-loaded x registers for this pair (shared across all 4 rows).
            half lo = (pair == 0) ? rx0 : (pair == 1) ? rx2 : (pair == 2) ? rx4 : rx6;
            half hi = (pair == 0) ? rx1 : (pair == 1) ? rx3 : (pair == 2) ? rx5 : rx7;

            s0 += (d0 * sm0a.x * float(q0b & 0xF) - dm0 * sm0a.y) * float(lo);
            s0 += (d0 * sm0b.x * float(q0b >> 4)  - dm0 * sm0b.y) * float(hi);
            s1 += (d1 * sm1a.x * float(q1b & 0xF) - dm1 * sm1a.y) * float(lo);
            s1 += (d1 * sm1b.x * float(q1b >> 4)  - dm1 * sm1b.y) * float(hi);
            s2 += (d2 * sm2a.x * float(q2b & 0xF) - dm2 * sm2a.y) * float(lo);
            s2 += (d2 * sm2b.x * float(q2b >> 4)  - dm2 * sm2b.y) * float(hi);
            s3 += (d3 * sm3a.x * float(q3b & 0xF) - dm3 * sm3a.y) * float(lo);
            s3 += (d3 * sm3b.x * float(q3b >> 4)  - dm3 * sm3b.y) * float(hi);
        }
    }

    // Single-simdgroup reduction — no threadgroup barrier needed.
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    s2 = simd_sum(s2);
    s3 = simd_sum(s3);

    if (lane == 0) {
        y[r0] = s0;
        if (v1) y[r0 + 1] = s1;
        if (v2) y[r0 + 2] = s2;
        if (v3) y[r0 + 3] = s3;
    }
}

// N_DST=4 Q8_0 decode matvec: four output rows per threadgroup, TG=32.
//
// One simdgroup computes 4 rows at once, reusing x (one value per lane per block)
// across all 4 row accumulators to reduce x bandwidth pressure.
kernel void dequant_matvec_q8_0_n4(
    device const Q8_0_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint lane                  [[thread_index_in_threadgroup]]
) {
    uint r0 = tg_id * NDST4_ROWS;
    if (r0 >= M) return;
    bool v1 = (r0 + 1) < M;
    bool v2 = (r0 + 2) < M;
    bool v3 = (r0 + 3) < M;

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    device const Q8_0_Block* row0 = A + r0 * blocks_per_row;
    device const Q8_0_Block* row1 = v1 ? A + (r0 + 1) * blocks_per_row : row0;
    device const Q8_0_Block* row2 = v2 ? A + (r0 + 2) * blocks_per_row : row0;
    device const Q8_0_Block* row3 = v3 ? A + (r0 + 3) * blocks_per_row : row0;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint idx = b * Q8_0_BLOCK_VALUES + lane;
        float xv = x[idx];

        float d0 = float(row0[b].d);
        float d1 = float(row1[b].d);
        float d2 = float(row2[b].d);
        float d3 = float(row3[b].d);

        int q0 = int(row0[b].qs[lane]);
        int q1 = int(row1[b].qs[lane]);
        int q2 = int(row2[b].qs[lane]);
        int q3 = int(row3[b].qs[lane]);

        s0 += d0 * float(q0) * xv;
        s1 += d1 * float(q1) * xv;
        s2 += d2 * float(q2) * xv;
        s3 += d3 * float(q3) * xv;
    }

    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    s2 = simd_sum(s2);
    s3 = simd_sum(s3);

    if (lane == 0) {
        y[r0] = s0;
        if (v1) y[r0 + 1] = s1;
        if (v2) y[r0 + 2] = s2;
        if (v3) y[r0 + 3] = s3;
    }
}

// R3 (Q6_K): register-based half x-caching, original stride structure.
//
// Each simdgroup pre-loads all 8 x-values it needs into register-resident
// half variables before the two-group compute block.  No extra barriers.
kernel void dequant_matvec_q6_k(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    device const Q6_K_Block* a_row = A + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        uint base = b * Q6_K_BLOCK_VALUES;
        uint l = simd_lane;  // 0..31

        // Pre-load all 8 x-values for this block as half (2 per group-pair).
        half rx0 = half(x[base +   0 + l]);  // group 0, q1
        half rx1 = half(x[base +  32 + l]);  // group 0, q2
        half rx2 = half(x[base +  64 + l]);  // group 0, q3
        half rx3 = half(x[base +  96 + l]);  // group 0, q4
        half rx4 = half(x[base + 128 + l]);  // group 1, q1
        half rx5 = half(x[base + 160 + l]);  // group 1, q2
        half rx6 = half(x[base + 192 + l]);  // group 1, q3
        half rx7 = half(x[base + 224 + l]);  // group 1, q4

        float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
        d = simd_broadcast(d, 0);
        device const uchar* ql     = a_row[b].ql;
        device const uchar* qh     = a_row[b].qh;
        device const char*  scales = a_row[b].scales;

        uint is = l / 16;  // scale row index: 0 or 1

        // Group 0: ql_idx=0, qh_idx=0, sc_idx=0
        {
            int q1 = int((ql[l] & 0x0F)      | ((qh[l] & 3)         << 4)) - 32;
            int q2 = int((ql[l + 32] & 0x0F)  | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = int((ql[l] >> 4)         | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = int((ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3) << 4)) - 32;

            sum += d * float(scales[is])     * float(q1) * float(rx0);
            sum += d * float(scales[is + 2]) * float(q2) * float(rx1);
            sum += d * float(scales[is + 4]) * float(q3) * float(rx2);
            sum += d * float(scales[is + 6]) * float(q4) * float(rx3);
        }

        // Group 1: ql_idx=64, qh_idx=32, sc_idx=8
        {
            int q1 = int((ql[64 + l] & 0x0F)      | ((qh[32 + l] & 3)         << 4)) - 32;
            int q2 = int((ql[64 + l + 32] & 0x0F)  | (((qh[32 + l] >> 2) & 3) << 4)) - 32;
            int q3 = int((ql[64 + l] >> 4)         | (((qh[32 + l] >> 4) & 3) << 4)) - 32;
            int q4 = int((ql[64 + l + 32] >> 4)    | (((qh[32 + l] >> 6) & 3) << 4)) - 32;

            sum += d * float(scales[8 + is])     * float(q1) * float(rx4);
            sum += d * float(scales[8 + is + 2]) * float(q2) * float(rx5);
            sum += d * float(scales[8 + is + 4]) * float(q3) * float(rx6);
            sum += d * float(scales[8 + is + 6]) * float(q4) * float(rx7);
        }
    }

    // SIMD-level reduction (32 → 1 per SIMD group)
    sum = simd_sum(sum);

    // Cross-SIMD reduction via threadgroup memory
    threadgroup float simd_sums[num_simd_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) {
            y[row] = sum;
        }
    }
}

// Q6_K decode matvec matching the same NR2 structure as Q4_K:
// 2 rows per simdgroup, 2 simdgroups per threadgroup (TG=64), with x values
// loaded once into registers and reused across both output rows.
constant uint Q6K_NR2_ROWS_PER_SG = 2;
constant uint Q6K_NR2_SG_PER_TG = 2;
constant uint Q6K_NR2_ROWS_PER_TG = Q6K_NR2_ROWS_PER_SG * Q6K_NR2_SG_PER_TG;

kernel void dequant_matvec_q6_k_nr2(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    uint first_row = (tg_id * Q6K_NR2_SG_PER_TG + simd_id) * Q6K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    bool valid1 = (first_row + 1) < M;
    device const Q6_K_Block* row0 = A + first_row * blocks_per_row;
    device const Q6_K_Block* row1 = valid1 ? row0 + blocks_per_row : row0;

    float sumf[Q6K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};
    float yl[16];

    ushort tid = simd_lane / 2;
    ushort ix = simd_lane % 2;
    ushort ip = tid / 8;      // 0 or 1
    ushort il = tid % 8;      // 0..7
    ushort l0 = 4 * il;
    ushort is = 8 * ip + l0 / 16;

    ushort y_offset = 128 * ip + l0;
    ushort q_offset_l = 64 * ip + l0;
    ushort q_offset_h = 32 * ip + l0;

    for (uint ib = ix; ib < blocks_per_row; ib += 2) {
        device const float* xv = x + ib * Q6_K_BLOCK_VALUES + y_offset;

        FOR_UNROLL (ushort l = 0; l < 4; ++l) {
            yl[4 * l + 0] = xv[l + 0];
            yl[4 * l + 1] = xv[l + 32];
            yl[4 * l + 2] = xv[l + 64];
            yl[4 * l + 3] = xv[l + 96];
        }

        for (ushort row = 0; row < Q6K_NR2_ROWS_PER_SG; ++row) {
            device const Q6_K_Block* row_ptr = (row == 0) ? row0 : row1;
            device const Q6_K_Block& blk = row_ptr[ib];
            device const uchar* q1 = blk.ql + q_offset_l;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = blk.qh + q_offset_h;
            device const char* sc = blk.scales + is;

            float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
            FOR_UNROLL (ushort l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += float(blk.d) * (
                sums[0] * float(sc[0]) +
                sums[1] * float(sc[2]) +
                sums[2] * float(sc[4]) +
                sums[3] * float(sc[6])
            );
        }
    }

    float sum0 = simd_sum(sumf[0]);
    float sum1 = simd_sum(sumf[1]);
    if (simd_lane == 0) {
        y[first_row] = sum0;
        if (valid1) {
            y[first_row + 1] = sum1;
        }
    }
}

// ── Simd-sum Batch GEMM (K-parallel, high TG count) ──────────────────
//
// C[N × C_STRIDE] = B[N × K] × dequant(A[M × K])^T
//
// Each TG (64 threads = 2 simdgroups) computes 8 output rows.
// Each simdgroup handles 4 rows; each thread handles 8 elements per superblock.
// K-reduction via simd_sum() — no threadgroup memory, no barriers.
//
// Grid = (ceil(M / 8), N).
// Register budget: ~4 acc + 8 x + ~4 per-row scl/off + misc ≈ 20 floats/thread.

constant uint SBLK_TG = 64;
constant uint SBLK_NS = SBLK_TG / 32;  // 2 simdgroups per TG
constant uint SBLK_NR = 4;             // output rows per simdgroup

// ── dequant_batch_q4_k_simd ──────────────────────────────────────────
//
// Per thread tiisg (0..31), element assignment within one 256-element Q4_K superblock:
//   pair_idx = tiisg / 8           → which of 4 64-element pairs
//   l_base   = (tiisg % 4) * 8    → byte start within pair (0,8,16,24)
//   qs_off   = pair_idx*32 + l_base → starting byte in blk.qs[128]
//   is_hi    = (tiisg % 8) >= 4   → take hi nibble (true) or lo nibble (false)
//   sub_blk  = pair_idx*2 + is_hi → which of 8 sub-blocks (for scale lookup)
//   x_off    = ib*256 + tiisg*8   → B vector element index (consecutive across threads)
kernel void dequant_batch_q4_k_simd(
    device const Q4_K_Block* A  [[buffer(0)]],  // [M × nb] Q4_K blocks
    device const float*      B  [[buffer(1)]],  // [N × K] float32
    device float*            C  [[buffer(2)]],  // [N × C_STRIDE] float32
    constant uint&           M  [[buffer(3)]],
    constant uint&           N  [[buffer(4)]],
    constant uint&           K  [[buffer(5)]],
    constant uint&    C_STRIDE  [[buffer(6)]],
    uint2 group_id  [[threadgroup_position_in_grid]],
    uint  sgitg     [[simdgroup_index_in_threadgroup]],
    uint  tiisg     [[thread_index_in_simdgroup]]
) {
    uint m_base = group_id.x * (SBLK_NS * SBLK_NR) + sgitg * SBLK_NR;
    uint n_tok  = group_id.y;
    if (m_base >= M || n_tok >= N) return;

    uint nb = K / Q4_K_BLOCK_VALUES;

    // Per-thread element assignment (constant for the entire kernel invocation).
    uint pair_idx = tiisg / 8;
    uint l_base   = (tiisg % 4) * 8;
    uint qs_off   = pair_idx * 32 + l_base;
    bool is_hi    = (tiisg % 8) >= 4;
    uint sub_blk  = pair_idx * 2 + (is_hi ? 1u : 0u);

    // Clamped row indices so reads stay in bounds for partial boundary TGs.
    uint rows = min(SBLK_NR, M - m_base);
    uint m0 = m_base;
    uint m1 = min(m_base + 1, M - 1);
    uint m2 = min(m_base + 2, M - 1);
    uint m3 = min(m_base + 3, M - 1);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint ib = 0; ib < nb; ib++) {
        uint x_off = n_tok * K + ib * Q4_K_BLOCK_VALUES + tiisg * 8;
        float4 xs_lo = float4(B[x_off+0], B[x_off+1], B[x_off+2], B[x_off+3]);
        float4 xs_hi = float4(B[x_off+4], B[x_off+5], B[x_off+6], B[x_off+7]);

#define Q4K_SIMD_ROW(acc, m_idx) \
        { \
            device const Q4_K_Block& blk = A[(m_idx) * nb + ib]; \
            float2 sm = get_scale_min_q4k(sub_blk, blk.scales); \
            float scl = float(blk.d) * sm.x; \
            float off = float(blk.dmin) * sm.y; \
            device const uchar* qp = blk.qs + qs_off; \
            uint4 blo = uint4(qp[0], qp[1], qp[2], qp[3]); \
            uint4 bhi = uint4(qp[4], qp[5], qp[6], qp[7]); \
            float4 flo = is_hi ? float4(blo >> 4) : float4(blo & 0xFu); \
            float4 fhi = is_hi ? float4(bhi >> 4) : float4(bhi & 0xFu); \
            acc += dot(flo * scl - off, xs_lo) + dot(fhi * scl - off, xs_hi); \
        }

        Q4K_SIMD_ROW(acc0, m0)
        Q4K_SIMD_ROW(acc1, m1)
        Q4K_SIMD_ROW(acc2, m2)
        Q4K_SIMD_ROW(acc3, m3)
#undef Q4K_SIMD_ROW
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (tiisg == 0) {
        uint c_off = n_tok * C_STRIDE + m_base;
                          C[c_off + 0] = acc0;
        if (rows > 1) C[c_off + 1] = acc1;
        if (rows > 2) C[c_off + 2] = acc2;
        if (rows > 3) C[c_off + 3] = acc3;
    }
}

// ── dequant_batch_q6_k_simd ──────────────────────────────────────────
//
// Per thread tiisg (0..31), element assignment within one 256-element Q6_K superblock:
//   group_idx  = tiisg / 16         → first or second 128-element half
//   q_type     = (tiisg % 16) / 4  → 0=q1 (lo ql[0..31],  qh bits[1:0]),
//                                     1=q2 (lo ql[32..63], qh bits[3:2]),
//                                     2=q3 (hi ql[0..31],  qh bits[5:4]),
//                                     3=q4 (hi ql[32..63], qh bits[7:6])
//   l_base     = (tiisg % 4) * 8   → byte start within 32-element sub (0,8,16,24)
//   ql_off     = group_idx*64 + (q_type odd ? 32 : 0) + l_base
//   qh_off     = group_idx*32 + l_base
//   qh_shift   = q_type * 2
//   sc_idx     = group_idx*8 + q_type*2 + (l_base>=16 ? 1 : 0)
//   x_off      = ib*256 + tiisg*8  (consecutive across 32 threads)
kernel void dequant_batch_q6_k_simd(
    device const Q6_K_Block* A  [[buffer(0)]],  // [M × nb] Q6_K blocks
    device const float*      B  [[buffer(1)]],  // [N × K] float32
    device float*            C  [[buffer(2)]],  // [N × C_STRIDE] float32
    constant uint&           M  [[buffer(3)]],
    constant uint&           N  [[buffer(4)]],
    constant uint&           K  [[buffer(5)]],
    constant uint&    C_STRIDE  [[buffer(6)]],
    uint2 group_id  [[threadgroup_position_in_grid]],
    uint  sgitg     [[simdgroup_index_in_threadgroup]],
    uint  tiisg     [[thread_index_in_simdgroup]]
) {
    uint m_base = group_id.x * (SBLK_NS * SBLK_NR) + sgitg * SBLK_NR;
    uint n_tok  = group_id.y;
    if (m_base >= M || n_tok >= N) return;

    uint nb = K / Q6_K_BLOCK_VALUES;

    // Per-thread element assignment (constant for the entire kernel invocation).
    uint group_idx   = tiisg / 16;
    uint q_type      = (tiisg % 16) / 4;
    uint l_base      = (tiisg % 4) * 8;
    bool ql_hi       = (q_type >= 2);          // use hi nibble of ql byte
    bool ql_second32 = (q_type & 1u) != 0u;    // use ql[base+32]
    uint ql_off      = group_idx * 64 + (ql_second32 ? 32u : 0u) + l_base;
    uint qh_off      = group_idx * 32 + l_base;
    uint qh_shift    = q_type * 2;
    uint sc_idx      = group_idx * 8 + q_type * 2 + (l_base >= 16 ? 1u : 0u);

    // Clamped row indices so reads stay in bounds for partial boundary TGs.
    uint rows = min(SBLK_NR, M - m_base);
    uint m0 = m_base;
    uint m1 = min(m_base + 1, M - 1);
    uint m2 = min(m_base + 2, M - 1);
    uint m3 = min(m_base + 3, M - 1);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint ib = 0; ib < nb; ib++) {
        uint x_off = n_tok * K + ib * Q6_K_BLOCK_VALUES + tiisg * 8;
        float4 xs_lo = float4(B[x_off+0], B[x_off+1], B[x_off+2], B[x_off+3]);
        float4 xs_hi = float4(B[x_off+4], B[x_off+5], B[x_off+6], B[x_off+7]);

#define Q6K_SIMD_ROW(acc, m_idx) \
        { \
            device const Q6_K_Block& blk = A[(m_idx) * nb + ib]; \
            float sc = float(blk.d) * float(blk.scales[sc_idx]); \
            device const uchar* ql = blk.ql + ql_off; \
            device const uchar* qh = blk.qh + qh_off; \
            uint4 ql_lo = uint4(ql[0], ql[1], ql[2], ql[3]); \
            uint4 ql_hi4 = uint4(ql[4], ql[5], ql[6], ql[7]); \
            uint4 qh_lo = uint4(qh[0], qh[1], qh[2], qh[3]); \
            uint4 qh_hi = uint4(qh[4], qh[5], qh[6], qh[7]); \
            uint4 lo_lo = ql_hi ? (ql_lo >> 4) : (ql_lo & 0xFu); \
            uint4 lo_hi = ql_hi ? (ql_hi4 >> 4) : (ql_hi4 & 0xFu); \
            uint4 h2_lo = (qh_lo >> qh_shift) & 3u; \
            uint4 h2_hi = (qh_hi >> qh_shift) & 3u; \
            int4 q_lo = int4(lo_lo | (h2_lo << 4)) - 32; \
            int4 q_hi = int4(lo_hi | (h2_hi << 4)) - 32; \
            acc += dot(float4(q_lo) * sc, xs_lo) + dot(float4(q_hi) * sc, xs_hi); \
        }

        Q6K_SIMD_ROW(acc0, m0)
        Q6K_SIMD_ROW(acc1, m1)
        Q6K_SIMD_ROW(acc2, m2)
        Q6K_SIMD_ROW(acc3, m3)
#undef Q6K_SIMD_ROW
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (tiisg == 0) {
        uint c_off = n_tok * C_STRIDE + m_base;
                          C[c_off + 0] = acc0;
        if (rows > 1) C[c_off + 1] = acc1;
        if (rows > 2) C[c_off + 2] = acc2;
        if (rows > 3) C[c_off + 3] = acc3;
    }
}
