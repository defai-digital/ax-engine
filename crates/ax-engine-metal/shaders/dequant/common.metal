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

// ── Q5_0 Block ─────────────────────────────────────────────────────────
//
// 22 bytes → 32 f32 values.
//   - 2 bytes:  f16 scale (d)
//   - 4 bytes:  high bits (qh), one bit per value packed as u32
//   - 16 bytes: 32 × 4-bit nibbles packed into 16 bytes (qs)
//
// Dequant: output = d * ((nibble | (qh_bit << 4)) - 16)
struct Q5_0_Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};

static_assert(sizeof(Q5_0_Block) == 22, "Q5_0_Block must be exactly 22 bytes");
constant uint Q5_0_BLOCK_VALUES = 32;

// ── Q5_1 Block ─────────────────────────────────────────────────────────
//
// 24 bytes → 32 f32 values.
//   - 2 bytes:  f16 scale (d)
//   - 2 bytes:  f16 minimum (m)
//   - 4 bytes:  high bits (qh), one bit per value packed as u32
//   - 16 bytes: 32 × 4-bit nibbles packed into 16 bytes (qs)
//
// Dequant: output = d * (nibble | (qh_bit << 4)) + m
// (Unlike Q5_0 which uses -16*d offset, Q5_1 stores explicit min)
struct Q5_1_Block {
    half d;
    half m;
    uint8_t qh[4];
    uint8_t qs[16];
};

static_assert(sizeof(Q5_1_Block) == 24, "Q5_1_Block must be exactly 24 bytes");
constant uint Q5_1_BLOCK_VALUES = 32;

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



// Same kernel for Q6_K format.
// Processes K in chunks of 64 values (half a 128-value group in Q6_K).



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

// Tile constants for 64x32 full-tile kernels (f16in dequant, BN=32).
constant uint D32_BM = 64;
constant uint D32_BN = 32;
constant uint D32_TG = 128;  // 4 simdgroups × 32 threads









constant uint DB32_BN = 32;
constant uint DB32_TG = 128;



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



// ═══════════════════════════════════════════════════════════════════════════
// dequant_batch_q4_k_blocked_silu — Fused SiLU activation + down projection
//
// Identical to dequant_batch_q4_k_blocked except the B-loading phase reads
// from two buffers (gate, up) and computes silu(gate) * up inline, replacing
// the separate SiLU activation dispatch.
//
// Saves 1 Metal dispatch per prefill layer.
// ═══════════════════════════════════════════════════════════════════════════

// silu_f is defined later in this file for decode matvec kernels.
// Forward-declare here so the batch kernel can use it.
inline float silu_f_batch(float x) { return x / (1.0f + exp(-x)); }



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

    if (il < 2) {
        for (int row = 0; row < 4; ++row) {
            uchar4 qv = *(device const uchar4*)(q + row * 4);
            uchar4 qhv = *(device const uchar4*)(qh + row * 4);
            uint4 lo = uint4(qv & uchar4(0x0Fu));
            uint4 hi = select(uint4(0u), uint4(16u), (qhv & uchar4(ul)) != uchar4(0u));
            float4 vals = float4(lo + hi);
            reg[row] = half4(dl * vals - ml);
        }
    } else {
        for (int row = 0; row < 4; ++row) {
            uchar4 qv = *(device const uchar4*)(q + row * 4);
            uchar4 qhv = *(device const uchar4*)(qh + row * 4);
            uint4 lo = uint4(qv & uchar4(0xF0u));
            uint4 hi = select(uint4(0u), uint4(256u), (qhv & uchar4(ul)) != uchar4(0u));
            float4 vals = float4(lo + hi);
            reg[row] = half4(dl * vals - ml);
        }
    }
}

// Q5_K blocked batch kernel (llama.cpp kernel_mul_mm geometry).


// Q5_K blocked batch kernel with f16 input and f32 output.


// ── Q5_K f16in full64/full32/tail32/small kernels ─────────────────────
//
// Adapted from the Q6_K f16in kernels, replacing Q6_K dequant with the
// Q5_K 3-phase dequant pattern: scale precompute -> barrier -> paired
// nibble + high-bit extraction.
//
// Q5_K_BLOCK_VALUES = 256, BK = 64, so pair = (kt % 256) / 64 in {0..3}.
// Each pair covers 64 values (32 bytes of qs, split into lo/hi nibbles,
// plus 1 high bit per value from qh[0..31]).









// Q5_K f16-input pair kernel (gate+up fused).
// Dual-output: dequants A0 and A1 (Q5_K) with shared B (half), produces C0 and C1.
// Uses P16_BM=32, P16_BN=64, P16_BK=64, P16_TG=256 (defined later near pair kernels).
// Placed here near other Q5_K f16in kernels; P16_ constants are forward-declared
// at point of use (line ~6630). This kernel must appear AFTER that constant block
// in the compilation unit, so we use DB_ constants (same values: 32/64/64/256).


// Forward declaration (defined later, after Q6_K blocked BM=64).
inline void dequantize_q6k_blocked(device const Q6_K_Block* xb, short il, thread half4x4& reg);

// Same for Q6_K:


// ── Fused SiLU activation + Q6_K blocked down projection ──────────────
//
// Same structure as dequant_batch_q6_k_blocked but reads gate + up buffers
// instead of plain B, applying silu(gate) * up during B-loading.
// Computes: C = dequant(A) × (silu(gate) ⊙ up)^T
//
// Tile: NR0=32, NR1=32, NK=32.  TG=128.
// Shared memory: sa[2 KB] + sb[2 KB] + staging[4 KB boundary] = 4-8 KB.



// Same batch kernel for Q6_K format.


// Dense f16 batch matmul (B-transposed layout):
//   C[N×M] = B[N×K] × A[M×K]^T
//
// A and B are half, accumulators/output are float.
// Uses the same tile shape as dequant_batch_* for easy routing parity.


// 64x64 full-tile fast path for dense f16 batch matmul (B-transposed).
// Uses D64_BM/D64_BN/D64_BK/D64_TG constants (defined below near f16in_full64).
// Preconditions enforced by dispatch:
// - M is divisible by 64
// - N is divisible by 64


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




// - C output remains f32 (avoids f16->f32 cast after matmul)




// Q5_0 batch dequant for blocked matmul.
//
// Each call produces 16 values (half a block of 32).
// il=0: low nibbles + high bits for positions 0-15
// il=1: high nibbles + high bits for positions 16-31
//
// Matches llama.cpp's dequantize_q5_0 template with half4x4 output.
inline void dequantize_q5_0_blocked(
    device const Q5_0_Block* xb,
    short il,
    thread half4x4& reg
) {
    device const uint16_t* qs = (device const uint16_t*)(xb->qs);
    const float d = float(xb->d);
    const float md = -16.0f * d;
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = *(device const uint32_t*)(xb->qh);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;

    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2 * i)) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2 * i + 1)) << gh_bk) & 0x10;
        const int32_t x0 = (((qs[i]) & mask) >> x_mv) | xh_0;
        const int32_t x1 = (((qs[i] >> 8) & mask) >> x_mv) | xh_1;
        reg[i / 2][2 * (i % 2) + 0] = half(d * float(x0) + md);
        reg[i / 2][2 * (i % 2) + 1] = half(d * float(x1) + md);
    }
}

// Q5_1 blocked dequant: same structure as Q5_0 but uses explicit per-block
// minimum `m` instead of Q5_0's fixed offset (-16 * d).
// Dequant: output = d * (nibble | (qh_bit << 4)) + m
inline void dequantize_q5_1_blocked(
    device const Q5_1_Block* xb,
    short il,
    thread half4x4& reg
) {
    device const uint16_t* qs = (device const uint16_t*)(xb->qs);
    const float d = float(xb->d);
    const float m = float(xb->m);
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = *(device const uint32_t*)(xb->qh);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;

    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2 * i)) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2 * i + 1)) << gh_bk) & 0x10;
        const int32_t x0 = (((qs[i]) & mask) >> x_mv) | xh_0;
        const int32_t x1 = (((qs[i] >> 8) & mask) >> x_mv) | xh_1;
        reg[i / 2][2 * (i % 2) + 0] = half(d * float(x0) + m);
        reg[i / 2][2 * (i % 2) + 1] = half(d * float(x1) + m);
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





// Q8_0 f32-input batch kernel.
// Q8_0 block: 34 bytes = 2-byte f16 scale (d) + 32 x i8 values.
// blocks_per_row = K / 32.




// Full-tile fast path for Q8_0 f16-input batch kernel.
// Preconditions enforced by dispatch:
// - M is divisible by DB_BM (32)
// - N is divisible by DB_BN (64)
// - K is divisible by DB_BK (64)


// Full-tile fast paths for f16-input kernels.
// Preconditions enforced by dispatch:
// - M is divisible by DB_BM
// - N is divisible by DB_BN





// 64x64 full-tile fast paths for f16-input kernels.
// Preconditions enforced by dispatch:
// - M is divisible by 64
// - N is divisible by 64
// - no boundary masking required



// 64x64 full-tile BK=32 variant for Q4_K f16in.
// This reduces per-iteration working set and increases K-loop iterations.
constant uint D64S_BM = 64;
constant uint D64S_BN = 64;
constant uint D64S_BK = 32;
constant uint D64S_TG = 256;













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





// f16-input pair kernels tuned for BN=64.
// Reuses one B tile for gate+up projections and writes f32 outputs.
constant uint P16_BM = 32;
constant uint P16_BN = 64;
constant uint P16_BK = 64;
constant uint P16_TG = 256;







// Full-tile fast path for Q8_0 f16-input pair kernel.
// Preconditions enforced by dispatch:
// - M is divisible by P16_BM (32)
// - N is divisible by P16_BN (64)
// - K is divisible by P16_BK (64)


// ── Batch Dequant+Matmul (small-N, f16 input) for Q8_0 ───────────────
//
// Small-N variant (BM=32, BN=32, BK=64, TG=128) with boundary checks.
// Q8_0 dequant: d * qs[i] per value. BK=64 covers 2 consecutive Q8_0 blocks.



// ── Batch Dequant+Matmul (small-N variant) ────────────────────────────
//
// Tuned for smaller prefill batches with lower threadgroup memory pressure.
// Uses BN=32 and TG=128.





// ── Batch Dequant+Matmul (small-N, f16 input) ────────────────────────
//
// Same tile as small kernels (BN=32, TG=128), but with f16 batch input.





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







inline float silu_mul_f32(float gate, float up) {
    return gate / (1.0f + exp(-gate)) * up;
}

constant float GELU_SQRT_2_PI = 0.7978845608f;

inline float gelu_mul_f32(float gate, float up) {
    float gate3 = gate * gate * gate;
    float inner = GELU_SQRT_2_PI * (gate + 0.044715f * gate3);
    return 0.5f * gate * (1.0f + tanh(inner)) * up;
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

inline float q4_k_block_dot_silu(
    device const Q4_K_Block* a_row,
    uint b,
    device const float* gate,
    device const float* up,
    uint simd_lane
) {
    uint base = b * Q4_K_BLOCK_VALUES;

    half rx0 = half(silu_mul_f32(gate[base +   0 + simd_lane], up[base +   0 + simd_lane]));
    half rx1 = half(silu_mul_f32(gate[base +  32 + simd_lane], up[base +  32 + simd_lane]));
    half rx2 = half(silu_mul_f32(gate[base +  64 + simd_lane], up[base +  64 + simd_lane]));
    half rx3 = half(silu_mul_f32(gate[base +  96 + simd_lane], up[base +  96 + simd_lane]));
    half rx4 = half(silu_mul_f32(gate[base + 128 + simd_lane], up[base + 128 + simd_lane]));
    half rx5 = half(silu_mul_f32(gate[base + 160 + simd_lane], up[base + 160 + simd_lane]));
    half rx6 = half(silu_mul_f32(gate[base + 192 + simd_lane], up[base + 192 + simd_lane]));
    half rx7 = half(silu_mul_f32(gate[base + 224 + simd_lane], up[base + 224 + simd_lane]));

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

inline float q5_k_block_dot_silu(
    device const Q5_K_Block* a_row,
    uint b,
    device const float* gate,
    device const float* up,
    uint simd_lane
) {
    uint base = b * Q5_K_BLOCK_VALUES;

    half rx0 = half(silu_mul_f32(gate[base +   0 + simd_lane], up[base +   0 + simd_lane]));
    half rx1 = half(silu_mul_f32(gate[base +  32 + simd_lane], up[base +  32 + simd_lane]));
    half rx2 = half(silu_mul_f32(gate[base +  64 + simd_lane], up[base +  64 + simd_lane]));
    half rx3 = half(silu_mul_f32(gate[base +  96 + simd_lane], up[base +  96 + simd_lane]));
    half rx4 = half(silu_mul_f32(gate[base + 128 + simd_lane], up[base + 128 + simd_lane]));
    half rx5 = half(silu_mul_f32(gate[base + 160 + simd_lane], up[base + 160 + simd_lane]));
    half rx6 = half(silu_mul_f32(gate[base + 192 + simd_lane], up[base + 192 + simd_lane]));
    half rx7 = half(silu_mul_f32(gate[base + 224 + simd_lane], up[base + 224 + simd_lane]));

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

inline float q5_k_block_dot_gelu(
    device const Q5_K_Block* a_row,
    uint b,
    device const float* gate,
    device const float* up,
    uint simd_lane
) {
    uint base = b * Q5_K_BLOCK_VALUES;

    half rx0 = half(gelu_mul_f32(gate[base +   0 + simd_lane], up[base +   0 + simd_lane]));
    half rx1 = half(gelu_mul_f32(gate[base +  32 + simd_lane], up[base +  32 + simd_lane]));
    half rx2 = half(gelu_mul_f32(gate[base +  64 + simd_lane], up[base +  64 + simd_lane]));
    half rx3 = half(gelu_mul_f32(gate[base +  96 + simd_lane], up[base +  96 + simd_lane]));
    half rx4 = half(gelu_mul_f32(gate[base + 128 + simd_lane], up[base + 128 + simd_lane]));
    half rx5 = half(gelu_mul_f32(gate[base + 160 + simd_lane], up[base + 160 + simd_lane]));
    half rx6 = half(gelu_mul_f32(gate[base + 192 + simd_lane], up[base + 192 + simd_lane]));
    half rx7 = half(gelu_mul_f32(gate[base + 224 + simd_lane], up[base + 224 + simd_lane]));

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








// Q5_K decode matvec with llama.cpp-style 4-way interleaved block traversal.
// This keeps TG=64 / 2 simdgroups / 1 row-per-simdgroup, but repartitions each
// simdgroup into 4 block streams so 8 lanes cooperate on one 256-value block.


constant uint Q5K_NR2_ROWS_PER_SG = 2;
constant uint Q5K_NR2_SG_PER_TG = 2;
constant uint Q5K_NR2_ROWS_PER_TG = Q5K_NR2_ROWS_PER_SG * Q5K_NR2_SG_PER_TG;

// Q5_K decode matvec with 2 rows per simdgroup and 2 simdgroups per
// threadgroup. This keeps the baseline TG=64 launch geometry but reuses the
// loaded x values across two rows inside each simdgroup.




// ═══════════════════════════════════════════════════════════════════════════
// G14: Fused SiLU(gate) * up + Down projection matvec kernels.
//
// Eliminates the separate SiLU*Up dispatch by computing silu(gate[k])*up[k]
// inline in the down-projection matvec inner loop. Reads from two input
// buffers (gate, up) instead of one; no intermediate buffer needed.
//
// Uses nr2 geometry (TG=64, 2 rows/SG, 4 rows/TG) to match the tuned
// single-output decode matvec performance.
// Grid: ceil(M / 4) threadgroups.
// ═══════════════════════════════════════════════════════════════════════════

inline float silu_f(float x) {
    return x / (1.0f + exp(-x));
}

// Q4_K fused SiLU+Down matvec — nr2 geometry.





// Q4_K decode matvec modeled on llama.cpp's multi-row structure:
// 2 rows per simdgroup, 2 simdgroups per threadgroup (TG=64).
// Each simdgroup reuses the loaded x values across both output rows and writes
// them independently, so no cross-simdgroup barrier or reduction is needed.
constant uint Q4K_NR2_ROWS_PER_SG = 2;
constant uint Q4K_NR2_SG_PER_TG = 2;
constant uint Q4K_NR2_ROWS_PER_TG = Q4K_NR2_ROWS_PER_SG * Q4K_NR2_SG_PER_TG;

// G13: Pair matvec with nr2 geometry — 2 rows/SG, TG=64, no cross-SG
// reduction. Preserves the tuned nr2 inner loop for Q4_K while computing
// two independent outputs (gate+up) per dispatch.
//
// Each SG loads x once per block and reuses it across 2 rows × 2 outputs.
// Grid: ceil(M / 4) threadgroups of 64 threads each.




// Q4_K decode matvec with ILP4 structure: 1 row per simdgroup, 2 SGs per TG
// (TG=64). Each thread strides over 4 blocks per outer iteration with 4
// independent accumulators (one per sub-block pair), giving the compiler
// maximum freedom to interleave independent arithmetic with memory loads.
// Adapted from Q5_K ILP4 — identical except no qh high-bit plane.





constant uint Q8_0_NR2_ROWS_PER_SG = 2;
constant uint Q8_0_NR2_SG_PER_TG = 2;



// ── Q8_0 ILP4 decode matvec: 4-way block unrolling ─────────────────
//
// 1 row per SG, 2 SGs per TG = 2 rows per TG, TG = 64.
// Processes 4 blocks per iteration with 4 independent accumulators
// to maximize instruction-level parallelism and hide memory latency.



// R3 (Q6_K): register-based half x-caching, original stride structure.
//
// Each simdgroup pre-loads all 8 x-values it needs into register-resident
// half variables before the two-group compute block.  No extra barriers.
inline float q6_k_block_dot(
    device const Q6_K_Block* a_row,
    uint b,
    device const float* x,
    uint simd_lane
) {
    uint base = b * Q6_K_BLOCK_VALUES;
    uint l = simd_lane;

    half rx0 = half(x[base +   0 + l]);
    half rx1 = half(x[base +  32 + l]);
    half rx2 = half(x[base +  64 + l]);
    half rx3 = half(x[base +  96 + l]);
    half rx4 = half(x[base + 128 + l]);
    half rx5 = half(x[base + 160 + l]);
    half rx6 = half(x[base + 192 + l]);
    half rx7 = half(x[base + 224 + l]);

    float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
    d = simd_broadcast(d, 0);
    device const uchar* ql = a_row[b].ql;
    device const uchar* qh = a_row[b].qh;
    device const char* scales = a_row[b].scales;

    uint is = l / 16;
    float sum = 0.0f;

    {
        int q1 = int((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) - 32;
        int q2 = int((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

        sum += d * float(scales[is]) * float(q1) * float(rx0);
        sum += d * float(scales[is + 2]) * float(q2) * float(rx1);
        sum += d * float(scales[is + 4]) * float(q3) * float(rx2);
        sum += d * float(scales[is + 6]) * float(q4) * float(rx3);
    }

    {
        int q1 = int((ql[64 + l] & 0x0F) | ((qh[32 + l] & 3) << 4)) - 32;
        int q2 = int((ql[64 + l + 32] & 0x0F) | (((qh[32 + l] >> 2) & 3) << 4)) - 32;
        int q3 = int((ql[64 + l] >> 4) | (((qh[32 + l] >> 4) & 3) << 4)) - 32;
        int q4 = int((ql[64 + l + 32] >> 4) | (((qh[32 + l] >> 6) & 3) << 4)) - 32;

        sum += d * float(scales[8 + is]) * float(q1) * float(rx4);
        sum += d * float(scales[8 + is + 2]) * float(q2) * float(rx5);
        sum += d * float(scales[8 + is + 4]) * float(q3) * float(rx6);
        sum += d * float(scales[8 + is + 6]) * float(q4) * float(rx7);
    }

    return sum;
}

inline float q6_k_block_dot_silu(
    device const Q6_K_Block* a_row,
    uint b,
    device const float* gate,
    device const float* up,
    uint simd_lane
) {
    uint base = b * Q6_K_BLOCK_VALUES;
    uint l = simd_lane;

    half rx0 = half(silu_mul_f32(gate[base +   0 + l], up[base +   0 + l]));
    half rx1 = half(silu_mul_f32(gate[base +  32 + l], up[base +  32 + l]));
    half rx2 = half(silu_mul_f32(gate[base +  64 + l], up[base +  64 + l]));
    half rx3 = half(silu_mul_f32(gate[base +  96 + l], up[base +  96 + l]));
    half rx4 = half(silu_mul_f32(gate[base + 128 + l], up[base + 128 + l]));
    half rx5 = half(silu_mul_f32(gate[base + 160 + l], up[base + 160 + l]));
    half rx6 = half(silu_mul_f32(gate[base + 192 + l], up[base + 192 + l]));
    half rx7 = half(silu_mul_f32(gate[base + 224 + l], up[base + 224 + l]));

    float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
    d = simd_broadcast(d, 0);
    device const uchar* ql = a_row[b].ql;
    device const uchar* qh = a_row[b].qh;
    device const char* scales = a_row[b].scales;

    uint is = l / 16;
    float sum = 0.0f;

    {
        int q1 = int((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) - 32;
        int q2 = int((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

        sum += d * float(scales[is]) * float(q1) * float(rx0);
        sum += d * float(scales[is + 2]) * float(q2) * float(rx1);
        sum += d * float(scales[is + 4]) * float(q3) * float(rx2);
        sum += d * float(scales[is + 6]) * float(q4) * float(rx3);
    }

    {
        int q1 = int((ql[64 + l] & 0x0F) | ((qh[32 + l] & 3) << 4)) - 32;
        int q2 = int((ql[64 + l + 32] & 0x0F) | (((qh[32 + l] >> 2) & 3) << 4)) - 32;
        int q3 = int((ql[64 + l] >> 4) | (((qh[32 + l] >> 4) & 3) << 4)) - 32;
        int q4 = int((ql[64 + l + 32] >> 4) | (((qh[32 + l] >> 6) & 3) << 4)) - 32;

        sum += d * float(scales[8 + is]) * float(q1) * float(rx4);
        sum += d * float(scales[8 + is + 2]) * float(q2) * float(rx5);
        sum += d * float(scales[8 + is + 4]) * float(q3) * float(rx6);
        sum += d * float(scales[8 + is + 6]) * float(q4) * float(rx7);
    }

    return sum;
}

inline float q6_k_block_dot_gelu(
    device const Q6_K_Block* a_row,
    uint b,
    device const float* gate,
    device const float* up,
    uint simd_lane
) {
    uint base = b * Q6_K_BLOCK_VALUES;
    uint l = simd_lane;

    half rx0 = half(gelu_mul_f32(gate[base +   0 + l], up[base +   0 + l]));
    half rx1 = half(gelu_mul_f32(gate[base +  32 + l], up[base +  32 + l]));
    half rx2 = half(gelu_mul_f32(gate[base +  64 + l], up[base +  64 + l]));
    half rx3 = half(gelu_mul_f32(gate[base +  96 + l], up[base +  96 + l]));
    half rx4 = half(gelu_mul_f32(gate[base + 128 + l], up[base + 128 + l]));
    half rx5 = half(gelu_mul_f32(gate[base + 160 + l], up[base + 160 + l]));
    half rx6 = half(gelu_mul_f32(gate[base + 192 + l], up[base + 192 + l]));
    half rx7 = half(gelu_mul_f32(gate[base + 224 + l], up[base + 224 + l]));

    float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
    d = simd_broadcast(d, 0);
    device const uchar* ql = a_row[b].ql;
    device const uchar* qh = a_row[b].qh;
    device const char* scales = a_row[b].scales;

    uint is = l / 16;
    float sum = 0.0f;

    {
        int q1 = int((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) - 32;
        int q2 = int((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

        sum += d * float(scales[is]) * float(q1) * float(rx0);
        sum += d * float(scales[is + 2]) * float(q2) * float(rx1);
        sum += d * float(scales[is + 4]) * float(q3) * float(rx2);
        sum += d * float(scales[is + 6]) * float(q4) * float(rx3);
    }

    {
        int q1 = int((ql[64 + l] & 0x0F) | ((qh[32 + l] & 3) << 4)) - 32;
        int q2 = int((ql[64 + l + 32] & 0x0F) | (((qh[32 + l] >> 2) & 3) << 4)) - 32;
        int q3 = int((ql[64 + l] >> 4) | (((qh[32 + l] >> 4) & 3) << 4)) - 32;
        int q4 = int((ql[64 + l + 32] >> 4) | (((qh[32 + l] >> 6) & 3) << 4)) - 32;

        sum += d * float(scales[8 + is]) * float(q1) * float(rx4);
        sum += d * float(scales[8 + is + 2]) * float(q2) * float(rx5);
        sum += d * float(scales[8 + is + 4]) * float(q3) * float(rx6);
        sum += d * float(scales[8 + is + 6]) * float(q4) * float(rx7);
    }

    return sum;
}









// Q6_K decode matvec matching the same NR2 structure as Q4_K:
// 2 rows per simdgroup, 2 simdgroups per threadgroup (TG=64), with x values
// loaded once into registers and reused across both output rows.
constant uint Q6K_NR2_ROWS_PER_SG = 2;
constant uint Q6K_NR2_SG_PER_TG = 2;
constant uint Q6K_NR2_ROWS_PER_TG = Q6K_NR2_ROWS_PER_SG * Q6K_NR2_SG_PER_TG;



// Q6_K decode matvec with ILP4 structure: 1 row per simdgroup, 2 SGs per TG
// (TG=64). Instead of the NR2 pattern (2 rows per SG, block stride 2), this
// kernel processes 1 row per SG with block stride 4 (ix = simd_lane % 4).
// The extra ILP comes from interleaving 4 independent block streams per thread.
// Thread decomposition within each block is identical to Q6_K NR2.



constant ushort EXT_R1PTG [[function_constant(2)]];



// ═══════════════════════════════════════════════════════════════════════════
// MoE mul_mat_id — unified expert dispatch (ported from llama.cpp)
//
// Stage 1 (map0): Build routing index — for each expert, which tokens use it.
// Stage 2 (main): Tiled Q4_K GEMM with expert routing.
//
// Replaces the per-expert dispatch loop (48 dispatches/layer → 2 dispatches).
// ═══════════════════════════════════════════════════════════════════════════

/// Stage 1: Build routing index from expert assignments.
///
/// expert_ids: [n_tokens, n_expert_used] — which experts each token uses.
/// tpe (tokens per expert): [n_expert] uint32 — count of tokens per expert.
/// hids: [n_expert, n_tokens] int32 — for expert e, hids[e*n_tokens + i] =
///   token_idx * n_expert_used + position_in_assignment for the i-th assigned token.
/// active_experts: [1 + n_expert] uint32 metadata where slot 0 stores the
///   active-expert count and slots 1..count store the compact expert list.
///
/// Grid: (1, 1, 1). TG: (n_expert, 1, 1). One thread per expert.


/// Stage 2: Tiled Q4_K GEMM with expert routing (compact grid).
///
/// Grid.z = min(n_expert, n_tokens * n_expert_used). Kernels early-exit when
/// group.z >= active_count and otherwise use active_experts[1 + group.z].
///
/// Grid: (ceil(n_tokens/32), ceil(M/32), n_active_upper_bound). TG: (128, 1, 1).


// ═══════════════════════════════════════════════════════════════════════════
// Optimized MoE mul_mat_id — 64×32 blocked tiles with half-precision compute.
//
// Adapted from dequant_batch_q4_k_blocked (NR0=64, NR1=32, NK=32) with
// MoE expert routing (active_experts, hids, tpe). Uses simdgroup_half8x8
// for 2× throughput vs the 32×32 float variant.
//
// Grid: (ceil(n_tokens/32), ceil(M/64), n_active_upper_bound). TG: 128.
// ═══════════════════════════════════════════════════════════════════════════


// Optimized MoE mul_mat_id for Q5_K — 64×32 blocked tiles with half compute.
// Same routing/layout as moe_mul_mat_id_q4_k_blocked, but uses Q5_K dequant.
