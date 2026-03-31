// AX Engine — Metal Tensor API matmul (Metal 4 / M5+)
//
// Uses mpp::tensor_ops::matmul2d for hardware-accelerated cooperative
// matrix multiply. Requires Metal 4 GPU family (M5/A19+).
//
// This file is compiled separately and may fail on SDKs that don't
// include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>.
// The build script and runtime both handle this gracefully.
//
// Tile: NR0=64 (M rows) × NR1=32 (N cols) × NK=32 (K reduction).
// Matches llama.cpp kernel_mul_mm tile geometry.
// TG: 128 threads (4 simdgroups via execution_simdgroups<4>).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════
// Tensor API general f32 matmul — C = A × B
//
// A: M × K row-major f32 (device)
// B: K × N row-major f32 (device)
// C: M × N row-major f32 (device)
//
// Uses half-precision threadgroup tiles with float accumulator, matching
// the non-tensor matmul_simdgroup_half_f32 precision profile.
// ═══════════════════════════════════════════════════════════════════════════

constant short TNS_NR0 = 64;   // M-tile (output rows)
constant short TNS_NR1 = 32;   // N-tile (output cols)
constant short TNS_NK  = 32;   // K-tile (reduction)

kernel void matmul_simdgroup_tensor_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup tile buffers (half precision for bandwidth).
    // sa: NR0 × NK half = 64×32×2 = 4096 bytes
    // sb: NR1 × NK half = 32×32×2 = 2048 bytes
    // Total: 6144 bytes (6 KB)
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    const int r0 = tgpig.y * TNS_NR0;  // M offset
    const int r1 = tgpig.x * TNS_NR1;  // N offset

    // Boundary clamping.
    const short nr0 = short(min(uint(TNS_NR0), M - uint(r0)));
    const short nr1 = short(min(uint(TNS_NR1), N - uint(r1)));

    // Create tensor wrappers over threadgroup memory.
    // tA: K×M tile (transposed for matmul2d).
    // tB: N×K tile.
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(TNS_NK, TNS_NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sb, dextents<int32_t, 2>(TNS_NR1, TNS_NK));

    // Configure matmul2d: C[NR1×NR0] += B[NR1×NK] × A[NK×NR0]^T
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            TNS_NR1, TNS_NR0, TNS_NK,
            false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    // ── K-loop ───────────────────────────────────────────────────────
    // 128 threads cooperatively load A and B tiles, then run tensor matmul.
    constexpr short NL0 = TNS_NK / 16;  // = 2
    constexpr short NL1 = TNS_NK / 8;   // = 4

    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));
    const short il0 = short(tiitg % NL0);
    const short iy  = short(8 * (tiitg % NL1));

    device const float* a_ptr = A + uint(r0 + lr0) * K;
    device const float* b_ptr = B + uint(r1 + lr1) * N;  // B is K×N, but we read row lr1 of transposed

    // Actually, A is M×K row-major and B is K×N row-major.
    // We load A row (r0+lr0) and B column (r1+lr1) per thread.
    // For B: need to read K elements at column offset (r1+lr1), stride N.

    for (uint loop_k = 0; loop_k < K; loop_k += TNS_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A tile: each thread loads 16 values from one row.
        // A[r0+lr0, loop_k + il0*16 .. +16] → sa in tensor_inline layout.
        {
            uint a_col = loop_k + il0 * 16;
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = i % 8;
                const short ly = (tiitg / NL0) % 8;
                half val = (uint(r0 + lr0) < M && a_col + i < K)
                    ? half(A[uint(r0 + lr0) * K + a_col + i])
                    : half(0.0h);
                *(sa + TNS_NK * (8 * sy + ly) + 8 * sx + lx) = val;
            }
        }

        // Load B tile: each thread loads 8 values from one column.
        // B[loop_k + iy .. +8, r1+lr1] → sb in tensor_inline layout.
        {
            const short sx = short(tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            for (short i = 0; i < 8; i++) {
                uint b_row = loop_k + iy + i;
                half val = (b_row < K && uint(r1 + lr1) < N)
                    ? half(B[b_row * N + uint(r1 + lr1)])
                    : half(0.0h);
                *(sb + TNS_NK * (8 * sy + ly) + 8 * sx + i) = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tensor matmul: cT += tB × tA^T
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }

    // ── Output write ─────────────────────────────────────────────────
    if (r0 + TNS_NR0 <= int(M) && r1 + TNS_NR1 <= int(N)) {
        // Full tile: direct write to device memory.
        device float* out = C + uint(r0) + uint(r1) * M;
        auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
            out, dextents<int32_t, 2>(int(M), TNS_NR1));
        cT.store(tC);
    } else {
        // Boundary tile: stage through threadgroup, bounds-checked write.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* sc = (threadgroup float*)(shmem);
        auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            sc, dextents<int32_t, 2>(TNS_NR0, TNS_NR1));
        cT.store(tC);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += TNS_NR1) {
                device float*  D = C + uint(r0) + uint(r1 + j) * M;
                device float4* D4 = (device float4*)D;
                threadgroup float*  Cs = sc + j * TNS_NR0;
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
