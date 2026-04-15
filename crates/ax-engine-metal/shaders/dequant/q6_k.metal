// Q6_K dequant and matvec kernels.
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
                device const Q6_K_Block& blk =
                    q6_k_block_ref(q6_k_row_ptr(A, global_r, blocks_per_row), block_idx);
                tg_A[r * DQ_BK + c] = q6_k_dequant_pair_value(blk, group, sub_pair, c);
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
    device const Q6_K_Block* x =
        q6_k_row_ptr(A, uint(r0 + lr0), blocks_per_row) + offset1;

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

kernel void dequant_batch_q6_k_blocked_silu(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* gate   [[buffer(1)]],
    device const float* up     [[buffer(2)]],
    device float* C            [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& N           [[buffer(5)]],
    constant uint& K           [[buffer(6)]],
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
    device const Q6_K_Block* x =
        q6_k_row_ptr(A, uint(r0 + lr0), blocks_per_row) + offset1;

    const short iy = short(8 * (tiitg % NL1));
    device const float* g_ptr = gate + uint(r1 + lr1) * K + iy;
    device const float* u_ptr = up   + uint(r1 + lr1) * K + iy;

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

        // Fused B-loading: silu(gate) * up
        {
            const short sx = short(tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            float4 gv0 = *(device float4*)(g_ptr);
            float4 gv1 = *(device float4*)(g_ptr + 4);
            float4 uv0 = *(device float4*)(u_ptr);
            float4 uv1 = *(device float4*)(u_ptr + 4);
            half4 s0 = half4(
                silu_f_batch(gv0[0]) * uv0[0],
                silu_f_batch(gv0[1]) * uv0[1],
                silu_f_batch(gv0[2]) * uv0[2],
                silu_f_batch(gv0[3]) * uv0[3]);
            half4 s1 = half4(
                silu_f_batch(gv1[0]) * uv1[0],
                silu_f_batch(gv1[1]) * uv1[1],
                silu_f_batch(gv1[2]) * uv1[2],
                silu_f_batch(gv1[3]) * uv1[3]);
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = half2x4(s0, s1);
        }

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;
        g_ptr += NK;
        u_ptr += NK;

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

kernel void dequant_batch_q6_k_blocked_gelu(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* gate   [[buffer(1)]],
    device const float* up     [[buffer(2)]],
    device float* C            [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& N           [[buffer(5)]],
    constant uint& K           [[buffer(6)]],
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
    device const Q6_K_Block* x =
        q6_k_row_ptr(A, uint(r0 + lr0), blocks_per_row) + offset1;

    const short iy = short(8 * (tiitg % NL1));
    device const float* g_ptr = gate + uint(r1 + lr1) * K + iy;
    device const float* u_ptr = up   + uint(r1 + lr1) * K + iy;

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

        // Match the tanh-approx GELU used by the standalone batch activation
        // path and the decode fused GELU-down kernels.
        {
            const short sx = short(tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            float4 gv0 = *(device float4*)(g_ptr);
            float4 gv1 = *(device float4*)(g_ptr + 4);
            float4 uv0 = *(device float4*)(u_ptr);
            float4 uv1 = *(device float4*)(u_ptr + 4);
            half4 s0 = half4(
                gelu_mul_f32(gv0[0], uv0[0]),
                gelu_mul_f32(gv0[1], uv0[1]),
                gelu_mul_f32(gv0[2], uv0[2]),
                gelu_mul_f32(gv0[3], uv0[3]));
            half4 s1 = half4(
                gelu_mul_f32(gv1[0], uv1[0]),
                gelu_mul_f32(gv1[1], uv1[1]),
                gelu_mul_f32(gv1[2], uv1[2]),
                gelu_mul_f32(gv1[3], uv1[3]));
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = half2x4(s0, s1);
        }

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;
        g_ptr += NK;
        u_ptr += NK;

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

    threadgroup half tg_A[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];
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
                device const Q6_K_Block& blk =
                    q6_k_block_ref(q6_k_row_ptr(A, global_r, blocks_per_row), block_idx);
                tg_A[r * DB_BK + c] = half(q6_k_dequant_pair_value(blk, group, sub_pair, c));
            } else {
                tg_A[r * DB_BK + c] = half(0.0f);
            }
        }

        // Cooperative load B tile
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
    device const Q6_K_Block* x =
        q6_k_row_ptr(A, uint(r0 + lr0), blocks_per_row) + offset1;

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

// ── Q6_K MoE expert mul_mat_id (blocked, f32 input) ──
//
// Uses the same routed layout as the Q4_K/Q5_K/Q8_0 MoE kernels:
// `tpe`/`hids` compact tokens per active expert, then write back into the flat
// [token_slot, M] output using the routed `hid`.
kernel void moe_mul_mat_id_q6_k_blocked(
    device const Q6_K_Block *weights   [[buffer(0)]],
    device const float *input          [[buffer(1)]],
    device const uint32_t *tpe         [[buffer(2)]],
    device const int32_t *hids         [[buffer(3)]],
    device float *output               [[buffer(4)]],
    constant uint &M                   [[buffer(5)]],
    constant uint &K                   [[buffer(6)]],
    constant uint &n_tokens            [[buffer(7)]],
    constant uint &n_expert_used       [[buffer(8)]],
    constant uint &weight_stride       [[buffer(9)]],
    device const uint32_t *active_meta [[buffer(10)]],
    constant uint &input_is_hid        [[buffer(11)]],
    threadgroup char *shmem            [[threadgroup(0)]],
    uint3 tgpig  [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint active_count = active_meta[0];
    if (tgpig.z >= active_count) return;
    device const uint32_t *active_experts = active_meta + 1;
    const uint expert = active_experts[tgpig.z];
    const uint n_assigned = tpe[expert];

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short QK_NL = 16;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    if (n_assigned == 0 || (uint)r1 >= n_assigned) return;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = short(min(uint(NR1), n_assigned - uint(r1)));
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    short il = il0;

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    device const Q6_K_Block *W = q6_k_advance_blocks(weights, ulong(expert) * ulong(weight_stride));
    const short offset1 = il0 / QK_NL;
    device const Q6_K_Block *x = W + uint(r0 + lr0) * blocks_per_row + offset1;

    device const int32_t *expert_hids = hids + expert * n_tokens;
    int32_t hid_for_lr1 = (r1 + lr1 < (int)n_assigned) ? expert_hids[r1 + lr1] : 0;
    uint input_row = input_is_hid != 0 ? uint(hid_for_lr1) : uint(hid_for_lr1) / n_expert_used;
    const short iy = short(8 * (tiitg % NL1));
    device const float *y = input + input_row * K + iy;

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
        *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
            (half2x4)(*(device float2x4 *)y);

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1) / QK_NL : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);

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

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *temp_str = ((threadgroup float *)shmem)
        + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                        NR0, ulong2(0, 0), false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort lane = tiitg % 32;
    for (short j = sgitg; j < nr1; j += 4) {
        int32_t hid = expert_hids[r1 + j];
        device float  *D  = output + hid * M + uint(r0);
        device float4 *D4 = (device float4 *)D;
        threadgroup float  *Cs  = ((threadgroup float *)shmem) + j * NR0;
        threadgroup float4 *C4 = (threadgroup float4 *)Cs;

        int i = lane;
        for (; i < nr0 / 4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }
        i = (4 * (nr0 / 4)) + lane;
        for (; i < nr0; i += 32) {
            *(D + i) = *(Cs + i);
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
                device const Q6_K_Block& blk =
                    q6_k_block_ref(q6_k_row_ptr(A, global_r, blocks_per_row), block_idx);
                tg_A[r * DB_BK + c] = half(q6_k_dequant_pair_value(blk, group, sub_pair, c));
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
            device const Q6_K_Block& blk =
                q6_k_block_ref(q6_k_row_ptr(A, global_r, blocks_per_row), block_idx);
            tg_A[r * DB_BK + c] = half(q6_k_dequant_pair_value(blk, group, sub_pair, c));
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
            device const Q6_K_Block& blk =
                q6_k_block_ref(q6_k_row_ptr(A, global_r, blocks_per_row), block_idx);
            tg_A[r * D64_BK + c] = half(q6_k_dequant_pair_value(blk, group, sub_pair, c));
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
            device const Q6_K_Block& blk =
                q6_k_block_ref(q6_k_row_ptr(A, global_r, blocks_per_row), block_idx);
            tg_A[r * D64_BK + c] = half(q6_k_dequant_pair_value(blk, group, sub_pair, c));
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
            device const Q6_K_Block& blk =
                q6_k_block_ref(q6_k_row_ptr(A, global_r, blocks_per_row), block_idx);
            tg_A[r * D64_BK + c] = half(q6_k_dequant_pair_value(blk, group, sub_pair, c));
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

    threadgroup half tg_A0[PB_BM * PB_BK];
    threadgroup half tg_A1[PB_BM * PB_BK];
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
                tg_A0[r * PB_BK + c] = half(q6_k_dequant_pair_value(blk0, group, sub_pair, c));
                tg_A1[r * PB_BK + c] = half(q6_k_dequant_pair_value(blk1, group, sub_pair, c));
            } else {
                tg_A0[r * PB_BK + c] = half(0.0f);
                tg_A1[r * PB_BK + c] = half(0.0f);
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

            simdgroup_half8x8 a00, a01, a02, a03;
            simdgroup_half8x8 a10, a11, a12, a13;
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
                tg_A0[r * P16_BK + c] = half(q6_k_dequant_pair_value(blk0, group, sub_pair, c));
                tg_A1[r * P16_BK + c] = half(q6_k_dequant_pair_value(blk1, group, sub_pair, c));
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
            device const Q6_K_Block& blk =
                q6_k_block_ref(q6_k_row_ptr(A, global_r, blocks_per_row), block_idx);
                tg_A[r * SB_BK + c] = half(q6_k_dequant_pair_value(blk, group, sub_pair, c));
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
                device const Q6_K_Block& blk =
                    q6_k_block_ref(q6_k_row_ptr(A, global_r, blocks_per_row), block_idx);
                tg_A[r * SB_BK + c] = half(q6_k_dequant_pair_value(blk, group, sub_pair, c));
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
    device const Q6_K_Block* a_row = q6_k_row_ptr(A, row, blocks_per_row);
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

        device const Q6_K_Block& blk = q6_k_block_ref(a_row, b);
        float d = (simd_lane == 0) ? float(blk.d) : 0.0f;
        d = simd_broadcast(d, 0);
        device const uchar* ql     = blk.ql;
        device const uchar* qh     = blk.qh;
        device const char*  scales = blk.scales;

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

kernel void dequant_matvec_pair_q6_k(
    device const Q6_K_Block* A0 [[buffer(0)]],
    device const Q6_K_Block* A1 [[buffer(1)]],
    device const float* x       [[buffer(2)]],
    device float* y0            [[buffer(3)]],
    device float* y1            [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& K            [[buffer(6)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    device const Q6_K_Block* a_row0 = q6_k_row_ptr(A0, row, blocks_per_row);
    device const Q6_K_Block* a_row1 = q6_k_row_ptr(A1, row, blocks_per_row);
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        sum0 += q6_k_block_dot(a_row0, b, x, simd_lane);
        sum1 += q6_k_block_dot(a_row1, b, x, simd_lane);
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);

    threadgroup float simd_sums0[num_simd_groups];
    threadgroup float simd_sums1[num_simd_groups];
    if (simd_lane == 0) {
        simd_sums0[simd_id] = sum0;
        simd_sums1[simd_id] = sum1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        sum0 = (simd_lane < num_simd_groups) ? simd_sums0[simd_lane] : 0.0f;
        sum1 = (simd_lane < num_simd_groups) ? simd_sums1[simd_lane] : 0.0f;
        sum0 = simd_sum(sum0);
        sum1 = simd_sum(sum1);
        if (simd_lane == 0) {
            y0[row] = sum0;
            y1[row] = sum1;
        }
    }
}

kernel void dequant_matvec_silu_down_q6_k(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* gate   [[buffer(1)]],
    device const float* up     [[buffer(2)]],
    device float* y            [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    device const Q6_K_Block* a_row = q6_k_row_ptr(A, row, blocks_per_row);
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;
    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        sum += q6_k_block_dot_silu(a_row, b, gate, up, simd_lane);
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

kernel void dequant_matvec_gelu_down_q6_k(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* gate   [[buffer(1)]],
    device const float* up     [[buffer(2)]],
    device float* y            [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint first_row = row * 4 + simd_id * 2;
    if (first_row >= M) return;

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    bool valid1 = (first_row + 1) < M;
    device const Q6_K_Block* row0 = q6_k_row_ptr(A, first_row, blocks_per_row);
    device const Q6_K_Block* row1 = valid1 ? q6_k_row_ptr(A, first_row + 1, blocks_per_row) : row0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        sum0 += q6_k_block_dot_gelu(row0, b, gate, up, simd_lane);
        if (valid1) {
            sum1 += q6_k_block_dot_gelu(row1, b, gate, up, simd_lane);
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        y[first_row] = sum0;
        if (valid1) {
            y[first_row + 1] = sum1;
        }
    }
}

kernel void moe_mul_mat_selected_q6_k_matvec(
    device const Q6_K_Block* weights   [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device const int32_t* selected     [[buffer(2)]],
    device float* output               [[buffer(3)]],
    constant uint& M                   [[buffer(4)]],
    constant uint& K                   [[buffer(5)]],
    constant uint& n_selected          [[buffer(6)]],
    constant uint& weight_stride       [[buffer(7)]],
    constant uint& input_is_slot_major [[buffer(8)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (tgpig.z >= n_selected || tgpig.x >= M) return;

    const uint slot = tgpig.z;
    const uint expert = uint(selected[slot]);
    const uint row = tgpig.x;
    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    const uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;
    threadgroup float simd_sums[num_simd_groups];

    device const Q6_K_Block* expert_base =
        q6_k_advance_blocks(weights, ulong(expert) * ulong(weight_stride));
    device const Q6_K_Block* a_row = q6_k_row_ptr(expert_base, row, blocks_per_row);
    device const float* x = input + input_row * K;

    float sum = 0.0f;
    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        sum += q6_k_block_dot(a_row, b, x, simd_lane);
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float row_sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
        row_sum = simd_sum(row_sum);
        if (simd_lane == 0) {
            output[slot * M + row] = row_sum;
        }
    }
}

kernel void moe_mul_mat_selected_q6_k_matvec_nr2(
    device const Q6_K_Block* weights   [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device const int32_t* selected     [[buffer(2)]],
    device float* output               [[buffer(3)]],
    constant uint& M                   [[buffer(4)]],
    constant uint& K                   [[buffer(5)]],
    constant uint& n_selected          [[buffer(6)]],
    constant uint& weight_stride       [[buffer(7)]],
    constant uint& input_is_slot_major [[buffer(8)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (tgpig.z >= n_selected) return;

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint slot = tgpig.z;
    const uint expert = uint(selected[slot]);
    const uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    const uint first_row = (tgpig.x * Q6K_NR2_SG_PER_TG + simd_id) * Q6K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    const bool valid1 = (first_row + 1) < M;
    device const Q6_K_Block* base =
        q6_k_advance_blocks(weights, ulong(expert) * ulong(weight_stride));
    device const Q6_K_Block* row0 = q6_k_row_ptr(base, first_row, blocks_per_row);
    device const Q6_K_Block* row1 =
        valid1 ? q6_k_row_ptr(base, first_row + 1, blocks_per_row) : row0;
    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    device const float* x = input + input_row * K;

    float sumf[Q6K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};
    float yl[16];

    ushort tid = simd_lane / 2;
    ushort ix = simd_lane % 2;
    ushort ip = tid / 8;
    ushort il = tid % 8;
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
            device const Q6_K_Block& blk = q6_k_block_ref(row_ptr, ib);
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
        device float* y_slot = output + slot * M;
        y_slot[first_row] = sum0;
        if (valid1) {
            y_slot[first_row + 1] = sum1;
        }
    }
}

kernel void moe_mul_mat_selected_pair_q6_k_matvec(
    device const Q6_K_Block* weights0  [[buffer(0)]],
    device const Q6_K_Block* weights1  [[buffer(1)]],
    device const float* input          [[buffer(2)]],
    device const int32_t* selected     [[buffer(3)]],
    device float* output0              [[buffer(4)]],
    device float* output1              [[buffer(5)]],
    constant uint& M                   [[buffer(6)]],
    constant uint& K                   [[buffer(7)]],
    constant uint& n_selected          [[buffer(8)]],
    constant uint& weight_stride0      [[buffer(9)]],
    constant uint& weight_stride1      [[buffer(10)]],
    constant uint& input_is_slot_major [[buffer(11)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (tgpig.z >= n_selected || tgpig.x >= M) return;

    const uint slot = tgpig.z;
    const uint expert = uint(selected[slot]);
    const uint row = tgpig.x;
    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    const uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;
    threadgroup float simd_sums0[num_simd_groups];
    threadgroup float simd_sums1[num_simd_groups];

    device const Q6_K_Block* a_row0 =
        weights0 + expert * weight_stride0 + row * blocks_per_row;
    device const Q6_K_Block* a_row1 =
        weights1 + expert * weight_stride1 + row * blocks_per_row;
    device const float* x = input + input_row * K;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        sum0 += q6_k_block_dot(a_row0, b, x, simd_lane);
        sum1 += q6_k_block_dot(a_row1, b, x, simd_lane);
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        simd_sums0[simd_id] = sum0;
        simd_sums1[simd_id] = sum1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float row_sum0 = (simd_lane < num_simd_groups) ? simd_sums0[simd_lane] : 0.0f;
        float row_sum1 = (simd_lane < num_simd_groups) ? simd_sums1[simd_lane] : 0.0f;
        row_sum0 = simd_sum(row_sum0);
        row_sum1 = simd_sum(row_sum1);
        if (simd_lane == 0) {
            output0[slot * M + row] = row_sum0;
            output1[slot * M + row] = row_sum1;
        }
    }
}

kernel void moe_mul_mat_selected_pair_q6_k_matvec_ilp4(
    device const Q6_K_Block* weights0  [[buffer(0)]],
    device const Q6_K_Block* weights1  [[buffer(1)]],
    device const float* input          [[buffer(2)]],
    device const int32_t* selected     [[buffer(3)]],
    device float* output0              [[buffer(4)]],
    device float* output1              [[buffer(5)]],
    constant uint& M                   [[buffer(6)]],
    constant uint& K                   [[buffer(7)]],
    constant uint& n_selected          [[buffer(8)]],
    constant uint& weight_stride0      [[buffer(9)]],
    constant uint& weight_stride1      [[buffer(10)]],
    constant uint& input_is_slot_major [[buffer(11)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (tgpig.z >= n_selected) return;

    const uint slot = tgpig.z;
    const uint expert = uint(selected[slot]);
    const uint first_row = tgpig.x * 2 + simd_id;
    if (first_row >= M) return;

    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    const uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    device const Q6_K_Block* row0 =
        weights0 + expert * weight_stride0 + first_row * blocks_per_row;
    device const Q6_K_Block* row1 =
        weights1 + expert * weight_stride1 + first_row * blocks_per_row;
    device const float* x = input + input_row * K;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    ushort ix = simd_lane % 2;

    for (uint ib = ix; ib < blocks_per_row; ib += 2) {
        device const float* x_block = x + ib * Q6_K_BLOCK_VALUES;
        sum0 += q6_k_block_dot_ilp2(row0[ib], x_block, simd_lane);
        sum1 += q6_k_block_dot_ilp2(row1[ib], x_block, simd_lane);
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        output0[slot * M + first_row] = sum0;
        output1[slot * M + first_row] = sum1;
    }
}

kernel void moe_mul_mat_selected_pair_q6_k_matvec_nr2(
    device const Q6_K_Block* weights0  [[buffer(0)]],
    device const Q6_K_Block* weights1  [[buffer(1)]],
    device const float* input          [[buffer(2)]],
    device const int32_t* selected     [[buffer(3)]],
    device float* output0              [[buffer(4)]],
    device float* output1              [[buffer(5)]],
    constant uint& M                   [[buffer(6)]],
    constant uint& K                   [[buffer(7)]],
    constant uint& n_selected          [[buffer(8)]],
    constant uint& weight_stride0      [[buffer(9)]],
    constant uint& weight_stride1      [[buffer(10)]],
    constant uint& input_is_slot_major [[buffer(11)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (tgpig.z >= n_selected) return;

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint slot = tgpig.z;
    const uint expert = uint(selected[slot]);
    const uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    const uint first_row = (tgpig.x * Q6K_NR2_SG_PER_TG + simd_id) * Q6K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    const bool valid1 = (first_row + 1) < M;
    device const Q6_K_Block* base0 = weights0 + expert * weight_stride0;
    device const Q6_K_Block* row00 = base0 + first_row * blocks_per_row;
    device const Q6_K_Block* row01 = valid1 ? row00 + blocks_per_row : row00;
    device const Q6_K_Block* base1 = weights1 + expert * weight_stride1;
    device const Q6_K_Block* row10 = base1 + first_row * blocks_per_row;
    device const Q6_K_Block* row11 = valid1 ? row10 + blocks_per_row : row10;
    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    device const float* x = input + input_row * K;

    float sumf0[Q6K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};
    float sumf1[Q6K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};
    float yl[16];

    ushort tid = simd_lane / 2;
    ushort ix = simd_lane % 2;
    ushort ip = tid / 8;
    ushort il = tid % 8;
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
            device const Q6_K_Block& blk0 = ((row == 0) ? row00 : row01)[ib];
            device const Q6_K_Block& blk1 = ((row == 0) ? row10 : row11)[ib];

            device const uchar* q01 = blk0.ql + q_offset_l;
            device const uchar* q02 = q01 + 32;
            device const uchar* q0h = blk0.qh + q_offset_h;
            device const char* sc0 = blk0.scales + is;

            device const uchar* q11 = blk1.ql + q_offset_l;
            device const uchar* q12 = q11 + 32;
            device const uchar* q1h = blk1.qh + q_offset_h;
            device const char* sc1 = blk1.scales + is;

            float4 sums0 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 sums1 = {0.0f, 0.0f, 0.0f, 0.0f};
            FOR_UNROLL (ushort l = 0; l < 4; ++l) {
                sums0[0] += yl[4 * l + 0] * ((int8_t)((q01[l] & 0xF) | ((q0h[l] & kmask1) << 4)) - 32);
                sums0[1] += yl[4 * l + 1] * ((int8_t)((q02[l] & 0xF) | ((q0h[l] & kmask2) << 2)) - 32);
                sums0[2] += yl[4 * l + 2] * ((int8_t)((q01[l]  >> 4) | ((q0h[l] & kmask3) << 0)) - 32);
                sums0[3] += yl[4 * l + 3] * ((int8_t)((q02[l]  >> 4) | ((q0h[l] & kmask4) >> 2)) - 32);

                sums1[0] += yl[4 * l + 0] * ((int8_t)((q11[l] & 0xF) | ((q1h[l] & kmask1) << 4)) - 32);
                sums1[1] += yl[4 * l + 1] * ((int8_t)((q12[l] & 0xF) | ((q1h[l] & kmask2) << 2)) - 32);
                sums1[2] += yl[4 * l + 2] * ((int8_t)((q11[l]  >> 4) | ((q1h[l] & kmask3) << 0)) - 32);
                sums1[3] += yl[4 * l + 3] * ((int8_t)((q12[l]  >> 4) | ((q1h[l] & kmask4) >> 2)) - 32);
            }

            sumf0[row] += float(blk0.d) * (
                sums0[0] * float(sc0[0]) +
                sums0[1] * float(sc0[2]) +
                sums0[2] * float(sc0[4]) +
                sums0[3] * float(sc0[6])
            );
            sumf1[row] += float(blk1.d) * (
                sums1[0] * float(sc1[0]) +
                sums1[1] * float(sc1[2]) +
                sums1[2] * float(sc1[4]) +
                sums1[3] * float(sc1[6])
            );
        }
    }

    float sum00 = simd_sum(sumf0[0]);
    float sum01 = simd_sum(sumf0[1]);
    float sum10 = simd_sum(sumf1[0]);
    float sum11 = simd_sum(sumf1[1]);
    if (simd_lane == 0) {
        device float* y_slot0 = output0 + slot * M;
        device float* y_slot1 = output1 + slot * M;
        y_slot0[first_row] = sum00;
        y_slot1[first_row] = sum10;
        if (valid1) {
            y_slot0[first_row + 1] = sum01;
            y_slot1[first_row + 1] = sum11;
        }
    }
}

kernel void moe_mul_mat_selected_weighted_q6_k_matvec(
    device const Q6_K_Block* weights   [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device const int32_t* selected     [[buffer(2)]],
    device const float* expert_weights [[buffer(3)]],
    device float* output               [[buffer(4)]],
    constant uint& M                   [[buffer(5)]],
    constant uint& K                   [[buffer(6)]],
    constant uint& n_selected          [[buffer(7)]],
    constant uint& weight_stride       [[buffer(8)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;
    threadgroup float simd_sums[num_simd_groups];
    float acc = 0.0f;

    for (uint slot = 0; slot < n_selected; ++slot) {
        const uint expert = uint(selected[slot]);
        const float route_weight = expert_weights[slot];
        device const Q6_K_Block* expert_base =
            q6_k_advance_blocks(weights, ulong(expert) * ulong(weight_stride));
        device const Q6_K_Block* a_row = q6_k_row_ptr(expert_base, row, blocks_per_row);
        device const float* x = input + slot * K;

        float sum = 0.0f;
        for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
            sum += q6_k_block_dot(a_row, b, x, simd_lane);
        }

        sum = simd_sum(sum);
        if (simd_lane == 0) {
            simd_sums[simd_id] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            float slot_sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
            slot_sum = simd_sum(slot_sum);
            if (simd_lane == 0) {
                acc += route_weight * slot_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_id == 0 && simd_lane == 0) {
        output[row] = acc;
    }
}

kernel void moe_mul_mat_selected_weighted_q6_k_matvec_nr2(
    device const Q6_K_Block* weights   [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device const int32_t* selected     [[buffer(2)]],
    device const float* expert_weights [[buffer(3)]],
    device float* output               [[buffer(4)]],
    constant uint& M                   [[buffer(5)]],
    constant uint& K                   [[buffer(6)]],
    constant uint& n_selected          [[buffer(7)]],
    constant uint& weight_stride       [[buffer(8)]],
    uint tg_id                         [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    const uint first_row = (tg_id * Q6K_NR2_SG_PER_TG + simd_id) * Q6K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    const bool valid1 = (first_row + 1) < M;
    float acc[Q6K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};
    float yl[16];

    ushort tid = simd_lane / 2;
    ushort ix = simd_lane % 2;
    ushort ip = tid / 8;
    ushort il = tid % 8;
    ushort l0 = 4 * il;
    ushort is = 8 * ip + l0 / 16;

    ushort y_offset = 128 * ip + l0;
    ushort q_offset_l = 64 * ip + l0;
    ushort q_offset_h = 32 * ip + l0;

    for (uint slot = 0; slot < n_selected; ++slot) {
        const uint expert = uint(selected[slot]);
        const float route_weight = expert_weights[slot];
        device const Q6_K_Block* base =
            q6_k_advance_blocks(weights, ulong(expert) * ulong(weight_stride));
        device const Q6_K_Block* row0 = q6_k_row_ptr(base, first_row, blocks_per_row);
        device const Q6_K_Block* row1 =
            valid1 ? q6_k_row_ptr(base, first_row + 1, blocks_per_row) : row0;
        device const float* x = input + slot * K;

        float sumf[Q6K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};

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
                device const Q6_K_Block& blk = q6_k_block_ref(row_ptr, ib);
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
            acc[0] += route_weight * sum0;
            if (valid1) {
                acc[1] += route_weight * sum1;
            }
        }
    }

    if (simd_lane == 0) {
        output[first_row] = acc[0];
        if (valid1) {
            output[first_row + 1] = acc[1];
        }
    }
}

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
    device const Q6_K_Block* row0 = q6_k_row_ptr(A, first_row, blocks_per_row);
    device const Q6_K_Block* row1 = valid1 ? q6_k_row_ptr(A, first_row + 1, blocks_per_row) : row0;

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
            device const Q6_K_Block& blk = q6_k_block_ref(row_ptr, ib);
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

kernel void dequant_matvec_q6_k_ilp4(
    device const Q6_K_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    uint first_row = tg_id * 2 + simd_id;
    if (first_row >= M) return;

    device const Q6_K_Block* a_row = q6_k_row_ptr(A, first_row, blocks_per_row);

    float sumf = 0.0f;
    ushort ix = simd_lane % 2;

    for (uint ib = ix; ib < blocks_per_row; ib += 2) {
        sumf += q6_k_block_dot_ilp2(
            q6_k_block_ref(a_row, ib),
            x + ib * Q6_K_BLOCK_VALUES,
            simd_lane
        );
    }

    float total = simd_sum(sumf);
    if (simd_lane == 0) {
        y[first_row] = total;
    }
}

// ── Q6_K MoE mul_mat_id (f32 tiles) ─────────────────────────────
//
// Uses the same Q4_K f32-tile approach: dequant A into f32 threadgroup
// memory, load B as f32. Avoids half-precision overflow when the MoE
// SiLU*up input has large magnitude (>65504).
kernel void moe_mul_mat_id_q6_k(
    device const Q6_K_Block *weights   [[buffer(0)]],
    device const float *input          [[buffer(1)]],
    device const uint32_t *tpe         [[buffer(2)]],
    device const int32_t *hids         [[buffer(3)]],
    device float *output               [[buffer(4)]],
    constant uint &M                   [[buffer(5)]],
    constant uint &K                   [[buffer(6)]],
    constant uint &n_tokens            [[buffer(7)]],
    constant uint &n_expert_used       [[buffer(8)]],
    constant uint &weight_stride       [[buffer(9)]],
    device const uint32_t *active_meta [[buffer(10)]],
    constant uint &input_is_hid        [[buffer(11)]],
    uint3 group_id  [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]])
{
    const uint active_count = active_meta[0];
    if (group_id.z >= active_count) return;
    device const uint32_t *active_experts = active_meta + 1;
    const uint expert = active_experts[group_id.z];
    const uint n_assigned = tpe[expert];
    const uint tile_row = group_id.y * DQ_BM;
    const uint tile_col = group_id.x * DQ_BN;

    if (n_assigned == 0 || tile_col >= n_assigned) return;

    const uint blocks_per_row = K / Q6_K_BLOCK_VALUES;
    device const Q6_K_Block *W = q6_k_advance_blocks(weights, ulong(expert) * ulong(weight_stride));
    device const int32_t *expert_hids = hids + expert * n_tokens;

    threadgroup float tg_A[DQ_BM * DQ_BK];
    threadgroup float tg_B[DQ_BK * DQ_BN];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    for (uint kt = 0; kt < K; kt += DQ_BK) {
        uint block_idx = kt / Q6_K_BLOCK_VALUES;
        uint in_block = kt % Q6_K_BLOCK_VALUES;
        uint group = in_block / 128;
        uint sub_pair = (in_block % 128) / 64;

        // Load A tile: dequant Q6_K weight rows into f32
        for (uint i = tid; i < DQ_BM * DQ_BK; i += DQ_TG) {
            uint r = i / DQ_BK;
            uint c = i % DQ_BK;
            uint global_r = tile_row + r;

            if (global_r < M) {
                device const Q6_K_Block& blk =
                    q6_k_block_ref(q6_k_row_ptr(W, global_r, blocks_per_row), block_idx);
                tg_A[r * DQ_BK + c] = q6_k_dequant_pair_value(blk, group, sub_pair, c);
            } else {
                tg_A[r * DQ_BK + c] = 0.0f;
            }
        }

        // Load B tile: f32 input rows via routing
        for (uint i = tid; i < DQ_BK * DQ_BN; i += DQ_TG) {
            uint r = i / DQ_BN;
            uint c = i % DQ_BN;
            uint slot = tile_col + c;
            uint global_k = kt + r;

            if (slot < n_assigned && global_k < K) {
                int32_t hid = expert_hids[slot];
                uint input_row = input_is_hid != 0 ? uint(hid) : uint(hid) / n_expert_used;
                tg_B[r * DQ_BN + c] = input[input_row * K + global_k];
            } else {
                tg_B[r * DQ_BN + c] = 0.0f;
            }
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
        uint slot = tile_col + c;

        if (gr < M && slot < n_assigned) {
            int32_t hid = expert_hids[slot];
            output[hid * M + gr] = out_tile[r * DQ_BN + c];
        }
    }
}
