// Q8_0 dequant and matvec kernels.
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

kernel void dequant_batch_q8_0(
    device const Q8_0_Block* A [[buffer(0)]],
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

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        // Cooperative dequant A tile
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

kernel void dequant_batch_q8_0_f16in_small(
    device const Q8_0_Block* A [[buffer(0)]],
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

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += SB_BK) {
        for (uint i = tid; i < SB_BM * SB_BK; i += SB_TG) {
            uint r = i / SB_BK;
            uint c = i % SB_BK;
            uint global_r = tile_m + r;
            uint global_k = kt + c;

            if (global_r < M && global_k < K) {
                uint block_idx = global_k / Q8_0_BLOCK_VALUES;
                uint in_block = global_k % Q8_0_BLOCK_VALUES;
                device const Q8_0_Block& blk = A[global_r * blocks_per_row + block_idx];
                float d = float(blk.d);
                int q = int(blk.qs[in_block]);
                tg_A[r * SB_BK + c] = half(d * float(q));
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

kernel void dequant_matvec_pair_q8_0(
    device const Q8_0_Block* A0 [[buffer(0)]],
    device const Q8_0_Block* A1 [[buffer(1)]],
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

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    device const Q8_0_Block* a_row0 = A0 + row * blocks_per_row;
    device const Q8_0_Block* a_row1 = A1 + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        float d0 = (simd_lane == 0) ? float(a_row0[b].d) : 0.0f;
        float d1 = (simd_lane == 0) ? float(a_row1[b].d) : 0.0f;
        d0 = simd_broadcast(d0, 0);
        d1 = simd_broadcast(d1, 0);

        uint base = b * Q8_0_BLOCK_VALUES;
        float xv = x[base + simd_lane];
        sum0 += d0 * float(int(a_row0[b].qs[simd_lane])) * xv;
        sum1 += d1 * float(int(a_row1[b].qs[simd_lane])) * xv;
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

kernel void dequant_matvec_silu_down_q8_0(
    device const Q8_0_Block* A [[buffer(0)]],
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

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    device const Q8_0_Block* a_row = A + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
        d = simd_broadcast(d, 0);

        uint base = b * Q8_0_BLOCK_VALUES;
        float xv = silu_mul_f32(gate[base + simd_lane], up[base + simd_lane]);
        sum += d * float(int(a_row[b].qs[simd_lane])) * xv;
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

kernel void dequant_matvec_gelu_down_q8_0(
    device const Q8_0_Block* A [[buffer(0)]],
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

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    device const Q8_0_Block* a_row = A + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
        d = simd_broadcast(d, 0);

        uint base = b * Q8_0_BLOCK_VALUES;
        float xv = gelu_mul_f32(gate[base + simd_lane], up[base + simd_lane]);
        sum += d * float(int(a_row[b].qs[simd_lane])) * xv;
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

kernel void moe_mul_mat_selected_q8_0_matvec(
    device const Q8_0_Block* weights   [[buffer(0)]],
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
    const uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;
    threadgroup float simd_sums[num_simd_groups];

    device const Q8_0_Block* a_row =
        weights + expert * weight_stride + row * blocks_per_row;
    device const float* x = input + input_row * K;

    float sum = 0.0f;
    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
        d = simd_broadcast(d, 0);

        uint base = b * Q8_0_BLOCK_VALUES;
        float xv = x[base + simd_lane];
        sum += d * float(int(a_row[b].qs[simd_lane])) * xv;
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

kernel void moe_mul_mat_selected_q8_0_matvec_nr2(
    device const Q8_0_Block* weights   [[buffer(0)]],
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

    const uint slot = tgpig.z;
    const uint expert = uint(selected[slot]);
    const uint first_row = (tgpig.x * Q8_0_NR2_SG_PER_TG + simd_id) * Q8_0_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    const uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    const bool valid1 = (first_row + 1) < M;
    device const Q8_0_Block* base = weights + expert * weight_stride;
    device const Q8_0_Block* row0 = base + first_row * blocks_per_row;
    device const Q8_0_Block* row1 = valid1 ? row0 + blocks_per_row : row0;
    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    device const float* x = input + input_row * K;

    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    for (uint ib = 0; ib < blocks_per_row; ib++) {
        float xv = x[ib * Q8_0_BLOCK_VALUES + simd_lane];

        float d0 = float(row0[ib].d);
        int q0 = int(row0[ib].qs[simd_lane]);
        sumf0 += d0 * float(q0) * xv;

        float d1 = float(row1[ib].d);
        int q1 = int(row1[ib].qs[simd_lane]);
        sumf1 += d1 * float(q1) * xv;
    }

    float sum0 = simd_sum(sumf0);
    float sum1 = simd_sum(sumf1);
    if (simd_lane == 0) {
        device float* y_slot = output + slot * M;
        y_slot[first_row] = sum0;
        if (valid1) {
            y_slot[first_row + 1] = sum1;
        }
    }
}

kernel void moe_mul_mat_selected_pair_q8_0_matvec(
    device const Q8_0_Block* weights0  [[buffer(0)]],
    device const Q8_0_Block* weights1  [[buffer(1)]],
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
    const uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;
    threadgroup float simd_sums0[num_simd_groups];
    threadgroup float simd_sums1[num_simd_groups];

    device const Q8_0_Block* a_row0 =
        weights0 + expert * weight_stride0 + row * blocks_per_row;
    device const Q8_0_Block* a_row1 =
        weights1 + expert * weight_stride1 + row * blocks_per_row;
    device const float* x = input + input_row * K;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        float d0 = (simd_lane == 0) ? float(a_row0[b].d) : 0.0f;
        float d1 = (simd_lane == 0) ? float(a_row1[b].d) : 0.0f;
        d0 = simd_broadcast(d0, 0);
        d1 = simd_broadcast(d1, 0);

        uint base = b * Q8_0_BLOCK_VALUES;
        float xv = x[base + simd_lane];
        sum0 += d0 * float(int(a_row0[b].qs[simd_lane])) * xv;
        sum1 += d1 * float(int(a_row1[b].qs[simd_lane])) * xv;
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

kernel void moe_mul_mat_selected_pair_q8_0_matvec_nr2(
    device const Q8_0_Block* weights0  [[buffer(0)]],
    device const Q8_0_Block* weights1  [[buffer(1)]],
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
    const uint first_row = (tgpig.x * Q8_0_NR2_SG_PER_TG + simd_id) * Q8_0_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    const uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    const bool valid1 = (first_row + 1) < M;
    device const Q8_0_Block* base0 = weights0 + expert * weight_stride0;
    device const Q8_0_Block* row00 = base0 + first_row * blocks_per_row;
    device const Q8_0_Block* row01 = valid1 ? row00 + blocks_per_row : row00;
    device const Q8_0_Block* base1 = weights1 + expert * weight_stride1;
    device const Q8_0_Block* row10 = base1 + first_row * blocks_per_row;
    device const Q8_0_Block* row11 = valid1 ? row10 + blocks_per_row : row10;
    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    device const float* x = input + input_row * K;

    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    for (uint ib = 0; ib < blocks_per_row; ib++) {
        float xv = x[ib * Q8_0_BLOCK_VALUES + simd_lane];

        sum00 += float(row00[ib].d) * float(int(row00[ib].qs[simd_lane])) * xv;
        sum01 += float(row01[ib].d) * float(int(row01[ib].qs[simd_lane])) * xv;
        sum10 += float(row10[ib].d) * float(int(row10[ib].qs[simd_lane])) * xv;
        sum11 += float(row11[ib].d) * float(int(row11[ib].qs[simd_lane])) * xv;
    }

    float row00_sum = simd_sum(sum00);
    float row01_sum = simd_sum(sum01);
    float row10_sum = simd_sum(sum10);
    float row11_sum = simd_sum(sum11);
    if (simd_lane == 0) {
        device float* y_slot0 = output0 + slot * M;
        device float* y_slot1 = output1 + slot * M;
        y_slot0[first_row] = row00_sum;
        y_slot1[first_row] = row10_sum;
        if (valid1) {
            y_slot0[first_row + 1] = row01_sum;
            y_slot1[first_row + 1] = row11_sum;
        }
    }
}

kernel void moe_mul_mat_selected_weighted_q8_0_matvec(
    device const Q8_0_Block* weights   [[buffer(0)]],
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

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;
    threadgroup float simd_sums[num_simd_groups];
    float acc = 0.0f;

    for (uint slot = 0; slot < n_selected; ++slot) {
        const uint expert = uint(selected[slot]);
        const float route_weight = expert_weights[slot];
        device const Q8_0_Block* a_row =
            weights + expert * weight_stride + row * blocks_per_row;
        device const float* x = input + slot * K;

        float sum = 0.0f;
        for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
            float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
            d = simd_broadcast(d, 0);

            uint base = b * Q8_0_BLOCK_VALUES;
            float xv = x[base + simd_lane];
            sum += d * float(int(a_row[b].qs[simd_lane])) * xv;
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

kernel void moe_mul_mat_selected_weighted_q8_0_matvec_nr2(
    device const Q8_0_Block* weights   [[buffer(0)]],
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
    const uint first_row = (tg_id * Q8_0_NR2_SG_PER_TG + simd_id) * Q8_0_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    const uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    const bool valid1 = (first_row + 1) < M;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint slot = 0; slot < n_selected; ++slot) {
        const uint expert = uint(selected[slot]);
        const float route_weight = expert_weights[slot];
        device const Q8_0_Block* base = weights + expert * weight_stride;
        device const Q8_0_Block* row0 = base + first_row * blocks_per_row;
        device const Q8_0_Block* row1 = valid1 ? row0 + blocks_per_row : row0;
        device const float* x = input + slot * K;

        float sumf0 = 0.0f;
        float sumf1 = 0.0f;
        for (uint ib = 0; ib < blocks_per_row; ib++) {
            float xv = x[ib * Q8_0_BLOCK_VALUES + simd_lane];
            sumf0 += float(row0[ib].d) * float(int(row0[ib].qs[simd_lane])) * xv;
            sumf1 += float(row1[ib].d) * float(int(row1[ib].qs[simd_lane])) * xv;
        }

        float slot_sum0 = simd_sum(sumf0);
        float slot_sum1 = simd_sum(sumf1);
        if (simd_lane == 0) {
            acc0 += route_weight * slot_sum0;
            if (valid1) {
                acc1 += route_weight * slot_sum1;
            }
        }
    }

    if (simd_lane == 0) {
        output[first_row] = acc0;
        if (valid1) {
            output[first_row + 1] = acc1;
        }
    }
}

kernel void dequant_matvec_q8_0_nr2(
    device const Q8_0_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    uint first_row = (tg_id * Q8_0_NR2_SG_PER_TG + simd_id) * Q8_0_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    bool valid1 = (first_row + 1) < M;
    device const Q8_0_Block* row0 = A + first_row * blocks_per_row;
    device const Q8_0_Block* row1 = valid1 ? row0 + blocks_per_row : row0;

    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    for (uint ib = 0; ib < blocks_per_row; ib++) {
        float xv = x[ib * Q8_0_BLOCK_VALUES + simd_lane];

        float d0 = float(row0[ib].d);
        int q0 = int(row0[ib].qs[simd_lane]);
        sumf0 += d0 * float(q0) * xv;

        float d1 = float(row1[ib].d);
        int q1 = int(row1[ib].qs[simd_lane]);
        sumf1 += d1 * float(q1) * xv;
    }

    float sum0 = simd_sum(sumf0);
    float sum1 = simd_sum(sumf1);
    if (simd_lane == 0) {
        y[first_row] = sum0;
        if (valid1) {
            y[first_row + 1] = sum1;
        }
    }
}

kernel void dequant_matvec_q8_0_ilp4(
    device const Q8_0_Block* A [[buffer(0)]],
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

    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    device const Q8_0_Block* a_row = A + first_row * blocks_per_row;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    uint ib = 0;
    // Process 4 blocks per iteration for ILP
    for (; ib + 3 < blocks_per_row; ib += 4) {
        float xv0 = x[(ib + 0) * Q8_0_BLOCK_VALUES + simd_lane];
        float xv1 = x[(ib + 1) * Q8_0_BLOCK_VALUES + simd_lane];
        float xv2 = x[(ib + 2) * Q8_0_BLOCK_VALUES + simd_lane];
        float xv3 = x[(ib + 3) * Q8_0_BLOCK_VALUES + simd_lane];

        sum0 += float(a_row[ib + 0].d) * float(int(a_row[ib + 0].qs[simd_lane])) * xv0;
        sum1 += float(a_row[ib + 1].d) * float(int(a_row[ib + 1].qs[simd_lane])) * xv1;
        sum2 += float(a_row[ib + 2].d) * float(int(a_row[ib + 2].qs[simd_lane])) * xv2;
        sum3 += float(a_row[ib + 3].d) * float(int(a_row[ib + 3].qs[simd_lane])) * xv3;
    }
    // Handle remaining blocks
    for (; ib < blocks_per_row; ib++) {
        float xv = x[ib * Q8_0_BLOCK_VALUES + simd_lane];
        sum0 += float(a_row[ib].d) * float(int(a_row[ib].qs[simd_lane])) * xv;
    }

    float total = simd_sum(sum0 + sum1 + sum2 + sum3);
    if (simd_lane == 0) {
        y[first_row] = total;
    }
}

// ── Q8_0 MoE expert mul_mat_id (blocked, f32 input) ──
//
// Dispatches per-expert batched matmul via routing tables.
// Grid: (ceil(n_assigned/32), ceil(M/64), n_active_experts)
// TG: 128 threads (4 simdgroups)
// Shared memory: 8192 bytes
kernel void moe_mul_mat_id_q8_0_blocked(
    device const Q8_0_Block *weights   [[buffer(0)]],
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

    // Weight pointer for this expert.
    uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    device const Q8_0_Block *W = weights + expert * weight_stride;
    device const Q8_0_Block *x = W + uint(r0 + lr0) * blocks_per_row;

    // Input pointer via routing.
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
        dequantize_q8_0_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        // B-tile: vectorized load.
        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
            (half2x4)(*(device float2x4 *)y);

        // Q8_0: one block covers 32 values = NK, advance by 1 block.
        il = 1 - il;
        x = (il == 0) ? x + 1 : x;
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

    // Output via routing index.
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

// ── Q8_0 MoE mul_mat_id (f32 tiles) ─────────────────────────────
//
// Uses the Q4_K f32-tile approach to avoid half-precision overflow
// when MoE SiLU*up input has large magnitude.
kernel void moe_mul_mat_id_q8_0(
    device const Q8_0_Block *weights   [[buffer(0)]],
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

    const uint blocks_per_row = K / Q8_0_BLOCK_VALUES;
    device const Q8_0_Block *W = weights + expert * weight_stride;
    device const int32_t *expert_hids = hids + expert * n_tokens;

    // Q8_0 has 32 values per block. DQ_BK=64 spans 2 blocks.
    threadgroup float tg_A[DQ_BM * DQ_BK];
    threadgroup float tg_B[DQ_BK * DQ_BN];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    for (uint kt = 0; kt < K; kt += DQ_BK) {
        // Load A tile: dequant Q8_0 weight rows into f32
        for (uint i = tid; i < DQ_BM * DQ_BK; i += DQ_TG) {
            uint r = i / DQ_BK;
            uint c = i % DQ_BK;
            uint global_r = tile_row + r;
            uint global_k = kt + c;

            if (global_r < M && global_k < K) {
                uint blk_idx = global_k / Q8_0_BLOCK_VALUES;
                uint in_blk = global_k % Q8_0_BLOCK_VALUES;
                device const Q8_0_Block& blk = W[global_r * blocks_per_row + blk_idx];
                tg_A[r * DQ_BK + c] = float(blk.d) * float(blk.qs[in_blk]);
            } else {
                tg_A[r * DQ_BK + c] = 0.0f;
            }
        }

        // Load B tile
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
