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

