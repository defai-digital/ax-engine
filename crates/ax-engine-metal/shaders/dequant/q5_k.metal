// Q5_K dequant and matvec kernels.
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

        const uchar shift_lo = uchar(pair * 2);
        const uchar shift_hi = uchar(pair * 2 + 1);
        for (uint i = tid; i < DB_BM * (DB_BK / 8); i += DB_TG) {
            uint r  = i / (DB_BK / 8);
            uint b4 = (i % (DB_BK / 8)) * 4;
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar4 qs_vec = *(device const uchar4*)(blk.qs + pair * 32 + b4);
                uchar4 qh_vec = *(device const uchar4*)(blk.qh + b4);
                uint4 lo = uint4(qs_vec & uchar4(0x0F));
                uint4 hi = uint4(qs_vec >> uchar4(4));
                lo |= uint4((qh_vec >> uchar4(shift_lo)) & uchar4(1)) << uint4(4);
                hi |= uint4((qh_vec >> uchar4(shift_hi)) & uchar4(1)) << uint4(4);
                float dsc1_f = float(row_dsc1[r]);
                float dmn1_f = float(row_dmin1[r]);
                float dsc2_f = float(row_dsc2[r]);
                float dmn2_f = float(row_dmin2[r]);
                *(threadgroup half4*)(tg_A + r * DB_BK + b4)      = half4(dsc1_f * float4(lo) - dmn1_f);
                *(threadgroup half4*)(tg_A + r * DB_BK + b4 + 32) = half4(dsc2_f * float4(hi) - dmn2_f);
            } else {
                *(threadgroup half4*)(tg_A + r * DB_BK + b4)      = half4(0.0h);
                *(threadgroup half4*)(tg_A + r * DB_BK + b4 + 32) = half4(0.0h);
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

        const uchar shift_lo = uchar(pair * 2);
        const uchar shift_hi = uchar(pair * 2 + 1);
        for (uint i = tid; i < DB_BM * (DB_BK / 8); i += DB_TG) {
            uint r  = i / (DB_BK / 8);
            uint b4 = (i % (DB_BK / 8)) * 4;
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar4 qs_vec = *(device const uchar4*)(blk.qs + pair * 32 + b4);
                uchar4 qh_vec = *(device const uchar4*)(blk.qh + b4);
                uint4 lo = uint4(qs_vec & uchar4(0x0F));
                uint4 hi = uint4(qs_vec >> uchar4(4));
                lo |= uint4((qh_vec >> uchar4(shift_lo)) & uchar4(1)) << uint4(4);
                hi |= uint4((qh_vec >> uchar4(shift_hi)) & uchar4(1)) << uint4(4);
                float dsc1_f = float(row_dsc1[r]);
                float dmn1_f = float(row_dmin1[r]);
                float dsc2_f = float(row_dsc2[r]);
                float dmn2_f = float(row_dmin2[r]);
                *(threadgroup half4*)(tg_A + r * DB_BK + b4)      = half4(dsc1_f * float4(lo) - dmn1_f);
                *(threadgroup half4*)(tg_A + r * DB_BK + b4 + 32) = half4(dsc2_f * float4(hi) - dmn2_f);
            } else {
                *(threadgroup half4*)(tg_A + r * DB_BK + b4)      = half4(0.0h);
                *(threadgroup half4*)(tg_A + r * DB_BK + b4 + 32) = half4(0.0h);
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

kernel void dequant_batch_q5_k_f16in_full64(
    device const Q5_K_Block* A [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& C_STRIDE    [[buffer(6)]],
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

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        uint block_idx = kt / Q5_K_BLOCK_VALUES;
        uint pair = (kt % Q5_K_BLOCK_VALUES) / D64_BK;

        // Phase 1: precompute d*scale and dmin*min.
        for (uint r = tid; r < D64_BM; r += D64_TG) {
            uint global_r = tile_m + r;
            device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
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

        // Phase 2: paired nibble + high-bit extraction.
        // Phase 2: vectorized uchar4 nibble + high-bit extraction (branchless).
        const uchar shift_lo = uchar(pair * 2);
        const uchar shift_hi = uchar(pair * 2 + 1);
        for (uint i = tid; i < D64_BM * (D64_BK / 8); i += D64_TG) {
            uint r  = i / (D64_BK / 8);
            uint b4 = (i % (D64_BK / 8)) * 4;
            uint global_r = tile_m + r;
            device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            uchar4 qs_vec = *(device const uchar4*)(blk.qs + pair * 32 + b4);
            uchar4 qh_vec = *(device const uchar4*)(blk.qh + b4);
            uint4 lo = uint4(qs_vec & uchar4(0x0F));
            uint4 hi = uint4(qs_vec >> uchar4(4));
            lo |= uint4((qh_vec >> uchar4(shift_lo)) & uchar4(1)) << uint4(4);
            hi |= uint4((qh_vec >> uchar4(shift_hi)) & uchar4(1)) << uint4(4);
            float dsc1_f = float(row_dsc1[r]);
            float dmn1_f = float(row_dmin1[r]);
            float dsc2_f = float(row_dsc2[r]);
            float dmn2_f = float(row_dmin2[r]);
            *(threadgroup half4*)(tg_A + r * D64_BK + b4)      = half4(dsc1_f * float4(lo) - dmn1_f);
            *(threadgroup half4*)(tg_A + r * D64_BK + b4 + 32) = half4(dsc2_f * float4(hi) - dmn2_f);
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

kernel void dequant_batch_q5_k_f16in_full32(
    device const Q5_K_Block* A [[buffer(0)]],
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

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        uint block_idx = kt / Q5_K_BLOCK_VALUES;
        uint pair = (kt % Q5_K_BLOCK_VALUES) / D64_BK;

        // Phase 1: precompute d*scale and dmin*min.
        for (uint r = tid; r < D32_BM; r += D32_TG) {
            uint global_r = tile_m + r;
            device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
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

        // Phase 2: vectorized uchar4 nibble + high-bit extraction (branchless).
        const uchar shift_lo = uchar(pair * 2);
        const uchar shift_hi = uchar(pair * 2 + 1);
        for (uint i = tid; i < D32_BM * (D64_BK / 8); i += D32_TG) {
            uint r  = i / (D64_BK / 8);
            uint b4 = (i % (D64_BK / 8)) * 4;
            uint global_r = tile_m + r;
            device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            uchar4 qs_vec = *(device const uchar4*)(blk.qs + pair * 32 + b4);
            uchar4 qh_vec = *(device const uchar4*)(blk.qh + b4);
            uint4 lo = uint4(qs_vec & uchar4(0x0F));
            uint4 hi = uint4(qs_vec >> uchar4(4));
            lo |= uint4((qh_vec >> uchar4(shift_lo)) & uchar4(1)) << uint4(4);
            hi |= uint4((qh_vec >> uchar4(shift_hi)) & uchar4(1)) << uint4(4);
            float dsc1_f = float(row_dsc1[r]);
            float dmn1_f = float(row_dmin1[r]);
            float dsc2_f = float(row_dsc2[r]);
            float dmn2_f = float(row_dmin2[r]);
            *(threadgroup half4*)(tg_A + r * D64_BK + b4)      = half4(dsc1_f * float4(lo) - dmn1_f);
            *(threadgroup half4*)(tg_A + r * D64_BK + b4 + 32) = half4(dsc2_f * float4(hi) - dmn2_f);
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

kernel void dequant_batch_q5_k_f16in_tail32(
    device const Q5_K_Block* A [[buffer(0)]],
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

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += D64_BK) {
        uint block_idx = kt / Q5_K_BLOCK_VALUES;
        uint pair = (kt % Q5_K_BLOCK_VALUES) / D64_BK;

        // Phase 1: precompute d*scale and dmin*min.
        for (uint r = tid; r < D32_BM; r += D32_TG) {
            uint global_r = tile_m + r;
            device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
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

        // Phase 2: vectorized uchar4 nibble + high-bit extraction (branchless).
        const uchar shift_lo = uchar(pair * 2);
        const uchar shift_hi = uchar(pair * 2 + 1);
        for (uint i = tid; i < D32_BM * (D64_BK / 8); i += D32_TG) {
            uint r  = i / (D64_BK / 8);
            uint b4 = (i % (D64_BK / 8)) * 4;
            uint global_r = tile_m + r;
            device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
            uchar4 qs_vec = *(device const uchar4*)(blk.qs + pair * 32 + b4);
            uchar4 qh_vec = *(device const uchar4*)(blk.qh + b4);
            uint4 lo = uint4(qs_vec & uchar4(0x0F));
            uint4 hi = uint4(qs_vec >> uchar4(4));
            lo |= uint4((qh_vec >> uchar4(shift_lo)) & uchar4(1)) << uint4(4);
            hi |= uint4((qh_vec >> uchar4(shift_hi)) & uchar4(1)) << uint4(4);
            float dsc1_f = float(row_dsc1[r]);
            float dmn1_f = float(row_dmin1[r]);
            float dsc2_f = float(row_dsc2[r]);
            float dmn2_f = float(row_dmin2[r]);
            *(threadgroup half4*)(tg_A + r * D64_BK + b4)      = half4(dsc1_f * float4(lo) - dmn1_f);
            *(threadgroup half4*)(tg_A + r * D64_BK + b4 + 32) = half4(dsc2_f * float4(hi) - dmn2_f);
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

kernel void dequant_batch_q5_k_f16in_small(
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

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += SB_BK) {
        uint block_idx = kt / Q5_K_BLOCK_VALUES;
        uint pair = (kt % Q5_K_BLOCK_VALUES) / SB_BK;

        // Phase 1: precompute d*scale and dmin*min.
        if (tid < SB_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
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

        // Phase 2: paired nibble + high-bit extraction.
        for (uint i = tid; i < SB_BM * (SB_BK / 2); i += SB_TG) {
            uint r = i / (SB_BK / 2);
            uint b = i % (SB_BK / 2);
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q5_K_Block& blk = A[global_r * blocks_per_row + block_idx];
                uchar byte = blk.qs[pair * 32 + b];
                uchar high_bits = blk.qh[b];
                float lo_q = float(byte & 0x0F) + (((high_bits >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
                float hi_q = float(byte >> 4) + (((high_bits >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);
                tg_A[r * SB_BK + b]      = half(float(row_dsc1[r]) * lo_q - float(row_dmin1[r]));
                tg_A[r * SB_BK + b + 32] = half(float(row_dsc2[r]) * hi_q - float(row_dmin2[r]));
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

kernel void dequant_batch_pair_q5_k_f16in(
    device const Q5_K_Block* A0 [[buffer(0)]],
    device const Q5_K_Block* A1 [[buffer(1)]],
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
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup half tg_A0[DB_BM * DB_BK];
    threadgroup half tg_A1[DB_BM * DB_BK];
    threadgroup half tg_B[DB_BN * DB_BK];
    threadgroup half row_dsc1_0[DB_BM];
    threadgroup half row_dmin1_0[DB_BM];
    threadgroup half row_dsc2_0[DB_BM];
    threadgroup half row_dmin2_0[DB_BM];
    threadgroup half row_dsc1_1[DB_BM];
    threadgroup half row_dmin1_1[DB_BM];
    threadgroup half row_dsc2_1[DB_BM];
    threadgroup half row_dmin2_1[DB_BM];

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

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;

    for (uint kt = 0; kt < K; kt += DB_BK) {
        uint block_idx = kt / Q5_K_BLOCK_VALUES;
        uint pair = (kt % Q5_K_BLOCK_VALUES) / DB_BK;

        // Phase 1: precompute d*scale and dmin*min for both A0 and A1.
        if (tid < DB_BM) {
            uint global_r = tile_m + tid;
            if (global_r < M) {
                device const Q5_K_Block& blk0 = A0[global_r * blocks_per_row + block_idx];
                float d0    = float(blk0.d);
                float dmin0 = float(blk0.dmin);
                float2 sm1_0 = get_scale_min_q4k(pair * 2, blk0.scales);
                float2 sm2_0 = get_scale_min_q4k(pair * 2 + 1, blk0.scales);
                row_dsc1_0[tid]  = half(d0 * sm1_0.x);
                row_dmin1_0[tid] = half(dmin0 * sm1_0.y);
                row_dsc2_0[tid]  = half(d0 * sm2_0.x);
                row_dmin2_0[tid] = half(dmin0 * sm2_0.y);

                device const Q5_K_Block& blk1 = A1[global_r * blocks_per_row + block_idx];
                float d1    = float(blk1.d);
                float dmin1 = float(blk1.dmin);
                float2 sm1_1 = get_scale_min_q4k(pair * 2, blk1.scales);
                float2 sm2_1 = get_scale_min_q4k(pair * 2 + 1, blk1.scales);
                row_dsc1_1[tid]  = half(d1 * sm1_1.x);
                row_dmin1_1[tid] = half(dmin1 * sm1_1.y);
                row_dsc2_1[tid]  = half(d1 * sm2_1.x);
                row_dmin2_1[tid] = half(dmin1 * sm2_1.y);
            } else {
                row_dsc1_0[tid]  = half(0.0f);
                row_dmin1_0[tid] = half(0.0f);
                row_dsc2_0[tid]  = half(0.0f);
                row_dmin2_0[tid] = half(0.0f);
                row_dsc1_1[tid]  = half(0.0f);
                row_dmin1_1[tid] = half(0.0f);
                row_dsc2_1[tid]  = half(0.0f);
                row_dmin2_1[tid] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: paired nibble + high-bit extraction for both A0 and A1.
        for (uint i = tid; i < DB_BM * (DB_BK / 2); i += DB_TG) {
            uint r = i / (DB_BK / 2);
            uint b = i % (DB_BK / 2);
            uint global_r = tile_m + r;
            if (global_r < M) {
                device const Q5_K_Block& blk0 = A0[global_r * blocks_per_row + block_idx];
                uchar byte0 = blk0.qs[pair * 32 + b];
                uchar high_bits0 = blk0.qh[b];
                float lo_q0 = float(byte0 & 0x0F) + (((high_bits0 >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
                float hi_q0 = float(byte0 >> 4) + (((high_bits0 >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);
                tg_A0[r * DB_BK + b]      = half(float(row_dsc1_0[r]) * lo_q0 - float(row_dmin1_0[r]));
                tg_A0[r * DB_BK + b + 32] = half(float(row_dsc2_0[r]) * hi_q0 - float(row_dmin2_0[r]));

                device const Q5_K_Block& blk1 = A1[global_r * blocks_per_row + block_idx];
                uchar byte1 = blk1.qs[pair * 32 + b];
                uchar high_bits1 = blk1.qh[b];
                float lo_q1 = float(byte1 & 0x0F) + (((high_bits1 >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
                float hi_q1 = float(byte1 >> 4) + (((high_bits1 >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);
                tg_A1[r * DB_BK + b]      = half(float(row_dsc1_1[r]) * lo_q1 - float(row_dmin1_1[r]));
                tg_A1[r * DB_BK + b + 32] = half(float(row_dsc2_1[r]) * hi_q1 - float(row_dmin2_1[r]));
            } else {
                tg_A0[r * DB_BK + b]      = half(0.0f);
                tg_A0[r * DB_BK + b + 32] = half(0.0f);
                tg_A1[r * DB_BK + b]      = half(0.0f);
                tg_A1[r * DB_BK + b + 32] = half(0.0f);
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

            simdgroup_half8x8 a00, a01, a02, a03;
            simdgroup_half8x8 a10, a11, a12, a13;
            simdgroup_load(a00, &tg_A0[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a01, &tg_A0[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a02, &tg_A0[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a03, &tg_A0[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a10, &tg_A1[0  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a11, &tg_A1[8  * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a12, &tg_A1[16 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);
            simdgroup_load(a13, &tg_A1[24 * DB_BK + kk * 8], DB_BK, ulong2(0,0), true);

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

    if (tile_n + DB_BN <= N && tile_m + DB_BM <= M) {
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

    threadgroup float out_tile[DB_BN * DB_BM];
    simdgroup_store(acc00, &out_tile[simd_id * 8 * DB_BM + 0],  DB_BM);
    simdgroup_store(acc01, &out_tile[simd_id * 8 * DB_BM + 8],  DB_BM);
    simdgroup_store(acc02, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc03, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C0[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_store(acc10, &out_tile[simd_id * 8 * DB_BM + 0],  DB_BM);
    simdgroup_store(acc11, &out_tile[simd_id * 8 * DB_BM + 8],  DB_BM);
    simdgroup_store(acc12, &out_tile[simd_id * 8 * DB_BM + 16], DB_BM);
    simdgroup_store(acc13, &out_tile[simd_id * 8 * DB_BM + 24], DB_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < DB_BN * DB_BM; i += DB_TG) {
        uint r = i / DB_BM;
        uint c = i % DB_BM;
        uint gn = tile_n + r;
        uint gm = tile_m + c;
        if (gn < N && gm < M) {
            C1[gn * M + gm] = out_tile[r * DB_BM + c];
        }
    }
}

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

kernel void dequant_matvec_silu_down_q5_k(
    device const Q5_K_Block* A [[buffer(0)]],
    device const float* gate   [[buffer(1)]],
    device const float* up     [[buffer(2)]],
    device float* y            [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
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
        sum += q5_k_block_dot_silu(a_row, b, gate, up, simd_lane);
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        y[first_row] = sum;
    }
}

kernel void moe_fused_silu_down_selected_weighted_q5_k_matvec(
    device const Q5_K_Block* A       [[buffer(0)]],
    device const float* gate         [[buffer(1)]],
    device const float* up           [[buffer(2)]],
    device const int32_t* selected   [[buffer(3)]],
    device const float* sel_weights  [[buffer(4)]],
    device float* y                  [[buffer(5)]],
    constant uint& M                 [[buffer(6)]],
    constant uint& K                 [[buffer(7)]],
    constant uint& n_selected        [[buffer(8)]],
    constant uint& weight_stride     [[buffer(9)]],
    uint tg_id                       [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    uint first_row = tg_id * 2 + simd_id;
    if (first_row >= M) return;

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    float acc = 0.0f;

    for (uint slot = 0; slot < n_selected; ++slot) {
        const uint expert = uint(selected[slot]);
        const float route_weight = sel_weights[slot];
        device const Q5_K_Block* a_row =
            A + expert * weight_stride + first_row * blocks_per_row;
        device const float* gate_row = gate + slot * K;
        device const float* up_row = up + slot * K;

        float sum = 0.0f;
        for (uint b = 0; b < blocks_per_row; ++b) {
            sum += q5_k_block_dot_silu(a_row, b, gate_row, up_row, simd_lane);
        }

        sum = simd_sum(sum);
        if (simd_lane == 0) {
            acc += route_weight * sum;
        }
    }

    if (simd_lane == 0) {
        y[first_row] = acc;
    }
}

kernel void moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8(
    device const Q5_K_Block* A       [[buffer(0)]],
    device const float* gate         [[buffer(1)]],
    device const float* up           [[buffer(2)]],
    device const int32_t* selected   [[buffer(3)]],
    device const float* sel_weights  [[buffer(4)]],
    device float* y                  [[buffer(5)]],
    constant uint& M                 [[buffer(6)]],
    constant uint& K                 [[buffer(7)]],
    constant uint& weight_stride     [[buffer(8)]],
    uint tg_id                       [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    uint first_row = tg_id * 2 + simd_id;
    if (first_row >= M) return;

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    float acc = 0.0f;

#define FUSED_Q5K_SLOT8_STEP(SLOT)                                              \
    {                                                                           \
        const uint expert = uint(selected[SLOT]);                               \
        const float route_weight = sel_weights[SLOT];                           \
        device const Q5_K_Block* a_row =                                        \
            A + expert * weight_stride + first_row * blocks_per_row;            \
        device const float* gate_row = gate + (SLOT) * K;                       \
        device const float* up_row = up + (SLOT) * K;                           \
        float sum = 0.0f;                                                       \
        for (uint b = 0; b < blocks_per_row; ++b) {                             \
            sum += q5_k_block_dot_silu(a_row, b, gate_row, up_row, simd_lane);  \
        }                                                                       \
        sum = simd_sum(sum);                                                    \
        if (simd_lane == 0) {                                                   \
            acc += route_weight * sum;                                          \
        }                                                                       \
    }

    FUSED_Q5K_SLOT8_STEP(0);
    FUSED_Q5K_SLOT8_STEP(1);
    FUSED_Q5K_SLOT8_STEP(2);
    FUSED_Q5K_SLOT8_STEP(3);
    FUSED_Q5K_SLOT8_STEP(4);
    FUSED_Q5K_SLOT8_STEP(5);
    FUSED_Q5K_SLOT8_STEP(6);
    FUSED_Q5K_SLOT8_STEP(7);

#undef FUSED_Q5K_SLOT8_STEP

    if (simd_lane == 0) {
        y[first_row] = acc;
    }
}

kernel void moe_fused_silu_down_selected_weighted_q5_k_matvec_nr2(
    device const Q5_K_Block* A       [[buffer(0)]],
    device const float* gate         [[buffer(1)]],
    device const float* up           [[buffer(2)]],
    device const int32_t* selected   [[buffer(3)]],
    device const float* sel_weights  [[buffer(4)]],
    device float* y                  [[buffer(5)]],
    constant uint& M                 [[buffer(6)]],
    constant uint& K                 [[buffer(7)]],
    constant uint& n_selected        [[buffer(8)]],
    constant uint& weight_stride     [[buffer(9)]],
    uint tg_id                       [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    uint first_row = (tg_id * Q5K_NR2_SG_PER_TG + simd_id) * Q5K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    bool valid1 = (first_row + 1) < M;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint slot = 0; slot < n_selected; ++slot) {
        const uint expert = uint(selected[slot]);
        const float route_weight = sel_weights[slot];
        device const Q5_K_Block* row0 =
            A + expert * weight_stride + first_row * blocks_per_row;
        device const Q5_K_Block* row1 = valid1 ? row0 + blocks_per_row : row0;
        device const float* gate_row = gate + slot * K;
        device const float* up_row = up + slot * K;

        float sumf0 = 0.0f;
        float sumf1 = 0.0f;

        for (uint b = 0; b < blocks_per_row; ++b) {
            uint base = b * Q5_K_BLOCK_VALUES;

            half rx0 = half(silu_mul_f32(gate_row[base +   0 + simd_lane], up_row[base +   0 + simd_lane]));
            half rx1 = half(silu_mul_f32(gate_row[base +  32 + simd_lane], up_row[base +  32 + simd_lane]));
            half rx2 = half(silu_mul_f32(gate_row[base +  64 + simd_lane], up_row[base +  64 + simd_lane]));
            half rx3 = half(silu_mul_f32(gate_row[base +  96 + simd_lane], up_row[base +  96 + simd_lane]));
            half rx4 = half(silu_mul_f32(gate_row[base + 128 + simd_lane], up_row[base + 128 + simd_lane]));
            half rx5 = half(silu_mul_f32(gate_row[base + 160 + simd_lane], up_row[base + 160 + simd_lane]));
            half rx6 = half(silu_mul_f32(gate_row[base + 192 + simd_lane], up_row[base + 192 + simd_lane]));
            half rx7 = half(silu_mul_f32(gate_row[base + 224 + simd_lane], up_row[base + 224 + simd_lane]));

            for (ushort row = 0; row < Q5K_NR2_ROWS_PER_SG; ++row) {
                device const Q5_K_Block& blk = (row == 0 ? row0 : row1)[b];
                float d = (simd_lane == 0) ? float(blk.d) : 0.0f;
                float dmin = (simd_lane == 0) ? float(blk.dmin) : 0.0f;
                d = simd_broadcast(d, 0);
                dmin = simd_broadcast(dmin, 0);
                device const uchar* scales = blk.scales;
                device const uchar* qh = blk.qh;
                device const uchar* qs = blk.qs;

                float block_sum = 0.0f;
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
                    block_sum += (d1 * lo_q - m1) * float(lo);
                    block_sum += (d2 * hi_q - m2) * float(hi);
                }

                if (row == 0) {
                    sumf0 += block_sum;
                } else {
                    sumf1 += block_sum;
                }
            }
        }

        float sum0 = simd_sum(sumf0);
        float sum1 = simd_sum(sumf1);
        if (simd_lane == 0) {
            acc0 += route_weight * sum0;
            if (valid1) {
                acc1 += route_weight * sum1;
            }
        }
    }

    if (simd_lane == 0) {
        y[first_row] = acc0;
        if (valid1) {
            y[first_row + 1] = acc1;
        }
    }
}

kernel void moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8_nr2(
    device const Q5_K_Block* A       [[buffer(0)]],
    device const float* gate         [[buffer(1)]],
    device const float* up           [[buffer(2)]],
    device const int32_t* selected   [[buffer(3)]],
    device const float* sel_weights  [[buffer(4)]],
    device float* y                  [[buffer(5)]],
    constant uint& M                 [[buffer(6)]],
    constant uint& K                 [[buffer(7)]],
    constant uint& weight_stride     [[buffer(8)]],
    uint tg_id                       [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    uint first_row = (tg_id * Q5K_NR2_SG_PER_TG + simd_id) * Q5K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    bool valid1 = (first_row + 1) < M;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

#define FUSED_Q5K_SLOT8_NR2_STEP(SLOT)                                         \
    {                                                                           \
        const uint expert = uint(selected[SLOT]);                               \
        const float route_weight = sel_weights[SLOT];                           \
        device const Q5_K_Block* row0 =                                         \
            A + expert * weight_stride + first_row * blocks_per_row;            \
        device const Q5_K_Block* row1 = valid1 ? row0 + blocks_per_row : row0; \
        device const float* gate_row = gate + (SLOT) * K;                       \
        device const float* up_row = up + (SLOT) * K;                           \
        float sumf0 = 0.0f;                                                     \
        float sumf1 = 0.0f;                                                     \
        for (uint b = 0; b < blocks_per_row; ++b) {                             \
            uint base = b * Q5_K_BLOCK_VALUES;                                  \
            half rx0 = half(silu_mul_f32(gate_row[base +   0 + simd_lane], up_row[base +   0 + simd_lane])); \
            half rx1 = half(silu_mul_f32(gate_row[base +  32 + simd_lane], up_row[base +  32 + simd_lane])); \
            half rx2 = half(silu_mul_f32(gate_row[base +  64 + simd_lane], up_row[base +  64 + simd_lane])); \
            half rx3 = half(silu_mul_f32(gate_row[base +  96 + simd_lane], up_row[base +  96 + simd_lane])); \
            half rx4 = half(silu_mul_f32(gate_row[base + 128 + simd_lane], up_row[base + 128 + simd_lane])); \
            half rx5 = half(silu_mul_f32(gate_row[base + 160 + simd_lane], up_row[base + 160 + simd_lane])); \
            half rx6 = half(silu_mul_f32(gate_row[base + 192 + simd_lane], up_row[base + 192 + simd_lane])); \
            half rx7 = half(silu_mul_f32(gate_row[base + 224 + simd_lane], up_row[base + 224 + simd_lane])); \
            for (ushort row = 0; row < Q5K_NR2_ROWS_PER_SG; ++row) {            \
                device const Q5_K_Block& blk = (row == 0 ? row0 : row1)[b];     \
                float d = (simd_lane == 0) ? float(blk.d) : 0.0f;               \
                float dmin = (simd_lane == 0) ? float(blk.dmin) : 0.0f;         \
                d = simd_broadcast(d, 0);                                       \
                dmin = simd_broadcast(dmin, 0);                                 \
                device const uchar* scales = blk.scales;                        \
                device const uchar* qh = blk.qh;                                \
                device const uchar* qs = blk.qs;                                \
                float block_sum = 0.0f;                                         \
                uchar high_bits = qh[simd_lane];                                \
                FOR_UNROLL (uint pair = 0; pair < 4; ++pair) {                  \
                    float2 sm1 = get_scale_min_q4k(pair * 2, scales);           \
                    float2 sm2 = get_scale_min_q4k(pair * 2 + 1, scales);       \
                    float d1 = d * sm1.x, m1 = dmin * sm1.y;                    \
                    float d2 = d * sm2.x, m2 = dmin * sm2.y;                    \
                    uchar byte = qs[pair * 32 + simd_lane];                     \
                    float lo_q = float(byte & 0x0F)                             \
                        + (((high_bits >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);  \
                    float hi_q = float(byte >> 4)                               \
                        + (((high_bits >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f); \
                    half lo = (pair == 0) ? rx0 : (pair == 1) ? rx2 : (pair == 2) ? rx4 : rx6; \
                    half hi = (pair == 0) ? rx1 : (pair == 1) ? rx3 : (pair == 2) ? rx5 : rx7; \
                    block_sum += (d1 * lo_q - m1) * float(lo);                  \
                    block_sum += (d2 * hi_q - m2) * float(hi);                  \
                }                                                               \
                if (row == 0) {                                                 \
                    sumf0 += block_sum;                                         \
                } else {                                                        \
                    sumf1 += block_sum;                                         \
                }                                                               \
            }                                                                   \
        }                                                                       \
        float sum0 = simd_sum(sumf0);                                           \
        float sum1 = simd_sum(sumf1);                                           \
        if (simd_lane == 0) {                                                   \
            acc0 += route_weight * sum0;                                        \
            if (valid1) {                                                       \
                acc1 += route_weight * sum1;                                    \
            }                                                                   \
        }                                                                       \
    }

    FUSED_Q5K_SLOT8_NR2_STEP(0);
    FUSED_Q5K_SLOT8_NR2_STEP(1);
    FUSED_Q5K_SLOT8_NR2_STEP(2);
    FUSED_Q5K_SLOT8_NR2_STEP(3);
    FUSED_Q5K_SLOT8_NR2_STEP(4);
    FUSED_Q5K_SLOT8_NR2_STEP(5);
    FUSED_Q5K_SLOT8_NR2_STEP(6);
    FUSED_Q5K_SLOT8_NR2_STEP(7);

#undef FUSED_Q5K_SLOT8_NR2_STEP

    if (simd_lane == 0) {
        y[first_row] = acc0;
        if (valid1) {
            y[first_row + 1] = acc1;
        }
    }
}

kernel void dequant_matvec_gelu_down_q5_k(
    device const Q5_K_Block* A [[buffer(0)]],
    device const float* gate   [[buffer(1)]],
    device const float* up     [[buffer(2)]],
    device float* y            [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
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
        sum += q5_k_block_dot_gelu(a_row, b, gate, up, simd_lane);
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        y[first_row] = sum;
    }
}

kernel void dequant_matvec_pair_q5_k(
    device const Q5_K_Block* A0 [[buffer(0)]],
    device const Q5_K_Block* A1 [[buffer(1)]],
    device const float* x       [[buffer(2)]],
    device float* y0            [[buffer(3)]],
    device float* y1            [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& K            [[buffer(6)]],
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    uint first_row = tg_id * 2 + simd_id;
    if (first_row >= M) return;

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    device const Q5_K_Block* a_row0 = A0 + first_row * blocks_per_row;
    device const Q5_K_Block* a_row1 = A1 + first_row * blocks_per_row;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        sum0 += q5_k_block_dot(a_row0, b, x, simd_lane);
        sum1 += q5_k_block_dot(a_row1, b, x, simd_lane);
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (simd_lane == 0) {
        y0[first_row] = sum0;
        y1[first_row] = sum1;
    }
}

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

kernel void moe_mul_mat_id_q5_k_blocked(
    device const Q5_K_Block *weights [[buffer(0)]],
    device const float *input        [[buffer(1)]],
    device const uint32_t *tpe       [[buffer(2)]],
    device const int32_t *hids       [[buffer(3)]],
    device float *output             [[buffer(4)]],
    constant uint &M                 [[buffer(5)]],
    constant uint &K                 [[buffer(6)]],
    constant uint &n_tokens          [[buffer(7)]],
    constant uint &n_expert_used     [[buffer(8)]],
    constant uint &weight_stride     [[buffer(9)]],
    device const uint32_t *active_meta [[buffer(10)]],
    constant uint &input_is_hid      [[buffer(11)]],
    threadgroup char *shmem          [[threadgroup(0)]],
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

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    device const Q5_K_Block *W = weights + expert * weight_stride;
    const short offset1 = il0 / QK_NL;
    device const Q5_K_Block *x = W + uint(r0 + lr0) * blocks_per_row + offset1;

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

kernel void moe_mul_mat_selected_q5_k_matvec(
    device const Q5_K_Block *weights       [[buffer(0)]],
    device const float *input              [[buffer(1)]],
    device const int32_t *selected_experts [[buffer(2)]],
    device float *output                   [[buffer(3)]],
    constant uint &M                       [[buffer(4)]],
    constant uint &K                       [[buffer(5)]],
    constant uint &n_selected              [[buffer(6)]],
    constant uint &weight_stride           [[buffer(7)]],
    constant uint &input_is_slot_major     [[buffer(8)]],
    uint3 tgpig                            [[threadgroup_position_in_grid]],
    uint simd_lane                         [[thread_index_in_simdgroup]],
    uint simd_id                           [[simdgroup_index_in_threadgroup]]
) {
    if (tgpig.z >= n_selected) return;

    const uint slot = tgpig.z;
    const uint expert = uint(selected_experts[slot]);
    const uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    const uint first_row = (tgpig.x * Q5K_NR2_SG_PER_TG + simd_id) * Q5K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    const bool valid1 = (first_row + 1) < M;
    device const Q5_K_Block *base = weights + expert * weight_stride;
    device const Q5_K_Block *row0 = base + first_row * blocks_per_row;
    device const Q5_K_Block *row1 = valid1 ? row0 + blocks_per_row : row0;
    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    device const float *x = input + input_row * K;

    float sumf[Q5K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint base_k = b * Q5_K_BLOCK_VALUES;

        half rx0 = half(x[base_k +   0 + simd_lane]);
        half rx1 = half(x[base_k +  32 + simd_lane]);
        half rx2 = half(x[base_k +  64 + simd_lane]);
        half rx3 = half(x[base_k +  96 + simd_lane]);
        half rx4 = half(x[base_k + 128 + simd_lane]);
        half rx5 = half(x[base_k + 160 + simd_lane]);
        half rx6 = half(x[base_k + 192 + simd_lane]);
        half rx7 = half(x[base_k + 224 + simd_lane]);

        for (ushort row = 0; row < Q5K_NR2_ROWS_PER_SG; ++row) {
            device const Q5_K_Block *row_ptr = (row == 0) ? row0 : row1;
            device const Q5_K_Block &blk = row_ptr[b];
            float d = (simd_lane == 0) ? float(blk.d) : 0.0f;
            float dmin = (simd_lane == 0) ? float(blk.dmin) : 0.0f;
            d = simd_broadcast(d, 0);
            dmin = simd_broadcast(dmin, 0);

            device const uchar *scales = blk.scales;
            device const uchar *qh = blk.qh;
            device const uchar *qs = blk.qs;

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
        device float *y_slot = output + slot * M;
        y_slot[first_row] = sum0;
        if (valid1) {
            y_slot[first_row + 1] = sum1;
        }
    }
}

kernel void moe_mul_mat_selected_q5_k_blocked(
    device const Q5_K_Block *weights       [[buffer(0)]],
    device const float *input              [[buffer(1)]],
    device const int32_t *selected_experts [[buffer(2)]],
    device float *output                   [[buffer(3)]],
    constant uint &M                       [[buffer(4)]],
    constant uint &K                       [[buffer(5)]],
    constant uint &n_selected              [[buffer(6)]],
    constant uint &weight_stride           [[buffer(7)]],
    constant uint &input_is_slot_major     [[buffer(8)]],
    threadgroup char *shmem                [[threadgroup(0)]],
    uint3 tgpig  [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    if (tgpig.z >= n_selected) return;

    const uint expert = uint(selected_experts[tgpig.z]);
    const uint slot = tgpig.z;

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short QK_NL = 16;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    if ((uint)r1 >= 1) return;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = 1;
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    short il = il0;

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    device const Q5_K_Block *W = weights + expert * weight_stride;
    const short offset1 = il0 / QK_NL;
    device const Q5_K_Block *x = W + uint(r0 + lr0) * blocks_per_row + offset1;
    const uint input_row = input_is_slot_major != 0 ? slot + uint(lr1) : uint(lr1);
    device const float *y = input + input_row * K + short(8 * (tiitg % NL1));

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
    if (sgitg == 0) {
        device float  *D  = output + slot * M + uint(r0);
        device float4 *D4 = (device float4 *)D;
        threadgroup float  *Cs  = ((threadgroup float *)shmem);
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

kernel void moe_mul_mat_selected_pair_q5_k_matvec(
    device const Q5_K_Block *weights0      [[buffer(0)]],
    device const Q5_K_Block *weights1      [[buffer(1)]],
    device const float *input              [[buffer(2)]],
    device const int32_t *selected_experts [[buffer(3)]],
    device float *output0                  [[buffer(4)]],
    device float *output1                  [[buffer(5)]],
    constant uint &M                       [[buffer(6)]],
    constant uint &K                       [[buffer(7)]],
    constant uint &n_selected              [[buffer(8)]],
    constant uint &weight_stride0          [[buffer(9)]],
    constant uint &weight_stride1          [[buffer(10)]],
    constant uint &input_is_slot_major     [[buffer(11)]],
    uint3 tgpig                            [[threadgroup_position_in_grid]],
    uint simd_lane                         [[thread_index_in_simdgroup]],
    uint simd_id                           [[simdgroup_index_in_threadgroup]]
) {
    if (tgpig.z >= n_selected) return;

    const uint slot = tgpig.z;
    const uint expert = uint(selected_experts[slot]);
    const uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    const uint first_row =
        (tgpig.x * Q5K_NR2_SG_PER_TG + simd_id) * Q5K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    const bool valid1 = (first_row + 1) < M;
    device const Q5_K_Block *base0 = weights0 + expert * weight_stride0;
    device const Q5_K_Block *row00 = base0 + first_row * blocks_per_row;
    device const Q5_K_Block *row01 = valid1 ? row00 + blocks_per_row : row00;
    device const Q5_K_Block *base1 = weights1 + expert * weight_stride1;
    device const Q5_K_Block *row10 = base1 + first_row * blocks_per_row;
    device const Q5_K_Block *row11 = valid1 ? row10 + blocks_per_row : row10;
    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    device const float *x = input + input_row * K;

    float sumf0[Q5K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};
    float sumf1[Q5K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint base_k = b * Q5_K_BLOCK_VALUES;

        half rx0 = half(x[base_k +   0 + simd_lane]);
        half rx1 = half(x[base_k +  32 + simd_lane]);
        half rx2 = half(x[base_k +  64 + simd_lane]);
        half rx3 = half(x[base_k +  96 + simd_lane]);
        half rx4 = half(x[base_k + 128 + simd_lane]);
        half rx5 = half(x[base_k + 160 + simd_lane]);
        half rx6 = half(x[base_k + 192 + simd_lane]);
        half rx7 = half(x[base_k + 224 + simd_lane]);

        for (ushort row = 0; row < Q5K_NR2_ROWS_PER_SG; ++row) {
            device const Q5_K_Block &blk0 = (row == 0 ? row00 : row01)[b];
            device const Q5_K_Block &blk1 = (row == 0 ? row10 : row11)[b];

            float d0 = (simd_lane == 0) ? float(blk0.d) : 0.0f;
            float dmin0 = (simd_lane == 0) ? float(blk0.dmin) : 0.0f;
            float d1 = (simd_lane == 0) ? float(blk1.d) : 0.0f;
            float dmin1 = (simd_lane == 0) ? float(blk1.dmin) : 0.0f;
            d0 = simd_broadcast(d0, 0);
            dmin0 = simd_broadcast(dmin0, 0);
            d1 = simd_broadcast(d1, 0);
            dmin1 = simd_broadcast(dmin1, 0);

            device const uchar *scales0 = blk0.scales;
            device const uchar *qh0 = blk0.qh;
            device const uchar *qs0 = blk0.qs;
            device const uchar *scales1 = blk1.scales;
            device const uchar *qh1 = blk1.qh;
            device const uchar *qs1 = blk1.qs;

            uchar high_bits0 = qh0[simd_lane];
            uchar high_bits1 = qh1[simd_lane];
            FOR_UNROLL (uint pair = 0; pair < 4; ++pair) {
                float2 sm10 = get_scale_min_q4k(pair * 2, scales0);
                float2 sm20 = get_scale_min_q4k(pair * 2 + 1, scales0);
                float d10 = d0 * sm10.x, m10 = dmin0 * sm10.y;
                float d20 = d0 * sm20.x, m20 = dmin0 * sm20.y;

                float2 sm11 = get_scale_min_q4k(pair * 2, scales1);
                float2 sm21 = get_scale_min_q4k(pair * 2 + 1, scales1);
                float d11 = d1 * sm11.x, m11 = dmin1 * sm11.y;
                float d21 = d1 * sm21.x, m21 = dmin1 * sm21.y;

                uchar byte0 = qs0[pair * 32 + simd_lane];
                float lo_q0 = float(byte0 & 0x0F)
                    + (((high_bits0 >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
                float hi_q0 = float(byte0 >> 4)
                    + (((high_bits0 >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);

                uchar byte1 = qs1[pair * 32 + simd_lane];
                float lo_q1 = float(byte1 & 0x0F)
                    + (((high_bits1 >> (pair * 2)) & 0x01) ? 16.0f : 0.0f);
                float hi_q1 = float(byte1 >> 4)
                    + (((high_bits1 >> (pair * 2 + 1)) & 0x01) ? 16.0f : 0.0f);

                half lo = (pair == 0) ? rx0 : (pair == 1) ? rx2 : (pair == 2) ? rx4 : rx6;
                half hi = (pair == 0) ? rx1 : (pair == 1) ? rx3 : (pair == 2) ? rx5 : rx7;
                sumf0[row] += (d10 * lo_q0 - m10) * float(lo);
                sumf0[row] += (d20 * hi_q0 - m20) * float(hi);
                sumf1[row] += (d11 * lo_q1 - m11) * float(lo);
                sumf1[row] += (d21 * hi_q1 - m21) * float(hi);
            }
        }
    }

    float sum00 = simd_sum(sumf0[0]);
    float sum01 = simd_sum(sumf0[1]);
    float sum10 = simd_sum(sumf1[0]);
    float sum11 = simd_sum(sumf1[1]);
    if (simd_lane == 0) {
        device float *y0_slot = output0 + slot * M;
        device float *y1_slot = output1 + slot * M;
        y0_slot[first_row] = sum00;
        y1_slot[first_row] = sum10;
        if (valid1) {
            y0_slot[first_row + 1] = sum01;
            y1_slot[first_row + 1] = sum11;
        }
    }
}

kernel void moe_mul_mat_selected_weighted_q5_k_blocked(
    device const Q5_K_Block *weights       [[buffer(0)]],
    device const float *input              [[buffer(1)]],
    device const int32_t *selected_experts [[buffer(2)]],
    device const float *expert_weights     [[buffer(3)]],
    device float *output                   [[buffer(4)]],
    constant uint &M                       [[buffer(5)]],
    constant uint &K                       [[buffer(6)]],
    constant uint &n_selected              [[buffer(7)]],
    constant uint &weight_stride           [[buffer(8)]],
    threadgroup char *shmem                [[threadgroup(0)]],
    uint3 tgpig  [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short QK_NL = 16;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    if ((uint)r1 >= 1) return;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = 1;
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    const ushort lane = tiitg % 32;

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    const uint vec4_count = uint(nr0 / 4);
    const uint scalar_base = 4 * vec4_count;
    float4 acc4 = float4(0.0f);
    float acc_tail = 0.0f;

    const uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    for (uint slot = 0; slot < n_selected; ++slot) {
        const uint expert = uint(selected_experts[slot]);
        const float route_weight = expert_weights[slot];
        short il = il0;

        device const Q5_K_Block *W = weights + expert * weight_stride;
        const short offset1 = il0 / QK_NL;
        device const Q5_K_Block *x = W + uint(r0 + lr0) * blocks_per_row + offset1;
        device const float *y = input + slot * K + short(8 * (tiitg % NL1));

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

        if (sgitg == 0) {
            threadgroup float *Cs = ((threadgroup float *)shmem);
            threadgroup float4 *C4 = (threadgroup float4 *)Cs;
            if (lane < vec4_count) {
                acc4 += route_weight * C4[lane];
            }
            const uint scalar_idx = scalar_base + lane;
            if (scalar_idx < uint(nr0)) {
                acc_tail += route_weight * Cs[scalar_idx];
            }
        }
    }

    if (sgitg == 0) {
        device float *D = output + uint(r0);
        device float4 *D4 = (device float4 *)D;
        if (lane < vec4_count) {
            D4[lane] = acc4;
        }
        const uint scalar_idx = scalar_base + lane;
        if (scalar_idx < uint(nr0)) {
            D[scalar_idx] = acc_tail;
        }
    }
}

kernel void moe_mul_mat_selected_pair_q5_k_blocked(
    device const Q5_K_Block *weights0      [[buffer(0)]],
    device const Q5_K_Block *weights1      [[buffer(1)]],
    device const float *input              [[buffer(2)]],
    device const int32_t *selected_experts [[buffer(3)]],
    device float *output0                  [[buffer(4)]],
    device float *output1                  [[buffer(5)]],
    constant uint &M                       [[buffer(6)]],
    constant uint &K                       [[buffer(7)]],
    constant uint &n_selected              [[buffer(8)]],
    constant uint &weight_stride0          [[buffer(9)]],
    constant uint &weight_stride1          [[buffer(10)]],
    constant uint &input_is_slot_major     [[buffer(11)]],
    threadgroup char *shmem                [[threadgroup(0)]],
    uint3 tgpig  [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    if (tgpig.z >= n_selected) return;

    const uint expert = uint(selected_experts[tgpig.z]);
    const uint slot = tgpig.z;

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short QK_NL = 16;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    if ((uint)r1 >= 1) return;

    const short nr0 = short(min(uint(NR0), M - uint(r0)));
    const short nr1 = 1;
    const short lr0 = short(min(short(tiitg / NL0), short(nr0 - 1)));
    const short lr1 = short(min(short(tiitg / NL1), short(nr1 - 1)));

    const short il0 = short(tiitg % NL0);
    short il = il0;

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    uint blocks_per_row = K / Q5_K_BLOCK_VALUES;
    device const Q5_K_Block *W0 = weights0 + expert * weight_stride0;
    device const Q5_K_Block *W1 = weights1 + expert * weight_stride1;
    const short offset1 = il0 / QK_NL;
    device const Q5_K_Block *x0 = W0 + uint(r0 + lr0) * blocks_per_row + offset1;
    device const Q5_K_Block *x1 = W1 + uint(r0 + lr0) * blocks_per_row + offset1;
    const uint input_row = input_is_slot_major != 0 ? slot + uint(lr1) : uint(lr1);
    device const float *y = input + input_row * K + short(8 * (tiitg % NL1));

    simdgroup_half8x8 ma0[4];
    simdgroup_half8x8 ma1[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc0[8];
    simdgroup_float8x8 mc1[8];
    for (short i = 0; i < 8; i++) {
        mc0[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc1[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a0;
        half4x4 temp_a1;
        dequantize_q5k_blocked(x0, il, temp_a0);
        dequantize_q5k_blocked(x1, il, temp_a1);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a0[i / 4][i % 4];
        }

        const short sx = short(tiitg % NL1);
        const short sy = (tiitg / NL1) / 8;
        const short ly = (tiitg / NL1) % 8;
        const short ib = 4 * sx + sy;
        *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
            (half2x4)(*(device float2x4 *)y);

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x0 = (il < 2) ? x0 + (2 + QK_NL - 1) / QK_NL : x0;
        x1 = (il < 2) ? x1 + (2 + QK_NL - 1) / QK_NL : x1;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma0 = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma0[i], lsma0 + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc0[i], mb[i / 4], ma0[i % 4], mc0[i]);
            }
            lsma0 += 8 * 64;
            lsmb += 4 * 64;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a1[i / 4][i % 4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma1 = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half *lsmb1 = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma1[i], lsma1 + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb1 + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc1[i], mb[i / 4], ma1[i % 4], mc1[i]);
            }
            lsma1 += 8 * 64;
            lsmb1 += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *temp_str = ((threadgroup float *)shmem)
        + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc0[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                        NR0, ulong2(0, 0), false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort lane = tiitg % 32;
    if (sgitg == 0) {
        device float  *D  = output0 + slot * M + uint(r0);
        device float4 *D4 = (device float4 *)D;
        threadgroup float  *Cs  = ((threadgroup float *)shmem);
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

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc1[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                        NR0, ulong2(0, 0), false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        device float  *D  = output1 + slot * M + uint(r0);
        device float4 *D4 = (device float4 *)D;
        threadgroup float  *Cs  = ((threadgroup float *)shmem);
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
