// Q4_0 dequant and matvec kernels.
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

