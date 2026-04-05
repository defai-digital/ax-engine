// Mixed or generic dequant kernels not tied to a single quant family.
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

kernel void moe_mul_mat_id_map0(
    device const int32_t *expert_ids [[buffer(0)]],
    device uint32_t *tpe             [[buffer(1)]],
    device int32_t *hids             [[buffer(2)]],
    constant uint &n_tokens          [[buffer(3)]],
    constant uint &n_expert_used     [[buffer(4)]],
    constant uint &n_expert          [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint ide = tid;  // This thread handles expert `ide`.
    if (ide >= n_expert) return;

    device int32_t *ids_out = hids + ide * n_tokens;
    uint count = 0;

    for (uint t = 0; t < n_tokens; t++) {
        device const int32_t *assigns = expert_ids + t * n_expert_used;
        for (uint k = 0; k < n_expert_used; k++) {
            if ((uint)assigns[k] == ide) {
                // Store flat index: token * n_expert_used + k
                ids_out[count] = t * n_expert_used + k;
                count++;
                break;
            }
        }
    }
    tpe[ide] = count;

    device uint32_t *active_experts = (device uint32_t *)(hids) +
        n_expert * n_tokens;  // Placed after hids data
    active_experts[ide] = ide;
}

