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

// BF16 weights × f32 inputs → f32 batch matmul, B-transposed layout.
// A[M × K] stored as native Metal `bfloat` (M3+, Metal 3+).
// B[N × K] in f32, cast to bfloat on load (bfloat has f32 exponent range, no overflow).
// Uses simdgroup_bfloat8x8 with f32 accumulators.
kernel void batch_matmul_btrans_bf16_f32(
    device const bfloat* A   [[buffer(0)]],  // [M × K] bfloat
    device const float* B    [[buffer(1)]],  // [N × K] f32
    device float* C          [[buffer(2)]],  // [N × M] f32
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    uint2 group_id           [[threadgroup_position_in_grid]],
    uint  tid                [[thread_index_in_threadgroup]],
    uint  simd_id            [[simdgroup_index_in_threadgroup]],
    uint  simd_lane          [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * DB_BM;
    uint tile_n = group_id.y * DB_BN;

    threadgroup bfloat tg_A[DB_BM * DB_BK];
    threadgroup bfloat tg_B[DB_BN * DB_BK];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    for (uint kt = 0; kt < K; kt += DB_BK) {
        for (uint i = tid; i < DB_BM * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gm = tile_m + r;
            uint gk = kt + c;
            tg_A[r * DB_BK + c] = (gm < M && gk < K) ? A[gm * K + gk] : bfloat(0.0f);
        }

        for (uint i = tid; i < DB_BN * DB_BK; i += DB_TG) {
            uint r = i / DB_BK;
            uint c = i % DB_BK;
            uint gn = tile_n + r;
            uint gk = kt + c;
            tg_B[r * DB_BK + c] = (gn < N && gk < K) ? bfloat(B[gn * K + gk]) : bfloat(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < DB_BK / 8; kk++) {
            simdgroup_bfloat8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * DB_BK + kk * 8], DB_BK);

            simdgroup_bfloat8x8 a0, a1, a2, a3;
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

// 64×64 full-tile fast path for bfloat weights × f32 inputs → f32 batch matmul.
// A[M × K] as native Metal `bfloat`, B[N × K] in f32. M must be a multiple of D64_BM.
kernel void batch_matmul_btrans_bf16_f32_full64(
    device const bfloat* A   [[buffer(0)]],  // [M × K] bfloat
    device const float* B    [[buffer(1)]],  // [N × K] f32
    device float* C          [[buffer(2)]],  // [N × C_STRIDE] f32
    constant uint& M         [[buffer(3)]],  // output cols for this dispatch
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    constant uint& C_STRIDE  [[buffer(6)]],  // destination row stride
    uint2 group_id           [[threadgroup_position_in_grid]],
    uint tid                 [[thread_index_in_threadgroup]],
    uint simd_id             [[simdgroup_index_in_threadgroup]],
    uint simd_lane           [[thread_index_in_simdgroup]]
) {
    uint tile_m = group_id.x * D64_BM;
    uint tile_n = group_id.y * D64_BN;

    threadgroup bfloat tg_A[D64_BM * D64_BK];
    threadgroup bfloat tg_B[D64_BN * D64_BK];

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
            tg_B[r * D64_BK + c] = bfloat(B[gn * K + gk]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < D64_BK / 8; kk++) {
            simdgroup_bfloat8x8 b_frag;
            simdgroup_load(b_frag, &tg_B[simd_id * 8 * D64_BK + kk * 8], D64_BK);

            simdgroup_bfloat8x8 a0, a1, a2, a3, a4, a5, a6, a7;
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

kernel void dequant_matvec_dense_bf16(
    device const uint16_t* A [[buffer(0)]],  // [M × K] in bf16 (little-endian)
    device const float* x    [[buffer(1)]],  // [K]
    device float* y          [[buffer(2)]],  // [M]
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    uint row      [[threadgroup_position_in_grid]],
    uint lid      [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;
    device const uint16_t* a_row = A + row * K;
    float sum = 0.0f;

    for (uint i = simd_id * 32 + simd_lane; i < K; i += num_simd_groups * 32) {
        // BF16 → float: zero-extend the 16-bit value into the upper half of a 32-bit word.
        uint32_t bits = (uint32_t)a_row[i] << 16;
        float w = as_type<float>(bits);
        sum += w * x[i];
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
    device uint32_t *active_meta     [[buffer(3)]],
    constant uint &n_tokens          [[buffer(4)]],
    constant uint &n_expert_used     [[buffer(5)]],
    constant uint &n_expert          [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]])
{
    device atomic_uint *active_count = (device atomic_uint *)active_meta;
    if (tid == 0) {
        atomic_store_explicit(active_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

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
    if (count > 0) {
        uint slot = atomic_fetch_add_explicit(active_count, 1u, memory_order_relaxed);
        active_meta[1 + slot] = ide;
    }
}

// ── Q5_0 blocked batch matmul (f16 input) ─────────────────────────────
//
// C[N×M] = B[N×K] × dequant(A[M×K])^T
// Same 64×32×32 tiling as Q8_0 blocked kernel.
// nl=2: each dequantize_q5_0_blocked call produces 16 values.
kernel void dequant_batch_q5_0_blocked_f16in(
    device const Q5_0_Block* A [[buffer(0)]],
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

    const short il0 = short(tiitg % NL0);
    short il = il0;

    uint blocks_per_row = K / Q5_0_BLOCK_VALUES;
    device const Q5_0_Block* x = A + uint(r0 + lr0) * blocks_per_row;

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
        dequantize_q5_0_blocked(x, il, temp_a);

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
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = *(device half2x4*)y;

        // Q5_0 nl=2: il cycles 0→1, x advances by 1 block when il wraps.
        il = 1 - il;
        x = (il == 0) ? x + 1 : x;
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

    // Boundary-checked output (always use BC path for MoE compatibility).
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

// ── Q5_0 matvec kernel ────────────────────────────────────────────────
//
// y[row] = dequant(A[row, :]) · x[:]
// Each threadgroup processes one row. Threads within a simdgroup each
// handle one element per block, then SIMD-reduce.
kernel void dequant_matvec_q5_0(
    device const Q5_0_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q5_0_BLOCK_VALUES;
    device const Q5_0_Block* a_row = A + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
        d = simd_broadcast(d, 0);

        // Extract the packed 5-bit value for this lane.
        // Lanes 0-15 handle the low nibble half, lanes 16-31 the high nibble half.
        uint qh_u32 = *(device const uint32_t *)a_row[b].qh;
        uint j = simd_lane & 15;  // index within the 16-byte qs array
        uint nibble;
        uint high_bit;
        if (simd_lane < 16) {
            nibble = (a_row[b].qs[j]) & 0x0F;
            high_bit = ((qh_u32 >> j) << 4) & 0x10;
        } else {
            nibble = (a_row[b].qs[j] >> 4) & 0x0F;
            high_bit = ((qh_u32 >> (j + 12))) & 0x10;
        }
        int q = int(nibble | high_bit) - 16;

        uint base = b * Q5_0_BLOCK_VALUES;
        sum += d * float(q) * x[base + simd_lane];
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

// ── Q5_1 dequant matvec ─────────────────────────────────────────────────
// y[row] = dot(dequant(A_q5_1[row, :]), x)
// Same as Q5_0 but uses per-block minimum `m` instead of fixed -16 offset.
kernel void dequant_matvec_q5_1(
    device const Q5_1_Block* A [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    uint blocks_per_row = K / Q5_1_BLOCK_VALUES;
    device const Q5_1_Block* a_row = A + row * blocks_per_row;
    constexpr uint num_simd_groups = DEQUANT_MATVEC_TG / 32;

    float sum = 0.0f;

    for (uint b = simd_id; b < blocks_per_row; b += num_simd_groups) {
        float d = (simd_lane == 0) ? float(a_row[b].d) : 0.0f;
        float m = (simd_lane == 0) ? float(a_row[b].m) : 0.0f;
        d = simd_broadcast(d, 0);
        m = simd_broadcast(m, 0);

        uint qh_u32 = *(device const uint32_t *)a_row[b].qh;
        uint j = simd_lane & 15;
        uint nibble;
        uint high_bit;
        if (simd_lane < 16) {
            nibble = (a_row[b].qs[j]) & 0x0F;
            high_bit = ((qh_u32 >> j) << 4) & 0x10;
        } else {
            nibble = (a_row[b].qs[j] >> 4) & 0x0F;
            high_bit = ((qh_u32 >> (j + 12))) & 0x10;
        }
        int q = int(nibble | high_bit);  // No -16: Q5_1 uses explicit m

        uint base = b * Q5_1_BLOCK_VALUES;
        sum += (d * float(q) + m) * x[base + simd_lane];
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


// ── Q5_1 blocked batch matmul with f16 input ────────────────────────────
// C[N×M] = B[N×K] × dequant(A[M×K])^T
// Identical to Q5_0 blocked batch but uses dequantize_q5_1_blocked (d*x + m).
kernel void dequant_batch_q5_1_blocked_f16in(
    device const Q5_1_Block* A [[buffer(0)]],
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

    const short il0 = short(tiitg % NL0);
    short il = il0;

    uint blocks_per_row = K / Q5_1_BLOCK_VALUES;
    device const Q5_1_Block* x = A + uint(r0 + lr0) * blocks_per_row;

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
        dequantize_q5_1_blocked(x, il, temp_a);

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
        *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = *(device half2x4*)y;

        il = 1 - il;
        x = (il == 0) ? x + 1 : x;
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
            for (; i < nr0 / 4; i++) { *(D4 + i) = *(C4 + i); }
            i *= 4;
            for (; i < nr0; i++) { *(D + i) = *(Cs + i); }
        }
    }
}
