// Q4_K dequant and matvec kernels.
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

kernel void dequant_batch_q4_k_blocked_silu(
    device const Q4_K_Block* A [[buffer(0)]],
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
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
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

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    const short offset1 = il0 / QK_NL;
    device const Q4_K_Block* x = A + uint(r0 + lr0) * blocks_per_row + offset1;

    const short iy = short(8 * (tiitg % NL1));
    // Two input pointers instead of one
    device const float* g_ptr = gate + uint(r1 + lr1) * K + iy;
    device const float* u_ptr = up   + uint(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q4k_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        // Fused B-loading: silu(gate) * up instead of plain B read
        {
            const short sx = short(tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            float4 gv0 = *(device float4*)(g_ptr);
            float4 gv1 = *(device float4*)(g_ptr + 4);
            float4 uv0 = *(device float4*)(u_ptr);
            float4 uv1 = *(device float4*)(u_ptr + 4);
            // silu(g) * u for each element, store as half2x4
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
                    tg_A0[r * PB_BK + c] = half(float(blk0.d) * sm0a.x * float(byte0 & 0x0F) - float(blk0.dmin) * sm0a.y);
                    tg_A1[r * PB_BK + c] = half(float(blk1.d) * sm1a.x * float(byte1 & 0x0F) - float(blk1.dmin) * sm1a.y);
                } else {
                    tg_A0[r * PB_BK + c] = half(float(blk0.d) * sm0b.x * float(byte0 >> 4) - float(blk0.dmin) * sm0b.y);
                    tg_A1[r * PB_BK + c] = half(float(blk1.d) * sm1b.x * float(byte1 >> 4) - float(blk1.dmin) * sm1b.y);
                }
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
                    tg_A0[r * P16_BK + c] = half(float(blk0.d) * sm0a.x * float(byte0 & 0x0F) - float(blk0.dmin) * sm0a.y);
                    tg_A1[r * P16_BK + c] = half(float(blk1.d) * sm1a.x * float(byte1 & 0x0F) - float(blk1.dmin) * sm1a.y);
                } else {
                    tg_A0[r * P16_BK + c] = half(float(blk0.d) * sm0b.x * float(byte0 >> 4) - float(blk0.dmin) * sm0b.y);
                    tg_A1[r * P16_BK + c] = half(float(blk1.d) * sm1b.x * float(byte1 >> 4) - float(blk1.dmin) * sm1b.y);
                }
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

kernel void dequant_matvec_silu_down_q4_k(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* gate   [[buffer(1)]],
    device const float* up     [[buffer(2)]],
    device float* y            [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    ushort ix = simd_lane / 8;
    ushort it = simd_lane % 8;
    ushort iq = it / 4;
    ushort ir = it % 4;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    // nr2: 2 SGs per TG, 2 rows per SG = 4 rows per TG
    constexpr uint SG_PER_TG = 2;
    constexpr uint ROWS_PER_SG = 2;
    uint first_row = (tg_id * SG_PER_TG + simd_id) * ROWS_PER_SG;
    if (first_row >= M) return;
    bool valid1 = (first_row + 1) < M;

    device const Q4_K_Block* a_row0 = A + first_row * blocks_per_row;
    device const Q4_K_Block* a_row1 = valid1 ? a_row0 + blocks_per_row : a_row0;

    float sumf0 = 0.0f, sumf1 = 0.0f;

    for (uint i = ix; i < blocks_per_row; i += 4) {
        // Compute silu(gate) * up inline instead of reading from pre-computed x
        uint base = i * Q4_K_BLOCK_VALUES;
        device const float* gp = gate + base + 64 * iq + 8 * ir;
        device const float* up_p = up + base + 64 * iq + 8 * ir;
        float yl0 = silu_f(gp[0]) * up_p[0];
        float yl1 = silu_f(gp[1]) * up_p[1];
        float yl2 = silu_f(gp[2]) * up_p[2];
        float yl3 = silu_f(gp[3]) * up_p[3];
        float yl4 = silu_f(gp[4]) * up_p[4];
        float yl5 = silu_f(gp[5]) * up_p[5];
        float yl6 = silu_f(gp[6]) * up_p[6];
        float yl7 = silu_f(gp[7]) * up_p[7];
        float yh0 = silu_f(gp[32]) * up_p[32];
        float yh1 = silu_f(gp[33]) * up_p[33];
        float yh2 = silu_f(gp[34]) * up_p[34];
        float yh3 = silu_f(gp[35]) * up_p[35];
        float yh4 = silu_f(gp[36]) * up_p[36];
        float yh5 = silu_f(gp[37]) * up_p[37];
        float yh6 = silu_f(gp[38]) * up_p[38];
        float yh7 = silu_f(gp[39]) * up_p[39];

        float suml0 = yl0 + yl1 + yl2 + yl3;
        float suml1 = yl4 + yl5 + yl6 + yl7;
        float sumh0 = yh0 + yh1 + yh2 + yh3;
        float sumh1 = yh4 + yh5 + yh6 + yh7;

        #define SILU_DOWN_Q4K_DOT(A_ROW, SUM_VAR)                              \
        {                                                                       \
            device const Q4_K_Block& blk = (A_ROW)[i];                         \
            device const ushort* sc16 = (device const ushort*)(blk.scales);    \
            device const uchar* sc8  = blk.scales;                             \
            float2 dh = float2(blk.d, blk.dmin);                              \
            device const uchar* qs = blk.qs + 32 * iq + 8 * ir;               \
            ushort s16_0 = sc16[iq];                                           \
            ushort s16_1 = sc16[iq + 4];                                       \
            float acc1 =                                                       \
                yl0 * (qs[0] & 0xF) + yl1 * (qs[1] & 0xF) +                   \
                yl2 * (qs[2] & 0xF) + yl3 * (qs[3] & 0xF) +                   \
                yl4 * (qs[4] & 0xF) + yl5 * (qs[5] & 0xF) +                   \
                yl6 * (qs[6] & 0xF) + yl7 * (qs[7] & 0xF);                    \
            float acc2 =                                                       \
                yh0 * (qs[0] >> 4) + yh1 * (qs[1] >> 4) +                     \
                yh2 * (qs[2] >> 4) + yh3 * (qs[3] >> 4) +                     \
                yh4 * (qs[4] >> 4) + yh5 * (qs[5] >> 4) +                     \
                yh6 * (qs[6] >> 4) + yh7 * (qs[7] >> 4);                      \
            (SUM_VAR) +=                                                       \
                dh[0] * (acc1 * float(s16_0 & kmask1) +                        \
                         acc2 * float((s16_1 & kmask2) | ((s16_0 & kmask3) >> 2)) * (1.f/16.f)) \
              - dh[1] * (suml0 * float(sc8[2*iq + 4 + 8*ir/32]) +             \
                         suml1 * float(sc8[2*iq + 5 + 8*ir/32]) +             \
                         sumh0 * float(sc8[2*iq + 4 + 8*ir/32 + 4]) +         \
                         sumh1 * float(sc8[2*iq + 5 + 8*ir/32 + 4])) * (1.f/256.f); \
        }

        SILU_DOWN_Q4K_DOT(a_row0, sumf0);
        if (valid1) SILU_DOWN_Q4K_DOT(a_row1, sumf1);
        #undef SILU_DOWN_Q4K_DOT
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    if (simd_lane == 0) {
        y[first_row] = sumf0;
        if (valid1) y[first_row + 1] = sumf1;
    }
}

kernel void dequant_matvec_gelu_down_q4_k(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* gate   [[buffer(1)]],
    device const float* up     [[buffer(2)]],
    device float* y            [[buffer(3)]],
    constant uint& M           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    ushort ix = simd_lane / 8;
    ushort it = simd_lane % 8;
    ushort iq = it / 4;
    ushort ir = it % 4;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    constexpr uint SG_PER_TG = 2;
    constexpr uint ROWS_PER_SG = 2;
    uint first_row = (tg_id * SG_PER_TG + simd_id) * ROWS_PER_SG;
    if (first_row >= M) return;
    bool valid1 = (first_row + 1) < M;

    device const Q4_K_Block* a_row0 = A + first_row * blocks_per_row;
    device const Q4_K_Block* a_row1 = valid1 ? a_row0 + blocks_per_row : a_row0;

    float sumf0 = 0.0f, sumf1 = 0.0f;

    for (uint i = ix; i < blocks_per_row; i += 4) {
        uint base = i * Q4_K_BLOCK_VALUES;
        device const float* gp = gate + base + 64 * iq + 8 * ir;
        device const float* up_p = up + base + 64 * iq + 8 * ir;
        float yl0 = gelu_mul_f32(gp[0], up_p[0]);
        float yl1 = gelu_mul_f32(gp[1], up_p[1]);
        float yl2 = gelu_mul_f32(gp[2], up_p[2]);
        float yl3 = gelu_mul_f32(gp[3], up_p[3]);
        float yl4 = gelu_mul_f32(gp[4], up_p[4]);
        float yl5 = gelu_mul_f32(gp[5], up_p[5]);
        float yl6 = gelu_mul_f32(gp[6], up_p[6]);
        float yl7 = gelu_mul_f32(gp[7], up_p[7]);
        float yh0 = gelu_mul_f32(gp[32], up_p[32]);
        float yh1 = gelu_mul_f32(gp[33], up_p[33]);
        float yh2 = gelu_mul_f32(gp[34], up_p[34]);
        float yh3 = gelu_mul_f32(gp[35], up_p[35]);
        float yh4 = gelu_mul_f32(gp[36], up_p[36]);
        float yh5 = gelu_mul_f32(gp[37], up_p[37]);
        float yh6 = gelu_mul_f32(gp[38], up_p[38]);
        float yh7 = gelu_mul_f32(gp[39], up_p[39]);

        float suml0 = yl0 + yl1 + yl2 + yl3;
        float suml1 = yl4 + yl5 + yl6 + yl7;
        float sumh0 = yh0 + yh1 + yh2 + yh3;
        float sumh1 = yh4 + yh5 + yh6 + yh7;

        #define GELU_DOWN_Q4K_DOT(A_ROW, SUM_VAR)                              \
        {                                                                       \
            device const Q4_K_Block& blk = (A_ROW)[i];                         \
            device const ushort* sc16 = (device const ushort*)(blk.scales);    \
            device const uchar* sc8  = blk.scales;                             \
            float2 dh = float2(blk.d, blk.dmin);                               \
            device const uchar* qs = blk.qs + 32 * iq + 8 * ir;                \
            ushort s16_0 = sc16[iq];                                            \
            ushort s16_1 = sc16[iq + 4];                                        \
            float acc1 =                                                        \
                yl0 * (qs[0] & 0xF) + yl1 * (qs[1] & 0xF) +                    \
                yl2 * (qs[2] & 0xF) + yl3 * (qs[3] & 0xF) +                    \
                yl4 * (qs[4] & 0xF) + yl5 * (qs[5] & 0xF) +                    \
                yl6 * (qs[6] & 0xF) + yl7 * (qs[7] & 0xF);                     \
            float acc2 =                                                        \
                yh0 * (qs[0] >> 4) + yh1 * (qs[1] >> 4) +                      \
                yh2 * (qs[2] >> 4) + yh3 * (qs[3] >> 4) +                      \
                yh4 * (qs[4] >> 4) + yh5 * (qs[5] >> 4) +                      \
                yh6 * (qs[6] >> 4) + yh7 * (qs[7] >> 4);                       \
            (SUM_VAR) +=                                                        \
                dh[0] * (acc1 * float(s16_0 & kmask1) +                        \
                         acc2 * float((s16_1 & kmask2) | ((s16_0 & kmask3) >> 2)) * (1.f/16.f)) \
              - dh[1] * (suml0 * float(sc8[2*iq + 4 + 8*ir/32]) +              \
                         suml1 * float(sc8[2*iq + 5 + 8*ir/32]) +              \
                         sumh0 * float(sc8[2*iq + 4 + 8*ir/32 + 4]) +          \
                         sumh1 * float(sc8[2*iq + 5 + 8*ir/32 + 4])) * (1.f/256.f); \
        }

        GELU_DOWN_Q4K_DOT(a_row0, sumf0);
        if (valid1) GELU_DOWN_Q4K_DOT(a_row1, sumf1);
        #undef GELU_DOWN_Q4K_DOT
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    if (simd_lane == 0) {
        y[first_row] = sumf0;
        if (valid1) y[first_row + 1] = sumf1;
    }
}

kernel void dequant_matvec_pair_q4_k(
    device const Q4_K_Block* A0 [[buffer(0)]],
    device const Q4_K_Block* A1 [[buffer(1)]],
    device const float* x       [[buffer(2)]],
    device float* y0            [[buffer(3)]],
    device float* y1            [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& K            [[buffer(6)]],
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    ushort ix = simd_lane / 8;
    ushort it = simd_lane % 8;
    ushort iq = it / 4;
    ushort ir = it % 4;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    uint first_row = (tg_id * Q4K_NR2_SG_PER_TG + simd_id) * Q4K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;
    bool valid1 = (first_row + 1) < M;

    device const Q4_K_Block* a0_row0 = A0 + first_row * blocks_per_row;
    device const Q4_K_Block* a0_row1 = A0 + (first_row + 1) * blocks_per_row;
    device const Q4_K_Block* a1_row0 = A1 + first_row * blocks_per_row;
    device const Q4_K_Block* a1_row1 = A1 + (first_row + 1) * blocks_per_row;

    float sum_a0_r0 = 0.0f, sum_a0_r1 = 0.0f;
    float sum_a1_r0 = 0.0f, sum_a1_r1 = 0.0f;

    for (uint i = ix; i < blocks_per_row; i += 4) {
        // Load x values once, reuse for all 4 dot products
        device const float* y_base = x + i * Q4_K_BLOCK_VALUES + 64 * iq + 8 * ir;
        float yl0 = y_base[0];
        float yl1 = y_base[1];
        float yl2 = y_base[2];
        float yl3 = y_base[3];
        float yl4 = y_base[4];
        float yl5 = y_base[5];
        float yl6 = y_base[6];
        float yl7 = y_base[7];
        float yh0 = y_base[32];
        float yh1 = y_base[33];
        float yh2 = y_base[34];
        float yh3 = y_base[35];
        float yh4 = y_base[36];
        float yh5 = y_base[37];
        float yh6 = y_base[38];
        float yh7 = y_base[39];

        float suml0 = yl0 + yl1 + yl2 + yl3;
        float suml1 = yl4 + yl5 + yl6 + yl7;
        float sumh0 = yh0 + yh1 + yh2 + yh3;
        float sumh1 = yh4 + yh5 + yh6 + yh7;

        // Macro: compute one Q4_K block dot for row r of weight A
        #define PAIR_Q4K_DOT(A_ROW, SUM_VAR)                                   \
        {                                                                       \
            device const Q4_K_Block& blk = (A_ROW)[i];                         \
            device const ushort* sc16 = (device const ushort*)(blk.scales);    \
            device const uchar* sc8  = blk.scales;                             \
            float2 dh = float2(blk.d, blk.dmin);                              \
            device const uchar* qs = blk.qs + 32 * iq + 8 * ir;               \
            ushort s16_0 = sc16[iq];                                           \
            ushort s16_1 = sc16[iq + 4];                                       \
            float acc1 =                                                       \
                yl0 * (qs[0] & 0xF) + yl1 * (qs[1] & 0xF) +                   \
                yl2 * (qs[2] & 0xF) + yl3 * (qs[3] & 0xF) +                   \
                yl4 * (qs[4] & 0xF) + yl5 * (qs[5] & 0xF) +                   \
                yl6 * (qs[6] & 0xF) + yl7 * (qs[7] & 0xF);                    \
            float acc2 =                                                       \
                yh0 * (qs[0] >> 4) + yh1 * (qs[1] >> 4) +                     \
                yh2 * (qs[2] >> 4) + yh3 * (qs[3] >> 4) +                     \
                yh4 * (qs[4] >> 4) + yh5 * (qs[5] >> 4) +                     \
                yh6 * (qs[6] >> 4) + yh7 * (qs[7] >> 4);                      \
            (SUM_VAR) +=                                                       \
                dh[0] * (acc1 * float(s16_0 & kmask1) +                        \
                         acc2 * float((s16_1 & kmask2) | ((s16_0 & kmask3) >> 2)) * (1.f/16.f)) \
              - dh[1] * (suml0 * float(sc8[2*iq + 4 + 8*ir/32]) +             \
                         suml1 * float(sc8[2*iq + 5 + 8*ir/32]) +             \
                         sumh0 * float(sc8[2*iq + 4 + 8*ir/32 + 4]) +         \
                         sumh1 * float(sc8[2*iq + 5 + 8*ir/32 + 4])) * (1.f/256.f); \
        }

        PAIR_Q4K_DOT(a0_row0, sum_a0_r0);
        if (valid1) PAIR_Q4K_DOT(a0_row1, sum_a0_r1);
        PAIR_Q4K_DOT(a1_row0, sum_a1_r0);
        if (valid1) PAIR_Q4K_DOT(a1_row1, sum_a1_r1);

        #undef PAIR_Q4K_DOT
    }

    // simd_sum reduction (no cross-SG barrier needed)
    sum_a0_r0 = simd_sum(sum_a0_r0);
    sum_a0_r1 = simd_sum(sum_a0_r1);
    sum_a1_r0 = simd_sum(sum_a1_r0);
    sum_a1_r1 = simd_sum(sum_a1_r1);

    if (simd_lane == 0) {
        y0[first_row] = sum_a0_r0;
        if (valid1) y0[first_row + 1] = sum_a0_r1;
        y1[first_row] = sum_a1_r0;
        if (valid1) y1[first_row + 1] = sum_a1_r1;
    }
}

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

kernel void moe_mul_mat_selected_q4_k_matvec(
    device const Q4_K_Block* A       [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const int32_t* selected   [[buffer(2)]],
    device float* y                  [[buffer(3)]],
    constant uint& M                 [[buffer(4)]],
    constant uint& K                 [[buffer(5)]],
    constant uint& n_selected        [[buffer(6)]],
    constant uint& weight_stride     [[buffer(7)]],
    constant uint& input_is_slot_major [[buffer(8)]],
    uint3 tgpig                      [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    if (tgpig.z >= n_selected) return;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const uint slot = tgpig.z;
    const uint expert = uint(selected[slot]);

    ushort ix = simd_lane / 8;
    ushort it = simd_lane % 8;
    ushort iq = it / 4;
    ushort ir = it % 4;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    uint first_row = (tgpig.x * Q4K_NR2_SG_PER_TG + simd_id) * Q4K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    bool valid1 = (first_row + 1) < M;
    device const Q4_K_Block* base = A + expert * weight_stride;
    device const Q4_K_Block* row0 = base + first_row * blocks_per_row;
    device const Q4_K_Block* row1 = valid1 ? row0 + blocks_per_row : row0;
    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    device const float* y4 = x + input_row * K + ix * Q4_K_BLOCK_VALUES + 64 * iq + 8 * ir;

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
        device float* y_slot = y + slot * M;
        y_slot[first_row] = sum0;
        if (valid1) {
            y_slot[first_row + 1] = sum1;
        }
    }
}

kernel void moe_mul_mat_selected_pair_q4_k_matvec(
    device const Q4_K_Block* A0      [[buffer(0)]],
    device const Q4_K_Block* A1      [[buffer(1)]],
    device const float* x            [[buffer(2)]],
    device const int32_t* selected   [[buffer(3)]],
    device float* y0                 [[buffer(4)]],
    device float* y1                 [[buffer(5)]],
    constant uint& M                 [[buffer(6)]],
    constant uint& K                 [[buffer(7)]],
    constant uint& n_selected        [[buffer(8)]],
    constant uint& weight_stride0    [[buffer(9)]],
    constant uint& weight_stride1    [[buffer(10)]],
    constant uint& input_is_slot_major [[buffer(11)]],
    uint3 tgpig                      [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    if (tgpig.z >= n_selected) return;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const uint slot = tgpig.z;
    const uint expert = uint(selected[slot]);

    ushort ix = simd_lane / 8;
    ushort it = simd_lane % 8;
    ushort iq = it / 4;
    ushort ir = it % 4;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    uint first_row = (tgpig.x * Q4K_NR2_SG_PER_TG + simd_id) * Q4K_NR2_ROWS_PER_SG;
    if (first_row >= M) return;

    bool valid1 = (first_row + 1) < M;
    device const Q4_K_Block* base0 = A0 + expert * weight_stride0;
    device const Q4_K_Block* row00 = base0 + first_row * blocks_per_row;
    device const Q4_K_Block* row01 = valid1 ? row00 + blocks_per_row : row00;
    device const Q4_K_Block* base1 = A1 + expert * weight_stride1;
    device const Q4_K_Block* row10 = base1 + first_row * blocks_per_row;
    device const Q4_K_Block* row11 = valid1 ? row10 + blocks_per_row : row10;

    const uint input_row = input_is_slot_major != 0 ? slot : 0;
    device const float* y4 = x + input_row * K + ix * Q4_K_BLOCK_VALUES + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf0[Q4K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};
    float sumf1[Q4K_NR2_ROWS_PER_SG] = {0.0f, 0.0f};

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
            device const Q4_K_Block& blk0 = ((row == 0) ? row00 : row01)[ib];
            device const Q4_K_Block& blk1 = ((row == 0) ? row10 : row11)[ib];
            device const ushort* sc0 = (device const ushort*) blk0.scales + iq;
            device const ushort* sc1p = (device const ushort*) blk1.scales + iq;
            device const ushort* q01 = (device const ushort*) blk0.qs + 16 * iq + 4 * ir;
            device const ushort* q11 = (device const ushort*) blk1.qs + 16 * iq + 4 * ir;
            device const half* dh0 = &blk0.d;
            device const half* dh1 = &blk1.d;

            sc16[0] = sc0[0] & kmask1;
            sc16[1] = sc0[2] & kmask1;
            sc16[2] = ((sc0[4] >> 0) & kmask2) | ((sc0[0] & kmask3) >> 2);
            sc16[3] = ((sc0[4] >> 4) & kmask2) | ((sc0[2] & kmask3) >> 2);
            device const ushort* q02 = q01 + 32;

            float4 acc10 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc20 = {0.0f, 0.0f, 0.0f, 0.0f};
            FOR_UNROLL (ushort i = 0; i < 4; ++i) {
                acc10[0] += yl[2 * i + 0] * (q01[i] & 0x000F);
                acc10[1] += yl[2 * i + 1] * (q01[i] & 0x0F00);
                acc10[2] += yl[2 * i + 8] * (q01[i] & 0x00F0);
                acc10[3] += yl[2 * i + 9] * (q01[i] & 0xF000);
                acc20[0] += yh[2 * i + 0] * (q02[i] & 0x000F);
                acc20[1] += yh[2 * i + 1] * (q02[i] & 0x0F00);
                acc20[2] += yh[2 * i + 8] * (q02[i] & 0x00F0);
                acc20[3] += yh[2 * i + 9] * (q02[i] & 0xF000);
            }
            sumf0[row] += float(dh0[0]) * (
                (acc10[0] + (1.0f / 256.0f) * acc10[1]) * float(sc8[0]) +
                (acc10[2] + (1.0f / 256.0f) * acc10[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                (acc20[0] + (1.0f / 256.0f) * acc20[1]) * float(sc8[4]) +
                (acc20[2] + (1.0f / 256.0f) * acc20[3]) * float(sc8[5]) * (1.0f / 16.0f)
            ) - float(dh0[1]) * (
                sumy[0] * float(sc8[2]) +
                sumy[1] * float(sc8[3]) +
                sumy[2] * float(sc8[6]) +
                sumy[3] * float(sc8[7])
            );

            sc16[0] = sc1p[0] & kmask1;
            sc16[1] = sc1p[2] & kmask1;
            sc16[2] = ((sc1p[4] >> 0) & kmask2) | ((sc1p[0] & kmask3) >> 2);
            sc16[3] = ((sc1p[4] >> 4) & kmask2) | ((sc1p[2] & kmask3) >> 2);
            device const ushort* q12 = q11 + 32;

            float4 acc11 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc21 = {0.0f, 0.0f, 0.0f, 0.0f};
            FOR_UNROLL (ushort i = 0; i < 4; ++i) {
                acc11[0] += yl[2 * i + 0] * (q11[i] & 0x000F);
                acc11[1] += yl[2 * i + 1] * (q11[i] & 0x0F00);
                acc11[2] += yl[2 * i + 8] * (q11[i] & 0x00F0);
                acc11[3] += yl[2 * i + 9] * (q11[i] & 0xF000);
                acc21[0] += yh[2 * i + 0] * (q12[i] & 0x000F);
                acc21[1] += yh[2 * i + 1] * (q12[i] & 0x0F00);
                acc21[2] += yh[2 * i + 8] * (q12[i] & 0x00F0);
                acc21[3] += yh[2 * i + 9] * (q12[i] & 0xF000);
            }
            sumf1[row] += float(dh1[0]) * (
                (acc11[0] + (1.0f / 256.0f) * acc11[1]) * float(sc8[0]) +
                (acc11[2] + (1.0f / 256.0f) * acc11[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                (acc21[0] + (1.0f / 256.0f) * acc21[1]) * float(sc8[4]) +
                (acc21[2] + (1.0f / 256.0f) * acc21[3]) * float(sc8[5]) * (1.0f / 16.0f)
            ) - float(dh1[1]) * (
                sumy[0] * float(sc8[2]) +
                sumy[1] * float(sc8[3]) +
                sumy[2] * float(sc8[6]) +
                sumy[3] * float(sc8[7])
            );
        }

        y4 += 4 * Q4_K_BLOCK_VALUES;
    }

    float sum00 = simd_sum(sumf0[0]);
    float sum01 = simd_sum(sumf0[1]);
    float sum10 = simd_sum(sumf1[0]);
    float sum11 = simd_sum(sumf1[1]);
    if (simd_lane == 0) {
        device float* y0_slot = y0 + slot * M;
        device float* y1_slot = y1 + slot * M;
        y0_slot[first_row] = sum00;
        y1_slot[first_row] = sum10;
        if (valid1) {
            y0_slot[first_row + 1] = sum01;
            y1_slot[first_row + 1] = sum11;
        }
    }
}

kernel void dequant_matvec_q4_k_ilp4(
    device const Q4_K_Block* A [[buffer(0)]],
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

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block* a_row = A + first_row * blocks_per_row;

    uint tid = simd_lane / 4;
    uint ix = simd_lane % 4;
    uint iq = tid / 4;
    uint ir = tid % 4;

    uint l0 = 8 * ir;
    uint q_offset = 32 * iq + l0;
    uint y_offset = 64 * iq + l0;

    float sum = 0.0f;
    for (uint b = ix; b < blocks_per_row; b += 4) {
        device const Q4_K_Block& blk = a_row[b];
        device const float* y1 = x + b * Q4_K_BLOCK_VALUES + y_offset;
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

            acc0 += yl0 * float(ql_lo_hi & 0x0F);
            acc1 += yl1 * float(ql_lo_hi >> 4);
            acc2 += yh0 * float(qh_lo_hi & 0x0F);
            acc3 += yh1 * float(qh_lo_hi >> 4);

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

kernel void dequant_small_batch_q4_k(
    device const Q4_K_Block* A [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    uint2 tgpig                [[threadgroup_position_in_grid]],
    ushort tiitg               [[thread_index_in_threadgroup]],
    ushort sgitg               [[simdgroup_index_in_threadgroup]]
) {
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 4;  // 32 / nxpsg
    constexpr ushort nsg = 2;

    ushort tx = tiitg % nxpsg;   // 0..7: K-reduction cooperating index
    ushort ty = tiitg / nxpsg;   // 0..3: weight-row index within SG

    // Weight row for this thread
    uint i01 = tgpig.x * (nypsg * nsg) + nypsg * sgitg + ty;
    if (i01 >= M) return;

    // First token index for this threadgroup
    uint i11_base = tgpig.y * EXT_R1PTG;

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block* a_row = A + i01 * blocks_per_row;

    // Accumulators for r1ptg tokens
    float sumf[5] = {};  // max r1ptg = 5

    // Activation pointers for each token
    // (each thread covers a strided K-range via tx)
    for (uint ich = tx; ich < blocks_per_row; ich += nxpsg) {
        // Dequantize one Q4_K block (256 values → 16 chunks of 16)
        device const Q4_K_Block& blk = a_row[ich];
        float d_all = float(blk.d);
        float dmin = float(blk.dmin);

        // Process all 16 chunks of this block
        for (uint ch = 0; ch < 16; ch++) {
            ushort is = (ch / 4) * 2;
            ushort il = ch & 3;
            device const uchar* q = blk.qs + (ch / 4) * 32 + 16 * (ch & 1);

            uchar2 sc = get_scale_min_k4_just2(is, il / 2, blk.scales);
            float dd = il < 2 ? d_all : d_all / 16.0f;
            float dl = dd * float(sc[0]);
            float ml = dmin * float(sc[1]);
            ushort mask = il < 2 ? 0x0F : 0xF0;

            // 16 dequantized values for this chunk
            uint k_base = ich * Q4_K_BLOCK_VALUES + ch * 16;

            for (ushort ir1 = 0; ir1 < EXT_R1PTG; ir1++) {
                uint i11 = i11_base + ir1;
                if (i11 >= N) break;
                device const float* y = B + i11 * K + k_base;

                float dot = 0.0f;
                for (ushort j = 0; j < 16; j++) {
                    dot += (dl * float(q[j] & mask) - ml) * y[j];
                }
                sumf[ir1] += dot;
            }
        }
    }

    // Reduction across nxpsg=8 threads via simd_shuffle_down
    for (ushort ir1 = 0; ir1 < EXT_R1PTG; ir1++) {
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 4);
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 2);
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 1);
    }

    // Write results (only tx==0 thread has the final sum)
    if (tx == 0) {
        for (ushort ir1 = 0; ir1 < EXT_R1PTG; ir1++) {
            uint i11 = i11_base + ir1;
            if (i11 < N) {
                C[i11 * M + i01] = sumf[ir1];
            }
        }
    }
}

kernel void moe_mul_mat_id_q4_k(
    device const Q4_K_Block *weights [[buffer(0)]],
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

    // Tile positions.
    const uint tile_col = group_id.x * DQ_BN;  // Token tile (column = token index)
    const uint tile_row = group_id.y * DQ_BM;  // Output row tile

    // Early exit: no tokens for this expert, or this tile is past the token count.
    if (n_assigned == 0 || tile_col >= n_assigned) return;

    // Expert weight base: expert * (M * blocks_per_row) blocks.
    const uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block *W = weights + expert * weight_stride;

    // hids for this expert.
    device const int32_t *expert_hids = hids + expert * n_tokens;

    threadgroup float tg_A[DQ_BM * DQ_BK];
    threadgroup float tg_B[DQ_BK * DQ_BN];

    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    acc0 = simdgroup_float8x8(0);
    acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0);
    acc3 = simdgroup_float8x8(0);

    for (uint kt = 0; kt < K; kt += DQ_BK) {
        uint block_idx = kt / Q4_K_BLOCK_VALUES;
        uint pair = (kt % Q4_K_BLOCK_VALUES) / DQ_BK;

        // Load A tile: dequant weight rows [tile_row..tile_row+32, kt..kt+64].
        for (uint i = tid; i < DQ_BM * DQ_BK; i += DQ_TG) {
            uint r = i / DQ_BK;
            uint c = i % DQ_BK;
            uint global_r = tile_row + r;

            if (global_r < M) {
                device const Q4_K_Block& blk = W[global_r * blocks_per_row + block_idx];
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

        // Load B tile: input rows for assigned tokens.
        // B[r, c] = input[token_of(hids[tile_col + c]), kt + r]
        for (uint i = tid; i < DQ_BK * DQ_BN; i += DQ_TG) {
            uint r = i / DQ_BN;  // K dimension
            uint c = i % DQ_BN;  // Token slot
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

        // Simdgroup matmul.
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

    // Store output via routing index.
    threadgroup float out_tile[DQ_BM * DQ_BN];
    simdgroup_store(acc0, &out_tile[simd_id * 8 * DQ_BN + 0],  DQ_BN);
    simdgroup_store(acc1, &out_tile[simd_id * 8 * DQ_BN + 8],  DQ_BN);
    simdgroup_store(acc2, &out_tile[simd_id * 8 * DQ_BN + 16], DQ_BN);
    simdgroup_store(acc3, &out_tile[simd_id * 8 * DQ_BN + 24], DQ_BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to output: output[hid * M + row] = out_tile[row, col]
    // hid = hids[expert * n_tokens + tile_col + c], output indexed by flat hid.
    for (uint i = tid; i < DQ_BM * DQ_BN; i += DQ_TG) {
        uint r = i / DQ_BN;
        uint c = i % DQ_BN;
        uint gr = tile_row + r;
        uint slot = tile_col + c;

        if (gr < M && slot < n_assigned) {
            int32_t hid = expert_hids[slot];
            // Output layout: [n_tokens * n_expert_used, M]
            output[hid * M + gr] = out_tile[r * DQ_BN + c];
        }
    }
}

kernel void moe_mul_mat_id_q4_k_blocked(
    device const Q4_K_Block *weights [[buffer(0)]],
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

    // Weight pointer for this expert.
    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block *W = weights + expert * weight_stride;
    const short offset1 = il0 / QK_NL;
    device const Q4_K_Block *x = W + uint(r0 + lr0) * blocks_per_row + offset1;

    // Input pointer via routing: hids[expert * n_tokens + r1 + lr1] gives flat index.
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
        dequantize_q4k_blocked(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        // B-tile: vectorized load (8 elements per thread via float2x4 → half2x4).
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

    // Output via routing index.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *temp_str = ((threadgroup float *)shmem)
        + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                        NR0, ulong2(0, 0), false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All 4 simdgroups write output in parallel (matching llama.cpp pattern).
    // Each simdgroup handles every 4th column, all 32 lanes write rows.
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

kernel void moe_mul_mat_selected_q4_k_blocked(
    device const Q4_K_Block *weights       [[buffer(0)]],
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

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block *W = weights + expert * weight_stride;
    const short offset1 = il0 / QK_NL;
    device const Q4_K_Block *x = W + uint(r0 + lr0) * blocks_per_row + offset1;
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
        dequantize_q4k_blocked(x, il, temp_a);

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

kernel void moe_mul_mat_selected_weighted_q4_k_blocked(
    device const Q4_K_Block *weights       [[buffer(0)]],
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

    const uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    for (uint slot = 0; slot < n_selected; ++slot) {
        const uint expert = uint(selected_experts[slot]);
        const float route_weight = expert_weights[slot];
        short il = il0;

        device const Q4_K_Block *W = weights + expert * weight_stride;
        const short offset1 = il0 / QK_NL;
        device const Q4_K_Block *x = W + uint(r0 + lr0) * blocks_per_row + offset1;
        device const float *y = input + slot * K + short(8 * (tiitg % NL1));

        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];
        simdgroup_float8x8 mc[8];
        for (short i = 0; i < 8; i++) {
            mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }

        for (uint loop_k = 0; loop_k < K; loop_k += NK) {
            half4x4 temp_a;
            dequantize_q4k_blocked(x, il, temp_a);

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
            device float *D = output + uint(r0);
            device float4 *D4 = (device float4 *)D;
            threadgroup float4 *C4 = (threadgroup float4 *)Cs;

            int i = lane;
            for (; i < nr0 / 4; i += 32) {
                *(D4 + i) = *(D4 + i) + route_weight * *(C4 + i);
            }
            i = (4 * (nr0 / 4)) + lane;
            for (; i < nr0; i += 32) {
                *(D + i) += route_weight * *(Cs + i);
            }
        }
    }
}

kernel void moe_mul_mat_selected_pair_q4_k_blocked(
    device const Q4_K_Block *weights0      [[buffer(0)]],
    device const Q4_K_Block *weights1      [[buffer(1)]],
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

    uint blocks_per_row = K / Q4_K_BLOCK_VALUES;
    device const Q4_K_Block *W0 = weights0 + expert * weight_stride0;
    device const Q4_K_Block *W1 = weights1 + expert * weight_stride1;
    const short offset1 = il0 / QK_NL;
    device const Q4_K_Block *x0 = W0 + uint(r0 + lr0) * blocks_per_row + offset1;
    device const Q4_K_Block *x1 = W1 + uint(r0 + lr0) * blocks_per_row + offset1;
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
        dequantize_q4k_blocked(x0, il, temp_a0);
        dequantize_q4k_blocked(x1, il, temp_a1);

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
