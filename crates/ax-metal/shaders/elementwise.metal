// AX Engine — Elementwise compute shaders
//
// GPU kernels for operations that were previously CPU-only in the forward pass:
//   1. rms_norm_f32       — in-place RMSNorm with weight
//   2. rms_norm_out_f32   — RMSNorm writing to separate output buffer
//   3. rope_f32           — Rotary Position Embedding on Q and K vectors
//   4. per_head_rms_norm_f32 — Per-head RMSNorm (Gemma3 QK norm)
//   5. gelu_elementwise_mul_f32 — GELU(gate) * up (Gemma3 FFN)
//   6. silu_elementwise_mul_f32 — SiLU(gate) * up (LLaMA FFN)
//   7. elementwise_add_f32 — a[i] += b[i]
//
// All kernels operate on f32 vectors. Reduction kernels use the same
// two-level reduction pattern as matvec_f32 in matmul.metal:
//   1. Strided accumulation across threads
//   2. simd_sum within SIMD groups (32 threads)
//   3. Cross-SIMD reduction via threadgroup memory

#include <metal_stdlib>
using namespace metal;

constant uint NORM_TG_SIZE = 256;

// ── RMSNorm (in-place) ──────────────────────────────────────────────
//
// x[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
//
// One threadgroup of 256 threads processes one vector.
// Grid: 1 threadgroup × 1.

kernel void rms_norm_f32(
    device float* x          [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    constant float& eps      [[buffer(3)]],
    uint lid                 [[thread_index_in_threadgroup]],
    uint simd_lane           [[thread_index_in_simdgroup]],
    uint simd_id             [[simdgroup_index_in_threadgroup]]
) {
    // Step 1: Each thread accumulates partial sum of squares
    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = x[i];
        sum_sq += v * v;
    }

    // Step 2: SIMD-level reduction
    sum_sq = simd_sum(sum_sq);

    // Step 3: Cross-SIMD reduction via threadgroup memory
    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD group reduces across all groups
    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;

    // Step 4: Normalize and apply weight
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

// ── RMSNorm (out-of-place) ──────────────────────────────────────────
//
// out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
//
// Same reduction pattern, writes to separate output buffer.

kernel void rms_norm_out_f32(
    device const float* x    [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    constant float& eps      [[buffer(4)]],
    uint lid                 [[thread_index_in_threadgroup]],
    uint simd_lane           [[thread_index_in_simdgroup]],
    uint simd_id             [[simdgroup_index_in_threadgroup]]
) {
    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = x[i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;

    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// ── RMSNorm Batch (out-of-place) ────────────────────────────────────
//
// For each row r in [0, n_rows):
//   out[r, i] = x[r, i] * weight[i] / sqrt(mean(x[r, :]^2) + eps)
//
// x/out are contiguous row-major buffers with row stride n.
// Grid: n_rows threadgroups × 256 threads (one threadgroup per row).

kernel void rms_norm_out_batch_f32(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out          [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    constant uint& n_rows      [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= n_rows) return;

    uint base = row * n;

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = x[base + i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;

    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        out[base + i] = x[base + i] * inv_rms * weight[i];
    }
}

// ── RMSNorm Batch (out-of-place, f16 output) ─────────────────────────
//
// For each row r in [0, n_rows):
//   out[r, i] = half(x[r, i] * weight[i] / sqrt(mean(x[r, :]^2) + eps))
//
// x is f32 row-major, out is f16 row-major with the same stride n.

kernel void rms_norm_out_batch_f16(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device half* out           [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    constant uint& n_rows      [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= n_rows) return;

    uint base = row * n;

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = x[base + i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;

    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        out[base + i] = half(x[base + i] * inv_rms * weight[i]);
    }
}

// ── RMSNorm Batch (in-place) ────────────────────────────────────────
//
// For each row r in [0, n_rows):
//   x[r, i] = x[r, i] * weight[i] / sqrt(mean(x[r, :]^2) + eps)
//
// x is contiguous row-major with row stride n.
// Grid: n_rows threadgroups × 256 threads (one threadgroup per row).

kernel void rms_norm_batch_f32(
    device float* x            [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    constant uint& n_rows      [[buffer(3)]],
    constant float& eps        [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= n_rows) return;

    uint base = row * n;

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = x[base + i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        x[base + i] = x[base + i] * inv_rms * weight[i];
    }
}

// ── Residual Add + RMSNorm Batch (in-place hidden + out norm) ──────────────
//
// For each row r in [0, n_rows):
//   tmp[i]      = hidden[r, i] + addend[r, i]
//   hidden[r,i] = tmp[i]
//   norm_out[r,i] = tmp[i] * weight[i] / sqrt(mean(tmp[:]²) + eps)
//
// hidden/addend/norm_out are contiguous row-major buffers with row stride n.
// Grid: n_rows threadgroups × 256 threads (one threadgroup per row).

kernel void residual_add_rms_norm_out_batch_f32(
    device float* hidden          [[buffer(0)]],
    device const float* addend    [[buffer(1)]],
    device const float* weight    [[buffer(2)]],
    device float* norm_out        [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    constant uint& n_rows         [[buffer(5)]],
    constant float& eps           [[buffer(6)]],
    uint row                      [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (row >= n_rows) return;

    uint base = row * n;

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = hidden[base + i] + addend[base + i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;

    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = hidden[base + i] + addend[base + i];
        hidden[base + i] = v;
        norm_out[base + i] = v * inv_rms * weight[i];
    }
}

// ── Residual Add + RMSNorm Batch (in-place hidden + out norm f16) ────────
//
// For each row r in [0, n_rows):
//   tmp[i]      = hidden[r, i] + addend[r, i]
//   hidden[r,i] = tmp[i]
//   norm_out[r,i] = half(tmp[i] * weight[i] / sqrt(mean(tmp[:]²) + eps))

kernel void residual_add_rms_norm_out_batch_f16(
    device float* hidden          [[buffer(0)]],
    device const float* addend    [[buffer(1)]],
    device const float* weight    [[buffer(2)]],
    device half* norm_out         [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    constant uint& n_rows         [[buffer(5)]],
    constant float& eps           [[buffer(6)]],
    uint row                      [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (row >= n_rows) return;

    uint base = row * n;

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = hidden[base + i] + addend[base + i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;

    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = hidden[base + i] + addend[base + i];
        hidden[base + i] = v;
        norm_out[base + i] = half(v * inv_rms * weight[i]);
    }
}

// ── RoPE ────────────────────────────────────────────────────────────
//
// Applies rotary position embeddings to Q and K vectors.
// Processes both Q (n_q_heads heads) and K (n_kv_heads heads) in one dispatch.
//
// For each pair (x[2i], x[2i+1]):
//   freq = 1 / (base ^ (2i / head_dim))
//   theta = position * freq
//   x'[2i]   = x[2i]*cos(theta) - x[2i+1]*sin(theta)
//   x'[2i+1] = x[2i]*sin(theta) + x[2i+1]*cos(theta)
//
// Grid: ceil(total_pairs / 256) threadgroups × 256 threads.
// total_pairs = (n_q_heads + n_kv_heads) * half_dim

kernel void rope_f32(
    device float* q            [[buffer(0)]],
    device float* k            [[buffer(1)]],
    constant uint& n_q_heads   [[buffer(2)]],
    constant uint& n_kv_heads  [[buffer(3)]],
    constant uint& head_dim    [[buffer(4)]],
    constant float& position   [[buffer(5)]],
    constant float& freq_base  [[buffer(6)]],
    uint gid                   [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint total_q_pairs = n_q_heads * half_dim;
    uint total_pairs = total_q_pairs + n_kv_heads * half_dim;

    if (gid >= total_pairs) return;

    // Determine which buffer and which head/pair index
    device float* buf;
    uint pair_in_buf;
    if (gid < total_q_pairs) {
        buf = q;
        pair_in_buf = gid;
    } else {
        buf = k;
        pair_in_buf = gid - total_q_pairs;
    }

    uint i = pair_in_buf % half_dim;  // pair index within head
    uint head = pair_in_buf / half_dim;
    uint offset = head * head_dim + 2 * i;

    float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
    float theta = position * freq;
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    float v0 = buf[offset];
    float v1 = buf[offset + 1];
    buf[offset]     = v0 * cos_t - v1 * sin_t;
    buf[offset + 1] = v0 * sin_t + v1 * cos_t;
}

// ── RoPE Batch ───────────────────────────────────────────────────────
//
// Applies RoPE to batched Q/K buffers:
//   Q: [n_rows, n_q_heads, head_dim]
//   K: [n_rows, n_kv_heads, head_dim]
//
// Position for row r:
//   pos = start_pos + r * pos_step
//
// Grid: ceil(total_pairs / 256) threadgroups × 256 threads.
// total_pairs = n_rows * (n_q_heads + n_kv_heads) * (head_dim/2)

kernel void rope_batch_f32(
    device float* q            [[buffer(0)]],
    device float* k            [[buffer(1)]],
    constant uint& n_rows      [[buffer(2)]],
    constant uint& n_q_heads   [[buffer(3)]],
    constant uint& n_kv_heads  [[buffer(4)]],
    constant uint& head_dim    [[buffer(5)]],
    constant float& start_pos  [[buffer(6)]],
    constant float& pos_step   [[buffer(7)]],
    constant float& freq_base  [[buffer(8)]],
    uint gid                   [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint pairs_per_row = (n_q_heads + n_kv_heads) * half_dim;
    uint total_pairs = n_rows * pairs_per_row;
    if (gid >= total_pairs) return;

    uint row = gid / pairs_per_row;
    uint local = gid % pairs_per_row;

    uint q_pairs = n_q_heads * half_dim;
    device float* buf;
    uint vec_base;
    uint i;
    if (local < q_pairs) {
        uint head = local / half_dim;
        i = local % half_dim;
        buf = q;
        vec_base = row * (n_q_heads * head_dim) + head * head_dim;
    } else {
        uint k_local = local - q_pairs;
        uint head = k_local / half_dim;
        i = k_local % half_dim;
        buf = k;
        vec_base = row * (n_kv_heads * head_dim) + head * head_dim;
    }

    uint offset = vec_base + 2 * i;
    float position = start_pos + float(row) * pos_step;
    float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
    float theta = position * freq;
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    float v0 = buf[offset];
    float v1 = buf[offset + 1];
    buf[offset]     = v0 * cos_t - v1 * sin_t;
    buf[offset + 1] = v0 * sin_t + v1 * cos_t;
}

// ── Per-Head RMSNorm ────────────────────────────────────────────────
//
// Applies RMSNorm independently to each head's vector.
// buf contains n_heads concatenated vectors of size head_dim.
// weight has length head_dim, shared across all heads.
//
// Grid: n_heads threadgroups × 256 threads.

kernel void per_head_rms_norm_f32(
    device float* buf          [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& n_heads     [[buffer(2)]],
    constant uint& head_dim    [[buffer(3)]],
    constant float& eps        [[buffer(4)]],
    uint head                  [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads) return;

    uint base = head * head_dim;

    // Accumulate sum of squares
    float sum_sq = 0.0f;
    for (uint i = lid; i < head_dim; i += NORM_TG_SIZE) {
        float v = buf[base + i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;

    for (uint i = lid; i < head_dim; i += NORM_TG_SIZE) {
        buf[base + i] = buf[base + i] * inv_rms * weight[i];
    }
}

// ── Per-Head RMSNorm Batch ──────────────────────────────────────────
//
// Applies per-head RMSNorm independently across batched rows.
// buf layout: [n_rows, n_heads, head_dim]
// weight layout: [head_dim] (shared for all rows and heads)
//
// Grid: (n_rows * n_heads) threadgroups × 256 threads.

kernel void per_head_rms_norm_batch_f32(
    device float* buf          [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& n_rows      [[buffer(2)]],
    constant uint& n_heads     [[buffer(3)]],
    constant uint& head_dim    [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint row_head              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint total_heads = n_rows * n_heads;
    if (row_head >= total_heads) return;

    uint row = row_head / n_heads;
    uint head = row_head % n_heads;
    uint base = row * (n_heads * head_dim) + head * head_dim;

    float sum_sq = 0.0f;
    for (uint i = lid; i < head_dim; i += NORM_TG_SIZE) {
        float v = buf[base + i];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;
    for (uint i = lid; i < head_dim; i += NORM_TG_SIZE) {
        buf[base + i] = buf[base + i] * inv_rms * weight[i];
    }
}

// ── Gemma3 QK Norm + RoPE Batch (Fused) ────────────────────────────
//
// For each row/head:
//   1) RMSNorm on head vector (using q_weight or k_weight)
//   2) Apply RoPE in-place
//
// Q layout: [n_rows, n_q_heads, head_dim]
// K layout: [n_rows, n_kv_heads, head_dim]
// q_weight/k_weight layout: [head_dim]
//
// Grid: (n_rows * (n_q_heads + n_kv_heads)) threadgroups × 256 threads.

kernel void qk_norm_rope_batch_f32(
    device float* q             [[buffer(0)]],
    device float* k             [[buffer(1)]],
    device const float* q_weight[[buffer(2)]],
    device const float* k_weight[[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    constant uint& n_q_heads    [[buffer(5)]],
    constant uint& n_kv_heads   [[buffer(6)]],
    constant uint& head_dim     [[buffer(7)]],
    constant float& eps         [[buffer(8)]],
    constant float& start_pos   [[buffer(9)]],
    constant float& pos_step    [[buffer(10)]],
    constant float& freq_base   [[buffer(11)]],
    uint row_head               [[threadgroup_position_in_grid]],
    uint lid                    [[thread_index_in_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    uint heads_per_row = n_q_heads + n_kv_heads;
    uint total_heads = n_rows * heads_per_row;
    if (row_head >= total_heads) return;

    uint row = row_head / heads_per_row;
    uint head_local = row_head % heads_per_row;

    device float* buf;
    device const float* weight;
    uint base;
    if (head_local < n_q_heads) {
        uint head = head_local;
        base = row * (n_q_heads * head_dim) + head * head_dim;
        buf = q;
        weight = q_weight;
    } else {
        uint head = head_local - n_q_heads;
        base = row * (n_kv_heads * head_dim) + head * head_dim;
        buf = k;
        weight = k_weight;
    }

    float sum_sq = 0.0f;
    for (uint i = lid; i < head_dim; i += NORM_TG_SIZE) {
        float v = buf[base + i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_inv_rms;
    if (simd_id == 0) {
        sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (simd_lane == 0) {
            shared_inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_inv_rms;
    float position = start_pos + float(row) * pos_step;
    uint half_dim = head_dim / 2;
    for (uint i = lid; i < half_dim; i += NORM_TG_SIZE) {
        uint off = base + 2 * i;
        float v0 = buf[off] * inv_rms * weight[2 * i];
        float v1 = buf[off + 1] * inv_rms * weight[2 * i + 1];

        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);

        buf[off] = v0 * cos_t - v1 * sin_t;
        buf[off + 1] = v0 * sin_t + v1 * cos_t;
    }
}

// ── GELU elementwise mul ────────────────────────────────────────────
//
// gate[i] = GELU(gate[i]) * up[i]
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Grid: ceil(n / 256) threadgroups × 256 threads.

constant float SQRT_2_PI = 0.7978845608f;

kernel void gelu_elementwise_mul_f32(
    device float* gate       [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = gate[gid];
    float x3 = x * x * x;
    float inner = SQRT_2_PI * (x + 0.044715f * x3);
    // Clamp to prevent tanh overflow: Metal's tanh uses exp(2x) which overflows f32
    // for |x| > ~44. tanh(10) is already 1.0 to f32 precision, so this is lossless.
    inner = clamp(inner, -10.0f, 10.0f);
    gate[gid] = 0.5f * x * (1.0f + tanh(inner)) * up[gid];
}

// ── GELU elementwise mul (batch) ────────────────────────────────────
//
// gate[row, i] = GELU(gate[row, i]) * up[row, i]
// Layout: gate/up are contiguous [n_rows * n]

kernel void gelu_elementwise_mul_batch_f32(
    device float* gate       [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    constant uint& n_rows    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint total = n * n_rows;
    if (gid >= total) return;
    float x = gate[gid];
    float x3 = x * x * x;
    float inner = SQRT_2_PI * (x + 0.044715f * x3);
    inner = clamp(inner, -10.0f, 10.0f);
    gate[gid] = 0.5f * x * (1.0f + tanh(inner)) * up[gid];
}

// ── SiLU elementwise mul ────────────────────────────────────────────
//
// gate[i] = SiLU(gate[i]) * up[i]
// SiLU(x) = x / (1 + exp(-x))
//
// Grid: ceil(n / 256) threadgroups × 256 threads.

kernel void silu_elementwise_mul_f32(
    device float* gate       [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = gate[gid];
    gate[gid] = x / (1.0f + exp(-x)) * up[gid];
}

// ── SiLU elementwise mul (batch) ────────────────────────────────────
//
// gate[row, i] = SiLU(gate[row, i]) * up[row, i]
// Layout: gate/up are contiguous [n_rows * n]

kernel void silu_elementwise_mul_batch_f32(
    device float* gate       [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    constant uint& n_rows    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint total = n * n_rows;
    if (gid >= total) return;
    float x = gate[gid];
    float s = x / (1.0f + exp(-x)); // SiLU
    gate[gid] = s * up[gid];
}

// ── SiLU elementwise mul (batch, f16 output) ───────────────────────
//
// out[row, i] = half(SiLU(gate[row, i]) * up[row, i])
// Layout: gate/up/out contiguous [n_rows * n]
kernel void silu_elementwise_mul_batch_f16(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device half* out         [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    constant uint& n_rows    [[buffer(4)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint total = n * n_rows;
    if (gid >= total) return;
    float x = gate[gid];
    float s = x / (1.0f + exp(-x)); // SiLU
    out[gid] = half(s * up[gid]);
}

// ── Elementwise add ─────────────────────────────────────────────────
//
// a[i] += b[i]
//
// Grid: ceil(n / 256) threadgroups × 256 threads.

kernel void elementwise_add_f32(
    device float* a          [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    a[gid] += b[gid];
}

// ── Elementwise Add Batch ────────────────────────────────────────────
//
// a[row, i] += b[row, i]
// Layout: a/b contiguous [n_rows * n]

kernel void elementwise_add_batch_f32(
    device float* a          [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    constant uint& n_rows    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint total = n * n_rows;
    if (gid >= total) return;
    a[gid] += b[gid];
}

// ── Type cast helpers ───────────────────────────────────────────────
//
// Used to stage f16 matmul inputs/outputs while keeping the rest of
// the pipeline in f32 during incremental migration.

kernel void cast_f32_to_f16(
    device const float* src  [[buffer(0)]],
    device half* dst         [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = half(src[gid]);
}

kernel void cast_f16_to_f32(
    device const half* src   [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = float(src[gid]);
}

// ── QKV split (batched) ─────────────────────────────────────────────
//
// Split fused QKV projection output:
//   src: [n_rows, q_dim + 2 * kv_dim]
//   q:   [n_rows, q_dim]
//   k:   [n_rows, kv_dim]
//   v:   [n_rows, kv_dim]
//
// Single pass over src with row-aware range routing.

kernel void qkv_split_batch_f32(
    device const float* src      [[buffer(0)]],
    device float* q              [[buffer(1)]],
    device float* k              [[buffer(2)]],
    device float* v              [[buffer(3)]],
    constant uint& n_rows        [[buffer(4)]],
    constant uint& q_dim         [[buffer(5)]],
    constant uint& kv_dim        [[buffer(6)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint fused_dim = q_dim + 2 * kv_dim;
    uint total = n_rows * fused_dim;
    if (gid >= total) return;

    uint row = gid / fused_dim;
    uint col = gid % fused_dim;
    uint src_idx = row * fused_dim + col;

    if (col < q_dim) {
        q[row * q_dim + col] = src[src_idx];
        return;
    }

    uint k_start = q_dim;
    uint v_start = q_dim + kv_dim;
    if (col < v_start) {
        uint kc = col - k_start;
        k[row * kv_dim + kc] = src[src_idx];
    } else {
        uint vc = col - v_start;
        v[row * kv_dim + vc] = src[src_idx];
    }
}

// ── QKV split + RoPE Batch (fused) ──────────────────────────────────
//
// Input:
//   src: [n_rows, q_dim + 2 * kv_dim] where [Q | K | V]
// Output:
//   q:   [n_rows, q_dim]    (RoPE applied)
//   k:   [n_rows, kv_dim]   (RoPE applied)
//   v:   [n_rows, kv_dim]   (copied)
//
// RoPE is applied per row with:
//   pos = start_pos + row * pos_step
//
// q_dim = n_q_heads * head_dim
// kv_dim = n_kv_heads * head_dim
kernel void qkv_split_rope_batch_f32(
    device const float* src      [[buffer(0)]],
    device float* q              [[buffer(1)]],
    device float* k              [[buffer(2)]],
    device float* v              [[buffer(3)]],
    constant uint& n_rows        [[buffer(4)]],
    constant uint& n_q_heads     [[buffer(5)]],
    constant uint& n_kv_heads    [[buffer(6)]],
    constant uint& head_dim      [[buffer(7)]],
    constant float& start_pos    [[buffer(8)]],
    constant float& pos_step     [[buffer(9)]],
    constant float& freq_base    [[buffer(10)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint q_dim = n_q_heads * head_dim;
    uint kv_dim = n_kv_heads * head_dim;
    uint fused_dim = q_dim + 2 * kv_dim;
    uint q_pairs = n_q_heads * half_dim;
    uint k_pairs = n_kv_heads * half_dim;
    uint items_per_row = q_pairs + k_pairs + kv_dim;
    uint total = n_rows * items_per_row;
    if (gid >= total) return;

    uint row = gid / items_per_row;
    uint local = gid % items_per_row;
    float position = start_pos + float(row) * pos_step;

    if (local < q_pairs) {
        uint head = local / half_dim;
        uint i = local % half_dim;
        uint src_base = row * fused_dim + head * head_dim + 2 * i;
        uint dst_base = row * q_dim + head * head_dim + 2 * i;
        float v0 = src[src_base];
        float v1 = src[src_base + 1];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        q[dst_base] = v0 * cos_t - v1 * sin_t;
        q[dst_base + 1] = v0 * sin_t + v1 * cos_t;
        return;
    }

    uint k_start = q_pairs;
    uint v_start = q_pairs + k_pairs;
    if (local < v_start) {
        uint k_local = local - k_start;
        uint head = k_local / half_dim;
        uint i = k_local % half_dim;
        uint src_base = row * fused_dim + q_dim + head * head_dim + 2 * i;
        uint dst_base = row * kv_dim + head * head_dim + 2 * i;
        float v0 = src[src_base];
        float v1 = src[src_base + 1];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        k[dst_base] = v0 * cos_t - v1 * sin_t;
        k[dst_base + 1] = v0 * sin_t + v1 * cos_t;
    } else {
        uint vc = local - v_start;
        uint src_idx = row * fused_dim + q_dim + kv_dim + vc;
        v[row * kv_dim + vc] = src[src_idx];
    }
}

// ── QKV split + RoPE + KV append Batch (f32 KV cache) ───────────────
//
// Splits fused QKV, applies RoPE to Q/K, writes:
// - Q/K/V scratch buffers
// - K/V into KV cache with row stride and offset
kernel void qkv_split_rope_append_kv_batch_f32(
    device const float* src      [[buffer(0)]],
    device float* q              [[buffer(1)]],
    device float* k              [[buffer(2)]],
    device float* v              [[buffer(3)]],
    device float* cache_k        [[buffer(4)]],
    device float* cache_v        [[buffer(5)]],
    constant uint& n_rows        [[buffer(6)]],
    constant uint& n_q_heads     [[buffer(7)]],
    constant uint& n_kv_heads    [[buffer(8)]],
    constant uint& head_dim      [[buffer(9)]],
    constant float& start_pos    [[buffer(10)]],
    constant float& pos_step     [[buffer(11)]],
    constant float& freq_base    [[buffer(12)]],
    constant uint& cache_offset  [[buffer(13)]],
    constant uint& cache_stride  [[buffer(14)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint q_dim = n_q_heads * head_dim;
    uint kv_dim = n_kv_heads * head_dim;
    uint fused_dim = q_dim + 2 * kv_dim;
    uint q_pairs = n_q_heads * half_dim;
    uint k_pairs = n_kv_heads * half_dim;
    uint items_per_row = q_pairs + k_pairs + kv_dim;
    uint total = n_rows * items_per_row;
    if (gid >= total) return;

    uint row = gid / items_per_row;
    uint local = gid % items_per_row;
    float position = start_pos + float(row) * pos_step;
    uint cache_row_base = cache_offset + row * cache_stride;
    uint src_row_base = row * fused_dim;
    uint q_row_base = row * q_dim;
    uint kv_row_base = row * kv_dim;

    if (local < q_pairs) {
        uint head = local / half_dim;
        uint i = local % half_dim;
        uint src_base = src_row_base + head * head_dim + 2 * i;
        uint dst_base = q_row_base + head * head_dim + 2 * i;
        float v0 = src[src_base];
        float v1 = src[src_base + 1];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        q[dst_base] = v0 * cos_t - v1 * sin_t;
        q[dst_base + 1] = v0 * sin_t + v1 * cos_t;
        return;
    }

    uint k_start = q_pairs;
    uint v_start = q_pairs + k_pairs;
    if (local < v_start) {
        uint k_local = local - k_start;
        uint head = k_local / half_dim;
        uint i = k_local % half_dim;
        uint src_base = src_row_base + q_dim + head * head_dim + 2 * i;
        uint dst_base = kv_row_base + head * head_dim + 2 * i;
        float v0 = src[src_base];
        float v1 = src[src_base + 1];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        float rk0 = v0 * cos_t - v1 * sin_t;
        float rk1 = v0 * sin_t + v1 * cos_t;
        k[dst_base] = rk0;
        k[dst_base + 1] = rk1;
        cache_k[cache_row_base + dst_base] = rk0;
        cache_k[cache_row_base + dst_base + 1] = rk1;
    } else {
        uint vc = local - v_start;
        uint src_idx = src_row_base + q_dim + kv_dim + vc;
        float vv = src[src_idx];
        v[kv_row_base + vc] = vv;
        cache_v[cache_row_base + vc] = vv;
    }
}

// ── QKV split + RoPE + KV append Batch (f16 KV cache) ───────────────
kernel void qkv_split_rope_append_kv_batch_f16(
    device const float* src      [[buffer(0)]],
    device float* q              [[buffer(1)]],
    device float* k              [[buffer(2)]],
    device float* v              [[buffer(3)]],
    device half* cache_k         [[buffer(4)]],
    device half* cache_v         [[buffer(5)]],
    constant uint& n_rows        [[buffer(6)]],
    constant uint& n_q_heads     [[buffer(7)]],
    constant uint& n_kv_heads    [[buffer(8)]],
    constant uint& head_dim      [[buffer(9)]],
    constant float& start_pos    [[buffer(10)]],
    constant float& pos_step     [[buffer(11)]],
    constant float& freq_base    [[buffer(12)]],
    constant uint& cache_offset  [[buffer(13)]],
    constant uint& cache_stride  [[buffer(14)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint q_dim = n_q_heads * head_dim;
    uint kv_dim = n_kv_heads * head_dim;
    uint fused_dim = q_dim + 2 * kv_dim;
    uint q_pairs = n_q_heads * half_dim;
    uint k_pairs = n_kv_heads * half_dim;
    uint items_per_row = q_pairs + k_pairs + kv_dim;
    uint total = n_rows * items_per_row;
    if (gid >= total) return;

    uint row = gid / items_per_row;
    uint local = gid % items_per_row;
    float position = start_pos + float(row) * pos_step;
    uint cache_row_base = cache_offset + row * cache_stride;
    uint src_row_base = row * fused_dim;
    uint q_row_base = row * q_dim;
    uint kv_row_base = row * kv_dim;

    if (local < q_pairs) {
        uint head = local / half_dim;
        uint i = local % half_dim;
        uint src_base = src_row_base + head * head_dim + 2 * i;
        uint dst_base = q_row_base + head * head_dim + 2 * i;
        float v0 = src[src_base];
        float v1 = src[src_base + 1];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        q[dst_base] = v0 * cos_t - v1 * sin_t;
        q[dst_base + 1] = v0 * sin_t + v1 * cos_t;
        return;
    }

    uint k_start = q_pairs;
    uint v_start = q_pairs + k_pairs;
    if (local < v_start) {
        uint k_local = local - k_start;
        uint head = k_local / half_dim;
        uint i = k_local % half_dim;
        uint src_base = src_row_base + q_dim + head * head_dim + 2 * i;
        uint dst_base = kv_row_base + head * head_dim + 2 * i;
        float v0 = src[src_base];
        float v1 = src[src_base + 1];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        float rk0 = v0 * cos_t - v1 * sin_t;
        float rk1 = v0 * sin_t + v1 * cos_t;
        k[dst_base] = rk0;
        k[dst_base + 1] = rk1;
        cache_k[cache_row_base + dst_base] = half(rk0);
        cache_k[cache_row_base + dst_base + 1] = half(rk1);
    } else {
        uint vc = local - v_start;
        uint src_idx = src_row_base + q_dim + kv_dim + vc;
        float vv = src[src_idx];
        v[kv_row_base + vc] = vv;
        cache_v[cache_row_base + vc] = half(vv);
    }
}

// ── KV cache append ───────────────────────────────────────────────
//
// Copy `count` floats from src into dst at byte offset `offset`.
// Used to append a single token's K or V vector to the GPU KV cache
// inside a command buffer (no CPU sync needed).
//
// Grid: ceil(count / 256) threadgroups × 256 threads.

kernel void kv_append_f32(
    device const float* src  [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& offset    [[buffer(2)]],   // offset in floats (seq_len * kv_stride)
    constant uint& count     [[buffer(3)]],   // number of floats (kv_stride)
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    dst[offset + gid] = src[gid];
}

// ── KV cache append (f16 destination) ───────────────────────────────
//
// Copy `count` floats from src into dst (f16) at offset.
// Used when GPU KV cache is stored in half precision.

kernel void kv_append_f16(
    device const float* src  [[buffer(0)]],
    device half* dst         [[buffer(1)]],
    constant uint& offset    [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    dst[offset + gid] = half(src[gid]);
}

// ── KV cache append (batched) ────────────────────────────────────────
//
// Copy src rows into dst with a destination row stride:
//   src: [n_rows, count]
//   dst: destination tensor where each row starts at
//        dst_offset + row * dst_row_stride
//
// Grid: ceil((n_rows * count) / 256) threadgroups × 256 threads.

kernel void kv_append_batch_f32(
    device const float* src      [[buffer(0)]],
    device float* dst            [[buffer(1)]],
    constant uint& dst_offset    [[buffer(2)]],   // starting offset in floats
    constant uint& dst_row_stride[[buffer(3)]],   // row stride in floats
    constant uint& count         [[buffer(4)]],   // floats per row
    constant uint& n_rows        [[buffer(5)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint total = n_rows * count;
    if (gid >= total) return;

    uint row = gid / count;
    uint col = gid % count;
    dst[dst_offset + row * dst_row_stride + col] = src[row * count + col];
}

// ── KV cache append batch (f16 destination) ──────────────────────────
//
// Copy src rows into dst (f16) with row stride.

kernel void kv_append_batch_f16(
    device const float* src      [[buffer(0)]],
    device half* dst             [[buffer(1)]],
    constant uint& dst_offset    [[buffer(2)]],
    constant uint& dst_row_stride[[buffer(3)]],
    constant uint& count         [[buffer(4)]],
    constant uint& n_rows        [[buffer(5)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint total = n_rows * count;
    if (gid >= total) return;

    uint row = gid / count;
    uint col = gid % count;
    dst[dst_offset + row * dst_row_stride + col] = half(src[row * count + col]);
}

// ── KV cache append batch pair (f32 destination) ─────────────────────
//
// Copy K and V src rows into K/V destination caches with identical layout.
kernel void kv_append_batch2_f32(
    device const float* src_k    [[buffer(0)]],
    device const float* src_v    [[buffer(1)]],
    device float* dst_k          [[buffer(2)]],
    device float* dst_v          [[buffer(3)]],
    constant uint& dst_offset    [[buffer(4)]],
    constant uint& dst_row_stride[[buffer(5)]],
    constant uint& count         [[buffer(6)]],
    constant uint& n_rows        [[buffer(7)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint total = n_rows * count;
    if (gid >= total) return;
    uint row = gid / count;
    uint col = gid % count;
    uint dst_idx = dst_offset + row * dst_row_stride + col;
    uint src_idx = row * count + col;
    dst_k[dst_idx] = src_k[src_idx];
    dst_v[dst_idx] = src_v[src_idx];
}

// ── KV cache append batch pair (f16 destination) ─────────────────────
kernel void kv_append_batch2_f16(
    device const float* src_k    [[buffer(0)]],
    device const float* src_v    [[buffer(1)]],
    device half* dst_k           [[buffer(2)]],
    device half* dst_v           [[buffer(3)]],
    constant uint& dst_offset    [[buffer(4)]],
    constant uint& dst_row_stride[[buffer(5)]],
    constant uint& count         [[buffer(6)]],
    constant uint& n_rows        [[buffer(7)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint total = n_rows * count;
    if (gid >= total) return;
    uint row = gid / count;
    uint col = gid % count;
    uint dst_idx = dst_offset + row * dst_row_stride + col;
    uint src_idx = row * count + col;
    dst_k[dst_idx] = half(src_k[src_idx]);
    dst_v[dst_idx] = half(src_v[src_idx]);
}
