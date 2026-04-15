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

// ── RMSNorm Batch (out-of-place, float4 vectorized) ──────────────────
//
// Same as rms_norm_out_batch_f32 but processes 4 elements per iteration
// via float4 dot product. Requires n % 4 == 0 (always true for LLM dims).
// Reference: llama.cpp kernel_rms_norm_fuse_impl<float4, F>.
kernel void rms_norm_out_batch_f32_vec4(
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
    uint n4 = n / 4;

    device const float4* x4 = (device const float4*)(x + base);
    device const float4* w4 = (device const float4*)weight;
    device float4* o4       = (device float4*)(out + base);

    float sum_sq = 0.0f;
    for (uint i = lid; i < n4; i += NORM_TG_SIZE) {
        sum_sq += dot(x4[i], x4[i]);
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

    for (uint i = lid; i < n4; i += NORM_TG_SIZE) {
        o4[i] = x4[i] * inv_rms * w4[i];
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

// ── Post-Attention RMSNorm + Residual Add + RMSNorm Batch ───────────────────
//
// Gemma3-specific fused handoff:
//   1. attn_norm[i] = addend[i] * post_weight[i] / sqrt(mean(addend[:]²) + eps)
//   2. hidden[i] = hidden[i] + attn_norm[i]
//   3. norm_out[i] = hidden[i] * residual_weight[i] / sqrt(mean(hidden[:]²) + eps)
//
// This replaces a standalone post-attention RMSNorm dispatch plus the existing
// residual_add_rms_norm_out_batch dispatch in the Gemma3 prefill path.

kernel void post_attn_norm_residual_add_rms_norm_out_batch_f32(
    device float* hidden                 [[buffer(0)]],
    device const float* addend           [[buffer(1)]],
    device const float* post_weight      [[buffer(2)]],
    device const float* residual_weight  [[buffer(3)]],
    device float* norm_out               [[buffer(4)]],
    constant uint& n                     [[buffer(5)]],
    constant uint& n_rows                [[buffer(6)]],
    constant float& eps                  [[buffer(7)]],
    uint row                             [[threadgroup_position_in_grid]],
    uint lid                             [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    if (row >= n_rows) return;

    uint base = row * n;

    float addend_sum_sq = 0.0f;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = addend[base + i];
        addend_sum_sq += v * v;
    }

    addend_sum_sq = simd_sum(addend_sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = addend_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float addend_inv_rms;
    if (simd_id == 0) {
        addend_sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        addend_sum_sq = simd_sum(addend_sum_sq);
        if (simd_lane == 0) {
            addend_inv_rms = rsqrt(addend_sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float hidden_sum_sq = 0.0f;
    float inv_rms_addend = addend_inv_rms;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float attn_norm = addend[base + i] * inv_rms_addend * post_weight[i];
        float v = hidden[base + i] + attn_norm;
        hidden_sum_sq += v * v;
    }

    hidden_sum_sq = simd_sum(hidden_sum_sq);

    if (simd_lane == 0) {
        simd_sums[simd_id] = hidden_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float hidden_inv_rms;
    if (simd_id == 0) {
        hidden_sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        hidden_sum_sq = simd_sum(hidden_sum_sq);
        if (simd_lane == 0) {
            hidden_inv_rms = rsqrt(hidden_sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms_hidden = hidden_inv_rms;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float attn_norm = addend[base + i] * inv_rms_addend * post_weight[i];
        float v = hidden[base + i] + attn_norm;
        hidden[base + i] = v;
        norm_out[base + i] = v * inv_rms_hidden * residual_weight[i];
    }
}

// ── Post-FFN RMSNorm + Residual Add + RMSNorm Batch ────────────────────────
//
// Gemma3-specific fused FFN handoff:
//   1. ffn_norm[i] = addend[i] * post_weight[i] / sqrt(mean(addend[:]²) + eps)
//   2. hidden[i] = hidden[i] + ffn_norm[i]
//   3. norm_out[i] = hidden[i] * residual_weight[i] / sqrt(mean(hidden[:]²) + eps)
//
// This replaces a standalone post-FFN RMSNorm dispatch plus the existing
// residual_add_rms_norm_out_batch dispatch on non-final Gemma3 layers.

kernel void post_ffn_norm_residual_add_rms_norm_out_batch_f32(
    device float* hidden                 [[buffer(0)]],
    device const float* addend           [[buffer(1)]],
    device const float* post_weight      [[buffer(2)]],
    device const float* residual_weight  [[buffer(3)]],
    device float* norm_out               [[buffer(4)]],
    constant uint& n                     [[buffer(5)]],
    constant uint& n_rows                [[buffer(6)]],
    constant float& eps                  [[buffer(7)]],
    uint row                             [[threadgroup_position_in_grid]],
    uint lid                             [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]]
) {
    if (row >= n_rows) return;

    uint base = row * n;

    float addend_sum_sq = 0.0f;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float v = addend[base + i];
        addend_sum_sq += v * v;
    }

    addend_sum_sq = simd_sum(addend_sum_sq);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = addend_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float addend_inv_rms;
    if (simd_id == 0) {
        addend_sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        addend_sum_sq = simd_sum(addend_sum_sq);
        if (simd_lane == 0) {
            addend_inv_rms = rsqrt(addend_sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float hidden_sum_sq = 0.0f;
    float inv_rms_addend = addend_inv_rms;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float ffn_norm = addend[base + i] * inv_rms_addend * post_weight[i];
        float v = hidden[base + i] + ffn_norm;
        hidden_sum_sq += v * v;
    }

    hidden_sum_sq = simd_sum(hidden_sum_sq);

    if (simd_lane == 0) {
        simd_sums[simd_id] = hidden_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float hidden_inv_rms;
    if (simd_id == 0) {
        hidden_sum_sq = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        hidden_sum_sq = simd_sum(hidden_sum_sq);
        if (simd_lane == 0) {
            hidden_inv_rms = rsqrt(hidden_sum_sq / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms_hidden = hidden_inv_rms;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        float ffn_norm = addend[base + i] * inv_rms_addend * post_weight[i];
        float v = hidden[base + i] + ffn_norm;
        hidden[base + i] = v;
        norm_out[base + i] = v * inv_rms_hidden * residual_weight[i];
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
// total_pairs = rows * (n_q_heads + n_kv_heads) * half_dim
//
// Single-row and batched RoPE share the same indexing math with only one
// structural difference: whether gid spans one row or a `[row, pair]` domain.
// Use a function constant so Metal compiles two specialized pipelines from one
// source kernel without keeping two almost-identical shader bodies.

constant bool ROPE_BATCHED [[function_constant(0)]];

// ── G11: YaRN RoPE helpers (matching llama.cpp) ────────────────────────
//
// When ext_factor == 0, rope_yarn reduces to vanilla cos/sin with
// mscale == attn_factor. For standard RoPE, pass ext_factor=0,
// attn_factor=1 and there is zero overhead.

inline float rope_yarn_ramp(float low, float high, int i0) {
    float y = (float(i0) / 2.0f - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

inline void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dim_low,
    float corr_dim_high, int i0, float ext_factor, float mscale,
    thread float& cos_theta, thread float& sin_theta
) {
    float theta = freq_scale * theta_extrap;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dim_low, corr_dim_high, i0) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    cos_theta = cos(theta) * mscale;
    sin_theta = sin(theta) * mscale;
}

inline float rope_yarn_corr_factor(uint n_dims, uint n_ctx_orig, float freq_base, float beta) {
    return float(n_dims) * log(float(n_ctx_orig) / (beta * 2.0f * M_PI_F)) / (2.0f * log(freq_base));
}

kernel void rope_f32_generic(
    device float* q            [[buffer(0)]],
    device float* k            [[buffer(1)]],
    constant uint& n_rows      [[buffer(2)]],
    constant uint& n_q_heads   [[buffer(3)]],
    constant uint& n_kv_heads  [[buffer(4)]],
    constant uint& head_dim    [[buffer(5)]],
    constant float& start_pos  [[buffer(6)]],
    constant float& pos_step   [[buffer(7)]],
    constant float& freq_base  [[buffer(8)]],
    // G11: YaRN parameters (ext_factor==0 → vanilla RoPE, backward compatible)
    constant float& freq_scale   [[buffer(9)]],
    constant float& ext_factor   [[buffer(10)]],
    constant float& attn_factor  [[buffer(11)]],
    constant float& beta_fast    [[buffer(12)]],
    constant float& beta_slow    [[buffer(13)]],
    constant uint& n_ctx_orig    [[buffer(14)]],
    constant uint& rope_dim      [[buffer(15)]],
    uint gid                   [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint rows = ROPE_BATCHED ? n_rows : 1;
    uint pairs_per_row = (n_q_heads + n_kv_heads) * half_dim;
    uint total_pairs = rows * pairs_per_row;
    if (gid >= total_pairs) return;

    uint row = ROPE_BATCHED ? (gid / pairs_per_row) : 0;
    uint local = ROPE_BATCHED ? (gid % pairs_per_row) : gid;

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

    uint rope_dim_eff = rope_dim == 0 ? head_dim : min(rope_dim, head_dim);
    uint rope_pairs = rope_dim_eff / 2;
    if (i >= rope_pairs) {
        return;
    }

    uint offset = vec_base + 2 * i;
    float position = start_pos + float(row) * pos_step;
    float theta_extrap = position / pow(freq_base, 2.0f * float(i) / float(rope_dim_eff));

    float cos_t, sin_t;
    if (ext_factor == 0.0f) {
        // Fast path: vanilla RoPE or linear scaling (no per-dim blend)
        float theta = freq_scale * theta_extrap;
        cos_t = cos(theta) * attn_factor;
        sin_t = sin(theta) * attn_factor;
    } else {
        // YaRN: per-dimension interpolation/extrapolation blend
        float corr_low  = max(0.0f, floor(rope_yarn_corr_factor(rope_dim_eff, n_ctx_orig, freq_base, beta_fast)));
        float corr_high = min(float(rope_dim_eff) - 1.0f, ceil(rope_yarn_corr_factor(rope_dim_eff, n_ctx_orig, freq_base, beta_slow)));
        rope_yarn(theta_extrap, freq_scale, corr_low, corr_high, int(2 * i),
                  ext_factor, attn_factor, cos_t, sin_t);
    }

    float v0 = buf[offset];
    float v1 = buf[offset + 1];
    buf[offset]     = v0 * cos_t - v1 * sin_t;
    buf[offset + 1] = v0 * sin_t + v1 * cos_t;
}

// NeoX partial RoPE rotates split-half pairs within the rotary prefix:
//   (x[i], x[i + rope_dim/2]) for i in [0, rope_dim/2).
// Qwen3.5 full-attention layers use this layout.
kernel void rope_neox_partial_batch_f32(
    device float* q            [[buffer(0)]],
    device float* k            [[buffer(1)]],
    constant uint& n_rows      [[buffer(2)]],
    constant uint& n_q_heads   [[buffer(3)]],
    constant uint& n_kv_heads  [[buffer(4)]],
    constant uint& head_dim    [[buffer(5)]],
    constant float& start_pos  [[buffer(6)]],
    constant float& pos_step   [[buffer(7)]],
    constant float& freq_base  [[buffer(8)]],
    constant float& freq_scale [[buffer(9)]],
    constant float& ext_factor [[buffer(10)]],
    constant float& attn_factor[[buffer(11)]],
    constant float& beta_fast  [[buffer(12)]],
    constant float& beta_slow  [[buffer(13)]],
    constant uint& n_ctx_orig  [[buffer(14)]],
    constant uint& rope_dim    [[buffer(15)]],
    uint gid                   [[thread_position_in_grid]]
) {
    uint rope_dim_eff = rope_dim == 0 ? head_dim : min(rope_dim, head_dim);
    uint rope_pairs = rope_dim_eff / 2;
    if (rope_pairs == 0) return;

    uint pairs_per_row = (n_q_heads + n_kv_heads) * rope_pairs;
    uint total_pairs = n_rows * pairs_per_row;
    if (gid >= total_pairs) return;

    uint row = gid / pairs_per_row;
    uint local = gid % pairs_per_row;

    uint q_pairs = n_q_heads * rope_pairs;
    device float* buf;
    uint vec_base;
    uint i;
    if (local < q_pairs) {
        uint head = local / rope_pairs;
        i = local % rope_pairs;
        buf = q;
        vec_base = row * (n_q_heads * head_dim) + head * head_dim;
    } else {
        uint k_local = local - q_pairs;
        uint head = k_local / rope_pairs;
        i = k_local % rope_pairs;
        buf = k;
        vec_base = row * (n_kv_heads * head_dim) + head * head_dim;
    }

    float position = start_pos + float(row) * pos_step;
    float theta_extrap = position / pow(freq_base, 2.0f * float(i) / float(rope_dim_eff));

    float cos_t, sin_t;
    if (ext_factor == 0.0f) {
        float theta = freq_scale * theta_extrap;
        cos_t = cos(theta) * attn_factor;
        sin_t = sin(theta) * attn_factor;
    } else {
        float corr_low  = max(0.0f, floor(rope_yarn_corr_factor(rope_dim_eff, n_ctx_orig, freq_base, beta_fast)));
        float corr_high = min(float(rope_dim_eff) - 1.0f, ceil(rope_yarn_corr_factor(rope_dim_eff, n_ctx_orig, freq_base, beta_slow)));
        rope_yarn(theta_extrap, freq_scale, corr_low, corr_high, int(2 * i),
                  ext_factor, attn_factor, cos_t, sin_t);
    }

    uint offset0 = vec_base + i;
    uint offset1 = vec_base + rope_pairs + i;
    float v0 = buf[offset0];
    float v1 = buf[offset1];
    buf[offset0] = v0 * cos_t - v1 * sin_t;
    buf[offset1] = v0 * sin_t + v1 * cos_t;
}

// Gemma4 global layers use NeoX split-half RoPE with explicit per-pair
// frequency multipliers from rope_freqs.weight.
kernel void rope_neox_partial_freq_factors_batch_f32(
    device float* q                    [[buffer(0)]],
    device float* k                    [[buffer(1)]],
    device const float* freq_factors   [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& n_q_heads           [[buffer(4)]],
    constant uint& n_kv_heads          [[buffer(5)]],
    constant uint& head_dim            [[buffer(6)]],
    constant float& start_pos          [[buffer(7)]],
    constant float& pos_step           [[buffer(8)]],
    constant float& freq_base          [[buffer(9)]],
    constant uint& rope_dim            [[buffer(10)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint rope_dim_eff = rope_dim == 0 ? head_dim : min(rope_dim, head_dim);
    uint rope_pairs = rope_dim_eff / 2;
    if (rope_pairs == 0) return;

    uint pairs_per_row = (n_q_heads + n_kv_heads) * rope_pairs;
    uint total_pairs = n_rows * pairs_per_row;
    if (gid >= total_pairs) return;

    uint row = gid / pairs_per_row;
    uint local = gid % pairs_per_row;

    uint q_pairs = n_q_heads * rope_pairs;
    device float* buf;
    uint vec_base;
    uint i;
    if (local < q_pairs) {
        uint head = local / rope_pairs;
        i = local % rope_pairs;
        buf = q;
        vec_base = row * (n_q_heads * head_dim) + head * head_dim;
    } else {
        uint k_local = local - q_pairs;
        uint head = k_local / rope_pairs;
        i = k_local % rope_pairs;
        buf = k;
        vec_base = row * (n_kv_heads * head_dim) + head * head_dim;
    }

    float position = start_pos + float(row) * pos_step;
    float base_freq = exp(-(log(freq_base) * 2.0f * float(i) / float(rope_dim_eff)));
    float theta = position * base_freq * freq_factors[i];
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    uint offset0 = vec_base + i;
    uint offset1 = vec_base + rope_pairs + i;
    float v0 = buf[offset0];
    float v1 = buf[offset1];
    buf[offset0] = v0 * cos_t - v1 * sin_t;
    buf[offset1] = v0 * sin_t + v1 * cos_t;
}

// ── Per-Head RMSNorm ────────────────────────────────────────────────
//
// Single-row and batched per-head RMSNorm differ only in whether the grid is
// indexed by `head` or `[row, head]`. Use a function constant so the compiler
// specializes away the unused row math in the single-row path.

constant bool PER_HEAD_RMS_BATCHED [[function_constant(1)]];

kernel void per_head_rms_norm_f32_generic(
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
    uint rows = PER_HEAD_RMS_BATCHED ? n_rows : 1;
    uint total_heads = rows * n_heads;
    if (row_head >= total_heads) return;

    uint row = PER_HEAD_RMS_BATCHED ? (row_head / n_heads) : 0;
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

// ── Per-head RMSNorm + SiLU(gate) * src (batch) ────────────────────
//
// gate[row, head, i] = SiLU(gate[row, head, i]) *
//                      src[row, head, i] * inv_rms(row, head) * weight[i]
//
// Used by the Qwen3.5 recurrent tail to fuse the post-SSM head-wise norm
// with the final SiLU gating pass into a single dispatch.

kernel void per_head_rms_norm_silu_mul_batch_f32(
    device float* gate         [[buffer(0)]],
    device const float* src    [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant uint& n_rows      [[buffer(3)]],
    constant uint& n_heads     [[buffer(4)]],
    constant uint& head_dim    [[buffer(5)]],
    constant float& eps        [[buffer(6)]],
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
        float v = src[base + i];
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
        uint idx = base + i;
        float x = gate[idx];
        float s = x / (1.0f + exp(-x));
        gate[idx] = s * (src[idx] * inv_rms * weight[i]);
    }
}

kernel void per_head_rms_norm_no_weight_f32_generic(
    device float* buf          [[buffer(0)]],
    constant uint& n_rows      [[buffer(1)]],
    constant uint& n_heads     [[buffer(2)]],
    constant uint& head_dim    [[buffer(3)]],
    constant float& eps        [[buffer(4)]],
    uint row_head              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint rows = PER_HEAD_RMS_BATCHED ? n_rows : 1;
    uint total_heads = rows * n_heads;
    if (row_head >= total_heads) return;

    uint row = PER_HEAD_RMS_BATCHED ? (row_head / n_heads) : 0;
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
        buf[base + i] = buf[base + i] * inv_rms;
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

// ── GELU elementwise mul (batch, vec4) ─────────────────────────────
//
// Vectorized: processes 4 elements per thread.
// total = n * n_rows must be divisible by 4 (always true for LLM dims).
// Grid: ceil(total/4 / TG) threadgroups.

kernel void gelu_elementwise_mul_batch_f32_vec4(
    device float* gate       [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    constant uint& n_rows    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint total4 = (n * n_rows) / 4;
    if (gid >= total4) return;
    device float4* g4 = (device float4*)gate;
    device const float4* u4 = (device const float4*)up;
    float4 x = g4[gid];
    float4 x3 = x * x * x;
    float4 inner = SQRT_2_PI * (x + 0.044715f * x3);
    inner = clamp(inner, -10.0f, 10.0f);
    g4[gid] = 0.5f * x * (1.0f + tanh(inner)) * u4[gid];
}

// ── GELU split mul (batch) ─────────────────────────────────────────
//
// src[row, i] is laid out as [gate | up] with width 2 * n.
// dst[row, i] = GELU(src[row, i]) * src[row, n + i]

kernel void gelu_split_mul_batch_f32(
    device const float* src  [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    constant uint& n_rows    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint total = n * n_rows;
    if (gid >= total) return;

    uint row = gid / n;
    uint col = gid % n;
    uint base = row * (2 * n);
    float x = src[base + col];
    float up = src[base + n + col];
    float x3 = x * x * x;
    float inner = SQRT_2_PI * (x + 0.044715f * x3);
    inner = clamp(inner, -10.0f, 10.0f);
    dst[gid] = 0.5f * x * (1.0f + tanh(inner)) * up;
}

// ── GELU inplace ────────────────────────────────────────────────────
//
// x[i] = GELU(x[i])
// Used by Falcon FFN (no gate projection).

kernel void gelu_inplace_f32(
    device float* x          [[buffer(0)]],
    constant uint& n         [[buffer(1)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float v = x[gid];
    float v3 = v * v * v;
    float inner = SQRT_2_PI * (v + 0.044715f * v3);
    inner = clamp(inner, -10.0f, 10.0f);
    x[gid] = 0.5f * v * (1.0f + tanh(inner));
}

kernel void gelu_inplace_batch_f32(
    device float* x          [[buffer(0)]],
    constant uint& n         [[buffer(1)]],
    constant uint& n_rows    [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint total = n * n_rows;
    if (gid >= total) return;
    float v = x[gid];
    float v3 = v * v * v;
    float inner = SQRT_2_PI * (v + 0.044715f * v3);
    inner = clamp(inner, -10.0f, 10.0f);
    x[gid] = 0.5f * v * (1.0f + tanh(inner));
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

// ── Sigmoid in-place ────────────────────────────────────────────────
//
// x[i] = sigmoid(x[i]) = 1 / (1 + exp(-x[i]))
//
// Used by Qwen3.5 recurrent layers: beta preparation.

kernel void sigmoid_inplace_f32(
    device float* x   [[buffer(0)]],
    constant uint& n  [[buffer(1)]],
    uint gid          [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    x[gid] = 1.0f / (1.0f + exp(-x[gid]));
}

kernel void sigmoid_inplace_f32_vec4(
    device float* x   [[buffer(0)]],
    constant uint& n  [[buffer(1)]],
    uint gid          [[thread_position_in_grid]]
) {
    uint n4 = n / 4;
    if (gid >= n4) return;
    device float4* x4 = (device float4*)x;
    float4 v = x4[gid];
    x4[gid] = 1.0f / (1.0f + exp(-v));
}

// ── Softplus + bias + mul ───────────────────────────────────────────
//
// alpha[i] = ln(1 + exp(alpha[i] + bias[i])) * a[i]
//
// Used by Qwen3.5 recurrent layers: alpha preparation (softplus of
// biased alpha, scaled by the A matrix).

kernel void softplus_bias_mul_f32(
    device float* alpha       [[buffer(0)]],
    device const float* bias  [[buffer(1)]],
    device const float* a     [[buffer(2)]],
    constant uint& n          [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float x = alpha[gid] + bias[gid];
    // Numerically stable softplus: log(1 + exp(x))
    // For large x, softplus ≈ x; for small x, use log(1+exp(x)).
    float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    alpha[gid] = sp * a[gid];
}

// ── Fused softplus(alpha + bias) * a + sigmoid(beta) ────────────────
//
// alpha[i] = ln(1 + exp(alpha[i] + bias[i])) * a[i]
// beta[i]  = sigmoid(beta[i])
//
// Used by Qwen3.5 recurrent decode to avoid a second elementwise dispatch
// for beta preparation.

kernel void softplus_bias_mul_sigmoid_pair_f32(
    device float* alpha       [[buffer(0)]],
    device float* beta        [[buffer(1)]],
    device const float* bias  [[buffer(2)]],
    device const float* a     [[buffer(3)]],
    constant uint& n          [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float x = alpha[gid] + bias[gid];
    float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    alpha[gid] = sp * a[gid];
    beta[gid] = 1.0f / (1.0f + exp(-beta[gid]));
}

// ── Batched softplus + bias + mul ───────────────────────────────────
//
// alpha[t, h] = ln(1 + exp(alpha[t, h] + bias[h])) * a[h]
//
// Used by Qwen3.5 recurrent batch handoff where alpha is laid out as
// token-major scalars with a per-head bias/A vector.

kernel void softplus_bias_mul_batch_f32(
    device float* alpha       [[buffer(0)]],
    device const float* bias  [[buffer(1)]],
    device const float* a     [[buffer(2)]],
    constant uint& n          [[buffer(3)]],
    constant uint& head_dim   [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uint head = gid % head_dim;
    float x = alpha[gid] + bias[head];
    float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    alpha[gid] = sp * a[head];
}

// ── Batched fused softplus(alpha + bias) * a + sigmoid(beta) ────────
//
// alpha[t, h] = ln(1 + exp(alpha[t, h] + bias[h])) * a[h]
// beta[t, h]  = sigmoid(beta[t, h])

kernel void softplus_bias_mul_sigmoid_pair_batch_f32(
    device float* alpha       [[buffer(0)]],
    device float* beta        [[buffer(1)]],
    device const float* bias  [[buffer(2)]],
    device const float* a     [[buffer(3)]],
    constant uint& n          [[buffer(4)]],
    constant uint& head_dim   [[buffer(5)]],
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uint head = gid % head_dim;
    float x = alpha[gid] + bias[head];
    float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    alpha[gid] = sp * a[head];
    beta[gid] = 1.0f / (1.0f + exp(-beta[gid]));
}

// ── L2 norm per head ────────────────────────────────────────────────
//
// For each head h of n_heads:
//   sum_sq = sum(x[h*dim..h*dim+dim]^2)
//   scale = 1 / sqrt(sum_sq + eps)
//   x[h*dim+i] *= scale
//
// Used by Qwen3.5 recurrent layers: L2 normalization of Q/K after conv.
// Similar to per_head_rms_norm but uses sum_sq (not mean_sq) for the scale.

kernel void l2_norm_per_head_f32(
    device float* x          [[buffer(0)]],
    constant uint& n_heads   [[buffer(1)]],
    constant uint& head_dim  [[buffer(2)]],
    constant float& eps      [[buffer(3)]],
    uint head                [[threadgroup_position_in_grid]],
    uint lid                 [[thread_index_in_threadgroup]],
    uint simd_lane           [[thread_index_in_simdgroup]],
    uint simd_id             [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads) return;
    device float* head_ptr = x + head * head_dim;

    // Strided sum of squares.
    float sum_sq = 0.0f;
    for (uint i = lid; i < head_dim; i += 256) {
        float v = head_ptr[i];
        sum_sq = fma(v, v, sum_sq);
    }

    // Simdgroup reduction.
    sum_sq = simd_sum(sum_sq);

    // Cross-simdgroup reduction.
    threadgroup float simd_sums[8]; // up to 8 simdgroups in TG=256
    if (simd_lane == 0) simd_sums[simd_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float total = (simd_lane < 8) ? simd_sums[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) simd_sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(simd_sums[0] + eps);

    for (uint i = lid; i < head_dim; i += 256) {
        head_ptr[i] *= scale;
    }
}

// ── Sigmoid elementwise mul ──────────────────────────────────────────
//
// out[i] = sigmoid(gate[i]) * out[i]
//
// Used by Qwen3.5 attention gate: apply sigmoid to the gate half of the
// Q projection, then multiply with the attention output in-place.

kernel void sigmoid_elementwise_mul_f32(
    device const float* gate [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float g = gate[gid];
    out[gid] = out[gid] / (1.0f + exp(-g));
}

// ── Sigmoid scalar mul ─────────────────────────────────────────────
//
// out[i] = sigmoid(gate[0]) * out[i]
//
// Used by Qwen3.5 shared-expert scalar gate path to avoid a separate
// sigmoid pass over the 1-element gate buffer and a broadcast-mul kernel.

kernel void sigmoid_scalar_mul_inplace_f32(
    device const float* gate [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    const float g = gate[0];
    out[gid] = out[gid] / (1.0f + exp(-g));
}

// ── Dense row dot + sigmoid scalar mul ────────────────────────────
//
// gate = dot(row[:k], x[:k])
// out[i] = sigmoid(gate) * out[i]
//
// Used by Qwen3.5 shared-expert scalar gate path to avoid a standalone
// matvec into a 1-element gate buffer followed by sigmoid_scalar_mul_inplace.

kernel void dense_row_dot_sigmoid_mul_inplace_f32(
    device const float* row [[buffer(0)]],
    device const float* x   [[buffer(1)]],
    device float* out       [[buffer(2)]],
    constant uint& k        [[buffer(3)]],
    constant uint& n        [[buffer(4)]],
    uint lid                [[thread_index_in_threadgroup]],
    uint simd_lane          [[thread_index_in_simdgroup]],
    uint simd_id            [[simdgroup_index_in_threadgroup]]
) {
    float dot_sum = 0.0f;
    for (uint i = lid; i < k; i += NORM_TG_SIZE) {
        dot_sum += row[i] * x[i];
    }

    dot_sum = simd_sum(dot_sum);

    constexpr uint n_groups = NORM_TG_SIZE / 32;
    threadgroup float simd_sums[n_groups];
    if (simd_lane == 0) {
        simd_sums[simd_id] = dot_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_gate;
    if (simd_id == 0) {
        dot_sum = (simd_lane < n_groups) ? simd_sums[simd_lane] : 0.0f;
        dot_sum = simd_sum(dot_sum);
        if (simd_lane == 0) {
            shared_gate = dot_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float gate = shared_gate;
    for (uint i = lid; i < n; i += NORM_TG_SIZE) {
        out[i] = out[i] / (1.0f + exp(-gate));
    }
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

// ── SiLU elementwise mul (batch, vec4) ─────────────────────────────
//
// Vectorized: processes 4 elements per thread.
// total = n * n_rows must be divisible by 4 (always true for LLM dims).
// Grid: ceil(total/4 / TG) threadgroups.

kernel void silu_elementwise_mul_batch_f32_vec4(
    device float* gate       [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    constant uint& n_rows    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint total4 = (n * n_rows) / 4;
    if (gid >= total4) return;
    device float4* g4 = (device float4*)gate;
    device const float4* u4 = (device const float4*)up;
    float4 x = g4[gid];
    float4 s = x / (1.0f + exp(-x)); // SiLU
    g4[gid] = s * u4[gid];
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

kernel void elementwise_add_batch_f32_vec4(
    device float* a          [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    constant uint& n_rows    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint total4 = (n * n_rows) / 4;
    if (gid >= total4) return;
    device float4* a4 = (device float4*)a;
    device const float4* b4 = (device const float4*)b;
    a4[gid] += b4[gid];
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

kernel void cast_f32_to_f16_vec4(
    device const float* src  [[buffer(0)]],
    device half* dst         [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint n4 = n / 4;
    if (gid >= n4) return;
    float4 v = ((device const float4*)src)[gid];
    ((device half4*)dst)[gid] = half4(v);
}

kernel void cast_f16_to_f32_vec4(
    device const half* src   [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint n4 = n / 4;
    if (gid >= n4) return;
    half4 v = ((device const half4*)src)[gid];
    ((device float4*)dst)[gid] = float4(v);
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

// ── Q+Gate split (batched) ───────────────────────────────────────────
//
// Split fused Q+gate projection output (Qwen3.5 full-attention layers):
//   src:  [n_rows, 2 * q_dim]   where [q_h0, g_h0, q_h1, g_h1, ...]
//   q:    [n_rows, q_dim]
//   gate: [n_rows, q_dim]

kernel void split_qgate_batch_f32(
    device const float* src      [[buffer(0)]],
    device float* q              [[buffer(1)]],
    device float* gate           [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& q_dim         [[buffer(4)]],
    constant uint& head_dim      [[buffer(5)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint total = n_rows * q_dim;
    if (gid >= total) return;

    uint row = gid / q_dim;
    uint q_col = gid % q_dim;
    uint head = q_col / head_dim;
    uint lane = q_col % head_dim;
    uint src_head_base = row * (2 * q_dim) + head * (2 * head_dim);

    q[gid] = src[src_head_base + lane];
    gate[gid] = src[src_head_base + head_dim + lane];
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
        uint cache_k_off = head * head_dim + 2 * i;
        cache_k[cache_row_base + cache_k_off] = rk0;
        cache_k[cache_row_base + cache_k_off + 1] = rk1;
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
        uint cache_k_off = head * head_dim + 2 * i;
        cache_k[cache_row_base + cache_k_off] = half(rk0);
        cache_k[cache_row_base + cache_k_off + 1] = half(rk1);
    } else {
        uint vc = local - v_start;
        uint src_idx = src_row_base + q_dim + kv_dim + vc;
        float vv = src[src_idx];
        v[kv_row_base + vc] = vv;
        cache_v[cache_row_base + vc] = half(vv);
    }
}

// ── QKV split + Q/K norm + RoPE + KV append Batch (f32 KV cache) ───────────
kernel void qkv_split_qk_norm_rope_append_kv_batch_f32(
    device const float* src        [[buffer(0)]],
    device float* q                [[buffer(1)]],
    device float* k                [[buffer(2)]],
    device float* v                [[buffer(3)]],
    device const float* q_weight   [[buffer(4)]],
    device const float* k_weight   [[buffer(5)]],
    device float* cache_k          [[buffer(6)]],
    device float* cache_v          [[buffer(7)]],
    constant uint& n_rows          [[buffer(8)]],
    constant uint& n_q_heads       [[buffer(9)]],
    constant uint& n_kv_heads      [[buffer(10)]],
    constant uint& head_dim        [[buffer(11)]],
    constant float& eps            [[buffer(12)]],
    constant float& start_pos      [[buffer(13)]],
    constant float& pos_step       [[buffer(14)]],
    constant float& freq_base      [[buffer(15)]],
    constant uint& cache_offset    [[buffer(16)]],
    constant uint& cache_stride    [[buffer(17)]],
    uint gid                       [[thread_position_in_grid]]
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
        uint head_base = src_row_base + head * head_dim;
        float sum_sq = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            float vv = src[head_base + j];
            sum_sq += vv * vv;
        }
        float inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        uint src_lo = head_base + i;
        uint src_hi = head_base + half_dim + i;
        uint dst_lo = q_row_base + head * head_dim + i;
        uint dst_hi = q_row_base + head * head_dim + half_dim + i;
        float v0 = src[src_lo] * inv_rms * q_weight[i];
        float v1 = src[src_hi] * inv_rms * q_weight[half_dim + i];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        q[dst_lo] = v0 * cos_t - v1 * sin_t;
        q[dst_hi] = v0 * sin_t + v1 * cos_t;
        return;
    }

    uint k_start = q_pairs;
    uint v_start = q_pairs + k_pairs;
    if (local < v_start) {
        uint k_local = local - k_start;
        uint head = k_local / half_dim;
        uint i = k_local % half_dim;
        uint head_base = src_row_base + q_dim + head * head_dim;
        float sum_sq = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            float vv = src[head_base + j];
            sum_sq += vv * vv;
        }
        float inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        uint src_lo = head_base + i;
        uint src_hi = head_base + half_dim + i;
        uint dst_lo = kv_row_base + head * head_dim + i;
        uint dst_hi = kv_row_base + head * head_dim + half_dim + i;
        float v0 = src[src_lo] * inv_rms * k_weight[i];
        float v1 = src[src_hi] * inv_rms * k_weight[half_dim + i];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        float rk0 = v0 * cos_t - v1 * sin_t;
        float rk1 = v0 * sin_t + v1 * cos_t;
        k[dst_lo] = rk0;
        k[dst_hi] = rk1;
        uint cache_k_lo = head * head_dim + i;
        uint cache_k_hi = head * head_dim + half_dim + i;
        cache_k[cache_row_base + cache_k_lo] = rk0;
        cache_k[cache_row_base + cache_k_hi] = rk1;
    } else {
        uint vc = local - v_start;
        uint src_idx = src_row_base + q_dim + kv_dim + vc;
        float vv = src[src_idx];
        v[kv_row_base + vc] = vv;
        cache_v[cache_row_base + vc] = vv;
    }
}

// ── QKV split + BIAS + Q/K norm + RoPE + KV append Batch (f32 KV cache) ────
//
// Same as qkv_split_qk_norm_rope_append_kv_batch_f32 but applies Q/K/V bias
// BEFORE norm+RoPE. Fuses 7 dispatches (split, 3 bias, norm+RoPE, 2 KV append)
// into 1. For Qwen3 which has per-layer QKV bias.
//
// Bias is broadcast: q_bias[d] added to every row's Q at dimension d.
kernel void qkv_split_bias_qknorm_rope_append_kv_batch_f32(
    device const float* src        [[buffer(0)]],
    device float* q                [[buffer(1)]],
    device float* k                [[buffer(2)]],
    device float* v                [[buffer(3)]],
    device const float* q_weight   [[buffer(4)]],
    device const float* k_weight   [[buffer(5)]],
    device float* cache_k          [[buffer(6)]],
    device float* cache_v          [[buffer(7)]],
    device const float* q_bias     [[buffer(8)]],
    device const float* k_bias     [[buffer(9)]],
    device const float* v_bias     [[buffer(10)]],
    constant uint& n_rows          [[buffer(11)]],
    constant uint& n_q_heads       [[buffer(12)]],
    constant uint& n_kv_heads      [[buffer(13)]],
    constant uint& head_dim        [[buffer(14)]],
    constant float& eps            [[buffer(15)]],
    constant float& start_pos      [[buffer(16)]],
    constant float& pos_step       [[buffer(17)]],
    constant float& freq_base      [[buffer(18)]],
    constant uint& cache_offset    [[buffer(19)]],
    constant uint& cache_stride    [[buffer(20)]],
    uint gid                       [[thread_position_in_grid]]
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
        // Q path: split + bias + per-head norm + RoPE
        uint head = local / half_dim;
        uint i = local % half_dim;
        uint head_base = src_row_base + head * head_dim;
        // Per-head RMSNorm with bias applied first
        float sum_sq = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            float vv = src[head_base + j] + q_bias[head * head_dim + j];
            sum_sq += vv * vv;
        }
        float inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        uint src_lo = head_base + i;
        uint src_hi = head_base + half_dim + i;
        uint dst_lo = q_row_base + head * head_dim + i;
        uint dst_hi = q_row_base + head * head_dim + half_dim + i;
        float v0 = (src[src_lo] + q_bias[head * head_dim + i]) * inv_rms * q_weight[i];
        float v1 =
            (src[src_hi] + q_bias[head * head_dim + half_dim + i]) * inv_rms * q_weight[half_dim + i];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        q[dst_lo] = v0 * cos_t - v1 * sin_t;
        q[dst_hi] = v0 * sin_t + v1 * cos_t;
        return;
    }

    uint k_start = q_pairs;
    uint v_start = q_pairs + k_pairs;
    if (local < v_start) {
        // K path: split + bias + per-head norm + RoPE + KV cache append
        uint k_local = local - k_start;
        uint head = k_local / half_dim;
        uint i = k_local % half_dim;
        uint head_base = src_row_base + q_dim + head * head_dim;
        float sum_sq = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            float vv = src[head_base + j] + k_bias[head * head_dim + j];
            sum_sq += vv * vv;
        }
        float inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        uint src_lo = head_base + i;
        uint src_hi = head_base + half_dim + i;
        uint dst_lo = kv_row_base + head * head_dim + i;
        uint dst_hi = kv_row_base + head * head_dim + half_dim + i;
        float v0 = (src[src_lo] + k_bias[head * head_dim + i]) * inv_rms * k_weight[i];
        float v1 =
            (src[src_hi] + k_bias[head * head_dim + half_dim + i]) * inv_rms * k_weight[half_dim + i];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        float rk0 = v0 * cos_t - v1 * sin_t;
        float rk1 = v0 * sin_t + v1 * cos_t;
        k[dst_lo] = rk0;
        k[dst_hi] = rk1;
        uint cache_k_lo = head * head_dim + i;
        uint cache_k_hi = head * head_dim + half_dim + i;
        cache_k[cache_row_base + cache_k_lo] = rk0;
        cache_k[cache_row_base + cache_k_hi] = rk1;
    } else {
        // V path: split + bias + KV cache append (no norm or RoPE for V)
        uint vc = local - v_start;
        float vv = src[src_row_base + q_dim + kv_dim + vc] + v_bias[vc];
        v[kv_row_base + vc] = vv;
        cache_v[cache_row_base + vc] = vv;
    }
}

// ── QKV split + BIAS + Q/K norm + RoPE + KV append Batch (f16 KV cache) ────
//
// Same as f32 bias kernel but writes half to KV cache.
// This is the DEFAULT path for Qwen3 (kv_f16 = true when max_seq_len ≥ 256).
kernel void qkv_split_bias_qknorm_rope_append_kv_batch_f16(
    device const float* src        [[buffer(0)]],
    device float* q                [[buffer(1)]],
    device float* k                [[buffer(2)]],
    device float* v                [[buffer(3)]],
    device const float* q_weight   [[buffer(4)]],
    device const float* k_weight   [[buffer(5)]],
    device half* cache_k           [[buffer(6)]],
    device half* cache_v           [[buffer(7)]],
    device const float* q_bias     [[buffer(8)]],
    device const float* k_bias     [[buffer(9)]],
    device const float* v_bias     [[buffer(10)]],
    constant uint& n_rows          [[buffer(11)]],
    constant uint& n_q_heads       [[buffer(12)]],
    constant uint& n_kv_heads      [[buffer(13)]],
    constant uint& head_dim        [[buffer(14)]],
    constant float& eps            [[buffer(15)]],
    constant float& start_pos      [[buffer(16)]],
    constant float& pos_step       [[buffer(17)]],
    constant float& freq_base      [[buffer(18)]],
    constant uint& cache_offset    [[buffer(19)]],
    constant uint& cache_stride    [[buffer(20)]],
    uint gid                       [[thread_position_in_grid]]
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
        uint head_base = src_row_base + head * head_dim;
        float sum_sq = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            float vv = src[head_base + j] + q_bias[head * head_dim + j];
            sum_sq += vv * vv;
        }
        float inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        uint src_lo = head_base + i;
        uint src_hi = head_base + half_dim + i;
        uint dst_lo = q_row_base + head * head_dim + i;
        uint dst_hi = q_row_base + head * head_dim + half_dim + i;
        float v0 = (src[src_lo] + q_bias[head * head_dim + i]) * inv_rms * q_weight[i];
        float v1 =
            (src[src_hi] + q_bias[head * head_dim + half_dim + i]) * inv_rms * q_weight[half_dim + i];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        q[dst_lo] = v0 * cos_t - v1 * sin_t;
        q[dst_hi] = v0 * sin_t + v1 * cos_t;
        return;
    }

    uint k_start = q_pairs;
    uint v_start = q_pairs + k_pairs;
    if (local < v_start) {
        uint k_local = local - k_start;
        uint head = k_local / half_dim;
        uint i = k_local % half_dim;
        uint head_base = src_row_base + q_dim + head * head_dim;
        float sum_sq = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            float vv = src[head_base + j] + k_bias[head * head_dim + j];
            sum_sq += vv * vv;
        }
        float inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        uint src_lo = head_base + i;
        uint src_hi = head_base + half_dim + i;
        uint dst_lo = kv_row_base + head * head_dim + i;
        uint dst_hi = kv_row_base + head * head_dim + half_dim + i;
        float v0 = (src[src_lo] + k_bias[head * head_dim + i]) * inv_rms * k_weight[i];
        float v1 =
            (src[src_hi] + k_bias[head * head_dim + half_dim + i]) * inv_rms * k_weight[half_dim + i];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        float rk0 = v0 * cos_t - v1 * sin_t;
        float rk1 = v0 * sin_t + v1 * cos_t;
        k[dst_lo] = rk0;
        k[dst_hi] = rk1;
        uint cache_k_lo = head * head_dim + i;
        uint cache_k_hi = head * head_dim + half_dim + i;
        cache_k[cache_row_base + cache_k_lo] = half(rk0);
        cache_k[cache_row_base + cache_k_hi] = half(rk1);
    } else {
        uint vc = local - v_start;
        float vv = src[src_row_base + q_dim + kv_dim + vc] + v_bias[vc];
        v[kv_row_base + vc] = vv;
        cache_v[cache_row_base + vc] = half(vv);
    }
}

// ── QKV split + Q/K norm + RoPE + KV append Batch (f16 KV cache) ───────────
kernel void qkv_split_qk_norm_rope_append_kv_batch_f16(
    device const float* src        [[buffer(0)]],
    device float* q                [[buffer(1)]],
    device float* k                [[buffer(2)]],
    device float* v                [[buffer(3)]],
    device const float* q_weight   [[buffer(4)]],
    device const float* k_weight   [[buffer(5)]],
    device half* cache_k           [[buffer(6)]],
    device half* cache_v           [[buffer(7)]],
    constant uint& n_rows          [[buffer(8)]],
    constant uint& n_q_heads       [[buffer(9)]],
    constant uint& n_kv_heads      [[buffer(10)]],
    constant uint& head_dim        [[buffer(11)]],
    constant float& eps            [[buffer(12)]],
    constant float& start_pos      [[buffer(13)]],
    constant float& pos_step       [[buffer(14)]],
    constant float& freq_base      [[buffer(15)]],
    constant uint& cache_offset    [[buffer(16)]],
    constant uint& cache_stride    [[buffer(17)]],
    uint gid                       [[thread_position_in_grid]]
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
        uint head_base = src_row_base + head * head_dim;
        float sum_sq = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            float vv = src[head_base + j];
            sum_sq += vv * vv;
        }
        float inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        uint src_lo = head_base + i;
        uint src_hi = head_base + half_dim + i;
        uint dst_lo = q_row_base + head * head_dim + i;
        uint dst_hi = q_row_base + head * head_dim + half_dim + i;
        float v0 = src[src_lo] * inv_rms * q_weight[i];
        float v1 = src[src_hi] * inv_rms * q_weight[half_dim + i];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        q[dst_lo] = v0 * cos_t - v1 * sin_t;
        q[dst_hi] = v0 * sin_t + v1 * cos_t;
        return;
    }

    uint k_start = q_pairs;
    uint v_start = q_pairs + k_pairs;
    if (local < v_start) {
        uint k_local = local - k_start;
        uint head = k_local / half_dim;
        uint i = k_local % half_dim;
        uint head_base = src_row_base + q_dim + head * head_dim;
        float sum_sq = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            float vv = src[head_base + j];
            sum_sq += vv * vv;
        }
        float inv_rms = rsqrt(sum_sq / float(head_dim) + eps);
        uint src_lo = head_base + i;
        uint src_hi = head_base + half_dim + i;
        uint dst_lo = kv_row_base + head * head_dim + i;
        uint dst_hi = kv_row_base + head * head_dim + half_dim + i;
        float v0 = src[src_lo] * inv_rms * k_weight[i];
        float v1 = src[src_hi] * inv_rms * k_weight[half_dim + i];
        float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(head_dim));
        float theta = position * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        float rk0 = v0 * cos_t - v1 * sin_t;
        float rk1 = v0 * sin_t + v1 * cos_t;
        k[dst_lo] = rk0;
        k[dst_hi] = rk1;
        uint cache_k_lo = head * head_dim + i;
        uint cache_k_hi = head * head_dim + half_dim + i;
        cache_k[cache_row_base + cache_k_lo] = half(rk0);
        cache_k[cache_row_base + cache_k_hi] = half(rk1);
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

// ═══════════════════════════════════════════════════════════════════════════
// MoE weighted elementwise add: dst[i] += scale * src[i]
// Used to accumulate weighted expert outputs.
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Q8_0 KV cache append — quantize f32 → Q8_0 blocks on GPU
//
// Q8_0 block: half d (2 bytes) + char qs[32] (32 bytes) = 34 bytes per 32 values.
// Each thread handles one Q8_0 block: finds max(abs), computes scale, quantizes.
//
// dst layout: [token_row][block_idx] where each block is 34 bytes.
// src layout: [token_row][kv_stride] where kv_stride = n_kv_heads * head_dim.
// ═══════════════════════════════════════════════════════════════════════════

constant uint Q8_KV_BLOCK_SIZE = 32;
constant uint Q8_KV_BLOCK_BYTES = 34;

kernel void kv_append_batch_q8_0(
    device const float* src          [[buffer(0)]],
    device uchar*       dst          [[buffer(1)]],
    constant uint& dst_row_offset    [[buffer(2)]],
    constant uint& blocks_per_row    [[buffer(3)]],
    constant uint& kv_stride         [[buffer(4)]],
    constant uint& n_rows            [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total_blocks = n_rows * blocks_per_row;
    if (gid >= total_blocks) return;

    uint row = gid / blocks_per_row;
    uint block_in_row = gid % blocks_per_row;

    device const float* block_src = src + row * kv_stride + block_in_row * Q8_KV_BLOCK_SIZE;
    uint dst_block_idx = dst_row_offset + row * blocks_per_row + block_in_row;
    device uchar* block_dst = dst + dst_block_idx * Q8_KV_BLOCK_BYTES;

    float amax = 0.0f;
    for (uint i = 0; i < Q8_KV_BLOCK_SIZE; i++) {
        amax = max(amax, abs(block_src[i]));
    }
    float d = amax / 127.0f;
    float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    *reinterpret_cast<device half*>(block_dst) = half(d);
    for (uint i = 0; i < Q8_KV_BLOCK_SIZE; i++) {
        int q = clamp(int(round(block_src[i] * id)), -128, 127);
        block_dst[2 + i] = uchar(char(q));
    }
}

kernel void kv_append_batch2_q8_0(
    device const float* src_k        [[buffer(0)]],
    device const float* src_v        [[buffer(1)]],
    device uchar*       dst_k        [[buffer(2)]],
    device uchar*       dst_v        [[buffer(3)]],
    constant uint& dst_row_offset    [[buffer(4)]],
    constant uint& blocks_per_row    [[buffer(5)]],
    constant uint& kv_stride         [[buffer(6)]],
    constant uint& n_rows            [[buffer(7)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total_blocks = n_rows * blocks_per_row;
    if (gid >= total_blocks) return;

    uint row = gid / blocks_per_row;
    uint block_in_row = gid % blocks_per_row;
    uint dst_block_idx = dst_row_offset + row * blocks_per_row + block_in_row;

    // Quantize K
    {
        device const float* bs = src_k + row * kv_stride + block_in_row * Q8_KV_BLOCK_SIZE;
        device uchar* bd = dst_k + dst_block_idx * Q8_KV_BLOCK_BYTES;
        float amax = 0.0f;
        for (uint i = 0; i < Q8_KV_BLOCK_SIZE; i++) amax = max(amax, abs(bs[i]));
        float d = amax / 127.0f;
        float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        *reinterpret_cast<device half*>(bd) = half(d);
        for (uint i = 0; i < Q8_KV_BLOCK_SIZE; i++)
            bd[2 + i] = uchar(char(clamp(int(round(bs[i] * id)), -128, 127)));
    }

    // Quantize V
    {
        device const float* bs = src_v + row * kv_stride + block_in_row * Q8_KV_BLOCK_SIZE;
        device uchar* bd = dst_v + dst_block_idx * Q8_KV_BLOCK_BYTES;
        float amax = 0.0f;
        for (uint i = 0; i < Q8_KV_BLOCK_SIZE; i++) amax = max(amax, abs(bs[i]));
        float d = amax / 127.0f;
        float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        *reinterpret_cast<device half*>(bd) = half(d);
        for (uint i = 0; i < Q8_KV_BLOCK_SIZE; i++)
            bd[2 + i] = uchar(char(clamp(int(round(bs[i] * id)), -128, 127)));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q4_0 KV cache append — quantize f32 → Q4_0 blocks on GPU
//
// Q4_0 block: half d (2 bytes) + nibble pairs[16] (16 bytes) = 18 bytes per 32 values.
// Each thread handles one Q4_0 block: finds amax, computes scale, packs nibbles.
//
// dst layout: [token_row][block_idx] where each block is 18 bytes.
// src layout: [token_row][kv_stride] where kv_stride = n_kv_heads * head_dim.
// ═══════════════════════════════════════════════════════════════════════════

constant uint Q4_KV_BLOCK_SIZE = 32;
constant uint Q4_KV_BLOCK_BYTES = 18;

kernel void kv_append_batch_q4_0(
    device const float* src          [[buffer(0)]],
    device uchar*       dst          [[buffer(1)]],
    constant uint& dst_row_offset    [[buffer(2)]],
    constant uint& blocks_per_row    [[buffer(3)]],
    constant uint& kv_stride         [[buffer(4)]],
    constant uint& n_rows            [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total_blocks = n_rows * blocks_per_row;
    if (gid >= total_blocks) return;

    uint row = gid / blocks_per_row;
    uint block_in_row = gid % blocks_per_row;

    device const float* block_src = src + row * kv_stride + block_in_row * Q4_KV_BLOCK_SIZE;
    uint dst_block_idx = dst_row_offset + row * blocks_per_row + block_in_row;
    device uchar* block_dst = dst + dst_block_idx * Q4_KV_BLOCK_BYTES;

    float amax = 0.0f;
    for (uint i = 0; i < Q4_KV_BLOCK_SIZE; i++) {
        amax = max(amax, abs(block_src[i]));
    }
    float d = amax / 15.0f;
    float id = (amax > 0.0f) ? (15.0f / amax) : 0.0f;
    *reinterpret_cast<device half*>(block_dst) = half(d);
    for (uint i = 0; i < 16; i++) {
        uint q_lo = clamp(int(round(block_src[i] * id + 8.0f)), 0, 15);
        uint q_hi = clamp(int(round(block_src[i + 16] * id + 8.0f)), 0, 15);
        block_dst[2 + i] = uchar((q_lo & 0x0F) | (q_hi << 4));
    }
}

kernel void kv_append_batch2_q4_0(
    device const float* src_k        [[buffer(0)]],
    device const float* src_v        [[buffer(1)]],
    device uchar*       dst_k        [[buffer(2)]],
    device uchar*       dst_v        [[buffer(3)]],
    constant uint& dst_row_offset    [[buffer(4)]],
    constant uint& blocks_per_row    [[buffer(5)]],
    constant uint& kv_stride         [[buffer(6)]],
    constant uint& n_rows            [[buffer(7)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total_blocks = n_rows * blocks_per_row;
    if (gid >= total_blocks) return;

    uint row = gid / blocks_per_row;
    uint block_in_row = gid % blocks_per_row;
    uint dst_block_idx = dst_row_offset + row * blocks_per_row + block_in_row;

    // Quantize K
    {
        device const float* bs = src_k + row * kv_stride + block_in_row * Q4_KV_BLOCK_SIZE;
        device uchar* bd = dst_k + dst_block_idx * Q4_KV_BLOCK_BYTES;
        float amax = 0.0f;
        for (uint i = 0; i < Q4_KV_BLOCK_SIZE; i++) amax = max(amax, abs(bs[i]));
        float d = amax / 15.0f;
        float id = (amax > 0.0f) ? (15.0f / amax) : 0.0f;
        *reinterpret_cast<device half*>(bd) = half(d);
        for (uint i = 0; i < 16; i++) {
            uint q_lo = clamp(int(round(bs[i] * id + 8.0f)), 0, 15);
            uint q_hi = clamp(int(round(bs[i + 16] * id + 8.0f)), 0, 15);
            bd[2 + i] = uchar((q_lo & 0x0F) | (q_hi << 4));
        }
    }

    // Quantize V
    {
        device const float* bs = src_v + row * kv_stride + block_in_row * Q4_KV_BLOCK_SIZE;
        device uchar* bd = dst_v + dst_block_idx * Q4_KV_BLOCK_BYTES;
        float amax = 0.0f;
        for (uint i = 0; i < Q4_KV_BLOCK_SIZE; i++) amax = max(amax, abs(bs[i]));
        float d = amax / 15.0f;
        float id = (amax > 0.0f) ? (15.0f / amax) : 0.0f;
        *reinterpret_cast<device half*>(bd) = half(d);
        for (uint i = 0; i < 16; i++) {
            uint q_lo = clamp(int(round(bs[i] * id + 8.0f)), 0, 15);
            uint q_hi = clamp(int(round(bs[i + 16] * id + 8.0f)), 0, 15);
            bd[2 + i] = uchar((q_lo & 0x0F) | (q_hi << 4));
        }
    }
}

kernel void elementwise_weighted_add_f32(
    device float* dst          [[buffer(0)]],
    device const float* src    [[buffer(1)]],
    constant float& scale      [[buffer(2)]],
    constant uint& count       [[buffer(3)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid < count) {
        dst[gid] += scale * src[gid];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// General-purpose elementwise ops for GDN chunked graph and other uses.
// All operate on flat f32 buffers of `count` elements.
// Grid: (ceil(count/256), 1, 1).  TG: 256.
// ═══════════════════════════════════════════════════════════════════════════

/// dst = exp(src)
kernel void elementwise_exp_f32(
    const device float *src [[buffer(0)]],
    device float *dst       [[buffer(1)]],
    constant uint &count    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) dst[gid] = exp(src[gid]);
}

/// dst = src_a * src_b (elementwise)
kernel void elementwise_mul_f32(
    const device float *src_a [[buffer(0)]],
    const device float *src_b [[buffer(1)]],
    device float *dst         [[buffer(2)]],
    constant uint &count      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) dst[gid] = src_a[gid] * src_b[gid];
}

/// dst = src_a + src_b (elementwise)
kernel void elementwise_add_out_f32(
    const device float *src_a [[buffer(0)]],
    const device float *src_b [[buffer(1)]],
    device float *dst         [[buffer(2)]],
    constant uint &count      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) dst[gid] = src_a[gid] + src_b[gid];
}

/// dst = src_a - src_b (elementwise)
kernel void elementwise_sub_f32(
    const device float *src_a [[buffer(0)]],
    const device float *src_b [[buffer(1)]],
    device float *dst         [[buffer(2)]],
    constant uint &count      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) dst[gid] = src_a[gid] - src_b[gid];
}

/// dst = -src (elementwise negate)
kernel void elementwise_neg_f32(
    const device float *src [[buffer(0)]],
    device float *dst       [[buffer(1)]],
    constant uint &count    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) dst[gid] = -src[gid];
}

/// dst = src * scale (elementwise broadcast scalar)
kernel void elementwise_scale_f32(
    const device float *src [[buffer(0)]],
    device float *dst       [[buffer(1)]],
    constant float &scale   [[buffer(2)]],
    constant uint &count    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) dst[gid] = src[gid] * scale;
}

// ═══════════════════════════════════════════════════════════════════════════
// MoE GPU gather/scatter kernels for expert dispatch.
// ═══════════════════════════════════════════════════════════════════════════

/// Gather rows by index: for each slot s, dst[s*dim..] = src[indices[s]*dim..].
/// indices is [n_slots], src is [n_tokens, dim], dst is [n_slots, dim].
/// Grid: (ceil(n_slots * dim / 256), 1, 1).  TG: 256.
kernel void moe_gather_rows_f32(
    const device float *src     [[buffer(0)]],
    const device uint  *indices [[buffer(1)]],
    device float *dst           [[buffer(2)]],
    constant uint &dim          [[buffer(3)]],
    constant uint &n_slots      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = n_slots * dim;
    if (gid >= total) return;
    const uint slot = gid / dim;
    const uint d = gid % dim;
    dst[gid] = src[indices[slot] * dim + d];
}

/// Weighted scatter-add: for each slot s, dst[indices[s]*dim + d] += weights[s] * src[s*dim + d].
/// indices is [n_slots], weights is [n_slots], src is [n_slots, dim], dst is [n_tokens, dim].
/// Grid: (ceil(n_slots * dim / 256), 1, 1).  TG: 256.
/// NOTE: Multiple slots may map to the same dst row. The caller must ensure
/// no two slots in the same dispatch map to the same index (split by expert).
kernel void moe_weighted_scatter_add_f32(
    const device float *src     [[buffer(0)]],
    const device uint  *indices [[buffer(1)]],
    const device float *weights [[buffer(2)]],
    device float *dst           [[buffer(3)]],
    constant uint &dim          [[buffer(4)]],
    constant uint &n_slots      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = n_slots * dim;
    if (gid >= total) return;
    const uint slot = gid / dim;
    const uint d = gid % dim;
    dst[indices[slot] * dim + d] += weights[slot] * src[gid];
}

/// Weighted per-token slot reduction:
/// dst[token, d] += sum_k weights[token, k] * src[token, k, d].
///
/// This is the safe MoE reduction path for routed experts because each token
/// intentionally contributes multiple expert slots. Reducing inside one thread
/// avoids the write races that would otherwise occur with scatter-add.
///
/// src layout:     [n_tokens * n_expert_used, dim]
/// weights layout: [n_tokens * n_expert_used]
/// dst layout:     [n_tokens, dim]
kernel void moe_weighted_reduce_slots_add_f32(
    const device float *src            [[buffer(0)]],
    const device float *weights        [[buffer(1)]],
    device float *dst                  [[buffer(2)]],
    constant uint &dim                 [[buffer(3)]],
    constant uint &n_tokens            [[buffer(4)]],
    constant uint &n_expert_used       [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = n_tokens * dim;
    if (gid >= total) return;

    const uint token = gid / dim;
    const uint d = gid % dim;
    const uint slot_base = token * n_expert_used;

    float acc = 0.0f;
    for (uint k = 0; k < n_expert_used; ++k) {
        const uint slot = slot_base + k;
        acc += weights[slot] * src[slot * dim + d];
    }

    dst[gid] += acc;
}

// Specialized routed-expert reduction for n_expert_used == 8.
// Processes 4 output dimensions per thread to reduce loop/control overhead
// on the Qwen3.5-35B-A3B single-token decode path.
kernel void moe_weighted_reduce_slots8_add_f32_vec4(
    const device float4 *src4          [[buffer(0)]],
    const device float *weights        [[buffer(1)]],
    device float4 *dst4                [[buffer(2)]],
    constant uint &dim                 [[buffer(3)]],
    constant uint &n_tokens            [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint dim4 = dim / 4;
    const uint total4 = n_tokens * dim4;
    if (gid >= total4) return;

    const uint token = gid / dim4;
    const uint d4 = gid % dim4;
    const uint slot_base = token * 8;

    float4 acc = float4(0.0f);
    acc += weights[slot_base + 0] * src4[(slot_base + 0) * dim4 + d4];
    acc += weights[slot_base + 1] * src4[(slot_base + 1) * dim4 + d4];
    acc += weights[slot_base + 2] * src4[(slot_base + 2) * dim4 + d4];
    acc += weights[slot_base + 3] * src4[(slot_base + 3) * dim4 + d4];
    acc += weights[slot_base + 4] * src4[(slot_base + 4) * dim4 + d4];
    acc += weights[slot_base + 5] * src4[(slot_base + 5) * dim4 + d4];
    acc += weights[slot_base + 6] * src4[(slot_base + 6) * dim4 + d4];
    acc += weights[slot_base + 7] * src4[(slot_base + 7) * dim4 + d4];

    dst4[token * dim4 + d4] += acc;
}

// ---------------------------------------------------------------------------
// argmax_f32 — GPU-side argmax for greedy (temperature=0) decode.
//
// Finds the index and value of the maximum element in an f32 buffer, avoiding
// the need to read back 248K+ logit floats to the CPU.
//
// Single threadgroup dispatch: TG = 1024.
// Phase 1: each thread scans ceil(n / TG) elements for a local max.
// Phase 2: within-simdgroup reduction via simd_max + simd_shuffle.
// Phase 3: cross-simdgroup reduction in threadgroup memory; thread 0 writes.
// ---------------------------------------------------------------------------
kernel void argmax_f32(
    device const float* data   [[buffer(0)]],
    device uint*  result_idx   [[buffer(1)]],
    device float* result_val   [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    constexpr uint TG = 1024;
    constexpr uint N_SIMD = TG / 32;  // 32 simdgroups

    // Phase 1: each thread finds its local max over a strided chunk.
    float local_max = -INFINITY;
    uint local_idx = 0;
    for (uint i = tid; i < n; i += TG) {
        float v = data[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Phase 2: within-simdgroup reduction.
    // Find the simdgroup-wide maximum value.
    float sg_max = simd_max(local_max);
    // Identify which lane holds the max (ties: take lowest lane index).
    bool is_max_lane = (local_max == sg_max);
    uint max_lane = simd_min(is_max_lane ? simd_lane : 32u);
    uint sg_idx = simd_shuffle(local_idx, max_lane);

    // Phase 3: cross-simdgroup reduction via threadgroup memory.
    threadgroup float sg_maxvals[N_SIMD];
    threadgroup uint  sg_maxidxs[N_SIMD];
    if (simd_lane == 0) {
        sg_maxvals[simd_id] = sg_max;
        sg_maxidxs[simd_id] = sg_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 reduces across all simdgroups and writes the result.
    if (tid == 0) {
        float best_val = sg_maxvals[0];
        uint best_idx = sg_maxidxs[0];
        for (uint s = 1; s < N_SIMD; s++) {
            if (sg_maxvals[s] > best_val) {
                best_val = sg_maxvals[s];
                best_idx = sg_maxidxs[s];
            }
        }
        *result_idx = best_idx;
        *result_val = best_val;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MoE GPU softmax + top-k — compute expert routing entirely on GPU.
//
// For each token: softmax over n_expert logits, select top n_expert_used,
// re-normalize selected weights, write expert_ids and expert_weights.
//
// Input:  router_logits [n_tokens, n_expert] f32
// Output: expert_ids    [n_tokens, n_expert_used] i32
//         expert_weights[n_tokens, n_expert_used] f32
//
// Grid: (n_tokens, 1, 1). TG: min(n_expert, 256).
// Each threadgroup processes one token's expert selection.
// ═══════════════════════════════════════════════════════════════════════════
kernel void moe_softmax_topk_f32(
    device const float *router_logits [[buffer(0)]],
    device int32_t *expert_ids        [[buffer(1)]],
    device float *expert_weights      [[buffer(2)]],
    constant uint &n_expert           [[buffer(3)]],
    constant uint &n_expert_used      [[buffer(4)]],
    threadgroup float *shmem          [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint ntg   [[threads_per_threadgroup]])
{
    const uint token = tgpig;
    device const float *logits = router_logits + token * n_expert;
    device int32_t *out_ids = expert_ids + token * n_expert_used;
    device float *out_wts = expert_weights + token * n_expert_used;

    threadgroup float *sh_logits = shmem;                    // [n_expert]
    threadgroup float *sh_probs  = shmem + n_expert;         // [n_expert]
    threadgroup int   *sh_sel    = (threadgroup int *)(sh_probs + n_expert); // [n_expert_used]
    threadgroup float *sh_wts    = (threadgroup float *)(sh_sel + n_expert_used);

    // Step 1: Load logits + find max (for stable softmax).
    float local_max = -INFINITY;
    for (uint i = tid; i < n_expert; i += ntg) {
        float v = logits[i];
        sh_logits[i] = v;
        local_max = max(local_max, v);
    }
    // Reduce max across threadgroup.
    threadgroup float *sh_reduce = sh_wts + n_expert_used; // temp
    sh_reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ntg / 2; s > 0; s >>= 1) {
        if (tid < s) sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gmax = sh_reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute exp(logit - max) and sum.
    float local_sum = 0.0f;
    for (uint i = tid; i < n_expert; i += ntg) {
        float e = exp(sh_logits[i] - gmax);
        sh_probs[i] = e;
        local_sum += e;
    }
    sh_reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ntg / 2; s > 0; s >>= 1) {
        if (tid < s) sh_reduce[tid] += sh_reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gsum = sh_reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Normalize to probabilities.
    float inv_sum = (gsum > 0.0f) ? 1.0f / gsum : 0.0f;
    for (uint i = tid; i < n_expert; i += ntg) {
        sh_probs[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Top-k selection (sequential on thread 0, n_expert_used is small ≤16).
    if (tid == 0) {
        for (uint k = 0; k < n_expert_used; k++) {
            float best_val = -1.0f;
            int best_idx = 0;
            for (uint i = 0; i < n_expert; i++) {
                if (sh_probs[i] > best_val) {
                    best_val = sh_probs[i];
                    best_idx = i;
                }
            }
            sh_sel[k] = best_idx;
            sh_wts[k] = best_val;
            sh_probs[best_idx] = -1.0f; // Mark as selected.
        }
        // Re-normalize selected weights.
        float sel_sum = 0.0f;
        for (uint k = 0; k < n_expert_used; k++) sel_sum += sh_wts[k];
        float inv_sel = (sel_sum > 0.0f) ? 1.0f / sel_sum : 0.0f;
        for (uint k = 0; k < n_expert_used; k++) {
            out_ids[k] = sh_sel[k];
            out_wts[k] = sh_wts[k] * inv_sel;
        }
    }
}

// Apply per-expert output scales to selected routing weights:
//   expert_weights[token, slot] *= expert_scales[expert_ids[token, slot]]
kernel void moe_apply_expert_scales_f32(
    device const int32_t *expert_ids   [[buffer(0)]],
    device float *expert_weights       [[buffer(1)]],
    device const float *expert_scales  [[buffer(2)]],
    constant uint &n_tokens            [[buffer(3)]],
    constant uint &n_expert_used       [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint total = n_tokens * n_expert_used;
    if (gid >= total) return;

    int eid = expert_ids[gid];
    if (eid < 0) return;
    expert_weights[gid] *= expert_scales[(uint)eid];
}
