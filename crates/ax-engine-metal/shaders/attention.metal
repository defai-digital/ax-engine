// AX Engine — Attention compute shader
//
// FlashAttention-style tiled attention for prefill with online softmax.
// Avoids materializing the full N×N attention matrix — processes the KV
// sequence in tiles of ATTN_TG tokens, maintaining running softmax state.
//
// Grid layout: (n_tokens, n_heads) threadgroups
//   Each threadgroup computes attention output for one (query_position, head).
//   Threads cooperate on Q·K dot products (Phase 1) and V accumulation (Phase 4).

// Use hardware-accelerated fast::exp for all softmax paths.
// ~2 ULP precision (vs ~1 ULP for standard exp) is sufficient for softmax
// since only relative magnitudes matter after normalization.
// Matches MLX and llama.cpp which both use fast:: math in attention.
#define ax_exp(x) fast::exp(x)
//
// Supports GQA: n_heads query heads share n_kv_heads key/value heads.
// Causal masking: query at position qi attends only to positions 0..qi.
//
// Memory layout (row-major):
//   Q: [n_tokens, n_heads, head_dim]
//   K: [n_tokens, n_kv_heads, head_dim]
//   V: [n_tokens, n_kv_heads, head_dim]
//   O: [n_tokens, n_heads, head_dim]

#include <metal_stdlib>
using namespace metal;

constant uint ATTN_TG = 256;   // threads per threadgroup
constant uint MAX_HD  = 512;   // max supported head_dim for generic kernels
constant uint N_SIMD  = ATTN_TG / 32;  // SIMD groups per threadgroup
constant uint ATTN_DEC2_TG = 128;      // alternative decode threadgroup size
constant uint N_DEC2_SIMD  = ATTN_DEC2_TG / 32;

kernel void attention_prefill_f32(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device float* O            [[buffer(3)]],
    constant uint& n_tokens    [[buffer(4)]],
    constant uint& n_heads     [[buffer(5)]],
    constant uint& n_kv_heads  [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 tg_id                [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint qi = tg_id.x;   // query position
    uint h  = tg_id.y;   // query head index
    if (qi >= n_tokens || h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride  = n_heads * head_dim;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));
    uint attend_len = qi + 1;  // causal: positions 0..qi inclusive

    // ── Threadgroup memory ───────────────────────────────────────────
    threadgroup float q_shared[MAX_HD];       // query vector
    threadgroup float out_acc[MAX_HD];        // output accumulator
    threadgroup float tile_scores[ATTN_TG];   // ax_exp(score) per KV token in tile
    threadgroup float simd_buf[N_SIMD];       // SIMD reduction scratch
    threadgroup float shared_max;             // running softmax max
    threadgroup float shared_sum;             // running softmax denominator

    // ── Initialize ───────────────────────────────────────────────────
    device const float* q_ptr = Q + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Tile loop over KV sequence ───────────────────────────────────
    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        // Phase 1: Each thread computes one Q·K score
        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = tile_start + lid;
            device const float* k_ptr = K + t * kv_stride + kv_h * head_dim;
            float s = 0.0f;
            // Vectorize dot product in 4-float chunks (major attention hotspot).
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(k_ptr[d], k_ptr[d + 1], k_ptr[d + 2], k_ptr[d + 3]);
                s += dot(qv, kv);
            }
            for (; d < head_dim; d++) {
                s += q_shared[d] * k_ptr[d];
            }
            my_score = s * scale;
        }

        // Phase 2: Find tile-wide max score (two-level SIMD reduction)
        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;  // broadcast via slot 0
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        // Phase 3: Online softmax update
        //   new_max = max(running_max, tile_max)
        //   correction = ax_exp(running_max - new_max)   [rescales prior output]
        //   exp_score = ax_exp(score - new_max)           [this tile's weights]
        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        // Reduce exp sum (two-level SIMD reduction)
        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            // Update running softmax state
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: Rescale existing output and accumulate weighted V
        //   out_acc = out_acc * correction + Σ(exp_score[s] * V[s])
        //   Threads split across head_dim dimensions.
        //   Hoist base pointer and use stride to avoid recomputing per-s.
        device const float* v_base =
            V + tile_start * kv_stride + kv_h * head_dim;
        for (uint d = lid; d < head_dim; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s    ] * v_base[s       * kv_stride + d];
                acc += tile_scores[s + 1] * v_base[(s + 1) * kv_stride + d];
                acc += tile_scores[s + 2] * v_base[(s + 2) * kv_stride + d];
                acc += tile_scores[s + 3] * v_base[(s + 3) * kv_stride + d];
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Finalize: normalize by softmax denominator ───────────────────
    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Specialized prefill kernel for head_dim == 256.
// Keeps the same ABI as attention_prefill_f32 but removes dynamic head_dim loops.
kernel void attention_prefill_f32_hd256(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device float* O            [[buffer(3)]],
    constant uint& n_tokens    [[buffer(4)]],
    constant uint& n_heads     [[buffer(5)]],
    constant uint& n_kv_heads  [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 tg_id                [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint qi = tg_id.x;
    uint h  = tg_id.y;
    if (qi >= n_tokens || h >= n_heads || head_dim != 256) return;

    constexpr uint HD = 256;
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride  = n_heads * HD;
    uint kv_stride = n_kv_heads * HD;
    constexpr float scale = 1.0f / 16.0f;  // rsqrt(256)
    uint attend_len = qi + 1;

    threadgroup float q_shared[HD];
    threadgroup float out_acc[HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + qi * q_stride + h * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = tile_start + lid;
            device const float* k_ptr = K + t * kv_stride + kv_h * HD;
            float s = 0.0f;
            for (uint d = 0; d < HD; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(k_ptr[d], k_ptr[d + 1], k_ptr[d + 2], k_ptr[d + 3]);
                s += dot(qv, kv);
            }
            my_score = s * scale;
        }

        float v0 = (lid < tile_len) ? my_score : -INFINITY;
        v0 = simd_max(v0);
        if (simd_lane == 0) simd_buf[simd_id] = v0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float* v_base = V + tile_start * kv_stride + kv_h * HD;
        for (uint d = lid; d < HD; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s    ] * v_base[s       * kv_stride + d];
                acc += tile_scores[s + 1] * v_base[(s + 1) * kv_stride + d];
                acc += tile_scores[s + 2] * v_base[(s + 2) * kv_stride + d];
                acc += tile_scores[s + 3] * v_base[(s + 3) * kv_stride + d];
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + qi * q_stride + h * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Batched prefill attention over existing KV cache.
//
// Q: [n_tokens, n_heads, head_dim] for the new suffix tokens.
// K/V cache: [capacity, n_kv_heads, head_dim] containing both restored prefix
// and newly appended suffix K/V for this layer.
//
// For query index qi in [0, n_tokens):
//   total_pos = base_seq_len + qi
//   attend_end = total_pos + 1
//   attend_start = max(0, attend_end - sliding_window) when sliding_window > 0
//                = 0 otherwise
//   attend_len = attend_end - attend_start
//
// Grid: (n_tokens, n_heads) threadgroups.
kernel void attention_prefill_cache_f32(
    device const float* Q          [[buffer(0)]],
    device const float* K_cache    [[buffer(1)]],
    device const float* V_cache    [[buffer(2)]],
    device float* O                [[buffer(3)]],
    constant uint& n_tokens        [[buffer(4)]],
    constant uint& n_heads         [[buffer(5)]],
    constant uint& n_kv_heads      [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],
    constant uint& base_seq_len    [[buffer(8)]],
    constant uint& sliding_window  [[buffer(9)]], // 0 disables sliding window
    constant uint& kv_row_stride   [[buffer(10)]],
    uint2 tg_id                    [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    uint qi = tg_id.x;
    uint h = tg_id.y;
    if (qi >= n_tokens || h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride = n_heads * head_dim;
    uint kv_stride = kv_row_stride;
    float scale = rsqrt(float(head_dim));

    uint attend_end = base_seq_len + qi + 1;
    uint attend_start = 0;
    if (sliding_window > 0 && attend_end > sliding_window) {
        attend_start = attend_end - sliding_window;
    }
    uint attend_len = attend_end - attend_start;

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const float* k_ptr = K_cache + t * kv_stride + kv_h * head_dim;
            float s = 0.0f;
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(k_ptr[d], k_ptr[d + 1], k_ptr[d + 2], k_ptr[d + 3]);
                s += dot(qv, kv);
            }
            for (; d < head_dim; d++) {
                s += q_shared[d] * k_ptr[d];
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float* v_base =
            V_cache + (attend_start + tile_start) * kv_stride + kv_h * head_dim;
        for (uint d = lid; d < head_dim; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
                acc += tile_scores[s + 1] * v_base[(s + 1) * kv_stride + d];
                acc += tile_scores[s + 2] * v_base[(s + 2) * kv_stride + d];
                acc += tile_scores[s + 3] * v_base[(s + 3) * kv_stride + d];
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Same as attention_prefill_cache_f32 but reads K/V cache as half.
kernel void attention_prefill_cache_f16kv(
    device const float* Q          [[buffer(0)]],
    device const half* K_cache     [[buffer(1)]],
    device const half* V_cache     [[buffer(2)]],
    device float* O                [[buffer(3)]],
    constant uint& n_tokens        [[buffer(4)]],
    constant uint& n_heads         [[buffer(5)]],
    constant uint& n_kv_heads      [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],
    constant uint& base_seq_len    [[buffer(8)]],
    constant uint& sliding_window  [[buffer(9)]],
    constant uint& kv_row_stride   [[buffer(10)]],
    uint2 tg_id                    [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    uint qi = tg_id.x;
    uint h = tg_id.y;
    if (qi >= n_tokens || h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride = n_heads * head_dim;
    uint kv_stride = kv_row_stride;
    float scale = rsqrt(float(head_dim));

    uint attend_end = base_seq_len + qi + 1;
    uint attend_start = 0;
    if (sliding_window > 0 && attend_end > sliding_window) {
        attend_start = attend_end - sliding_window;
    }
    uint attend_len = attend_end - attend_start;

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const half* k_ptr = K_cache + t * kv_stride + kv_h * head_dim;
            float s = 0.0f;
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(*(device const half4*)(k_ptr + d));
                s += dot(qv, kv);
            }
            for (; d < head_dim; d++) {
                s += q_shared[d] * float(k_ptr[d]);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half* v_base =
            V_cache + (attend_start + tile_start) * kv_stride + kv_h * head_dim;
        for (uint d = lid; d < head_dim; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s] * float(v_base[s * kv_stride + d]);
                acc += tile_scores[s + 1] * float(v_base[(s + 1) * kv_stride + d]);
                acc += tile_scores[s + 2] * float(v_base[(s + 2) * kv_stride + d]);
                acc += tile_scores[s + 3] * float(v_base[(s + 3) * kv_stride + d]);
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * float(v_base[s * kv_stride + d]);
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// ── Decode attention ──────────────────────────────────────────────────
//
// Single-token decode attention: Q is a single token, K/V are from the
// GPU KV cache (all previous tokens). Uses the same online softmax
// approach as prefill but optimized for the single-query case.
//
// Memory layout:
//   Q: [n_heads * head_dim]         — single query token
//   K_cache: [capacity * kv_stride] — K cache, only attend_len tokens used
//   V_cache: [capacity * kv_stride] — V cache
//   O: [n_heads * head_dim]         — output for single query token
//
// Grid: n_heads threadgroups × ATTN_TG threads per threadgroup.
// One threadgroup per query head.
//
// Supports GQA and sliding window (via attend_start / attend_len).

kernel void attention_decode_f32(
    device const float* Q         [[buffer(0)]],
    device const float* K_cache   [[buffer(1)]],
    device const float* V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],   // first token to attend to
    constant uint& attend_len     [[buffer(8)]],   // number of tokens to attend
    constant uint& kv_row_stride  [[buffer(9)]],
    uint head                     [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = kv_row_stride;
    float scale = rsqrt(float(head_dim));

    // ── Threadgroup memory ──────────────────────────────────────────
    threadgroup float q_shared[MAX_HD];       // query vector for this head
    threadgroup float out_acc[MAX_HD];        // output accumulator
    threadgroup float tile_scores[ATTN_TG];   // ax_exp(score) per KV token in tile
    threadgroup float simd_buf[N_SIMD];       // SIMD reduction scratch
    threadgroup float shared_max;             // running softmax max
    threadgroup float shared_sum;             // running softmax denominator

    // ── Load query vector ───────────────────────────────────────────
    device const float* q_ptr = Q + head * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Tile loop over KV tokens ────────────────────────────────────
    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        // Phase 1: Each thread computes one Q·K score
        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const float* k_ptr = K_cache + t * kv_stride + kv_h * head_dim;
            float s = 0.0f;
            // Vectorize dot product in 4-float chunks (major attention hotspot).
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(k_ptr[d], k_ptr[d + 1], k_ptr[d + 2], k_ptr[d + 3]);
                s += dot(qv, kv);
            }
            for (; d < head_dim; d++) {
                s += q_shared[d] * k_ptr[d];
            }
            my_score = s * scale;
        }

        // Phase 2: Tile-wide max (two-level SIMD reduction)
        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        // Phase 3: Online softmax update
        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        // Reduce exp sum
        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: Rescale and accumulate weighted V
        device const float* v_base =
            V_cache + (attend_start + tile_start) * kv_stride + kv_h * head_dim;
        for (uint d = lid; d < head_dim; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s    ] * v_base[s       * kv_stride + d];
                acc += tile_scores[s + 1] * v_base[(s + 1) * kv_stride + d];
                acc += tile_scores[s + 2] * v_base[(s + 2) * kv_stride + d];
                acc += tile_scores[s + 3] * v_base[(s + 3) * kv_stride + d];
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Finalize: normalize by softmax denominator ──────────────────
    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + head * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Decode attention variant with f16 KV cache (K/V read as half, compute in float).
kernel void attention_decode_f16kv(
    device const float* Q         [[buffer(0)]],
    device const half* K_cache    [[buffer(1)]],
    device const half* V_cache    [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    constant uint& kv_row_stride  [[buffer(9)]],
    uint head                     [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = kv_row_stride;
    float scale = rsqrt(float(head_dim));

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + head * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const half* k_ptr = K_cache + t * kv_stride + kv_h * head_dim;
            float s = 0.0f;
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(*(device const half4*)(k_ptr + d));
                s += dot(qv, kv);
            }
            for (; d < head_dim; d++) {
                s += q_shared[d] * float(k_ptr[d]);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half* v_base =
            V_cache + (attend_start + tile_start) * kv_stride + kv_h * head_dim;
        for (uint d = lid; d < head_dim; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s    ] * float(v_base[s       * kv_stride + d]);
                acc += tile_scores[s + 1] * float(v_base[(s + 1) * kv_stride + d]);
                acc += tile_scores[s + 2] * float(v_base[(s + 2) * kv_stride + d]);
                acc += tile_scores[s + 3] * float(v_base[(s + 3) * kv_stride + d]);
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * float(v_base[s * kv_stride + d]);
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + head * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Decode attention specialized for head_dim == 256 (f32 KV).
kernel void attention_decode_f32_hd256(
    device const float* Q         [[buffer(0)]],
    device const float* K_cache   [[buffer(1)]],
    device const float* V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head                     [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads || head_dim != 256) return;

    constexpr uint HD = 256;
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = n_kv_heads * HD;
    constexpr float scale = 1.0f / 16.0f;  // rsqrt(256)

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + head * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const float* k_ptr = K_cache + t * kv_stride + kv_h * HD;
            float s = 0.0f;
            for (uint d = 0; d < HD; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(k_ptr[d], k_ptr[d + 1], k_ptr[d + 2], k_ptr[d + 3]);
                s += dot(qv, kv);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float* v_base =
            V_cache + (attend_start + tile_start) * kv_stride + kv_h * HD;
        for (uint d = lid; d < HD; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
                acc += tile_scores[s + 1] * v_base[(s + 1) * kv_stride + d];
                acc += tile_scores[s + 2] * v_base[(s + 2) * kv_stride + d];
                acc += tile_scores[s + 3] * v_base[(s + 3) * kv_stride + d];
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + head * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Decode attention specialized for head_dim == 256 (f16 KV).
kernel void attention_decode_f16kv_hd256(
    device const float* Q         [[buffer(0)]],
    device const half* K_cache    [[buffer(1)]],
    device const half* V_cache    [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head                     [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads || head_dim != 256) return;

    constexpr uint HD = 256;
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = n_kv_heads * HD;
    constexpr float scale = 1.0f / 16.0f;

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + head * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const half* k_ptr = K_cache + t * kv_stride + kv_h * HD;
            float s = 0.0f;
            for (uint d = 0; d < HD; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(*(device const half4*)(k_ptr + d));
                s += dot(qv, kv);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half* v_base =
            V_cache + (attend_start + tile_start) * kv_stride + kv_h * HD;
        for (uint d = lid; d < HD; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s] * float(v_base[s * kv_stride + d]);
                acc += tile_scores[s + 1] * float(v_base[(s + 1) * kv_stride + d]);
                acc += tile_scores[s + 2] * float(v_base[(s + 2) * kv_stride + d]);
                acc += tile_scores[s + 3] * float(v_base[(s + 3) * kv_stride + d]);
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * float(v_base[s * kv_stride + d]);
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + head * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Decode attention specialized for head_dim == 128 (f32 KV).
// TG = ATTN_DEC2_TG = 128 threads — one thread per head_dim element.
// Constexpr HD=128 lets the compiler unroll the 32-iteration dot-product loop.
kernel void attention_decode_f32_hd128(
    device const float* Q         [[buffer(0)]],
    device const float* K_cache   [[buffer(1)]],
    device const float* V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head                     [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads || head_dim != 128) return;

    constexpr uint HD   = 128;
    constexpr uint TG   = ATTN_DEC2_TG;   // 128
    constexpr uint NS   = N_DEC2_SIMD;    // 4
    constexpr float scale = 0.088388f;    // rsqrt(128)

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = n_kv_heads * HD;

    threadgroup float q_shared[HD];
    threadgroup float out_acc[HD];
    threadgroup float tile_scores[TG];
    threadgroup float simd_buf[NS];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    // TG == HD: each thread owns exactly one head_dim element.
    q_shared[lid] = Q[head * HD + lid];
    out_acc[lid]  = 0.0f;
    if (lid == 0) { shared_max = -INFINITY; shared_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += TG) {
        uint tile_len = min(TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const float* k_ptr = K_cache + t * kv_stride + kv_h * HD;
            float s = 0.0f;
            for (uint d = 0; d < HD; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d+1], q_shared[d+2], q_shared[d+3]);
                float4 kv = float4(k_ptr[d], k_ptr[d+1], k_ptr[d+2], k_ptr[d+3]);
                s += dot(qv, kv);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < NS; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max  = shared_max;
        float new_max   = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < NS; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V accumulation: each thread accumulates one output dimension.
        device const float* v_base = V_cache + (attend_start + tile_start) * kv_stride + kv_h * HD;
        float acc = out_acc[lid] * correction;
        uint s = 0;
        for (; s + 3 < tile_len; s += 4) {
            acc += tile_scores[s]   * v_base[s * kv_stride + lid];
            acc += tile_scores[s+1] * v_base[(s+1) * kv_stride + lid];
            acc += tile_scores[s+2] * v_base[(s+2) * kv_stride + lid];
            acc += tile_scores[s+3] * v_base[(s+3) * kv_stride + lid];
        }
        for (; s < tile_len; s++) acc += tile_scores[s] * v_base[s * kv_stride + lid];
        out_acc[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    O[head * HD + lid] = out_acc[lid] * inv_sum;
}

// Decode attention specialized for head_dim == 128 (f16 KV).
// TG = ATTN_DEC2_TG = 128 threads — one thread per head_dim element.
kernel void attention_decode_f16kv_hd128(
    device const float* Q         [[buffer(0)]],
    device const half*  K_cache   [[buffer(1)]],
    device const half*  V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head                     [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads || head_dim != 128) return;

    constexpr uint HD   = 128;
    constexpr uint TG   = ATTN_DEC2_TG;   // 128
    constexpr uint NS   = N_DEC2_SIMD;    // 4
    constexpr float scale = 0.088388f;    // rsqrt(128)

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = n_kv_heads * HD;

    threadgroup float q_shared[HD];
    threadgroup float out_acc[HD];
    threadgroup float tile_scores[TG];
    threadgroup float simd_buf[NS];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    q_shared[lid] = Q[head * HD + lid];
    out_acc[lid]  = 0.0f;
    if (lid == 0) { shared_max = -INFINITY; shared_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += TG) {
        uint tile_len = min(TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const half* k_ptr = K_cache + t * kv_stride + kv_h * HD;
            float s = 0.0f;
            for (uint d = 0; d < HD; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d+1], q_shared[d+2], q_shared[d+3]);
                float4 kv = float4(*(device const half4*)(k_ptr + d));
                s += dot(qv, kv);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < NS; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max   = shared_max;
        float new_max    = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < NS; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half* v_base = V_cache + (attend_start + tile_start) * kv_stride + kv_h * HD;
        float acc = out_acc[lid] * correction;
        uint s = 0;
        for (; s + 3 < tile_len; s += 4) {
            acc += tile_scores[s]   * float(v_base[s * kv_stride + lid]);
            acc += tile_scores[s+1] * float(v_base[(s+1) * kv_stride + lid]);
            acc += tile_scores[s+2] * float(v_base[(s+2) * kv_stride + lid]);
            acc += tile_scores[s+3] * float(v_base[(s+3) * kv_stride + lid]);
        }
        for (; s < tile_len; s++) acc += tile_scores[s] * float(v_base[s * kv_stride + lid]);
        out_acc[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    O[head * HD + lid] = out_acc[lid] * inv_sum;
}

// Decode attention specialized for head_dim == 128 (f16 KV), 2 heads per TG.
// TG = 256 threads split into two 128-thread partitions.
// Each partition computes one head using the same algorithm as hd128 kernel.
kernel void attention_decode_f16kv_hd128_n2(
    device const float* Q         [[buffer(0)]],
    device const half*  K_cache   [[buffer(1)]],
    device const half*  V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint tg_id                    [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head_dim != 128) return;

    constexpr uint HD = 128;
    constexpr uint PART_TG = 128;
    constexpr uint NS_PART = 4;  // 128 threads = 4 simdgroups
    constexpr float scale = 0.088388f;  // rsqrt(128)

    uint head0 = tg_id * 2;
    if (head0 >= n_heads) return;
    uint head1 = head0 + 1;
    bool valid1 = head1 < n_heads;

    uint part = lid / PART_TG;     // 0 or 1
    uint lane = lid & (PART_TG - 1); // 0..127
    bool active = (part == 0) || valid1;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h0 = head0 / heads_per_kv;
    uint kv_h1 = valid1 ? (head1 / heads_per_kv) : kv_h0;
    uint kv_stride = n_kv_heads * HD;

    threadgroup float q0[HD];
    threadgroup float q1[HD];
    threadgroup float out0[HD];
    threadgroup float out1[HD];
    threadgroup float tile_scores0[PART_TG];
    threadgroup float tile_scores1[PART_TG];
    threadgroup float simd_buf[8];      // 2 partitions × 4 simdgroups
    threadgroup float shared_max0;
    threadgroup float shared_max1;
    threadgroup float shared_sum0;
    threadgroup float shared_sum1;

    if (part == 0) {
        q0[lane] = Q[head0 * HD + lane];
        out0[lane] = 0.0f;
    } else {
        q1[lane] = valid1 ? Q[head1 * HD + lane] : 0.0f;
        out1[lane] = 0.0f;
    }
    if (lid == 0) {
        shared_max0 = -INFINITY;
        shared_sum0 = 0.0f;
        shared_max1 = -INFINITY;
        shared_sum1 = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += PART_TG) {
        uint tile_len = min(PART_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (active && lane < tile_len) {
            uint t = attend_start + tile_start + lane;
            uint kv_h = (part == 0) ? kv_h0 : kv_h1;
            device const half* k_ptr = K_cache + t * kv_stride + kv_h * HD;
            float s = 0.0f;
            if (part == 0) {
                for (uint d = 0; d < HD; d += 4) {
                    float4 qv = float4(q0[d], q0[d + 1], q0[d + 2], q0[d + 3]);
                    float4 kv = float4(*(device const half4*)(k_ptr + d));
                    s += dot(qv, kv);
                }
            } else {
                for (uint d = 0; d < HD; d += 4) {
                    float4 qv = float4(q1[d], q1[d + 1], q1[d + 2], q1[d + 3]);
                    float4 kv = float4(*(device const half4*)(k_ptr + d));
                    s += dot(qv, kv);
                }
            }
            my_score = s * scale;
        }

        float v = (active && lane < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint simd_base = part * NS_PART;
        if (lane == 0) {
            float m = simd_buf[simd_base];
            for (uint i = 1; i < NS_PART; i++) {
                m = max(m, simd_buf[simd_base + i]);
            }
            simd_buf[simd_base] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[simd_base];

        float prev_max = (part == 0) ? shared_max0 : shared_max1;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (active && lane < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        if (part == 0) {
            tile_scores0[lane] = exp_s;
        } else {
            tile_scores1[lane] = exp_s;
        }

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < NS_PART; i++) {
                tile_sum += simd_buf[simd_base + i];
            }
            if (part == 0) {
                shared_max0 = new_max;
                shared_sum0 = shared_sum0 * correction + tile_sum;
            } else {
                shared_max1 = new_max;
                shared_sum1 = shared_sum1 * correction + tile_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint kv_h = (part == 0) ? kv_h0 : kv_h1;
        device const half* v_base = V_cache + (attend_start + tile_start) * kv_stride + kv_h * HD;
        float acc = ((part == 0) ? out0[lane] : out1[lane]) * correction;
        uint s = 0;
        if (part == 0) {
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores0[s] * float(v_base[s * kv_stride + lane]);
                acc += tile_scores0[s + 1] * float(v_base[(s + 1) * kv_stride + lane]);
                acc += tile_scores0[s + 2] * float(v_base[(s + 2) * kv_stride + lane]);
                acc += tile_scores0[s + 3] * float(v_base[(s + 3) * kv_stride + lane]);
            }
            for (; s < tile_len; s++) {
                acc += tile_scores0[s] * float(v_base[s * kv_stride + lane]);
            }
            out0[lane] = acc;
        } else {
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores1[s] * float(v_base[s * kv_stride + lane]);
                acc += tile_scores1[s + 1] * float(v_base[(s + 1) * kv_stride + lane]);
                acc += tile_scores1[s + 2] * float(v_base[(s + 2) * kv_stride + lane]);
                acc += tile_scores1[s + 3] * float(v_base[(s + 3) * kv_stride + lane]);
            }
            for (; s < tile_len; s++) {
                acc += tile_scores1[s] * float(v_base[s * kv_stride + lane]);
            }
            out1[lane] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (part == 0) {
        float inv_sum0 = (shared_sum0 > 0.0f) ? (1.0f / shared_sum0) : 0.0f;
        O[head0 * HD + lane] = out0[lane] * inv_sum0;
    } else if (valid1) {
        float inv_sum1 = (shared_sum1 > 0.0f) ? (1.0f / shared_sum1) : 0.0f;
        O[head1 * HD + lane] = out1[lane] * inv_sum1;
    }
}

// Split-K decode attention specialized for head_dim == 128 (f16 KV).
// Grid: (n_chunks, n_heads), TG=128. Each threadgroup computes one
// (head, chunk) partial attention result over a contiguous KV slice.
kernel void attention_decode_splitk_f16kv_hd128_partial(
    device const float* Q         [[buffer(0)]],
    device const half*  K_cache   [[buffer(1)]],
    device const half*  V_cache   [[buffer(2)]],
    device float* partial_out     [[buffer(3)]],
    device float* partial_lse     [[buffer(4)]],
    constant uint& n_heads        [[buffer(5)]],
    constant uint& n_kv_heads     [[buffer(6)]],
    constant uint& head_dim       [[buffer(7)]],
    constant uint& attend_start   [[buffer(8)]],
    constant uint& attend_len     [[buffer(9)]],
    constant uint& chunk_size     [[buffer(10)]],
    constant uint& n_chunks       [[buffer(11)]],
    uint2 tg_id                   [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head_dim != 128) return;

    uint chunk = tg_id.x;
    uint head = tg_id.y;
    if (chunk >= n_chunks || head >= n_heads) return;

    constexpr uint HD = 128;
    constexpr uint TG = ATTN_DEC2_TG;
    constexpr uint NS = N_DEC2_SIMD;
    constexpr float scale = 0.088388f;

    uint chunk_start = attend_start + chunk * chunk_size;
    uint attend_end = attend_start + attend_len;
    if (chunk_start >= attend_end) return;
    uint chunk_end = min(chunk_start + chunk_size, attend_end);
    uint chunk_len = chunk_end - chunk_start;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = n_kv_heads * HD;

    threadgroup float q_shared[HD];
    threadgroup float out_acc[HD];
    threadgroup float tile_scores[TG];
    threadgroup float simd_buf[NS];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    q_shared[lid] = Q[head * HD + lid];
    out_acc[lid] = 0.0f;
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < chunk_len; tile_start += TG) {
        uint tile_len = min(TG, chunk_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = chunk_start + tile_start + lid;
            device const half* k_ptr = K_cache + t * kv_stride + kv_h * HD;
            float s = 0.0f;
            for (uint d = 0; d < HD; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(*(device const half4*)(k_ptr + d));
                s += dot(qv, kv);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < NS; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < NS; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half* v_base = V_cache + (chunk_start + tile_start) * kv_stride + kv_h * HD;
        float acc = out_acc[lid] * correction;
        uint s = 0;
        for (; s + 3 < tile_len; s += 4) {
            acc += tile_scores[s] * float(v_base[s * kv_stride + lid]);
            acc += tile_scores[s + 1] * float(v_base[(s + 1) * kv_stride + lid]);
            acc += tile_scores[s + 2] * float(v_base[(s + 2) * kv_stride + lid]);
            acc += tile_scores[s + 3] * float(v_base[(s + 3) * kv_stride + lid]);
        }
        for (; s < tile_len; s++) acc += tile_scores[s] * float(v_base[s * kv_stride + lid]);
        out_acc[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    uint base = (head * n_chunks + chunk) * HD;
    partial_out[base + lid] = out_acc[lid] * inv_sum;
    if (lid == 0) {
        partial_lse[head * n_chunks + chunk] =
            (shared_sum > 0.0f) ? (shared_max + log(shared_sum)) : -INFINITY;
    }
}

kernel void attention_decode_splitk_f16kv_hd128_reduce(
    device const float* partial_out [[buffer(0)]],
    device const float* partial_lse [[buffer(1)]],
    device float* O                 [[buffer(2)]],
    constant uint& n_heads          [[buffer(3)]],
    constant uint& head_dim         [[buffer(4)]],
    constant uint& n_chunks         [[buffer(5)]],
    uint head                       [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    if (head >= n_heads || head_dim != 128) return;

    constexpr uint HD = 128;
    threadgroup float shared_max;
    threadgroup float shared_denom;

    if (lid == 0) {
        float max_lse = -INFINITY;
        for (uint c = 0; c < n_chunks; c++) {
            max_lse = max(max_lse, partial_lse[head * n_chunks + c]);
        }
        float denom = 0.0f;
        for (uint c = 0; c < n_chunks; c++) {
            denom += ax_exp(partial_lse[head * n_chunks + c] - max_lse);
        }
        shared_max = max_lse;
        shared_denom = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint c = 0; c < n_chunks; c++) {
        float w = ax_exp(partial_lse[head * n_chunks + c] - shared_max);
        acc += w * partial_out[(head * n_chunks + c) * HD + lid];
    }
    O[head * HD + lid] = (shared_denom > 0.0f) ? (acc / shared_denom) : 0.0f;
}

// Split-K decode attention specialized for head_dim == 256 (f16 KV).
// Grid: (n_chunks, n_heads), TG=256.
kernel void attention_decode_splitk_f16kv_hd256_partial(
    device const float* Q         [[buffer(0)]],
    device const half* K_cache    [[buffer(1)]],
    device const half* V_cache    [[buffer(2)]],
    device float* partial_out     [[buffer(3)]],
    device float* partial_lse     [[buffer(4)]],
    constant uint& n_heads        [[buffer(5)]],
    constant uint& n_kv_heads     [[buffer(6)]],
    constant uint& head_dim       [[buffer(7)]],
    constant uint& attend_start   [[buffer(8)]],
    constant uint& attend_len     [[buffer(9)]],
    constant uint& chunk_size     [[buffer(10)]],
    constant uint& n_chunks       [[buffer(11)]],
    uint2 tg_id                   [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head_dim != 256) return;

    uint chunk = tg_id.x;
    uint head = tg_id.y;
    if (chunk >= n_chunks || head >= n_heads) return;

    constexpr uint HD = 256;
    constexpr float scale = 1.0f / 16.0f;

    uint chunk_start = attend_start + chunk * chunk_size;
    uint attend_end = attend_start + attend_len;
    if (chunk_start >= attend_end) return;
    uint chunk_end = min(chunk_start + chunk_size, attend_end);
    uint chunk_len = chunk_end - chunk_start;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = n_kv_heads * HD;

    threadgroup float q_shared[HD];
    threadgroup float out_acc[HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    q_shared[lid] = Q[head * HD + lid];
    out_acc[lid] = 0.0f;
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < chunk_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, chunk_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = chunk_start + tile_start + lid;
            device const half* k_ptr = K_cache + t * kv_stride + kv_h * HD;
            float s = 0.0f;
            for (uint d = 0; d < HD; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(*(device const half4*)(k_ptr + d));
                s += dot(qv, kv);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half* v_base = V_cache + (chunk_start + tile_start) * kv_stride + kv_h * HD;
        float acc = out_acc[lid] * correction;
        uint s = 0;
        for (; s + 3 < tile_len; s += 4) {
            acc += tile_scores[s] * float(v_base[s * kv_stride + lid]);
            acc += tile_scores[s + 1] * float(v_base[(s + 1) * kv_stride + lid]);
            acc += tile_scores[s + 2] * float(v_base[(s + 2) * kv_stride + lid]);
            acc += tile_scores[s + 3] * float(v_base[(s + 3) * kv_stride + lid]);
        }
        for (; s < tile_len; s++) acc += tile_scores[s] * float(v_base[s * kv_stride + lid]);
        out_acc[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    uint base = (head * n_chunks + chunk) * HD;
    partial_out[base + lid] = out_acc[lid] * inv_sum;
    if (lid == 0) {
        partial_lse[head * n_chunks + chunk] =
            (shared_sum > 0.0f) ? (shared_max + log(shared_sum)) : -INFINITY;
    }
}

kernel void attention_decode_splitk_f16kv_hd256_reduce(
    device const float* partial_out [[buffer(0)]],
    device const float* partial_lse [[buffer(1)]],
    device float* O                 [[buffer(2)]],
    constant uint& n_heads          [[buffer(3)]],
    constant uint& head_dim         [[buffer(4)]],
    constant uint& n_chunks         [[buffer(5)]],
    uint head                       [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    if (head >= n_heads || head_dim != 256) return;

    constexpr uint HD = 256;
    threadgroup float shared_max;
    threadgroup float shared_denom;

    if (lid == 0) {
        float max_lse = -INFINITY;
        for (uint c = 0; c < n_chunks; c++) {
            max_lse = max(max_lse, partial_lse[head * n_chunks + c]);
        }
        float denom = 0.0f;
        for (uint c = 0; c < n_chunks; c++) {
            denom += ax_exp(partial_lse[head * n_chunks + c] - max_lse);
        }
        shared_max = max_lse;
        shared_denom = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint c = 0; c < n_chunks; c++) {
        float w = ax_exp(partial_lse[head * n_chunks + c] - shared_max);
        acc += w * partial_out[(head * n_chunks + c) * HD + lid];
    }
    O[head * HD + lid] = (shared_denom > 0.0f) ? (acc / shared_denom) : 0.0f;
}

// Alternative decode variant with smaller threadgroup size.
kernel void attention_decode_f32_v2(
    device const float* Q         [[buffer(0)]],
    device const float* K_cache   [[buffer(1)]],
    device const float* V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head                     [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_DEC2_TG];
    threadgroup float simd_buf[N_DEC2_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + head * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_DEC2_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_DEC2_TG) {
        uint tile_len = min(ATTN_DEC2_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const float* k_ptr = K_cache + t * kv_stride + kv_h * head_dim;
            float s = 0.0f;
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(k_ptr[d], k_ptr[d + 1], k_ptr[d + 2], k_ptr[d + 3]);
                s += dot(qv, kv);
            }
            for (; d < head_dim; d++) {
                s += q_shared[d] * k_ptr[d];
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_DEC2_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_DEC2_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float* v_base =
            V_cache + (attend_start + tile_start) * kv_stride + kv_h * head_dim;
        for (uint d = lid; d < head_dim; d += ATTN_DEC2_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s    ] * v_base[s       * kv_stride + d];
                acc += tile_scores[s + 1] * v_base[(s + 1) * kv_stride + d];
                acc += tile_scores[s + 2] * v_base[(s + 2) * kv_stride + d];
                acc += tile_scores[s + 3] * v_base[(s + 3) * kv_stride + d];
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + head * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_DEC2_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Alternative f16-KV decode variant with smaller threadgroup size.
kernel void attention_decode_f16kv_v2(
    device const float* Q         [[buffer(0)]],
    device const half* K_cache    [[buffer(1)]],
    device const half* V_cache    [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head                     [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head / heads_per_kv;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_DEC2_TG];
    threadgroup float simd_buf[N_DEC2_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + head * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_DEC2_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_DEC2_TG) {
        uint tile_len = min(ATTN_DEC2_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            device const half* k_ptr = K_cache + t * kv_stride + kv_h * head_dim;
            float s = 0.0f;
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(*(device const half4*)(k_ptr + d));
                s += dot(qv, kv);
            }
            for (; d < head_dim; d++) {
                s += q_shared[d] * float(k_ptr[d]);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_DEC2_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_DEC2_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half* v_base =
            V_cache + (attend_start + tile_start) * kv_stride + kv_h * head_dim;
        for (uint d = lid; d < head_dim; d += ATTN_DEC2_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s    ] * float(v_base[s       * kv_stride + d]);
                acc += tile_scores[s + 1] * float(v_base[(s + 1) * kv_stride + d]);
                acc += tile_scores[s + 2] * float(v_base[(s + 2) * kv_stride + d]);
                acc += tile_scores[s + 3] * float(v_base[(s + 3) * kv_stride + d]);
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * float(v_base[s * kv_stride + d]);
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + head * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_DEC2_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Alternative prefill variant tuned for higher occupancy on medium sequence
// lengths by reducing threadgroup size.
constant uint ATTN2_TG = 128;
constant uint N2_SIMD  = ATTN2_TG / 32;

kernel void attention_prefill_f32_v2(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device float* O            [[buffer(3)]],
    constant uint& n_tokens    [[buffer(4)]],
    constant uint& n_heads     [[buffer(5)]],
    constant uint& n_kv_heads  [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 tg_id                [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint qi = tg_id.x;
    uint h  = tg_id.y;
    if (qi >= n_tokens || h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride  = n_heads * head_dim;
    uint kv_stride = n_kv_heads * head_dim;
    float scale = rsqrt(float(head_dim));
    uint attend_len = qi + 1;

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN2_TG];
    threadgroup float simd_buf[N2_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN2_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN2_TG) {
        uint tile_len = min(ATTN2_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = tile_start + lid;
            device const float* k_ptr = K + t * kv_stride + kv_h * head_dim;
            float s = 0.0f;
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(k_ptr[d], k_ptr[d + 1], k_ptr[d + 2], k_ptr[d + 3]);
                s += dot(qv, kv);
            }
            for (; d < head_dim; d++) {
                s += q_shared[d] * k_ptr[d];
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N2_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N2_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float* v_base =
            V + tile_start * kv_stride + kv_h * head_dim;
        for (uint d = lid; d < head_dim; d += ATTN2_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s    ] * v_base[s       * kv_stride + d];
                acc += tile_scores[s + 1] * v_base[(s + 1) * kv_stride + d];
                acc += tile_scores[s + 2] * v_base[(s + 2) * kv_stride + d];
                acc += tile_scores[s + 3] * v_base[(s + 3) * kv_stride + d];
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN2_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// Specialized v2 prefill for head_dim == 128 (Llama/Qwen class models).
// Keeps ATTN2_TG scheduling while removing dynamic head_dim loops.
kernel void attention_prefill_f32_v2_hd128(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device float* O            [[buffer(3)]],
    constant uint& n_tokens    [[buffer(4)]],
    constant uint& n_heads     [[buffer(5)]],
    constant uint& n_kv_heads  [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 tg_id                [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    uint qi = tg_id.x;
    uint h  = tg_id.y;
    if (qi >= n_tokens || h >= n_heads || head_dim != 128) return;

    constexpr uint HD = 128;
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride  = n_heads * HD;
    uint kv_stride = n_kv_heads * HD;
    constexpr float scale = 0.08838834764831843f;  // rsqrt(128)
    uint attend_len = qi + 1;

    threadgroup float q_shared[HD];
    threadgroup float out_acc[HD];
    threadgroup float tile_scores[ATTN2_TG];
    threadgroup float simd_buf[N2_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + qi * q_stride + h * HD;
    for (uint d = lid; d < HD; d += ATTN2_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN2_TG) {
        uint tile_len = min(ATTN2_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = tile_start + lid;
            device const float* k_ptr = K + t * kv_stride + kv_h * HD;
            float s = 0.0f;
            for (uint d = 0; d < HD; d += 4) {
                float4 qv = float4(q_shared[d], q_shared[d + 1], q_shared[d + 2], q_shared[d + 3]);
                float4 kv = float4(k_ptr[d], k_ptr[d + 1], k_ptr[d + 2], k_ptr[d + 3]);
                s += dot(qv, kv);
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N2_SIMD; i++) {
                m = max(m, simd_buf[i]);
            }
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = 0.0f;
        if (lid < tile_len) {
            exp_s = ax_exp(my_score - new_max);
        }
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N2_SIMD; i++) {
                tile_sum += simd_buf[i];
            }
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float* v_base = V + tile_start * kv_stride + kv_h * HD;
        for (uint d = lid; d < HD; d += ATTN2_TG) {
            float acc = out_acc[d] * correction;
            uint s = 0;
            for (; s + 3 < tile_len; s += 4) {
                acc += tile_scores[s    ] * v_base[s       * kv_stride + d];
                acc += tile_scores[s + 1] * v_base[(s + 1) * kv_stride + d];
                acc += tile_scores[s + 2] * v_base[(s + 2) * kv_stride + d];
                acc += tile_scores[s + 3] * v_base[(s + 3) * kv_stride + d];
            }
            for (; s < tile_len; s++) {
                acc += tile_scores[s] * v_base[s * kv_stride + d];
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + qi * q_stride + h * HD;
    for (uint d = lid; d < HD; d += ATTN2_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// AX prefill specialization for head_dim == 128:
// - BR=8 queries per threadgroup (one simdgroup per query)
// - BC=32 keys per tile (one lane per key)
// - TG=256 threads (8 simdgroups)
//
// This increases query-level parallelism vs one-query-per-threadgroup kernels.
kernel void attention_prefill_f32_ax_hd128(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device float* O            [[buffer(3)]],
    constant uint& n_tokens    [[buffer(4)]],
    constant uint& n_heads     [[buffer(5)]],
    constant uint& n_kv_heads  [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 tg_id                [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint HD = 128;
    constexpr uint BR = 8;
    constexpr uint BC = 32;
    constexpr float scale = 0.08838834764831843f;  // rsqrt(128)

    if (head_dim != HD) return;

    uint h = tg_id.y;
    if (h >= n_heads) return;

    uint qi = tg_id.x * BR + simd_id;
    if (simd_id >= BR || qi >= n_tokens) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride = n_heads * HD;
    uint kv_stride = n_kv_heads * HD;
    uint attend_len = qi + 1;  // causal

    // Each lane owns 4 contiguous elements of the head vector.
    uint d0 = simd_lane * 4;
    device const float* q_ptr = Q + qi * q_stride + h * HD;
    float4 qv = float4(
        q_ptr[d0],
        q_ptr[d0 + 1],
        q_ptr[d0 + 2],
        q_ptr[d0 + 3]
    );
    float4 outv = float4(0.0f);

    // Per-query online softmax state, replicated per lane within the simdgroup.
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint tile_start = 0; tile_start < attend_len; tile_start += BC) {
        uint tile_len = min(BC, attend_len - tile_start);

        // One score per lane (lane -> one key in the tile).
        float score = -INFINITY;
        if (simd_lane < tile_len) {
            uint t = tile_start + simd_lane;
            device const float* k_ptr = K + t * kv_stride + kv_h * HD + d0;
            float4 kv = float4(k_ptr[0], k_ptr[1], k_ptr[2], k_ptr[3]);
            float partial = dot(qv, kv);
            float dot_full = simd_sum(partial);
            score = dot_full * scale;
        }

        float tile_max = simd_max(score);
        float prev_max = running_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (simd_lane < tile_len) ? ax_exp(score - new_max) : 0.0f;
        float tile_sum = simd_sum(exp_s);
        running_max = new_max;
        running_sum = running_sum * correction + tile_sum;

        // Weighted V accumulation for this lane's 4 output dims.
        float4 acc = outv * correction;
        for (uint j = 0; j < tile_len; j++) {
            float w = simd_shuffle(exp_s, j);
            uint t = tile_start + j;
            device const float* v_ptr = V + t * kv_stride + kv_h * HD + d0;
            float4 vv = float4(v_ptr[0], v_ptr[1], v_ptr[2], v_ptr[3]);
            acc += w * vv;
        }
        outv = acc;
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    outv *= inv_sum;

    device float* o_ptr = O + qi * q_stride + h * HD + d0;
    o_ptr[0] = outv[0];
    o_ptr[1] = outv[1];
    o_ptr[2] = outv[2];
    o_ptr[3] = outv[3];
}

// Same as attention_prefill_f32_ax_hd128, but writes output in f16.
kernel void attention_prefill_f16_ax_hd128(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device half* O             [[buffer(3)]],
    constant uint& n_tokens    [[buffer(4)]],
    constant uint& n_heads     [[buffer(5)]],
    constant uint& n_kv_heads  [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 tg_id                [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint HD = 128;
    constexpr uint BR = 8;
    constexpr uint BC = 32;
    constexpr float scale = 0.08838834764831843f;  // rsqrt(128)

    if (head_dim != HD) return;
    uint h = tg_id.y;
    if (h >= n_heads) return;

    uint qi = tg_id.x * BR + simd_id;
    if (simd_id >= BR || qi >= n_tokens) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride = n_heads * HD;
    uint kv_stride = n_kv_heads * HD;
    uint attend_len = qi + 1;

    uint d0 = simd_lane * 4;
    device const float* q_ptr = Q + qi * q_stride + h * HD;
    float4 qv = float4(q_ptr[d0], q_ptr[d0 + 1], q_ptr[d0 + 2], q_ptr[d0 + 3]);
    float4 outv = float4(0.0f);

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint tile_start = 0; tile_start < attend_len; tile_start += BC) {
        uint tile_len = min(BC, attend_len - tile_start);

        float score = -INFINITY;
        if (simd_lane < tile_len) {
            uint t = tile_start + simd_lane;
            device const float* k_ptr = K + t * kv_stride + kv_h * HD + d0;
            float4 kv = float4(k_ptr[0], k_ptr[1], k_ptr[2], k_ptr[3]);
            score = simd_sum(dot(qv, kv)) * scale;
        }

        float tile_max = simd_max(score);
        float prev_max = running_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (simd_lane < tile_len) ? ax_exp(score - new_max) : 0.0f;
        float tile_sum = simd_sum(exp_s);
        running_max = new_max;
        running_sum = running_sum * correction + tile_sum;

        float4 acc = outv * correction;
        for (uint j = 0; j < tile_len; j++) {
            float w = simd_shuffle(exp_s, j);
            uint t = tile_start + j;
            device const float* v_ptr = V + t * kv_stride + kv_h * HD + d0;
            float4 vv = float4(v_ptr[0], v_ptr[1], v_ptr[2], v_ptr[3]);
            acc += w * vv;
        }
        outv = acc;
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    outv *= inv_sum;

    device half* o_ptr = O + qi * q_stride + h * HD + d0;
    o_ptr[0] = half(outv[0]);
    o_ptr[1] = half(outv[1]);
    o_ptr[2] = half(outv[2]);
    o_ptr[3] = half(outv[3]);
}

// AX HD128 prefill with explicit K/V tile staging in threadgroup memory.
// BC=32 keeps shared memory usage bounded (K+V = 32 * 128 * 4 * 2 = 32KB).
kernel void attention_prefill_f32_ax_hd128_smem(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device float* O            [[buffer(3)]],
    constant uint& n_tokens    [[buffer(4)]],
    constant uint& n_heads     [[buffer(5)]],
    constant uint& n_kv_heads  [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 tg_id                [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint HD = 128;
    constexpr uint BR = 8;
    constexpr uint BC = 32;
    constexpr float scale = 0.08838834764831843f;  // rsqrt(128)

    if (head_dim != HD) return;
    uint h = tg_id.y;
    if (h >= n_heads) return;

    uint qi = tg_id.x * BR + simd_id;
    if (simd_id >= BR || qi >= n_tokens) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride = n_heads * HD;
    uint kv_stride = n_kv_heads * HD;
    uint attend_len = qi + 1;

    uint d0 = simd_lane * 4;
    device const float* q_ptr = Q + qi * q_stride + h * HD;
    float4 qv = float4(q_ptr[d0], q_ptr[d0 + 1], q_ptr[d0 + 2], q_ptr[d0 + 3]);
    float4 outv = float4(0.0f);

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    threadgroup float k_tile[BC * HD];
    threadgroup float v_tile[BC * HD];

    for (uint tile_start = 0; tile_start < attend_len; tile_start += BC) {
        uint tile_len = min(BC, attend_len - tile_start);

        // Stage K/V tile once per threadgroup, reused by all BR queries in this TG.
        for (uint i = tid; i < BC * HD; i += ATTN_TG) {
            uint j = i / HD;
            uint d = i % HD;
            if (j < tile_len) {
                uint t = tile_start + j;
                device const float* k_ptr = K + t * kv_stride + kv_h * HD;
                device const float* v_ptr = V + t * kv_stride + kv_h * HD;
                k_tile[i] = k_ptr[d];
                v_tile[i] = v_ptr[d];
            } else {
                k_tile[i] = 0.0f;
                v_tile[i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = -INFINITY;
        if (simd_lane < tile_len) {
            uint k_off = simd_lane * HD + d0;
            float4 kv = float4(
                k_tile[k_off],
                k_tile[k_off + 1],
                k_tile[k_off + 2],
                k_tile[k_off + 3]
            );
            score = simd_sum(dot(qv, kv)) * scale;
        }

        float tile_max = simd_max(score);
        float prev_max = running_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (simd_lane < tile_len) ? ax_exp(score - new_max) : 0.0f;
        float tile_sum = simd_sum(exp_s);
        running_max = new_max;
        running_sum = running_sum * correction + tile_sum;

        float4 acc = outv * correction;
        for (uint j = 0; j < tile_len; j++) {
            float w = simd_shuffle(exp_s, j);
            uint v_off = j * HD + d0;
            float4 vv = float4(
                v_tile[v_off],
                v_tile[v_off + 1],
                v_tile[v_off + 2],
                v_tile[v_off + 3]
            );
            acc += w * vv;
        }
        outv = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    outv *= inv_sum;

    device float* o_ptr = O + qi * q_stride + h * HD + d0;
    o_ptr[0] = outv[0];
    o_ptr[1] = outv[1];
    o_ptr[2] = outv[2];
    o_ptr[3] = outv[3];
}

// AX HD128 prefill with f16 K/V threadgroup staging.
// Same BR/BC as smem variant, but halves shared-memory footprint.
kernel void attention_prefill_f32_ax_hd128_smem_f16(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device float* O            [[buffer(3)]],
    constant uint& n_tokens    [[buffer(4)]],
    constant uint& n_heads     [[buffer(5)]],
    constant uint& n_kv_heads  [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 tg_id                [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint HD = 128;
    constexpr uint BR = 8;
    constexpr uint BC = 32;
    constexpr float scale = 0.08838834764831843f;  // rsqrt(128)

    if (head_dim != HD) return;
    uint h = tg_id.y;
    if (h >= n_heads) return;

    uint qi = tg_id.x * BR + simd_id;
    if (simd_id >= BR || qi >= n_tokens) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride = n_heads * HD;
    uint kv_stride = n_kv_heads * HD;
    uint attend_len = qi + 1;

    uint d0 = simd_lane * 4;
    device const float* q_ptr = Q + qi * q_stride + h * HD;
    float4 qv = float4(q_ptr[d0], q_ptr[d0 + 1], q_ptr[d0 + 2], q_ptr[d0 + 3]);
    float4 outv = float4(0.0f);

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    threadgroup half k_tile[BC * HD];
    threadgroup half v_tile[BC * HD];

    for (uint tile_start = 0; tile_start < attend_len; tile_start += BC) {
        uint tile_len = min(BC, attend_len - tile_start);

        for (uint i = tid; i < BC * HD; i += ATTN_TG) {
            uint j = i / HD;
            uint d = i % HD;
            if (j < tile_len) {
                uint t = tile_start + j;
                device const float* k_ptr = K + t * kv_stride + kv_h * HD;
                device const float* v_ptr = V + t * kv_stride + kv_h * HD;
                k_tile[i] = half(k_ptr[d]);
                v_tile[i] = half(v_ptr[d]);
            } else {
                k_tile[i] = half(0.0f);
                v_tile[i] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = -INFINITY;
        if (simd_lane < tile_len) {
            uint k_off = simd_lane * HD + d0;
            float4 kv = float4(
                float(k_tile[k_off]),
                float(k_tile[k_off + 1]),
                float(k_tile[k_off + 2]),
                float(k_tile[k_off + 3])
            );
            score = simd_sum(dot(qv, kv)) * scale;
        }

        float tile_max = simd_max(score);
        float prev_max = running_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (simd_lane < tile_len) ? ax_exp(score - new_max) : 0.0f;
        float tile_sum = simd_sum(exp_s);
        running_max = new_max;
        running_sum = running_sum * correction + tile_sum;

        float4 acc = outv * correction;
        for (uint j = 0; j < tile_len; j++) {
            float w = simd_shuffle(exp_s, j);
            uint v_off = j * HD + d0;
            float4 vv = float4(
                float(v_tile[v_off]),
                float(v_tile[v_off + 1]),
                float(v_tile[v_off + 2]),
                float(v_tile[v_off + 3])
            );
            acc += w * vv;
        }
        outv = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    outv *= inv_sum;

    device float* o_ptr = O + qi * q_stride + h * HD + d0;
    o_ptr[0] = outv[0];
    o_ptr[1] = outv[1];
    o_ptr[2] = outv[2];
    o_ptr[3] = outv[3];
}

// AX prefill specialization for head_dim == 128, BC=64.
// Each lane computes two logits per tile (offsets +0 and +32) to reduce
// tile-loop overhead on long prompts.
kernel void attention_prefill_f32_ax_hd128_bc64(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device float* O            [[buffer(3)]],
    constant uint& n_tokens    [[buffer(4)]],
    constant uint& n_heads     [[buffer(5)]],
    constant uint& n_kv_heads  [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 tg_id                [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint HD = 128;
    constexpr uint BR = 8;
    constexpr uint BC = 64;
    constexpr float scale = 0.08838834764831843f;  // rsqrt(128)

    if (head_dim != HD) return;
    uint h = tg_id.y;
    if (h >= n_heads) return;

    uint qi = tg_id.x * BR + simd_id;
    if (simd_id >= BR || qi >= n_tokens) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride = n_heads * HD;
    uint kv_stride = n_kv_heads * HD;
    uint attend_len = qi + 1;

    uint d0 = simd_lane * 4;
    device const float* q_ptr = Q + qi * q_stride + h * HD;
    float4 qv = float4(q_ptr[d0], q_ptr[d0 + 1], q_ptr[d0 + 2], q_ptr[d0 + 3]);
    float4 outv = float4(0.0f);

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint tile_start = 0; tile_start < attend_len; tile_start += BC) {
        uint tile_len = min(BC, attend_len - tile_start);
        bool has0 = simd_lane < tile_len;
        bool has1 = (simd_lane + 32) < tile_len;

        float score0 = -INFINITY;
        float score1 = -INFINITY;

        if (has0) {
            uint t0 = tile_start + simd_lane;
            device const float* k0 = K + t0 * kv_stride + kv_h * HD + d0;
            float4 kv0 = float4(k0[0], k0[1], k0[2], k0[3]);
            score0 = simd_sum(dot(qv, kv0)) * scale;
        }
        if (has1) {
            uint t1 = tile_start + simd_lane + 32;
            device const float* k1 = K + t1 * kv_stride + kv_h * HD + d0;
            float4 kv1 = float4(k1[0], k1[1], k1[2], k1[3]);
            score1 = simd_sum(dot(qv, kv1)) * scale;
        }

        float tile_max = max(simd_max(score0), simd_max(score1));
        float prev_max = running_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp0 = has0 ? ax_exp(score0 - new_max) : 0.0f;
        float exp1 = has1 ? ax_exp(score1 - new_max) : 0.0f;
        float tile_sum = simd_sum(exp0 + exp1);
        running_max = new_max;
        running_sum = running_sum * correction + tile_sum;

        float4 acc = outv * correction;
        uint half_len0 = min(tile_len, 32u);
        for (uint j = 0; j < half_len0; j++) {
            float w = simd_shuffle(exp0, j);
            uint t = tile_start + j;
            device const float* v_ptr = V + t * kv_stride + kv_h * HD + d0;
            float4 vv = float4(v_ptr[0], v_ptr[1], v_ptr[2], v_ptr[3]);
            acc += w * vv;
        }
        if (tile_len > 32) {
            uint half_len1 = tile_len - 32;
            for (uint j = 0; j < half_len1; j++) {
                float w = simd_shuffle(exp1, j);
                uint t = tile_start + 32 + j;
                device const float* v_ptr = V + t * kv_stride + kv_h * HD + d0;
                float4 vv = float4(v_ptr[0], v_ptr[1], v_ptr[2], v_ptr[3]);
                acc += w * vv;
            }
        }
        outv = acc;
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    outv *= inv_sum;

    device float* o_ptr = O + qi * q_stride + h * HD + d0;
    o_ptr[0] = outv[0];
    o_ptr[1] = outv[1];
    o_ptr[2] = outv[2];
    o_ptr[3] = outv[3];
}

// ── FA2-style multi-query prefill, HD=128 (base prefill path) ─────────────
//
// Grid: (ceil(n_tokens / 8), n_heads) threadgroups.
constant uint FA2H_Q   = 8;
constant uint FA2H_KV  = 16;
constant uint FA2H_TG  = 256;
constant uint FA2H_HD  = 128;
constant uint FA2H_D_PER_THREAD = FA2H_HD / (FA2H_TG / FA2H_Q); // 4

kernel void attention_prefill_f32_fa2_hd128(
    device const float* Q         [[buffer(0)]],
    device const float* K         [[buffer(1)]],
    device const float* V         [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_tokens       [[buffer(4)]],
    constant uint& n_heads        [[buffer(5)]],
    constant uint& n_kv_heads     [[buffer(6)]],
    constant uint& head_dim       [[buffer(7)]],
    uint2 tg_id                   [[threadgroup_position_in_grid]],
    uint  lid                     [[thread_index_in_threadgroup]],
    uint  simd_lane               [[thread_index_in_simdgroup]],
    uint  simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2H_Q;
    uint h      = tg_id.y;
    if (h >= n_heads || head_dim != FA2H_HD) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2H_HD;
    uint kv_stride    = n_kv_heads * FA2H_HD;
    constexpr float scale = 0.08838834764831843f; // 1/sqrt(128)

    threadgroup half  q_tile[FA2H_Q * FA2H_HD];
    threadgroup half  kv_tile[FA2H_KV * FA2H_HD];
    threadgroup float scores[FA2H_Q * FA2H_KV];
    threadgroup float out_acc[FA2H_Q * FA2H_HD];
    threadgroup float running_max[FA2H_Q];
    threadgroup float running_sum[FA2H_Q];
    threadgroup float correction[FA2H_Q];

    if (lid < FA2H_Q) {
        running_max[lid] = -INFINITY;
        running_sum[lid] = 0.0f;
    }
    for (uint i = lid; i < FA2H_Q * FA2H_HD; i += FA2H_TG) {
        out_acc[i] = 0.0f;
    }

    for (uint i = lid; i < FA2H_Q * FA2H_HD; i += FA2H_TG) {
        uint qi = i / FA2H_HD;
        uint d  = i % FA2H_HD;
        uint gqi = tile_q + qi;
        q_tile[i] = (gqi < n_tokens) ? half(Q[gqi * q_stride + h * FA2H_HD + d]) : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint attend_range = min(tile_q + FA2H_Q, n_tokens);
    for (uint tile_kv = 0; tile_kv < attend_range; tile_kv += FA2H_KV) {
        uint kv_tile_len = min(FA2H_KV, attend_range - tile_kv);

        for (uint i = lid; i < FA2H_KV * FA2H_HD; i += FA2H_TG) {
            uint ki  = i / FA2H_HD;
            uint d   = i % FA2H_HD;
            uint gkv = tile_kv + ki;
            kv_tile[i] = (ki < kv_tile_len)
                ? half(K[gkv * kv_stride + kv_h * FA2H_HD + d])
                : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < FA2H_Q * FA2H_KV) {
            uint qi = lid / FA2H_KV;
            uint ki = lid % FA2H_KV;
            uint gqi = tile_q + qi;
            uint gkv = tile_kv + ki;
            if (gqi < n_tokens && ki < kv_tile_len && gkv <= gqi) {
                float s = 0.0f;
                for (uint d = 0; d < FA2H_HD; d++) {
                    s += float(q_tile[qi * FA2H_HD + d]) * float(kv_tile[ki * FA2H_HD + d]);
                }
                scores[lid] = s * scale;
            } else {
                scores[lid] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < FA2H_Q) {
            uint qi = lid;
            float prev_max = running_max[qi];
            float tile_max_qi = -INFINITY;
            for (uint ki = 0; ki < FA2H_KV; ki++) {
                tile_max_qi = max(tile_max_qi, scores[qi * FA2H_KV + ki]);
            }
            float new_max = max(prev_max, tile_max_qi);
            float corr    = ax_exp(prev_max - new_max);
            correction[qi] = corr;

            float qi_sum = 0.0f;
            for (uint ki = 0; ki < FA2H_KV; ki++) {
                float e = ax_exp(scores[qi * FA2H_KV + ki] - new_max);
                scores[qi * FA2H_KV + ki] = e;
                qi_sum += e;
            }
            running_max[qi] = new_max;
            running_sum[qi] = running_sum[qi] * corr + qi_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint qi     = lid / (FA2H_TG / FA2H_Q);
            uint d_base = (lid % (FA2H_TG / FA2H_Q)) * FA2H_D_PER_THREAD;
            float corr  = correction[qi];
            for (uint di = 0; di < FA2H_D_PER_THREAD; di++) {
                out_acc[qi * FA2H_HD + d_base + di] *= corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lid; i < FA2H_KV * FA2H_HD; i += FA2H_TG) {
            uint ki  = i / FA2H_HD;
            uint d   = i % FA2H_HD;
            uint gkv = tile_kv + ki;
            kv_tile[i] = (ki < kv_tile_len)
                ? half(V[gkv * kv_stride + kv_h * FA2H_HD + d])
                : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint qi     = lid / (FA2H_TG / FA2H_Q);
            uint d_base = (lid % (FA2H_TG / FA2H_Q)) * FA2H_D_PER_THREAD;
            for (uint di = 0; di < FA2H_D_PER_THREAD; di++) {
                uint d = d_base + di;
                float acc = out_acc[qi * FA2H_HD + d];
                for (uint ki = 0; ki < kv_tile_len; ki++) {
                    acc += scores[qi * FA2H_KV + ki] * float(kv_tile[ki * FA2H_HD + d]);
                }
                out_acc[qi * FA2H_HD + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < FA2H_Q * FA2H_HD; i += FA2H_TG) {
        uint qi = i / FA2H_HD;
        uint d  = i % FA2H_HD;
        uint gqi = tile_q + qi;
        if (gqi < n_tokens) {
            float inv_sum = (running_sum[qi] > 0.0f) ? (1.0f / running_sum[qi]) : 0.0f;
            O[gqi * q_stride + h * FA2H_HD + d] = out_acc[i] * inv_sum;
        }
    }
}

// ── sdpa_vector decode — HD=256, f16 KV cache ─────────────────────────────
//
// MLX sdpa_vector pattern adapted for head_dim=256 and f16 KV cache.
//
// Thread layout (TG = SDPA_TG = 256 threads, SDPA_N_SG = 8 simdgroups):
//   - Each simdgroup handles ONE KV position per tile iteration.
//   - Each lane within the simdgroup owns SDPA_D_PER_LANE = HD/32 = 8 consecutive
//     D-elements.  The simdgroup computes the full Q·K dot product via simd_sum.
//   - V reads are 8 consecutive halves per lane = 128-bit aligned sequential loads,
//     eliminating the strided-D scatter in the original kernel.
//
// Each simdgroup independently maintains online-softmax state (running_max,
// running_sum, o_acc[8]) processing every SDPA_N_SG-th KV position.
// After all tiles the N_SG partial states are merged via the standard
// log-sum-exp merge into the final output (written by simdgroup 0).
//
// TG memory: ~8.1 KB (tg_sg_acc[8][256] floats + small bookkeeping).
//
// Grid: n_heads threadgroups.

constant uint SDPA_TG         = 256;
constant uint SDPA_N_SG       = SDPA_TG / 32;   // 8 simdgroups
constant uint SDPA_HD         = 256;
constant uint SDPA_D_PER_LANE = SDPA_HD / 32;   // 8

kernel void attention_decode_sdpa_hd256(
    device const float* Q        [[buffer(0)]],
    device const half*  K_cache  [[buffer(1)]],
    device const half*  V_cache  [[buffer(2)]],
    device float* O              [[buffer(3)]],
    constant uint& n_heads       [[buffer(4)]],
    constant uint& n_kv_heads    [[buffer(5)]],
    constant uint& head_dim      [[buffer(6)]],
    constant uint& attend_start  [[buffer(7)]],
    constant uint& attend_len    [[buffer(8)]],
    uint head                    [[threadgroup_position_in_grid]],
    uint lid                     [[thread_index_in_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    if (head >= n_heads || head_dim != SDPA_HD) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = head / heads_per_kv;
    uint kv_stride    = n_kv_heads * SDPA_HD;
    constexpr float scale = 1.0f / 16.0f;  // rsqrt(256)

    // Each lane owns D[d_base .. d_base + SDPA_D_PER_LANE)
    uint d_base = simd_lane * SDPA_D_PER_LANE;

    // Load this lane's Q slice into registers.
    device const float* q_ptr = Q + head * SDPA_HD + d_base;
    float q_reg[SDPA_D_PER_LANE];
    for (uint i = 0; i < SDPA_D_PER_LANE; i++) {
        q_reg[i] = q_ptr[i];
    }

    // Per-simdgroup online-softmax state (independent per simdgroup).
    float sg_max = -INFINITY;
    float sg_sum = 0.0f;
    float sg_acc[SDPA_D_PER_LANE];
    for (uint i = 0; i < SDPA_D_PER_LANE; i++) sg_acc[i] = 0.0f;

    // Each simdgroup processes positions: simd_id, simd_id + N_SG, simd_id + 2*N_SG, ...
    for (uint pos = simd_id; pos < attend_len; pos += SDPA_N_SG) {
        uint t_abs = attend_start + pos;

        // Q·K dot product: each lane computes partial dot over 8 D-elements,
        // simd_sum gives the full dot product (same in all 32 lanes).
        device const half* k_ptr = K_cache + t_abs * kv_stride + kv_h * SDPA_HD + d_base;
        float partial = 0.0f;
        for (uint i = 0; i < SDPA_D_PER_LANE; i++) {
            partial += q_reg[i] * float(k_ptr[i]);
        }
        float dot = simd_sum(partial) * scale;

        // Online softmax update.
        float prev_max = sg_max;
        float new_max  = max(prev_max, dot);
        float corr     = ax_exp(prev_max - new_max);
        float exp_w    = ax_exp(dot - new_max);

        // Accumulate weighted V.  Sequential reads: 8 consecutive halves.
        device const half* v_ptr = V_cache + t_abs * kv_stride + kv_h * SDPA_HD + d_base;
        for (uint i = 0; i < SDPA_D_PER_LANE; i++) {
            sg_acc[i] = sg_acc[i] * corr + exp_w * float(v_ptr[i]);
        }
        sg_max = new_max;
        sg_sum = sg_sum * corr + exp_w;
    }

    // ── Merge N_SG partial states via threadgroup memory ─────────────────
    // tg_sg_acc: [N_SG][HD] = 8*256 f32 = 8 KB
    // tg_sg_max/sum: [N_SG] = 32+32 bytes
    threadgroup float tg_sg_acc[SDPA_N_SG * SDPA_HD];  // 8KB
    threadgroup float tg_sg_max[SDPA_N_SG];
    threadgroup float tg_sg_sum[SDPA_N_SG];
    threadgroup float tg_gmax;
    threadgroup float tg_gsum;

    // Each lane writes its 8 acc values into tg_sg_acc[simd_id][d_base..d_base+8].
    for (uint i = 0; i < SDPA_D_PER_LANE; i++) {
        tg_sg_acc[simd_id * SDPA_HD + d_base + i] = sg_acc[i];
    }
    if (simd_lane == 0) {
        tg_sg_max[simd_id] = sg_max;
        tg_sg_sum[simd_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 merges partial softmax states.
    if (lid == 0) {
        float gmax = tg_sg_max[0];
        for (uint g = 1; g < SDPA_N_SG; g++) gmax = max(gmax, tg_sg_max[g]);
        float gsum = 0.0f;
        for (uint g = 0; g < SDPA_N_SG; g++) {
            gsum += tg_sg_sum[g] * ax_exp(tg_sg_max[g] - gmax);
        }
        tg_gmax = gmax;
        tg_gsum = gsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup 0 writes the output: each of its 32 lanes handles 8 D-elements.
    if (simd_id == 0) {
        float gmax     = tg_gmax;
        float inv_gsum = (tg_gsum > 0.0f) ? (1.0f / tg_gsum) : 0.0f;
        device float* o_ptr = O + head * SDPA_HD + d_base;
        for (uint i = 0; i < SDPA_D_PER_LANE; i++) {
            float total = 0.0f;
            for (uint g = 0; g < SDPA_N_SG; g++) {
                total += tg_sg_acc[g * SDPA_HD + d_base + i] * ax_exp(tg_sg_max[g] - gmax);
            }
            o_ptr[i] = total * inv_gsum;
        }
    }
}

// ── FA2-style multi-query prefill, HD=256, f16 KV cache ───────────────────
//
// Processes FA2_Q=8 query tokens per threadgroup, amortizing K/V loads 8×.
// Each KV tile of FA2_KV=16 positions is loaded once into TG memory
// and shared across all 8 query rows.
//
// Thread layout (FA2_TG=256):
//   - lid 0..127: score computation — qi = lid/FA2_KV, ki = lid%FA2_KV
//   - lid 0..7  : per-qi softmax state update (tile_max, exp_scores, running state)
//   - All 256   : out_acc rescale and V accumulation
//                 qi = lid/32, d_start = (lid%32)*8
//
// TG memory budget:
//   q_tile[8][256] half   = 4KB
//   kv_tile[16][256] half = 8KB   (reused for K then V)
//   scores[8][16]  float  = 512B
//   out_acc[8][256] float = 8KB
//   running_max/sum[8]    = 64B
//   correction[8]         = 32B
//   Total ≈ 20.6KB < 32KB ✓
//
// ABI identical to attention_prefill_cache_f16kv (same buffer slots).
// Grid: (ceil(n_tokens / FA2_Q), n_heads) threadgroups.

constant uint FA2_Q   = 8;
constant uint FA2_KV  = 16;
constant uint FA2_TG  = 256;   // FA2_Q * (FA2_TG / FA2_Q) = 8 * 32 threads per qi
constant uint FA2_HD  = 256;
constant uint FA2_D_PER_THREAD = FA2_HD / (FA2_TG / FA2_Q);  // 256/(256/8) = 8

kernel void attention_prefill_cache_fa2_hd256(
    device const float* Q          [[buffer(0)]],
    device const half*  K_cache    [[buffer(1)]],
    device const half*  V_cache    [[buffer(2)]],
    device float* O                [[buffer(3)]],
    constant uint& n_tokens        [[buffer(4)]],
    constant uint& n_heads         [[buffer(5)]],
    constant uint& n_kv_heads      [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],
    constant uint& base_seq_len    [[buffer(8)]],
    constant uint& sliding_window  [[buffer(9)]],
    uint2 tg_id                    [[threadgroup_position_in_grid]],
    uint  lid                      [[thread_index_in_threadgroup]],
    uint  simd_lane                [[thread_index_in_simdgroup]],
    uint  simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2_Q;
    uint h      = tg_id.y;
    if (h >= n_heads || head_dim != FA2_HD) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2_HD;
    uint kv_stride    = n_kv_heads * FA2_HD;
    constexpr float scale = 1.0f / 16.0f;

    threadgroup half  q_tile[FA2_Q * FA2_HD];         // 4 KB
    threadgroup half  kv_tile[FA2_KV * FA2_HD];       // 8 KB
    threadgroup float scores[FA2_Q * FA2_KV];         // 512 B
    threadgroup float out_acc[FA2_Q * FA2_HD];        // 8 KB
    threadgroup float running_max[FA2_Q];
    threadgroup float running_sum[FA2_Q];
    threadgroup float correction[FA2_Q];

    // Initialize per-qi running state and out_acc.
    if (lid < FA2_Q) {
        running_max[lid] = -INFINITY;
        running_sum[lid] = 0.0f;
    }
    for (uint i = lid; i < FA2_Q * FA2_HD; i += FA2_TG) {
        out_acc[i] = 0.0f;
    }

    // Load Q tile [FA2_Q][FA2_HD] half.
    for (uint i = lid; i < FA2_Q * FA2_HD; i += FA2_TG) {
        uint qi = i / FA2_HD;
        uint d  = i % FA2_HD;
        uint gqi = tile_q + qi;
        q_tile[i] = (gqi < n_tokens) ? half(Q[gqi * q_stride + h * FA2_HD + d]) : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Determine KV attend range: union across all 8 queries.
    //   Most conservative: first query's window start, last query's window end.
    uint last_qi_abs = base_seq_len + min(tile_q + FA2_Q, n_tokens);
    uint max_attend_end   = last_qi_abs;
    uint min_attend_start = 0;
    if (sliding_window > 0) {
        uint first_qi_abs = base_seq_len + tile_q;
        if (first_qi_abs >= sliding_window) {
            min_attend_start = first_qi_abs - sliding_window + 1;
        }
    }
    if (max_attend_end <= min_attend_start) {
        // Nothing to attend — write zeros and return.
        for (uint i = lid; i < FA2_Q * FA2_HD; i += FA2_TG) {
            uint qi = i / FA2_HD;
            uint d  = i % FA2_HD;
            uint gqi = tile_q + qi;
            if (gqi < n_tokens) {
                O[gqi * q_stride + h * FA2_HD + d] = 0.0f;
            }
        }
        return;
    }
    uint attend_range = max_attend_end - min_attend_start;

    // KV tile loop.
    for (uint tile_kv = 0; tile_kv < attend_range; tile_kv += FA2_KV) {
        uint kv_tile_len = min(FA2_KV, attend_range - tile_kv);

        // ── Load K tile ─────────────────────────────────────────────────
        for (uint i = lid; i < FA2_KV * FA2_HD; i += FA2_TG) {
            uint ki  = i / FA2_HD;
            uint d   = i % FA2_HD;
            uint gkv = min_attend_start + tile_kv + ki;
            kv_tile[i] = (ki < kv_tile_len) ?
                K_cache[gkv * kv_stride + kv_h * FA2_HD + d] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Score computation (threads 0..FA2_Q*FA2_KV-1 = 0..127) ────
        if (lid < FA2_Q * FA2_KV) {
            uint qi = lid / FA2_KV;
            uint ki = lid % FA2_KV;
            uint gqi = tile_q + qi;
            uint gkv = min_attend_start + tile_kv + ki;
            uint qi_abs_pos = base_seq_len + gqi;  // causal: attend up to qi_abs_pos
            if (gqi < n_tokens && ki < kv_tile_len && gkv <= qi_abs_pos) {
                float s = 0.0f;
                for (uint d = 0; d < FA2_HD; d++) {
                    s += float(q_tile[qi * FA2_HD + d]) * float(kv_tile[ki * FA2_HD + d]);
                }
                scores[lid] = s * scale;
            } else {
                scores[lid] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Per-qi softmax state update (threads 0..FA2_Q-1 = 0..7) ───
        if (lid < FA2_Q) {
            uint qi = lid;
            float prev_max = running_max[qi];
            float tile_max_qi = -INFINITY;
            for (uint ki = 0; ki < FA2_KV; ki++) {
                tile_max_qi = max(tile_max_qi, scores[qi * FA2_KV + ki]);
            }
            float new_max = max(prev_max, tile_max_qi);
            float corr    = ax_exp(prev_max - new_max);
            correction[qi] = corr;

            float qi_sum = 0.0f;
            for (uint ki = 0; ki < FA2_KV; ki++) {
                float e = ax_exp(scores[qi * FA2_KV + ki] - new_max);
                scores[qi * FA2_KV + ki] = e;
                qi_sum += e;
            }
            running_max[qi] = new_max;
            running_sum[qi] = running_sum[qi] * corr + qi_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Rescale out_acc (all 256 threads) ──────────────────────────
        // Thread layout: qi = lid/32, d_base = (lid%32)*FA2_D_PER_THREAD
        {
            uint qi     = lid / (FA2_TG / FA2_Q);
            uint d_base = (lid % (FA2_TG / FA2_Q)) * FA2_D_PER_THREAD;
            float corr  = correction[qi];
            for (uint di = 0; di < FA2_D_PER_THREAD; di++) {
                out_acc[qi * FA2_HD + d_base + di] *= corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Load V tile (reuse kv_tile buffer) ─────────────────────────
        for (uint i = lid; i < FA2_KV * FA2_HD; i += FA2_TG) {
            uint ki  = i / FA2_HD;
            uint d   = i % FA2_HD;
            uint gkv = min_attend_start + tile_kv + ki;
            kv_tile[i] = (ki < kv_tile_len) ?
                V_cache[gkv * kv_stride + kv_h * FA2_HD + d] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── V accumulation (all 256 threads) ───────────────────────────
        // Each thread handles FA2_D_PER_THREAD = 8 D-elements for one qi.
        {
            uint qi     = lid / (FA2_TG / FA2_Q);
            uint d_base = (lid % (FA2_TG / FA2_Q)) * FA2_D_PER_THREAD;
            for (uint di = 0; di < FA2_D_PER_THREAD; di++) {
                uint d = d_base + di;
                float acc = out_acc[qi * FA2_HD + d];
                for (uint ki = 0; ki < kv_tile_len; ki++) {
                    acc += scores[qi * FA2_KV + ki] * float(kv_tile[ki * FA2_HD + d]);
                }
                out_acc[qi * FA2_HD + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write output ─────────────────────────────────────────────────────
    for (uint i = lid; i < FA2_Q * FA2_HD; i += FA2_TG) {
        uint qi = i / FA2_HD;
        uint d  = i % FA2_HD;
        uint gqi = tile_q + qi;
        if (gqi < n_tokens) {
            float inv_sum = (running_sum[qi] > 0.0f) ? (1.0f / running_sum[qi]) : 0.0f;
            O[gqi * q_stride + h * FA2_HD + d] = out_acc[i] * inv_sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FA2 prefill with simdgroup matrix operations — head_dim=128
//
// Uses Apple Silicon simdgroup_half8x8 hardware matrix units for both
// QK^T and O=S*V, matching llama.cpp's flash_attn_ext approach.
//
// Parallelism strategy (matches llama.cpp):
//   QK^T phase: simdgroups divide KV dimension (NC=2 blocks of 8 KV each)
//   Softmax:    simdgroups divide Q  dimension (NQ=2 queries each)
//   OV phase:   simdgroups divide DV dimension (NO=4 blocks of 8 dims each)
//
// Grid: (ceil(n_tokens/8), n_heads) threadgroups.
// TG: 128 threads = 4 simdgroups × 32 threads.
//
// Threadgroup memory: ~10 KB
//   sq[8×128] float = 4 KB   Q tile (float for type-matched K MMA)
//   so[8×128] float = 4 KB   O accumulator (float throughout)
//   ss[8×64]  float = 2 KB   softmax scores (float for type-matched V MMA)
//   No K/V scratch: K and V loaded directly from device float* via simdgroup_load
//
// G21: All-float MMA eliminates K/V threadgroup staging entirely.
// simdgroup_load(float8x8, device float*, ...) reads K/V directly from
// device memory. On UMA this is a single read instead of
// device→TG→registers (double copy). Barriers inside the QK^T and S×V
// inner loops are eliminated.
// ═══════════════════════════════════════════════════════════════════════════

constant uint FA2S_Q   = 8;      // queries per threadgroup
constant uint FA2S_C   = 64;     // KV tokens per tile
constant uint FA2S_NSG = 4;      // simdgroups per threadgroup
constant uint FA2S_NW  = 32;     // threads per simdgroup
constant uint FA2S_TG  = FA2S_NW * FA2S_NSG;  // 128 threads
constant uint FA2S_HD  = 128;    // head_dim

constant uint FA2S_NQ  = FA2S_Q / FA2S_NSG;            // 2 queries per SG (softmax)
constant uint FA2S_NC  = (FA2S_C / 8) / FA2S_NSG;      // 2 KV-blocks per SG (QK^T)
constant uint FA2S_NO  = (FA2S_HD / 8) / FA2S_NSG;     // 4 output-blocks per SG (OV)

kernel void attention_prefill_fa2_simd_hd128(
    device const float* Q_buf   [[buffer(0)]],
    device const float* K_buf   [[buffer(1)]],
    device const float* V_buf   [[buffer(2)]],
    device float* O_buf         [[buffer(3)]],
    constant uint& n_tokens     [[buffer(4)]],
    constant uint& n_heads      [[buffer(5)]],
    constant uint& n_kv_heads   [[buffer(6)]],
    constant uint& head_dim     [[buffer(7)]],
    uint2 tg_id       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2S_Q;
    uint h      = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2S_HD;
    uint kv_stride    = n_kv_heads * FA2S_HD;
    constexpr float inv_sqrt_hd = 0.08838834764831843f; // 1/sqrt(128)

    // ── Threadgroup memory (no K/V scratch) ──────────────────────────
    threadgroup float sq[FA2S_Q * FA2S_HD];                 // 4 KB: Q tile (float)
    threadgroup float so[FA2S_Q * FA2S_HD];                 // 4 KB: O accumulator (float)
    threadgroup float ss[FA2S_Q * FA2S_C];                  // 2 KB: scores (float)

    // Per-query online softmax state (register, per simdgroup)
    float S[FA2S_NQ] = {};
    float M[FA2S_NQ];
    for (uint i = 0; i < FA2S_NQ; i++) M[i] = -INFINITY;

    // ── Load Q into threadgroup as float ─────────────────────────────
    for (uint i = lid; i < FA2S_Q * FA2S_HD; i += FA2S_TG) {
        uint qi = i / FA2S_HD;
        uint d  = i % FA2S_HD;
        uint gqi = tile_q + qi;
        sq[i] = (gqi < n_tokens)
            ? Q_buf[gqi * q_stride + h * FA2S_HD + d]
            : 0.0f;
    }

    // Zero O accumulator
    for (uint i = lid; i < FA2S_Q * FA2S_HD; i += FA2S_TG) {
        so[i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint attend_range = min(tile_q + FA2S_Q, n_tokens);

    // ── Main tile loop over KV sequence ──────────────────────────────
    for (uint tile_kv = 0; tile_kv < attend_range; tile_kv += FA2S_C) {
        uint kv_tile_len = min(uint(FA2S_C), attend_range - tile_kv);

        // ── Phase 1: QK^T via simdgroup_float8x8 ────────────────────
        // K loaded DIRECTLY from device float* — no TG staging.
        // Each simdgroup handles NC=2 blocks of 8 KV tokens.
        for (uint cc = 0; cc < FA2S_NC; cc++) {
            uint block_idx = cc * FA2S_NSG + simd_id;
            uint gkv_base  = tile_kv + block_idx * 8;

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            // K base pointer for this 8-token block
            device const float* k_block = K_buf + gkv_base * kv_stride
                                                + kv_h * FA2S_HD;

            for (uint dk = 0; dk < FA2S_HD; dk += 16) {
                simdgroup_float8x8 mq0, mq1, mk0, mk1;
                // Q from threadgroup (float)
                simdgroup_load(mq0, sq + dk,     FA2S_HD);
                simdgroup_load(mq1, sq + dk + 8, FA2S_HD);
                // K directly from device (float), transposed
                // Guard: if gkv_base+7 >= attend_range, some rows are OOB.
                // simdgroup_load from device with OOB rows reads garbage,
                // but those rows' QK^T contributions are masked to -INF in
                // softmax (causal mask: gkv > gqi → -INF). Safe.
                simdgroup_load(mk0, k_block + dk,     kv_stride, ulong2(0, 0), true);
                simdgroup_load(mk1, k_block + dk + 8, kv_stride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }

            // Store QK^T scores as float
            simdgroup_store(mqk, ss + block_idx * 8, FA2S_C);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 2: Online softmax (parallel over Q) ────────────────
        for (uint jj = 0; jj < FA2S_NQ; jj++) {
            uint j   = jj * FA2S_NSG + simd_id;
            uint gqi = tile_q + j;
            float prev_max = M[jj];

            uint ki0 = simd_lane * 2;
            uint ki1 = ki0 + 1;
            uint gkv0 = tile_kv + ki0;
            uint gkv1 = tile_kv + ki1;

            // Read scores (already float), apply scale + causal mask
            float s0 = -INFINITY, s1 = -INFINITY;
            if (ki0 < kv_tile_len && gkv0 <= gqi) {
                s0 = ss[j * FA2S_C + ki0] * inv_sqrt_hd;
            }
            if (ki1 < kv_tile_len && gkv1 <= gqi) {
                s1 = ss[j * FA2S_C + ki1] * inv_sqrt_hd;
            }

            float new_max = simd_max(max(max(s0, s1), prev_max));
            float ms = ax_exp(prev_max - new_max);
            float e0 = ax_exp(s0 - new_max);
            float e1 = ax_exp(s1 - new_max);
            float tile_sum = simd_sum(e0 + e1);

            M[jj] = new_max;
            S[jj] = S[jj] * ms + tile_sum;

            // Store softmax probabilities as float
            ss[j * FA2S_C + ki0] = e0;
            ss[j * FA2S_C + ki1] = e1;

            // Correct O accumulator: so[j, :] *= ms
            for (uint d = simd_lane; d < FA2S_HD; d += FA2S_NW) {
                so[j * FA2S_HD + d] *= ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 3: O += S × V via simdgroup_float8x8 ──────────────
        // V loaded DIRECTLY from device float* — no TG staging.

        // Load O accumulators into simdgroup registers
        simdgroup_float8x8 lo[FA2S_NO];
        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2S_NO; oo++) {
                simdgroup_load(lo[oo], sop, FA2S_HD);
                sop += 8 * FA2S_NSG;
            }
        }

        for (uint cc = 0; cc < FA2S_C / 8; cc++) {
            simdgroup_float8x8 ms_mat;
            simdgroup_load(ms_mat, ss + 8 * cc, FA2S_C);
            uint gkv_base = tile_kv + cc * 8;

            for (uint oo = 0; oo < FA2S_NO; oo++) {
                uint dv_block  = oo * FA2S_NSG + simd_id;
                uint dv_offset = kv_h * FA2S_HD + dv_block * 8;

                // V directly from device (float), no staging
                simdgroup_float8x8 mv;
                simdgroup_load(mv, V_buf + gkv_base * kv_stride + dv_offset, kv_stride);
                simdgroup_multiply_accumulate(lo[oo], ms_mat, mv, lo[oo]);
            }
        }

        // Store O accumulators back to threadgroup (float)
        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2S_NO; oo++) {
                simdgroup_store(lo[oo], sop, FA2S_HD);
                sop += 8 * FA2S_NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write final output (divide by softmax sum) ───────────────────
    for (uint jj = 0; jj < FA2S_NQ; jj++) {
        uint j   = jj * FA2S_NSG + simd_id;
        uint gqi = tile_q + j;
        if (gqi < n_tokens && S[jj] > 0.0f) {
            float inv_sum = 1.0f / S[jj];
            for (uint d = simd_lane; d < FA2S_HD; d += FA2S_NW) {
                O_buf[gqi * q_stride + h * FA2S_HD + d] =
                    so[j * FA2S_HD + d] * inv_sum;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FA2 v2: Mixed-precision prefill attention — head_dim=128
//
// Same FA2 architecture as fa2_simd_hd128 but with mixed-precision MMA:
// - Q stored as half in TG memory (2 KB vs 4 KB)
// - Scores stored as half (128 B vs 512 B)
// - QK^T: half(Q) × float(K) → float accumulator (uses hardware mixed MMA)
// - S×V:  half(S) × float(V) → float accumulator (uses hardware mixed MMA)
// - Softmax: computed in float (numerical stability)
// - O accumulator: float throughout
//
// The mixed-precision MMA leverages Apple Silicon's simdgroup matrix
// hardware which provides higher throughput when at least one operand is half.
//
// TG memory: sq(2KB half) + so(4KB float) + ss(128B half) = ~6.1 KB
// Grid: (ceil(n_tokens / 8), n_heads) threadgroups.
// ═══════════════════════════════════════════════════════════════════════════

constant uint FA2V2_Q   = 8;
constant uint FA2V2_C   = 64;
constant uint FA2V2_NSG = 4;
constant uint FA2V2_NW  = 32;
constant uint FA2V2_TG  = FA2V2_NW * FA2V2_NSG;
constant uint FA2V2_HD  = 128;

constant uint FA2V2_NQ  = FA2V2_Q / FA2V2_NSG;
constant uint FA2V2_NC  = (FA2V2_C / 8) / FA2V2_NSG;
constant uint FA2V2_NO  = (FA2V2_HD / 8) / FA2V2_NSG;

kernel void attention_prefill_fa2v2_hd128(
    device const float* Q_buf   [[buffer(0)]],
    device const float* K_buf   [[buffer(1)]],
    device const float* V_buf   [[buffer(2)]],
    device float* O_buf         [[buffer(3)]],
    constant uint& n_tokens     [[buffer(4)]],
    constant uint& n_heads      [[buffer(5)]],
    constant uint& n_kv_heads   [[buffer(6)]],
    constant uint& head_dim     [[buffer(7)]],
    uint2 tg_id       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2V2_Q;
    uint h      = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2V2_HD;
    uint kv_stride    = n_kv_heads * FA2V2_HD;
    constexpr float inv_sqrt_hd = 0.08838834764831843f; // 1/sqrt(128)

    // ── TG memory: Q as half, O as float, scores as float ───────
    threadgroup half  sq[FA2V2_Q * FA2V2_HD];        // 2 KB
    threadgroup float so[FA2V2_Q * FA2V2_HD];        // 4 KB
    threadgroup float ss[FA2V2_Q * FA2V2_C];         // 2 KB (float for QK^T + softmax)
    threadgroup half  sp[FA2V2_Q * FA2V2_C];         // 1 KB (half for S×V MMA)

    // Per-query online softmax state
    float S[FA2V2_NQ] = {};
    float M[FA2V2_NQ];
    for (uint i = 0; i < FA2V2_NQ; i++) M[i] = -INFINITY;

    // ── Load Q into TG as half ───────────────────────────────────
    for (uint i = lid; i < FA2V2_Q * FA2V2_HD; i += FA2V2_TG) {
        uint qi = i / FA2V2_HD;
        uint d  = i % FA2V2_HD;
        uint gqi = tile_q + qi;
        sq[i] = (gqi < n_tokens)
            ? half(Q_buf[gqi * q_stride + h * FA2V2_HD + d])
            : half(0.0h);
    }
    for (uint i = lid; i < FA2V2_Q * FA2V2_HD; i += FA2V2_TG) {
        so[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint attend_range = min(tile_q + FA2V2_Q, n_tokens);

    // ── Main tile loop over KV sequence ──────────────────────────
    for (uint tile_kv = 0; tile_kv < attend_range; tile_kv += FA2V2_C) {
        uint kv_tile_len = min(uint(FA2V2_C), attend_range - tile_kv);

        // ── Phase 1: QK^T — half(Q) × float(K) → float accumulator ──
        for (uint cc = 0; cc < FA2V2_NC; cc++) {
            uint block_idx = cc * FA2V2_NSG + simd_id;
            uint gkv_base  = tile_kv + block_idx * 8;

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            device const float* k_block = K_buf + gkv_base * kv_stride
                                                + kv_h * FA2V2_HD;

            for (uint dk = 0; dk < FA2V2_HD; dk += 16) {
                simdgroup_half8x8 mq0, mq1;
                simdgroup_float8x8 mk0, mk1;
                // Q from half TG
                simdgroup_load(mq0, sq + dk,     FA2V2_HD);
                simdgroup_load(mq1, sq + dk + 8, FA2V2_HD);
                // K from float device, transposed
                simdgroup_load(mk0, k_block + dk,     kv_stride, ulong2(0, 0), true);
                simdgroup_load(mk1, k_block + dk + 8, kv_stride, ulong2(0, 0), true);
                // Mixed MMA: half × float → float
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }

            // Store float QK^T scores
            simdgroup_store(mqk, ss + block_idx * 8, FA2V2_C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 2: Online softmax (float) ──────────────────────
        for (uint jj = 0; jj < FA2V2_NQ; jj++) {
            uint j   = jj * FA2V2_NSG + simd_id;
            uint gqi = tile_q + j;
            float prev_max = M[jj];

            uint ki0 = simd_lane * 2;
            uint ki1 = ki0 + 1;
            uint gkv0 = tile_kv + ki0;
            uint gkv1 = tile_kv + ki1;

            float s0 = -INFINITY, s1 = -INFINITY;
            if (ki0 < kv_tile_len && gkv0 <= gqi) {
                s0 = ss[j * FA2V2_C + ki0] * inv_sqrt_hd;
            }
            if (ki1 < kv_tile_len && gkv1 <= gqi) {
                s1 = ss[j * FA2V2_C + ki1] * inv_sqrt_hd;
            }

            float new_max = simd_max(max(max(s0, s1), prev_max));
            float ms = ax_exp(prev_max - new_max);
            float e0 = ax_exp(s0 - new_max);
            float e1 = ax_exp(s1 - new_max);
            float tile_sum = simd_sum(e0 + e1);

            M[jj] = new_max;
            S[jj] = S[jj] * ms + tile_sum;

            // Store softmax probs as half for S×V MMA
            sp[j * FA2V2_C + ki0] = half(e0);
            sp[j * FA2V2_C + ki1] = half(e1);

            // Correct O accumulator
            for (uint d = simd_lane; d < FA2V2_HD; d += FA2V2_NW) {
                so[j * FA2V2_HD + d] *= ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 3: O += half(S) × float(V) → float ────────────
        simdgroup_float8x8 lo[FA2V2_NO];
        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2V2_NO; oo++) {
                simdgroup_load(lo[oo], sop, FA2V2_HD);
                sop += 8 * FA2V2_NSG;
            }
        }

        for (uint cc = 0; cc < FA2V2_C / 8; cc++) {
            simdgroup_half8x8 ms_mat;
            simdgroup_load(ms_mat, sp + 8 * cc, FA2V2_C);
            uint gkv_base = tile_kv + cc * 8;

            for (uint oo = 0; oo < FA2V2_NO; oo++) {
                uint dv_block  = oo * FA2V2_NSG + simd_id;
                uint dv_offset = kv_h * FA2V2_HD + dv_block * 8;

                // V from float device
                simdgroup_float8x8 mv;
                simdgroup_load(mv, V_buf + gkv_base * kv_stride + dv_offset, kv_stride);
                // Mixed MMA: half(S) × float(V) → float(O)
                simdgroup_multiply_accumulate(lo[oo], ms_mat, mv, lo[oo]);
            }
        }

        // Store O back to TG
        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2V2_NO; oo++) {
                simdgroup_store(lo[oo], sop, FA2V2_HD);
                sop += 8 * FA2V2_NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write final output ───────────────────────────────────────
    for (uint jj = 0; jj < FA2V2_NQ; jj++) {
        uint j   = jj * FA2V2_NSG + simd_id;
        uint gqi = tile_q + j;
        if (gqi < n_tokens) {
            float inv_sum = (S[jj] > 0.0f) ? (1.0f / S[jj]) : 0.0f;
            for (uint d = simd_lane; d < FA2V2_HD; d += FA2V2_NW)
                O_buf[gqi * q_stride + h * FA2V2_HD + d] =
                    so[j * FA2V2_HD + d] * inv_sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FA2 full-half prefill attention — head_dim=128
//
// Like fa2v2_hd128 but stages K and V through threadgroup memory as half,
// enabling half×half→float simdgroup MMA for both QK^T and S×V.
// This matches llama.cpp's kernel_flash_attn_ext strategy which uses
// simdgroup_half8x8 for both operands with float accumulators.
//
// Key change from fa2v2: K and V are read from device float*, cast to half
// in TG scratch per-SG, then loaded as simdgroup_half8x8 for MMA.
//
// TG memory: sq(2KB half) + so(4KB float) + ss(2KB float) + sp(1KB half)
//            + sk(4KB half per-SG K/V staging) = ~13 KB
// Grid: (ceil(n_tokens / 8), n_heads) threadgroups.
// ═══════════════════════════════════════════════════════════════════════════

constant uint FA2FH_Q   = 8;
constant uint FA2FH_C   = 64;
constant uint FA2FH_NSG = 4;
constant uint FA2FH_NW  = 32;
constant uint FA2FH_TG  = FA2FH_NW * FA2FH_NSG;
constant uint FA2FH_HD  = 128;

constant uint FA2FH_NQ  = FA2FH_Q / FA2FH_NSG;            // 2
constant uint FA2FH_NC  = (FA2FH_C / 8) / FA2FH_NSG;      // 2
constant uint FA2FH_NO  = (FA2FH_HD / 8) / FA2FH_NSG;     // 4

kernel void attention_prefill_fa2_half_hd128(
    device const float* Q_buf   [[buffer(0)]],
    device const float* K_buf   [[buffer(1)]],
    device const float* V_buf   [[buffer(2)]],
    device float* O_buf         [[buffer(3)]],
    constant uint& n_tokens     [[buffer(4)]],
    constant uint& n_heads      [[buffer(5)]],
    constant uint& n_kv_heads   [[buffer(6)]],
    constant uint& head_dim     [[buffer(7)]],
    uint2 tg_id       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2FH_Q;
    uint h      = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2FH_HD;
    uint kv_stride    = n_kv_heads * FA2FH_HD;
    constexpr float inv_sqrt_hd = 0.08838834764831843f; // 1/sqrt(128)

    // ── TG memory ────────────────────────────────────────────────────
    threadgroup half  sq[FA2FH_Q * FA2FH_HD];          // 2 KB: Q (half)
    threadgroup float so[FA2FH_Q * FA2FH_HD];          // 4 KB: O accumulator (float)
    threadgroup float ss[FA2FH_Q * FA2FH_C];           // 2 KB: scores (float, QK^T + softmax)
    threadgroup half  sp[FA2FH_Q * FA2FH_C];           // 1 KB: softmax probs (half, for S×V)
    // Per-SG K/V staging scratch: 8 rows × 128 cols × half = 1 KB per SG, 4 KB total.
    // Reused for V staging in Phase 3.
    threadgroup half  sk[FA2FH_NSG * 8 * FA2FH_HD];   // 4 KB: K/V staging

    // Per-query online softmax state
    float S[FA2FH_NQ] = {};
    float M[FA2FH_NQ];
    for (uint i = 0; i < FA2FH_NQ; i++) M[i] = -INFINITY;

    // ── Load Q from device float → TG half ───────────────────────────
    for (uint i = lid; i < FA2FH_Q * FA2FH_HD; i += FA2FH_TG) {
        uint qi = i / FA2FH_HD;
        uint d  = i % FA2FH_HD;
        uint gqi = tile_q + qi;
        sq[i] = (gqi < n_tokens)
            ? half(Q_buf[gqi * q_stride + h * FA2FH_HD + d])
            : half(0.0h);
    }
    for (uint i = lid; i < FA2FH_Q * FA2FH_HD; i += FA2FH_TG) {
        so[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint attend_range = min(tile_q + FA2FH_Q, n_tokens);

    // ── Main tile loop over KV sequence ──────────────────────────────
    for (uint tile_kv = 0; tile_kv < attend_range; tile_kv += FA2FH_C) {
        uint kv_tile_len = min(uint(FA2FH_C), attend_range - tile_kv);

        // ── Phase 1: QK^T — half(Q) × half(K) → float accumulator ──
        // Each SG stages its own 8-row K block into sk, then uses
        // half×half simdgroup MMA.
        for (uint cc = 0; cc < FA2FH_NC; cc++) {
            uint block_idx = cc * FA2FH_NSG + simd_id;
            uint gkv_base  = tile_kv + block_idx * 8;

            // Stage 8 K rows from device float → per-SG TG half scratch.
            // All 32 lanes cooperate: each loads 128/32 = 4 elements per row,
            // covering 8 rows × 128 cols = 1024 half values.
            threadgroup half* my_sk = sk + simd_id * 8 * FA2FH_HD;
            for (uint r = 0; r < 8; r++) {
                uint gkv = gkv_base + r;
                device const float* k_row = K_buf + gkv * kv_stride + kv_h * FA2FH_HD;
                for (uint d = simd_lane; d < FA2FH_HD; d += FA2FH_NW) {
                    my_sk[r * FA2FH_HD + d] = (gkv < attend_range)
                        ? half(k_row[d]) : half(0.0h);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            for (uint dk = 0; dk < FA2FH_HD; dk += 16) {
                simdgroup_half8x8 mq0, mq1, mk0, mk1;
                // Q from half TG
                simdgroup_load(mq0, sq + dk,     FA2FH_HD);
                simdgroup_load(mq1, sq + dk + 8, FA2FH_HD);
                // K from half TG scratch, transposed
                simdgroup_load(mk0, my_sk + dk,     FA2FH_HD, ulong2(0, 0), true);
                simdgroup_load(mk1, my_sk + dk + 8, FA2FH_HD, ulong2(0, 0), true);
                // half × half → float accumulator
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }

            // Store float QK^T scores
            simdgroup_store(mqk, ss + block_idx * 8, FA2FH_C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 2: Online softmax (float) ──────────────────────────
        for (uint jj = 0; jj < FA2FH_NQ; jj++) {
            uint j   = jj * FA2FH_NSG + simd_id;
            uint gqi = tile_q + j;
            float prev_max = M[jj];

            uint ki0 = simd_lane * 2;
            uint ki1 = ki0 + 1;
            uint gkv0 = tile_kv + ki0;
            uint gkv1 = tile_kv + ki1;

            float s0 = -INFINITY, s1 = -INFINITY;
            if (ki0 < kv_tile_len && gkv0 <= gqi) {
                s0 = ss[j * FA2FH_C + ki0] * inv_sqrt_hd;
            }
            if (ki1 < kv_tile_len && gkv1 <= gqi) {
                s1 = ss[j * FA2FH_C + ki1] * inv_sqrt_hd;
            }

            float new_max = simd_max(max(max(s0, s1), prev_max));
            float ms = ax_exp(prev_max - new_max);
            float e0 = ax_exp(s0 - new_max);
            float e1 = ax_exp(s1 - new_max);
            float tile_sum = simd_sum(e0 + e1);

            M[jj] = new_max;
            S[jj] = S[jj] * ms + tile_sum;

            // Store softmax probs as half for S×V MMA
            sp[j * FA2FH_C + ki0] = half(e0);
            sp[j * FA2FH_C + ki1] = half(e1);

            // Correct O accumulator
            for (uint d = simd_lane; d < FA2FH_HD; d += FA2FH_NW) {
                so[j * FA2FH_HD + d] *= ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 3: O += half(S) × half(V) → float ─────────────────
        // V staged through TG scratch (reuse sk) per 8-token block.
        simdgroup_float8x8 lo[FA2FH_NO];
        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2FH_NO; oo++) {
                simdgroup_load(lo[oo], sop, FA2FH_HD);
                sop += 8 * FA2FH_NSG;
            }
        }

        for (uint cc = 0; cc < FA2FH_C / 8; cc++) {
            uint gkv_base = tile_kv + cc * 8;

            // Stage 8 V rows from device float → TG half scratch.
            // All SGs cooperate on the same V block (shared across output dims).
            threadgroup half* v_scratch = sk; // Reuse sk for V staging
            for (uint i = lid; i < 8 * FA2FH_HD; i += FA2FH_TG) {
                uint r = i / FA2FH_HD;
                uint d = i % FA2FH_HD;
                uint gkv = gkv_base + r;
                v_scratch[i] = (gkv < attend_range)
                    ? half(V_buf[gkv * kv_stride + kv_h * FA2FH_HD + d])
                    : half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 ms_mat;
            simdgroup_load(ms_mat, sp + 8 * cc, FA2FH_C);

            for (uint oo = 0; oo < FA2FH_NO; oo++) {
                uint dv_block  = oo * FA2FH_NSG + simd_id;
                uint dv_offset = dv_block * 8;

                // V from half TG scratch
                simdgroup_half8x8 mv;
                simdgroup_load(mv, v_scratch + dv_offset, FA2FH_HD);
                // half × half → float accumulator
                simdgroup_multiply_accumulate(lo[oo], ms_mat, mv, lo[oo]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store O back to TG
        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2FH_NO; oo++) {
                simdgroup_store(lo[oo], sop, FA2FH_HD);
                sop += 8 * FA2FH_NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write final output ───────────────────────────────────────────
    for (uint jj = 0; jj < FA2FH_NQ; jj++) {
        uint j   = jj * FA2FH_NSG + simd_id;
        uint gqi = tile_q + j;
        if (gqi < n_tokens) {
            float inv_sum = (S[jj] > 0.0f) ? (1.0f / S[jj]) : 0.0f;
            for (uint d = simd_lane; d < FA2FH_HD; d += FA2FH_NW)
                O_buf[gqi * q_stride + h * FA2FH_HD + d] =
                    so[j * FA2FH_HD + d] * inv_sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FA2 prefill with simdgroup matrix operations — head_dim=256
//
// Same architecture as HD=128 but for head_dim=256 (Gemma3 12B/27B).
// Key changes: FA2S256_NO=8 output blocks, 16 QK^T iterations per block,
// FA2S256_C=32 (smaller KV tile to fit 32KB TG memory).
// TG memory: 8×256×4 (Q) + 8×256×4 (O) + 8×32×4 (scores) = ~17 KB.
// ═══════════════════════════════════════════════════════════════════════════

constant uint FA2S256_Q   = 8;      // queries per threadgroup
constant uint FA2S256_C   = 32;     // KV tokens per tile (reduced for hd256)
constant uint FA2S256_NSG = 4;      // simdgroups per threadgroup
constant uint FA2S256_NW  = 32;     // threads per simdgroup
constant uint FA2S256_TG  = FA2S256_NW * FA2S256_NSG;  // 128 threads
constant uint FA2S256_HD  = 256;    // head_dim

constant uint FA2S256_NQ  = FA2S256_Q / FA2S256_NSG;            // 2 queries per SG
constant uint FA2S256_NC  = (FA2S256_C / 8) / FA2S256_NSG;      // 1 KV-block per SG
constant uint FA2S256_NO  = (FA2S256_HD / 8) / FA2S256_NSG;     // 8 output-blocks per SG

kernel void attention_prefill_fa2_simd_hd256(
    device const float* Q_buf   [[buffer(0)]],
    device const float* K_buf   [[buffer(1)]],
    device const float* V_buf   [[buffer(2)]],
    device float* O_buf         [[buffer(3)]],
    constant uint& n_tokens     [[buffer(4)]],
    constant uint& n_heads      [[buffer(5)]],
    constant uint& n_kv_heads   [[buffer(6)]],
    constant uint& head_dim     [[buffer(7)]],
    uint2 tg_id       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2S256_Q;
    uint h      = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2S256_HD;
    uint kv_stride    = n_kv_heads * FA2S256_HD;
    constexpr float inv_sqrt_hd = 0.0625f; // 1/sqrt(256) = 1/16

    threadgroup float sq[FA2S256_Q * FA2S256_HD];   // 8 KB: Q tile
    threadgroup float so[FA2S256_Q * FA2S256_HD];   // 8 KB: O accumulator
    threadgroup float ss[FA2S256_Q * FA2S256_C];    // 1 KB: scores

    float S[FA2S256_NQ] = {};
    float M[FA2S256_NQ];
    for (uint i = 0; i < FA2S256_NQ; i++) M[i] = -INFINITY;

    // Load Q tile
    for (uint i = lid; i < FA2S256_Q * FA2S256_HD; i += FA2S256_TG) {
        uint qi = i / FA2S256_HD;
        uint d  = i % FA2S256_HD;
        uint gqi = tile_q + qi;
        sq[i] = (gqi < n_tokens)
            ? Q_buf[gqi * q_stride + h * FA2S256_HD + d]
            : 0.0f;
    }
    for (uint i = lid; i < FA2S256_Q * FA2S256_HD; i += FA2S256_TG) {
        so[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint attend_range = min(tile_q + FA2S256_Q, n_tokens);

    for (uint tile_kv = 0; tile_kv < attend_range; tile_kv += FA2S256_C) {
        uint kv_tile_len = min(uint(FA2S256_C), attend_range - tile_kv);

        // Phase 1: QK^T via simdgroup_float8x8
        for (uint cc = 0; cc < FA2S256_NC; cc++) {
            uint block_idx = cc * FA2S256_NSG + simd_id;
            uint gkv_base  = tile_kv + block_idx * 8;

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
            device const float* k_block = K_buf + gkv_base * kv_stride
                                                + kv_h * FA2S256_HD;

            // 256 / 16 = 16 iterations
            for (uint dk = 0; dk < FA2S256_HD; dk += 16) {
                simdgroup_float8x8 mq0, mq1, mk0, mk1;
                simdgroup_load(mq0, sq + dk,     FA2S256_HD);
                simdgroup_load(mq1, sq + dk + 8, FA2S256_HD);
                simdgroup_load(mk0, k_block + dk,     kv_stride, ulong2(0, 0), true);
                simdgroup_load(mk1, k_block + dk + 8, kv_stride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }
            simdgroup_store(mqk, ss + block_idx * 8, FA2S256_C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: Online softmax
        for (uint jj = 0; jj < FA2S256_NQ; jj++) {
            uint j   = jj * FA2S256_NSG + simd_id;
            uint gqi = tile_q + j;
            float prev_max = M[jj];

            uint ki0 = simd_lane * 2;
            uint ki1 = ki0 + 1;
            // For C=32: only 32 KV tokens, lane 16..31 process ki0=32..63 which are OOB
            float s0 = -INFINITY, s1 = -INFINITY;
            if (ki0 < kv_tile_len && (tile_kv + ki0) <= gqi) {
                s0 = ss[j * FA2S256_C + ki0] * inv_sqrt_hd;
            }
            if (ki1 < kv_tile_len && (tile_kv + ki1) <= gqi) {
                s1 = ss[j * FA2S256_C + ki1] * inv_sqrt_hd;
            }

            float new_max = simd_max(max(max(s0, s1), prev_max));
            float ms = ax_exp(prev_max - new_max);
            float e0 = ax_exp(s0 - new_max);
            float e1 = ax_exp(s1 - new_max);
            float tile_sum = simd_sum(e0 + e1);

            M[jj] = new_max;
            S[jj] = S[jj] * ms + tile_sum;

            ss[j * FA2S256_C + ki0] = e0;
            ss[j * FA2S256_C + ki1] = e1;

            for (uint d = simd_lane; d < FA2S256_HD; d += FA2S256_NW) {
                so[j * FA2S256_HD + d] *= ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: O += S × V via simdgroup_float8x8
        simdgroup_float8x8 lo[FA2S256_NO];
        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2S256_NO; oo++) {
                simdgroup_load(lo[oo], sop, FA2S256_HD);
                sop += 8 * FA2S256_NSG;
            }
        }

        for (uint cc = 0; cc < FA2S256_C / 8; cc++) {
            simdgroup_float8x8 ms_mat;
            simdgroup_load(ms_mat, ss + 8 * cc, FA2S256_C);
            uint gkv_base = tile_kv + cc * 8;

            for (uint oo = 0; oo < FA2S256_NO; oo++) {
                uint dv_block  = oo * FA2S256_NSG + simd_id;
                uint dv_offset = kv_h * FA2S256_HD + dv_block * 8;

                simdgroup_float8x8 mv;
                simdgroup_load(mv, V_buf + gkv_base * kv_stride + dv_offset, kv_stride);
                simdgroup_multiply_accumulate(lo[oo], ms_mat, mv, lo[oo]);
            }
        }

        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2S256_NO; oo++) {
                simdgroup_store(lo[oo], sop, FA2S256_HD);
                sop += 8 * FA2S256_NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (uint jj = 0; jj < FA2S256_NQ; jj++) {
        uint j   = jj * FA2S256_NSG + simd_id;
        uint gqi = tile_q + j;
        if (gqi < n_tokens && S[jj] > 0.0f) {
            float inv_sum = 1.0f / S[jj];
            for (uint d = simd_lane; d < FA2S256_HD; d += FA2S256_NW) {
                O_buf[gqi * q_stride + h * FA2S256_HD + d] =
                    so[j * FA2S256_HD + d] * inv_sum;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FA2 v2: D-blocked mixed-precision prefill attention — head_dim=256
//
// Solves the HD=256 register pressure problem:
// - Current kernel: NO=8 output registers per SG → register spill
// - This kernel: D_BLOCK=64 → NO=2 per iteration, 4 D-block iterations
//
// Three-phase inner loop per KV tile:
// Phase A: Accumulate QK^T across 4 D-blocks (half(Q)×float(K)→float)
// Phase B: Online softmax in float, correct O accumulator
// Phase C: Accumulate S×V across 4 D-blocks (half(S)×float(V)→float)
//
// Key improvements over current hd256 kernel:
// 1. C=64 KV tokens per tile (vs C=32) — QK^T only needs D_BLOCK inner loop
// 2. NO=2 register pressure (vs NO=8) — eliminates register spill
// 3. Mixed precision MMA — half operand for higher throughput
//
// TG memory: sq(4KB half) + so(8KB float) + ss(2KB float) + sp(0.5KB half) = ~14.5KB
// Grid: (ceil(n_tokens / 8), n_heads) threadgroups.
// ═══════════════════════════════════════════════════════════════════════════

constant uint FA2V2_256_Q   = 8;
constant uint FA2V2_256_C   = 64;     // KV tokens per tile (doubled from 32!)
constant uint FA2V2_256_NSG = 4;
constant uint FA2V2_256_NW  = 32;
constant uint FA2V2_256_TG  = 128;
constant uint FA2V2_256_HD  = 256;
constant uint FA2V2_256_DB  = 64;     // D-block size
constant uint FA2V2_256_NDB = 4;      // HD / DB iterations

constant uint FA2V2_256_NQ  = FA2V2_256_Q / FA2V2_256_NSG;              // 2
constant uint FA2V2_256_NC  = (FA2V2_256_C / 8) / FA2V2_256_NSG;        // 2
constant uint FA2V2_256_NO  = (FA2V2_256_DB / 8) / FA2V2_256_NSG;       // 2

kernel void attention_prefill_fa2v2_hd256(
    device const float* Q_buf   [[buffer(0)]],
    device const float* K_buf   [[buffer(1)]],
    device const float* V_buf   [[buffer(2)]],
    device float* O_buf         [[buffer(3)]],
    constant uint& n_tokens     [[buffer(4)]],
    constant uint& n_heads      [[buffer(5)]],
    constant uint& n_kv_heads   [[buffer(6)]],
    constant uint& head_dim     [[buffer(7)]],
    uint2 tg_id       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2V2_256_Q;
    uint h      = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2V2_256_HD;
    uint kv_stride    = n_kv_heads * FA2V2_256_HD;
    constexpr float inv_sqrt_hd = 0.0625f; // 1/sqrt(256)

    // ── TG memory ────────────────────────────────────────────────
    threadgroup half  sq[FA2V2_256_Q * FA2V2_256_HD];     // 4 KB: Q (half, full HD)
    threadgroup float so[FA2V2_256_Q * FA2V2_256_HD];     // 8 KB: O accumulator (float)
    threadgroup float ss[FA2V2_256_Q * FA2V2_256_C];      // 2 KB: QK^T scores (float)
    threadgroup half  sp[FA2V2_256_Q * FA2V2_256_C];      // 1 KB: softmax probs (half)

    // Per-query online softmax state
    float S[FA2V2_256_NQ] = {};
    float M[FA2V2_256_NQ];
    for (uint i = 0; i < FA2V2_256_NQ; i++) M[i] = -INFINITY;

    // ── Load Q into TG as half (full HD, once) ───────────────────
    for (uint i = lid; i < FA2V2_256_Q * FA2V2_256_HD; i += FA2V2_256_TG) {
        uint qi = i / FA2V2_256_HD;
        uint d  = i % FA2V2_256_HD;
        uint gqi = tile_q + qi;
        sq[i] = (gqi < n_tokens)
            ? half(Q_buf[gqi * q_stride + h * FA2V2_256_HD + d])
            : half(0.0h);
    }
    for (uint i = lid; i < FA2V2_256_Q * FA2V2_256_HD; i += FA2V2_256_TG) {
        so[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint attend_range = min(tile_q + FA2V2_256_Q, n_tokens);

    // ── Main tile loop over KV sequence ──────────────────────────
    for (uint tile_kv = 0; tile_kv < attend_range; tile_kv += FA2V2_256_C) {
        uint kv_tile_len = min(uint(FA2V2_256_C), attend_range - tile_kv);

        // ── Phase A: QK^T across 4 D-blocks ─────────────────────
        // Each SG accumulates its block's QK^T across ALL D-blocks
        // in registers, then stores once.
        for (uint cc = 0; cc < FA2V2_256_NC; cc++) {
            uint block_idx = cc * FA2V2_256_NSG + simd_id;
            uint gkv_base  = tile_kv + block_idx * 8;

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            // Accumulate across all 4 D-blocks (full head_dim dot product)
            for (uint db = 0; db < FA2V2_256_NDB; db++) {
                uint d_off = db * FA2V2_256_DB;
                device const float* k_block = K_buf + gkv_base * kv_stride
                                                    + kv_h * FA2V2_256_HD + d_off;

                for (uint dk = 0; dk < FA2V2_256_DB; dk += 16) {
                    simdgroup_half8x8 mq0, mq1;
                    simdgroup_float8x8 mk0, mk1;
                    simdgroup_load(mq0, sq + d_off + dk,     FA2V2_256_HD);
                    simdgroup_load(mq1, sq + d_off + dk + 8, FA2V2_256_HD);
                    simdgroup_load(mk0, k_block + dk,     kv_stride, ulong2(0, 0), true);
                    simdgroup_load(mk1, k_block + dk + 8, kv_stride, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                    simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                }
            }

            // Store complete QK^T scores
            simdgroup_store(mqk, ss + block_idx * 8, FA2V2_256_C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase B: Online softmax (float) ─────────────────────
        for (uint jj = 0; jj < FA2V2_256_NQ; jj++) {
            uint j   = jj * FA2V2_256_NSG + simd_id;
            uint gqi = tile_q + j;
            float prev_max = M[jj];

            uint ki0 = simd_lane * 2;
            uint ki1 = ki0 + 1;
            uint gkv0 = tile_kv + ki0;
            uint gkv1 = tile_kv + ki1;

            float s0 = -INFINITY, s1 = -INFINITY;
            if (ki0 < kv_tile_len && gkv0 <= gqi) {
                s0 = ss[j * FA2V2_256_C + ki0] * inv_sqrt_hd;
            }
            if (ki1 < kv_tile_len && gkv1 <= gqi) {
                s1 = ss[j * FA2V2_256_C + ki1] * inv_sqrt_hd;
            }

            float new_max = simd_max(max(max(s0, s1), prev_max));
            float ms = ax_exp(prev_max - new_max);
            float e0 = ax_exp(s0 - new_max);
            float e1 = ax_exp(s1 - new_max);
            float tile_sum = simd_sum(e0 + e1);

            M[jj] = new_max;
            S[jj] = S[jj] * ms + tile_sum;

            sp[j * FA2V2_256_C + ki0] = half(e0);
            sp[j * FA2V2_256_C + ki1] = half(e1);

            // Correct O across full HD
            for (uint d = simd_lane; d < FA2V2_256_HD; d += FA2V2_256_NW) {
                so[j * FA2V2_256_HD + d] *= ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase C: O += S × V across 4 D-blocks ──────────────
        for (uint db = 0; db < FA2V2_256_NDB; db++) {
            uint d_off = db * FA2V2_256_DB;

            // Load O accumulator slice (only NO=2 registers!)
            simdgroup_float8x8 lo[FA2V2_256_NO];
            {
                threadgroup float* sop = so + d_off + 8 * simd_id;
                for (uint oo = 0; oo < FA2V2_256_NO; oo++) {
                    simdgroup_load(lo[oo], sop, FA2V2_256_HD);
                    sop += 8 * FA2V2_256_NSG;
                }
            }

            for (uint cc = 0; cc < FA2V2_256_C / 8; cc++) {
                simdgroup_half8x8 ms_mat;
                simdgroup_load(ms_mat, sp + 8 * cc, FA2V2_256_C);
                uint gkv_base = tile_kv + cc * 8;

                for (uint oo = 0; oo < FA2V2_256_NO; oo++) {
                    uint dv_block  = oo * FA2V2_256_NSG + simd_id;
                    uint dv_offset = kv_h * FA2V2_256_HD + d_off + dv_block * 8;

                    simdgroup_float8x8 mv;
                    simdgroup_load(mv, V_buf + gkv_base * kv_stride + dv_offset, kv_stride);
                    simdgroup_multiply_accumulate(lo[oo], ms_mat, mv, lo[oo]);
                }
            }

            // Store O slice back
            {
                threadgroup float* sop = so + d_off + 8 * simd_id;
                for (uint oo = 0; oo < FA2V2_256_NO; oo++) {
                    simdgroup_store(lo[oo], sop, FA2V2_256_HD);
                    sop += 8 * FA2V2_256_NSG;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Write final output ───────────────────────────────────────
    for (uint jj = 0; jj < FA2V2_256_NQ; jj++) {
        uint j   = jj * FA2V2_256_NSG + simd_id;
        uint gqi = tile_q + j;
        if (gqi < n_tokens) {
            float inv_sum = (S[jj] > 0.0f) ? (1.0f / S[jj]) : 0.0f;
            for (uint d = simd_lane; d < FA2V2_256_HD; d += FA2V2_256_NW)
                O_buf[gqi * q_stride + h * FA2V2_256_HD + d] =
                    so[j * FA2V2_256_HD + d] * inv_sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FA2 prefill with simdgroup matrix operations — head_dim=64
//
// Same architecture as HD=128 but for head_dim=64 (Qwen3-0.6B, Gemma etc.).
// FA2S64_NO=2 (half the output blocks), 4 QK^T iterations (not 8).
// TG memory: ~7 KB (vs ~13 KB for HD=128).
// ═══════════════════════════════════════════════════════════════════════════

constant uint FA2S64_Q   = 8;
constant uint FA2S64_C   = 64;
constant uint FA2S64_NSG = 4;
constant uint FA2S64_NW  = 32;
constant uint FA2S64_TG  = FA2S64_NW * FA2S64_NSG;
constant uint FA2S64_HD  = 64;

constant uint FA2S64_NQ  = FA2S64_Q / FA2S64_NSG;
constant uint FA2S64_NC  = (FA2S64_C / 8) / FA2S64_NSG;
constant uint FA2S64_NO  = (FA2S64_HD / 8) / FA2S64_NSG;  // 2

kernel void attention_prefill_fa2_simd_hd64(
    device const float* Q_buf   [[buffer(0)]],
    device const float* K_buf   [[buffer(1)]],
    device const float* V_buf   [[buffer(2)]],
    device float* O_buf         [[buffer(3)]],
    constant uint& n_tokens     [[buffer(4)]],
    constant uint& n_heads      [[buffer(5)]],
    constant uint& n_kv_heads   [[buffer(6)]],
    constant uint& head_dim     [[buffer(7)]],
    uint2 tg_id       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2S64_Q;
    uint h      = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2S64_HD;
    uint kv_stride    = n_kv_heads * FA2S64_HD;
    constexpr float inv_sqrt_hd = 0.125f; // 1/sqrt(64)

    threadgroup half  sq[FA2S64_Q * FA2S64_HD];
    threadgroup half  so[FA2S64_Q * FA2S64_HD];
    threadgroup half  ss[FA2S64_Q * FA2S64_C];
    threadgroup half  sk[FA2S64_NSG * 8 * FA2S64_HD];

    float S[FA2S64_NQ] = {};
    float M[FA2S64_NQ];
    for (uint i = 0; i < FA2S64_NQ; i++) M[i] = -INFINITY;

    for (uint i = lid; i < FA2S64_Q * FA2S64_HD; i += FA2S64_TG) {
        uint qi = i / FA2S64_HD;
        uint d  = i % FA2S64_HD;
        uint gqi = tile_q + qi;
        sq[i] = (gqi < n_tokens)
            ? half(Q_buf[gqi * q_stride + h * FA2S64_HD + d]) : half(0.0h);
    }
    for (uint i = lid; i < FA2S64_Q * FA2S64_HD; i += FA2S64_TG) {
        so[i] = half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint attend_range = min(tile_q + FA2S64_Q, n_tokens);

    for (uint tile_kv = 0; tile_kv < attend_range; tile_kv += FA2S64_C) {
        // QK^T
        for (uint cc = 0; cc < FA2S64_NC; cc++) {
            uint block_idx = cc * FA2S64_NSG + simd_id;
            uint gkv_base  = tile_kv + block_idx * 8;
            threadgroup half* my_sk = sk + simd_id * (8 * FA2S64_HD);
            for (uint i = simd_lane; i < 8 * FA2S64_HD; i += FA2S64_NW) {
                uint ki = i / FA2S64_HD;
                uint d  = i % FA2S64_HD;
                uint gkv = gkv_base + ki;
                my_sk[i] = (gkv < attend_range)
                    ? half(K_buf[gkv * kv_stride + kv_h * FA2S64_HD + d]) : half(0.0h);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
            for (uint dk = 0; dk < FA2S64_HD; dk += 16) {
                simdgroup_half8x8 mq0, mq1, mk0, mk1;
                simdgroup_load(mq0, sq + dk,     FA2S64_HD);
                simdgroup_load(mq1, sq + dk + 8, FA2S64_HD);
                simdgroup_load(mk0, my_sk + dk,     FA2S64_HD, ulong2(0, 0), true);
                simdgroup_load(mk1, my_sk + dk + 8, FA2S64_HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }
            simdgroup_store(mqk, ss + block_idx * 8, FA2S64_C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax
        for (uint jj = 0; jj < FA2S64_NQ; jj++) {
            uint j   = jj * FA2S64_NSG + simd_id;
            uint gqi = tile_q + j;
            float prev_max = M[jj];
            uint ki0 = simd_lane * 2;
            uint ki1 = ki0 + 1;
            float s0 = -INFINITY, s1 = -INFINITY;
            if (ki0 < min(uint(FA2S64_C), attend_range - tile_kv) && tile_kv + ki0 <= gqi)
                s0 = float(ss[j * FA2S64_C + ki0]) * inv_sqrt_hd;
            if (ki1 < min(uint(FA2S64_C), attend_range - tile_kv) && tile_kv + ki1 <= gqi)
                s1 = float(ss[j * FA2S64_C + ki1]) * inv_sqrt_hd;
            float new_max = simd_max(max(max(s0, s1), prev_max));
            float ms = ax_exp(prev_max - new_max);
            float e0 = ax_exp(s0 - new_max);
            float e1 = ax_exp(s1 - new_max);
            M[jj] = new_max;
            S[jj] = S[jj] * ms + simd_sum(e0 + e1);
            ss[j * FA2S64_C + ki0] = half(e0);
            ss[j * FA2S64_C + ki1] = half(e1);
            for (uint d = simd_lane; d < FA2S64_HD; d += FA2S64_NW)
                so[j * FA2S64_HD + d] = half(float(so[j * FA2S64_HD + d]) * ms);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += S × V
        simdgroup_half8x8 lo[FA2S64_NO];
        {
            threadgroup half* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2S64_NO; oo++) {
                simdgroup_load(lo[oo], sop, FA2S64_HD);
                sop += 8 * FA2S64_NSG;
            }
        }
        for (uint cc = 0; cc < FA2S64_C / 8; cc++) {
            simdgroup_half8x8 ms_mat;
            simdgroup_load(ms_mat, ss + 8 * cc, FA2S64_C);
            for (uint oo = 0; oo < FA2S64_NO; oo++) {
                uint dv_block  = oo * FA2S64_NSG + simd_id;
                uint dv_offset = kv_h * FA2S64_HD + dv_block * 8;
                uint gkv_base  = tile_kv + cc * 8;
                threadgroup half* my_sv = sk + simd_id * 64;
                for (uint i = simd_lane; i < 64; i += FA2S64_NW) {
                    uint vi = i / 8, d = i % 8;
                    uint gkv = gkv_base + vi;
                    my_sv[i] = (gkv < attend_range)
                        ? half(V_buf[gkv * kv_stride + dv_offset + d]) : half(0.0h);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_half8x8 mv;
                simdgroup_load(mv, my_sv, 8);
                simdgroup_multiply_accumulate(lo[oo], ms_mat, mv, lo[oo]);
            }
        }
        {
            threadgroup half* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2S64_NO; oo++) {
                simdgroup_store(lo[oo], sop, FA2S64_HD);
                sop += 8 * FA2S64_NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint jj = 0; jj < FA2S64_NQ; jj++) {
        uint j   = jj * FA2S64_NSG + simd_id;
        uint gqi = tile_q + j;
        if (gqi < n_tokens && S[jj] > 0.0f) {
            float inv_sum = 1.0f / S[jj];
            for (uint d = simd_lane; d < FA2S64_HD; d += FA2S64_NW)
                O_buf[gqi * q_stride + h * FA2S64_HD + d] =
                    float(so[j * FA2S64_HD + d]) * inv_sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MLX-style parallel sdpa_vector decode attention (HD=128)
//
// 32 simdgroups × 32 lanes = 1024 threads per TG.
// Each simdgroup independently processes a disjoint slice of KV sequence.
// Cross-simdgroup merge uses transpose trick + simd_sum for O(1) reduction.
//
// Based on the MLX sdpa_vector parallel merge pattern.
// Grid: (1, n_heads, 1) threadgroups.
// ═══════════════════════════════════════════════════════════════════════════

template <int HD>
[[kernel]] void attention_decode_sdpa_parallel(
    device const float* Q          [[buffer(0)]],
    device const float* K          [[buffer(1)]],
    device const float* V          [[buffer(2)]],
    device float* O                [[buffer(3)]],
    constant uint& n_heads         [[buffer(4)]],
    constant uint& n_kv_heads      [[buffer(5)]],
    constant uint& seq_len         [[buffer(6)]],
    uint2 tg_id                    [[threadgroup_position_in_grid]],
    uint simd_gid                  [[simdgroup_index_in_threadgroup]],
    uint simd_lid                  [[thread_index_in_simdgroup]]
) {
    constexpr int BN = 32;          // simdgroups per TG
    constexpr int BD = 32;          // SIMD width
    constexpr int EPT = HD / BD;    // elements per thread
    // rsqrt(HD) — precomputed per specialization
    const float scale = (HD == 128) ? 0.08838834764831843f
                      : (HD == 256) ? 0.0625f
                      : 1.0f / sqrt(float(HD));

    // Shared memory for cross-simdgroup merge
    threadgroup float tg_outputs[BN * BD];
    threadgroup float tg_max[BN];
    threadgroup float tg_sum[BN];

    uint h = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint N = seq_len;

    // Per-thread registers
    float q[EPT];
    float o[EPT];
    for (int i = 0; i < EPT; i++) o[i] = 0.0f;

    // Load query (each thread loads EPT elements)
    device const float* q_ptr = Q + h * HD + simd_lid * EPT;
    for (int i = 0; i < EPT; i++) {
        q[i] = q_ptr[i] * scale;
    }

    // KV base pointers — stride by n_kv_heads * HD per position
    uint kv_stride = n_kv_heads * HD;
    device const float* k_base = K + kv_h * HD + simd_lid * EPT;
    device const float* v_base = V + kv_h * HD + simd_lid * EPT;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Phase 1: Each simdgroup processes KV positions i = simd_gid, simd_gid+BN, ...
    for (uint i = simd_gid; i < N; i += BN) {
        device const float* k_ptr = k_base + i * kv_stride;
        device const float* v_ptr = v_base + i * kv_stride;

        // Q·K dot product across all lanes
        float score = 0.0f;
        for (int j = 0; j < EPT; j++) {
            score += q[j] * k_ptr[j];
        }
        score = simd_sum(score);

        // Online softmax update
        float new_max = max(max_score, score);
        float factor = ax_exp(max_score - new_max);
        float exp_score = ax_exp(score - new_max);

        // Rescale accumulated output and add new weighted V
        for (int j = 0; j < EPT; j++) {
            o[j] = o[j] * factor + exp_score * v_ptr[j];
        }
        sum_exp = sum_exp * factor + exp_score;
        max_score = new_max;
    }

    // Phase 2: Cross-simdgroup merge via transpose trick
    // Step 2a: Share max and sum across simdgroups
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel max reduction: each lane reads one simdgroup's max
    float sg_max = (simd_lid < BN) ? tg_max[simd_lid] : -INFINITY;
    float global_max = simd_max(sg_max);
    float factor = (simd_lid < BN) ? ax_exp(tg_max[simd_lid] - global_max) : 0.0f;

    // My simdgroup's correction factor
    float my_factor = ax_exp(max_score - global_max);

    // Parallel sum reduction
    float corrected_sum = (simd_lid < BN) ? tg_sum[simd_lid] * factor : 0.0f;
    float global_sum = simd_sum(corrected_sum);

    // Step 2b: Aggregate outputs via transpose
    for (int i = 0; i < EPT; i++) {
        // Write transposed: each simdgroup writes to column simd_gid
        tg_outputs[simd_lid * BD + simd_gid] = o[i] * my_factor;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Read transposed: each lane reads from row simd_gid
        float val = tg_outputs[simd_gid * BD + simd_lid];
        o[i] = simd_sum(val) / global_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output (only lane 0 of each simdgroup writes its EPT elements)
    if (simd_lid == 0) {
        device float* o_ptr = O + h * HD + simd_gid * EPT;
        for (int i = 0; i < EPT; i++) {
            o_ptr[i] = o[i];
        }
    }
}

// f32 KV variants
template [[host_name("attention_decode_sdpa_parallel_hd128")]]
[[kernel]] void attention_decode_sdpa_parallel<128>(
    device const float*, device const float*, device const float*,
    device float*, constant uint&, constant uint&, constant uint&,
    uint2, uint, uint);

template [[host_name("attention_decode_sdpa_parallel_hd256")]]
[[kernel]] void attention_decode_sdpa_parallel<256>(
    device const float*, device const float*, device const float*,
    device float*, constant uint&, constant uint&, constant uint&,
    uint2, uint, uint);

// ═══════════════════════════════════════════════════════════════════════════
// f16 KV variant — same algorithm, reads K/V as half and converts to float.
// ═══════════════════════════════════════════════════════════════════════════

template <int HD>
[[kernel]] void attention_decode_sdpa_parallel_f16kv(
    device const float* Q          [[buffer(0)]],
    device const half*  K          [[buffer(1)]],
    device const half*  V          [[buffer(2)]],
    device float* O                [[buffer(3)]],
    constant uint& n_heads         [[buffer(4)]],
    constant uint& n_kv_heads      [[buffer(5)]],
    constant uint& seq_len         [[buffer(6)]],
    uint2 tg_id                    [[threadgroup_position_in_grid]],
    uint simd_gid                  [[simdgroup_index_in_threadgroup]],
    uint simd_lid                  [[thread_index_in_simdgroup]]
) {
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int EPT = HD / BD;
    const float scale = (HD == 128) ? 0.08838834764831843f
                      : (HD == 256) ? 0.0625f
                      : 1.0f / sqrt(float(HD));

    threadgroup float tg_outputs[BN * BD];
    threadgroup float tg_max[BN];
    threadgroup float tg_sum[BN];

    uint h = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint N = seq_len;

    float q[EPT];
    float o[EPT];
    for (int i = 0; i < EPT; i++) o[i] = 0.0f;

    device const float* q_ptr = Q + h * HD + simd_lid * EPT;
    for (int i = 0; i < EPT; i++) q[i] = q_ptr[i] * scale;

    uint kv_stride = n_kv_heads * HD;
    device const half* k_base = K + kv_h * HD + simd_lid * EPT;
    device const half* v_base = V + kv_h * HD + simd_lid * EPT;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    for (uint i = simd_gid; i < N; i += BN) {
        device const half* k_ptr = k_base + i * kv_stride;
        device const half* v_ptr = v_base + i * kv_stride;

        float score = 0.0f;
        for (int j = 0; j < EPT; j++) {
            score += q[j] * float(k_ptr[j]);
        }
        score = simd_sum(score);

        float new_max = max(max_score, score);
        float factor = ax_exp(max_score - new_max);
        float exp_score = ax_exp(score - new_max);

        for (int j = 0; j < EPT; j++) {
            o[j] = o[j] * factor + exp_score * float(v_ptr[j]);
        }
        sum_exp = sum_exp * factor + exp_score;
        max_score = new_max;
    }

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sg_max = (simd_lid < BN) ? tg_max[simd_lid] : -INFINITY;
    float global_max = simd_max(sg_max);
    float factor = (simd_lid < BN) ? ax_exp(tg_max[simd_lid] - global_max) : 0.0f;
    float my_factor = ax_exp(max_score - global_max);
    float corrected_sum = (simd_lid < BN) ? tg_sum[simd_lid] * factor : 0.0f;
    float global_sum = simd_sum(corrected_sum);

    for (int i = 0; i < EPT; i++) {
        tg_outputs[simd_lid * BD + simd_gid] = o[i] * my_factor;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float val = tg_outputs[simd_gid * BD + simd_lid];
        o[i] = simd_sum(val) / global_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device float* o_ptr = O + h * HD + simd_gid * EPT;
        for (int i = 0; i < EPT; i++) o_ptr[i] = o[i];
    }
}

template [[host_name("attention_decode_sdpa_parallel_f16kv_hd128")]]
[[kernel]] void attention_decode_sdpa_parallel_f16kv<128>(
    device const float*, device const half*, device const half*,
    device float*, constant uint&, constant uint&, constant uint&,
    uint2, uint, uint);

template [[host_name("attention_decode_sdpa_parallel_f16kv_hd256")]]
[[kernel]] void attention_decode_sdpa_parallel_f16kv<256>(
    device const float*, device const half*, device const half*,
    device float*, constant uint&, constant uint&, constant uint&,
    uint2, uint, uint);

// ═══════════════════════════════════════════════════════════════════════════
// GQA-aware sdpa_parallel: multiple query heads per TG, sharing KV reads.
//
// For GQA models (e.g. Llama 3: 32 query heads / 8 KV heads = ratio 4),
// all query heads sharing a KV head are processed in the SAME threadgroup.
// This reduces KV memory reads by the GQA ratio (4× for Llama 3).
//
// Grid: (1, n_kv_heads, 1) — one TG per KV head.
// TG: 1024 threads (32 simdgroups).
// Simdgroups partitioned: 32/gqa_ratio SGs per query head.
// For GQA=4: 8 SGs per query, 4 query heads per TG.
// For GQA=1 (MHA): equivalent to sdpa_parallel (32 SGs, 1 query).
// ═══════════════════════════════════════════════════════════════════════════

template <int HD>
[[kernel]] void attention_decode_sdpa_gqa_f16kv(
    device const float* Q          [[buffer(0)]],
    device const half*  K          [[buffer(1)]],
    device const half*  V          [[buffer(2)]],
    device float* O                [[buffer(3)]],
    constant uint& n_heads         [[buffer(4)]],
    constant uint& n_kv_heads      [[buffer(5)]],
    constant uint& seq_len         [[buffer(6)]],
    uint2 tg_id                    [[threadgroup_position_in_grid]],
    uint simd_gid                  [[simdgroup_index_in_threadgroup]],
    uint simd_lid                  [[thread_index_in_simdgroup]]
) {
    constexpr int BN = 32;          // total simdgroups per TG
    constexpr int BD = 32;          // SIMD width
    constexpr int EPT = HD / BD;    // elements per thread
    const float scale = (HD == 128) ? 0.08838834764831843f
                      : (HD == 256) ? 0.0625f
                      : 1.0f / sqrt(float(HD));

    uint kv_h = tg_id.y;
    if (kv_h >= n_kv_heads) return;

    uint gqa_ratio = n_heads / n_kv_heads;
    uint sgs_per_q = BN / gqa_ratio;     // SGs dedicated to each query head
    uint local_q = simd_gid / sgs_per_q; // which query head within this TG
    uint local_sg = simd_gid % sgs_per_q;// which SG within the query group
    uint h = kv_h * gqa_ratio + local_q; // global query head index

    if (h >= n_heads) return;

    uint N = seq_len;

    // Shared merge buffers — partitioned by query head
    threadgroup float tg_outputs[BN * BD];
    threadgroup float tg_max[BN];
    threadgroup float tg_sum[BN];

    // Load query (each thread loads EPT elements of its query head)
    float q[EPT];
    float o[EPT];
    for (int i = 0; i < EPT; i++) o[i] = 0.0f;

    device const float* q_ptr = Q + h * HD + simd_lid * EPT;
    for (int i = 0; i < EPT; i++) q[i] = q_ptr[i] * scale;

    // KV base — same for all query heads in this TG
    uint kv_stride = n_kv_heads * HD;
    device const half* k_base = K + kv_h * HD + simd_lid * EPT;
    device const half* v_base = V + kv_h * HD + simd_lid * EPT;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // KV loop: each query group iterates with step = sgs_per_q
    // All query groups read the SAME KV data → SLC cache reuse
    for (uint i = local_sg; i < N; i += sgs_per_q) {
        device const half* k_ptr = k_base + i * kv_stride;
        device const half* v_ptr = v_base + i * kv_stride;

        float score = 0.0f;
        for (int j = 0; j < EPT; j++) {
            score += q[j] * float(k_ptr[j]);
        }
        score = simd_sum(score);

        float new_max = max(max_score, score);
        float factor = ax_exp(max_score - new_max);
        float exp_score = ax_exp(score - new_max);

        for (int j = 0; j < EPT; j++) {
            o[j] = o[j] * factor + exp_score * float(v_ptr[j]);
        }
        sum_exp = sum_exp * factor + exp_score;
        max_score = new_max;
    }

    // Cross-SG merge: each query group merges independently
    // Use the tg_max/tg_sum/tg_outputs arrays with per-query offsets
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel max/sum within this query group (sgs_per_q SGs)
    // Read this query's SG range: [local_q * sgs_per_q .. (local_q+1) * sgs_per_q)
    float sg_max = -INFINITY;
    if (simd_lid < sgs_per_q) {
        sg_max = tg_max[local_q * sgs_per_q + simd_lid];
    }
    float global_max = simd_max(sg_max);
    float my_factor = ax_exp(max_score - global_max);

    float corrected_sum = 0.0f;
    if (simd_lid < sgs_per_q) {
        corrected_sum = tg_sum[local_q * sgs_per_q + simd_lid]
                      * ax_exp(tg_max[local_q * sgs_per_q + simd_lid] - global_max);
    }
    float global_sum = simd_sum(corrected_sum);

    // Transpose merge within the query group's SG range
    // Use a slice of tg_outputs for this query: [local_q * sgs_per_q * BD ..]
    uint out_base = local_q * sgs_per_q * BD;
    for (int i = 0; i < EPT; i++) {
        if (simd_lid < sgs_per_q) {
            tg_outputs[out_base + simd_lid * BD + local_sg] = o[i] * my_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float val = 0.0f;
        if (simd_lid < sgs_per_q) {
            val = tg_outputs[out_base + local_sg * BD + simd_lid];
        }
        o[i] = simd_sum(val) / global_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output — lane 0 of each query group writes
    if (simd_lid == 0) {
        device float* o_ptr = O + h * HD + local_sg * EPT;
        for (int i = 0; i < EPT; i++) {
            o_ptr[i] = o[i];
        }
    }
}

template [[host_name("attention_decode_sdpa_gqa_f16kv_hd128")]]
[[kernel]] void attention_decode_sdpa_gqa_f16kv<128>(
    device const float*, device const half*, device const half*,
    device float*, constant uint&, constant uint&, constant uint&,
    uint2, uint, uint);

template [[host_name("attention_decode_sdpa_gqa_f16kv_hd256")]]
[[kernel]] void attention_decode_sdpa_gqa_f16kv<256>(
    device const float*, device const half*, device const half*,
    device float*, constant uint&, constant uint&, constant uint&,
    uint2, uint, uint);

// ═══════════════════════════════════════════════════════════════════════════
// FA2 SIMD cached prefill — f16 KV cache, head_dim=128
//
// Extends FA2 SIMD HD128 to the cached prefill path (multi-turn continuation).
// Key additions vs local FA2 SIMD:
//   - Reads K/V from f16 KV cache with base_seq_len offset
//   - Causal mask uses absolute positions: gkv <= base_seq_len + gqi
//   - Sliding window: per-query attend_start check
//   - Mask tile-skip: starts KV loop at min_attend_start (rounded to tile),
//     skipping tiles fully outside all 8 queries' attend windows
//   - K/V staged through threadgroup half scratch (simdgroup_half8x8 matmul)
//
// Grid: (ceil(n_tokens / 8), n_heads) × 128 threads
// TG memory: ~14 KB (sq 4KB + so 4KB + ss 2KB + sk 4KB)
// ═══════════════════════════════════════════════════════════════════════════

constant uint FA2SC_Q   = 8;       // queries per threadgroup
constant uint FA2SC_C   = 64;      // KV tokens per tile
constant uint FA2SC_NSG = 4;       // simdgroups per threadgroup
constant uint FA2SC_NW  = 32;      // threads per simdgroup
constant uint FA2SC_TG  = 128;     // total threads
constant uint FA2SC_HD  = 128;     // constexpr head_dim

constant uint FA2SC_NQ  = 2;       // queries per SG (softmax)  = Q / NSG
constant uint FA2SC_NC  = 2;       // KV-blocks per SG (QK^T)   = C / 8 / NSG
constant uint FA2SC_NO  = 4;       // output-blocks per SG (OV) = HD / 8 / NSG

kernel void attention_prefill_fa2_simd_cached_f16kv_hd128(
    device const float* Q_buf       [[buffer(0)]],
    device const half*  K_cache     [[buffer(1)]],
    device const half*  V_cache     [[buffer(2)]],
    device float*       O_buf       [[buffer(3)]],
    constant uint& n_tokens         [[buffer(4)]],
    constant uint& n_heads          [[buffer(5)]],
    constant uint& n_kv_heads       [[buffer(6)]],
    constant uint& head_dim         [[buffer(7)]],
    constant uint& base_seq_len     [[buffer(8)]],
    constant uint& sliding_window   [[buffer(9)]],
    uint2 tg_id       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2SC_Q;
    uint h      = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2SC_HD;
    uint kv_stride    = n_kv_heads * FA2SC_HD;
    constexpr float inv_sqrt_hd = 0.08838834764831843f; // 1/sqrt(128)

    // ── Threadgroup memory ───────────────────────────────────────────
    threadgroup float sq[FA2SC_Q * FA2SC_HD];               // 4 KB: Q tile (float)
    threadgroup float so[FA2SC_Q * FA2SC_HD];               // 4 KB: O accumulator (float)
    threadgroup float ss[FA2SC_Q * FA2SC_C];                // 2 KB: scores (float)
    threadgroup half  sk[FA2SC_NSG * 8 * FA2SC_HD];         // 4 KB: K/V staging (half)

    // Per-query online softmax state (register, per simdgroup)
    float S[FA2SC_NQ] = {};
    float M[FA2SC_NQ];
    for (uint i = 0; i < FA2SC_NQ; i++) M[i] = -INFINITY;

    // ── Compute attend range with mask tile-skip ─────────────────────
    // First and last query absolute positions in this tile
    uint min_qi = tile_q;
    uint max_qi = min(tile_q + FA2SC_Q - 1, n_tokens - 1);
    uint min_qi_abs = base_seq_len + min_qi;
    uint max_qi_abs = base_seq_len + max_qi;

    // max_attend_end: last KV position any query in this tile can see + 1
    uint max_attend_end = max_qi_abs + 1;

    // min_attend_start: earliest KV position the most restrictive query needs
    // (for sliding window, the first query has the tightest window start)
    uint min_attend_start = 0;
    if (sliding_window > 0 && (min_qi_abs + 1) > sliding_window) {
        min_attend_start = (min_qi_abs + 1) - sliding_window;
    }

    // Round min_attend_start down to tile boundary for aligned iteration
    uint tile_start = (min_attend_start / FA2SC_C) * FA2SC_C;

    // ── Load Q into threadgroup as float ─────────────────────────────
    for (uint i = lid; i < FA2SC_Q * FA2SC_HD; i += FA2SC_TG) {
        uint qi = i / FA2SC_HD;
        uint d  = i % FA2SC_HD;
        uint gqi = tile_q + qi;
        sq[i] = (gqi < n_tokens)
            ? Q_buf[gqi * q_stride + h * FA2SC_HD + d]
            : 0.0f;
    }

    // Zero O accumulator
    for (uint i = lid; i < FA2SC_Q * FA2SC_HD; i += FA2SC_TG) {
        so[i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Main tile loop over KV cache ─────────────────────────────────
    for (uint tile_kv = tile_start; tile_kv < max_attend_end; tile_kv += FA2SC_C) {
        uint kv_tile_len = min(uint(FA2SC_C), max_attend_end - tile_kv);

        // ── Phase 1: QK^T via simdgroup_half8x8 ─────────────────────
        // Stage K from f16 cache into threadgroup half scratch, then matmul.
        for (uint cc = 0; cc < FA2SC_NC; cc++) {
            uint block_idx = cc * FA2SC_NSG + simd_id;
            uint gkv_base  = tile_kv + block_idx * 8;

            // Stage 8 K rows from f16 cache into per-SG scratch
            threadgroup half* my_sk = sk + simd_id * (8 * FA2SC_HD);
            for (uint i = simd_lane; i < 8 * FA2SC_HD; i += FA2SC_NW) {
                uint ki = i / FA2SC_HD;
                uint d  = i % FA2SC_HD;
                uint gkv = gkv_base + ki;
                my_sk[i] = (gkv < max_attend_end)
                    ? K_cache[gkv * kv_stride + kv_h * FA2SC_HD + d]
                    : half(0.0h);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Q (float TG) × K^T (half scratch) → scores (float)
            // Use simdgroup_float8x8 for Q, load K as half→float promotion
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            for (uint dk = 0; dk < FA2SC_HD; dk += 16) {
                simdgroup_float8x8 mq0, mq1;
                simdgroup_half8x8 mk0, mk1;
                // Q from threadgroup (float)
                simdgroup_load(mq0, sq + dk,     FA2SC_HD);
                simdgroup_load(mq1, sq + dk + 8, FA2SC_HD);
                // K from half scratch, transposed
                simdgroup_load(mk0, my_sk + dk,     FA2SC_HD, ulong2(0, 0), true);
                simdgroup_load(mk1, my_sk + dk + 8, FA2SC_HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }

            simdgroup_store(mqk, ss + block_idx * 8, FA2SC_C);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 2: Online softmax with causal + sliding window mask ─
        for (uint jj = 0; jj < FA2SC_NQ; jj++) {
            uint j   = jj * FA2SC_NSG + simd_id;
            uint gqi = tile_q + j;
            uint gqi_abs = base_seq_len + gqi;
            float prev_max = M[jj];

            // Per-query sliding window start
            uint qi_attend_start = 0;
            if (sliding_window > 0 && (gqi_abs + 1) > sliding_window) {
                qi_attend_start = (gqi_abs + 1) - sliding_window;
            }

            uint ki0 = simd_lane * 2;
            uint ki1 = ki0 + 1;
            uint gkv0 = tile_kv + ki0;
            uint gkv1 = tile_kv + ki1;

            // Causal mask: gkv <= gqi_abs
            // Sliding window mask: gkv >= qi_attend_start
            float s0 = -INFINITY, s1 = -INFINITY;
            if (ki0 < kv_tile_len && gkv0 <= gqi_abs && gkv0 >= qi_attend_start) {
                s0 = ss[j * FA2SC_C + ki0] * inv_sqrt_hd;
            }
            if (ki1 < kv_tile_len && gkv1 <= gqi_abs && gkv1 >= qi_attend_start) {
                s1 = ss[j * FA2SC_C + ki1] * inv_sqrt_hd;
            }

            float new_max = simd_max(max(max(s0, s1), prev_max));
            float ms = ax_exp(prev_max - new_max);
            float e0 = ax_exp(s0 - new_max);
            float e1 = ax_exp(s1 - new_max);
            float tile_sum = simd_sum(e0 + e1);

            M[jj] = new_max;
            S[jj] = S[jj] * ms + tile_sum;

            // Store softmax probabilities
            ss[j * FA2SC_C + ki0] = e0;
            ss[j * FA2SC_C + ki1] = e1;

            // Correct O accumulator: so[j, :] *= ms
            for (uint d = simd_lane; d < FA2SC_HD; d += FA2SC_NW) {
                so[j * FA2SC_HD + d] *= ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 3: O += S × V via simdgroup_float8x8 ──────────────
        // Load O accumulators into simdgroup registers
        simdgroup_float8x8 lo[FA2SC_NO];
        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2SC_NO; oo++) {
                simdgroup_load(lo[oo], sop, FA2SC_HD);
                sop += 8 * FA2SC_NSG;
            }
        }

        for (uint cc = 0; cc < FA2SC_C / 8; cc++) {
            simdgroup_float8x8 ms_mat;
            simdgroup_load(ms_mat, ss + 8 * cc, FA2SC_C);
            uint gkv_base = tile_kv + cc * 8;

            for (uint oo = 0; oo < FA2SC_NO; oo++) {
                uint dv_block  = oo * FA2SC_NSG + simd_id;
                uint dv_offset = kv_h * FA2SC_HD + dv_block * 8;

                // Stage V from f16 cache into per-SG scratch
                threadgroup half* my_sv = sk + simd_id * 64;
                for (uint i = simd_lane; i < 64; i += FA2SC_NW) {
                    uint vi = i / 8, d = i % 8;
                    uint gkv = gkv_base + vi;
                    my_sv[i] = (gkv < max_attend_end)
                        ? V_cache[gkv * kv_stride + dv_offset + d]
                        : half(0.0h);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // V from half scratch → float matmul
                simdgroup_half8x8 mv;
                simdgroup_load(mv, my_sv, 8);
                simdgroup_multiply_accumulate(lo[oo], ms_mat, mv, lo[oo]);
            }
        }

        // Store O accumulators back to threadgroup
        {
            threadgroup float* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2SC_NO; oo++) {
                simdgroup_store(lo[oo], sop, FA2SC_HD);
                sop += 8 * FA2SC_NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write final output ───────────────────────────────────────────
    for (uint jj = 0; jj < FA2SC_NQ; jj++) {
        uint j   = jj * FA2SC_NSG + simd_id;
        uint gqi = tile_q + j;
        if (gqi < n_tokens && S[jj] > 0.0f) {
            float inv_sum = 1.0f / S[jj];
            for (uint d = simd_lane; d < FA2SC_HD; d += FA2SC_NW) {
                O_buf[gqi * q_stride + h * FA2SC_HD + d] =
                    so[j * FA2SC_HD + d] * inv_sum;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FA2 SIMD cached prefill — f16 KV cache, head_dim=64
//
// Same architecture as cached HD=128 but for head_dim=64.
// Grid: (ceil(n_tokens / 8), n_heads) × 128 threads
// TG memory: ~7 KB (sq 1KB + so 1KB + ss 2KB + sk 2KB + sv 256B)
// ═══════════════════════════════════════════════════════════════════════════

constant uint FA2SC64_Q   = 8;
constant uint FA2SC64_C   = 64;
constant uint FA2SC64_NSG = 4;
constant uint FA2SC64_NW  = 32;
constant uint FA2SC64_TG  = 128;
constant uint FA2SC64_HD  = 64;

constant uint FA2SC64_NQ  = 2;      // Q / NSG
constant uint FA2SC64_NC  = 2;      // C / 8 / NSG
constant uint FA2SC64_NO  = 2;      // HD / 8 / NSG

kernel void attention_prefill_fa2_simd_cached_f16kv_hd64(
    device const float* Q_buf       [[buffer(0)]],
    device const half*  K_cache     [[buffer(1)]],
    device const half*  V_cache     [[buffer(2)]],
    device float*       O_buf       [[buffer(3)]],
    constant uint& n_tokens         [[buffer(4)]],
    constant uint& n_heads          [[buffer(5)]],
    constant uint& n_kv_heads       [[buffer(6)]],
    constant uint& head_dim         [[buffer(7)]],
    constant uint& base_seq_len     [[buffer(8)]],
    constant uint& sliding_window   [[buffer(9)]],
    uint2 tg_id       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2SC64_Q;
    uint h      = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2SC64_HD;
    uint kv_stride    = n_kv_heads * FA2SC64_HD;
    constexpr float inv_sqrt_hd = 0.125f; // 1/sqrt(64)

    threadgroup half  sq[FA2SC64_Q * FA2SC64_HD];               // 1 KB
    threadgroup half  so[FA2SC64_Q * FA2SC64_HD];               // 1 KB
    threadgroup half  ss[FA2SC64_Q * FA2SC64_C];                // 1 KB
    threadgroup half  sk[FA2SC64_NSG * 8 * FA2SC64_HD];         // 2 KB

    float S[FA2SC64_NQ] = {};
    float M[FA2SC64_NQ];
    for (uint i = 0; i < FA2SC64_NQ; i++) M[i] = -INFINITY;

    // ── Attend range with tile-skip ──────────────────────────────────
    uint min_qi = tile_q;
    uint max_qi = min(tile_q + FA2SC64_Q - 1, n_tokens - 1);
    uint min_qi_abs = base_seq_len + min_qi;
    uint max_qi_abs = base_seq_len + max_qi;
    uint max_attend_end = max_qi_abs + 1;
    uint min_attend_start = 0;
    if (sliding_window > 0 && (min_qi_abs + 1) > sliding_window) {
        min_attend_start = (min_qi_abs + 1) - sliding_window;
    }
    uint tile_start = (min_attend_start / FA2SC64_C) * FA2SC64_C;

    // ── Load Q as half ───────────────────────────────────────────────
    for (uint i = lid; i < FA2SC64_Q * FA2SC64_HD; i += FA2SC64_TG) {
        uint qi = i / FA2SC64_HD;
        uint d  = i % FA2SC64_HD;
        uint gqi = tile_q + qi;
        sq[i] = (gqi < n_tokens)
            ? half(Q_buf[gqi * q_stride + h * FA2SC64_HD + d]) : half(0.0h);
    }
    for (uint i = lid; i < FA2SC64_Q * FA2SC64_HD; i += FA2SC64_TG) {
        so[i] = half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Main KV tile loop ────────────────────────────────────────────
    for (uint tile_kv = tile_start; tile_kv < max_attend_end; tile_kv += FA2SC64_C) {

        // QK^T: stage K from f16 cache, use simdgroup_half8x8
        for (uint cc = 0; cc < FA2SC64_NC; cc++) {
            uint block_idx = cc * FA2SC64_NSG + simd_id;
            uint gkv_base  = tile_kv + block_idx * 8;
            threadgroup half* my_sk = sk + simd_id * (8 * FA2SC64_HD);
            for (uint i = simd_lane; i < 8 * FA2SC64_HD; i += FA2SC64_NW) {
                uint ki = i / FA2SC64_HD;
                uint d  = i % FA2SC64_HD;
                uint gkv = gkv_base + ki;
                my_sk[i] = (gkv < max_attend_end)
                    ? K_cache[gkv * kv_stride + kv_h * FA2SC64_HD + d]
                    : half(0.0h);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
            for (uint dk = 0; dk < FA2SC64_HD; dk += 16) {
                simdgroup_half8x8 mq0, mq1, mk0, mk1;
                simdgroup_load(mq0, sq + dk,     FA2SC64_HD);
                simdgroup_load(mq1, sq + dk + 8, FA2SC64_HD);
                simdgroup_load(mk0, my_sk + dk,     FA2SC64_HD, ulong2(0, 0), true);
                simdgroup_load(mk1, my_sk + dk + 8, FA2SC64_HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }
            simdgroup_store(mqk, ss + block_idx * 8, FA2SC64_C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax with causal + sliding window
        for (uint jj = 0; jj < FA2SC64_NQ; jj++) {
            uint j   = jj * FA2SC64_NSG + simd_id;
            uint gqi = tile_q + j;
            uint gqi_abs = base_seq_len + gqi;
            float prev_max = M[jj];

            uint qi_attend_start = 0;
            if (sliding_window > 0 && (gqi_abs + 1) > sliding_window) {
                qi_attend_start = (gqi_abs + 1) - sliding_window;
            }

            uint ki0 = simd_lane * 2;
            uint ki1 = ki0 + 1;
            uint gkv0 = tile_kv + ki0;
            uint gkv1 = tile_kv + ki1;
            uint kv_tile_len = min(uint(FA2SC64_C), max_attend_end - tile_kv);

            float s0 = -INFINITY, s1 = -INFINITY;
            if (ki0 < kv_tile_len && gkv0 <= gqi_abs && gkv0 >= qi_attend_start)
                s0 = float(ss[j * FA2SC64_C + ki0]) * inv_sqrt_hd;
            if (ki1 < kv_tile_len && gkv1 <= gqi_abs && gkv1 >= qi_attend_start)
                s1 = float(ss[j * FA2SC64_C + ki1]) * inv_sqrt_hd;

            float new_max = simd_max(max(max(s0, s1), prev_max));
            float ms = ax_exp(prev_max - new_max);
            float e0 = ax_exp(s0 - new_max);
            float e1 = ax_exp(s1 - new_max);
            M[jj] = new_max;
            S[jj] = S[jj] * ms + simd_sum(e0 + e1);
            ss[j * FA2SC64_C + ki0] = half(e0);
            ss[j * FA2SC64_C + ki1] = half(e1);
            for (uint d = simd_lane; d < FA2SC64_HD; d += FA2SC64_NW)
                so[j * FA2SC64_HD + d] = half(float(so[j * FA2SC64_HD + d]) * ms);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += S × V
        simdgroup_half8x8 lo[FA2SC64_NO];
        {
            threadgroup half* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2SC64_NO; oo++) {
                simdgroup_load(lo[oo], sop, FA2SC64_HD);
                sop += 8 * FA2SC64_NSG;
            }
        }
        for (uint cc = 0; cc < FA2SC64_C / 8; cc++) {
            simdgroup_half8x8 ms_mat;
            simdgroup_load(ms_mat, ss + 8 * cc, FA2SC64_C);
            for (uint oo = 0; oo < FA2SC64_NO; oo++) {
                uint dv_block  = oo * FA2SC64_NSG + simd_id;
                uint dv_offset = kv_h * FA2SC64_HD + dv_block * 8;
                uint gkv_base  = tile_kv + cc * 8;
                threadgroup half* my_sv = sk + simd_id * 64;
                for (uint i = simd_lane; i < 64; i += FA2SC64_NW) {
                    uint vi = i / 8, d = i % 8;
                    uint gkv = gkv_base + vi;
                    my_sv[i] = (gkv < max_attend_end)
                        ? V_cache[gkv * kv_stride + dv_offset + d]
                        : half(0.0h);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_half8x8 mv;
                simdgroup_load(mv, my_sv, 8);
                simdgroup_multiply_accumulate(lo[oo], ms_mat, mv, lo[oo]);
            }
        }
        {
            threadgroup half* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2SC64_NO; oo++) {
                simdgroup_store(lo[oo], sop, FA2SC64_HD);
                sop += 8 * FA2SC64_NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write output ─────────────────────────────────────────────────
    for (uint jj = 0; jj < FA2SC64_NQ; jj++) {
        uint j   = jj * FA2SC64_NSG + simd_id;
        uint gqi = tile_q + j;
        if (gqi < n_tokens && S[jj] > 0.0f) {
            float inv_sum = 1.0f / S[jj];
            for (uint d = simd_lane; d < FA2SC64_HD; d += FA2SC64_NW)
                O_buf[gqi * q_stride + h * FA2SC64_HD + d] =
                    float(so[j * FA2SC64_HD + d]) * inv_sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FA2 SIMD cached prefill — f16 KV cache, head_dim=256
//
// Same architecture as cached HD=128 but for head_dim=256 (Gemma3).
// KV tile reduced to 32 tokens to fit 32KB TG memory limit.
// All TG memory is half to stay within 32KB (sq 4KB + so 4KB + ss 0.5KB + sk 16KB = 24.5KB).
// Grid: (ceil(n_tokens / 8), n_heads) × 128 threads
// TG memory: ~25 KB (sq 4KB + so 4KB + ss 0.5KB + sk 16KB)
// ═══════════════════════════════════════════════════════════════════════════

constant uint FA2SC256_Q   = 8;
constant uint FA2SC256_C   = 32;      // reduced KV tile for HD=256
constant uint FA2SC256_NSG = 4;
constant uint FA2SC256_NW  = 32;
constant uint FA2SC256_TG  = 128;
constant uint FA2SC256_HD  = 256;

constant uint FA2SC256_NQ  = 2;       // Q / NSG
constant uint FA2SC256_NC  = 1;       // C / 8 / NSG = 32/8/4 = 1
constant uint FA2SC256_NO  = 8;       // HD / 8 / NSG = 256/8/4 = 8

kernel void attention_prefill_fa2_simd_cached_f16kv_hd256(
    device const float* Q_buf       [[buffer(0)]],
    device const half*  K_cache     [[buffer(1)]],
    device const half*  V_cache     [[buffer(2)]],
    device float*       O_buf       [[buffer(3)]],
    constant uint& n_tokens         [[buffer(4)]],
    constant uint& n_heads          [[buffer(5)]],
    constant uint& n_kv_heads       [[buffer(6)]],
    constant uint& head_dim         [[buffer(7)]],
    constant uint& base_seq_len     [[buffer(8)]],
    constant uint& sliding_window   [[buffer(9)]],
    uint2 tg_id       [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]]
) {
    uint tile_q = tg_id.x * FA2SC256_Q;
    uint h      = tg_id.y;
    if (h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h         = h / heads_per_kv;
    uint q_stride     = n_heads * FA2SC256_HD;
    uint kv_stride    = n_kv_heads * FA2SC256_HD;
    constexpr float inv_sqrt_hd = 0.0625f; // 1/sqrt(256)

    threadgroup half  sq[FA2SC256_Q * FA2SC256_HD];              // 4 KB
    threadgroup half  so[FA2SC256_Q * FA2SC256_HD];              // 4 KB
    threadgroup half  ss[FA2SC256_Q * FA2SC256_C];               // 0.5 KB
    threadgroup half  sk[FA2SC256_NSG * 8 * FA2SC256_HD];        // 16 KB

    float S[FA2SC256_NQ] = {};
    float M[FA2SC256_NQ];
    for (uint i = 0; i < FA2SC256_NQ; i++) M[i] = -INFINITY;

    // ── Attend range with tile-skip ──────────────────────────────────
    uint min_qi = tile_q;
    uint max_qi = min(tile_q + FA2SC256_Q - 1, n_tokens - 1);
    uint min_qi_abs = base_seq_len + min_qi;
    uint max_qi_abs = base_seq_len + max_qi;
    uint max_attend_end = max_qi_abs + 1;
    uint min_attend_start = 0;
    if (sliding_window > 0 && (min_qi_abs + 1) > sliding_window) {
        min_attend_start = (min_qi_abs + 1) - sliding_window;
    }
    uint tile_start_kv = (min_attend_start / FA2SC256_C) * FA2SC256_C;

    // ── Load Q as half ─────────────────────────────────────────────
    for (uint i = lid; i < FA2SC256_Q * FA2SC256_HD; i += FA2SC256_TG) {
        uint qi = i / FA2SC256_HD;
        uint d  = i % FA2SC256_HD;
        uint gqi = tile_q + qi;
        sq[i] = (gqi < n_tokens)
            ? half(Q_buf[gqi * q_stride + h * FA2SC256_HD + d])
            : half(0.0h);
    }
    for (uint i = lid; i < FA2SC256_Q * FA2SC256_HD; i += FA2SC256_TG) {
        so[i] = half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Main KV tile loop ────────────────────────────────────────────
    for (uint tile_kv = tile_start_kv; tile_kv < max_attend_end; tile_kv += FA2SC256_C) {
        uint kv_tile_len = min(uint(FA2SC256_C), max_attend_end - tile_kv);

        // Phase 1: QK^T — stage K from f16 cache
        for (uint cc = 0; cc < FA2SC256_NC; cc++) {
            uint block_idx = cc * FA2SC256_NSG + simd_id;
            uint gkv_base  = tile_kv + block_idx * 8;

            threadgroup half* my_sk = sk + simd_id * (8 * FA2SC256_HD);
            for (uint i = simd_lane; i < 8 * FA2SC256_HD; i += FA2SC256_NW) {
                uint ki = i / FA2SC256_HD;
                uint d  = i % FA2SC256_HD;
                uint gkv = gkv_base + ki;
                my_sk[i] = (gkv < max_attend_end)
                    ? K_cache[gkv * kv_stride + kv_h * FA2SC256_HD + d]
                    : half(0.0h);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8>(half(0.0h));
            for (uint dk = 0; dk < FA2SC256_HD; dk += 16) {
                simdgroup_half8x8 mq0, mq1, mk0, mk1;
                simdgroup_load(mq0, sq + dk,     FA2SC256_HD);
                simdgroup_load(mq1, sq + dk + 8, FA2SC256_HD);
                simdgroup_load(mk0, my_sk + dk,     FA2SC256_HD, ulong2(0, 0), true);
                simdgroup_load(mk1, my_sk + dk + 8, FA2SC256_HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }
            simdgroup_store(mqk, ss + block_idx * 8, FA2SC256_C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: Online softmax with causal + sliding window
        for (uint jj = 0; jj < FA2SC256_NQ; jj++) {
            uint j   = jj * FA2SC256_NSG + simd_id;
            uint gqi = tile_q + j;
            uint gqi_abs = base_seq_len + gqi;
            float prev_max = M[jj];

            uint qi_attend_start = 0;
            if (sliding_window > 0 && (gqi_abs + 1) > sliding_window) {
                qi_attend_start = (gqi_abs + 1) - sliding_window;
            }

            // FA2SC256_C=32 → each lane handles 1 KV token (32 lanes, 32 tokens)
            uint ki0 = simd_lane;
            uint gkv0 = tile_kv + ki0;

            float s0 = -INFINITY;
            if (ki0 < kv_tile_len && gkv0 <= gqi_abs && gkv0 >= qi_attend_start) {
                s0 = float(ss[j * FA2SC256_C + ki0]) * inv_sqrt_hd;
            }

            float new_max = simd_max(max(s0, prev_max));
            float ms = ax_exp(prev_max - new_max);
            float e0 = ax_exp(s0 - new_max);
            float tile_sum = simd_sum(e0);

            M[jj] = new_max;
            S[jj] = S[jj] * ms + tile_sum;

            ss[j * FA2SC256_C + ki0] = half(e0);

            for (uint d = simd_lane; d < FA2SC256_HD; d += FA2SC256_NW) {
                so[j * FA2SC256_HD + d] = half(float(so[j * FA2SC256_HD + d]) * ms);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: O += S × V
        simdgroup_half8x8 lo[FA2SC256_NO];
        {
            threadgroup half* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2SC256_NO; oo++) {
                simdgroup_load(lo[oo], sop, FA2SC256_HD);
                sop += 8 * FA2SC256_NSG;
            }
        }

        for (uint cc = 0; cc < FA2SC256_C / 8; cc++) {
            simdgroup_half8x8 ms_mat;
            simdgroup_load(ms_mat, ss + 8 * cc, FA2SC256_C);
            uint gkv_base = tile_kv + cc * 8;

            for (uint oo = 0; oo < FA2SC256_NO; oo++) {
                uint dv_block  = oo * FA2SC256_NSG + simd_id;
                uint dv_offset = kv_h * FA2SC256_HD + dv_block * 8;

                threadgroup half* my_sv = sk + simd_id * 64;
                for (uint i = simd_lane; i < 64; i += FA2SC256_NW) {
                    uint vi = i / 8, d = i % 8;
                    uint gkv = gkv_base + vi;
                    my_sv[i] = (gkv < max_attend_end)
                        ? V_cache[gkv * kv_stride + dv_offset + d]
                        : half(0.0h);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_half8x8 mv;
                simdgroup_load(mv, my_sv, 8);
                simdgroup_multiply_accumulate(lo[oo], ms_mat, mv, lo[oo]);
            }
        }

        {
            threadgroup half* sop = so + 8 * simd_id;
            for (uint oo = 0; oo < FA2SC256_NO; oo++) {
                simdgroup_store(lo[oo], sop, FA2SC256_HD);
                sop += 8 * FA2SC256_NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write output ─────────────────────────────────────────────────
    for (uint jj = 0; jj < FA2SC256_NQ; jj++) {
        uint j   = jj * FA2SC256_NSG + simd_id;
        uint gqi = tile_q + j;
        if (gqi < n_tokens && S[jj] > 0.0f) {
            float inv_sum = 1.0f / S[jj];
            for (uint d = simd_lane; d < FA2SC256_HD; d += FA2SC256_NW) {
                O_buf[gqi * q_stride + h * FA2SC256_HD + d] =
                    float(so[j * FA2SC256_HD + d]) * inv_sum;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q8_0 KV decode attention — head_dim=128
//
// Same algorithm as attention_decode_f16kv_hd128 but reads K/V from Q8_0
// quantized cache with inline dequant: value = float(qs[i]) * float(d).
//
// Q8_0 block layout: half d (2B) + char qs[32] (32B) = 34 bytes per 32 values.
// KV cache layout: [capacity × blocks_per_row] Q8_0 blocks,
// where blocks_per_row = n_kv_heads × (head_dim / 32).
//
// Grid: (n_heads) × 128 threads
// ═══════════════════════════════════════════════════════════════════════════

constant uint Q8_BLK = 32;   // values per Q8_0 block
constant uint Q8_BLK_BYTES = 34;

// Inline dequant: read one f32 value from Q8_0 cache
inline float q8_dequant(device const uchar* cache, uint token, uint kv_h, uint head_dim, uint blocks_per_row, uint d) {
    uint block_in_head = d / Q8_BLK;
    uint in_block = d % Q8_BLK;
    uint block_idx = token * blocks_per_row + kv_h * (head_dim / Q8_BLK) + block_in_head;
    device const uchar* blk = cache + block_idx * Q8_BLK_BYTES;
    float scale = float(*reinterpret_cast<device const half*>(blk));
    float q = float(char(blk[2 + in_block]));
    return q * scale;
}

kernel void attention_decode_q8kv_hd128(
    device const float* Q         [[buffer(0)]],
    device const uchar* K_cache   [[buffer(1)]],
    device const uchar* V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head_id                  [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head_id >= n_heads || head_dim != 128) return;

    constexpr uint HD   = 128;
    constexpr uint TG   = ATTN_DEC2_TG;   // 128
    constexpr uint NS   = N_DEC2_SIMD;    // 4
    constexpr float scale = 0.088388f;    // rsqrt(128)

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head_id / heads_per_kv;
    uint blocks_per_row = n_kv_heads * (HD / Q8_BLK);

    threadgroup float q_shared[HD];
    threadgroup float out_acc[HD];
    threadgroup float tile_scores[TG];
    threadgroup float simd_buf[NS];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    q_shared[lid] = Q[head_id * HD + lid];
    out_acc[lid]  = 0.0f;
    if (lid == 0) { shared_max = -INFINITY; shared_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += TG) {
        uint tile_len = min(TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            // Q·K dot product with inline Q8_0 dequant
            float s = 0.0f;
            // Process 32 elements at a time (one Q8_0 block)
            uint blocks_per_head = HD / Q8_BLK;
            uint k_block_base = t * blocks_per_row + kv_h * blocks_per_head;
            for (uint bi = 0; bi < blocks_per_head; bi++) {
                device const uchar* blk = K_cache + (k_block_base + bi) * Q8_BLK_BYTES;
                float blk_scale = float(*reinterpret_cast<device const half*>(blk));
                uint d_base = bi * Q8_BLK;
                for (uint i = 0; i < Q8_BLK; i += 4) {
                    float4 qv = float4(q_shared[d_base+i], q_shared[d_base+i+1],
                                       q_shared[d_base+i+2], q_shared[d_base+i+3]);
                    float4 kv = float4(float(char(blk[2+i])), float(char(blk[2+i+1])),
                                       float(char(blk[2+i+2])), float(char(blk[2+i+3])));
                    s += dot(qv, kv * blk_scale);
                }
            }
            my_score = s * scale;
        }

        // Online softmax (identical to f16kv variant)
        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < NS; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max   = shared_max;
        float new_max    = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < NS; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V accumulation with inline Q8_0 dequant
        float acc = out_acc[lid] * correction;
        uint v_d = lid; // lid maps 1:1 to head_dim element
        uint v_block_in_head = v_d / Q8_BLK;
        uint v_in_block = v_d % Q8_BLK;
        for (uint s = 0; s < tile_len; s++) {
            uint t = attend_start + tile_start + s;
            uint v_block_idx = t * blocks_per_row + kv_h * (HD / Q8_BLK) + v_block_in_head;
            device const uchar* v_blk = V_cache + v_block_idx * Q8_BLK_BYTES;
            float v_scale = float(*reinterpret_cast<device const half*>(v_blk));
            float v_val = float(char(v_blk[2 + v_in_block])) * v_scale;
            acc += tile_scores[s] * v_val;
        }
        out_acc[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    O[head_id * HD + lid] = out_acc[lid] * inv_sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Q8_0 KV decode attention — head_dim=256
//
// Same as Q8_0 HD=128 but for Gemma3 (head_dim=256).
// TG=256 threads, threads loop over head_dim for V accumulation.
// Grid: (n_heads) × 256 threads
// ═══════════════════════════════════════════════════════════════════════════

kernel void attention_decode_q8kv_hd256(
    device const float* Q         [[buffer(0)]],
    device const uchar* K_cache   [[buffer(1)]],
    device const uchar* V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head_id                  [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head_id >= n_heads || head_dim != 256) return;

    constexpr uint HD   = 256;
    constexpr float scale = 1.0f / 16.0f; // rsqrt(256)

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head_id / heads_per_kv;
    uint blocks_per_head = HD / Q8_BLK; // 8
    uint blocks_per_row = n_kv_heads * blocks_per_head;

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + head_id * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) { shared_max = -INFINITY; shared_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        // Q·K dot product with inline Q8_0 dequant
        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            float s = 0.0f;
            uint k_block_base = t * blocks_per_row + kv_h * blocks_per_head;
            for (uint bi = 0; bi < blocks_per_head; bi++) {
                device const uchar* blk = K_cache + (k_block_base + bi) * Q8_BLK_BYTES;
                float blk_scale = float(*reinterpret_cast<device const half*>(blk));
                uint d_base = bi * Q8_BLK;
                for (uint i = 0; i < Q8_BLK; i += 4) {
                    float4 qv = float4(q_shared[d_base+i], q_shared[d_base+i+1],
                                       q_shared[d_base+i+2], q_shared[d_base+i+3]);
                    float4 kv = float4(float(char(blk[2+i])), float(char(blk[2+i+1])),
                                       float(char(blk[2+i+2])), float(char(blk[2+i+3])));
                    s += dot(qv, kv * blk_scale);
                }
            }
            my_score = s * scale;
        }

        // Online softmax
        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max   = shared_max;
        float new_max    = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V accumulation with inline Q8_0 dequant (loop over head_dim)
        for (uint d = lid; d < HD; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint v_block_in_head = d / Q8_BLK;
            uint v_in_block = d % Q8_BLK;
            for (uint sv = 0; sv < tile_len; sv++) {
                uint t = attend_start + tile_start + sv;
                uint v_block_idx = t * blocks_per_row + kv_h * blocks_per_head + v_block_in_head;
                device const uchar* v_blk = V_cache + v_block_idx * Q8_BLK_BYTES;
                float v_scale = float(*reinterpret_cast<device const half*>(v_blk));
                float v_val = float(char(v_blk[2 + v_in_block])) * v_scale;
                acc += tile_scores[sv] * v_val;
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + head_id * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q4_0 KV decode attention — head_dim=128
//
// Q4_0 block: half d (2B) + uchar qs[16] (16B) = 18 bytes per 32 values.
// Nibble layout: byte[i] low=element[i], high=element[i+16].
// Dequant: (nibble - 8) * d
// Grid: (n_heads) × 128 threads
// ═══════════════════════════════════════════════════════════════════════════

constant uint Q4_BLK = 32;
constant uint Q4_BLK_BYTES = 18;

kernel void attention_decode_q4kv_hd128(
    device const float* Q         [[buffer(0)]],
    device const uchar* K_cache   [[buffer(1)]],
    device const uchar* V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head_id                  [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head_id >= n_heads || head_dim != 128) return;

    constexpr uint HD   = 128;
    constexpr uint TG   = ATTN_DEC2_TG;
    constexpr uint NS   = N_DEC2_SIMD;
    constexpr float scale = 0.088388f;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head_id / heads_per_kv;
    uint blocks_per_head = HD / Q4_BLK; // 4
    uint blocks_per_row = n_kv_heads * blocks_per_head;

    threadgroup float q_shared[HD];
    threadgroup float out_acc[HD];
    threadgroup float tile_scores[TG];
    threadgroup float simd_buf[NS];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    q_shared[lid] = Q[head_id * HD + lid];
    out_acc[lid]  = 0.0f;
    if (lid == 0) { shared_max = -INFINITY; shared_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += TG) {
        uint tile_len = min(TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            float s = 0.0f;
            uint k_block_base = t * blocks_per_row + kv_h * blocks_per_head;
            for (uint bi = 0; bi < blocks_per_head; bi++) {
                device const uchar* blk = K_cache + (k_block_base + bi) * Q4_BLK_BYTES;
                float blk_scale = float(*reinterpret_cast<device const half*>(blk));
                uint d_base = bi * Q4_BLK;
                // Unpack nibbles: byte[i] low → element[i], high → element[i+16]
                for (uint i = 0; i < 16; i++) {
                    uchar byte_val = blk[2 + i];
                    float k_lo = float(int(byte_val & 0x0F) - 8) * blk_scale;
                    float k_hi = float(int(byte_val >> 4) - 8) * blk_scale;
                    s += q_shared[d_base + i] * k_lo + q_shared[d_base + i + 16] * k_hi;
                }
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < NS; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max   = shared_max;
        float new_max    = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < NS; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V accumulation: dequant Q4_0 per element
        float acc = out_acc[lid] * correction;
        uint v_d = lid;
        uint v_block_in_head = v_d / Q4_BLK;
        uint v_in_block = v_d % Q4_BLK;
        uint v_byte_idx = (v_in_block < 16) ? v_in_block : (v_in_block - 16);
        bool v_is_hi = (v_in_block >= 16);
        for (uint sv = 0; sv < tile_len; sv++) {
            uint t = attend_start + tile_start + sv;
            uint v_block_idx = t * blocks_per_row + kv_h * blocks_per_head + v_block_in_head;
            device const uchar* v_blk = V_cache + v_block_idx * Q4_BLK_BYTES;
            float v_scale = float(*reinterpret_cast<device const half*>(v_blk));
            uchar v_byte = v_blk[2 + v_byte_idx];
            int nibble = v_is_hi ? int(v_byte >> 4) : int(v_byte & 0x0F);
            float v_val = float(nibble - 8) * v_scale;
            acc += tile_scores[sv] * v_val;
        }
        out_acc[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    O[head_id * HD + lid] = out_acc[lid] * inv_sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Q4_0 KV decode attention — head_dim=256
// Grid: (n_heads) × 256 threads
// ═══════════════════════════════════════════════════════════════════════════

kernel void attention_decode_q4kv_hd256(
    device const float* Q         [[buffer(0)]],
    device const uchar* K_cache   [[buffer(1)]],
    device const uchar* V_cache   [[buffer(2)]],
    device float* O               [[buffer(3)]],
    constant uint& n_heads        [[buffer(4)]],
    constant uint& n_kv_heads     [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& attend_start   [[buffer(7)]],
    constant uint& attend_len     [[buffer(8)]],
    uint head_id                  [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (head_id >= n_heads || head_dim != 256) return;

    constexpr uint HD   = 256;
    constexpr float scale = 1.0f / 16.0f;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = head_id / heads_per_kv;
    uint blocks_per_head = HD / Q4_BLK; // 8
    uint blocks_per_row = n_kv_heads * blocks_per_head;

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + head_id * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) { shared_max = -INFINITY; shared_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start + tile_start + lid;
            float s = 0.0f;
            uint k_block_base = t * blocks_per_row + kv_h * blocks_per_head;
            for (uint bi = 0; bi < blocks_per_head; bi++) {
                device const uchar* blk = K_cache + (k_block_base + bi) * Q4_BLK_BYTES;
                float blk_scale = float(*reinterpret_cast<device const half*>(blk));
                uint d_base = bi * Q4_BLK;
                for (uint i = 0; i < 16; i++) {
                    uchar byte_val = blk[2 + i];
                    float k_lo = float(int(byte_val & 0x0F) - 8) * blk_scale;
                    float k_hi = float(int(byte_val >> 4) - 8) * blk_scale;
                    s += q_shared[d_base + i] * k_lo + q_shared[d_base + i + 16] * k_hi;
                }
            }
            my_score = s * scale;
        }

        float v = (lid < tile_len) ? my_score : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) simd_buf[simd_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max   = shared_max;
        float new_max    = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lid; d < HD; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint v_block_in_head = d / Q4_BLK;
            uint v_in_block = d % Q4_BLK;
            uint v_byte_idx = (v_in_block < 16) ? v_in_block : (v_in_block - 16);
            bool v_is_hi = (v_in_block >= 16);
            for (uint sv = 0; sv < tile_len; sv++) {
                uint t = attend_start + tile_start + sv;
                uint v_block_idx = t * blocks_per_row + kv_h * blocks_per_head + v_block_in_head;
                device const uchar* v_blk = V_cache + v_block_idx * Q4_BLK_BYTES;
                float v_scale = float(*reinterpret_cast<device const half*>(v_blk));
                uchar v_byte = v_blk[2 + v_byte_idx];
                int nibble = v_is_hi ? int(v_byte >> 4) : int(v_byte & 0x0F);
                float v_val = float(nibble - 8) * v_scale;
                acc += tile_scores[sv] * v_val;
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + head_id * HD;
    for (uint d = lid; d < HD; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q4_0 KV cached prefill attention — generic head_dim
// Grid: (n_tokens, n_heads) × 256 threads
// ═══════════════════════════════════════════════════════════════════════════

kernel void attention_prefill_cache_q4kv(
    device const float* Q          [[buffer(0)]],
    device const uchar* K_cache    [[buffer(1)]],
    device const uchar* V_cache    [[buffer(2)]],
    device float* O                [[buffer(3)]],
    constant uint& n_tokens        [[buffer(4)]],
    constant uint& n_heads         [[buffer(5)]],
    constant uint& n_kv_heads      [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],
    constant uint& base_seq_len    [[buffer(8)]],
    constant uint& sliding_window  [[buffer(9)]],
    uint2 tg_id                    [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    uint qi = tg_id.x;
    uint h = tg_id.y;
    if (qi >= n_tokens || h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride = n_heads * head_dim;
    float attn_scale = rsqrt(float(head_dim));
    uint blocks_per_head = head_dim / Q4_BLK;
    uint blocks_per_row = n_kv_heads * blocks_per_head;

    uint attend_end = base_seq_len + qi + 1;
    uint attend_start_pos = 0;
    if (sliding_window > 0 && attend_end > sliding_window) {
        attend_start_pos = attend_end - sliding_window;
    }
    uint attend_len = attend_end - attend_start_pos;

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) { shared_max = -INFINITY; shared_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start_pos + tile_start + lid;
            float s = 0.0f;
            uint k_block_base = t * blocks_per_row + kv_h * blocks_per_head;
            for (uint bi = 0; bi < blocks_per_head; bi++) {
                device const uchar* blk = K_cache + (k_block_base + bi) * Q4_BLK_BYTES;
                float blk_scale = float(*reinterpret_cast<device const half*>(blk));
                uint d_base = bi * Q4_BLK;
                for (uint i = 0; i < 16; i++) {
                    uchar byte_val = blk[2 + i];
                    float k_lo = float(int(byte_val & 0x0F) - 8) * blk_scale;
                    float k_hi = float(int(byte_val >> 4) - 8) * blk_scale;
                    s += q_shared[d_base + i] * k_lo + q_shared[d_base + i + 16] * k_hi;
                }
            }
            my_score = s * attn_scale;
        }

        float sv = (lid < tile_len) ? my_score : -INFINITY;
        sv = simd_max(sv);
        if (simd_lane == 0) simd_buf[simd_id] = sv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lid; d < head_dim; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint v_block_in_head = d / Q4_BLK;
            uint v_in_block = d % Q4_BLK;
            uint v_byte_idx = (v_in_block < 16) ? v_in_block : (v_in_block - 16);
            bool v_is_hi = (v_in_block >= 16);
            for (uint ssv = 0; ssv < tile_len; ssv++) {
                uint t = attend_start_pos + tile_start + ssv;
                uint v_block_idx = t * blocks_per_row + kv_h * blocks_per_head + v_block_in_head;
                device const uchar* v_blk = V_cache + v_block_idx * Q4_BLK_BYTES;
                float v_scale = float(*reinterpret_cast<device const half*>(v_blk));
                uchar v_byte = v_blk[2 + v_byte_idx];
                int nibble = v_is_hi ? int(v_byte >> 4) : int(v_byte & 0x0F);
                float v_val = float(nibble - 8) * v_scale;
                acc += tile_scores[ssv] * v_val;
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q8_0 KV cached prefill attention — generic head_dim
//
// Same structure as attention_prefill_cache_f16kv but reads from Q8_0 cache.
// Grid: (n_tokens, n_heads) × 256 threads
// ═══════════════════════════════════════════════════════════════════════════

kernel void attention_prefill_cache_q8kv(
    device const float* Q          [[buffer(0)]],
    device const uchar* K_cache    [[buffer(1)]],
    device const uchar* V_cache    [[buffer(2)]],
    device float* O                [[buffer(3)]],
    constant uint& n_tokens        [[buffer(4)]],
    constant uint& n_heads         [[buffer(5)]],
    constant uint& n_kv_heads      [[buffer(6)]],
    constant uint& head_dim        [[buffer(7)]],
    constant uint& base_seq_len    [[buffer(8)]],
    constant uint& sliding_window  [[buffer(9)]],
    uint2 tg_id                    [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    uint qi = tg_id.x;
    uint h = tg_id.y;
    if (qi >= n_tokens || h >= n_heads) return;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint q_stride = n_heads * head_dim;
    float attn_scale = rsqrt(float(head_dim));
    uint blocks_per_head = head_dim / Q8_BLK;
    uint blocks_per_row = n_kv_heads * blocks_per_head;

    uint attend_end = base_seq_len + qi + 1;
    uint attend_start_pos = 0;
    if (sliding_window > 0 && attend_end > sliding_window) {
        attend_start_pos = attend_end - sliding_window;
    }
    uint attend_len = attend_end - attend_start_pos;

    threadgroup float q_shared[MAX_HD];
    threadgroup float out_acc[MAX_HD];
    threadgroup float tile_scores[ATTN_TG];
    threadgroup float simd_buf[N_SIMD];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    device const float* q_ptr = Q + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (lid == 0) { shared_max = -INFINITY; shared_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < attend_len; tile_start += ATTN_TG) {
        uint tile_len = min(ATTN_TG, attend_len - tile_start);

        // Q·K dot product with inline Q8_0 dequant
        float my_score = -INFINITY;
        if (lid < tile_len) {
            uint t = attend_start_pos + tile_start + lid;
            float s = 0.0f;
            uint k_block_base = t * blocks_per_row + kv_h * blocks_per_head;
            for (uint bi = 0; bi < blocks_per_head; bi++) {
                device const uchar* blk = K_cache + (k_block_base + bi) * Q8_BLK_BYTES;
                float blk_scale = float(*reinterpret_cast<device const half*>(blk));
                uint d_base = bi * Q8_BLK;
                for (uint i = 0; i < Q8_BLK; i += 4) {
                    float4 qv = float4(q_shared[d_base+i], q_shared[d_base+i+1],
                                       q_shared[d_base+i+2], q_shared[d_base+i+3]);
                    float4 kv = float4(float(char(blk[2+i])), float(char(blk[2+i+1])),
                                       float(char(blk[2+i+2])), float(char(blk[2+i+3])));
                    s += dot(qv, kv * blk_scale);
                }
            }
            my_score = s * attn_scale;
        }

        // Online softmax (same as f16kv variant)
        float sv = (lid < tile_len) ? my_score : -INFINITY;
        sv = simd_max(sv);
        if (simd_lane == 0) simd_buf[simd_id] = sv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float m = simd_buf[0];
            for (uint i = 1; i < N_SIMD; i++) m = max(m, simd_buf[i]);
            simd_buf[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = simd_buf[0];

        float prev_max = shared_max;
        float new_max = max(prev_max, tile_max);
        float correction = ax_exp(prev_max - new_max);

        float exp_s = (lid < tile_len) ? ax_exp(my_score - new_max) : 0.0f;
        tile_scores[lid] = exp_s;

        float es = simd_sum(exp_s);
        if (simd_lane == 0) simd_buf[simd_id] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tile_sum = 0.0f;
            for (uint i = 0; i < N_SIMD; i++) tile_sum += simd_buf[i];
            shared_max = new_max;
            shared_sum = shared_sum * correction + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V accumulation with inline Q8_0 dequant
        for (uint d = lid; d < head_dim; d += ATTN_TG) {
            float acc = out_acc[d] * correction;
            uint v_block_in_head = d / Q8_BLK;
            uint v_in_block = d % Q8_BLK;
            for (uint sv = 0; sv < tile_len; sv++) {
                uint t = attend_start_pos + tile_start + sv;
                uint v_block_idx = t * blocks_per_row + kv_h * blocks_per_head + v_block_in_head;
                device const uchar* v_blk = V_cache + v_block_idx * Q8_BLK_BYTES;
                float v_scale = float(*reinterpret_cast<device const half*>(v_blk));
                float v_val = float(char(v_blk[2 + v_in_block])) * v_scale;
                acc += tile_scores[sv] * v_val;
            }
            out_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (shared_sum > 0.0f) ? (1.0f / shared_sum) : 0.0f;
    device float* o_ptr = O + qi * q_stride + h * head_dim;
    for (uint d = lid; d < head_dim; d += ATTN_TG) {
        o_ptr[d] = out_acc[d] * inv_sum;
    }
}
