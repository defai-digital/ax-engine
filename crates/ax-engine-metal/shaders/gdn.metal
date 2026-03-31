// GDN (Gated Delta Net) Metal kernels adapted from mistral.rs.
//
// AX keeps the Qwen3.5 causal-conv path in token-major f32 layout:
// - input/output: [seq_len, conv_dim]
// - kernel: [conv_cache_len + 1, conv_dim]

// fast::exp for gate decay — precision sufficient for state scaling.
#define ax_exp(x) fast::exp(x)
// - conv_state: [conv_cache_len, conv_dim]
//
// This matches the existing CPU helper and avoids extra layout transforms in
// the first Metal port of the recurrent conv path.

#include <metal_stdlib>
using namespace metal;

constant uint QWEN35_MAX_CONV_CACHE = 8;
constant uint QWEN35_GDN_PACK_TG = 256;

[[kernel]] void qwen35_causal_conv_sequence_f32(
    const device float *input [[buffer(0)]],
    const device float *weights [[buffer(1)]],
    device float *conv_state [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &seq_len [[buffer(4)]],
    constant uint &conv_cache_len [[buffer(5)]],
    constant uint &conv_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint ch = gid;
    if (ch >= conv_dim) return;

    float history[QWEN35_MAX_CONV_CACHE];
    for (uint i = 0; i < QWEN35_MAX_CONV_CACHE; ++i) {
        history[i] = 0.0f;
    }
    for (uint i = 0; i < conv_cache_len; ++i) {
        history[i] = conv_state[i * conv_dim + ch];
    }

    const uint current_weight_base = conv_cache_len * conv_dim + ch;
    for (uint t = 0; t < seq_len; ++t) {
        const uint offset = t * conv_dim + ch;
        const float x = input[offset];
        float acc = x * weights[current_weight_base];
        for (uint i = 0; i < conv_cache_len; ++i) {
            acc = fma(history[i], weights[i * conv_dim + ch], acc);
        }
        output[offset] = acc / (1.0f + exp(-acc));

        if (conv_cache_len > 0) {
            for (uint i = 0; i + 1 < conv_cache_len; ++i) {
                history[i] = history[i + 1];
            }
            history[conv_cache_len - 1] = x;
        }
    }

    for (uint i = 0; i < conv_cache_len; ++i) {
        conv_state[i * conv_dim + ch] = history[i];
    }
}

/// Parallel causal conv: one thread per (channel, token) pair.
/// For tokens t >= conv_cache_len, all inputs come from the input buffer.
/// For tokens t < conv_cache_len, reads initial history from conv_state.
/// Grid: (conv_dim, seq_len, 1).  Threads: (min(conv_dim, 256), 1, 1).
[[kernel]] void qwen35_causal_conv_sequence_parallel_f32(
    const device float *input [[buffer(0)]],
    const device float *weights [[buffer(1)]],
    device float *conv_state [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &seq_len [[buffer(4)]],
    constant uint &conv_cache_len [[buffer(5)]],
    constant uint &conv_dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint ch = gid.x;
    const uint t = gid.y;
    if (ch >= conv_dim || t >= seq_len) return;

    const uint offset = t * conv_dim + ch;
    const float x = input[offset];
    float acc = x * weights[conv_cache_len * conv_dim + ch];

    for (uint i = 0; i < conv_cache_len; ++i) {
        // history[i] corresponds to the input at position (t - conv_cache_len + i)
        const int src_t = (int)t - (int)conv_cache_len + (int)i;
        float h;
        if (src_t < 0) {
            // Read from initial conv_state: slot (conv_cache_len + src_t)
            h = conv_state[((uint)((int)conv_cache_len + src_t)) * conv_dim + ch];
        } else {
            h = input[(uint)src_t * conv_dim + ch];
        }
        acc = fma(h, weights[i * conv_dim + ch], acc);
    }
    output[offset] = acc / (1.0f + exp(-acc));

    // Update conv_state: after processing seq_len tokens, the state should
    // hold the last conv_cache_len input values.
    // This kernel is only dispatched when seq_len >= conv_cache_len (enforced
    // by the dispatch code), so state[j] = input[seq_len - conv_cache_len + j].
    const uint tail_start = seq_len - conv_cache_len;
    if (t >= tail_start) {
        conv_state[(t - tail_start) * conv_dim + ch] = x;
    }
}

[[kernel]] void qwen35_prepare_single_token_gdn_qkv_f32(
    const device float *conv_out [[buffer(0)]],
    device float *q_out [[buffer(1)]],
    device float *k_out [[buffer(2)]],
    device float *v_out [[buffer(3)]],
    constant uint &group_count [[buffer(4)]],
    constant uint &time_step_rank [[buffer(5)]],
    constant uint &state_size [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const uint value_dim = time_step_rank * state_size;
    if (gid >= value_dim) return;

    const uint repeats = time_step_rank / group_count;
    const uint dst_head = gid / state_size;
    const uint lane = gid % state_size;
    const uint src_head = dst_head / repeats;
    const uint src_base = src_head * state_size;
    const uint key_dim = group_count * state_size;

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (uint idx = 0; idx < state_size; ++idx) {
        const float q = conv_out[src_base + idx];
        const float k = conv_out[key_dim + src_base + idx];
        q_sum_sq = fma(q, q, q_sum_sq);
        k_sum_sq = fma(k, k, k_sum_sq);
    }

    const float q_inv = rsqrt(q_sum_sq + eps);
    const float k_inv = rsqrt(k_sum_sq + eps);
    q_out[gid] = conv_out[src_base + lane] * q_inv;
    k_out[gid] = conv_out[key_dim + src_base + lane] * k_inv;
    v_out[gid] = conv_out[2 * key_dim + gid];
}

[[kernel]] void qwen35_prepare_multi_token_gdn_qk_f32(
    const device float *conv_out [[buffer(0)]],
    device float *q_out [[buffer(1)]],
    device float *k_out [[buffer(2)]],
    constant uint &n_tokens [[buffer(3)]],
    constant uint &group_count [[buffer(4)]],
    constant uint &time_step_rank [[buffer(5)]],
    constant uint &state_size [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint token_src = tg_pos.y;
    const uint token_idx = token_src / group_count;
    if (token_idx >= n_tokens) return;

    const uint src_head = token_src % group_count;
    const uint repeats = time_step_rank / group_count;
    const uint key_dim = group_count * state_size;
    const uint value_dim = time_step_rank * state_size;
    const uint conv_dim = 2 * key_dim + value_dim;
    const uint conv_token_base = token_idx * conv_dim;
    const uint src_base = src_head * state_size;

    threadgroup float q_sum[QWEN35_GDN_PACK_TG];
    threadgroup float k_sum[QWEN35_GDN_PACK_TG];

    float q_local = 0.0f;
    float k_local = 0.0f;
    for (uint lane = tid; lane < state_size; lane += QWEN35_GDN_PACK_TG) {
        const float q = conv_out[conv_token_base + src_base + lane];
        const float k = conv_out[conv_token_base + key_dim + src_base + lane];
        q_local = fma(q, q, q_local);
        k_local = fma(k, k, k_local);
    }
    q_sum[tid] = q_local;
    k_sum[tid] = k_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = QWEN35_GDN_PACK_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_sum[tid] += q_sum[tid + stride];
            k_sum[tid] += k_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float q_inv = rsqrt(q_sum[0] + eps);
    const float k_inv = rsqrt(k_sum[0] + eps);
    for (uint lane = tid; lane < state_size; lane += QWEN35_GDN_PACK_TG) {
        const float q = conv_out[conv_token_base + src_base + lane] * q_inv;
        const float k = conv_out[conv_token_base + key_dim + src_base + lane] * k_inv;
        for (uint rep = 0; rep < repeats; ++rep) {
            const uint dst_head = src_head * repeats + rep;
            const uint dst =
                dst_head * n_tokens * state_size + token_idx * state_size + lane;
            q_out[dst] = q;
            k_out[dst] = k;
        }
    }
}

[[kernel]] void qwen35_prepare_multi_token_gdn_vgb_f32(
    const device float *conv_out [[buffer(0)]],
    const device float *alpha_in [[buffer(1)]],
    const device float *beta_in [[buffer(2)]],
    device float *v_out [[buffer(3)]],
    device float *gate_out [[buffer(4)]],
    device float *beta_out [[buffer(5)]],
    constant uint &n_tokens [[buffer(6)]],
    constant uint &group_count [[buffer(7)]],
    constant uint &time_step_rank [[buffer(8)]],
    constant uint &state_size [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint lane = gid.x;
    if (lane >= state_size) return;

    const uint token_head = gid.y;
    const uint token_idx = token_head / time_step_rank;
    if (token_idx >= n_tokens) return;

    const uint dst_head = token_head % time_step_rank;
    const uint key_dim = group_count * state_size;
    const uint value_dim = time_step_rank * state_size;
    const uint conv_dim = 2 * key_dim + value_dim;
    const uint conv_token_base = token_idx * conv_dim;
    const uint dst = dst_head * n_tokens * state_size + token_idx * state_size + lane;
    const uint src = conv_token_base + 2 * key_dim + dst_head * state_size + lane;
    v_out[dst] = conv_out[src];

    if (lane == 0) {
        const uint scalar_idx = token_idx * time_step_rank + dst_head;
        const uint scalar_dst = dst_head * n_tokens + token_idx;
        gate_out[scalar_dst] = alpha_in[scalar_idx];
        beta_out[scalar_dst] = beta_in[scalar_idx];
    }
}

[[kernel]] void qwen35_prepare_multi_token_gdn_vgb_ab_f16(
    const device float *conv_out [[buffer(0)]],
    const device half *alpha_in [[buffer(1)]],
    const device half *beta_in [[buffer(2)]],
    device float *v_out [[buffer(3)]],
    device float *gate_out [[buffer(4)]],
    device float *beta_out [[buffer(5)]],
    constant uint &n_tokens [[buffer(6)]],
    constant uint &group_count [[buffer(7)]],
    constant uint &time_step_rank [[buffer(8)]],
    constant uint &state_size [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint lane = gid.x;
    if (lane >= state_size) return;

    const uint token_head = gid.y;
    const uint token_idx = token_head / time_step_rank;
    if (token_idx >= n_tokens) return;

    const uint dst_head = token_head % time_step_rank;
    const uint key_dim = group_count * state_size;
    const uint value_dim = time_step_rank * state_size;
    const uint conv_dim = 2 * key_dim + value_dim;
    const uint conv_token_base = token_idx * conv_dim;
    const uint dst = dst_head * n_tokens * state_size + token_idx * state_size + lane;
    const uint src = conv_token_base + 2 * key_dim + dst_head * state_size + lane;
    v_out[dst] = conv_out[src];

    if (lane == 0) {
        const uint scalar_idx = token_idx * time_step_rank + dst_head;
        const uint scalar_dst = dst_head * n_tokens + token_idx;
        gate_out[scalar_dst] = float(alpha_in[scalar_idx]);
        beta_out[scalar_dst] = float(beta_in[scalar_idx]);
    }
}

[[kernel]] void qwen35_unpack_bhsk_to_token_major_f32(
    const device float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &n_tokens [[buffer(2)]],
    constant uint &n_heads [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint lane = gid.x;
    if (lane >= head_dim) return;

    const uint token_head = gid.y;
    const uint token_idx = token_head / n_heads;
    if (token_idx >= n_tokens) return;
    const uint head = token_head % n_heads;

    const uint src = head * n_tokens * head_dim + token_idx * head_dim + lane;
    const uint dst = token_idx * n_heads * head_dim + head * head_dim + lane;
    output[dst] = input[src];
}

template <int BK, int BV>
[[kernel]] void qwen35_single_token_gated_delta_fused_kernel(
    const device float *conv_out [[buffer(0)]],
    const device float *gate [[buffer(1)]],
    const device float *beta [[buffer(2)]],
    device float *state [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &group_count [[buffer(5)]],
    constant uint &time_step_rank [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    const uint v_tile = tgpig.x;
    const uint bh = tgpig.y;
    const uint v_idx = v_tile * BV + tid;
    if (bh >= time_step_rank || v_idx >= BK) return;

    const uint repeats = time_step_rank / group_count;
    const uint src_head = bh / repeats;
    const uint key_dim = group_count * BK;
    const uint src_base = src_head * BK;
    const uint value_base = 2 * key_dim + bh * BK;
    const float scale = rsqrt((float)BK);

    threadgroup float q_buf[BK];
    threadgroup float k_buf[BK];
    constexpr uint simd_groups = BV / 32;
    threadgroup float q_simd_sum[simd_groups];
    threadgroup float k_simd_sum[simd_groups];
    threadgroup float q_inv_shared;
    threadgroup float k_inv_shared;

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (uint j = tid; j < BK; j += BV) {
        const float q_val = conv_out[src_base + j];
        const float k_val = conv_out[key_dim + src_base + j];
        q_buf[j] = q_val;
        k_buf[j] = k_val;
        q_sum_sq = fma(q_val, q_val, q_sum_sq);
        k_sum_sq = fma(k_val, k_val, k_sum_sq);
    }

    q_sum_sq = simd_sum(q_sum_sq);
    k_sum_sq = simd_sum(k_sum_sq);
    if (simd_lane == 0) {
        q_simd_sum[simd_id] = q_sum_sq;
        k_simd_sum[simd_id] = k_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        q_sum_sq = (simd_lane < simd_groups) ? q_simd_sum[simd_lane] : 0.0f;
        k_sum_sq = (simd_lane < simd_groups) ? k_simd_sum[simd_lane] : 0.0f;
        q_sum_sq = simd_sum(q_sum_sq);
        k_sum_sq = simd_sum(k_sum_sq);
        if (simd_lane == 0) {
            q_inv_shared = rsqrt(q_sum_sq + eps);
            k_inv_shared = rsqrt(k_sum_sq + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float q_inv = q_inv_shared;
    const float k_inv = k_inv_shared;
    for (uint j = tid; j < BK; j += BV) {
        q_buf[j] *= q_inv;
        k_buf[j] *= k_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *state_bh = state + bh * BK * BK;
    device float *out_bh = output + bh * BK;
    float s[BK];
    for (uint j = 0; j < BK; ++j) {
        s[j] = state_bh[j * BK + v_idx];
    }

    const float decay = ax_exp(gate[bh]);
    const float beta_t = beta[bh];
    const float v_t = conv_out[value_base + v_idx];

    float kv_mem = 0.0f;
    for (uint j = 0; j < BK; ++j) {
        s[j] *= decay;
        kv_mem = fma(s[j], k_buf[j], kv_mem);
    }

    const float delta = (v_t - kv_mem) * beta_t;
    float y_t = 0.0f;
    for (uint j = 0; j < BK; ++j) {
        s[j] = fma(k_buf[j], delta, s[j]);
        y_t = fma(s[j], q_buf[j], y_t);
    }

    out_bh[v_idx] = y_t * scale;
    for (uint j = 0; j < BK; ++j) {
        state_bh[j * BK + v_idx] = s[j];
    }
}

template <int BK, int BV>
[[kernel]] void gated_delta_rule_kernel(
    const device float *q [[buffer(0)]],
    const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]],
    const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant uint &seq_len [[buffer(7)]],
    constant uint &v_dim [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint v_tile = tgpig.x;
    const uint bh = tgpig.y;
    const uint v_idx = v_tile * BV + tid;

    if (v_idx >= v_dim) return;

    const device float *q_bh = q + bh * seq_len * BK;
    const device float *k_bh = k + bh * seq_len * BK;
    const device float *v_bh = v + bh * seq_len * v_dim;
    const device float *g_bh = g + bh * seq_len;
    const device float *beta_bh = beta + bh * seq_len;
    device float *state_bh = state + bh * BK * v_dim;
    device float *out_bh = output + bh * seq_len * v_dim;

    threadgroup float k_buf[BK];
    threadgroup float q_buf[BK];
    const float scale = rsqrt((float)BK);

    float s[BK];
    for (uint j = 0; j < BK; j++) {
        s[j] = state_bh[j * v_dim + v_idx];
    }

    for (uint t = 0; t < seq_len; t++) {
        for (uint j = tid; j < BK; j += BV) {
            k_buf[j] = k_bh[t * BK + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float decay = ax_exp(g_bh[t]);
        float beta_t = beta_bh[t];
        float v_t = v_bh[t * v_dim + v_idx];

        float kv_mem = 0.0f;
        for (uint j = 0; j < BK; j++) {
            s[j] *= decay;
            kv_mem = fma(s[j], k_buf[j], kv_mem);
        }

        float delta = (v_t - kv_mem) * beta_t;

        for (uint j = tid; j < BK; j += BV) {
            q_buf[j] = q_bh[t * BK + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float y_t = 0.0f;
        for (uint j = 0; j < BK; j++) {
            s[j] = fma(k_buf[j], delta, s[j]);
            y_t = fma(s[j], q_buf[j], y_t);
        }

        out_bh[t * v_dim + v_idx] = y_t * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint j = 0; j < BK; j++) {
        state_bh[j * v_dim + v_idx] = s[j];
    }
}

template <int BV, int MAX_K>
[[kernel]] void gated_delta_rule_kernel_fallback(
    const device float *q [[buffer(0)]],
    const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]],
    const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant uint &seq_len [[buffer(7)]],
    constant uint &k_dim [[buffer(8)]],
    constant uint &v_dim [[buffer(9)]],
    threadgroup float *shared_mem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint v_tile = tgpig.x;
    const uint bh = tgpig.y;
    const uint v_idx = v_tile * BV + tid;

    if (v_idx >= v_dim) return;

    const device float *q_bh = q + bh * seq_len * k_dim;
    const device float *k_bh = k + bh * seq_len * k_dim;
    const device float *v_bh = v + bh * seq_len * v_dim;
    const device float *g_bh = g + bh * seq_len;
    const device float *beta_bh = beta + bh * seq_len;
    device float *state_bh = state + bh * k_dim * v_dim;
    device float *out_bh = output + bh * seq_len * v_dim;

    threadgroup float *k_buf = shared_mem;
    threadgroup float *q_buf = shared_mem + k_dim;
    const float scale = rsqrt((float)k_dim);

    float s[MAX_K];
    for (uint j = 0; j < k_dim; j++) {
        s[j] = state_bh[j * v_dim + v_idx];
    }

    for (uint t = 0; t < seq_len; t++) {
        for (uint j = tid; j < k_dim; j += BV) {
            k_buf[j] = k_bh[t * k_dim + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float decay = ax_exp(g_bh[t]);
        float beta_t = beta_bh[t];
        float v_t = v_bh[t * v_dim + v_idx];

        float kv_mem = 0.0f;
        for (uint j = 0; j < k_dim; j++) {
            s[j] *= decay;
            kv_mem = fma(s[j], k_buf[j], kv_mem);
        }

        float delta = (v_t - kv_mem) * beta_t;

        for (uint j = tid; j < k_dim; j += BV) {
            q_buf[j] = q_bh[t * k_dim + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float y_t = 0.0f;
        for (uint j = 0; j < k_dim; j++) {
            s[j] = fma(k_buf[j], delta, s[j]);
            y_t = fma(s[j], q_buf[j], y_t);
        }

        out_bh[t * v_dim + v_idx] = y_t * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint j = 0; j < k_dim; j++) {
        state_bh[j * v_dim + v_idx] = s[j];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// gated_delta_rule_simd — llama.cpp-style fused GDN with simd_sum reductions
//
// Ports kernel_gated_delta_net_impl from llama.cpp:
//   - State S_v dimension distributed across threadgroups (not just threads)
//   - Each thread holds NSG state elements in registers (NSG=4)
//   - simd_sum for k-dimension reductions (32 threads × 4 = 128 k-elements)
//   - Grid: (S_v/NSG, n_heads, n_seqs) = (32, 32, 1) = 1024 TGs
//   - TG: (32, NSG, 1) = 128 threads
//   - 32x more TGs than the old kernel for the same head count
//
// Memory: state in registers (NSG floats per thread = 16 bytes), no TG memory.
// ═══════════════════════════════════════════════════════════════════════════
template <int NSG>
[[kernel]] void gated_delta_rule_simd(
    const device float *q [[buffer(0)]],
    const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]],
    const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant uint &seq_len [[buffer(7)]],
    constant uint &v_dim [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint tx = tpitg.x;     // SIMD lane: 0..31
    const uint ty = tpitg.y;     // sub-row within TG tile: 0..NSG-1
    const uint bh = tgpig.y;     // head index
    const uint i20 = tgpig.x * NSG + ty;  // v-dimension element index

    if (i20 >= v_dim) return;

    const uint k_dim = v_dim;    // S_k == S_v for Qwen3.5
    const float scale = rsqrt((float)k_dim);

    // State pointer: state is [n_heads, k_dim, v_dim], stored row-major in k-dim.
    // We load row i20 from the k-dim × v_dim state matrix for head bh.
    // state_bh[j * v_dim + i20] for j = 0..k_dim-1
    device const float *s_ptr = state + bh * k_dim * v_dim + i20;

    // Each thread loads NSG consecutive elements of the k-dimension.
    // 32 threads × NSG=4 = 128 = full k-dimension coverage.
    float ls[NSG];
    for (short j = 0; j < NSG; j++) {
        const uint is = tx * NSG + j;
        ls[j] = (is < k_dim) ? s_ptr[is * v_dim] : 0.0f;
    }

    // Input pointers — layout: [n_heads, seq_len, k_dim/v_dim]
    const device float *q_bh = q + bh * seq_len * k_dim;
    const device float *k_bh = k + bh * seq_len * k_dim;
    const device float *v_bh = v + bh * seq_len * v_dim;
    const device float *g_bh = g + bh * seq_len;
    const device float *beta_bh = beta + bh * seq_len;
    device float *out_bh = output + bh * seq_len * v_dim;

    // Token-serial loop (same as llama.cpp autoregressive path).
    for (uint t = 0; t < seq_len; t++) {
        // 1. state *= exp(gate)  +  s_k = sum(state * k)
        const float g_exp = ax_exp(g_bh[t]);
        float s_k = 0.0f;
        if (NSG == 4) {
            float4 ls4 = *(thread float4*)(&ls[0]);
            float4 k4 = *(device const float4*)(&k_bh[t * k_dim + tx * 4]);
            ls4 *= g_exp;
            *(thread float4*)(&ls[0]) = ls4;
            s_k = dot(ls4, k4);
        } else {
            for (short j = 0; j < NSG; j++) {
                const uint is = tx * NSG + j;
                ls[j] *= g_exp;
                s_k += ls[j] * k_bh[t * k_dim + is];
            }
        }
        // Reduce s_k across all 32 SIMD lanes (covers full k_dim).
        s_k = simd_sum(s_k);

        // 2. delta = (v - s_k) * beta
        const float d = (v_bh[t * v_dim + i20] - s_k) * beta_bh[t];

        // 3. state += k * delta  +  y = sum(state * q)
        float y = 0.0f;
        if (NSG == 4) {
            float4 ls4 = *(thread float4*)(&ls[0]);
            float4 k4 = *(device const float4*)(&k_bh[t * k_dim + tx * 4]);
            float4 q4 = *(device const float4*)(&q_bh[t * k_dim + tx * 4]);
            ls4 += k4 * d;
            *(thread float4*)(&ls[0]) = ls4;
            y = dot(ls4, q4);
        } else {
            for (short j = 0; j < NSG; j++) {
                const uint is = tx * NSG + j;
                ls[j] += k_bh[t * k_dim + is] * d;
                y += ls[j] * q_bh[t * k_dim + is];
            }
        }
        y = simd_sum(y);

        // 4. Write output (only lane 0 has the reduced sum).
        if (tx == 0) {
            out_bh[t * v_dim + i20] = y * scale;
        }
    }

    // Write back state.
    device float *state_out = state + bh * k_dim * v_dim + i20;
    for (short j = 0; j < NSG; j++) {
        const uint is = tx * NSG + j;
        if (is < k_dim) {
            state_out[is * v_dim] = ls[j];
        }
    }
}

template [[host_name("gated_delta_rule_simd_4")]] [[kernel]]
void gated_delta_rule_simd<4>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint3, uint3);

template [[host_name("gated_delta_rule_simd_2")]] [[kernel]]
void gated_delta_rule_simd<2>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint3, uint3);

template [[host_name("gated_delta_rule_simd_1")]] [[kernel]]
void gated_delta_rule_simd<1>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint3, uint3);

template <int BT, int BK, int BV>
[[kernel]] void chunked_gated_delta_rule_kernel(
    const device float *q [[buffer(0)]],
    const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]],
    const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant uint &seq_len [[buffer(7)]],
    constant uint &v_dim [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint v_tile = tgpig.x;
    const uint bh = tgpig.y;
    const uint v_idx = v_tile * BV + tid;

    if (v_idx >= v_dim) return;

    const uint num_chunks = (seq_len + BT - 1) / BT;

    const device float *q_bh = q + bh * seq_len * BK;
    const device float *k_bh = k + bh * seq_len * BK;
    const device float *v_bh = v + bh * seq_len * v_dim;
    const device float *g_bh = g + bh * seq_len;
    const device float *beta_bh = beta + bh * seq_len;
    device float *state_bh = state + bh * BK * v_dim;
    device float *out_bh = output + bh * seq_len * v_dim;

    threadgroup float k_chunk[BT * BK];
    threadgroup float kk_dot[BT * BT];
    threadgroup float gcum[BT];
    threadgroup float decay[BT];     // precomputed exp(gcum[i])
    threadgroup float decay_inv[BT]; // precomputed 1/exp(gcum[i])
    threadgroup float beta_s[BT];
    threadgroup float q_buf[BK];

    const float scale = rsqrt((float)BK);

    float s[BK];
    for (uint j = 0; j < BK; ++j) {
        s[j] = state_bh[j * v_dim + v_idx];
    }

    float delta_arr[BT];

    for (uint c = 0; c < num_chunks; ++c) {
        const uint chunk_start = c * BT;
        const uint chunk_len = min((uint)BT, seq_len - chunk_start);

        for (uint t = 0; t < chunk_len; ++t) {
            for (uint j = tid; j < BK; j += BV) {
                k_chunk[t * BK + j] = k_bh[(chunk_start + t) * BK + j];
            }
        }
        if (tid < chunk_len) {
            beta_s[tid] = beta_bh[chunk_start + tid];
            gcum[tid] = g_bh[chunk_start + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Prefix sum of gate values (parallel scan).
        for (uint stride = 1; stride < BT; stride <<= 1) {
            float prev = 0.0f;
            if (tid < chunk_len && tid >= stride) {
                prev = gcum[tid - stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < chunk_len && tid >= stride) {
                gcum[tid] += prev;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Precompute decay factors — eliminates all exp() from hot loops.
        if (tid < chunk_len) {
            const float d = ax_exp(gcum[tid]);
            decay[tid] = d;
            decay_inv[tid] = 1.0f / d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < chunk_len * chunk_len; idx += BV) {
            const uint i = idx / chunk_len;
            const uint j = idx % chunk_len;
            if (j < i) {
                float dot = 0.0f;
                for (uint d = 0; d < BK; ++d) {
                    dot = fma(k_chunk[i * BK + d], k_chunk[j * BK + d], dot);
                }
                kk_dot[i * BT + j] = dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Delta computation: uses precomputed decay[] and decay_inv[].
        for (uint i = 0; i < chunk_len; ++i) {
            const float v_i = v_bh[(chunk_start + i) * v_dim + v_idx];
            const float decay_i = decay[i];
            const float beta_i = beta_s[i];

            float kv_mem = 0.0f;
            for (uint d = 0; d < BK; ++d) {
                kv_mem = fma(s[d] * decay_i, k_chunk[i * BK + d], kv_mem);
            }

            float rhs = beta_i * (v_i - kv_mem);
            for (uint j = 0; j < i; ++j) {
                const float a_ij = beta_i * kk_dot[i * BT + j] * decay_i * decay_inv[j];
                rhs -= a_ij * delta_arr[j];
            }
            delta_arr[i] = rhs;
        }

        // Output computation: uses precomputed decay[]/decay_inv[].
        for (uint i = 0; i < chunk_len; ++i) {
            for (uint j = tid; j < BK; j += BV) {
                q_buf[j] = q_bh[(chunk_start + i) * BK + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float decay_i = decay[i];
            float o_val = 0.0f;
            for (uint d = 0; d < BK; ++d) {
                o_val = fma(q_buf[d], s[d] * decay_i, o_val);
            }

            for (uint j = 0; j <= i; ++j) {
                float qk_dot = 0.0f;
                for (uint d = 0; d < BK; ++d) {
                    qk_dot = fma(q_buf[d], k_chunk[j * BK + d], qk_dot);
                }
                o_val += qk_dot * delta_arr[j] * decay_i * decay_inv[j];
            }

            out_bh[(chunk_start + i) * v_dim + v_idx] = o_val * scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // State update: uses precomputed decay[].
        const float decay_total = decay[chunk_len - 1];
        for (uint d = 0; d < BK; ++d) {
            float s_new = s[d] * decay_total;
            for (uint t = 0; t < chunk_len; ++t) {
                s_new += k_chunk[t * BK + d] * delta_arr[t] * decay_total * decay_inv[t];
            }
            s[d] = s_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint j = 0; j < BK; ++j) {
        state_bh[j * v_dim + v_idx] = s[j];
    }
}

template [[host_name("gated_delta_rule_128_64")]] [[kernel]]
void gated_delta_rule_kernel<128, 64>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint2, uint);

template [[host_name("gated_delta_rule_128_128")]] [[kernel]]
void gated_delta_rule_kernel<128, 128>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint2, uint);

template [[host_name("gated_delta_rule_64_64")]] [[kernel]]
void gated_delta_rule_kernel<64, 64>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint2, uint);

template [[host_name("qwen35_single_token_gated_delta_fused_128_64")]] [[kernel]]
void qwen35_single_token_gated_delta_fused_kernel<128, 64>(
    const device float*, const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&, constant float&,
    uint2, uint, uint, uint);

template [[host_name("qwen35_single_token_gated_delta_fused_64_64")]] [[kernel]]
void qwen35_single_token_gated_delta_fused_kernel<64, 64>(
    const device float*, const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&, constant float&,
    uint2, uint, uint, uint);

template [[host_name("gated_delta_rule_fallback")]] [[kernel]]
void gated_delta_rule_kernel_fallback<64, 256>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&, constant uint&,
    threadgroup float*,
    uint2, uint);

template [[host_name("chunked_gated_delta_rule_32_128_64")]] [[kernel]]
void chunked_gated_delta_rule_kernel<32, 128, 64>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint2, uint);

template [[host_name("chunked_gated_delta_rule_32_128_128")]] [[kernel]]
void chunked_gated_delta_rule_kernel<32, 128, 128>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint2, uint);

template [[host_name("chunked_gated_delta_rule_32_64_64")]] [[kernel]]
void chunked_gated_delta_rule_kernel<32, 64, 64>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint2, uint);
