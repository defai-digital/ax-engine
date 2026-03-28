// GDN (Gated Delta Net) Metal kernels adapted from mistral.rs.
//
// AX keeps the Qwen3.5 causal-conv path in token-major f32 layout:
// - input/output: [seq_len, conv_dim]
// - kernel: [conv_cache_len + 1, conv_dim]
// - conv_state: [conv_cache_len, conv_dim]
//
// This matches the existing CPU helper and avoids extra layout transforms in
// the first Metal port of the recurrent conv path.

#include <metal_stdlib>
using namespace metal;

constant uint QWEN35_MAX_CONV_CACHE = 8;

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

        float decay = exp(g_bh[t]);
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

        float decay = exp(g_bh[t]);
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

        for (uint i = 0; i < chunk_len; ++i) {
            const float v_i = v_bh[(chunk_start + i) * v_dim + v_idx];
            const float decay_i = exp(gcum[i]);
            const float beta_i = beta_s[i];

            float kv_mem = 0.0f;
            for (uint d = 0; d < BK; ++d) {
                kv_mem = fma(s[d] * decay_i, k_chunk[i * BK + d], kv_mem);
            }

            float rhs = beta_i * (v_i - kv_mem);
            for (uint j = 0; j < i; ++j) {
                const float a_ij = beta_i * kk_dot[i * BT + j] * exp(gcum[i] - gcum[j]);
                rhs -= a_ij * delta_arr[j];
            }
            delta_arr[i] = rhs;
        }

        for (uint i = 0; i < chunk_len; ++i) {
            for (uint j = tid; j < BK; j += BV) {
                q_buf[j] = q_bh[(chunk_start + i) * BK + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float decay_i = exp(gcum[i]);
            float o_val = 0.0f;
            for (uint d = 0; d < BK; ++d) {
                o_val = fma(q_buf[d], s[d] * decay_i, o_val);
            }

            for (uint j = 0; j <= i; ++j) {
                float qk_dot = 0.0f;
                for (uint d = 0; d < BK; ++d) {
                    qk_dot = fma(q_buf[d], k_chunk[j * BK + d], qk_dot);
                }
                o_val += qk_dot * delta_arr[j] * exp(gcum[i] - gcum[j]);
            }

            out_bh[(chunk_start + i) * v_dim + v_idx] = o_val * scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float g_total = gcum[chunk_len - 1];
        for (uint d = 0; d < BK; ++d) {
            float s_new = s[d] * exp(g_total);
            for (uint t = 0; t < chunk_len; ++t) {
                s_new += k_chunk[t * BK + d] * delta_arr[t] * exp(g_total - gcum[t]);
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

template [[host_name("gated_delta_rule_64_64")]] [[kernel]]
void gated_delta_rule_kernel<64, 64>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint2, uint);

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

template [[host_name("chunked_gated_delta_rule_32_64_64")]] [[kernel]]
void chunked_gated_delta_rule_kernel<32, 64, 64>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant uint&, constant uint&,
    uint2, uint);
