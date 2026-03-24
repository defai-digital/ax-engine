// GDN (Gated Delta Net) Metal kernels adapted from mistral.rs.

#include <metal_stdlib>
using namespace metal;

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

        out_bh[t * v_dim + v_idx] = y_t;
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

        out_bh[t * v_dim + v_idx] = y_t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint j = 0; j < k_dim; j++) {
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
