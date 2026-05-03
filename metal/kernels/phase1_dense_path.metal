#include <metal_stdlib>

using namespace metal;

// These kernels intentionally keep the first AX v4 Metal surface narrow.
// They now execute a repo-owned numeric bring-up path while still keeping the
// contract surface small enough for early validation and toolchain hardening.

struct CacheDispatchParams {
    uint element_count;
    uint head_size;
};

struct AttentionDispatchParams {
    uint element_count;
    uint num_seqs;
    uint head_count;
    uint head_dim;
};

struct GatherDispatchParams {
    uint element_count;
    uint num_seqs;
    uint block_size_tokens;
    uint block_table_stride;
    uint head_size;
};

struct CopyBlockDispatchParams {
    uint num_pairs;
    uint numel_per_block_key;
    uint numel_per_block_value;
    uint head_size;
};

struct ScaleParams {
    uint element_count;
    float scale;
};

struct LogitsProjectionParams {
    uint vocab_rows;
    uint projection_cols;
    uint input_width;
};

struct Q4KMProjectionParams {
    uint n_rows;        // output rows
    uint input_width;   // hidden dimension (must be a multiple of 256)
};

struct BatchedLogitsProjectionParams {
    uint token_count;
    uint vocab_rows;
    uint projection_cols;
    uint input_width;
    uint hidden_stride;
};

struct LogitsArgmaxParams {
    uint element_count;
};

struct BatchedLogitsArgmaxParams {
    uint token_count;
    uint vocab_rows;
};

struct RmsNormParams {
    uint element_count;
    float epsilon;
    float weight_offset;
};

struct BatchedRmsNormParams {
    uint head_count;
    uint head_dim;
    float epsilon;
    float weight_offset;
};

struct FfnGateProductParams {
    uint element_count;
};

struct RopeDispatchParams {
    uint query_head_count;
    uint key_head_count;
    uint head_dim;
    uint rope_style;
    uint rotary_dim;
};

struct BatchedRopeDispatchParams {
    uint token_count;
    uint query_head_count;
    uint key_head_count;
    uint head_dim;
    uint rope_style;
    float freq_base;
    uint rotary_dim;
};

struct GroupedKvExpandParams {
    uint output_element_count;
    uint kv_head_count;
    uint heads_per_kv;
    uint head_dim;
};

struct EmbeddingGatherParams {
    uint token_count;
    uint embedding_rows;
    uint hidden_dim;
    float scale;
};

struct VectorAddParams {
    uint element_count;
};

struct RowScaleParams {
    uint row_count;
    uint row_width;
};

struct RowVectorScaleParams {
    uint row_count;
    uint row_width;
};

struct LinearGatedDeltaParams {
    uint batch_size;
    uint num_key_heads;
    uint num_value_heads;
    uint key_head_dim;
    uint value_head_dim;
    uint repeat_factor;
};

struct LinearAttentionConvParams {
    uint batch_size;
    uint conv_dim;
    uint conv_kernel_dim;
};

kernel void reshape_and_cache(
    device const float* key [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* key_cache [[buffer(2)]],
    device float* value_cache [[buffer(3)]],
    device const uint* slot_mapping [[buffer(4)]],
    constant CacheDispatchParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    uint token_id = gid / params.head_size;
    uint lane_id = gid % params.head_size;
    uint slot = slot_mapping[token_id];
    uint cache_index = slot * params.head_size + lane_id;
    uint source_index = token_id * params.head_size + lane_id;
    key_cache[cache_index] = key[source_index];
    value_cache[cache_index] = value[source_index];
}

// Per-head fused online-softmax attention.
// Dispatch: n_tokens * head_count threadgroups, head_dim threads per TG.
// Each TG handles one (token, head) pair using online softmax (single KV pass).
// Threads collaborate on QK dot products via SIMD+threadgroup reduction,
// then each thread independently accumulates its value dimension.
kernel void paged_decode_attention(
    device const float* query          [[buffer(0)]],
    device const float* key_gathered   [[buffer(1)]],
    device const float* value_gathered [[buffer(2)]],
    device const uint*  cu_seq_lens    [[buffer(3)]],
    device const uint*  scheduled_cu_seq_lens [[buffer(4)]],
    device float* output               [[buffer(5)]],
    constant AttentionDispatchParams& params [[buffer(6)]],
    uint tg_pos [[threadgroup_position_in_grid]],   // token_id * head_count + head_id
    uint lid    [[thread_index_in_threadgroup]],     // 0 .. head_dim-1
    uint lane   [[thread_index_in_simdgroup]],
    uint simd   [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]]
) {
    if (params.head_dim == 0) return;

    uint head_size = params.head_count * params.head_dim;
    uint n_tokens  = params.element_count / head_size;
    uint token_id  = tg_pos / params.head_count;
    uint head_id   = tg_pos % params.head_count;
    if (token_id >= n_tokens) return;

    threadgroup float smem[32];  // per-SIMD partial sums (max 32 SIMD groups = 1024-thread TG)

    // Binary search: find which request this token belongs to
    uint batch_lo = 0, batch_hi = params.num_seqs;
    while (batch_lo < batch_hi) {
        uint mid = (batch_lo + batch_hi + 1) / 2;
        if (scheduled_cu_seq_lens[mid] <= token_id) batch_lo = mid;
        else batch_hi = mid - 1;
    }
    uint context_begin = cu_seq_lens[batch_lo];
    uint context_end   = cu_seq_lens[batch_lo + 1];

    uint q_base    = token_id * head_size + head_id * params.head_dim;
    float q_val    = (lid < params.head_dim) ? query[q_base + lid] : 0.0f;
    float inv_scale = rsqrt(float(params.head_dim));

    // Online softmax accumulators (per-thread, for this thread's value dimension)
    float m   = -INFINITY;
    float d   = 0.0f;
    float acc = 0.0f;

    for (uint c = context_begin; c < context_end; c++) {
        uint kv_base = c * head_size + head_id * params.head_dim;

        // Compute dot(q, k[c]) collaboratively across the TG
        float k_val = (lid < params.head_dim) ? key_gathered[kv_base + lid] : 0.0f;
        float partial_qk = simd_sum(q_val * k_val);
        if (lane == 0) smem[simd] = partial_qk;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = 0.0f;
        if (simd == 0) {
            float v = (lane < n_simd) ? smem[lane] : 0.0f;
            score = simd_sum(v) * inv_scale;
            if (lane == 0) smem[0] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        score = smem[0];

        // Online softmax update for this thread's output dimension (lid)
        float m_new    = max(m, score);
        float exp_diff = exp(m - m_new);
        float exp_s    = exp(score - m_new);

        float v_val = (lid < params.head_dim) ? value_gathered[kv_base + lid] : 0.0f;
        d   = d   * exp_diff + exp_s;
        acc = acc * exp_diff + exp_s * v_val;
        m   = m_new;
    }

    if (lid < params.head_dim) {
        output[q_base + lid] = acc / max(d, 0.000001f);
    }
}

kernel void gather_kv_cache(
    device const float* key_cache [[buffer(0)]],
    device const float* value_cache [[buffer(1)]],
    device const uint* block_table [[buffer(2)]],
    device const uint* cu_seq_lens [[buffer(3)]],
    device float* key_gathered [[buffer(4)]],
    device float* value_gathered [[buffer(5)]],
    constant GatherDispatchParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    uint token_id = gid / params.head_size;
    uint lane_id = gid % params.head_size;

    uint batch_lo = 0;
    uint batch_hi = params.num_seqs;
    while (batch_lo < batch_hi) {
        uint batch_mid = (batch_lo + batch_hi + 1) / 2;
        if (cu_seq_lens[batch_mid] <= token_id) {
            batch_lo = batch_mid;
        } else {
            batch_hi = batch_mid - 1;
        }
    }

    uint batch_id = batch_lo;
    uint batch_offset = token_id - cu_seq_lens[batch_id];
    uint block_index = batch_offset / params.block_size_tokens;
    uint block_offset = batch_offset % params.block_size_tokens;
    uint block_base = block_table[batch_id * params.block_table_stride + block_index];
    uint slot = block_base + block_offset;
    uint source_index = slot * params.head_size + lane_id;
    uint target_index = token_id * params.head_size + lane_id;

    key_gathered[target_index] = key_cache[source_index];
    value_gathered[target_index] = value_cache[source_index];
}

kernel void copy_blocks(
    device const float* key_source [[buffer(0)]],
    device const float* value_source [[buffer(1)]],
    device float* key_target [[buffer(2)]],
    device float* value_target [[buffer(3)]],
    device const uint2* block_base_mapping [[buffer(4)]],
    constant CopyBlockDispatchParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint elements_per_pair = max(params.numel_per_block_key, params.numel_per_block_value);
    uint total_elements = params.num_pairs * elements_per_pair;
    if (gid >= total_elements) {
        return;
    }

    uint pair_index = gid / elements_per_pair;
    uint block_offset = gid % elements_per_pair;
    uint2 pair = block_base_mapping[pair_index];
    uint source_key_base = pair.x * params.head_size;
    uint target_key_base = pair.y * params.head_size;
    if (block_offset < params.numel_per_block_key) {
        key_target[target_key_base + block_offset] = key_source[source_key_base + block_offset];
    }
    if (block_offset < params.numel_per_block_value) {
        value_target[target_key_base + block_offset] =
            value_source[source_key_base + block_offset];
    }
}

// Deferred for Phase 1 runtime dispatch, but kept explicit in the checked-in
// manifest so future remap / offload work does not silently invent a new
// entry-point name later.
kernel void swap_blocks(
    device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const uint2* block_mapping [[buffer(2)]],
    constant CacheDispatchParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    uint2 pair = block_mapping[gid];
    float tmp = src[pair.x];
    src[pair.x] = dst[pair.y];
    dst[pair.y] = tmp;
}

kernel void kv_scale_update(
    device float* key_cache [[buffer(0)]],
    device float* value_cache [[buffer(1)]],
    constant ScaleParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    key_cache[gid] *= params.scale;
    value_cache[gid] *= params.scale;
}

kernel void linear_gated_delta_step_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* g [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device const float* state_in [[buffer(5)]],
    device float* output [[buffer(6)]],
    device float* state_out [[buffer(7)]],
    constant LinearGatedDeltaParams& params [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (gid.z >= params.batch_size * params.num_value_heads || gid.y >= params.value_head_dim) {
        return;
    }

    uint batch_idx = gid.z / params.num_value_heads;
    uint value_head_idx = gid.z % params.num_value_heads;
    uint key_head_idx = value_head_idx / max(params.repeat_factor, 1u);
    if (key_head_idx >= params.num_key_heads) {
        return;
    }

    uint key_dim = params.key_head_dim;
    uint value_dim = params.value_head_dim;
    uint qk_base = (batch_idx * params.num_key_heads + key_head_idx) * key_dim;
    uint value_base = (batch_idx * params.num_value_heads + value_head_idx) * value_dim;
    uint state_base = ((batch_idx * params.num_value_heads + value_head_idx) * value_dim + gid.y) * key_dim;
    float decay = g[batch_idx * params.num_value_heads + value_head_idx];
    float blend = beta[batch_idx * params.num_value_heads + value_head_idx];

    float kv_mem = 0.0f;
    constexpr uint simd_width = 32;
    for (uint key_lane = lane; key_lane < key_dim; key_lane += simd_width) {
        uint state_index = state_base + key_lane;
        float decayed = state_in[state_index] * decay;
        state_out[state_index] = decayed;
        kv_mem += decayed * k[qk_base + key_lane];
    }
    kv_mem = simd_sum(kv_mem);

    float delta = (v[value_base + gid.y] - kv_mem) * blend;
    float out = 0.0f;
    for (uint key_lane = lane; key_lane < key_dim; key_lane += simd_width) {
        uint state_index = state_base + key_lane;
        float updated = state_out[state_index] + k[qk_base + key_lane] * delta;
        state_out[state_index] = updated;
        out += updated * q[qk_base + key_lane];
    }
    out = simd_sum(out);
    if (lane == 0) {
        output[value_base + gid.y] = out;
    }
}

kernel void linear_attention_conv1d_f32(
    device const float* qkv [[buffer(0)]],
    device const float* conv_weight [[buffer(1)]],
    device const float* state_in [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* state_out [[buffer(4)]],
    constant LinearAttentionConvParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_elements = params.batch_size * params.conv_dim;
    if (gid >= total_elements || params.conv_dim == 0 || params.conv_kernel_dim == 0) {
        return;
    }

    uint batch_idx = gid / params.conv_dim;
    uint channel = gid % params.conv_dim;
    uint qkv_base = batch_idx * params.conv_dim;
    uint state_batch_base = batch_idx * max(params.conv_kernel_dim - 1, 0u) * params.conv_dim;
    float acc = 0.0f;
    for (uint tap = 0; tap + 1 < params.conv_kernel_dim; ++tap) {
        uint state_index = state_batch_base + tap * params.conv_dim + channel;
        acc += state_in[state_index] * conv_weight[channel * params.conv_kernel_dim + tap];
    }
    acc += qkv[qkv_base + channel]
        * conv_weight[channel * params.conv_kernel_dim + (params.conv_kernel_dim - 1)];
    output[gid] = acc / (1.0f + exp(-acc));

    if (params.conv_kernel_dim > 1) {
        for (uint tap = 0; tap + 2 < params.conv_kernel_dim; ++tap) {
            uint dst_index = state_batch_base + tap * params.conv_dim + channel;
            uint src_index = state_batch_base + (tap + 1) * params.conv_dim + channel;
            state_out[dst_index] = state_in[src_index];
        }
        uint tail_index = state_batch_base + (params.conv_kernel_dim - 2) * params.conv_dim + channel;
        state_out[tail_index] = qkv[qkv_base + channel];
    }
}

kernel void linear_attention_conv1d_f16(
    device const float* qkv [[buffer(0)]],
    device const half* conv_weight [[buffer(1)]],
    device const float* state_in [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* state_out [[buffer(4)]],
    constant LinearAttentionConvParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_elements = params.batch_size * params.conv_dim;
    if (gid >= total_elements || params.conv_dim == 0 || params.conv_kernel_dim == 0) {
        return;
    }

    uint batch_idx = gid / params.conv_dim;
    uint channel = gid % params.conv_dim;
    uint qkv_base = batch_idx * params.conv_dim;
    uint state_batch_base = batch_idx * max(params.conv_kernel_dim - 1, 0u) * params.conv_dim;
    float acc = 0.0f;
    for (uint tap = 0; tap + 1 < params.conv_kernel_dim; ++tap) {
        uint state_index = state_batch_base + tap * params.conv_dim + channel;
        acc += state_in[state_index] * float(conv_weight[channel * params.conv_kernel_dim + tap]);
    }
    acc += qkv[qkv_base + channel]
        * float(conv_weight[channel * params.conv_kernel_dim + (params.conv_kernel_dim - 1)]);
    output[gid] = acc / (1.0f + exp(-acc));

    if (params.conv_kernel_dim > 1) {
        for (uint tap = 0; tap + 2 < params.conv_kernel_dim; ++tap) {
            uint dst_index = state_batch_base + tap * params.conv_dim + channel;
            uint src_index = state_batch_base + (tap + 1) * params.conv_dim + channel;
            state_out[dst_index] = state_in[src_index];
        }
        uint tail_index = state_batch_base + (params.conv_kernel_dim - 2) * params.conv_dim + channel;
        state_out[tail_index] = qkv[qkv_base + channel];
    }
}

kernel void linear_attention_conv1d_bf16(
    device const float* qkv [[buffer(0)]],
    device const bfloat* conv_weight [[buffer(1)]],
    device const float* state_in [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* state_out [[buffer(4)]],
    constant LinearAttentionConvParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_elements = params.batch_size * params.conv_dim;
    if (gid >= total_elements || params.conv_dim == 0 || params.conv_kernel_dim == 0) {
        return;
    }

    uint batch_idx = gid / params.conv_dim;
    uint channel = gid % params.conv_dim;
    uint qkv_base = batch_idx * params.conv_dim;
    uint state_batch_base = batch_idx * max(params.conv_kernel_dim - 1, 0u) * params.conv_dim;
    float acc = 0.0f;
    for (uint tap = 0; tap + 1 < params.conv_kernel_dim; ++tap) {
        uint state_index = state_batch_base + tap * params.conv_dim + channel;
        acc += state_in[state_index] * float(conv_weight[channel * params.conv_kernel_dim + tap]);
    }
    acc += qkv[qkv_base + channel]
        * float(conv_weight[channel * params.conv_kernel_dim + (params.conv_kernel_dim - 1)]);
    output[gid] = acc / (1.0f + exp(-acc));

    if (params.conv_kernel_dim > 1) {
        for (uint tap = 0; tap + 2 < params.conv_kernel_dim; ++tap) {
            uint dst_index = state_batch_base + tap * params.conv_dim + channel;
            uint src_index = state_batch_base + (tap + 1) * params.conv_dim + channel;
            state_out[dst_index] = state_in[src_index];
        }
        uint tail_index = state_batch_base + (params.conv_kernel_dim - 2) * params.conv_dim + channel;
        state_out[tail_index] = qkv[qkv_base + channel];
    }
}

kernel void gather_embedding_rows_f32(
    device const uint* token_ids [[buffer(0)]],
    device const float* embedding [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant EmbeddingGatherParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.token_count * params.hidden_dim || params.embedding_rows == 0) {
        return;
    }

    uint token_index = gid / params.hidden_dim;
    uint lane_index = gid % params.hidden_dim;
    uint row_index = token_ids[token_index] % params.embedding_rows;
    output[gid] = embedding[row_index * params.hidden_dim + lane_index] * params.scale;
}

kernel void gather_embedding_rows_f16(
    device const uint* token_ids [[buffer(0)]],
    device const half* embedding [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant EmbeddingGatherParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.token_count * params.hidden_dim || params.embedding_rows == 0) {
        return;
    }

    uint token_index = gid / params.hidden_dim;
    uint lane_index = gid % params.hidden_dim;
    uint row_index = token_ids[token_index] % params.embedding_rows;
    output[gid] = float(embedding[row_index * params.hidden_dim + lane_index]) * params.scale;
}

kernel void gather_embedding_rows_bf16(
    device const uint* token_ids [[buffer(0)]],
    device const bfloat* embedding [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant EmbeddingGatherParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.token_count * params.hidden_dim || params.embedding_rows == 0) {
        return;
    }

    uint token_index = gid / params.hidden_dim;
    uint lane_index = gid % params.hidden_dim;
    uint row_index = token_ids[token_index] % params.embedding_rows;
    output[gid] = float(embedding[row_index * params.hidden_dim + lane_index]) * params.scale;
}

kernel void vector_add_f32(
    device const float* input [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant VectorAddParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    output[gid] = input[gid] + delta[gid];
}

kernel void row_scale_f32(
    device const float* input [[buffer(0)]],
    device const float* row_scales [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RowScaleParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint element_count = params.row_count * params.row_width;
    if (gid >= element_count || params.row_width == 0) {
        return;
    }

    uint row_index = gid / params.row_width;
    output[gid] = input[gid] * row_scales[row_index];
}

kernel void row_vector_scale_f32(
    device const float* input [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RowVectorScaleParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint element_count = params.row_count * params.row_width;
    if (gid >= element_count || params.row_width == 0) {
        return;
    }

    uint lane_index = gid % params.row_width;
    output[gid] = input[gid] * scales[lane_index];
}

// Projection kernels use one simd-group (32 threads) per output row.
// Each thread handles every 32nd input element; simd_sum reduces to the dot
// product. This achieves near-memory-bandwidth-limited throughput for GEMV
// (single-token decode) by fully utilising the GPU's SIMD parallelism.
// Dispatch: vocab_rows * 32 total threads, threadgroup size 32.

kernel void decode_logits_projection_f32(
    device const float* hidden     [[buffer(0)]],
    device const float* projection [[buffer(1)]],
    device float*       logits     [[buffer(2)]],
    constant LogitsProjectionParams& params [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint out_row = gid / 32;
    if (out_row >= params.vocab_rows) return;

    uint row_base = out_row * params.projection_cols;
    float partial = 0.0f;
    for (uint col = lane; col < params.input_width; col += 32) {
        partial += projection[row_base + col] * hidden[col];
    }
    if (lane == 0) {
        logits[out_row] = simd_sum(partial);
    }
}

kernel void decode_logits_projection_f16(
    device const float* hidden     [[buffer(0)]],
    device const half*  projection [[buffer(1)]],
    device float*       logits     [[buffer(2)]],
    constant LogitsProjectionParams& params [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint out_row = gid / 32;
    if (out_row >= params.vocab_rows) return;

    uint row_base = out_row * params.projection_cols;
    float partial = 0.0f;
    for (uint col = lane; col < params.input_width; col += 32) {
        partial += float(projection[row_base + col]) * hidden[col];
    }
    if (lane == 0) {
        logits[out_row] = simd_sum(partial);
    }
}

kernel void decode_logits_projection_bf16(
    device const float*  hidden     [[buffer(0)]],
    device const bfloat* projection [[buffer(1)]],
    device float*        logits     [[buffer(2)]],
    constant LogitsProjectionParams& params [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint out_row = gid / 32;
    if (out_row >= params.vocab_rows) return;

    uint row_base = out_row * params.projection_cols;
    float partial = 0.0f;
    for (uint col = lane; col < params.input_width; col += 32) {
        partial += float(projection[row_base + col]) * hidden[col];
    }
    if (lane == 0) {
        logits[out_row] = simd_sum(partial);
    }
}

// Batched projection: one simd-group per (token, output_row) pair.
// Dispatch: token_count * vocab_rows * 32 total threads, threadgroup size 32.

kernel void decode_logits_projection_batched_f32(
    device const float* hidden     [[buffer(0)]],
    device const float* projection [[buffer(1)]],
    device float*       logits     [[buffer(2)]],
    constant BatchedLogitsProjectionParams& params [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint element_count = params.token_count * params.vocab_rows;
    uint flat_index = gid / 32;
    if (flat_index >= element_count || params.hidden_stride == 0) return;

    uint token_index = flat_index / params.vocab_rows;
    uint vocab_index = flat_index % params.vocab_rows;
    uint row_base    = vocab_index * params.projection_cols;
    uint hidden_base = token_index * params.hidden_stride;
    float partial = 0.0f;
    for (uint col = lane; col < params.input_width; col += 32) {
        partial += projection[row_base + col] * hidden[hidden_base + col];
    }
    if (lane == 0) {
        logits[flat_index] = simd_sum(partial);
    }
}

kernel void decode_logits_projection_batched_f16(
    device const float* hidden     [[buffer(0)]],
    device const half*  projection [[buffer(1)]],
    device float*       logits     [[buffer(2)]],
    constant BatchedLogitsProjectionParams& params [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint element_count = params.token_count * params.vocab_rows;
    uint flat_index = gid / 32;
    if (flat_index >= element_count || params.hidden_stride == 0) return;

    uint token_index = flat_index / params.vocab_rows;
    uint vocab_index = flat_index % params.vocab_rows;
    uint row_base    = vocab_index * params.projection_cols;
    uint hidden_base = token_index * params.hidden_stride;
    float partial = 0.0f;
    for (uint col = lane; col < params.input_width; col += 32) {
        partial += float(projection[row_base + col]) * hidden[hidden_base + col];
    }
    if (lane == 0) {
        logits[flat_index] = simd_sum(partial);
    }
}

kernel void decode_logits_projection_batched_bf16(
    device const float*  hidden     [[buffer(0)]],
    device const bfloat* projection [[buffer(1)]],
    device float*        logits     [[buffer(2)]],
    constant BatchedLogitsProjectionParams& params [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint element_count = params.token_count * params.vocab_rows;
    uint flat_index = gid / 32;
    if (flat_index >= element_count || params.hidden_stride == 0) return;

    uint token_index = flat_index / params.vocab_rows;
    uint vocab_index = flat_index % params.vocab_rows;
    uint row_base    = vocab_index * params.projection_cols;
    uint hidden_base = token_index * params.hidden_stride;
    float partial = 0.0f;
    for (uint col = lane; col < params.input_width; col += 32) {
        partial += float(projection[row_base + col]) * hidden[hidden_base + col];
    }
    if (lane == 0) {
        logits[flat_index] = simd_sum(partial);
    }
}

// Parallel argmax using SIMD max + min-index selection.
// Dispatch with ARGMAX_TG_SIZE threads (1 threadgroup).
// smem_val/smem_idx hold per-SIMD-group results (max 32 SIMD groups for TG ≤ 1024).
kernel void logits_argmax_f32(
    device const float* logits [[buffer(0)]],
    device uint* best_index [[buffer(1)]],
    constant LogitsArgmaxParams& params [[buffer(2)]],
    uint gid   [[thread_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint simd  [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total [[threads_per_threadgroup]]
) {
    uint n = params.element_count;
    if (n == 0) return;

    threadgroup float smem_val[32];
    threadgroup uint  smem_idx[32];

    float local_val = -INFINITY;
    uint  local_idx = 0;
    for (uint i = gid; i < n; i += total) {
        float v = logits[i];
        if (v > local_val) { local_val = v; local_idx = i; }
    }

    float simd_max_v = simd_max(local_val);
    uint  candidate  = (local_val >= simd_max_v) ? local_idx : UINT_MAX;
    uint  simd_best  = simd_min(candidate);

    if (lane == 0) { smem_val[simd] = simd_max_v; smem_idx[simd] = simd_best; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float best_v = smem_val[0];
        uint  best_i = smem_idx[0];
        for (uint s = 1; s < n_simd; s++) {
            if (smem_val[s] > best_v) { best_v = smem_val[s]; best_i = smem_idx[s]; }
        }
        best_index[0] = best_i;
    }
}

// Parallel batched argmax: 1 threadgroup per token.
// Dispatch: token_count * ARGMAX_TG_SIZE threads, ARGMAX_TG_SIZE per TG.
kernel void logits_argmax_batched_f32(
    device const float* logits [[buffer(0)]],
    device uint* best_index [[buffer(1)]],
    constant BatchedLogitsArgmaxParams& params [[buffer(2)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint lid    [[thread_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint simd   [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total  [[threads_per_threadgroup]]
) {
    if (tg_pos >= params.token_count || params.vocab_rows == 0) return;

    threadgroup float smem_val[32];
    threadgroup uint  smem_idx[32];

    uint base = tg_pos * params.vocab_rows;
    uint n    = params.vocab_rows;

    float local_val = -INFINITY;
    uint  local_idx = 0;
    for (uint i = lid; i < n; i += total) {
        float v = logits[base + i];
        if (v > local_val) { local_val = v; local_idx = i; }
    }

    float simd_max_v = simd_max(local_val);
    uint  candidate  = (local_val >= simd_max_v) ? local_idx : UINT_MAX;
    uint  simd_best  = simd_min(candidate);

    if (lane == 0) { smem_val[simd] = simd_max_v; smem_idx[simd] = simd_best; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float best_v = smem_val[0];
        uint  best_i = smem_idx[0];
        for (uint s = 1; s < n_simd; s++) {
            if (smem_val[s] > best_v) { best_v = smem_val[s]; best_i = smem_idx[s]; }
        }
        best_index[tg_pos] = best_i;
    }
}

// Parallel argmax + logprob.  Two stride-loop passes (argmax then logsumexp)
// within the same dispatch, communicating via threadgroup memory.
// Dispatch: ARGMAX_TG_SIZE threads, 1 threadgroup.
kernel void sample_argmax_logprob_f32(
    device const float* logits [[buffer(0)]],
    device uint* best_index [[buffer(1)]],
    device float* best_logprob [[buffer(2)]],
    constant LogitsArgmaxParams& params [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint simd  [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total [[threads_per_threadgroup]]
) {
    uint n = params.element_count;
    if (n == 0) return;

    threadgroup float smem_val[32];
    threadgroup uint  smem_idx[32];

    // Pass 1: parallel argmax
    float local_val = -INFINITY;
    uint  local_idx = 0;
    for (uint i = gid; i < n; i += total) {
        float v = logits[i];
        if (v > local_val) { local_val = v; local_idx = i; }
    }

    float simd_max_v = simd_max(local_val);
    uint  candidate  = (local_val >= simd_max_v) ? local_idx : UINT_MAX;
    uint  simd_best  = simd_min(candidate);

    if (lane == 0) { smem_val[simd] = simd_max_v; smem_idx[simd] = simd_best; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best_score = smem_val[0];
    uint  winner_idx = smem_idx[0];
    if (lid == 0) {
        for (uint s = 1; s < n_simd; s++) {
            if (smem_val[s] > best_score) { best_score = smem_val[s]; winner_idx = smem_idx[s]; }
        }
        smem_val[0] = best_score;
        smem_idx[0] = winner_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best_score = smem_val[0];
    winner_idx = smem_idx[0];

    // Pass 2: parallel logsumexp
    float partial_exp = 0.0f;
    for (uint i = gid; i < n; i += total) {
        partial_exp += exp(logits[i] - best_score);
    }
    float simd_exp = simd_sum(partial_exp);
    if (lane == 0) smem_val[simd] = simd_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float total_exp = 0.0f;
        for (uint s = 0; s < n_simd; s++) total_exp += smem_val[s];
        if (isfinite(total_exp) && total_exp > 0.0f) {
            best_index[0]  = winner_idx;
            best_logprob[0] = -log(total_exp);
        }
    }
}


// Parallel batched argmax + logprob: 1 threadgroup per token, 2-pass within TG.
// Dispatch: token_count * ARGMAX_TG_SIZE threads, ARGMAX_TG_SIZE per TG.
kernel void sample_argmax_logprob_batched_f32(
    device const float* logits [[buffer(0)]],
    device uint*  best_index  [[buffer(1)]],
    device float* best_logprob [[buffer(2)]],
    constant BatchedLogitsArgmaxParams& params [[buffer(3)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint lid    [[thread_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint simd   [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total  [[threads_per_threadgroup]]
) {
    if (tg_pos >= params.token_count || params.vocab_rows == 0) return;

    threadgroup float smem_val[32];
    threadgroup uint  smem_idx[32];

    uint base = tg_pos * params.vocab_rows;
    uint n    = params.vocab_rows;

    // Pass 1: parallel argmax
    float local_val = -INFINITY;
    uint  local_idx = 0;
    for (uint i = lid; i < n; i += total) {
        float v = logits[base + i];
        if (v > local_val) { local_val = v; local_idx = i; }
    }

    float simd_max_v = simd_max(local_val);
    uint  candidate  = (local_val >= simd_max_v) ? local_idx : UINT_MAX;
    uint  simd_best  = simd_min(candidate);

    if (lane == 0) { smem_val[simd] = simd_max_v; smem_idx[simd] = simd_best; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best_score = smem_val[0];
    uint  winner     = smem_idx[0];
    if (lid == 0) {
        for (uint s = 1; s < n_simd; s++) {
            if (smem_val[s] > best_score) { best_score = smem_val[s]; winner = smem_idx[s]; }
        }
        smem_val[0] = best_score;
        smem_idx[0] = winner;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best_score = smem_val[0];
    winner     = smem_idx[0];

    // Pass 2: parallel logsumexp
    float partial_exp = 0.0f;
    for (uint i = lid; i < n; i += total) {
        partial_exp += exp(logits[base + i] - best_score);
    }
    float simd_exp = simd_sum(partial_exp);
    if (lane == 0) smem_val[simd] = simd_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float total_exp = 0.0f;
        for (uint s = 0; s < n_simd; s++) total_exp += smem_val[s];
        if (isfinite(total_exp) && total_exp > 0.0f) {
            best_index[tg_pos]  = winner;
            best_logprob[tg_pos] = -log(total_exp);
        }
    }
}

// Parallel rms_norm: stride-loop over input, SIMD + threadgroup reduction.
// Dispatch: NORM_TG_SIZE threads in 1 threadgroup (NORM_TG_SIZE ≤ 1024).
// Works for any element_count; threads stride through the vector.
kernel void rms_norm_f32(
    device const float* input   [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]],
    uint simd  [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total [[threads_per_threadgroup]]
) {
    uint n = params.element_count;
    if (n == 0) return;

    threadgroup float smem[32];

    float sum_sq = 0.0f;
    for (uint i = gid; i < n; i += total) { float v = input[i]; sum_sq += v * v; }

    sum_sq = simd_sum(sum_sq);
    if (lane == 0) smem[simd] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0) {
        float v = (lane < n_simd) ? smem[lane] : 0.0f;
        float total_sq = simd_sum(v);
        if (lane == 0) smem[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean_sq = smem[0] / float(n);
    float denom = sqrt(mean_sq + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) return;

    for (uint i = gid; i < n; i += total) {
        output[i] = (input[i] / denom) * (weights[i] + params.weight_offset);
    }
}

kernel void rms_norm_f16(
    device const float* input   [[buffer(0)]],
    device const half*  weights [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]],
    uint simd  [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total [[threads_per_threadgroup]]
) {
    uint n = params.element_count;
    if (n == 0) return;

    threadgroup float smem[32];

    float sum_sq = 0.0f;
    for (uint i = gid; i < n; i += total) { float v = input[i]; sum_sq += v * v; }

    sum_sq = simd_sum(sum_sq);
    if (lane == 0) smem[simd] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0) {
        float v = (lane < n_simd) ? smem[lane] : 0.0f;
        float total_sq = simd_sum(v);
        if (lane == 0) smem[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean_sq = smem[0] / float(n);
    float denom = sqrt(mean_sq + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) return;

    for (uint i = gid; i < n; i += total) {
        output[i] = (input[i] / denom) * (float(weights[i]) + params.weight_offset);
    }
}

kernel void rms_norm_bf16(
    device const float*  input   [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device float*        output  [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]],
    uint simd  [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total [[threads_per_threadgroup]]
) {
    uint n = params.element_count;
    if (n == 0) return;

    threadgroup float smem[32];

    float sum_sq = 0.0f;
    for (uint i = gid; i < n; i += total) { float v = input[i]; sum_sq += v * v; }

    sum_sq = simd_sum(sum_sq);
    if (lane == 0) smem[simd] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0) {
        float v = (lane < n_simd) ? smem[lane] : 0.0f;
        float total_sq = simd_sum(v);
        if (lane == 0) smem[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean_sq = smem[0] / float(n);
    float denom = sqrt(mean_sq + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) return;

    for (uint i = gid; i < n; i += total) {
        output[i] = (input[i] / denom) * (float(weights[i]) + params.weight_offset);
    }
}

// Parallel batched rms_norm: 1 threadgroup per head, stride-loop within TG.
// Dispatch: NORM_TG_SIZE * head_count total threads, NORM_TG_SIZE per TG.
// Each TG handles one head, identified by threadgroup_position_in_grid.
kernel void rms_norm_batched_f32(
    device const float* input   [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant BatchedRmsNormParams& params [[buffer(3)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint lid    [[thread_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint simd   [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total  [[threads_per_threadgroup]]
) {
    uint n = params.head_dim;
    uint head_idx = tg_idx;
    if (head_idx >= params.head_count || n == 0) return;

    uint head_base = head_idx * n;

    threadgroup float smem[32];

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += total) { float v = input[head_base + i]; sum_sq += v * v; }

    sum_sq = simd_sum(sum_sq);
    if (lane == 0) smem[simd] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0) {
        float v = (lane < n_simd) ? smem[lane] : 0.0f;
        float total_sq = simd_sum(v);
        if (lane == 0) smem[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean_sq = smem[0] / float(n);
    float denom = sqrt(mean_sq + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) return;

    for (uint i = lid; i < n; i += total) {
        output[head_base + i] = (input[head_base + i] / denom) *
            (weights[i] + params.weight_offset);
    }
}

kernel void rms_norm_batched_f16(
    device const float* input   [[buffer(0)]],
    device const half*  weights [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant BatchedRmsNormParams& params [[buffer(3)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint lid    [[thread_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint simd   [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total  [[threads_per_threadgroup]]
) {
    uint n = params.head_dim;
    uint head_idx = tg_idx;
    if (head_idx >= params.head_count || n == 0) return;

    uint head_base = head_idx * n;

    threadgroup float smem[32];

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += total) { float v = input[head_base + i]; sum_sq += v * v; }

    sum_sq = simd_sum(sum_sq);
    if (lane == 0) smem[simd] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0) {
        float v = (lane < n_simd) ? smem[lane] : 0.0f;
        float total_sq = simd_sum(v);
        if (lane == 0) smem[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean_sq = smem[0] / float(n);
    float denom = sqrt(mean_sq + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) return;

    for (uint i = lid; i < n; i += total) {
        output[head_base + i] = (input[head_base + i] / denom) *
            (float(weights[i]) + params.weight_offset);
    }
}

kernel void rms_norm_batched_bf16(
    device const float*  input   [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device float*        output  [[buffer(2)]],
    constant BatchedRmsNormParams& params [[buffer(3)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint lid    [[thread_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint simd   [[simdgroup_index_in_threadgroup]],
    uint n_simd [[simdgroups_per_threadgroup]],
    uint total  [[threads_per_threadgroup]]
) {
    uint n = params.head_dim;
    uint head_idx = tg_idx;
    if (head_idx >= params.head_count || n == 0) return;

    uint head_base = head_idx * n;

    threadgroup float smem[32];

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += total) { float v = input[head_base + i]; sum_sq += v * v; }

    sum_sq = simd_sum(sum_sq);
    if (lane == 0) smem[simd] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0) {
        float v = (lane < n_simd) ? smem[lane] : 0.0f;
        float total_sq = simd_sum(v);
        if (lane == 0) smem[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean_sq = smem[0] / float(n);
    float denom = sqrt(mean_sq + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) return;

    for (uint i = lid; i < n; i += total) {
        output[head_base + i] = (input[head_base + i] / denom) *
            (float(weights[i]) + params.weight_offset);
    }
}


kernel void ffn_gate_silu_product_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant FfnGateProductParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    float gate_value = gate[gid];
    float activated = gate_value / (1.0f + exp(-gate_value));
    output[gid] = activated * up[gid];
}

kernel void ffn_gate_gelu_approx_product_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant FfnGateProductParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    float gate_value = gate[gid];
    float cubic = gate_value * gate_value * gate_value;
    float inner = tanh(0.7978846f * (gate_value + 0.044715f * cubic));
    float activated = 0.5f * gate_value * (1.0f + inner);
    output[gid] = activated * up[gid];
}

kernel void linear_attention_gate_silu_f32(
    device const float* gate [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant FfnGateProductParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    float gate_value = gate[gid];
    float activated = gate_value / (1.0f + exp(-gate_value));
    output[gid] = activated * input[gid];
}

kernel void attention_output_gate_sigmoid_product_f32(
    device const float* gate [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant FfnGateProductParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    float gate_value = gate[gid];
    float activated = 1.0f / (1.0f + exp(-gate_value));
    output[gid] = activated * input[gid];
}

kernel void linear_attention_beta_sigmoid_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant FfnGateProductParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    float value = input[gid];
    output[gid] = 1.0f / (1.0f + exp(-value));
}

kernel void linear_attention_decay_f32(
    device const float* input [[buffer(0)]],
    device const float* a_log [[buffer(1)]],
    device const float* dt_bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant FfnGateProductParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    float summed = input[gid] + dt_bias[gid];
    float softplus = summed > 20.0f ? summed : log(1.0f + exp(summed));
    output[gid] = exp(-exp(a_log[gid]) * softplus);
}

// Parallel apply_rope_f32: one thread per (head, rotary-pair).
// Dispatch: (q_heads + k_heads) * (rotary_dim / 2) threads.
// Threads gid < q_heads*half_dim operate on query; remainder on key.
// No serial loops; each thread applies one independent 2D rotation.
kernel void apply_rope_f32(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    constant RopeDispatchParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (params.head_dim == 0 || (params.head_dim % 2) != 0
        || params.rotary_dim == 0 || params.rotary_dim > params.head_dim
        || (params.rotary_dim % 2) != 0) {
        return;
    }

    uint half_dim = params.rotary_dim / 2;
    uint q_total  = params.query_head_count * half_dim;
    uint k_total  = params.key_head_count   * half_dim;
    if (gid >= q_total + k_total) return;

    bool is_query = (gid < q_total);
    uint flat     = is_query ? gid : gid - q_total;
    uint head_idx = flat / half_dim;
    uint index    = flat % half_dim;

    float cos_theta = cos_table[index];
    float sin_theta = sin_table[index];
    device float* vec = is_query ? query : key;
    uint head_base    = head_idx * params.head_dim;

    if (params.rope_style == 0) {
        uint low  = head_base + index;
        uint high = head_base + half_dim + index;
        float lv = vec[low], hv = vec[high];
        vec[low]  = lv * cos_theta - hv * sin_theta;
        vec[high] = lv * sin_theta + hv * cos_theta;
    } else {
        uint pb = head_base + index * 2;
        float ev = vec[pb], ov = vec[pb + 1];
        vec[pb]     = ev * cos_theta - ov * sin_theta;
        vec[pb + 1] = ov * cos_theta + ev * sin_theta;
    }
}

kernel void apply_rope_batched_f32(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const float* positions [[buffer(2)]],
    constant BatchedRopeDispatchParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.token_count || params.head_dim == 0 || (params.head_dim % 2) != 0
        || params.rotary_dim == 0 || params.rotary_dim > params.head_dim
        || (params.rotary_dim % 2) != 0) {
        return;
    }

    float position = positions[gid];
    if (!isfinite(position) || !isfinite(params.freq_base) || params.freq_base <= 0.0f) {
        return;
    }

    uint half_dim = params.rotary_dim / 2;
    float neg_log_base = -log(params.freq_base);
    float dim_inv = 2.0f / float(params.rotary_dim);
    uint query_token_base = gid * params.query_head_count * params.head_dim;
    uint key_token_base = gid * params.key_head_count * params.head_dim;

    for (uint head = 0; head < params.query_head_count; head++) {
        uint head_base = query_token_base + head * params.head_dim;
        for (uint index = 0; index < half_dim; index++) {
            float freq = exp(neg_log_base * float(index) * dim_inv);
            float theta = position * freq;
            float cos_theta = cos(theta);
            float sin_theta = sin(theta);
            if (params.rope_style == 0) {
                uint low_index = head_base + index;
                uint high_index = head_base + half_dim + index;
                float low_value = query[low_index];
                float high_value = query[high_index];
                query[low_index] = low_value * cos_theta - high_value * sin_theta;
                query[high_index] = low_value * sin_theta + high_value * cos_theta;
            } else {
                uint pair_base = head_base + index * 2;
                float even_value = query[pair_base];
                float odd_value = query[pair_base + 1];
                query[pair_base] = even_value * cos_theta - odd_value * sin_theta;
                query[pair_base + 1] = odd_value * cos_theta + even_value * sin_theta;
            }
        }
    }

    for (uint head = 0; head < params.key_head_count; head++) {
        uint head_base = key_token_base + head * params.head_dim;
        for (uint index = 0; index < half_dim; index++) {
            float freq = exp(neg_log_base * float(index) * dim_inv);
            float theta = position * freq;
            float cos_theta = cos(theta);
            float sin_theta = sin(theta);
            if (params.rope_style == 0) {
                uint low_index = head_base + index;
                uint high_index = head_base + half_dim + index;
                float low_value = key[low_index];
                float high_value = key[high_index];
                key[low_index] = low_value * cos_theta - high_value * sin_theta;
                key[high_index] = low_value * sin_theta + high_value * cos_theta;
            } else {
                uint pair_base = head_base + index * 2;
                float even_value = key[pair_base];
                float odd_value = key[pair_base + 1];
                key[pair_base] = even_value * cos_theta - odd_value * sin_theta;
                key[pair_base + 1] = odd_value * cos_theta + even_value * sin_theta;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K_M GEMV: output[row] = dot(hidden, W_q4[row])
//
// Block layout (from llama.cpp ggml-quants.h / ggml-quants.c):
//   struct block_q4_K {  // 144 bytes per 256 elements
//     half d;            // [0..1]  super-block scale
//     half dmin;         // [2..3]  super-block min-scale
//     uint8 scales[12];  // [4..15] 6-bit packed (scale, min) for 8 sub-blocks
//     uint8 qs[128];     // [16..143] packed nibbles
//   };
//   Nibble layout: qs[j*32 + l] encodes
//     low nibble  → element[j*64 + l]        (sub-block j*2 + 0)
//     high nibble → element[j*64 + 32 + l]   (sub-block j*2 + 1)
//
// Dequant formula (matches dequantize_row_q4_K in ggml-quants.c):
//   For group j in {0,1,2,3}  (64 elements each):
//     get_scale_min_k4(is+0) → d1, m1  (for elements j*64 + 0..31)
//     get_scale_min_k4(is+1) → d2, m2  (for elements j*64 + 32..63)
//     low  nibble: w = d1 * (qs & 0xF) - m1
//     high nibble: w = d2 * (qs >> 4)  - m2
//
// Dispatch: n_rows * PROJECTION_SIMD_WIDTH total threads,
//           threadgroup = PROJECTION_SIMD_WIDTH (32).
//   gid / 32 = output row;  lane = gid % 32 = position in SIMD group.
//   Each lane strides over blocks: for (ib = lane; ib < n_blocks; ib += 32).
// ---------------------------------------------------------------------------

// Exact port of llama.cpp get_scale_min_k4() used in dequantize_row_q4_K.
static inline void q4k_get_scale_min(int j,
                                     device const uint8_t* scales,
                                     thread float* sc,
                                     thread float* mn,
                                     float d, float dmin) {
    uint8_t raw_sc, raw_mn;
    if (j < 4) {
        raw_sc = scales[j]     & 63u;
        raw_mn = scales[j + 4] & 63u;
    } else {
        raw_sc = (scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6u) << 4u);
        raw_mn = (scales[j + 4] >> 4u)   | ((scales[j]     >> 6u) << 4u);
    }
    *sc = d    * float(raw_sc);
    *mn = dmin * float(raw_mn);
}

kernel void decode_projection_q4km(
    device const float*   hidden   [[buffer(0)]],
    device const uint8_t* weights  [[buffer(1)]],  // raw block_q4_K bytes, row-major
    device float*         output   [[buffer(2)]],
    constant Q4KMProjectionParams& params [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // 2 rows per TG (one per simdgroup), 32 threads per simdgroup.
    // Dispatch: dispatch_thread_groups(ceil(n_rows/2), 1, 1), threadgroup (32, 2, 1).
    const uint row = tgpig.x * 2u + (uint)sgitg;
    if (row >= params.n_rows) return;

    const uint n_blocks = params.input_width / 256u;
    device const uint8_t* row_w = weights + row * n_blocks * 144u;

    float sumf = 0.f;

    // Thread tiisg (0..31) processes 8 elements per block across 4 groups of 64:
    //   group j: qs[j*32 + tiisg] → lo nibble: element j*64+tiisg (sub-block j*2)
    //                              → hi nibble: element j*64+32+tiisg (sub-block j*2+1)
    // After simd_sum, all 32 threads together cover all 256 elements of each block.
    for (uint ib = 0; ib < n_blocks; ib++) {
        device const uint8_t* blk    = row_w + ib * 144u;
        device const uint8_t* scales = blk + 4u;
        device const uint8_t* qs     = blk + 16u;

        const float d    = float(((device const half*)blk)[0]);
        const float dmin = float(((device const half*)blk)[1]);

        for (int j = 0; j < 4; j++) {
            float sc_lo, mn_lo, sc_hi, mn_hi;
            q4k_get_scale_min(j * 2,     scales, &sc_lo, &mn_lo, d, dmin);
            q4k_get_scale_min(j * 2 + 1, scales, &sc_hi, &mn_hi, d, dmin);

            uint8_t byte = qs[(uint)j * 32u + (uint)tiisg];
            float qlo = float(byte & 0xFu);
            float qhi = float(byte >> 4u);

            uint elem_base = ib * 256u + (uint)j * 64u;
            float hlo = hidden[elem_base + (uint)tiisg];
            float hhi = hidden[elem_base + (uint)tiisg + 32u];

            sumf += sc_lo * hlo * qlo - mn_lo * hlo
                  + sc_hi * hhi * qhi - mn_hi * hhi;
        }
    }

    float sum_all = simd_sum(sumf);
    if (tiisg == 0) {
        output[row] = sum_all;
    }
}

kernel void expand_grouped_kv_heads_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant GroupedKvExpandParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.output_element_count || params.head_dim == 0 || params.heads_per_kv == 0) {
        return;
    }

    uint head_index = gid / params.head_dim;
    uint lane_index = gid % params.head_dim;
    uint kv_head_index = head_index / params.heads_per_kv;
    if (kv_head_index >= params.kv_head_count) {
        return;
    }

    uint input_index = kv_head_index * params.head_dim + lane_index;
    output[gid] = input[input_index];
}
