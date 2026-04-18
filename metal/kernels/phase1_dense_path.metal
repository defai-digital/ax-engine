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
};

struct BatchedRopeDispatchParams {
    uint token_count;
    uint query_head_count;
    uint key_head_count;
    uint head_dim;
    uint rope_style;
    float freq_base;
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

kernel void paged_decode_attention(
    device const float* query [[buffer(0)]],
    device const float* key_gathered [[buffer(1)]],
    device const float* value_gathered [[buffer(2)]],
    device const uint* cu_seq_lens [[buffer(3)]],
    device const uint* scheduled_cu_seq_lens [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant AttentionDispatchParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) {
        return;
    }

    uint head_size = params.head_count * params.head_dim;
    uint token_id = gid / head_size;
    uint token_lane = gid % head_size;
    uint head_id = token_lane / params.head_dim;
    uint lane_id = token_lane % params.head_dim;

    uint batch_lo = 0;
    uint batch_hi = params.num_seqs;
    while (batch_lo < batch_hi) {
        uint batch_mid = (batch_lo + batch_hi + 1) / 2;
        if (scheduled_cu_seq_lens[batch_mid] <= token_id) {
            batch_lo = batch_mid;
        } else {
            batch_hi = batch_mid - 1;
        }
    }

    uint batch_id = batch_lo;
    uint context_begin = cu_seq_lens[batch_id];
    uint context_end = cu_seq_lens[batch_id + 1];
    uint head_query_base = token_id * head_size + head_id * params.head_dim;
    float max_score = -INFINITY;

    for (uint context_index = context_begin; context_index < context_end; context_index++) {
        uint context_base = context_index * head_size + head_id * params.head_dim;
        float score = 0.0f;
        for (uint lane = 0; lane < params.head_dim; lane++) {
            score += query[head_query_base + lane] * key_gathered[context_base + lane];
        }
        score /= sqrt(float(params.head_dim));
        max_score = max(max_score, score);
    }

    float weight_sum = 0.0f;
    float accum = 0.0f;
    for (uint context_index = context_begin; context_index < context_end; context_index++) {
        uint context_base = context_index * head_size + head_id * params.head_dim;
        float score = 0.0f;
        for (uint lane = 0; lane < params.head_dim; lane++) {
            score += query[head_query_base + lane] * key_gathered[context_base + lane];
        }
        score /= sqrt(float(params.head_dim));
        float weight = exp(score - max_score);
        weight_sum += weight;
        accum += weight * value_gathered[context_base + lane_id];
    }

    output[gid] = accum / max(weight_sum, 0.000001f);
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

kernel void decode_logits_projection_f32(
    device const float* hidden [[buffer(0)]],
    device const float* projection [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant LogitsProjectionParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.vocab_rows) {
        return;
    }

    uint row_base = gid * params.projection_cols;
    float score = 0.0f;
    for (uint lane = 0; lane < params.input_width; lane++) {
        score += projection[row_base + lane] * hidden[lane];
    }
    logits[gid] = score;
}

kernel void decode_logits_projection_f16(
    device const float* hidden [[buffer(0)]],
    device const half* projection [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant LogitsProjectionParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.vocab_rows) {
        return;
    }

    uint row_base = gid * params.projection_cols;
    float score = 0.0f;
    for (uint lane = 0; lane < params.input_width; lane++) {
        score += float(projection[row_base + lane]) * hidden[lane];
    }
    logits[gid] = score;
}

kernel void decode_logits_projection_bf16(
    device const float* hidden [[buffer(0)]],
    device const bfloat* projection [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant LogitsProjectionParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.vocab_rows) {
        return;
    }

    uint row_base = gid * params.projection_cols;
    float score = 0.0f;
    for (uint lane = 0; lane < params.input_width; lane++) {
        score += float(projection[row_base + lane]) * hidden[lane];
    }
    logits[gid] = score;
}

kernel void decode_logits_projection_batched_f32(
    device const float* hidden [[buffer(0)]],
    device const float* projection [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant BatchedLogitsProjectionParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint element_count = params.token_count * params.vocab_rows;
    if (gid >= element_count || params.hidden_stride == 0) {
        return;
    }

    uint token_index = gid / params.vocab_rows;
    uint vocab_index = gid % params.vocab_rows;
    uint row_base = vocab_index * params.projection_cols;
    uint hidden_base = token_index * params.hidden_stride;
    float score = 0.0f;
    for (uint lane = 0; lane < params.input_width; lane++) {
        score += projection[row_base + lane] * hidden[hidden_base + lane];
    }
    logits[gid] = score;
}

kernel void decode_logits_projection_batched_f16(
    device const float* hidden [[buffer(0)]],
    device const half* projection [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant BatchedLogitsProjectionParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint element_count = params.token_count * params.vocab_rows;
    if (gid >= element_count || params.hidden_stride == 0) {
        return;
    }

    uint token_index = gid / params.vocab_rows;
    uint vocab_index = gid % params.vocab_rows;
    uint row_base = vocab_index * params.projection_cols;
    uint hidden_base = token_index * params.hidden_stride;
    float score = 0.0f;
    for (uint lane = 0; lane < params.input_width; lane++) {
        score += float(projection[row_base + lane]) * hidden[hidden_base + lane];
    }
    logits[gid] = score;
}

kernel void decode_logits_projection_batched_bf16(
    device const float* hidden [[buffer(0)]],
    device const bfloat* projection [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant BatchedLogitsProjectionParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint element_count = params.token_count * params.vocab_rows;
    if (gid >= element_count || params.hidden_stride == 0) {
        return;
    }

    uint token_index = gid / params.vocab_rows;
    uint vocab_index = gid % params.vocab_rows;
    uint row_base = vocab_index * params.projection_cols;
    uint hidden_base = token_index * params.hidden_stride;
    float score = 0.0f;
    for (uint lane = 0; lane < params.input_width; lane++) {
        score += float(projection[row_base + lane]) * hidden[hidden_base + lane];
    }
    logits[gid] = score;
}

kernel void logits_argmax_f32(
    device const float* logits [[buffer(0)]],
    device uint* best_index [[buffer(1)]],
    constant LogitsArgmaxParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0 || params.element_count == 0) {
        return;
    }

    uint best = 0;
    float best_score = logits[0];
    for (uint index = 1; index < params.element_count; index++) {
        float score = logits[index];
        if (score > best_score) {
            best_score = score;
            best = index;
        }
    }
    best_index[0] = best;
}

kernel void logits_argmax_batched_f32(
    device const float* logits [[buffer(0)]],
    device uint* best_index [[buffer(1)]],
    constant BatchedLogitsArgmaxParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.token_count || params.vocab_rows == 0) {
        return;
    }

    uint base = gid * params.vocab_rows;
    uint best = 0;
    float best_score = logits[base];
    for (uint index = 1; index < params.vocab_rows; index++) {
        float score = logits[base + index];
        if (score > best_score) {
            best_score = score;
            best = index;
        }
    }
    best_index[gid] = best;
}

kernel void sample_argmax_logprob_f32(
    device const float* logits [[buffer(0)]],
    device uint* best_index [[buffer(1)]],
    device float* best_logprob [[buffer(2)]],
    constant LogitsArgmaxParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0 || params.element_count == 0) {
        return;
    }

    uint best = 0;
    float best_score = logits[0];
    for (uint index = 1; index < params.element_count; index++) {
        float score = logits[index];
        if (score > best_score) {
            best_score = score;
            best = index;
        }
    }

    float normalizer = 0.0f;
    for (uint index = 0; index < params.element_count; index++) {
        normalizer += exp(logits[index] - best_score);
    }
    if (!isfinite(normalizer) || normalizer <= 0.0f) {
        return;
    }

    best_index[0] = best;
    best_logprob[0] = best_score - best_score - log(normalizer);
}

kernel void sample_argmax_logprob_batched_f32(
    device const float* logits [[buffer(0)]],
    device uint* best_index [[buffer(1)]],
    device float* best_logprob [[buffer(2)]],
    constant BatchedLogitsArgmaxParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.token_count || params.vocab_rows == 0) {
        return;
    }

    uint base = gid * params.vocab_rows;
    uint best = 0;
    float best_score = logits[base];
    for (uint index = 1; index < params.vocab_rows; index++) {
        float score = logits[base + index];
        if (score > best_score) {
            best_score = score;
            best = index;
        }
    }

    float normalizer = 0.0f;
    for (uint index = 0; index < params.vocab_rows; index++) {
        normalizer += exp(logits[base + index] - best_score);
    }
    if (!isfinite(normalizer) || normalizer <= 0.0f) {
        return;
    }

    best_index[gid] = best;
    best_logprob[gid] = best_score - best_score - log(normalizer);
}

kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0 || params.element_count == 0) {
        return;
    }

    float mean_square = 0.0f;
    for (uint index = 0; index < params.element_count; index++) {
        float value = input[index];
        mean_square += value * value;
    }
    mean_square /= float(params.element_count);

    float denom = sqrt(mean_square + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) {
        return;
    }

    for (uint index = 0; index < params.element_count; index++) {
        float normalized = (input[index] / denom) * (weights[index] + params.weight_offset);
        output[index] = normalized;
    }
}

kernel void rms_norm_f16(
    device const float* input [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0 || params.element_count == 0) {
        return;
    }

    float mean_square = 0.0f;
    for (uint index = 0; index < params.element_count; index++) {
        float value = input[index];
        mean_square += value * value;
    }
    mean_square /= float(params.element_count);

    float denom = sqrt(mean_square + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) {
        return;
    }

    for (uint index = 0; index < params.element_count; index++) {
        float normalized =
            (input[index] / denom) * (float(weights[index]) + params.weight_offset);
        output[index] = normalized;
    }
}

kernel void rms_norm_bf16(
    device const float* input [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0 || params.element_count == 0) {
        return;
    }

    float mean_square = 0.0f;
    for (uint index = 0; index < params.element_count; index++) {
        float value = input[index];
        mean_square += value * value;
    }
    mean_square /= float(params.element_count);

    float denom = sqrt(mean_square + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) {
        return;
    }

    for (uint index = 0; index < params.element_count; index++) {
        float normalized =
            (input[index] / denom) * (float(weights[index]) + params.weight_offset);
        output[index] = normalized;
    }
}

kernel void rms_norm_batched_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant BatchedRmsNormParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint element_count = params.head_count * params.head_dim;
    if (gid >= element_count || params.head_dim == 0) {
        return;
    }

    uint head_index = gid / params.head_dim;
    uint lane_index = gid % params.head_dim;
    uint head_base = head_index * params.head_dim;
    float mean_square = 0.0f;
    for (uint lane = 0; lane < params.head_dim; lane++) {
        float value = input[head_base + lane];
        mean_square += value * value;
    }
    mean_square /= float(params.head_dim);

    float denom = sqrt(mean_square + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) {
        return;
    }

    float normalized =
        (input[head_base + lane_index] / denom) * (weights[lane_index] + params.weight_offset);
    output[gid] = normalized;
}

kernel void rms_norm_batched_f16(
    device const float* input [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant BatchedRmsNormParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint element_count = params.head_count * params.head_dim;
    if (gid >= element_count || params.head_dim == 0) {
        return;
    }

    uint head_index = gid / params.head_dim;
    uint lane_index = gid % params.head_dim;
    uint head_base = head_index * params.head_dim;
    float mean_square = 0.0f;
    for (uint lane = 0; lane < params.head_dim; lane++) {
        float value = input[head_base + lane];
        mean_square += value * value;
    }
    mean_square /= float(params.head_dim);

    float denom = sqrt(mean_square + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) {
        return;
    }

    float normalized = (input[head_base + lane_index] / denom) *
        (float(weights[lane_index]) + params.weight_offset);
    output[gid] = normalized;
}

kernel void rms_norm_batched_bf16(
    device const float* input [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant BatchedRmsNormParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint element_count = params.head_count * params.head_dim;
    if (gid >= element_count || params.head_dim == 0) {
        return;
    }

    uint head_index = gid / params.head_dim;
    uint lane_index = gid % params.head_dim;
    uint head_base = head_index * params.head_dim;
    float mean_square = 0.0f;
    for (uint lane = 0; lane < params.head_dim; lane++) {
        float value = input[head_base + lane];
        mean_square += value * value;
    }
    mean_square /= float(params.head_dim);

    float denom = sqrt(mean_square + params.epsilon);
    if (!isfinite(denom) || denom <= 0.0f) {
        return;
    }

    float normalized = (input[head_base + lane_index] / denom) *
        (float(weights[lane_index]) + params.weight_offset);
    output[gid] = normalized;
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

kernel void apply_rope_f32(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    constant RopeDispatchParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0 || params.head_dim == 0 || (params.head_dim % 2) != 0) {
        return;
    }

    uint half_dim = params.head_dim / 2;
    for (uint head = 0; head < params.query_head_count; head++) {
        uint head_base = head * params.head_dim;
        for (uint index = 0; index < half_dim; index++) {
            float cos_theta = cos_table[index];
            float sin_theta = sin_table[index];
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
        uint head_base = head * params.head_dim;
        for (uint index = 0; index < half_dim; index++) {
            float cos_theta = cos_table[index];
            float sin_theta = sin_table[index];
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

kernel void apply_rope_batched_f32(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const float* positions [[buffer(2)]],
    constant BatchedRopeDispatchParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.token_count || params.head_dim == 0 || (params.head_dim % 2) != 0) {
        return;
    }

    float position = positions[gid];
    if (!isfinite(position) || !isfinite(params.freq_base) || params.freq_base <= 0.0f) {
        return;
    }

    uint half_dim = params.head_dim / 2;
    float neg_log_base = -log(params.freq_base);
    float dim_inv = 2.0f / float(params.head_dim);
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
