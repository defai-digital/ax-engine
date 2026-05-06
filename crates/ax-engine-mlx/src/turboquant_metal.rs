use std::sync::OnceLock;

use ax_engine_core::MlxTurboQuantPreset;
use mlx_sys::{KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, eval};

use crate::turboquant::{
    TurboQuantAttentionPartitionStats, TurboQuantCodecError, TurboQuantCompressedBlockBuffer,
    TurboQuantFusedDecodeLaunchDescriptor, hadamard_in_place, merge_attention_partition_stats,
};

static TURBOQUANT_FUSED_COLD_DECODE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static TURBOQUANT_FUSED_COLD_DECODE_HEAD_SERIAL_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static TURBOQUANT_FUSED_COLD_DECODE_SCORE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static TURBOQUANT_FUSED_COLD_DECODE_HEAD_STATS_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static TURBOQUANT_FUSED_COLD_DECODE_VALUE_SUM_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

fn validate_query_head_mapping(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    actual_queries: usize,
) -> Result<(), TurboQuantCodecError> {
    if actual_queries != descriptor.n_query_heads {
        return Err(TurboQuantCodecError::MismatchedKvHeadCount {
            expected: descriptor.n_query_heads,
            actual: actual_queries,
        });
    }
    if descriptor.n_query_heads == 0
        || descriptor.n_kv_heads == 0
        || !descriptor
            .n_query_heads
            .is_multiple_of(descriptor.n_kv_heads)
    {
        return Err(TurboQuantCodecError::MismatchedKvHeadCount {
            expected: descriptor.n_kv_heads,
            actual: descriptor.n_query_heads,
        });
    }
    Ok(())
}

pub fn turboquant_fused_cold_decode_metal(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    buffer: &TurboQuantCompressedBlockBuffer,
    queries: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
    if descriptor.preset != MlxTurboQuantPreset::K8V4 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset,
        });
    }
    if descriptor.key_bits != 8 || descriptor.value_bits != 4 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset,
        });
    }
    if descriptor.cold_tokens == 0 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::FullPrecisionOnly,
        });
    }
    validate_query_head_mapping(descriptor, queries.len())?;

    let mut rotated_queries = Vec::with_capacity(descriptor.n_query_heads * descriptor.head_dim);
    for query in queries {
        if query.len() != descriptor.head_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: descriptor.head_dim,
                actual: query.len(),
            });
        }
        let mut rotated = query.clone();
        hadamard_in_place(&mut rotated)?;
        rotated_queries.extend(rotated);
    }

    let compressed = MlxArray::from_raw_data(
        buffer.as_bytes().as_ptr(),
        buffer.as_bytes().len(),
        &[buffer.as_bytes().len() as i32],
        MlxDtype::Uint8,
    );
    let query = MlxArray::from_raw_data(
        rotated_queries.as_ptr().cast(),
        rotated_queries.len() * std::mem::size_of::<f32>(),
        &[descriptor.n_query_heads as i32, descriptor.head_dim as i32],
        MlxDtype::Float32,
    );

    let kernel = TURBOQUANT_FUSED_COLD_DECODE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "turboquant_fused_cold_decode_k8v4",
            &["compressed", "rotated_query"],
            &["output"],
            TURBOQUANT_FUSED_COLD_DECODE_KERNEL_SOURCE,
            TURBOQUANT_FUSED_COLD_DECODE_KERNEL_HEADER,
            true,
        )
    });

    let outputs = kernel.apply_with_template(
        &[&compressed, &query],
        &[KernelOutputSpec {
            shape: vec![descriptor.n_query_heads as i32, descriptor.head_dim as i32],
            dtype: MlxDtype::Float32,
        }],
        &[
            KernelTemplateArg::Int {
                name: "COLD_TOKENS",
                value: descriptor.cold_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "HEADS",
                value: descriptor.n_query_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "KV_HEADS",
                value: descriptor.n_kv_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "HEAD_DIM",
                value: descriptor.head_dim as i32,
            },
            KernelTemplateArg::Int {
                name: "BLOCK_TOKENS",
                value: descriptor.block_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "BLOCK_BYTES",
                value: descriptor.block_bytes as i32,
            },
            KernelTemplateArg::Int {
                name: "TOKEN_STRIDE_BYTES",
                value: descriptor.token_stride_bytes as i32,
            },
            KernelTemplateArg::Int {
                name: "SLOT_BYTES",
                value: descriptor.slot_bytes_per_head as i32,
            },
            KernelTemplateArg::Int {
                name: "KEY_PAYLOAD_OFFSET",
                value: descriptor.key_payload_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "KEY_NORM_OFFSET",
                value: descriptor.key_norm_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_PAYLOAD_OFFSET",
                value: descriptor.value_payload_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_MINS_OFFSET",
                value: descriptor.value_mins_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_SCALES_OFFSET",
                value: descriptor.value_scales_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_GROUP_SIZE",
                value: descriptor.value_group_size as i32,
            },
        ],
        (
            descriptor.head_dim as i32,
            descriptor.n_query_heads as i32,
            1,
        ),
        (1, 1, 1),
        None,
    );

    let output = outputs
        .into_iter()
        .next()
        .expect("TurboQuant fused decode output");
    eval(&[&output]);
    Ok(output
        .data_f32()
        .chunks(descriptor.head_dim)
        .map(|chunk| chunk.to_vec())
        .collect())
}

pub fn turboquant_fused_cold_decode_metal_head_serial(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    buffer: &TurboQuantCompressedBlockBuffer,
    queries: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
    if descriptor.preset != MlxTurboQuantPreset::K8V4 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset,
        });
    }
    if descriptor.key_bits != 8 || descriptor.value_bits != 4 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset,
        });
    }
    if descriptor.cold_tokens == 0 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::FullPrecisionOnly,
        });
    }
    if descriptor.head_dim != 128 || descriptor.n_query_heads != descriptor.n_kv_heads {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::UnsupportedHeadDim,
        });
    }
    validate_query_head_mapping(descriptor, queries.len())?;

    let mut rotated_queries = Vec::with_capacity(descriptor.n_kv_heads * descriptor.head_dim);
    for query in queries {
        if query.len() != descriptor.head_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: descriptor.head_dim,
                actual: query.len(),
            });
        }
        let mut rotated = query.clone();
        hadamard_in_place(&mut rotated)?;
        rotated_queries.extend(rotated);
    }

    let compressed = MlxArray::from_raw_data(
        buffer.as_bytes().as_ptr(),
        buffer.as_bytes().len(),
        &[buffer.as_bytes().len() as i32],
        MlxDtype::Uint8,
    );
    let query = MlxArray::from_raw_data(
        rotated_queries.as_ptr().cast(),
        rotated_queries.len() * std::mem::size_of::<f32>(),
        &[descriptor.n_kv_heads as i32, descriptor.head_dim as i32],
        MlxDtype::Float32,
    );

    let kernel = TURBOQUANT_FUSED_COLD_DECODE_HEAD_SERIAL_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "turboquant_fused_cold_decode_k8v4_head_serial",
            &["compressed", "rotated_query"],
            &["output"],
            TURBOQUANT_FUSED_COLD_DECODE_HEAD_SERIAL_KERNEL_SOURCE,
            TURBOQUANT_FUSED_COLD_DECODE_KERNEL_HEADER,
            true,
        )
    });

    let outputs = kernel.apply_with_template(
        &[&compressed, &query],
        &[KernelOutputSpec {
            shape: vec![descriptor.n_kv_heads as i32, descriptor.head_dim as i32],
            dtype: MlxDtype::Float32,
        }],
        &[
            KernelTemplateArg::Int {
                name: "COLD_TOKENS",
                value: descriptor.cold_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "HEADS",
                value: descriptor.n_kv_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "HEAD_DIM",
                value: descriptor.head_dim as i32,
            },
            KernelTemplateArg::Int {
                name: "BLOCK_TOKENS",
                value: descriptor.block_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "BLOCK_BYTES",
                value: descriptor.block_bytes as i32,
            },
            KernelTemplateArg::Int {
                name: "TOKEN_STRIDE_BYTES",
                value: descriptor.token_stride_bytes as i32,
            },
            KernelTemplateArg::Int {
                name: "SLOT_BYTES",
                value: descriptor.slot_bytes_per_head as i32,
            },
            KernelTemplateArg::Int {
                name: "KEY_PAYLOAD_OFFSET",
                value: descriptor.key_payload_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "KEY_NORM_OFFSET",
                value: descriptor.key_norm_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_PAYLOAD_OFFSET",
                value: descriptor.value_payload_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_MINS_OFFSET",
                value: descriptor.value_mins_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_SCALES_OFFSET",
                value: descriptor.value_scales_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_GROUP_SIZE",
                value: descriptor.value_group_size as i32,
            },
        ],
        (descriptor.n_kv_heads as i32, 1, 1),
        (1, 1, 1),
        None,
    );

    let output = outputs
        .into_iter()
        .next()
        .expect("TurboQuant fused decode output");
    eval(&[&output]);
    Ok(output
        .data_f32()
        .chunks(descriptor.head_dim)
        .map(|chunk| chunk.to_vec())
        .collect())
}

pub fn turboquant_fused_cold_decode_metal_two_stage(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    buffer: &TurboQuantCompressedBlockBuffer,
    queries: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
    turboquant_fused_cold_decode_metal_two_stage_partition_stats(descriptor, buffer, queries)?
        .into_iter()
        .map(|stats| merge_attention_partition_stats(&[stats]))
        .collect()
}

pub fn turboquant_fused_cold_decode_metal_two_stage_partition_stats(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    buffer: &TurboQuantCompressedBlockBuffer,
    queries: &[Vec<f32>],
) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
    turboquant_fused_cold_decode_metal_two_stage_stats(descriptor, buffer, queries)
}

fn turboquant_fused_cold_decode_metal_two_stage_stats(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    buffer: &TurboQuantCompressedBlockBuffer,
    queries: &[Vec<f32>],
) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
    if descriptor.preset != MlxTurboQuantPreset::K8V4 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset,
        });
    }
    if descriptor.key_bits != 8 || descriptor.value_bits != 4 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset,
        });
    }
    if descriptor.cold_tokens == 0 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::FullPrecisionOnly,
        });
    }
    validate_query_head_mapping(descriptor, queries.len())?;

    let mut rotated_queries = Vec::with_capacity(descriptor.n_query_heads * descriptor.head_dim);
    for query in queries {
        if query.len() != descriptor.head_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: descriptor.head_dim,
                actual: query.len(),
            });
        }
        let mut rotated = query.clone();
        hadamard_in_place(&mut rotated)?;
        rotated_queries.extend(rotated);
    }

    let compressed = MlxArray::from_raw_data(
        buffer.as_bytes().as_ptr(),
        buffer.as_bytes().len(),
        &[buffer.as_bytes().len() as i32],
        MlxDtype::Uint8,
    );
    let query = MlxArray::from_raw_data(
        rotated_queries.as_ptr().cast(),
        rotated_queries.len() * std::mem::size_of::<f32>(),
        &[descriptor.n_query_heads as i32, descriptor.head_dim as i32],
        MlxDtype::Float32,
    );

    let score_kernel = TURBOQUANT_FUSED_COLD_DECODE_SCORE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "turboquant_fused_cold_decode_k8v4_scores",
            &["compressed", "rotated_query"],
            &["scores"],
            TURBOQUANT_FUSED_COLD_DECODE_SCORE_KERNEL_SOURCE,
            TURBOQUANT_FUSED_COLD_DECODE_KERNEL_HEADER,
            true,
        )
    });
    let score_outputs = score_kernel.apply_with_template(
        &[&compressed, &query],
        &[KernelOutputSpec {
            shape: vec![
                descriptor.n_query_heads as i32,
                descriptor.cold_tokens as i32,
            ],
            dtype: MlxDtype::Float32,
        }],
        &[
            KernelTemplateArg::Int {
                name: "COLD_TOKENS",
                value: descriptor.cold_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "HEADS",
                value: descriptor.n_query_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "KV_HEADS",
                value: descriptor.n_kv_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "HEAD_DIM",
                value: descriptor.head_dim as i32,
            },
            KernelTemplateArg::Int {
                name: "BLOCK_TOKENS",
                value: descriptor.block_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "BLOCK_BYTES",
                value: descriptor.block_bytes as i32,
            },
            KernelTemplateArg::Int {
                name: "TOKEN_STRIDE_BYTES",
                value: descriptor.token_stride_bytes as i32,
            },
            KernelTemplateArg::Int {
                name: "SLOT_BYTES",
                value: descriptor.slot_bytes_per_head as i32,
            },
            KernelTemplateArg::Int {
                name: "KEY_PAYLOAD_OFFSET",
                value: descriptor.key_payload_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "KEY_NORM_OFFSET",
                value: descriptor.key_norm_offset_in_slot as i32,
            },
        ],
        (
            descriptor.cold_tokens as i32,
            descriptor.n_query_heads as i32,
            1,
        ),
        (1, 1, 1),
        None,
    );
    let scores = score_outputs
        .into_iter()
        .next()
        .expect("TurboQuant score output");

    let stats_kernel = TURBOQUANT_FUSED_COLD_DECODE_HEAD_STATS_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "turboquant_fused_cold_decode_k8v4_head_stats",
            &["scores"],
            &["max_scores", "exp_sums"],
            TURBOQUANT_FUSED_COLD_DECODE_HEAD_STATS_KERNEL_SOURCE,
            TURBOQUANT_FUSED_COLD_DECODE_KERNEL_HEADER,
            true,
        )
    });
    let stats_outputs = stats_kernel.apply_with_template(
        &[&scores],
        &[
            KernelOutputSpec {
                shape: vec![descriptor.n_query_heads as i32],
                dtype: MlxDtype::Float32,
            },
            KernelOutputSpec {
                shape: vec![descriptor.n_query_heads as i32],
                dtype: MlxDtype::Float32,
            },
        ],
        &[
            KernelTemplateArg::Int {
                name: "COLD_TOKENS",
                value: descriptor.cold_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "HEADS",
                value: descriptor.n_query_heads as i32,
            },
        ],
        (descriptor.n_query_heads as i32, 1, 1),
        (1, 1, 1),
        None,
    );
    let mut stats_outputs = stats_outputs.into_iter();
    let max_scores = stats_outputs
        .next()
        .expect("TurboQuant partition max-score output");
    let exp_sums = stats_outputs
        .next()
        .expect("TurboQuant partition exp-sum output");

    let value_sum_kernel = TURBOQUANT_FUSED_COLD_DECODE_VALUE_SUM_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "turboquant_fused_cold_decode_k8v4_value_sum",
            &["compressed", "scores", "max_scores"],
            &["weighted_value_sum"],
            TURBOQUANT_FUSED_COLD_DECODE_VALUE_SUM_KERNEL_SOURCE,
            TURBOQUANT_FUSED_COLD_DECODE_KERNEL_HEADER,
            true,
        )
    });
    let value_outputs = value_sum_kernel.apply_with_template(
        &[&compressed, &scores, &max_scores],
        &[KernelOutputSpec {
            shape: vec![descriptor.n_query_heads as i32, descriptor.head_dim as i32],
            dtype: MlxDtype::Float32,
        }],
        &[
            KernelTemplateArg::Int {
                name: "COLD_TOKENS",
                value: descriptor.cold_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "HEADS",
                value: descriptor.n_query_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "KV_HEADS",
                value: descriptor.n_kv_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "HEAD_DIM",
                value: descriptor.head_dim as i32,
            },
            KernelTemplateArg::Int {
                name: "BLOCK_TOKENS",
                value: descriptor.block_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "BLOCK_BYTES",
                value: descriptor.block_bytes as i32,
            },
            KernelTemplateArg::Int {
                name: "TOKEN_STRIDE_BYTES",
                value: descriptor.token_stride_bytes as i32,
            },
            KernelTemplateArg::Int {
                name: "SLOT_BYTES",
                value: descriptor.slot_bytes_per_head as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_PAYLOAD_OFFSET",
                value: descriptor.value_payload_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_MINS_OFFSET",
                value: descriptor.value_mins_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_SCALES_OFFSET",
                value: descriptor.value_scales_offset_in_slot as i32,
            },
            KernelTemplateArg::Int {
                name: "VALUE_GROUP_SIZE",
                value: descriptor.value_group_size as i32,
            },
        ],
        (
            descriptor.head_dim as i32,
            descriptor.n_query_heads as i32,
            1,
        ),
        (1, 1, 1),
        None,
    );

    let weighted_value_sum = value_outputs
        .into_iter()
        .next()
        .expect("TurboQuant partition weighted-value output");
    eval(&[&max_scores, &exp_sums, &weighted_value_sum]);

    let max_scores = max_scores.data_f32();
    let exp_sums = exp_sums.data_f32();
    let weighted_value_sum = weighted_value_sum.data_f32();
    let mut stats = Vec::with_capacity(descriptor.n_query_heads);
    for head_index in 0..descriptor.n_query_heads {
        let offset = head_index * descriptor.head_dim;
        stats.push(TurboQuantAttentionPartitionStats {
            token_count: descriptor.cold_tokens,
            value_dim: descriptor.head_dim,
            max_score: max_scores[head_index],
            exp_sum: exp_sums[head_index],
            weighted_value_sum: weighted_value_sum[offset..offset + descriptor.head_dim].to_vec(),
        });
    }

    Ok(stats)
}

const TURBOQUANT_FUSED_COLD_DECODE_KERNEL_HEADER: &str = r#"
    inline float tq_read_f32(const device uint8_t* bytes, int offset) {
      uint bits = ((uint)bytes[offset])
        | (((uint)bytes[offset + 1]) << 8)
        | (((uint)bytes[offset + 2]) << 16)
        | (((uint)bytes[offset + 3]) << 24);
      return as_type<float>(bits);
    }

    inline float tq_centroid_k8(uint8_t index) {
      return -1.0f + (2.0f * ((float)index) / 255.0f);
    }

    inline uint8_t tq_unpack_v4(const device uint8_t* bytes, int payload_offset, int dim) {
      uint8_t packed = bytes[payload_offset + dim / 2];
      return (dim & 1) == 0 ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    }
"#;

const TURBOQUANT_FUSED_COLD_DECODE_KERNEL_SOURCE: &str = r#"
    const int dim = thread_position_in_grid.x;
    const int head = thread_position_in_grid.y;
    if (dim >= HEAD_DIM || head >= HEADS) {
      return;
    }
    const int kv_head = head / (HEADS / KV_HEADS);

    const float inv_sqrt_dim = rsqrt((float)HEAD_DIM);
    float max_score = -3.4028234663852886e+38f;

    for (int token = 0; token < COLD_TOKENS; ++token) {
      const int block = token / BLOCK_TOKENS;
      const int token_offset = token - block * BLOCK_TOKENS;
      const int slot = block * BLOCK_BYTES
        + token_offset * TOKEN_STRIDE_BYTES
        + kv_head * SLOT_BYTES;
      const int key_payload = slot + KEY_PAYLOAD_OFFSET;
      const float key_norm = tq_read_f32(compressed, slot + KEY_NORM_OFFSET);

      float score = 0.0f;
      for (int kdim = 0; kdim < HEAD_DIM; ++kdim) {
        const float centroid = tq_centroid_k8(compressed[key_payload + kdim]);
        score += rotated_query[head * HEAD_DIM + kdim] * centroid;
      }
      score *= key_norm * inv_sqrt_dim;
      max_score = max(max_score, score);
    }

    float denom = 0.0f;
    float weighted = 0.0f;
    for (int token = 0; token < COLD_TOKENS; ++token) {
      const int block = token / BLOCK_TOKENS;
      const int token_offset = token - block * BLOCK_TOKENS;
      const int slot = block * BLOCK_BYTES
        + token_offset * TOKEN_STRIDE_BYTES
        + kv_head * SLOT_BYTES;
      const int key_payload = slot + KEY_PAYLOAD_OFFSET;
      const float key_norm = tq_read_f32(compressed, slot + KEY_NORM_OFFSET);

      float score = 0.0f;
      for (int kdim = 0; kdim < HEAD_DIM; ++kdim) {
        const float centroid = tq_centroid_k8(compressed[key_payload + kdim]);
        score += rotated_query[head * HEAD_DIM + kdim] * centroid;
      }
      score *= key_norm * inv_sqrt_dim;

      const float weight = exp(score - max_score);
      const int group = dim / VALUE_GROUP_SIZE;
      const int value_payload = slot + VALUE_PAYLOAD_OFFSET;
      const float value_min = tq_read_f32(compressed, slot + VALUE_MINS_OFFSET + group * 4);
      const float value_scale = tq_read_f32(compressed, slot + VALUE_SCALES_OFFSET + group * 4);
      const float value = value_min + value_scale * (float)tq_unpack_v4(compressed, value_payload, dim);

      denom += weight;
      weighted += weight * value;
    }

    output[head * HEAD_DIM + dim] = weighted / max(denom, 1.17549435e-38f);
"#;

const TURBOQUANT_FUSED_COLD_DECODE_HEAD_SERIAL_KERNEL_SOURCE: &str = r#"
    const int head = thread_position_in_grid.x;
    if (head >= HEADS || HEAD_DIM != 128) {
      return;
    }

    const float inv_sqrt_dim = rsqrt((float)HEAD_DIM);
    float max_score = -3.4028234663852886e+38f;

    for (int token = 0; token < COLD_TOKENS; ++token) {
      const int block = token / BLOCK_TOKENS;
      const int token_offset = token - block * BLOCK_TOKENS;
      const int slot = block * BLOCK_BYTES
        + token_offset * TOKEN_STRIDE_BYTES
        + head * SLOT_BYTES;
      const int key_payload = slot + KEY_PAYLOAD_OFFSET;
      const float key_norm = tq_read_f32(compressed, slot + KEY_NORM_OFFSET);

      float score = 0.0f;
      for (int kdim = 0; kdim < 128; ++kdim) {
        const float centroid = tq_centroid_k8(compressed[key_payload + kdim]);
        score += rotated_query[head * 128 + kdim] * centroid;
      }
      score *= key_norm * inv_sqrt_dim;
      max_score = max(max_score, score);
    }

    float denom = 0.0f;
    float weighted[128];
    for (int dim = 0; dim < 128; ++dim) {
      weighted[dim] = 0.0f;
    }

    for (int token = 0; token < COLD_TOKENS; ++token) {
      const int block = token / BLOCK_TOKENS;
      const int token_offset = token - block * BLOCK_TOKENS;
      const int slot = block * BLOCK_BYTES
        + token_offset * TOKEN_STRIDE_BYTES
        + head * SLOT_BYTES;
      const int key_payload = slot + KEY_PAYLOAD_OFFSET;
      const float key_norm = tq_read_f32(compressed, slot + KEY_NORM_OFFSET);

      float score = 0.0f;
      for (int kdim = 0; kdim < 128; ++kdim) {
        const float centroid = tq_centroid_k8(compressed[key_payload + kdim]);
        score += rotated_query[head * 128 + kdim] * centroid;
      }
      score *= key_norm * inv_sqrt_dim;

      const float weight = exp(score - max_score);
      const int value_payload = slot + VALUE_PAYLOAD_OFFSET;
      denom += weight;
      for (int dim = 0; dim < 128; ++dim) {
        const int group = dim / VALUE_GROUP_SIZE;
        const float value_min = tq_read_f32(compressed, slot + VALUE_MINS_OFFSET + group * 4);
        const float value_scale = tq_read_f32(compressed, slot + VALUE_SCALES_OFFSET + group * 4);
        const float value = value_min + value_scale * (float)tq_unpack_v4(compressed, value_payload, dim);
        weighted[dim] += weight * value;
      }
    }

    const float safe_denom = max(denom, 1.17549435e-38f);
    for (int dim = 0; dim < 128; ++dim) {
      output[head * 128 + dim] = weighted[dim] / safe_denom;
    }
"#;

const TURBOQUANT_FUSED_COLD_DECODE_SCORE_KERNEL_SOURCE: &str = r#"
    const int token = thread_position_in_grid.x;
    const int head = thread_position_in_grid.y;
    if (token >= COLD_TOKENS || head >= HEADS) {
      return;
    }
    const int kv_head = head / (HEADS / KV_HEADS);

    const int block = token / BLOCK_TOKENS;
    const int token_offset = token - block * BLOCK_TOKENS;
    const int slot = block * BLOCK_BYTES
      + token_offset * TOKEN_STRIDE_BYTES
      + kv_head * SLOT_BYTES;
    const int key_payload = slot + KEY_PAYLOAD_OFFSET;
    const float key_norm = tq_read_f32(compressed, slot + KEY_NORM_OFFSET);
    const float inv_sqrt_dim = rsqrt((float)HEAD_DIM);

    float score = 0.0f;
    for (int kdim = 0; kdim < HEAD_DIM; ++kdim) {
      const float centroid = tq_centroid_k8(compressed[key_payload + kdim]);
      score += rotated_query[head * HEAD_DIM + kdim] * centroid;
    }
    scores[head * COLD_TOKENS + token] = score * key_norm * inv_sqrt_dim;
"#;

const TURBOQUANT_FUSED_COLD_DECODE_HEAD_STATS_KERNEL_SOURCE: &str = r#"
    const int head = thread_position_in_grid.x;
    if (head >= HEADS) {
      return;
    }

    float max_score = -3.4028234663852886e+38f;
    for (int token = 0; token < COLD_TOKENS; ++token) {
      max_score = max(max_score, scores[head * COLD_TOKENS + token]);
    }

    float denom = 0.0f;
    for (int token = 0; token < COLD_TOKENS; ++token) {
      denom += exp(scores[head * COLD_TOKENS + token] - max_score);
    }

    max_scores[head] = max_score;
    exp_sums[head] = denom;
"#;

const TURBOQUANT_FUSED_COLD_DECODE_VALUE_SUM_KERNEL_SOURCE: &str = r#"
    const int dim = thread_position_in_grid.x;
    const int head = thread_position_in_grid.y;
    if (dim >= HEAD_DIM || head >= HEADS) {
      return;
    }
    const int kv_head = head / (HEADS / KV_HEADS);

    const float max_score = max_scores[head];
    float weighted = 0.0f;
    for (int token = 0; token < COLD_TOKENS; ++token) {
      const int block = token / BLOCK_TOKENS;
      const int token_offset = token - block * BLOCK_TOKENS;
      const int slot = block * BLOCK_BYTES
        + token_offset * TOKEN_STRIDE_BYTES
        + kv_head * SLOT_BYTES;
      const float weight = exp(scores[head * COLD_TOKENS + token] - max_score);
      const int group = dim / VALUE_GROUP_SIZE;
      const int value_payload = slot + VALUE_PAYLOAD_OFFSET;
      const float value_min = tq_read_f32(compressed, slot + VALUE_MINS_OFFSET + group * 4);
      const float value_scale = tq_read_f32(compressed, slot + VALUE_SCALES_OFFSET + group * 4);
      const float value = value_min + value_scale * (float)tq_unpack_v4(compressed, value_payload, dim);

      weighted += weight * value;
    }

    weighted_value_sum[head * HEAD_DIM + dim] = weighted;
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::{
        TurboQuantBlockLayout, TurboQuantBlockLayoutConfig, TurboQuantCompressedDecodePlan,
    };

    #[test]
    fn turboquant_fused_cold_decode_metal_matches_reference_for_k8v4() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 256,
            n_kv_heads: 1,
            head_dim: 128,
            value_group_size: 32,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 2, 0).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let tokens = [
            (
                (0..128)
                    .map(|idx| ((idx % 17) as f32 - 8.0) / 16.0)
                    .collect::<Vec<_>>(),
                (0..128)
                    .map(|idx| ((idx % 11) as f32 - 5.0) / 8.0)
                    .collect::<Vec<_>>(),
            ),
            (
                (0..128)
                    .map(|idx| ((idx % 13) as f32 - 6.0) / 12.0)
                    .collect::<Vec<_>>(),
                (0..128)
                    .map(|idx| ((idx % 7) as f32 - 3.0) / 6.0)
                    .collect::<Vec<_>>(),
            ),
        ];
        for (token_index, token) in tokens.iter().enumerate() {
            buffer
                .write_token(token_index, std::slice::from_ref(token))
                .expect("token should compress");
        }
        let queries = vec![
            ((0..128)
                .map(|idx| ((idx % 19) as f32 - 9.0) / 10.0)
                .collect::<Vec<_>>()),
        ];
        let descriptor = plan
            .fused_decode_launch_descriptor(&buffer, &queries)
            .expect("descriptor should build");

        let expected = buffer
            .debug_decode_attention_for_all_heads(&queries, 2)
            .expect("reference decode should work");
        let expected_stats = buffer
            .debug_decode_partition_stats_for_all_heads(&queries, 2)
            .expect("reference partition stats should work");
        let actual_stats = turboquant_fused_cold_decode_metal_two_stage_partition_stats(
            descriptor, &buffer, &queries,
        )
        .expect("two-stage Metal partition stats should launch");
        assert_eq!(actual_stats.len(), expected_stats.len());
        for (expected, actual) in expected_stats.iter().zip(&actual_stats) {
            assert_eq!(actual.token_count, expected.token_count);
            assert_eq!(actual.value_dim, expected.value_dim);
            assert!((actual.max_score - expected.max_score).abs() < 0.0001);
            assert!((actual.exp_sum - expected.exp_sum).abs() < 0.0001);
            for (expected, actual) in expected
                .weighted_value_sum
                .iter()
                .zip(&actual.weighted_value_sum)
            {
                assert!((actual - expected).abs() < 0.0001);
            }
        }

        for (variant, actual) in [
            (
                "dim_parallel",
                turboquant_fused_cold_decode_metal(descriptor, &buffer, &queries)
                    .expect("Metal fused decode should launch"),
            ),
            (
                "head_serial",
                turboquant_fused_cold_decode_metal_head_serial(descriptor, &buffer, &queries)
                    .expect("head-serial Metal fused decode should launch"),
            ),
            (
                "two_stage_scores",
                turboquant_fused_cold_decode_metal_two_stage(descriptor, &buffer, &queries)
                    .expect("two-stage Metal fused decode should launch"),
            ),
        ] {
            let report = crate::turboquant::compare_decode_outputs(&expected, &actual)
                .expect("comparison should work");

            assert!(
                report.max_abs_diff < 0.0001,
                "{variant} Metal fused decode diverged from reference: {report:?}"
            );
        }
    }

    #[test]
    fn turboquant_two_stage_metal_matches_reference_for_256_dim_gqa() {
        assert_two_stage_metal_matches_reference_for_dim(256);
    }

    #[test]
    fn turboquant_two_stage_metal_matches_reference_for_512_dim_gqa() {
        assert_two_stage_metal_matches_reference_for_dim(512);
    }

    fn assert_two_stage_metal_matches_reference_for_dim(head_dim: usize) {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: MlxTurboQuantPreset::K8V4,
            block_tokens: 256,
            n_kv_heads: 2,
            head_dim,
            value_group_size: 32,
        })
        .expect("layout should build");
        let plan = TurboQuantCompressedDecodePlan::new(layout, 2, 0).expect("plan should build");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        for token_index in 0..2 {
            let heads = vec![
                (
                    (0..head_dim)
                        .map(|idx| ((idx % 17) as f32 - 8.0) / 16.0 + token_index as f32 * 0.01)
                        .collect::<Vec<_>>(),
                    (0..head_dim)
                        .map(|idx| ((idx % 11) as f32 - 5.0) / 8.0 - token_index as f32 * 0.02)
                        .collect::<Vec<_>>(),
                ),
                (
                    (0..head_dim)
                        .map(|idx| ((idx % 13) as f32 - 6.0) / 12.0 - token_index as f32 * 0.01)
                        .collect::<Vec<_>>(),
                    (0..head_dim)
                        .map(|idx| ((idx % 7) as f32 - 3.0) / 6.0 + token_index as f32 * 0.02)
                        .collect::<Vec<_>>(),
                ),
            ];
            buffer
                .write_token(token_index, &heads)
                .expect("token should compress");
        }
        let queries = (0..4)
            .map(|head_index| {
                (0..head_dim)
                    .map(|idx| ((idx % 19) as f32 - 9.0) / 10.0 + head_index as f32 * 0.005)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let descriptor = plan
            .fused_decode_launch_descriptor(&buffer, &queries)
            .expect("descriptor should build");
        assert_eq!(descriptor.n_query_heads, 4);
        assert_eq!(descriptor.n_kv_heads, 2);
        assert_eq!(descriptor.head_dim, head_dim);

        let expected = buffer
            .debug_decode_partition_stats_for_all_heads(&queries, 2)
            .expect("reference partition stats should work");
        let actual = turboquant_fused_cold_decode_metal_two_stage_partition_stats(
            descriptor, &buffer, &queries,
        )
        .expect("two-stage Metal partition stats should launch");

        assert_eq!(actual.len(), expected.len());
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_eq!(actual.token_count, expected.token_count);
            assert_eq!(actual.value_dim, expected.value_dim);
            assert!((actual.max_score - expected.max_score).abs() < 0.0001);
            assert!((actual.exp_sum - expected.exp_sum).abs() < 0.0001);
            for (expected, actual) in expected
                .weighted_value_sum
                .iter()
                .zip(&actual.weighted_value_sum)
            {
                assert!((actual - expected).abs() < 0.0001);
            }
        }
    }
}
