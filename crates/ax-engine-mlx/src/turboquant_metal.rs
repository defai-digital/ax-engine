use std::sync::OnceLock;

use ax_engine_core::TurboQuantPreset;
use mlx_sys::{KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, eval};

use crate::turboquant::{
    TurboQuantAttentionPartitionStats, TurboQuantAttentionPartitionStatsBatch,
    TurboQuantCodecError, TurboQuantCompressedBlockBuffer, TurboQuantFusedDecodeCandidateStatus,
    TurboQuantFusedDecodeLaunchDescriptor, TurboQuantRotationSigns, TurboQuantVectorRole,
    merge_attention_partition_stats, randomized_hadamard_in_place,
    turboquant_query_head_to_kv_head, turboquant_rotation_seed,
};

static TURBOQUANT_FUSED_KEY_ENCODE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static TURBOQUANT_FUSED_COLD_DECODE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static TURBOQUANT_FUSED_COLD_DECODE_HEAD_SERIAL_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static TURBOQUANT_FUSED_COLD_DECODE_SCORE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static TURBOQUANT_FUSED_COLD_DECODE_HEAD_STATS_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static TURBOQUANT_FUSED_COLD_DECODE_VALUE_SUM_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantFusedKeyEncodeResult {
    pub token_start: usize,
    pub token_count: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub packed_key_bytes: Vec<u8>,
    pub key_norms: Vec<f32>,
}

pub fn turboquant_fused_key_encode_metal_k8(
    keys: &MlxArray,
    token_start: usize,
    token_count: usize,
) -> Result<TurboQuantFusedKeyEncodeResult, TurboQuantCodecError> {
    let shape = keys.shape();
    if shape.len() != 4 || shape[0] != 1 {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: 4,
            actual: shape.len(),
        });
    }
    if keys.dtype() != MlxDtype::Float32 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: TurboQuantFusedDecodeCandidateStatus::UnsupportedPreset,
        });
    }
    let n_kv_heads = usize::try_from(shape[1]).unwrap_or(usize::MAX);
    let capacity_tokens = usize::try_from(shape[2]).unwrap_or(usize::MAX);
    let head_dim = usize::try_from(shape[3]).unwrap_or(usize::MAX);
    if token_count == 0 {
        return Ok(TurboQuantFusedKeyEncodeResult {
            token_start,
            token_count,
            n_kv_heads,
            head_dim,
            packed_key_bytes: Vec::new(),
            key_norms: Vec::new(),
        });
    }
    if token_start.saturating_add(token_count) > capacity_tokens {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: capacity_tokens,
            actual: token_start.saturating_add(token_count),
        });
    }
    if head_dim == 0 || !head_dim.is_power_of_two() || head_dim > 512 {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: TurboQuantFusedDecodeCandidateStatus::UnsupportedHeadDim,
        });
    }

    let mut signs = Vec::with_capacity(n_kv_heads.saturating_mul(head_dim));
    for head_index in 0..n_kv_heads {
        let seed = turboquant_rotation_seed(head_dim, head_index, TurboQuantVectorRole::Key);
        let head_signs = TurboQuantRotationSigns::new(head_dim, seed)?;
        for dim in 0..head_dim {
            signs.push(head_signs.sign_at(dim));
        }
    }
    let signs = MlxArray::from_raw_data(
        signs.as_ptr().cast(),
        std::mem::size_of_val(signs.as_slice()),
        &[n_kv_heads as i32, head_dim as i32],
        MlxDtype::Float32,
    );

    let kernel = TURBOQUANT_FUSED_KEY_ENCODE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "turboquant_fused_key_encode_k8_rht",
            &["keys", "signs"],
            &["packed", "norms"],
            TURBOQUANT_FUSED_KEY_ENCODE_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let vector_count = token_count.saturating_mul(n_kv_heads);
    let outputs = kernel.apply_with_template(
        &[keys, &signs],
        &[
            KernelOutputSpec {
                shape: vec![vector_count as i32, head_dim as i32],
                dtype: MlxDtype::Uint8,
            },
            KernelOutputSpec {
                shape: vec![vector_count as i32],
                dtype: MlxDtype::Float32,
            },
        ],
        &[
            KernelTemplateArg::Int {
                name: "TOKEN_START",
                value: token_start as i32,
            },
            KernelTemplateArg::Int {
                name: "TOKEN_COUNT",
                value: token_count as i32,
            },
            KernelTemplateArg::Int {
                name: "KV_HEADS",
                value: n_kv_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "CAPACITY_TOKENS",
                value: capacity_tokens as i32,
            },
            KernelTemplateArg::Int {
                name: "HEAD_DIM",
                value: head_dim as i32,
            },
        ],
        (vector_count as i32 * head_dim as i32, 1, 1),
        (head_dim as i32, 1, 1),
        None,
    );
    let mut outputs = outputs.into_iter();
    let packed = outputs
        .next()
        .expect("TurboQuant fused key encode packed output");
    let norms = outputs
        .next()
        .expect("TurboQuant fused key encode norm output");
    eval(&[&packed, &norms]);
    let packed_key_bytes =
        unsafe { std::slice::from_raw_parts(packed.data_raw(), packed.nbytes()).to_vec() };

    Ok(TurboQuantFusedKeyEncodeResult {
        token_start,
        token_count,
        n_kv_heads,
        head_dim,
        packed_key_bytes,
        key_norms: norms.data_f32().to_vec(),
    })
}

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

fn validate_fused_decode_launch(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    queries: &[Vec<f32>],
) -> Result<(), TurboQuantCodecError> {
    validate_fused_decode_launch_for_query_count(descriptor, queries.len())?;
    for query in queries {
        if query.len() != descriptor.head_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: descriptor.head_dim,
                actual: query.len(),
            });
        }
    }
    Ok(())
}

fn validate_fused_decode_launch_flat(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    query_values: &[f32],
    n_query_heads: usize,
) -> Result<(), TurboQuantCodecError> {
    validate_fused_decode_launch_for_query_count(descriptor, n_query_heads)?;
    let expected_len = n_query_heads.saturating_mul(descriptor.head_dim);
    if query_values.len() != expected_len {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: expected_len,
            actual: query_values.len(),
        });
    }
    Ok(())
}

fn validate_fused_decode_launch_for_query_count(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    n_query_heads: usize,
) -> Result<(), TurboQuantCodecError> {
    if descriptor.preset != TurboQuantPreset::K8V4 {
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
    validate_query_head_mapping(descriptor, n_query_heads)
}

fn rotated_query_values_from_flat(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    query_values: &[f32],
    n_query_heads: usize,
) -> Result<Vec<f32>, TurboQuantCodecError> {
    validate_fused_decode_launch_flat(descriptor, query_values, n_query_heads)?;
    let mut rotated_queries = Vec::with_capacity(query_values.len());
    for (query_head_index, query) in query_values.chunks_exact(descriptor.head_dim).enumerate() {
        let mut rotated = query.to_vec();
        rotate_query_for_descriptor(&mut rotated, descriptor, query_head_index)?;
        rotated_queries.extend(rotated);
    }
    Ok(rotated_queries)
}

fn rotate_query_for_descriptor(
    query: &mut [f32],
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    query_head_index: usize,
) -> Result<(), TurboQuantCodecError> {
    let kv_head_index = turboquant_query_head_to_kv_head(
        query_head_index,
        descriptor.n_query_heads,
        descriptor.n_kv_heads,
    )?;
    let seed = turboquant_rotation_seed(
        descriptor.head_dim,
        kv_head_index,
        TurboQuantVectorRole::Key,
    );
    randomized_hadamard_in_place(query, seed)
}

pub fn turboquant_fused_cold_decode_metal(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    buffer: &TurboQuantCompressedBlockBuffer,
    queries: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
    validate_fused_decode_launch(descriptor, queries)?;

    let mut rotated_queries = Vec::with_capacity(descriptor.n_query_heads * descriptor.head_dim);
    for (query_head_index, query) in queries.iter().enumerate() {
        if query.len() != descriptor.head_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: descriptor.head_dim,
                actual: query.len(),
            });
        }
        let mut rotated = query.clone();
        rotate_query_for_descriptor(&mut rotated, descriptor, query_head_index)?;
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
        std::mem::size_of_val(rotated_queries.as_slice()),
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
            descriptor.head_dim as i32 * 32,
            descriptor.n_query_heads as i32,
            1,
        ),
        (32, 1, 1),
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
    validate_fused_decode_launch(descriptor, queries)?;
    if descriptor.head_dim != 128 || descriptor.n_query_heads != descriptor.n_kv_heads {
        return Err(TurboQuantCodecError::FusedDecodeLaunchRejected {
            status: crate::turboquant::TurboQuantFusedDecodeCandidateStatus::UnsupportedHeadDim,
        });
    }

    let mut rotated_queries = Vec::with_capacity(descriptor.n_kv_heads * descriptor.head_dim);
    for (query_head_index, query) in queries.iter().enumerate() {
        if query.len() != descriptor.head_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: descriptor.head_dim,
                actual: query.len(),
            });
        }
        let mut rotated = query.clone();
        rotate_query_for_descriptor(&mut rotated, descriptor, query_head_index)?;
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
        std::mem::size_of_val(rotated_queries.as_slice()),
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
    let compressed = MlxArray::from_raw_data(
        buffer.as_bytes().as_ptr(),
        buffer.as_bytes().len(),
        &[buffer.as_bytes().len() as i32],
        MlxDtype::Uint8,
    );
    turboquant_fused_cold_decode_metal_two_stage_stats(descriptor, &compressed, queries)
}

pub fn turboquant_fused_cold_decode_metal_two_stage_partition_stats_flat(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    buffer: &TurboQuantCompressedBlockBuffer,
    query_values: &[f32],
    n_query_heads: usize,
) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
    let compressed = MlxArray::from_raw_data(
        buffer.as_bytes().as_ptr(),
        buffer.as_bytes().len(),
        &[buffer.as_bytes().len() as i32],
        MlxDtype::Uint8,
    );
    turboquant_fused_cold_decode_metal_two_stage_partition_stats_with_compressed_array_flat(
        descriptor,
        &compressed,
        query_values,
        n_query_heads,
    )
}

pub fn turboquant_fused_cold_decode_metal_two_stage_partition_stats_with_compressed_array(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    compressed: &MlxArray,
    queries: &[Vec<f32>],
) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
    turboquant_fused_cold_decode_metal_two_stage_stats(descriptor, compressed, queries)
}

pub fn turboquant_fused_cold_decode_metal_two_stage_partition_stats_with_compressed_array_flat(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    compressed: &MlxArray,
    query_values: &[f32],
    n_query_heads: usize,
) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
    let batch =
        turboquant_fused_cold_decode_metal_two_stage_partition_stats_batch_with_compressed_array_flat(
            descriptor,
            compressed,
            query_values,
            n_query_heads,
        )?;
    (0..batch.query_heads())
        .map(|head_index| batch.partition_stats(head_index))
        .collect()
}

pub fn turboquant_fused_cold_decode_metal_two_stage_sparse_partition_stats_flat(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    buffer: &TurboQuantCompressedBlockBuffer,
    query_values: &[f32],
    n_query_heads: usize,
    sparse_value_threshold: f32,
) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
    let compressed = MlxArray::from_raw_data(
        buffer.as_bytes().as_ptr(),
        buffer.as_bytes().len(),
        &[buffer.as_bytes().len() as i32],
        MlxDtype::Uint8,
    );
    let batch =
        turboquant_fused_cold_decode_metal_two_stage_stats_batch_flat_with_sparse_threshold(
            descriptor,
            &compressed,
            query_values,
            n_query_heads,
            sparse_value_threshold,
        )?;
    (0..batch.query_heads())
        .map(|head_index| batch.partition_stats(head_index))
        .collect()
}

pub fn turboquant_fused_cold_decode_metal_two_stage_partition_stats_batch_with_compressed_array_flat(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    compressed: &MlxArray,
    query_values: &[f32],
    n_query_heads: usize,
) -> Result<TurboQuantAttentionPartitionStatsBatch, TurboQuantCodecError> {
    turboquant_fused_cold_decode_metal_two_stage_partition_stats_batch_with_compressed_array_flat_sparse_threshold(
        descriptor,
        compressed,
        query_values,
        n_query_heads,
        0.0,
    )
}

pub fn turboquant_fused_cold_decode_metal_two_stage_partition_stats_batch_with_compressed_array_flat_sparse_threshold(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    compressed: &MlxArray,
    query_values: &[f32],
    n_query_heads: usize,
    sparse_value_threshold: f32,
) -> Result<TurboQuantAttentionPartitionStatsBatch, TurboQuantCodecError> {
    let rotated_queries = rotated_query_values_from_flat(descriptor, query_values, n_query_heads)?;
    turboquant_fused_cold_decode_metal_two_stage_stats_batch_with_rotated_queries_and_sparse_threshold(
        descriptor,
        compressed,
        &rotated_queries,
        sparse_value_threshold,
    )
}

fn turboquant_fused_cold_decode_metal_two_stage_stats_batch_flat_with_sparse_threshold(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    compressed: &MlxArray,
    query_values: &[f32],
    n_query_heads: usize,
    sparse_value_threshold: f32,
) -> Result<TurboQuantAttentionPartitionStatsBatch, TurboQuantCodecError> {
    let rotated_queries = rotated_query_values_from_flat(descriptor, query_values, n_query_heads)?;
    turboquant_fused_cold_decode_metal_two_stage_stats_batch_with_rotated_queries_and_sparse_threshold(
        descriptor,
        compressed,
        &rotated_queries,
        sparse_value_threshold,
    )
}

fn turboquant_fused_cold_decode_metal_two_stage_stats(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    compressed: &MlxArray,
    queries: &[Vec<f32>],
) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
    validate_fused_decode_launch(descriptor, queries)?;

    let mut rotated_queries = Vec::with_capacity(descriptor.n_query_heads * descriptor.head_dim);
    for (query_head_index, query) in queries.iter().enumerate() {
        if query.len() != descriptor.head_dim {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: descriptor.head_dim,
                actual: query.len(),
            });
        }
        let mut rotated = query.clone();
        rotate_query_for_descriptor(&mut rotated, descriptor, query_head_index)?;
        rotated_queries.extend(rotated);
    }

    turboquant_fused_cold_decode_metal_two_stage_stats_with_rotated_queries(
        descriptor,
        compressed,
        &rotated_queries,
    )
}

fn turboquant_fused_cold_decode_metal_two_stage_stats_with_rotated_queries(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    compressed: &MlxArray,
    rotated_queries: &[f32],
) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
    let batch = turboquant_fused_cold_decode_metal_two_stage_stats_batch_with_rotated_queries(
        descriptor,
        compressed,
        rotated_queries,
    )?;
    (0..batch.query_heads())
        .map(|head_index| batch.partition_stats(head_index))
        .collect()
}

fn turboquant_fused_cold_decode_metal_two_stage_stats_batch_with_rotated_queries(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    compressed: &MlxArray,
    rotated_queries: &[f32],
) -> Result<TurboQuantAttentionPartitionStatsBatch, TurboQuantCodecError> {
    turboquant_fused_cold_decode_metal_two_stage_stats_batch_with_rotated_queries_and_sparse_threshold(
        descriptor,
        compressed,
        rotated_queries,
        0.0,
    )
}

fn turboquant_fused_cold_decode_metal_two_stage_stats_batch_with_rotated_queries_and_sparse_threshold(
    descriptor: TurboQuantFusedDecodeLaunchDescriptor,
    compressed: &MlxArray,
    rotated_queries: &[f32],
    sparse_value_threshold: f32,
) -> Result<TurboQuantAttentionPartitionStatsBatch, TurboQuantCodecError> {
    let query = MlxArray::from_raw_data(
        rotated_queries.as_ptr().cast(),
        std::mem::size_of_val(rotated_queries),
        &[descriptor.n_query_heads as i32, descriptor.head_dim as i32],
        MlxDtype::Float32,
    );

    let score_kernel = TURBOQUANT_FUSED_COLD_DECODE_SCORE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "turboquant_fused_cold_decode_k8v4_scores_simd32",
            &["compressed", "rotated_query"],
            &["scores"],
            TURBOQUANT_FUSED_COLD_DECODE_SCORE_KERNEL_SOURCE,
            TURBOQUANT_FUSED_COLD_DECODE_KERNEL_HEADER,
            true,
        )
    });
    let score_outputs = score_kernel.apply_with_template(
        &[compressed, &query],
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
            descriptor.cold_tokens as i32 * 32,
            descriptor.n_query_heads as i32,
            1,
        ),
        (32, 1, 1),
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
        (descriptor.n_query_heads as i32 * 32, 1, 1),
        (32, 1, 1),
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
            &[
                "compressed",
                "scores",
                "max_scores",
                "exp_sums",
                "threshold",
            ],
            &["weighted_value_sum"],
            TURBOQUANT_FUSED_COLD_DECODE_VALUE_SUM_KERNEL_SOURCE,
            TURBOQUANT_FUSED_COLD_DECODE_KERNEL_HEADER,
            true,
        )
    });
    let sparse_threshold = [sparse_value_threshold.max(0.0)];
    let sparse_threshold = MlxArray::from_raw_data(
        sparse_threshold.as_ptr().cast(),
        std::mem::size_of_val(sparse_threshold.as_slice()),
        &[1],
        MlxDtype::Float32,
    );
    let value_outputs = value_sum_kernel.apply_with_template(
        &[
            compressed,
            &scores,
            &max_scores,
            &exp_sums,
            &sparse_threshold,
        ],
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
        (descriptor.value_group_size as i32, 1, 1),
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
    Ok(TurboQuantAttentionPartitionStatsBatch {
        token_count: descriptor.cold_tokens,
        value_dim: descriptor.head_dim,
        max_scores: max_scores.to_vec(),
        exp_sums: exp_sums.to_vec(),
        weighted_value_sums: weighted_value_sum.to_vec(),
    })
}

const TURBOQUANT_FUSED_KEY_ENCODE_KERNEL_SOURCE: &str = r#"
    const int lane = (int)thread_position_in_threadgroup.x;
    const int vector = (int)thread_position_in_grid.x / HEAD_DIM;
    if (vector >= TOKEN_COUNT * KV_HEADS || lane >= HEAD_DIM) {
      return;
    }
    const int token = vector / KV_HEADS;
    const int head = vector - token * KV_HEADS;
    const int source_token = TOKEN_START + token;
    const int source_offset = ((head * CAPACITY_TOKENS + source_token) * HEAD_DIM) + lane;

    threadgroup float values[512];
    threadgroup float reductions[512];

    const float input_value = keys[source_offset];
    values[lane] = input_value;
    reductions[lane] = input_value * input_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
      if (lane < stride) {
        reductions[lane] += reductions[lane + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float vec_norm = sqrt(reductions[0]);
    const float safe_norm = max(vec_norm, 1.19209290e-7f);
    values[lane] = (values[lane] / safe_norm) * signs[head * HEAD_DIM + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int h = 1;
    while (h < HEAD_DIM) {
      const int block = lane / (2 * h);
      const int offset = lane - block * 2 * h;
      if (offset < h) {
        const int j = block * 2 * h + offset;
        const float left = values[j];
        const float right = values[j + h];
        values[j] = left + right;
        values[j + h] = left - right;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      h <<= 1;
    }

    const float rotated = values[lane] * rsqrt((float)HEAD_DIM);
    const float projected = (rotated + 1.0f) * 127.5f + 0.5f - 0.000001f;
    const int centroid = clamp((int)floor(projected), 0, 255);
    packed[vector * HEAD_DIM + lane] = (uint8_t)centroid;
    if (lane == 0) {
      norms[vector] = vec_norm;
    }
"#;

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
    const int lane = (int)thread_position_in_threadgroup.x;
    const int dim = (int)thread_position_in_grid.x / 32;
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

      float partial = 0.0f;
      for (int kdim = lane; kdim < HEAD_DIM; kdim += 32) {
        const float centroid = tq_centroid_k8(compressed[key_payload + kdim]);
        partial += rotated_query[head * HEAD_DIM + kdim] * centroid;
      }
      const float score = simd_sum(partial) * key_norm * inv_sqrt_dim;
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

      float partial = 0.0f;
      for (int kdim = lane; kdim < HEAD_DIM; kdim += 32) {
        const float centroid = tq_centroid_k8(compressed[key_payload + kdim]);
        partial += rotated_query[head * HEAD_DIM + kdim] * centroid;
      }
      const float score = simd_sum(partial) * key_norm * inv_sqrt_dim;

      const float weight = exp(score - max_score);
      if (lane == 0) {
        const int group = dim / VALUE_GROUP_SIZE;
        const int value_payload = slot + VALUE_PAYLOAD_OFFSET;
        const float value_min = tq_read_f32(compressed, slot + VALUE_MINS_OFFSET + group * 4);
        const float value_scale = tq_read_f32(compressed, slot + VALUE_SCALES_OFFSET + group * 4);
        const float value = value_min + value_scale * (float)tq_unpack_v4(compressed, value_payload, dim);

        denom += weight;
        weighted += weight * value;
      }
    }

    if (lane == 0) {
      output[head * HEAD_DIM + dim] = weighted / max(denom, 1.17549435e-38f);
    }
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
    const int lane = (int)thread_position_in_threadgroup.x;
    const int token = (int)thread_position_in_grid.x / 32;
    const int head = (int)thread_position_in_grid.y;
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

    float partial = 0.0f;
    for (int kdim = lane; kdim < HEAD_DIM; kdim += 32) {
      const float centroid = tq_centroid_k8(compressed[key_payload + kdim]);
      partial += rotated_query[head * HEAD_DIM + kdim] * centroid;
    }
    const float score = simd_sum(partial);

    if (lane == 0) {
      scores[head * COLD_TOKENS + token] = score * key_norm * inv_sqrt_dim;
    }
"#;

const TURBOQUANT_FUSED_COLD_DECODE_HEAD_STATS_KERNEL_SOURCE: &str = r#"
    const int lane = (int)thread_position_in_threadgroup.x;
    const int head = (int)thread_position_in_grid.x / 32;
    if (head >= HEADS) {
      return;
    }

    float max_score = -3.4028234663852886e+38f;
    for (int token = lane; token < COLD_TOKENS; token += 32) {
      max_score = max(max_score, scores[head * COLD_TOKENS + token]);
    }
    max_score = simd_max(max_score);

    float denom = 0.0f;
    for (int token = lane; token < COLD_TOKENS; token += 32) {
      denom += exp(scores[head * COLD_TOKENS + token] - max_score);
    }
    denom = simd_sum(denom);

    if (lane == 0) {
      max_scores[head] = max_score;
      exp_sums[head] = denom;
    }
"#;

const TURBOQUANT_FUSED_COLD_DECODE_VALUE_SUM_KERNEL_SOURCE: &str = r#"
    const int dim = (int)thread_position_in_grid.x;
    const int group = dim / VALUE_GROUP_SIZE;
    const int head = thread_position_in_grid.y;
    if (head >= HEADS) {
      return;
    }
    const int kv_head = head / (HEADS / KV_HEADS);

    const float max_score = max_scores[head];
    const float denom = max(exp_sums[head], 1.17549435e-38f);
    const float min_weight = max(threshold[0], 0.0f);
    float weighted = 0.0f;
    for (int token = 0; token < COLD_TOKENS; ++token) {
      const int block = token / BLOCK_TOKENS;
      const int token_offset = token - block * BLOCK_TOKENS;
      const int slot = block * BLOCK_BYTES
        + token_offset * TOKEN_STRIDE_BYTES
        + kv_head * SLOT_BYTES;
      const float weight = exp(scores[head * COLD_TOKENS + token] - max_score);
      if ((weight / denom) < min_weight) {
        continue;
      }
      if (dim < HEAD_DIM) {
        const int value_payload = slot + VALUE_PAYLOAD_OFFSET;
        const float value_min = tq_read_f32(compressed, slot + VALUE_MINS_OFFSET + group * 4);
        const float value_scale = tq_read_f32(compressed, slot + VALUE_SCALES_OFFSET + group * 4);
        const float value = value_min + value_scale * (float)tq_unpack_v4(compressed, value_payload, dim);
        weighted += weight * value;
      }
    }

    if (dim < HEAD_DIM) {
      weighted_value_sum[head * HEAD_DIM + dim] = weighted;
    }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::{
        TurboQuantBlockLayout, TurboQuantBlockLayoutConfig, TurboQuantCompressedDecodePlan,
        encode_key_vector_for_head,
    };

    #[test]
    fn turboquant_dim_parallel_decode_kernel_uses_simd_sum_for_qk_scores() {
        assert!(TURBOQUANT_FUSED_COLD_DECODE_KERNEL_SOURCE.contains("simd_sum(partial)"));
        assert!(
            !TURBOQUANT_FUSED_COLD_DECODE_KERNEL_SOURCE
                .contains("for (int kdim = 0; kdim < HEAD_DIM; ++kdim)")
        );
    }

    #[test]
    fn turboquant_fused_key_encode_metal_matches_cpu_k8v4() {
        let n_kv_heads = 2usize;
        let capacity_tokens = 3usize;
        let head_dim = 8usize;
        let mut data = vec![0.0f32; n_kv_heads * capacity_tokens * head_dim];
        for head in 0..n_kv_heads {
            for token in 0..capacity_tokens {
                for dim in 0..head_dim {
                    let idx = (head * capacity_tokens + token) * head_dim + dim;
                    data[idx] = ((idx % 23) as f32 - 11.0) / 13.0 + token as f32 * 0.03
                        - head as f32 * 0.02;
                }
            }
        }
        let keys = MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[
                1,
                n_kv_heads as i32,
                capacity_tokens as i32,
                head_dim as i32,
            ],
            MlxDtype::Float32,
        );

        let actual = turboquant_fused_key_encode_metal_k8(&keys, 1, 2).expect("Metal key encode");

        assert_eq!(actual.token_start, 1);
        assert_eq!(actual.token_count, 2);
        assert_eq!(actual.n_kv_heads, n_kv_heads);
        assert_eq!(actual.head_dim, head_dim);
        assert_eq!(actual.packed_key_bytes.len(), 2 * n_kv_heads * head_dim);
        assert_eq!(actual.key_norms.len(), 2 * n_kv_heads);

        for rel_token in 0..actual.token_count {
            let source_token = actual.token_start + rel_token;
            for head in 0..n_kv_heads {
                let vector_index = rel_token * n_kv_heads + head;
                let source_offset = (head * capacity_tokens + source_token) * head_dim;
                let expected = encode_key_vector_for_head(
                    &data[source_offset..source_offset + head_dim],
                    TurboQuantPreset::K8V4,
                    head,
                )
                .expect("CPU key encode");
                let packed_start = vector_index * head_dim;
                assert_eq!(
                    &actual.packed_key_bytes[packed_start..packed_start + head_dim],
                    expected.packed_indices.as_slice()
                );
                assert!(
                    (actual.key_norms[vector_index] - expected.l2_norm).abs() < 1e-6,
                    "norm mismatch for token {source_token} head {head}: actual={} expected={}",
                    actual.key_norms[vector_index],
                    expected.l2_norm
                );
            }
        }
    }

    #[test]
    fn turboquant_fused_cold_decode_metal_matches_reference_for_k8v4() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
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
        let flat_queries = queries
            .iter()
            .flat_map(|query| query.iter().copied())
            .collect::<Vec<_>>();
        let sparse_zero_stats =
            turboquant_fused_cold_decode_metal_two_stage_sparse_partition_stats_flat(
                descriptor,
                &buffer,
                &flat_queries,
                queries.len(),
                0.0,
            )
            .expect("sparse threshold=0 stats should launch");
        assert_eq!(sparse_zero_stats.len(), actual_stats.len());
        for (expected, actual) in actual_stats.iter().zip(&sparse_zero_stats) {
            assert_eq!(actual.token_count, expected.token_count);
            assert_eq!(actual.value_dim, expected.value_dim);
            assert!((actual.max_score - expected.max_score).abs() < 0.0001);
            assert!((actual.exp_sum - expected.exp_sum).abs() < 0.0001);
            for (dim, (expected, actual)) in expected
                .weighted_value_sum
                .iter()
                .zip(&actual.weighted_value_sum)
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() < 0.0001,
                    "sparse threshold=0 mismatch at dim {dim}: actual={actual} expected={expected}"
                );
            }
        }
        let sparse_all_skipped =
            turboquant_fused_cold_decode_metal_two_stage_sparse_partition_stats_flat(
                descriptor,
                &buffer,
                &flat_queries,
                queries.len(),
                2.0,
            )
            .expect("sparse all-skipped stats should launch");
        for stats in sparse_all_skipped {
            assert!(stats.weighted_value_sum.iter().all(|value| *value == 0.0));
        }
        assert_eq!(actual_stats.len(), expected_stats.len());
        for (expected, actual) in expected_stats.iter().zip(&actual_stats) {
            assert_eq!(actual.token_count, expected.token_count);
            assert_eq!(actual.value_dim, expected.value_dim);
            assert!((actual.max_score - expected.max_score).abs() < 0.0001);
            assert!((actual.exp_sum - expected.exp_sum).abs() < 0.0001);
            for (dim, (expected, actual)) in expected
                .weighted_value_sum
                .iter()
                .zip(&actual.weighted_value_sum)
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() < 0.0001,
                    "weighted value mismatch at dim {dim}: actual={actual} expected={expected}"
                );
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
        assert_two_stage_metal_matches_reference_for_dim(256, 32);
    }

    #[test]
    fn turboquant_two_stage_metal_matches_reference_for_512_dim_gqa() {
        assert_two_stage_metal_matches_reference_for_dim(512, 32);
    }

    #[test]
    fn turboquant_two_stage_metal_matches_reference_for_value_group_size_above_simd_width() {
        // Regression: the value-sum kernel previously fetched value mins/scales on
        // lane 0 and used simd_broadcast_first, which only covers the first 32-wide
        // SIMD-group of the threadgroup. Group sizes above 32 silently decoded the
        // remaining dims of each group with min=0/scale=0.
        assert_two_stage_metal_matches_reference_for_dim(256, 64);
        assert_two_stage_metal_matches_reference_for_dim(512, 128);
    }

    fn assert_two_stage_metal_matches_reference_for_dim(head_dim: usize, value_group_size: usize) {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 256,
            n_kv_heads: 2,
            head_dim,
            value_group_size,
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
            for (dim, (expected, actual)) in expected
                .weighted_value_sum
                .iter()
                .zip(&actual.weighted_value_sum)
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() < 0.0001,
                    "weighted value mismatch at dim {dim}: actual={actual} expected={expected}"
                );
            }
        }
    }
}
