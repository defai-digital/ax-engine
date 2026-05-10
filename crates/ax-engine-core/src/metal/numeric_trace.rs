#[cfg(target_os = "macos")]
use super::MetalDispatchArena;
#[cfg(target_os = "macos")]
use super::buffer_io::read_shared_buffer_prefix;
use super::{
    MetalDispatchKvCacheSeed, MetalDispatchKvCacheSnapshot, MetalDispatchNumericTrace,
    MetalDispatchStagedInputs, MetalDispatchWorkload, MetalNumericValidationSummary,
    MetalRuntimeError, ReferenceAttentionConfig, merge_copy_targets_into_cache_snapshot,
    synthetic_staged_inputs,
};

#[derive(Clone, Debug)]
pub(super) struct ReferenceNumericPath {
    pub(super) key_cache: Vec<f32>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) value_cache: Vec<f32>,
    pub(super) gather_key: Vec<f32>,
    pub(super) gather_value: Vec<f32>,
    pub(super) attention_output: Vec<f32>,
    pub(super) copy_key: Vec<f32>,
    pub(super) copy_value: Vec<f32>,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn reference_numeric_path(workload: &MetalDispatchWorkload) -> ReferenceNumericPath {
    let staged_inputs = synthetic_staged_inputs(workload);
    reference_numeric_path_with_inputs(workload, &staged_inputs)
}

pub(super) fn reference_numeric_path_with_inputs(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
) -> ReferenceNumericPath {
    reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
        workload,
        staged_inputs,
        None,
        None,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn reference_numeric_path_with_inputs_and_cache_seed(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    cache_seed: Option<MetalDispatchKvCacheSeed<'_>>,
) -> ReferenceNumericPath {
    reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
        workload,
        staged_inputs,
        cache_seed,
        None,
    )
}

pub(super) fn reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    cache_seed: Option<MetalDispatchKvCacheSeed<'_>>,
    attention_config: Option<ReferenceAttentionConfig>,
) -> ReferenceNumericPath {
    let head_size = workload.numeric_layout.head_size() as usize;
    let head_count = workload.numeric_layout.head_count as usize;
    let head_dim = workload.numeric_layout.head_dim as usize;
    let attention_config = attention_config.unwrap_or_else(|| {
        ReferenceAttentionConfig::from_head_dim(head_dim).unwrap_or(ReferenceAttentionConfig {
            softmax_scale: 1.0,
            softcap: None,
        })
    });
    let scheduled_tokens = workload.token_elements as usize;
    let gather_tokens = workload.kv_metadata.gather_token_count() as usize;
    let mut key_cache = vec![0.0_f32; workload.slot_numeric_capacity() as usize];
    let mut value_cache = vec![0.0_f32; workload.slot_numeric_capacity() as usize];
    if let Some(cache_seed) = cache_seed {
        key_cache.copy_from_slice(cache_seed.key_cache);
        value_cache.copy_from_slice(cache_seed.value_cache);
    }

    for token_id in 0..scheduled_tokens {
        let slot = workload.kv_metadata.slot_mapping[token_id] as usize;
        let source_base = token_id * head_size;
        let cache_base = slot * head_size;
        key_cache[cache_base..cache_base + head_size]
            .copy_from_slice(&staged_inputs.key[source_base..source_base + head_size]);
        value_cache[cache_base..cache_base + head_size]
            .copy_from_slice(&staged_inputs.value[source_base..source_base + head_size]);
    }

    let mut gather_key = vec![0.0_f32; workload.gather_numeric_elements() as usize];
    let mut gather_value = vec![0.0_f32; workload.gather_numeric_elements() as usize];
    for token_id in 0..gather_tokens {
        let batch_id = batch_id_for_token(&workload.kv_metadata.cu_seq_lens, token_id as u32);
        let batch_offset = token_id as u32 - workload.kv_metadata.cu_seq_lens[batch_id];
        let block_index = (batch_offset / workload.kv_metadata.block_size_tokens) as usize;
        let block_offset = (batch_offset % workload.kv_metadata.block_size_tokens) as usize;
        let block_base = workload.kv_metadata.gather_block_table
            [batch_id * workload.kv_metadata.gather_block_table_stride as usize + block_index]
            as usize;
        let slot = block_base + block_offset;
        let source_base = slot * head_size;
        let target_base = token_id * head_size;
        gather_key[target_base..target_base + head_size]
            .copy_from_slice(&key_cache[source_base..source_base + head_size]);
        gather_value[target_base..target_base + head_size]
            .copy_from_slice(&value_cache[source_base..source_base + head_size]);
    }

    let mut attention_output = vec![0.0_f32; workload.attention_numeric_elements() as usize];
    for token_id in 0..scheduled_tokens {
        let batch_id =
            batch_id_for_token(&workload.kv_metadata.scheduled_cu_seq_lens, token_id as u32);
        let context_begin = workload.kv_metadata.cu_seq_lens[batch_id] as usize;
        let context_end = workload.kv_metadata.cu_seq_lens[batch_id + 1] as usize;
        let query_base = token_id * head_size;

        for head in 0..head_count {
            let head_query_base = query_base + head * head_dim;
            let mut max_score = f32::NEG_INFINITY;

            for context_index in context_begin..context_end {
                let context_base = context_index * head_size + head * head_dim;
                let score = configured_attention_score(
                    &staged_inputs.query[head_query_base..head_query_base + head_dim],
                    &gather_key[context_base..context_base + head_dim],
                    attention_config,
                );
                max_score = max_score.max(score);
            }

            let mut weight_sum = 0.0_f32;
            let mut accum = vec![0.0_f32; head_dim];
            for context_index in context_begin..context_end {
                let context_base = context_index * head_size + head * head_dim;
                let score = configured_attention_score(
                    &staged_inputs.query[head_query_base..head_query_base + head_dim],
                    &gather_key[context_base..context_base + head_dim],
                    attention_config,
                );
                let weight = (score - max_score).exp();
                weight_sum += weight;
                for lane in 0..head_dim {
                    accum[lane] += weight * gather_value[context_base + lane];
                }
            }

            for lane in 0..head_dim {
                attention_output[head_query_base + lane] = accum[lane] / weight_sum.max(0.000001);
            }
        }
    }

    let mut copy_key = vec![0.0_f32; workload.slot_numeric_capacity() as usize];
    let mut copy_value = vec![0.0_f32; workload.slot_numeric_capacity() as usize];
    for pair in &workload.kv_metadata.copy_block_mapping {
        let source_base = pair[0] as usize * head_size;
        let target_base = pair[1] as usize * head_size;
        let block_width = workload.block_numeric_elements() as usize;
        copy_key[target_base..target_base + block_width]
            .copy_from_slice(&key_cache[source_base..source_base + block_width]);
        copy_value[target_base..target_base + block_width]
            .copy_from_slice(&value_cache[source_base..source_base + block_width]);
    }

    ReferenceNumericPath {
        key_cache,
        value_cache,
        gather_key,
        gather_value,
        attention_output,
        copy_key,
        copy_value,
    }
}

pub(super) fn batch_id_for_token(cu_seq_lens: &[u32], token_id: u32) -> usize {
    let mut lo = 0_usize;
    let mut hi = cu_seq_lens.len().saturating_sub(1);

    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if cu_seq_lens[mid] <= token_id {
            lo = mid;
        } else {
            hi = mid.saturating_sub(1);
        }
    }

    lo.min(cu_seq_lens.len().saturating_sub(2))
}

pub(super) fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

pub(super) fn configured_attention_score(
    query: &[f32],
    key: &[f32],
    attention_config: ReferenceAttentionConfig,
) -> f32 {
    let mut score = dot_product(query, key) * attention_config.softmax_scale;
    if let Some(softcap) = attention_config.softcap {
        score = softcap * (score / softcap).tanh();
    }
    score
}

#[cfg(target_os = "macos")]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn validate_numeric_trace_against_reference(
    workload: &MetalDispatchWorkload,
    trace: &MetalDispatchNumericTrace,
) -> Result<MetalNumericValidationSummary, MetalRuntimeError> {
    let staged_inputs = synthetic_staged_inputs(workload);
    validate_numeric_trace_against_inputs(workload, &staged_inputs, trace)
}

#[cfg(target_os = "macos")]
pub(super) fn validate_numeric_trace_against_inputs(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    trace: &MetalDispatchNumericTrace,
) -> Result<MetalNumericValidationSummary, MetalRuntimeError> {
    validate_numeric_trace_against_inputs_and_cache_seed(workload, staged_inputs, None, None, trace)
}

#[cfg(target_os = "macos")]
pub(super) fn validate_numeric_trace_against_inputs_and_cache_seed(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    cache_seed: Option<MetalDispatchKvCacheSeed<'_>>,
    attention_config: Option<ReferenceAttentionConfig>,
    trace: &MetalDispatchNumericTrace,
) -> Result<MetalNumericValidationSummary, MetalRuntimeError> {
    let reference = reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
        workload,
        staged_inputs,
        cache_seed,
        attention_config,
    );

    let expected_key_cache_checksum =
        validate_numeric_checksum("key_cache", trace.key_cache_checksum, &reference.key_cache)?;
    let expected_gather_output_checksum = validate_numeric_pair_checksum(
        "gather_kv",
        trace.gather_output_checksum,
        &reference.gather_key,
        &reference.gather_value,
    )?;
    let expected_copy_output_checksum = validate_numeric_pair_checksum(
        "copy_blocks",
        trace.copy_output_checksum,
        &reference.copy_key,
        &reference.copy_value,
    )?;
    let (expected_attention_output_checksum, attention_max_abs_diff_microunits) =
        validate_attention_output_against_reference(
            &reference.attention_output,
            &trace.attention_output_bits,
            trace.attention_output_checksum,
        )?;

    Ok(MetalNumericValidationSummary {
        expected_key_cache_checksum,
        expected_attention_output_checksum,
        expected_gather_output_checksum,
        expected_copy_output_checksum,
        attention_max_abs_diff_microunits,
    })
}

#[cfg(target_os = "macos")]
fn validate_numeric_checksum(
    stage: &'static str,
    actual_checksum: u64,
    expected_values: &[f32],
) -> Result<u64, MetalRuntimeError> {
    let expected_checksum = checksum_f32_slice(expected_values);
    if actual_checksum == expected_checksum {
        return Ok(expected_checksum);
    }

    Err(MetalRuntimeError::NumericValidationMismatch {
        stage,
        message: format!(
            "checksum mismatch; actual={actual_checksum:#018x}, expected={expected_checksum:#018x}"
        ),
    })
}

#[cfg(target_os = "macos")]
fn validate_numeric_pair_checksum(
    stage: &'static str,
    actual_checksum: u64,
    expected_left: &[f32],
    expected_right: &[f32],
) -> Result<u64, MetalRuntimeError> {
    let expected_checksum = checksum_f32_pair(expected_left, expected_right);
    if actual_checksum == expected_checksum {
        return Ok(expected_checksum);
    }

    Err(MetalRuntimeError::NumericValidationMismatch {
        stage,
        message: format!(
            "checksum mismatch; actual={actual_checksum:#018x}, expected={expected_checksum:#018x}"
        ),
    })
}

#[cfg(target_os = "macos")]
fn validate_attention_output_against_reference(
    expected_values: &[f32],
    actual_bits: &[u32],
    actual_checksum: u64,
) -> Result<(u64, u32), MetalRuntimeError> {
    const ABS_TOLERANCE: f32 = 1.0e-4;
    const REL_TOLERANCE: f32 = 5.0e-4;

    let expected_checksum = checksum_f32_slice(expected_values);
    if actual_bits.len() != expected_values.len() {
        return Err(MetalRuntimeError::NumericValidationMismatch {
            stage: "attention_output",
            message: format!(
                "element count mismatch; actual={}, expected={}, actual_checksum={actual_checksum:#018x}, expected_checksum={expected_checksum:#018x}",
                actual_bits.len(),
                expected_values.len(),
            ),
        });
    }

    let mut max_abs_diff = 0.0_f32;
    for (index, (&actual_bits, &expected)) in actual_bits.iter().zip(expected_values).enumerate() {
        let actual = f32::from_bits(actual_bits);
        if !actual.is_finite() {
            return Err(MetalRuntimeError::NumericValidationMismatch {
                stage: "attention_output",
                message: format!("non-finite value at index {index}: actual={actual}"),
            });
        }

        let abs_diff = (actual - expected).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        let scale = actual.abs().max(expected.abs()).max(1.0);
        if abs_diff > ABS_TOLERANCE && abs_diff > scale * REL_TOLERANCE {
            return Err(MetalRuntimeError::NumericValidationMismatch {
                stage: "attention_output",
                message: format!(
                    "value drift at index {index}: actual={actual}, expected={expected}, abs_diff={abs_diff}, max_abs_diff={max_abs_diff}, actual_checksum={actual_checksum:#018x}, expected_checksum={expected_checksum:#018x}",
                ),
            });
        }
    }

    Ok((
        expected_checksum,
        (max_abs_diff * 1_000_000.0)
            .round()
            .clamp(0.0, u32::MAX as f32) as u32,
    ))
}

#[cfg(target_os = "macos")]
pub(super) fn capture_numeric_trace(
    workload: &MetalDispatchWorkload,
    arena: &MetalDispatchArena,
) -> MetalDispatchNumericTrace {
    let key_cache = read_shared_buffer_prefix(&arena.key_cache, workload.slot_numeric_capacity());
    let attention_output = read_shared_buffer_prefix(
        &arena.attention_output,
        workload.attention_numeric_elements(),
    );
    let gather_key =
        read_shared_buffer_prefix(&arena.kv_key_gathered, workload.gather_numeric_elements());
    let gather_value =
        read_shared_buffer_prefix(&arena.kv_value_gathered, workload.gather_numeric_elements());
    let copy_key =
        read_shared_buffer_prefix(&arena.copy_key_target, workload.slot_numeric_capacity());
    let copy_value =
        read_shared_buffer_prefix(&arena.copy_value_target, workload.slot_numeric_capacity());

    MetalDispatchNumericTrace {
        attention_output_bits: attention_output
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        key_cache_checksum: checksum_f32_slice(&key_cache),
        attention_output_checksum: checksum_f32_slice(&attention_output),
        gather_output_checksum: checksum_f32_pair(&gather_key, &gather_value),
        copy_output_checksum: checksum_f32_pair(&copy_key, &copy_value),
        validation: None,
    }
}

#[cfg(target_os = "macos")]
pub(super) fn capture_numeric_cache_snapshot(
    workload: &MetalDispatchWorkload,
    arena: &MetalDispatchArena,
) -> MetalDispatchKvCacheSnapshot {
    let mut key_cache =
        read_shared_buffer_prefix(&arena.key_cache, workload.slot_numeric_capacity());
    let mut value_cache =
        read_shared_buffer_prefix(&arena.value_cache, workload.slot_numeric_capacity());
    let copy_key =
        read_shared_buffer_prefix(&arena.copy_key_target, workload.slot_numeric_capacity());
    let copy_value =
        read_shared_buffer_prefix(&arena.copy_value_target, workload.slot_numeric_capacity());
    merge_copy_targets_into_cache_snapshot(
        workload,
        &mut key_cache,
        &mut value_cache,
        &copy_key,
        &copy_value,
    );

    MetalDispatchKvCacheSnapshot {
        key_cache,
        value_cache,
    }
}

pub(super) fn checksum_f32_slice(values: &[f32]) -> u64 {
    checksum_words(values.iter().map(|value| value.to_bits()))
}

pub(super) fn checksum_f32_pair(left: &[f32], right: &[f32]) -> u64 {
    checksum_words(
        left.iter()
            .map(|value| value.to_bits())
            .chain(right.iter().map(|value| value.to_bits())),
    )
}

pub(super) fn checksum_words(words: impl IntoIterator<Item = u32>) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    words.into_iter().fold(OFFSET_BASIS, |checksum, word| {
        checksum
            .wrapping_mul(PRIME)
            .wrapping_add(u64::from(word))
            .rotate_left(7)
    })
}
