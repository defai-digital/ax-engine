use super::*;

#[cfg(target_os = "macos")]
pub(super) fn build_model_stage_rope_tables(
    rotary_dim: usize,
    position: f32,
    freq_base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = rotary_dim / 2;
    let neg_log_base = -(freq_base.ln());
    let dim_inv = 2.0 / rotary_dim as f32;
    let mut cos_table = Vec::with_capacity(half_dim);
    let mut sin_table = Vec::with_capacity(half_dim);

    for pair_index in 0..half_dim {
        let freq = (neg_log_base * pair_index as f32 * dim_inv).exp();
        let theta = position * freq;
        let (sin_theta, cos_theta) = theta.sin_cos();
        cos_table.push(cos_theta);
        sin_table.push(sin_theta);
    }

    (cos_table, sin_table)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_split_half_rope_in_place(
    values: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let half_dim = cos_table.len();
    if values.len() != half_dim.saturating_mul(2) || sin_table.len() != half_dim {
        return;
    }

    let (low, high) = values.split_at_mut(half_dim);
    for index in 0..half_dim {
        let cos_theta = cos_table[index];
        let sin_theta = sin_table[index];
        let low_value = low[index];
        let high_value = high[index];
        low[index] = low_value * cos_theta - high_value * sin_theta;
        high[index] = low_value * sin_theta + high_value * cos_theta;
    }
}

#[cfg(target_os = "macos")]
pub(super) fn apply_interleaved_rope_in_place(
    values: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let half_dim = cos_table.len();
    if values.len() != half_dim.saturating_mul(2) || sin_table.len() != half_dim {
        return;
    }

    for index in 0..half_dim {
        let pair_base = index.saturating_mul(2);
        let cos_theta = cos_table[index];
        let sin_theta = sin_table[index];
        let even = values[pair_base];
        let odd = values[pair_base + 1];
        values[pair_base] = even * cos_theta - odd * sin_theta;
        values[pair_base + 1] = odd * cos_theta + even * sin_theta;
    }
}

#[cfg(target_os = "macos")]
pub(super) fn apply_rope_style_in_place(
    values: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    rope_style: ModelStageRopeStyle,
) {
    match rope_style {
        ModelStageRopeStyle::None => {}
        ModelStageRopeStyle::Neox => apply_split_half_rope_in_place(values, cos_table, sin_table),
        ModelStageRopeStyle::Interleaved => {
            apply_interleaved_rope_in_place(values, cos_table, sin_table)
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) fn apply_model_stage_rope_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    query: &[f32],
    key: &[f32],
    position: f32,
    stage_dims: ModelStageDims,
    rope_style: ModelStageRopeStyle,
    freq_base: f32,
    rotary_dim: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let bringup = bringup?;
    if !position.is_finite() || !freq_base.is_finite() || freq_base <= 0.0 {
        return None;
    }

    let head_dim = stage_dims.head_dim;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return None;
    }

    let query_len = stage_dims.q_heads.checked_mul(head_dim)?;
    let key_len = stage_dims.kv_heads.checked_mul(head_dim)?;
    if query.len() != query_len || key.len() != key_len {
        return None;
    }
    let pipeline_index = bringup.state.optional_kernel_dispatch_plan.apply_rope_f32?;
    let kernel_name = "apply_rope_f32";
    let feedback_key = rope_feedback_key(
        kernel_name,
        stage_dims.q_heads,
        stage_dims.kv_heads,
        head_dim,
        rotary_dim,
        rope_style,
    );
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }
    let (cos_table, sin_table) = build_model_stage_rope_tables(rotary_dim, position, freq_base);

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let query_buffer = new_shared_buffer_with_data(&bringup.state.device, query);
            let key_buffer = new_shared_buffer_with_data(&bringup.state.device, key);
            let cos_buffer = new_shared_buffer_with_data(&bringup.state.device, &cos_table);
            let sin_buffer = new_shared_buffer_with_data(&bringup.state.device, &sin_table);

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.apply_rope");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.apply_rope.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&query_buffer), 0);
            encoder.set_buffer(1, Some(&key_buffer), 0);
            encoder.set_buffer(2, Some(&cos_buffer), 0);
            encoder.set_buffer(3, Some(&sin_buffer), 0);
            set_model_stage_rope_dispatch_params(
                encoder,
                4,
                saturating_usize_to_u32(stage_dims.q_heads),
                saturating_usize_to_u32(stage_dims.kv_heads),
                saturating_usize_to_u32(head_dim),
                rope_style_dispatch_value(rope_style),
                saturating_usize_to_u32(rotary_dim),
            );
            let rope_threads = {
                let half_dim = rotary_dim / 2;
                let total = (stage_dims.q_heads + stage_dims.kv_heads) * half_dim;
                (total as u64).max(1)
            };
            encoder.dispatch_threads(MTLSize::new(rope_threads, 1, 1), MTLSize::new(32, 1, 1));

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let native_query =
                read_shared_buffer_prefix(&query_buffer, saturating_usize_to_u32(query_len));
            let native_key =
                read_shared_buffer_prefix(&key_buffer, saturating_usize_to_u32(key_len));
            (native_query.len() == query_len
                && native_key.len() == key_len
                && native_query.iter().all(|value| value.is_finite())
                && native_key.iter().all(|value| value.is_finite()))
            .then_some((native_query, native_key))
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
type BatchedModelStageRopeRows = (Vec<Vec<f32>>, Vec<Vec<f32>>);

#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity)]
pub(super) fn apply_batched_model_stage_rope_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    query_rows: &[Vec<f32>],
    key_rows: &[Vec<f32>],
    positions: &[u32],
    stage_dims: ModelStageDims,
    rope_style: ModelStageRopeStyle,
    freq_base: f32,
    rotary_dim: usize,
) -> Option<BatchedModelStageRopeRows> {
    let bringup = bringup?;
    if !freq_base.is_finite() || freq_base <= 0.0 {
        return None;
    }

    let head_dim = stage_dims.head_dim;
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
    {
        return None;
    }
    if query_rows.is_empty()
        || query_rows.len() != key_rows.len()
        || query_rows.len() != positions.len()
    {
        return None;
    }

    let token_count = query_rows.len();
    let query_len = stage_dims.q_heads.checked_mul(head_dim)?;
    let key_len = stage_dims.kv_heads.checked_mul(head_dim)?;
    if query_rows.iter().any(|row| row.len() != query_len)
        || key_rows.iter().any(|row| row.len() != key_len)
    {
        return None;
    }
    if token_count == 1 {
        let (native_query, native_key) = apply_model_stage_rope_with_optional_native_path(
            Some(bringup),
            query_rows.first()?,
            key_rows.first()?,
            *positions.first()? as f32,
            stage_dims,
            rope_style,
            freq_base,
            rotary_dim,
        )?;
        return Some((vec![native_query], vec![native_key]));
    }
    let pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .apply_rope_batched_f32?;
    let kernel_name = "apply_rope_batched_f32";
    let feedback_key = batched_rope_feedback_key(
        kernel_name,
        token_count,
        stage_dims.q_heads,
        stage_dims.kv_heads,
        head_dim,
        rotary_dim,
        rope_style,
    );
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }
    let mut flattened_query = Vec::with_capacity(token_count.checked_mul(query_len)?);
    let mut flattened_key = Vec::with_capacity(token_count.checked_mul(key_len)?);
    for row in query_rows {
        flattened_query.extend_from_slice(row);
    }
    for row in key_rows {
        flattened_key.extend_from_slice(row);
    }
    let positions = positions
        .iter()
        .map(|position| *position as f32)
        .collect::<Vec<_>>();

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let query_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_query);
            let key_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_key);
            let positions_buffer = new_shared_buffer_with_data(&bringup.state.device, &positions);

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.apply_rope_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.apply_rope_batched.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&query_buffer), 0);
            encoder.set_buffer(1, Some(&key_buffer), 0);
            encoder.set_buffer(2, Some(&positions_buffer), 0);
            set_batched_model_stage_rope_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(token_count),
                saturating_usize_to_u32(stage_dims.q_heads),
                saturating_usize_to_u32(stage_dims.kv_heads),
                saturating_usize_to_u32(head_dim),
                rope_style_dispatch_value(rope_style),
                freq_base,
                saturating_usize_to_u32(rotary_dim),
            );
            encoder.dispatch_threads(
                MTLSize::new(token_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(token_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let native_query = read_shared_buffer_prefix(
                &query_buffer,
                saturating_usize_to_u32(token_count.checked_mul(query_len)?),
            );
            let native_key = read_shared_buffer_prefix(
                &key_buffer,
                saturating_usize_to_u32(token_count.checked_mul(key_len)?),
            );
            if native_query.len() != token_count.checked_mul(query_len)?
                || native_key.len() != token_count.checked_mul(key_len)?
                || native_query.iter().any(|value| !value.is_finite())
                || native_key.iter().any(|value| !value.is_finite())
            {
                return None;
            }

            Some((
                native_query
                    .chunks_exact(query_len)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
                native_key
                    .chunks_exact(key_len)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
            ))
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
pub(super) fn rope_style_dispatch_value(rope_style: ModelStageRopeStyle) -> u32 {
    match rope_style {
        ModelStageRopeStyle::Neox => 0,
        ModelStageRopeStyle::Interleaved => 1,
        ModelStageRopeStyle::None => 2,
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity)]
pub(super) fn project_attention_qkv_with_dims_and_tally(
    artifacts: &NativeModelArtifacts,
    attention_qkv: &MetalAttentionQkvBindings,
    buffers: &MetalNativeModelBufferBindings,
    input: &[f32],
    stage_dims: ModelStageDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, PrefixAttentionExecutionTally)> {
    let query_output_dim = stage_dims.q_heads.saturating_mul(stage_dims.head_dim);
    let kv_output_dim = stage_dims.kv_head_size();

    match attention_qkv {
        MetalAttentionQkvBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let q_rows = artifacts.manifest().attention_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let packed_q_rows = if artifacts.manifest().attn_output_gate {
                q_rows.saturating_mul(2)
            } else {
                q_rows
            };
            let k_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let v_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            if q_rows < query_output_dim
                || packed_q_rows < q_rows
                || k_rows < kv_output_dim
                || v_rows < kv_output_dim
            {
                return None;
            }

            let (query, query_tally) = project_matrix_head_prefix_with_tally(
                packed,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input,
                bringup,
            )?;
            let (key, key_tally) = project_matrix_head_prefix_with_tally(
                packed,
                packed_q_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input,
                bringup,
            )?;
            let (value, value_tally) = project_matrix_head_prefix_with_tally(
                packed,
                packed_q_rows + k_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input,
                bringup,
            )?;
            Some((
                query,
                key,
                value,
                query_tally.merge(key_tally).merge(value_tally),
            ))
        }
        MetalAttentionQkvBindings::Split {
            q,
            k,
            v,
            value_from_key,
        } => {
            let q_binding = buffers.binding_for(q)?;
            let k_binding = buffers.binding_for(k)?;
            let q_source_head_dim = if artifacts.manifest().attn_output_gate {
                stage_dims.head_dim.checked_mul(2)?
            } else {
                stage_dims.head_dim
            };
            let (query, query_tally) = project_matrix_head_prefix_with_tally(
                q_binding,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                q_source_head_dim,
                input,
                bringup,
            )?;
            let (key, key_tally) = project_matrix_head_prefix_with_tally(
                k_binding,
                0,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                stage_dims.head_dim,
                input,
                bringup,
            )?;
            let (value, value_tally) = if *value_from_key {
                (key.clone(), PrefixAttentionExecutionTally::default())
            } else {
                let v_binding = buffers.binding_for(v.as_ref()?)?;
                project_matrix_head_prefix_with_tally(
                    v_binding,
                    0,
                    stage_dims.kv_heads,
                    stage_dims.head_dim,
                    stage_dims.head_dim,
                    input,
                    bringup,
                )?
            };
            Some((
                query,
                key,
                value,
                query_tally.merge(key_tally).merge(value_tally),
            ))
        }
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity)]
pub(super) fn project_batched_attention_qkv_with_dims_and_tally(
    artifacts: &NativeModelArtifacts,
    attention_qkv: &MetalAttentionQkvBindings,
    buffers: &MetalNativeModelBufferBindings,
    input_rows: &[Vec<f32>],
    input_width: usize,
    stage_dims: ModelStageDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    PrefixAttentionExecutionTally,
)> {
    let query_output_dim = stage_dims.q_heads.saturating_mul(stage_dims.head_dim);
    let kv_output_dim = stage_dims.kv_head_size();

    match attention_qkv {
        MetalAttentionQkvBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let q_rows = artifacts.manifest().attention_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let packed_q_rows = if artifacts.manifest().attn_output_gate {
                q_rows.saturating_mul(2)
            } else {
                q_rows
            };
            let k_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let v_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            if q_rows < query_output_dim
                || packed_q_rows < q_rows
                || k_rows < kv_output_dim
                || v_rows < kv_output_dim
            {
                return None;
            }

            let (query, query_tally) = project_batched_matrix_head_prefix_with_tally(
                packed,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input_rows,
                input_width,
                bringup,
            )?;
            let (key, key_tally) = project_batched_matrix_head_prefix_with_tally(
                packed,
                packed_q_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input_rows,
                input_width,
                bringup,
            )?;
            let (value, value_tally) = project_batched_matrix_head_prefix_with_tally(
                packed,
                packed_q_rows + k_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input_rows,
                input_width,
                bringup,
            )?;
            Some((
                query,
                key,
                value,
                query_tally.merge(key_tally).merge(value_tally),
            ))
        }
        MetalAttentionQkvBindings::Split {
            q,
            k,
            v,
            value_from_key,
        } => {
            let q_binding = buffers.binding_for(q)?;
            let k_binding = buffers.binding_for(k)?;
            let q_source_head_dim = if artifacts.manifest().attn_output_gate {
                stage_dims.head_dim.checked_mul(2)?
            } else {
                stage_dims.head_dim
            };
            let (query, query_tally) = project_batched_matrix_head_prefix_with_tally(
                q_binding,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                q_source_head_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            let (key, key_tally) = project_batched_matrix_head_prefix_with_tally(
                k_binding,
                0,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                stage_dims.head_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            let (value, value_tally) = if *value_from_key {
                (key.clone(), PrefixAttentionExecutionTally::default())
            } else {
                let v_binding = buffers.binding_for(v.as_ref()?)?;
                project_batched_matrix_head_prefix_with_tally(
                    v_binding,
                    0,
                    stage_dims.kv_heads,
                    stage_dims.head_dim,
                    stage_dims.head_dim,
                    input_rows,
                    input_width,
                    bringup,
                )?
            };
            Some((
                query,
                key,
                value,
                query_tally.merge(key_tally).merge(value_tally),
            ))
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) fn expand_grouped_kv_heads_cpu(
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Option<Vec<f32>> {
    if kv_heads == 0
        || q_heads == 0
        || head_dim == 0
        || q_heads < kv_heads
        || !q_heads.is_multiple_of(kv_heads)
        || values.len() != kv_heads.checked_mul(head_dim)?
    {
        return None;
    }

    if q_heads == kv_heads {
        return Some(values.to_vec());
    }

    let heads_per_kv = q_heads / kv_heads;
    let mut expanded = Vec::with_capacity(q_heads.checked_mul(head_dim)?);
    for kv_head in 0..kv_heads {
        let base = kv_head.checked_mul(head_dim)?;
        let end = base.checked_add(head_dim)?;
        let head = values.get(base..end)?;
        for _ in 0..heads_per_kv {
            expanded.extend_from_slice(head);
        }
    }
    Some(expanded)
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(super) fn expand_grouped_kv_heads_with_path(
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<f32>> {
    if let Some(expanded) = expand_grouped_kv_heads_with_optional_native_path(
        bringup, values, q_heads, kv_heads, head_dim,
    ) {
        return Some(expanded);
    }

    expand_grouped_kv_heads_cpu(values, q_heads, kv_heads, head_dim)
}

#[cfg(target_os = "macos")]
pub(super) fn expand_batched_grouped_kv_heads_with_path(
    rows: &[Vec<f32>],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<Vec<f32>>> {
    if rows.is_empty() {
        return Some(Vec::new());
    }
    let input_row_width = kv_heads.checked_mul(head_dim)?;
    if rows.iter().any(|row| row.len() != input_row_width) {
        return None;
    }
    if rows.len() == 1 {
        return expand_grouped_kv_heads_with_path(
            rows.first()?,
            q_heads,
            kv_heads,
            head_dim,
            bringup,
        )
        .map(|row| vec![row]);
    }

    if let Some(expanded) = expand_batched_grouped_kv_heads_with_optional_native_path(
        rows, q_heads, kv_heads, head_dim, bringup,
    ) {
        return Some(expanded);
    }
    if rows.len() > 1 && batched_grouped_kv_expand_split_retry_worthwhile(bringup) {
        let split_index = rows.len() / 2;
        let (left_rows, right_rows) = rows.split_at(split_index);
        let mut left_expanded = expand_batched_grouped_kv_heads_with_path(
            left_rows, q_heads, kv_heads, head_dim, bringup,
        )?;
        let right_expanded = expand_batched_grouped_kv_heads_with_path(
            right_rows, q_heads, kv_heads, head_dim, bringup,
        )?;
        left_expanded.extend(right_expanded);
        return Some(left_expanded);
    }

    rows.iter()
        .map(|row| expand_grouped_kv_heads_cpu(row, q_heads, kv_heads, head_dim))
        .collect()
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(super) fn expand_grouped_kv_heads_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if kv_heads == 0
        || q_heads == 0
        || head_dim == 0
        || q_heads < kv_heads
        || !q_heads.is_multiple_of(kv_heads)
        || values.len() != kv_heads.checked_mul(head_dim)?
    {
        return None;
    }

    if q_heads == kv_heads {
        return Some(values.to_vec());
    }

    let output_element_count = q_heads.checked_mul(head_dim)?;
    let heads_per_kv = q_heads / kv_heads;
    let pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .expand_grouped_kv_heads_f32?;
    let kernel_name = "expand_grouped_kv_heads_f32";
    let feedback_key = grouped_kv_expand_feedback_key(kernel_name, q_heads, kv_heads, head_dim);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.expand_grouped_kv_heads");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.expand_grouped_kv_heads.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            set_grouped_kv_expand_dispatch_params(
                encoder,
                2,
                saturating_usize_to_u32(output_element_count),
                saturating_usize_to_u32(kv_heads),
                saturating_usize_to_u32(heads_per_kv),
                saturating_usize_to_u32(head_dim),
            );
            encoder.dispatch_threads(
                MTLSize::new(output_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let expanded = read_shared_buffer_prefix(
                &output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            (expanded.len() == output_element_count
                && expanded.iter().all(|value| value.is_finite()))
            .then_some(expanded)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
pub(super) fn expand_batched_grouped_kv_heads_with_optional_native_path(
    rows: &[Vec<f32>],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    if rows.is_empty() || kv_heads == 0 || q_heads == 0 || head_dim == 0 {
        return None;
    }
    if q_heads < kv_heads || !q_heads.is_multiple_of(kv_heads) {
        return None;
    }
    let input_row_width = kv_heads.checked_mul(head_dim)?;
    if rows.iter().any(|row| row.len() != input_row_width) {
        return None;
    }

    if q_heads == kv_heads {
        return Some(rows.to_vec());
    }

    let token_count = rows.len();
    let flattened_input_len = token_count.checked_mul(input_row_width)?;
    let output_row_width = q_heads.checked_mul(head_dim)?;
    let output_element_count = token_count.checked_mul(output_row_width)?;
    let heads_per_kv = q_heads / kv_heads;
    let pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .expand_grouped_kv_heads_f32?;
    let kernel_name = "expand_grouped_kv_heads_f32";
    let feedback_key = batched_grouped_kv_expand_feedback_key(
        kernel_name,
        token_count,
        q_heads,
        kv_heads,
        head_dim,
    );
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let mut flattened_input = Vec::with_capacity(flattened_input_len);
    for row in rows {
        flattened_input.extend_from_slice(row);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_input);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.expand_grouped_kv_heads_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.expand_grouped_kv_heads_batched.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            set_grouped_kv_expand_dispatch_params(
                encoder,
                2,
                saturating_usize_to_u32(output_element_count),
                saturating_usize_to_u32(token_count.checked_mul(kv_heads)?),
                saturating_usize_to_u32(heads_per_kv),
                saturating_usize_to_u32(head_dim),
            );
            encoder.dispatch_threads(
                MTLSize::new(output_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let expanded = read_shared_buffer_prefix(
                &output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            if expanded.len() != output_element_count
                || expanded.iter().any(|value| !value.is_finite())
            {
                return None;
            }

            Some(
                expanded
                    .chunks_exact(output_row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

// Model accessor functions moved to model_helpers.rs

#[cfg(target_os = "macos")]
pub(super) fn apply_rms_norm_with_weights_in_place(
    values: &mut [f32],
    weights: &[f32],
    epsilon: f32,
    weight_offset: f32,
) -> Option<()> {
    if values.len() != weights.len() || values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0
    {
        return None;
    }

    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    if !mean_square.is_finite() {
        return None;
    }

    let denom = (mean_square + epsilon).sqrt();
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }

    for (value, weight) in values.iter_mut().zip(weights.iter().copied()) {
        let normalized = (*value / denom) * (weight + weight_offset);
        if !normalized.is_finite() {
            return None;
        }
        *value = normalized;
    }

    Some(())
}

#[cfg(target_os = "macos")]
pub(super) fn apply_rms_norm_without_weights_in_place(
    values: &mut [f32],
    epsilon: f32,
) -> Option<()> {
    if values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }

    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    if !mean_square.is_finite() {
        return None;
    }

    let denom = (mean_square + epsilon).sqrt();
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }

    for value in values.iter_mut() {
        let normalized = *value / denom;
        if !normalized.is_finite() {
            return None;
        }
        *value = normalized;
    }

    Some(())
}

#[cfg(target_os = "macos")]
pub(super) fn apply_rms_norm_without_weights_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    epsilon: f32,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .rms_norm_kernel(NativeTensorDataType::F32)?;
    let feedback_key = rms_norm_feedback_key(kernel_name, values.len());
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let zero_weights = vec![0.0_f32; values.len()];
            let weight_buffer = new_shared_buffer_with_data(&bringup.state.device, &zero_weights);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(values.len()),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.rms_norm_no_weight");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.rms_norm_no_weight.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_rms_norm_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(values.len()),
                epsilon,
                1.0,
            );
            encoder.dispatch_threads(
                MTLSize::new(NORM_TG_SIZE, 1, 1),
                MTLSize::new(NORM_TG_SIZE, 1, 1),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(values.len()));
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(super) fn apply_rms_norm_with_binding_in_place(
    values: &mut [f32],
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<()> {
    apply_rms_norm_with_binding_in_place_with_path(
        values,
        weight_binding,
        epsilon,
        weight_offset,
        bringup,
    )
    .map(|_| ())
}

#[cfg(target_os = "macos")]
pub(super) fn apply_rms_norm_with_binding_in_place_with_path(
    values: &mut [f32],
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    if let Some(output) = apply_rms_norm_with_optional_native_path(
        bringup,
        values,
        weight_binding,
        epsilon,
        weight_offset,
    ) {
        values.copy_from_slice(&output);
        return Some(true);
    }

    let weights = tensor_prefix_f32(weight_binding, values.len())?;
    apply_rms_norm_with_weights_in_place(values, &weights, epsilon, weight_offset)?;
    round_slice_to_native_dtype(values, weight_binding.native_dtype);
    Some(false)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_rms_norm_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    let (rms_norm_kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .rms_norm_kernel(weight_binding.native_dtype)?;
    if values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    let feedback_key = rms_norm_feedback_key(rms_norm_kernel_name, values.len());
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let weight_len = tensor_element_count(&weight_binding.meta.spec)?;
    if values.len() > weight_len {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        rms_norm_kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(values.len()),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.rms_norm");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.rms_norm.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_binding.native_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_rms_norm_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(values.len()),
                epsilon,
                weight_offset,
            );
            encoder.dispatch_threads(
                MTLSize::new(NORM_TG_SIZE, 1, 1),
                MTLSize::new(NORM_TG_SIZE, 1, 1),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(values.len()));
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(super) fn apply_per_head_rms_norm_with_binding_in_place(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<()> {
    apply_per_head_rms_norm_with_binding_in_place_with_tally(
        values,
        head_count,
        head_dim,
        weight_binding,
        epsilon,
        weight_offset,
        bringup,
    )
    .map(|_| ())
}

#[cfg(target_os = "macos")]
pub(super) fn apply_per_head_rms_norm_with_binding_in_place_with_tally(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if head_count == 0 || head_dim == 0 || values.len() != head_count.checked_mul(head_dim)? {
        return None;
    }

    if let Some(output) = apply_batched_per_head_rms_norm_with_optional_native_path(
        bringup,
        values,
        head_count,
        head_dim,
        weight_binding,
        epsilon,
        weight_offset,
    ) {
        values.copy_from_slice(&output);
        return Some(
            PrefixAttentionExecutionTally::default().record_rms_norm_elements(values.len(), true),
        );
    }

    let mut tally = PrefixAttentionExecutionTally::default();
    let single_native_bringup = single_rms_norm_retry_worthwhile(bringup, weight_binding, head_dim)
        .then_some(bringup)
        .flatten();
    for head in values.chunks_exact_mut(head_dim) {
        let used_native = apply_rms_norm_with_binding_in_place_with_path(
            head,
            weight_binding,
            epsilon,
            weight_offset,
            single_native_bringup,
        )?;
        tally = tally.record_rms_norm_elements(head.len(), used_native);
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_row_rms_norm_with_binding_in_place_with_tally(
    rows: &mut [Vec<f32>],
    row_width: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if row_width == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    if rows.is_empty() || rows.iter().any(|row| row.len() < row_width) {
        return None;
    }

    let single_native_bringup =
        single_rms_norm_retry_worthwhile(bringup, weight_binding, row_width)
            .then_some(bringup)
            .flatten();
    if rows.len() == 1 && single_native_bringup.is_some() {
        let row = rows.first_mut()?.get_mut(..row_width)?;
        let used_native = apply_rms_norm_with_binding_in_place_with_path(
            row,
            weight_binding,
            epsilon,
            weight_offset,
            single_native_bringup,
        )?;
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_rms_norm_elements(row_width, used_native),
        );
    }

    let row_count = rows.len();
    let flattened_element_count = row_count.checked_mul(row_width)?;
    if let Some(bringup) = bringup {
        let allow_batched_native =
            batched_rms_norm_feedback_binding(bringup, weight_binding, row_count, row_width)
                .is_some_and(|(_, _, feedback_key)| {
                    optional_kernel_allowed(bringup, &feedback_key)
                });
        if allow_batched_native {
            let mut flattened = Vec::with_capacity(flattened_element_count);
            for row in rows.iter() {
                flattened.extend_from_slice(row.get(..row_width)?);
            }
            if let Some(output) = apply_batched_per_head_rms_norm_with_optional_native_path(
                Some(bringup),
                &flattened,
                row_count,
                row_width,
                weight_binding,
                epsilon,
                weight_offset,
            ) {
                for (row, normalized) in rows.iter_mut().zip(output.chunks_exact(row_width)) {
                    row.get_mut(..row_width)?.copy_from_slice(normalized);
                }
                return Some(
                    PrefixAttentionExecutionTally::default()
                        .record_rms_norm_elements(flattened_element_count, true),
                );
            }
        }
    }
    if rows.len() > 1 && batched_rms_norm_split_retry_worthwhile(bringup, weight_binding) {
        let split_index = rows.len() / 2;
        let (left_rows, right_rows) = rows.split_at_mut(split_index);
        let left_tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
            left_rows,
            row_width,
            weight_binding,
            epsilon,
            weight_offset,
            bringup,
        )?;
        let right_tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
            right_rows,
            row_width,
            weight_binding,
            epsilon,
            weight_offset,
            bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }

    let mut tally = PrefixAttentionExecutionTally::default();
    for row in rows.iter_mut() {
        let used_native = apply_rms_norm_with_binding_in_place_with_path(
            row.get_mut(..row_width)?,
            weight_binding,
            epsilon,
            weight_offset,
            single_native_bringup,
        )?;
        tally = tally.record_rms_norm_elements(row_width, used_native);
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_per_head_rms_norm_rows_with_tally(
    rows: &mut [Vec<f32>],
    head_count: usize,
    head_dim: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if head_count == 0 || head_dim == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    let row_width = head_count.checked_mul(head_dim)?;
    if rows.is_empty() || rows.iter().any(|row| row.len() != row_width) {
        return None;
    }

    let total_heads = rows.len().checked_mul(head_count)?;
    let mut per_head_rows = Vec::with_capacity(total_heads);
    for row in rows.iter() {
        for head in row.chunks_exact(head_dim) {
            per_head_rows.push(head.to_vec());
        }
    }
    let tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
        &mut per_head_rows,
        head_dim,
        weight_binding,
        epsilon,
        weight_offset,
        bringup,
    )?;

    let mut normalized_heads = per_head_rows.into_iter();
    for row in rows.iter_mut() {
        for head in row.chunks_exact_mut(head_dim) {
            let normalized = normalized_heads.next()?;
            head.copy_from_slice(&normalized);
        }
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_per_head_rms_norm_rows_without_weights_with_tally(
    rows: &mut [Vec<f32>],
    head_count: usize,
    head_dim: usize,
    epsilon: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if head_count == 0 || head_dim == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    let row_width = head_count.checked_mul(head_dim)?;
    if rows.is_empty() || rows.iter().any(|row| row.len() != row_width) {
        return None;
    }

    let total_heads = rows.len().checked_mul(head_count)?;
    let mut per_head_rows = Vec::with_capacity(total_heads);
    for row in rows.iter() {
        for head in row.chunks_exact(head_dim) {
            per_head_rows.push(head.to_vec());
        }
    }

    let tally = apply_batched_row_rms_norm_without_weights_in_place_with_tally(
        &mut per_head_rows,
        head_dim,
        epsilon,
        bringup,
    )?;

    let mut normalized_heads = per_head_rows.into_iter();
    for row in rows.iter_mut() {
        for head in row.chunks_exact_mut(head_dim) {
            let normalized = normalized_heads.next()?;
            head.copy_from_slice(&normalized);
        }
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_row_rms_norm_without_weights_in_place_with_tally(
    rows: &mut [Vec<f32>],
    row_width: usize,
    epsilon: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if row_width == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    if rows.is_empty() || rows.iter().any(|row| row.len() < row_width) {
        return None;
    }

    let single_native_bringup =
        single_rms_norm_without_weights_retry_worthwhile(bringup, row_width)
            .then_some(bringup)
            .flatten();
    if rows.len() == 1 && single_native_bringup.is_some() {
        let row = rows.first_mut()?.get_mut(..row_width)?;
        let used_native = if let Some(output) =
            apply_rms_norm_without_weights_with_optional_native_path(
                single_native_bringup,
                row,
                epsilon,
            ) {
            row.copy_from_slice(&output);
            true
        } else {
            apply_rms_norm_without_weights_in_place(row, epsilon)?;
            false
        };
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_rms_norm_elements(row_width, used_native),
        );
    }

    let row_count = rows.len();
    let flattened_element_count = row_count.checked_mul(row_width)?;
    if let Some(bringup) = bringup {
        let (kernel_name, _) = bringup
            .state
            .optional_kernel_dispatch_plan
            .batched_rms_norm_kernel(NativeTensorDataType::F32)?;
        let feedback_key = batched_rms_norm_feedback_key(kernel_name, row_count, row_width);
        if optional_kernel_allowed(bringup, &feedback_key) {
            let mut flattened = Vec::with_capacity(flattened_element_count);
            for row in rows.iter() {
                flattened.extend_from_slice(row.get(..row_width)?);
            }
            if let Some(output) =
                apply_batched_per_head_rms_norm_without_weights_with_optional_native_path(
                    Some(bringup),
                    &flattened,
                    row_count,
                    row_width,
                    epsilon,
                )
            {
                for (row, normalized) in rows.iter_mut().zip(output.chunks_exact(row_width)) {
                    row.get_mut(..row_width)?.copy_from_slice(normalized);
                }
                return Some(
                    PrefixAttentionExecutionTally::default()
                        .record_rms_norm_elements(flattened_element_count, true),
                );
            }
        }
    }
    if rows.len() > 1 && batched_rms_norm_without_weights_split_retry_worthwhile(bringup) {
        let split_index = rows.len() / 2;
        let (left_rows, right_rows) = rows.split_at_mut(split_index);
        let left_tally = apply_batched_row_rms_norm_without_weights_in_place_with_tally(
            left_rows, row_width, epsilon, bringup,
        )?;
        let right_tally = apply_batched_row_rms_norm_without_weights_in_place_with_tally(
            right_rows, row_width, epsilon, bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }

    let mut tally = PrefixAttentionExecutionTally::default();
    for row in rows.iter_mut() {
        let used_native = if let Some(output) =
            apply_rms_norm_without_weights_with_optional_native_path(
                single_native_bringup,
                row.get(..row_width)?,
                epsilon,
            ) {
            row.get_mut(..row_width)?.copy_from_slice(&output);
            true
        } else {
            apply_rms_norm_without_weights_in_place(row.get_mut(..row_width)?, epsilon)?;
            false
        };
        tally = tally.record_rms_norm_elements(row_width, used_native);
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_per_head_rms_norm_without_weights_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    head_count: usize,
    head_dim: usize,
    epsilon: f32,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if head_count == 0
        || head_dim == 0
        || values.len() != head_count.checked_mul(head_dim)?
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_rms_norm_kernel(NativeTensorDataType::F32)?;
    let feedback_key = batched_rms_norm_feedback_key(kernel_name, head_count, head_dim);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let element_count = saturating_usize_to_u32(values.len());
    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let zero_weights = vec![0.0_f32; head_dim];
            let weight_buffer = new_shared_buffer_with_data(&bringup.state.device, &zero_weights);
            let output_buffer =
                new_zeroed_shared_buffer::<f32>(&bringup.state.device, element_count.max(1));

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.rms_norm_batched_no_weight");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.rms_norm_batched_no_weight.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_rms_norm_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(head_count),
                saturating_usize_to_u32(head_dim),
                epsilon,
                1.0,
            );
            encoder.dispatch_threads(
                MTLSize::new(
                    NORM_TG_SIZE.saturating_mul(u64::from(saturating_usize_to_u32(head_count))),
                    1,
                    1,
                ),
                MTLSize::new(NORM_TG_SIZE, 1, 1),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output = read_shared_buffer_prefix(&output_buffer, element_count);
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_per_head_rms_norm_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    head_count: usize,
    head_dim: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if head_count == 0
        || head_dim == 0
        || values.len() != head_count.checked_mul(head_dim)?
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return None;
    }

    let weight_len = tensor_element_count(&weight_binding.meta.spec)?;
    if head_dim > weight_len {
        return None;
    }

    let (kernel_name, pipeline_index, feedback_key) =
        batched_rms_norm_feedback_binding(bringup, weight_binding, head_count, head_dim)?;
    let element_count = saturating_usize_to_u32(values.len());
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let output_buffer =
                new_zeroed_shared_buffer::<f32>(&bringup.state.device, element_count.max(1));

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.rms_norm_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.rms_norm_batched.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_binding.native_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_rms_norm_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(head_count),
                saturating_usize_to_u32(head_dim),
                epsilon,
                weight_offset,
            );
            encoder.dispatch_threads(
                MTLSize::new(
                    NORM_TG_SIZE.saturating_mul(u64::from(saturating_usize_to_u32(head_count))),
                    1,
                    1,
                ),
                MTLSize::new(NORM_TG_SIZE, 1, 1),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output = read_shared_buffer_prefix(&output_buffer, element_count);
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ModelFfnActivation {
    Silu,
    GeluApprox,
}

#[cfg(target_os = "macos")]
pub(super) fn add_in_place(values: &mut [f32], delta: &[f32]) {
    for (value, addition) in values.iter_mut().zip(delta.iter()) {
        *value += *addition;
    }
}

#[cfg(target_os = "macos")]
pub(super) fn apply_vector_add_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    delta: &[f32],
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if values.is_empty() || values.len() != delta.len() {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .vector_add_kernel()?;
    let feedback_key = vector_add_feedback_key(kernel_name, values.len());
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let delta_buffer = new_shared_buffer_with_data(&bringup.state.device, delta);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(values.len()),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.vector_add");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.vector_add.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&delta_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_vector_add_dispatch_params(encoder, 3, saturating_usize_to_u32(values.len()));
            encoder.dispatch_threads(
                MTLSize::new(values.len().max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(values.len().max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(values.len()));
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_row_scale_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    rows: &[Vec<f32>],
    row_width: usize,
    row_scales: &[f32],
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    if row_width == 0 {
        return Some(vec![Vec::new(); rows.len()]);
    }
    if rows.is_empty()
        || rows.len() != row_scales.len()
        || rows.iter().any(|row| row.len() < row_width)
    {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .row_scale_kernel()?;
    let feedback_key = batched_row_scale_feedback_key(kernel_name, rows.len(), row_width);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let element_count = rows.len().checked_mul(row_width)?;
    let mut flattened_rows = Vec::with_capacity(element_count);
    for row in rows {
        flattened_rows.extend_from_slice(row.get(..row_width)?);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_rows);
            let scale_buffer = new_shared_buffer_with_data(&bringup.state.device, row_scales);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.row_scale");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.row_scale.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&scale_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_row_scale_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(rows.len()),
                saturating_usize_to_u32(row_width),
            );
            encoder.dispatch_threads(
                MTLSize::new(element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(element_count));
            if output.len() != element_count || output.iter().any(|value| !value.is_finite()) {
                return None;
            }
            Some(
                output
                    .chunks_exact(row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_row_vector_scale_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    rows: &[Vec<f32>],
    row_width: usize,
    scales: &[f32],
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    if row_width == 0 {
        return Some(vec![Vec::new(); rows.len()]);
    }
    if rows.is_empty() || scales.len() < row_width || rows.iter().any(|row| row.len() < row_width) {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .row_vector_scale_kernel()?;
    let feedback_key = batched_row_vector_scale_feedback_key(kernel_name, rows.len(), row_width);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let element_count = rows.len().checked_mul(row_width)?;
    let mut flattened_rows = Vec::with_capacity(element_count);
    for row in rows {
        flattened_rows.extend_from_slice(row.get(..row_width)?);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_rows);
            let scale_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &scales[..row_width]);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.row_vector_scale");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.row_vector_scale.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&scale_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_row_vector_scale_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(rows.len()),
                saturating_usize_to_u32(row_width),
            );
            encoder.dispatch_threads(
                MTLSize::new(element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(element_count));
            if output.len() != element_count || output.iter().any(|value| !value.is_finite()) {
                return None;
            }
            Some(
                output
                    .chunks_exact(row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_row_vector_scale_in_place_with_path(
    rows: &mut [Vec<f32>],
    row_width: usize,
    scales: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    if row_width == 0 {
        return Some(false);
    }
    if rows.is_empty() || scales.len() < row_width || rows.iter().any(|row| row.len() < row_width) {
        return None;
    }

    if let Some(output_rows) =
        apply_batched_row_vector_scale_with_optional_native_path(bringup, rows, row_width, scales)
    {
        for (row, output_row) in rows.iter_mut().zip(output_rows) {
            row.get_mut(..row_width)?.copy_from_slice(&output_row);
        }
        return Some(true);
    }

    for row in rows {
        for (value, scale) in row.get_mut(..row_width)?.iter_mut().zip(scales.iter()) {
            *value *= *scale;
        }
    }
    Some(false)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_row_scale_in_place_with_path(
    rows: &mut [Vec<f32>],
    row_width: usize,
    row_scales: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    if row_width == 0 {
        return Some(false);
    }
    if rows.is_empty()
        || rows.len() != row_scales.len()
        || rows.iter().any(|row| row.len() < row_width)
    {
        return None;
    }

    if let Some(output_rows) =
        apply_batched_row_scale_with_optional_native_path(bringup, rows, row_width, row_scales)
    {
        for (row, output_row) in rows.iter_mut().zip(output_rows) {
            row.get_mut(..row_width)?.copy_from_slice(&output_row);
        }
        return Some(true);
    }

    for (row, scale) in rows.iter_mut().zip(row_scales.iter().copied()) {
        for value in row.get_mut(..row_width)? {
            *value *= scale;
        }
    }
    Some(false)
}

#[cfg(target_os = "macos")]
pub(super) fn add_in_place_with_path_and_result_dtype(
    values: &mut [f32],
    delta: &[f32],
    result_dtype: NativeTensorDataType,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    if values.len() != delta.len() {
        return None;
    }
    if let Some(output) = apply_vector_add_with_optional_native_path(bringup, values, delta) {
        values.copy_from_slice(&output);
        return Some(true);
    }
    add_in_place(values, delta);
    round_slice_to_native_dtype(values, result_dtype);
    Some(false)
}

#[cfg(target_os = "macos")]
pub(super) fn add_batched_rows_in_place_with_direct_decode_tally_and_result_dtype(
    rows: &mut [Vec<f32>],
    delta_rows: &[Vec<f32>],
    row_width: usize,
    result_dtype: NativeTensorDataType,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<DirectDecodeNativeDenseTally> {
    Some(
        direct_decode_native_dense_tally_from_prefix_attention_tally(
            add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
                rows,
                delta_rows,
                row_width,
                result_dtype,
                bringup,
            )?,
        ),
    )
}

#[cfg(target_os = "macos")]
pub(super) fn add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
    rows: &mut [Vec<f32>],
    delta_rows: &[Vec<f32>],
    row_width: usize,
    result_dtype: NativeTensorDataType,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if row_width == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    if rows.len() != delta_rows.len()
        || rows.iter().any(|row| row.len() < row_width)
        || delta_rows.iter().any(|row| row.len() < row_width)
    {
        return None;
    }

    let single_native_bringup = single_vector_add_retry_worthwhile(bringup, row_width)
        .then_some(bringup)
        .flatten();
    if rows.len() == 1 && single_native_bringup.is_some() {
        let used_native = add_in_place_with_path_and_result_dtype(
            rows.first_mut()?.get_mut(..row_width)?,
            delta_rows.first()?.get(..row_width)?,
            result_dtype,
            single_native_bringup,
        )?;
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_residual_add_elements(row_width, used_native),
        );
    }

    let row_count = rows.len();
    let element_count = row_count.checked_mul(row_width)?;
    let mut flattened_rows = Vec::with_capacity(element_count);
    let mut flattened_deltas = Vec::with_capacity(element_count);
    for (row, delta_row) in rows.iter().zip(delta_rows.iter()) {
        flattened_rows.extend_from_slice(row.get(..row_width)?);
        flattened_deltas.extend_from_slice(delta_row.get(..row_width)?);
    }

    if let Some(output) =
        apply_vector_add_with_optional_native_path(bringup, &flattened_rows, &flattened_deltas)
    {
        for (row, output_row) in rows.iter_mut().zip(output.chunks_exact(row_width)) {
            row.get_mut(..row_width)?.copy_from_slice(output_row);
        }
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_residual_add_elements(element_count, true),
        );
    }
    if rows.len() > 1 && batched_vector_add_split_retry_worthwhile(bringup) {
        let split_index = rows.len() / 2;
        let (left_rows, right_rows) = rows.split_at_mut(split_index);
        let (left_delta_rows, right_delta_rows) = delta_rows.split_at(split_index);
        let left_tally = add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
            left_rows,
            left_delta_rows,
            row_width,
            result_dtype,
            bringup,
        )?;
        let right_tally = add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
            right_rows,
            right_delta_rows,
            row_width,
            result_dtype,
            bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }

    let mut tally = PrefixAttentionExecutionTally::default();
    for (row, delta_row) in rows.iter_mut().zip(delta_rows.iter()) {
        let used_native = add_in_place_with_path_and_result_dtype(
            row.get_mut(..row_width)?,
            delta_row.get(..row_width)?,
            result_dtype,
            single_native_bringup,
        )?;
        tally = tally.record_residual_add_elements(row_width, used_native);
    }
    Some(tally)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_model_gate_up_product(
    artifacts: &NativeModelArtifacts,
    gate: &mut [f32],
    up: &[f32],
) {
    let activation = native_model_ffn_activation(artifacts);
    for (gate_value, up_value) in gate.iter_mut().zip(up.iter()) {
        let activated = match activation {
            ModelFfnActivation::Silu => silu(*gate_value),
            ModelFfnActivation::GeluApprox => gelu_approx(*gate_value),
        };
        *gate_value = activated * *up_value;
    }
}

#[cfg(target_os = "macos")]
pub(super) fn apply_model_gate_up_product_with_path(
    artifacts: &NativeModelArtifacts,
    gate: &mut [f32],
    up: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    let activation = native_model_ffn_activation(artifacts);
    if let Some(output) =
        apply_model_gate_up_product_with_optional_native_path(bringup, activation, gate, up, None)
    {
        gate.copy_from_slice(&output);
        return Some(true);
    }

    apply_model_gate_up_product(artifacts, gate, up);
    Some(false)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_batched_model_gate_up_product_in_place_with_tally(
    artifacts: &NativeModelArtifacts,
    gate_rows: &mut [Vec<f32>],
    up_rows: &[Vec<f32>],
    row_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<DirectDecodeNativeDenseTally> {
    if row_width == 0 {
        return Some(DirectDecodeNativeDenseTally::default());
    }
    if gate_rows.len() != up_rows.len()
        || gate_rows.iter().any(|row| row.len() < row_width)
        || up_rows.iter().any(|row| row.len() < row_width)
    {
        return None;
    }

    let row_count = gate_rows.len();
    let flattened_len = row_count.checked_mul(row_width)?;
    let activation = native_model_ffn_activation(artifacts);
    let single_native_bringup =
        single_ffn_gate_product_retry_worthwhile(bringup, activation, row_width)
            .then_some(bringup)
            .flatten();
    if gate_rows.len() == 1 && single_native_bringup.is_some() {
        let used_native = apply_model_gate_up_product_with_path(
            artifacts,
            gate_rows.first_mut()?.get_mut(..row_width)?,
            up_rows.first()?.get(..row_width)?,
            single_native_bringup,
        )?;
        return Some(
            DirectDecodeNativeDenseTally::default()
                .record_ffn_activation_elements(row_width, used_native),
        );
    }

    if let Some(bringup) = bringup {
        let allow_batched_native =
            ffn_gate_product_feedback_binding(bringup, activation, row_count, row_width)
                .is_some_and(|(_, _, feedback_key)| {
                    optional_kernel_allowed(bringup, &feedback_key)
                });
        if allow_batched_native {
            let mut flattened_gate = Vec::with_capacity(flattened_len);
            let mut flattened_up = Vec::with_capacity(flattened_len);
            for (gate_row, up_row) in gate_rows.iter().zip(up_rows.iter()) {
                flattened_gate.extend_from_slice(gate_row.get(..row_width)?);
                flattened_up.extend_from_slice(up_row.get(..row_width)?);
            }

            if let Some(output) = apply_model_gate_up_product_with_optional_native_path(
                Some(bringup),
                activation,
                &flattened_gate,
                &flattened_up,
                Some((row_count, row_width)),
            ) {
                for (gate_row, output_row) in
                    gate_rows.iter_mut().zip(output.chunks_exact(row_width))
                {
                    gate_row.get_mut(..row_width)?.copy_from_slice(output_row);
                }
                return Some(
                    DirectDecodeNativeDenseTally::default()
                        .record_ffn_activation_elements(flattened_len, true),
                );
            }
        }
    }
    if gate_rows.len() > 1 && batched_ffn_gate_product_split_retry_worthwhile(bringup, activation) {
        let split_index = gate_rows.len() / 2;
        let (left_gate_rows, right_gate_rows) = gate_rows.split_at_mut(split_index);
        let (left_up_rows, right_up_rows) = up_rows.split_at(split_index);
        let left_tally = apply_batched_model_gate_up_product_in_place_with_tally(
            artifacts,
            left_gate_rows,
            left_up_rows,
            row_width,
            bringup,
        )?;
        let right_tally = apply_batched_model_gate_up_product_in_place_with_tally(
            artifacts,
            right_gate_rows,
            right_up_rows,
            row_width,
            bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }

    let mut tally = DirectDecodeNativeDenseTally::default();
    for (gate_row, up_row) in gate_rows.iter_mut().zip(up_rows.iter()) {
        let used_native = apply_model_gate_up_product_with_path(
            artifacts,
            gate_row.get_mut(..row_width)?,
            up_row.get(..row_width)?,
            single_native_bringup,
        )?;
        tally = tally.record_ffn_activation_elements(row_width, used_native);
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_model_gate_up_product_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    activation: ModelFfnActivation,
    gate: &[f32],
    up: &[f32],
    batched_shape: Option<(usize, usize)>,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if gate.is_empty() || gate.len() != up.len() {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .ffn_gate_product_kernel(activation)?;
    let feedback_key = batched_shape
        .map(|(row_count, row_width)| {
            batched_ffn_gate_product_feedback_key(kernel_name, row_count, row_width)
        })
        .unwrap_or_else(|| ffn_gate_product_feedback_key(kernel_name, gate.len()));
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let gate_buffer = new_shared_buffer_with_data(&bringup.state.device, gate);
            let up_buffer = new_shared_buffer_with_data(&bringup.state.device, up);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(gate.len()),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.ffn_gate_product");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.ffn_gate_product.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&gate_buffer), 0);
            encoder.set_buffer(1, Some(&up_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_ffn_gate_product_dispatch_params(encoder, 3, saturating_usize_to_u32(gate.len()));
            encoder.dispatch_threads(
                MTLSize::new(gate.len().max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(gate.len().max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(gate.len()));
            (output.len() == gate.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
pub(super) fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[cfg(target_os = "macos")]
pub(super) fn gelu_approx(value: f32) -> f32 {
    let cubic = value * value * value;
    let inner = (0.797_884_6_f32 * (value + 0.044_715_f32 * cubic)).tanh();
    0.5 * value * (1.0 + inner)
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(super) fn project_matrix_rows(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<f32>> {
    project_matrix_rows_with_path(binding, row_offset, output_dim, input, bringup)
        .map(|(output, _)| output)
}

#[cfg(target_os = "macos")]
pub(super) fn project_matrix_rows_with_path(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, bool)> {
    let (rows, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if row_offset.checked_add(output_dim)? > rows || input.len() > cols {
        return None;
    }

    if let Some(output) = project_matrix_rows_with_optional_native_path(
        bringup, binding, row_offset, output_dim, input,
    ) {
        return Some((output, true));
    }

    let mut output = Vec::with_capacity(output_dim);
    for row in row_offset..row_offset + output_dim {
        let weights = tensor_matrix_row_prefix_f32(binding, row, input.len())?;
        output.push(dot_product(&weights, input));
    }
    round_slice_to_native_dtype(&mut output, binding.native_dtype);
    Some((output, false))
}

#[cfg(target_os = "macos")]
pub(super) fn project_batched_matrix_rows_with_tally(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, DirectDecodeNativeDenseTally)> {
    if input_rows.is_empty() {
        return Some((Vec::new(), DirectDecodeNativeDenseTally::default()));
    }
    if input_rows.iter().any(|row| row.len() < input_width) {
        return None;
    }

    let single_native_bringup =
        single_projection_retry_worthwhile(bringup, binding, output_dim, input_width)
            .then_some(bringup)
            .flatten();
    if input_rows.len() == 1 && single_native_bringup.is_some() {
        let (projected, used_native) = project_matrix_rows_with_path(
            binding,
            row_offset,
            output_dim,
            input_rows.first()?.get(..input_width)?,
            single_native_bringup,
        )?;
        return Some((
            vec![projected],
            DirectDecodeNativeDenseTally::default().record_projection_rows(output_dim, used_native),
        ));
    }

    if let Some(output_rows) = project_batched_matrix_rows_with_optional_native_path(
        bringup,
        binding,
        row_offset,
        output_dim,
        input_rows,
        input_width,
    ) {
        return Some((
            output_rows,
            DirectDecodeNativeDenseTally::default()
                .record_projection_rows(input_rows.len().checked_mul(output_dim)?, true),
        ));
    }
    if input_rows.len() > 1 && batched_projection_split_retry_worthwhile(bringup, binding) {
        let split_index = input_rows.len() / 2;
        let (left_rows, right_rows) = input_rows.split_at(split_index);
        let (mut left_projected_rows, left_tally) = project_batched_matrix_rows_with_tally(
            binding,
            row_offset,
            output_dim,
            left_rows,
            input_width,
            bringup,
        )?;
        let (right_projected_rows, right_tally) = project_batched_matrix_rows_with_tally(
            binding,
            row_offset,
            output_dim,
            right_rows,
            input_width,
            bringup,
        )?;
        left_projected_rows.extend(right_projected_rows);
        return Some((left_projected_rows, left_tally.merge(right_tally)));
    }

    let mut projected_rows = Vec::with_capacity(input_rows.len());
    let mut tally = DirectDecodeNativeDenseTally::default();
    for row in input_rows {
        let (projected, used_native) = project_matrix_rows_with_path(
            binding,
            row_offset,
            output_dim,
            row.get(..input_width)?,
            single_native_bringup,
        )?;
        tally = tally.record_projection_rows(output_dim, used_native);
        projected_rows.push(projected);
    }
    Some((projected_rows, tally))
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(super) fn project_matrix_head_prefix(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    head_count: usize,
    projected_head_dim: usize,
    full_head_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<f32>> {
    project_matrix_head_prefix_with_tally(
        binding,
        row_offset,
        head_count,
        projected_head_dim,
        full_head_dim,
        input,
        bringup,
    )
    .map(|(output, _)| output)
}

#[cfg(target_os = "macos")]
pub(super) fn project_matrix_head_prefix_with_tally(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    head_count: usize,
    projected_head_dim: usize,
    full_head_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, PrefixAttentionExecutionTally)> {
    let (rows, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input.len() > cols || projected_head_dim > full_head_dim {
        return None;
    }

    if projected_head_dim == full_head_dim {
        let (projected, used_native) = project_matrix_rows_with_path(
            binding,
            row_offset,
            head_count.checked_mul(projected_head_dim)?,
            input,
            bringup,
        )?;
        let tally = PrefixAttentionExecutionTally::default()
            .record_projection_rows(head_count.checked_mul(projected_head_dim)?, used_native);
        return Some((projected, tally));
    }

    let mut output = Vec::with_capacity(head_count.checked_mul(projected_head_dim)?);
    let mut tally = PrefixAttentionExecutionTally::default();
    let single_native_bringup =
        single_projection_retry_worthwhile(bringup, binding, projected_head_dim, input.len())
            .then_some(bringup)
            .flatten();
    for head in 0..head_count {
        let head_row_offset = row_offset.checked_add(head.checked_mul(full_head_dim)?)?;
        if head_row_offset.checked_add(projected_head_dim)? > rows {
            return None;
        }

        let (projected, used_native) = project_matrix_rows_with_path(
            binding,
            head_row_offset,
            projected_head_dim,
            input,
            single_native_bringup,
        )?;
        tally = tally.record_projection_rows(projected_head_dim, used_native);
        output.extend(projected);
    }
    Some((output, tally))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) fn project_batched_matrix_head_prefix_with_tally(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    head_count: usize,
    projected_head_dim: usize,
    full_head_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    let (rows, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input_width > cols || projected_head_dim > full_head_dim {
        return None;
    }
    if input_rows.is_empty() || input_rows.iter().any(|row| row.len() < input_width) {
        return None;
    }

    if projected_head_dim == full_head_dim {
        let (projected, tally) = project_batched_matrix_rows_with_tally(
            binding,
            row_offset,
            head_count.checked_mul(projected_head_dim)?,
            input_rows,
            input_width,
            bringup,
        )?;
        return Some((
            projected,
            prefix_attention_tally_from_native_dense_tally(tally),
        ));
    }

    let row_count = input_rows.len();
    let output_width = head_count.checked_mul(projected_head_dim)?;

    // Build multi-projection tasks for all heads.
    let mut head_tasks = Vec::with_capacity(head_count);
    for head in 0..head_count {
        let head_row_offset = row_offset.checked_add(head.checked_mul(full_head_dim)?)?;
        if head_row_offset.checked_add(projected_head_dim)? > rows {
            return None;
        }
        head_tasks.push(MultiProjectionTask {
            binding,
            row_offset: head_row_offset,
            output_dim: projected_head_dim,
        });
    }

    // Try multi-projection batch: all heads in a single command buffer.
    if let Some(bringup) = bringup {
        let multi_result = if row_count == 1 {
            let input = input_rows.first()?.get(..input_width)?;
            project_multi_matrix_rows_with_optional_native_path(
                bringup,
                &head_tasks,
                input,
                input_width,
            )
            .map(|outputs| outputs.into_iter().map(|v| vec![v]).collect::<Vec<_>>())
        } else {
            project_multi_batched_matrix_rows_with_optional_native_path(
                bringup,
                &head_tasks,
                input_rows,
                input_width,
            )
        };

        if let Some(head_outputs) = multi_result {
            if head_outputs.len() == head_count {
                let total_rows = row_count.checked_mul(output_width)?;
                let tally = prefix_attention_tally_from_native_dense_tally(
                    DirectDecodeNativeDenseTally::default()
                        .record_projection_rows(total_rows, true),
                );
                let mut output_rows = vec![Vec::with_capacity(output_width); row_count];
                for head_rows in head_outputs {
                    for (output_row, head_row) in output_rows.iter_mut().zip(head_rows) {
                        output_row.extend(head_row);
                    }
                }
                return Some((output_rows, tally));
            }
        }
    }

    // Fallback: individual dispatches per head.
    let mut output_rows = vec![Vec::with_capacity(output_width); row_count];
    let mut tally = PrefixAttentionExecutionTally::default();
    for head in 0..head_count {
        let head_row_offset = row_offset.checked_add(head.checked_mul(full_head_dim)?)?;
        if head_row_offset.checked_add(projected_head_dim)? > rows {
            return None;
        }

        let (projected_rows, projected_tally) = project_batched_matrix_rows_with_tally(
            binding,
            head_row_offset,
            projected_head_dim,
            input_rows,
            input_width,
            bringup,
        )?;
        tally = tally.merge(prefix_attention_tally_from_native_dense_tally(
            projected_tally,
        ));
        for (output_row, projected_row) in output_rows.iter_mut().zip(projected_rows) {
            output_row.extend(projected_row);
        }
    }
    Some((output_rows, tally))
}

#[cfg(target_os = "macos")]
pub(super) fn project_batched_matrix_rows_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_projection_kernel(binding.native_dtype)?;

    let (_, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input_width > cols || output_dim == 0 || input_rows.is_empty() {
        return None;
    }

    let row_count = input_rows.len();
    let hidden_stride = input_width;
    let feedback_key = batched_projection_feedback_key(
        projection_kernel_name,
        row_count,
        output_dim,
        input_width,
        hidden_stride,
        cols,
    );
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let row_byte_offset = row_offset
        .checked_mul(cols)?
        .checked_mul(native_dtype_size_bytes(binding.native_dtype))?;
    let output_element_count = row_count.checked_mul(output_dim)?;
    let mut flattened_input = Vec::with_capacity(row_count.checked_mul(hidden_stride)?);
    for row in input_rows {
        flattened_input.extend_from_slice(row.get(..input_width)?);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()
    .and_then(|projection_pipeline| {
        autoreleasepool(|| {
            let hidden_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &flattened_input);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.project_matrix_rows_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.project_matrix_rows_batched.compute");

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(row_count),
                saturating_usize_to_u32(output_dim),
                saturating_usize_to_u32(cols),
                saturating_usize_to_u32(input_width),
                saturating_usize_to_u32(hidden_stride),
            );
            encoder.dispatch_threads(
                MTLSize::new(
                    projection_dispatch_threads(output_element_count.max(1)),
                    1,
                    1,
                ),
                MTLSize::new(PROJECTION_SIMD_WIDTH, 1, 1),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output = read_shared_buffer_prefix(
                &output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            if output.len() != output_element_count || output.iter().any(|value| !value.is_finite())
            {
                return None;
            }
            Some(
                output
                    .chunks_exact(output_dim)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

/// A single projection task within a multi-projection batch dispatch.
/// All tasks share the same input buffer but project through different weight matrices.
#[cfg(target_os = "macos")]
pub(super) struct MultiProjectionTask<'a> {
    pub(super) binding: &'a MetalNativeTensorBufferBinding,
    pub(super) row_offset: usize,
    pub(super) output_dim: usize,
}

/// Dispatches multiple independent matrix projections in a single Metal command buffer.
///
/// Each task reads from the same input row(s) but uses a different weight matrix and
/// produces an independent output. This eliminates per-projection command buffer
/// overhead when projections are data-independent (e.g. Q/K/V or FFN gate/up).
///
/// Returns `None` if any task cannot be dispatched natively (caller falls back to
/// individual dispatches). Follows the engine's feedback-key pattern: each task's
/// kernel is checked against the feedback state before attempting dispatch.
#[cfg(target_os = "macos")]
pub(super) fn project_multi_matrix_rows_with_optional_native_path(
    bringup: &MetalRuntimeBringup,
    tasks: &[MultiProjectionTask<'_>],
    input: &[f32],
    input_width: usize,
) -> Option<Vec<Vec<f32>>> {
    if tasks.is_empty() {
        return Some(Vec::new());
    }

    // All tasks must use the same dtype (same projection kernel).
    let first_dtype = tasks[0].binding.native_dtype;
    if tasks.iter().any(|t| t.binding.native_dtype != first_dtype) {
        return None;
    }

    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .projection_kernel(first_dtype)?;

    // Validate every task and check feedback before creating any Metal resources.
    for task in tasks {
        let (_, cols) = tensor_matrix_dimensions(&task.binding.meta.spec)?;
        if input_width > cols || task.output_dim == 0 {
            return None;
        }
        if task.row_offset.checked_add(task.output_dim)?
            > tensor_matrix_dimensions(&task.binding.meta.spec)?.0
        {
            return None;
        }
        let feedback_key =
            projection_feedback_key(projection_kernel_name, task.output_dim, input_width, cols);
        if !optional_kernel_allowed(bringup, &feedback_key) {
            return None;
        }
    }

    let projection_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()?;

    autoreleasepool(|| {
        let hidden_buffer = new_shared_buffer_with_data(&bringup.state.device, input);

        // Pre-allocate one output buffer per task.
        let output_buffers: Vec<_> = tasks
            .iter()
            .map(|task| {
                new_zeroed_shared_buffer::<f32>(
                    &bringup.state.device,
                    saturating_usize_to_u32(task.output_dim),
                )
            })
            .collect();

        let command_buffer = bringup.state.command_queue.new_command_buffer();
        command_buffer.set_label("ax.phase1.project_matrix_rows.multi");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("ax.phase1.project_matrix_rows.multi.compute");

        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let (_, cols) = tensor_matrix_dimensions(&task.binding.meta.spec)?;
            let row_byte_offset = if task.binding.native_dtype == NativeTensorDataType::Q4Km {
                q4km_row_byte_offset(task.row_offset, cols)?
            } else {
                task.row_offset
                    .checked_mul(cols)?
                    .checked_mul(native_dtype_size_bytes(task.binding.native_dtype))?
            };

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&task.binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(output_buffer), 0);
            if task.binding.native_dtype == NativeTensorDataType::Q4Km {
                set_q4km_projection_dispatch_params(
                    encoder,
                    3,
                    saturating_usize_to_u32(task.output_dim),
                    saturating_usize_to_u32(input_width),
                );
                let (tg_count, tg_size) = q4km_dispatch(task.output_dim.max(1));
                encoder.dispatch_thread_groups(tg_count, tg_size);
            } else {
                set_logits_projection_dispatch_params(
                    encoder,
                    3,
                    saturating_usize_to_u32(task.output_dim),
                    saturating_usize_to_u32(cols),
                    saturating_usize_to_u32(input_width),
                );
                encoder.dispatch_threads(
                    MTLSize::new(projection_dispatch_threads(task.output_dim.max(1)), 1, 1),
                    MTLSize::new(PROJECTION_SIMD_WIDTH, 1, 1),
                );
            }
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer_status(command_buffer.status());
        if status != MetalCommandBufferStatus::Completed {
            return None;
        }

        let mut outputs = Vec::with_capacity(tasks.len());
        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let output =
                read_shared_buffer_prefix(output_buffer, saturating_usize_to_u32(task.output_dim));
            if output.len() != task.output_dim || output.iter().any(|v| !v.is_finite()) {
                return None;
            }
            outputs.push(output);
        }

        Some(outputs)
    })
}

/// Batched variant of `project_multi_matrix_rows_with_optional_native_path` for
/// multiple input rows (e.g. prefill with N tokens). Each task projects all input
/// rows through its weight matrix in a single command buffer dispatch.
#[cfg(target_os = "macos")]
pub(super) fn project_multi_batched_matrix_rows_with_optional_native_path(
    bringup: &MetalRuntimeBringup,
    tasks: &[MultiProjectionTask<'_>],
    input_rows: &[Vec<f32>],
    input_width: usize,
) -> Option<Vec<Vec<Vec<f32>>>> {
    if tasks.is_empty() {
        return Some(Vec::new());
    }
    if input_rows.is_empty() {
        return Some(tasks.iter().map(|_| Vec::new()).collect());
    }

    let row_count = input_rows.len();
    let first_dtype = tasks[0].binding.native_dtype;
    if tasks.iter().any(|t| t.binding.native_dtype != first_dtype) {
        return None;
    }

    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_projection_kernel(first_dtype)?;

    for task in tasks {
        let (_, cols) = tensor_matrix_dimensions(&task.binding.meta.spec)?;
        if input_width > cols || task.output_dim == 0 {
            return None;
        }
        if task.row_offset.checked_add(task.output_dim)?
            > tensor_matrix_dimensions(&task.binding.meta.spec)?.0
        {
            return None;
        }
        let feedback_key = batched_projection_feedback_key(
            projection_kernel_name,
            row_count,
            task.output_dim,
            input_width,
            input_width,
            cols,
        );
        if !optional_kernel_allowed(bringup, &feedback_key) {
            return None;
        }
    }

    let projection_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()?;

    let hidden_stride = input_width;
    let mut flattened_input = Vec::with_capacity(row_count.checked_mul(hidden_stride)?);
    for row in input_rows {
        flattened_input.extend_from_slice(row.get(..input_width)?);
    }

    autoreleasepool(|| {
        let hidden_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_input);

        let output_buffers: Vec<_> = tasks
            .iter()
            .map(|task| {
                let element_count = row_count.checked_mul(task.output_dim).unwrap_or(0);
                new_zeroed_shared_buffer::<f32>(
                    &bringup.state.device,
                    saturating_usize_to_u32(element_count),
                )
            })
            .collect();

        let command_buffer = bringup.state.command_queue.new_command_buffer();
        command_buffer.set_label("ax.phase1.project_matrix_rows_batched.multi");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("ax.phase1.project_matrix_rows_batched.multi.compute");

        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let (_, cols) = tensor_matrix_dimensions(&task.binding.meta.spec)?;
            let row_byte_offset = task
                .row_offset
                .checked_mul(cols)?
                .checked_mul(native_dtype_size_bytes(task.binding.native_dtype))?;
            let output_element_count = row_count.checked_mul(task.output_dim)?;

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&task.binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(output_buffer), 0);
            set_batched_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(row_count),
                saturating_usize_to_u32(task.output_dim),
                saturating_usize_to_u32(cols),
                saturating_usize_to_u32(input_width),
                saturating_usize_to_u32(hidden_stride),
            );
            encoder.dispatch_threads(
                MTLSize::new(
                    projection_dispatch_threads(output_element_count.max(1)),
                    1,
                    1,
                ),
                MTLSize::new(PROJECTION_SIMD_WIDTH, 1, 1),
            );
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer_status(command_buffer.status());
        if status != MetalCommandBufferStatus::Completed {
            return None;
        }

        let mut outputs = Vec::with_capacity(tasks.len());
        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let output_element_count = row_count.checked_mul(task.output_dim)?;
            let output = read_shared_buffer_prefix(
                output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            if output.len() != output_element_count || output.iter().any(|v| !v.is_finite()) {
                return None;
            }
            let chunked: Vec<Vec<f32>> = output
                .chunks_exact(task.output_dim)
                .map(|chunk| chunk.to_vec())
                .collect();
            outputs.push(chunked);
        }

        Some(outputs)
    })
}

#[cfg(target_os = "macos")]
pub(super) fn project_matrix_rows_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input: &[f32],
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .projection_kernel(binding.native_dtype)?;

    let (_, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input.len() > cols {
        return None;
    }
    if output_dim == 0 {
        return Some(Vec::new());
    }
    let feedback_key =
        projection_feedback_key(projection_kernel_name, output_dim, input.len(), cols);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    // Compute byte offset for the row_offset-th row in the weight matrix.
    let row_byte_offset = if binding.native_dtype == NativeTensorDataType::Q4Km {
        q4km_row_byte_offset(row_offset, cols)?
    } else {
        row_offset
            .checked_mul(cols)?
            .checked_mul(native_dtype_size_bytes(binding.native_dtype))?
    };

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()
    .and_then(|projection_pipeline| {
        autoreleasepool(|| {
            let hidden_buffer = new_shared_buffer_with_data(&bringup.state.device, input);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_dim),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.project_matrix_rows");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.project_matrix_rows.compute");

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(&output_buffer), 0);

            if binding.native_dtype == NativeTensorDataType::Q4Km {
                set_q4km_projection_dispatch_params(
                    encoder,
                    3,
                    saturating_usize_to_u32(output_dim),
                    saturating_usize_to_u32(input.len()),
                );
                let (tg_count, tg_size) = q4km_dispatch(output_dim.max(1));
                encoder.dispatch_thread_groups(tg_count, tg_size);
            } else {
                set_logits_projection_dispatch_params(
                    encoder,
                    3,
                    saturating_usize_to_u32(output_dim),
                    saturating_usize_to_u32(cols),
                    saturating_usize_to_u32(input.len()),
                );
                encoder.dispatch_threads(
                    MTLSize::new(projection_dispatch_threads(output_dim.max(1)), 1, 1),
                    MTLSize::new(PROJECTION_SIMD_WIDTH, 1, 1),
                );
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(output_dim));
            (output.len() == output_dim && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}
