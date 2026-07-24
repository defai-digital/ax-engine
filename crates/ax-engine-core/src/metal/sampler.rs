use super::*;

#[derive(Debug)]
pub struct MetalBringupSampler {
    bringup: MetalRuntimeBringup,
    // Pre-allocated fixed-size output buffers reused across sample calls (buffer(1) and buffer(2)
    // in sample_argmax_logprob_f32: argmax output u32, logprob output f32).
    #[cfg(target_os = "macos")]
    argmax_out: Mutex<Buffer>,
    #[cfg(target_os = "macos")]
    logprob_out: Mutex<Buffer>,
}

impl MetalBringupSampler {
    pub fn from_build_dir(path: impl AsRef<Path>) -> Result<Self, MetalRuntimeError> {
        Self::from_bringup(MetalRuntimeBringup::from_build_dir(path)?)
    }

    pub fn from_assets(assets: MetalKernelAssets) -> Result<Self, MetalRuntimeError> {
        Self::from_bringup(MetalRuntimeBringup::from_assets(assets)?)
    }

    fn from_bringup(bringup: MetalRuntimeBringup) -> Result<Self, MetalRuntimeError> {
        #[cfg(target_os = "macos")]
        let (argmax_out, logprob_out) = {
            let device = &bringup.state.device;
            (
                Mutex::new(new_zeroed_shared_buffer::<u32>(device, 1)),
                Mutex::new(new_zeroed_shared_buffer::<f32>(device, 1)),
            )
        };
        Ok(Self {
            bringup,
            #[cfg(target_os = "macos")]
            argmax_out,
            #[cfg(target_os = "macos")]
            logprob_out,
        })
    }

    #[cfg(target_os = "macos")]
    fn sample_argmax_logprob_native_available(&self) -> bool {
        self.bringup
            .state
            .optional_kernel_dispatch_plan
            .sample_argmax_logprob_f32
            .is_some()
    }

    #[cfg(target_os = "macos")]
    fn sample_argmax_logprob_with_optional_native_path(
        &self,
        logits: &[f32],
    ) -> Option<(u32, f32)> {
        let kernel_name = "sample_argmax_logprob_f32";
        if logits.is_empty() {
            return None;
        }
        let pipeline_index = self
            .bringup
            .state
            .optional_kernel_dispatch_plan
            .sample_argmax_logprob_f32?;

        let feedback_key = logits_argmax_feedback_key(kernel_name, logits.len());
        let argmax_guard = self.argmax_out.lock().ok()?;
        let logprob_guard = self.logprob_out.lock().ok()?;

        let output = find_optional_pipeline_handle_by_index(
            &self.bringup.state,
            &self.bringup.metallib.path,
            kernel_name,
            pipeline_index,
        )
        .ok()
        .and_then(|pipeline| {
            autoreleasepool(|| {
                let logits_buffer = new_shared_buffer_with_data(&self.bringup.state.device, logits);

                let command_buffer = self.bringup.state.command_queue.new_command_buffer();
                command_buffer.set_label("ax.phase1.sample_argmax_logprob");
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_label("ax.phase1.sample_argmax_logprob.compute");

                encoder.set_compute_pipeline_state(&pipeline.pipeline);
                encoder.set_buffer(0, Some(&logits_buffer), 0);
                encoder.set_buffer(1, Some(&*argmax_guard), 0);
                encoder.set_buffer(2, Some(&*logprob_guard), 0);
                set_logits_argmax_dispatch_params(
                    encoder,
                    3,
                    saturating_usize_to_u32(logits.len()),
                );
                encoder.dispatch_threads(
                    MTLSize::new(ARGMAX_TG_SIZE, 1, 1),
                    MTLSize::new(ARGMAX_TG_SIZE, 1, 1),
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                let command_buffer_status = command_buffer_status(command_buffer.status());
                if command_buffer_status != MetalCommandBufferStatus::Completed {
                    return None;
                }

                let token_id = read_shared_u32_buffer_prefix(&argmax_guard, 1)
                    .into_iter()
                    .next()?;
                let logprob = read_shared_buffer_prefix(&logprob_guard, 1)
                    .into_iter()
                    .next()?;
                logprob.is_finite().then_some((token_id, logprob))
            })
        });
        record_optional_kernel_result(&self.bringup, &feedback_key, output.is_some());
        output
    }

    #[cfg(target_os = "macos")]
    fn sample_argmax_logprob_batched_with_optional_native_path(
        &self,
        logits_rows: &[&[f32]],
    ) -> Option<Vec<(u32, f32)>> {
        let kernel_name = "sample_argmax_logprob_batched_f32";
        if logits_rows.is_empty() {
            return Some(Vec::new());
        }
        let pipeline_index = self
            .bringup
            .state
            .optional_kernel_dispatch_plan
            .sample_argmax_logprob_batched_f32?;
        let vocab_rows = logits_rows.first()?.len();
        if vocab_rows == 0 || logits_rows.iter().any(|row| row.len() != vocab_rows) {
            return None;
        }

        let token_count = logits_rows.len();
        let feedback_key = batched_sampler_feedback_key(kernel_name, token_count, vocab_rows);
        if !optional_kernel_allowed(&self.bringup, &feedback_key) {
            return None;
        }
        let logits_element_count = token_count.checked_mul(vocab_rows)?;
        let mut flattened_logits = Vec::with_capacity(logits_element_count);
        for row in logits_rows {
            flattened_logits.extend_from_slice(row);
        }

        let output = find_optional_pipeline_handle_by_index(
            &self.bringup.state,
            &self.bringup.metallib.path,
            kernel_name,
            pipeline_index,
        )
        .ok()
        .and_then(|pipeline| {
            autoreleasepool(|| {
                let logits_buffer =
                    new_shared_buffer_with_data(&self.bringup.state.device, &flattened_logits);
                let argmax_buffer = new_zeroed_shared_buffer::<u32>(
                    &self.bringup.state.device,
                    saturating_usize_to_u32(token_count),
                );
                let logprob_buffer = new_zeroed_shared_buffer::<f32>(
                    &self.bringup.state.device,
                    saturating_usize_to_u32(token_count),
                );

                let command_buffer = self.bringup.state.command_queue.new_command_buffer();
                command_buffer.set_label("ax.phase1.sample_argmax_logprob_batched");
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_label("ax.phase1.sample_argmax_logprob_batched.compute");

                encoder.set_compute_pipeline_state(&pipeline.pipeline);
                encoder.set_buffer(0, Some(&logits_buffer), 0);
                encoder.set_buffer(1, Some(&argmax_buffer), 0);
                encoder.set_buffer(2, Some(&logprob_buffer), 0);
                set_batched_logits_argmax_dispatch_params(
                    encoder,
                    3,
                    saturating_usize_to_u32(token_count),
                    saturating_usize_to_u32(vocab_rows),
                );
                encoder.dispatch_threads(
                    MTLSize::new(
                        ARGMAX_TG_SIZE
                            .saturating_mul(u64::from(saturating_usize_to_u32(token_count))),
                        1,
                        1,
                    ),
                    MTLSize::new(ARGMAX_TG_SIZE, 1, 1),
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                let command_buffer_status = command_buffer_status(command_buffer.status());
                if command_buffer_status != MetalCommandBufferStatus::Completed {
                    return None;
                }

                let token_ids = read_shared_u32_buffer_prefix(
                    &argmax_buffer,
                    saturating_usize_to_u32(token_count),
                );
                let logprobs = read_shared_buffer_prefix(
                    &logprob_buffer,
                    saturating_usize_to_u32(token_count),
                );
                if token_ids.len() != token_count
                    || logprobs.len() != token_count
                    || logprobs.iter().any(|value| !value.is_finite())
                {
                    return None;
                }

                Some(token_ids.into_iter().zip(logprobs).collect())
            })
        });
        record_optional_kernel_result(&self.bringup, &feedback_key, output.is_some());
        output
    }
}

impl TokenSampler for MetalBringupSampler {
    fn sample(&self, input: SamplerInput) -> Vec<SampledToken> {
        let requests = input.requests;
        #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
        let mut sampled_from_logits = vec![None; requests.len()];

        #[cfg(target_os = "macos")]
        {
            for (logits_width, indices) in
                grouped_sampler_request_indices_by_logits_width(&requests)
            {
                let allow_single_native = self.sample_argmax_logprob_native_available();
                if let Some(group_results) = collect_grouped_sampler_results_with_item_fallback(
                    &indices,
                    &mut |group_indices| {
                        if group_indices.len() < 2 {
                            return None;
                        }
                        let feedback_key =
                            sampler_batched_group_feedback_key(group_indices.len(), logits_width);
                        if !optional_kernel_allowed(&self.bringup, &feedback_key) {
                            return None;
                        }
                        let logits_rows = group_indices
                            .iter()
                            .map(|index| {
                                requests
                                    .get(*index)
                                    .and_then(|request| request.logits.as_deref())
                            })
                            .collect::<Option<Vec<_>>>()?;
                        let output = self
                            .sample_argmax_logprob_batched_with_optional_native_path(&logits_rows);
                        let (output, success) =
                            validate_batched_sampler_group_output(output, group_indices.len());
                        record_optional_kernel_result(&self.bringup, &feedback_key, success);
                        output
                    },
                    &mut |request_index| {
                        requests.get(request_index).and_then(|request| {
                            request.logits.as_ref().and_then(|logits| {
                                allow_single_native
                                    .then(|| {
                                        self.sample_argmax_logprob_with_optional_native_path(logits)
                                    })
                                    .flatten()
                                    .or_else(|| sample_argmax_with_logprob(logits))
                            })
                        })
                    },
                ) {
                    for (request_index, result) in group_results {
                        sampled_from_logits[request_index] = Some(result);
                    }
                }
            }
        }

        requests
            .into_iter()
            .enumerate()
            .map(|(index, request)| {
                let sampled_from_logits = sampled_from_logits[index].or_else(|| {
                    request.logits.as_ref().and_then(|logits| {
                        #[cfg(target_os = "macos")]
                        {
                            self.sample_argmax_logprob_with_optional_native_path(logits)
                                .or_else(|| sample_argmax_with_logprob(logits))
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            sample_argmax_with_logprob(logits)
                        }
                    })
                });
                let invalid_logits = request.logits.is_some() && sampled_from_logits.is_none();
                let token_id = if invalid_logits {
                    0
                } else {
                    sampled_from_logits
                        .map(|(token_id, _)| token_id)
                        .unwrap_or_else(|| request.previous_token.saturating_add(1))
                };
                let logprob = sampled_from_logits
                    .map(|(_, logprob)| logprob)
                    .or_else(|| (!invalid_logits).then_some(0.0));
                let stop_reason = if invalid_logits {
                    Some(StopReason::Error)
                } else if request.generated_len.saturating_add(1) >= request.max_output_tokens {
                    Some(StopReason::MaxOutputTokens)
                } else {
                    None
                };

                SampledToken {
                    request_id: request.request_id,
                    token_id,
                    stop_reason,
                    logprob,
                }
            })
            .collect()
    }
}

pub(super) fn grouped_sampler_request_indices_by_logits_width(
    requests: &[SamplerRequest],
) -> BTreeMap<usize, Vec<usize>> {
    let mut grouped_request_indices = BTreeMap::<usize, Vec<usize>>::new();
    for (index, request) in requests.iter().enumerate() {
        if let Some(logits) = request.logits.as_ref() {
            grouped_request_indices
                .entry(logits.len())
                .or_default()
                .push(index);
        }
    }
    grouped_request_indices
}

#[cfg(target_os = "macos")]
pub(super) fn collect_grouped_sampler_results_with_item_fallback<T>(
    indices: &[usize],
    process_group: &mut impl FnMut(&[usize]) -> Option<Vec<T>>,
    process_item: &mut impl FnMut(usize) -> Option<T>,
) -> Option<Vec<(usize, T)>> {
    if indices.is_empty() {
        return Some(Vec::new());
    }
    if indices.len() > 1 {
        if let Some(results) = process_group(indices) {
            if results.len() != indices.len() {
                return None;
            }
            return Some(indices.iter().copied().zip(results).collect());
        }
        let split_index = indices.len() / 2;
        let mut left_results = collect_grouped_sampler_results_with_item_fallback(
            &indices[..split_index],
            process_group,
            process_item,
        )?;
        let right_results = collect_grouped_sampler_results_with_item_fallback(
            &indices[split_index..],
            process_group,
            process_item,
        )?;
        left_results.extend(right_results);
        return Some(left_results);
    }

    let request_index = *indices.first()?;
    Some(vec![(request_index, process_item(request_index)?)])
}

#[cfg(target_os = "macos")]
pub(super) fn ffn_gate_product_feedback_binding(
    bringup: &MetalRuntimeBringup,
    activation: ModelFfnActivation,
    row_count: usize,
    row_width: usize,
) -> Option<(&'static str, usize, MetalOptionalKernelFeedbackKey)> {
    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .ffn_gate_product_kernel(activation)?;
    Some((
        kernel_name,
        pipeline_index,
        batched_ffn_gate_product_feedback_key(kernel_name, row_count, row_width),
    ))
}

#[cfg(target_os = "macos")]
pub(super) fn validate_batched_sampler_group_output<T>(
    output: Option<Vec<T>>,
    expected_len: usize,
) -> (Option<Vec<T>>, bool) {
    match output {
        Some(results) if results.len() == expected_len => (Some(results), true),
        Some(_) | None => (None, false),
    }
}

#[cfg(target_os = "macos")]
pub(super) fn validate_model_bound_direct_decode_group_output(
    output: Option<ModelBoundDirectDecodeResult>,
    expected_request_ids: &[crate::ids::RequestId],
) -> (Option<ModelBoundDirectDecodeResult>, bool) {
    let expected_ids = expected_request_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let Some(result) = output else {
        return (None, false);
    };
    let token_ids = result
        .tokens
        .iter()
        .map(|(request_id, _)| *request_id)
        .collect::<BTreeSet<_>>();
    let logits_request_ids = result
        .logits_outputs
        .iter()
        .map(|output| output.request_id)
        .collect::<BTreeSet<_>>();
    let has_complete_logits_payload = result.logits_outputs.is_empty()
        || (result.logits_outputs.len() == expected_request_ids.len()
            && logits_request_ids == expected_ids);
    let success = result.tokens.len() == expected_request_ids.len()
        && token_ids == expected_ids
        && has_complete_logits_payload;
    if success {
        (Some(result), true)
    } else {
        (None, false)
    }
}
