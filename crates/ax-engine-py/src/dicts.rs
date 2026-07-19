use ax_engine_sdk::{
    CapabilityReport, EngineStepReport, GenerateResponse, GenerateRouteReport,
    GenerateStreamEvent as SdkGenerateStreamEvent, HostReport, MetalDispatchKernelStepReport,
    MetalDispatchNumericStepReport, MetalDispatchStepReport, MetalDispatchValidationStepReport,
    MetalToolchainReport, NativeRuntimeStatus, NativeSourceQuantization, RuntimeReport,
    SessionRequestReport, ToolStatusReport,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

pub(crate) fn runtime_dict<'py>(py: Python<'py>, runtime: &RuntimeReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("selected_backend", enum_label(py, runtime.selected_backend))
        .expect("selected_backend should serialize");
    dict.set_item("support_tier", enum_label(py, runtime.support_tier))
        .expect("support_tier should serialize");
    dict.set_item(
        "resolution_policy",
        enum_label(py, runtime.resolution_policy),
    )
    .expect("resolution_policy should serialize");
    dict.set_item("capabilities", capability_dict(py, &runtime.capabilities))
        .expect("capabilities should serialize");
    if let Some(fallback_reason) = runtime.fallback_reason.as_deref() {
        dict.set_item("fallback_reason", fallback_reason)
            .expect("fallback_reason should serialize");
    }
    dict.set_item("host", host_dict(py, &runtime.host))
        .expect("host should serialize");
    dict.set_item(
        "metal_toolchain",
        metal_toolchain_dict(py, &runtime.metal_toolchain),
    )
    .expect("metal_toolchain should serialize");
    if let Some(native_runtime) = runtime.mlx_runtime.as_ref() {
        dict.set_item("mlx_runtime", mlx_runtime_dict(py, native_runtime))
            .expect("mlx_runtime should serialize");
    }
    if let Some(native_model) = runtime.mlx_model.as_ref() {
        dict.set_item("mlx_model", native_model_dict(py, native_model))
            .expect("mlx_model should serialize");
    }
    dict.unbind()
}

fn mlx_runtime_dict<'py>(
    py: Python<'py>,
    mlx_runtime: &ax_engine_sdk::NativeRuntimeReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("runner", enum_label(py, mlx_runtime.runner))
        .expect("mlx_runtime.runner should serialize");
    if let Some(source) = mlx_runtime.artifacts_source {
        dict.set_item("artifacts_source", enum_label(py, source))
            .expect("mlx_runtime.artifacts_source should serialize");
    }
    dict.unbind()
}

fn native_model_dict<'py>(
    py: Python<'py>,
    native_model: &ax_engine_sdk::NativeModelReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item(
        "artifacts_source",
        enum_label(py, native_model.artifacts_source),
    )
    .expect("native_model.artifacts_source should serialize");
    dict.set_item("model_family", &native_model.model_family)
        .expect("native_model.model_family should serialize");
    dict.set_item("tensor_format", enum_label(py, native_model.tensor_format))
        .expect("native_model.tensor_format should serialize");
    if let Some(source_quantization) = native_model.source_quantization.as_ref() {
        dict.set_item(
            "source_quantization",
            native_source_quantization_dict(py, source_quantization),
        )
        .expect("native_model.source_quantization should serialize");
    }
    dict.set_item(
        "runtime_status",
        native_runtime_status_dict(py, &native_model.runtime_status),
    )
    .expect("native_model.runtime_status should serialize");
    dict.set_item("layer_count", native_model.layer_count)
        .expect("native_model.layer_count should serialize");
    dict.set_item("tensor_count", native_model.tensor_count)
        .expect("native_model.tensor_count should serialize");
    dict.set_item("tie_word_embeddings", native_model.tie_word_embeddings)
        .expect("native_model.tie_word_embeddings should serialize");
    dict.set_item("is_moe", native_model.is_moe)
        .expect("native_model.is_moe should serialize");
    dict.set_item("is_hybrid_attention", native_model.is_hybrid_attention)
        .expect("native_model.is_hybrid_attention should serialize");
    if let Some(interval) = native_model.hybrid_full_attention_interval {
        dict.set_item("hybrid_full_attention_interval", interval)
            .expect("native_model.hybrid_full_attention_interval should serialize");
    }
    if let Some(dim) = native_model.mla_kv_latent_dim {
        dict.set_item("mla_kv_latent_dim", dim)
            .expect("native_model.mla_kv_latent_dim should serialize");
    }
    if let Some(experts) = native_model.moe_active_experts {
        dict.set_item("moe_active_experts", experts)
            .expect("native_model.moe_active_experts should serialize");
    }
    dict.set_item("bindings_prepared", native_model.bindings_prepared)
        .expect("native_model.bindings_prepared should serialize");
    dict.set_item("buffers_bound", native_model.buffers_bound)
        .expect("native_model.buffers_bound should serialize");
    dict.set_item("buffer_count", native_model.buffer_count)
        .expect("native_model.buffer_count should serialize");
    dict.set_item("buffer_bytes", native_model.buffer_bytes)
        .expect("native_model.buffer_bytes should serialize");
    dict.set_item(
        "source_quantized_binding_count",
        native_model.source_quantized_binding_count,
    )
    .expect("native_model.source_quantized_binding_count should serialize");
    dict.set_item(
        "source_q4_k_binding_count",
        native_model.source_q4_k_binding_count,
    )
    .expect("native_model.source_q4_k_binding_count should serialize");
    dict.set_item(
        "source_q5_k_binding_count",
        native_model.source_q5_k_binding_count,
    )
    .expect("native_model.source_q5_k_binding_count should serialize");
    dict.set_item(
        "source_q6_k_binding_count",
        native_model.source_q6_k_binding_count,
    )
    .expect("native_model.source_q6_k_binding_count should serialize");
    dict.set_item(
        "source_q8_0_binding_count",
        native_model.source_q8_0_binding_count,
    )
    .expect("native_model.source_q8_0_binding_count should serialize");
    dict.unbind()
}

fn native_source_quantization_dict<'py>(
    py: Python<'py>,
    source_quantization: &NativeSourceQuantization,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("format", &source_quantization.format)
        .expect("source_quantization.format should serialize");
    let counts_dict = PyDict::new(py);
    for (key, count) in &source_quantization.tensor_type_counts {
        counts_dict
            .set_item(key, *count)
            .expect("source_quantization.tensor_type_counts should serialize");
    }
    dict.set_item("tensor_type_counts", counts_dict)
        .expect("source_quantization.tensor_type_counts should serialize");
    dict.set_item(
        "quantized_tensor_count",
        source_quantization.quantized_tensor_count,
    )
    .expect("source_quantization.quantized_tensor_count should serialize");
    dict.set_item(
        "contains_quantized_tensors",
        source_quantization.contains_quantized_tensors,
    )
    .expect("source_quantization.contains_quantized_tensors should serialize");
    dict.unbind()
}

fn native_runtime_status_dict<'py>(
    py: Python<'py>,
    runtime_status: &NativeRuntimeStatus,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("ready", runtime_status.ready)
        .expect("runtime_status.ready should serialize");
    dict.set_item("blockers", runtime_status.blockers.clone())
        .expect("runtime_status.blockers should serialize");
    dict.set_item("notes", runtime_status.notes.clone())
        .expect("runtime_status.notes should serialize");
    dict.unbind()
}

fn capability_dict<'py>(py: Python<'py>, capabilities: &CapabilityReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("text_generation", capabilities.text_generation)
        .expect("capabilities should serialize");
    dict.set_item("token_streaming", capabilities.token_streaming)
        .expect("capabilities should serialize");
    dict.set_item("deterministic_mode", capabilities.deterministic_mode)
        .expect("capabilities should serialize");
    dict.set_item("prefix_reuse", capabilities.prefix_reuse)
        .expect("capabilities should serialize");
    dict.set_item(
        "long_context_validation",
        enum_label(py, capabilities.long_context_validation),
    )
    .expect("capabilities should serialize");
    dict.set_item(
        "benchmark_metrics",
        enum_label(py, capabilities.benchmark_metrics),
    )
    .expect("capabilities should serialize");
    dict.unbind()
}

fn host_dict<'py>(py: Python<'py>, host: &HostReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("os", &host.os)
        .expect("host.os should serialize");
    dict.set_item("arch", &host.arch)
        .expect("host.arch should serialize");
    if let Some(detected_soc) = host.detected_soc.as_deref() {
        dict.set_item("detected_soc", detected_soc)
            .expect("host.detected_soc should serialize");
    }
    dict.set_item("supported_mlx_runtime", host.supported_mlx_runtime)
        .expect("host.supported_mlx_runtime should serialize");
    dict.set_item(
        "unsupported_host_override_active",
        host.unsupported_host_override_active,
    )
    .expect("host.unsupported_host_override_active should serialize");
    dict.unbind()
}

fn metal_toolchain_dict<'py>(py: Python<'py>, toolchain: &MetalToolchainReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("fully_available", toolchain.fully_available)
        .expect("toolchain.fully_available should serialize");
    dict.set_item("metal", tool_status_dict(py, &toolchain.metal))
        .expect("toolchain.metal should serialize");
    dict.set_item("metallib", tool_status_dict(py, &toolchain.metallib))
        .expect("toolchain.metallib should serialize");
    dict.set_item("metal_ar", tool_status_dict(py, &toolchain.metal_ar))
        .expect("toolchain.metal_ar should serialize");
    dict.unbind()
}

fn tool_status_dict<'py>(py: Python<'py>, tool: &ToolStatusReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("available", tool.available)
        .expect("tool.available should serialize");
    if let Some(version) = tool.version.as_deref() {
        dict.set_item("version", version)
            .expect("tool.version should serialize");
    }
    dict.unbind()
}

pub(crate) fn route_dict<'py>(py: Python<'py>, route: &GenerateRouteReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    if let Some(value) = route.execution_plan.as_deref() {
        dict.set_item("execution_plan", value)
            .expect("execution_plan should serialize");
    }
    if let Some(value) = route.attention_route.as_deref() {
        dict.set_item("attention_route", value)
            .expect("attention_route should serialize");
    }
    if let Some(value) = route.kv_mode.as_deref() {
        dict.set_item("kv_mode", value)
            .expect("kv_mode should serialize");
    }
    if let Some(value) = route.prefix_cache_path.as_deref() {
        dict.set_item("prefix_cache_path", value)
            .expect("prefix_cache_path should serialize");
    }
    if let Some(value) = route.barrier_mode.as_deref() {
        dict.set_item("barrier_mode", value)
            .expect("barrier_mode should serialize");
    }
    if !route.crossover_decisions.is_empty() {
        let crossover = PyDict::new(py);
        for (key, value) in &route.crossover_decisions {
            crossover
                .set_item(key, value)
                .expect("crossover decision should serialize");
        }
        dict.set_item("crossover_decisions", crossover)
            .expect("crossover decisions should serialize");
    }
    dict.unbind()
}

pub(crate) fn request_report_dict<'py>(
    py: Python<'py>,
    report: &SessionRequestReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("request_id", report.request_id)
        .expect("request_id should serialize");
    dict.set_item("model_id", report.model_id.as_str())
        .expect("model_id should serialize");
    dict.set_item("state", enum_label(py, report.state))
        .expect("state should serialize");
    dict.set_item("prompt_tokens", report.prompt_tokens.clone())
        .expect("prompt_tokens should serialize");
    dict.set_item("processed_prompt_tokens", report.processed_prompt_tokens)
        .expect("processed_prompt_tokens should serialize");
    dict.set_item("output_tokens", report.output_tokens.clone())
        .expect("output_tokens should serialize");
    if has_observed_token_logprobs(&report.output_token_logprobs) {
        dict.set_item(
            "output_token_logprobs",
            report.output_token_logprobs.clone(),
        )
        .expect("output_token_logprobs should serialize");
    }
    dict.set_item("prompt_len", report.prompt_len)
        .expect("prompt_len should serialize");
    dict.set_item("output_len", report.output_len)
        .expect("output_len should serialize");
    dict.set_item("max_output_tokens", report.max_output_tokens)
        .expect("max_output_tokens should serialize");
    dict.set_item("cancel_requested", report.cancel_requested)
        .expect("cancel_requested should serialize");
    if let Some(finish_reason) = report.finish_reason {
        dict.set_item("finish_reason", enum_label(py, finish_reason))
            .expect("finish_reason should serialize");
    }
    if let Some(terminal_stop_reason) = report.terminal_stop_reason {
        dict.set_item("terminal_stop_reason", enum_label(py, terminal_stop_reason))
            .expect("terminal_stop_reason should serialize");
    }
    if let Some(execution_plan_ref) = report.execution_plan_ref.as_deref() {
        dict.set_item("execution_plan_ref", execution_plan_ref)
            .expect("execution_plan_ref should serialize");
    }
    dict.set_item("route", route_dict(py, &report.route))
        .expect("route should serialize");
    dict.unbind()
}

pub(crate) fn step_report_dict<'py>(py: Python<'py>, report: &EngineStepReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    if let Some(step_id) = report.step_id {
        dict.set_item("step_id", step_id)
            .expect("step_id should serialize");
    }
    dict.set_item("scheduled_requests", report.scheduled_requests)
        .expect("scheduled_requests should serialize");
    dict.set_item("scheduled_tokens", report.scheduled_tokens)
        .expect("scheduled_tokens should serialize");
    dict.set_item("ttft_events", report.ttft_events)
        .expect("ttft_events should serialize");
    dict.set_item("prefix_hits", report.prefix_hits)
        .expect("prefix_hits should serialize");
    dict.set_item("kv_usage_blocks", report.kv_usage_blocks)
        .expect("kv_usage_blocks should serialize");
    dict.set_item("evictions", report.evictions)
        .expect("evictions should serialize");
    dict.set_item("preempted_requests", report.preempted_requests)
        .expect("preempted_requests should serialize");
    dict.set_item("preempted_tokens", report.preempted_tokens)
        .expect("preempted_tokens should serialize");
    dict.set_item("cpu_time_us", report.cpu_time_us)
        .expect("cpu_time_us should serialize");
    dict.set_item("runner_time_us", report.runner_time_us)
        .expect("runner_time_us should serialize");
    if let Some(route) = report.route.as_ref() {
        dict.set_item("route", route_dict(py, route))
            .expect("route should serialize");
    }
    if let Some(metal_dispatch) = report.metal_dispatch.as_ref() {
        dict.set_item("metal_dispatch", metal_dispatch_dict(py, metal_dispatch))
            .expect("metal_dispatch should serialize");
    }
    dict.unbind()
}

fn metal_dispatch_dict<'py>(py: Python<'py>, report: &MetalDispatchStepReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("command_queue_label", report.command_queue_label.as_str())
        .expect("command_queue_label should serialize");
    dict.set_item("command_buffer_label", report.command_buffer_label.as_str())
        .expect("command_buffer_label should serialize");
    dict.set_item(
        "command_buffer_status",
        enum_label(py, report.command_buffer_status),
    )
    .expect("command_buffer_status should serialize");
    dict.set_item("runtime_device_name", report.runtime_device_name.as_str())
        .expect("runtime_device_name should serialize");
    dict.set_item(
        "runtime_required_pipeline_count",
        report.runtime_required_pipeline_count,
    )
    .expect("runtime_required_pipeline_count should serialize");
    dict.set_item(
        "runtime_max_thread_execution_width",
        report.runtime_max_thread_execution_width,
    )
    .expect("runtime_max_thread_execution_width should serialize");
    dict.set_item(
        "runtime_model_conditioned_inputs",
        report.runtime_model_conditioned_inputs,
    )
    .expect("runtime_model_conditioned_inputs should serialize");
    dict.set_item(
        "runtime_real_model_tensor_inputs",
        report.runtime_real_model_tensor_inputs,
    )
    .expect("runtime_real_model_tensor_inputs should serialize");
    dict.set_item(
        "runtime_complete_model_forward_supported",
        report.runtime_complete_model_forward_supported,
    )
    .expect("runtime_complete_model_forward_supported should serialize");
    dict.set_item(
        "runtime_model_bindings_prepared",
        report.runtime_model_bindings_prepared,
    )
    .expect("runtime_model_bindings_prepared should serialize");
    dict.set_item(
        "runtime_model_buffers_bound",
        report.runtime_model_buffers_bound,
    )
    .expect("runtime_model_buffers_bound should serialize");
    dict.set_item(
        "runtime_model_buffer_count",
        report.runtime_model_buffer_count,
    )
    .expect("runtime_model_buffer_count should serialize");
    dict.set_item(
        "runtime_model_buffer_bytes",
        report.runtime_model_buffer_bytes,
    )
    .expect("runtime_model_buffer_bytes should serialize");
    if let Some(model_family) = report.runtime_model_family.as_deref() {
        dict.set_item("runtime_model_family", model_family)
            .expect("runtime_model_family should serialize");
    }
    dict.set_item(
        "execution_direct_decode_token_count",
        report.execution_direct_decode_token_count,
    )
    .expect("execution_direct_decode_token_count should serialize");
    dict.set_item(
        "execution_direct_decode_checksum_lo",
        report.execution_direct_decode_checksum_lo,
    )
    .expect("execution_direct_decode_checksum_lo should serialize");
    dict.set_item(
        "execution_logits_output_count",
        report.execution_logits_output_count,
    )
    .expect("execution_logits_output_count should serialize");
    dict.set_item(
        "execution_remaining_logits_handle_count",
        report.execution_remaining_logits_handle_count,
    )
    .expect("execution_remaining_logits_handle_count should serialize");
    dict.set_item(
        "execution_model_bound_ffn_decode",
        report.execution_model_bound_ffn_decode,
    )
    .expect("execution_model_bound_ffn_decode should serialize");
    dict.set_item(
        "execution_real_model_forward_completed",
        report.execution_real_model_forward_completed,
    )
    .expect("execution_real_model_forward_completed should serialize");
    dict.set_item(
        "execution_prefix_native_dispatch_count",
        report.execution_prefix_native_dispatch_count,
    )
    .expect("execution_prefix_native_dispatch_count should serialize");
    dict.set_item(
        "execution_prefix_cpu_reference_dispatch_count",
        report.execution_prefix_cpu_reference_dispatch_count,
    )
    .expect("execution_prefix_cpu_reference_dispatch_count should serialize");
    dict.set_item(
        "execution_qkv_projection_token_count",
        report.execution_qkv_projection_token_count,
    )
    .expect("execution_qkv_projection_token_count should serialize");
    dict.set_item(
        "execution_layer_continuation_token_count",
        report.execution_layer_continuation_token_count,
    )
    .expect("execution_layer_continuation_token_count should serialize");
    dict.set_item(
        "execution_logits_projection_token_count",
        report.execution_logits_projection_token_count,
    )
    .expect("execution_logits_projection_token_count should serialize");
    dict.set_item(
        "execution_logits_vocab_scan_row_count",
        report.execution_logits_vocab_scan_row_count,
    )
    .expect("execution_logits_vocab_scan_row_count should serialize");
    dict.set_item(
        "binary_archive_state",
        enum_label(py, report.binary_archive_state),
    )
    .expect("binary_archive_state should serialize");
    dict.set_item(
        "binary_archive_attached_pipeline_count",
        report.binary_archive_attached_pipeline_count,
    )
    .expect("binary_archive_attached_pipeline_count should serialize");
    dict.set_item(
        "binary_archive_serialized",
        report.binary_archive_serialized,
    )
    .expect("binary_archive_serialized should serialize");
    dict.set_item("arena_token_capacity", report.arena_token_capacity)
        .expect("arena_token_capacity should serialize");
    dict.set_item("arena_slot_capacity", report.arena_slot_capacity)
        .expect("arena_slot_capacity should serialize");
    dict.set_item(
        "arena_attention_ref_capacity",
        report.arena_attention_ref_capacity,
    )
    .expect("arena_attention_ref_capacity should serialize");
    dict.set_item(
        "arena_gather_ref_capacity",
        report.arena_gather_ref_capacity,
    )
    .expect("arena_gather_ref_capacity should serialize");
    dict.set_item(
        "arena_gather_output_capacity",
        report.arena_gather_output_capacity,
    )
    .expect("arena_gather_output_capacity should serialize");
    dict.set_item("arena_copy_pair_capacity", report.arena_copy_pair_capacity)
        .expect("arena_copy_pair_capacity should serialize");
    dict.set_item("arena_sequence_capacity", report.arena_sequence_capacity)
        .expect("arena_sequence_capacity should serialize");
    dict.set_item("arena_reused_existing", report.arena_reused_existing)
        .expect("arena_reused_existing should serialize");
    dict.set_item("arena_grew_existing", report.arena_grew_existing)
        .expect("arena_grew_existing should serialize");

    let kernels = report
        .kernels
        .iter()
        .map(|kernel| metal_dispatch_kernel_dict(py, kernel))
        .collect::<Vec<_>>();
    dict.set_item("kernels", kernels)
        .expect("kernels should serialize");
    dict.set_item("numeric", metal_dispatch_numeric_dict(py, &report.numeric))
        .expect("numeric should serialize");
    dict.unbind()
}

fn metal_dispatch_kernel_dict<'py>(
    py: Python<'py>,
    kernel: &MetalDispatchKernelStepReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("function_name", kernel.function_name.as_str())
        .expect("function_name should serialize");
    dict.set_item("element_count", kernel.element_count)
        .expect("element_count should serialize");
    dict.set_item("threads_per_grid_width", kernel.threads_per_grid_width)
        .expect("threads_per_grid_width should serialize");
    dict.set_item(
        "threads_per_threadgroup_width",
        kernel.threads_per_threadgroup_width,
    )
    .expect("threads_per_threadgroup_width should serialize");
    dict.unbind()
}

fn metal_dispatch_numeric_dict<'py>(
    py: Python<'py>,
    numeric: &MetalDispatchNumericStepReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("key_cache_checksum", numeric.key_cache_checksum)
        .expect("key_cache_checksum should serialize");
    dict.set_item(
        "attention_output_checksum",
        numeric.attention_output_checksum,
    )
    .expect("attention_output_checksum should serialize");
    dict.set_item("gather_output_checksum", numeric.gather_output_checksum)
        .expect("gather_output_checksum should serialize");
    dict.set_item("copy_output_checksum", numeric.copy_output_checksum)
        .expect("copy_output_checksum should serialize");
    if let Some(validation) = numeric.validation.as_ref() {
        dict.set_item("validation", metal_dispatch_validation_dict(py, validation))
            .expect("validation should serialize");
    }
    dict.unbind()
}

fn metal_dispatch_validation_dict<'py>(
    py: Python<'py>,
    validation: &MetalDispatchValidationStepReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item(
        "expected_key_cache_checksum",
        validation.expected_key_cache_checksum,
    )
    .expect("expected_key_cache_checksum should serialize");
    dict.set_item(
        "expected_attention_output_checksum",
        validation.expected_attention_output_checksum,
    )
    .expect("expected_attention_output_checksum should serialize");
    dict.set_item(
        "expected_gather_output_checksum",
        validation.expected_gather_output_checksum,
    )
    .expect("expected_gather_output_checksum should serialize");
    dict.set_item(
        "expected_copy_output_checksum",
        validation.expected_copy_output_checksum,
    )
    .expect("expected_copy_output_checksum should serialize");
    dict.set_item(
        "attention_max_abs_diff_microunits",
        validation.attention_max_abs_diff_microunits,
    )
    .expect("attention_max_abs_diff_microunits should serialize");
    dict.unbind()
}

pub(crate) fn generate_response_dict<'py>(
    py: Python<'py>,
    response: &GenerateResponse,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("request_id", response.request_id)
        .expect("request_id should serialize");
    dict.set_item("model_id", response.model_id.as_str())
        .expect("model_id should serialize");
    dict.set_item("prompt_tokens", response.prompt_tokens.clone())
        .expect("prompt_tokens should serialize");
    if let Some(prompt_text) = response.prompt_text.as_deref() {
        dict.set_item("prompt_text", prompt_text)
            .expect("prompt_text should serialize");
    }
    dict.set_item("output_tokens", response.output_tokens.clone())
        .expect("output_tokens should serialize");
    if has_observed_token_logprobs(&response.output_token_logprobs) {
        dict.set_item(
            "output_token_logprobs",
            response.output_token_logprobs.clone(),
        )
        .expect("output_token_logprobs should serialize");
    }
    if let Some(output_text) = response.output_text.as_deref() {
        dict.set_item("output_text", output_text)
            .expect("output_text should serialize");
    }
    dict.set_item("status", enum_label(py, response.status))
        .expect("status should serialize");
    if let Some(finish_reason) = response.finish_reason {
        dict.set_item("finish_reason", enum_label(py, finish_reason))
            .expect("finish_reason should serialize");
    }
    dict.set_item("step_count", response.step_count)
        .expect("step_count should serialize");
    if let Some(ttft_step) = response.ttft_step {
        dict.set_item("ttft_step", ttft_step)
            .expect("ttft_step should serialize");
    }
    dict.set_item("route", route_dict(py, &response.route))
        .expect("route should serialize");
    dict.set_item("runtime", runtime_dict(py, &response.runtime))
        .expect("runtime should serialize");
    dict.unbind()
}

pub(crate) fn stream_event_dict<'py>(
    py: Python<'py>,
    event: &SdkGenerateStreamEvent,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("event", event.event_name())
        .expect("event name should serialize");

    match event {
        SdkGenerateStreamEvent::Request(payload) => {
            dict.set_item("request", request_report_dict(py, &payload.request))
                .expect("request payload should serialize");
            dict.set_item("runtime", runtime_dict(py, &payload.runtime))
                .expect("runtime payload should serialize");
        }
        SdkGenerateStreamEvent::Step(payload) => {
            dict.set_item("request", request_report_dict(py, &payload.request))
                .expect("request payload should serialize");
            dict.set_item("step", step_report_dict(py, &payload.step))
                .expect("step payload should serialize");
            dict.set_item("delta_tokens", payload.delta_tokens.clone())
                .expect("delta tokens should serialize");
            if has_observed_token_logprobs(&payload.delta_token_logprobs) {
                dict.set_item("delta_token_logprobs", payload.delta_token_logprobs.clone())
                    .expect("delta token logprobs should serialize");
            }
            if let Some(delta_text) = payload.delta_text.as_deref() {
                dict.set_item("delta_text", delta_text)
                    .expect("delta text should serialize");
            }
        }
        SdkGenerateStreamEvent::Response(payload) => {
            dict.set_item("response", generate_response_dict(py, &payload.response))
                .expect("response payload should serialize");
        }
    }

    dict.unbind()
}

fn has_observed_token_logprobs(logprobs: &[Option<f32>]) -> bool {
    logprobs.iter().any(Option::is_some)
}

fn enum_label<T>(py: Python<'_>, value: T) -> Py<PyAny>
where
    T: serde::Serialize,
{
    let string = serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string());
    PyString::new(py, &string).into_any().unbind()
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use ax_engine_sdk::SessionRequestReport;
    use pyo3::types::{PyAny, PyDictMethods, PyList};
    use serde_json::{Map, Value, json};

    pub(crate) fn dict_string(dict: &Bound<'_, PyDict>, key: &str) -> String {
        dict.get_item(key)
            .unwrap()
            .unwrap()
            .extract::<String>()
            .unwrap()
    }

    pub(crate) fn dict_tokens(dict: &Bound<'_, PyDict>, key: &str) -> Vec<u32> {
        dict.get_item(key)
            .unwrap()
            .unwrap()
            .extract::<Vec<u32>>()
            .unwrap()
    }

    fn py_any_to_json(value: &Bound<'_, PyAny>) -> Value {
        if value.is_none() {
            return Value::Null;
        }
        if let Ok(dict) = value.cast::<PyDict>() {
            return py_dict_to_json(dict);
        }
        if let Ok(list) = value.cast::<PyList>() {
            return Value::Array(list.iter().map(|item| py_any_to_json(&item)).collect());
        }
        if let Ok(string) = value.extract::<String>() {
            return Value::String(string);
        }
        if let Ok(boolean) = value.extract::<bool>() {
            return Value::Bool(boolean);
        }
        if let Ok(number) = value.extract::<i64>() {
            return Value::Number(number.into());
        }
        if let Ok(number) = value.extract::<u64>() {
            return Value::Number(number.into());
        }
        if let Ok(number) = value.extract::<f64>() {
            return Value::Number(
                serde_json::Number::from_f64(number)
                    .expect("finite python float should convert to json number"),
            );
        }

        panic!("unsupported python value in test json conversion");
    }

    pub(crate) fn py_dict_to_json(dict: &Bound<'_, PyDict>) -> Value {
        let mut object = Map::new();
        for (key, value) in dict.iter() {
            object.insert(
                key.extract::<String>()
                    .expect("python dict keys should be strings"),
                py_any_to_json(&value),
            );
        }
        Value::Object(object)
    }

    pub(crate) fn normalize_measurement_fields(value: &mut Value) {
        match value {
            Value::Object(map) => {
                // Python response dictionaries intentionally omit the SDK's
                // performance report, so exclude it from cross-surface test
                // comparisons along with lower-level timing measurements.
                map.remove("performance");
                map.remove("cpu_time_us");
                map.remove("runner_time_us");
                for value in map.values_mut() {
                    normalize_measurement_fields(value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    normalize_measurement_fields(value);
                }
            }
            _ => {}
        }
    }

    fn strip_unobserved_logprobs(value: &mut Value) {
        match value {
            Value::Object(map) => {
                for key in ["output_token_logprobs", "delta_token_logprobs"] {
                    let remove = map.get(key).is_some_and(|value| {
                        value
                            .as_array()
                            .is_some_and(|values| values.iter().all(Value::is_null))
                    });
                    if remove {
                        map.remove(key);
                    }
                }
                for value in map.values_mut() {
                    strip_unobserved_logprobs(value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    strip_unobserved_logprobs(value);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn sdk_stream_event_json(event: &SdkGenerateStreamEvent) -> Value {
        let mut value = match event {
            SdkGenerateStreamEvent::Request(payload) => json!({
                "event": "request",
                "runtime": payload.runtime,
                "request": payload.request,
            }),
            SdkGenerateStreamEvent::Step(payload) => {
                let mut json = json!({
                    "event": "step",
                    "request": payload.request,
                    "step": payload.step,
                    "delta_tokens": payload.delta_tokens,
                });
                if !payload.delta_token_logprobs.is_empty() {
                    json.as_object_mut()
                        .expect("step event json should be object")
                        .insert(
                            "delta_token_logprobs".to_string(),
                            json!(payload.delta_token_logprobs),
                        );
                }
                if let Some(delta_text) = payload.delta_text.as_deref() {
                    json.as_object_mut()
                        .expect("step event json should be object")
                        .insert("delta_text".to_string(), json!(delta_text));
                }
                json
            }
            SdkGenerateStreamEvent::Response(payload) => json!({
                "event": "response",
                "response": payload.response,
            }),
        };
        strip_unobserved_logprobs(&mut value);
        value
    }

    pub(crate) fn sdk_request_report_json(report: SessionRequestReport) -> Value {
        let mut value = serde_json::to_value(report).expect("sdk request report should serialize");
        strip_unobserved_logprobs(&mut value);
        value
    }
}
