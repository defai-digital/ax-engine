use super::*;

pub(crate) fn build_environment_json(
    run_id: &str,
    command: &str,
    manifest_path: &Path,
    output_root: &Path,
    started_at_unix_s: u64,
    execution: &RuntimeResult,
) -> Result<Value, CliError> {
    let completed_at_unix_s = unix_timestamp_secs().unwrap_or(started_at_unix_s);
    let manifest_fingerprint = file_fingerprint_fnv1a64(manifest_path)?;
    let cwd = env::current_dir().map_err(|error| {
        CliError::Runtime(format!(
            "failed to resolve current working directory for benchmark provenance: {error}"
        ))
    })?;
    let hostname = env::var("HOSTNAME")
        .or_else(|_| env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    let logical_cpu_count = thread::available_parallelism()
        .ok()
        .map(|parallelism| parallelism.get());
    let system_model = detect_system_model().unwrap_or_else(|| "unknown".to_string());
    let soc = detect_soc().unwrap_or_else(|| "unknown".to_string());
    let memory_bytes = detect_memory_bytes();
    let memory_gb = memory_bytes.map(bytes_to_gib);
    let os_version = detect_os_version().unwrap_or_else(|| "unknown".to_string());
    let os_build = detect_os_build().unwrap_or_else(|| "unknown".to_string());
    let kernel_release = detect_kernel_release().unwrap_or_else(|| "unknown".to_string());
    let metal_driver =
        env::var("AX_METAL_DRIVER").unwrap_or_else(|_| default_metal_driver().to_string());
    let device = if system_model != "unknown" {
        system_model.clone()
    } else {
        hostname.clone()
    };
    let mut route_json = serialize_route_metadata(&execution.observation.route_metadata);
    annotate_route_json_with_decode_batching_opportunity(
        &mut route_json,
        &execution.observation.step_trace,
    );

    Ok(json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "run_id": run_id,
        "engine_commit": env::var("AX_ENGINE_COMMIT").unwrap_or_else(|_| "unknown".to_string()),
        "machine": {
            "device": device,
            "hostname": hostname,
            "system_model": system_model,
            "soc": soc,
            "memory_gb": memory_gb,
            "memory_bytes": memory_bytes,
            "arch": env::consts::ARCH,
            "logical_cpu_count": logical_cpu_count
        },
        "software": {
            "os_family": env::consts::OS,
            "os_version": os_version,
            "os_build": os_build,
            "kernel_release": kernel_release,
            "metal_driver": metal_driver,
            "tool_mode": execution.tool_mode,
            "ax_engine_bench_version": env!("CARGO_PKG_VERSION")
        },
        "runtime": serialize_runtime_metadata(
            &execution.runtime,
            execution.observation.runtime_report.as_ref(),
        ),
        "route": route_json,
        "benchmark": {
            "started_at": format!("unix:{started_at_unix_s}"),
            "started_at_unix_s": started_at_unix_s,
            "completed_at_unix_s": completed_at_unix_s,
            "command": format!(
                "ax-engine-bench {} --manifest {} --output-root {}",
                command,
                manifest_path.display(),
                output_root.display()
            ),
            "subcommand": command,
            "schema_family": "ax.engine_bench.v1",
            "manifest_path": manifest_path.display().to_string(),
            "manifest_fingerprint_fnv1a64": manifest_fingerprint,
            "output_root": output_root.display().to_string(),
            "cwd": cwd.display().to_string()
        }
    }))
}

#[allow(
    clippy::expect_used,
    reason = "the metrics field is an object literal created in this function"
)]
pub(crate) fn build_metrics_json(run_id: &str, execution: &RuntimeResult) -> Value {
    let mut metrics_json = json!({
        "schema_version": "ax.engine_bench.metrics.v1",
        "run_id": run_id,
        "status": execution.status_label(),
        "runtime": serialize_runtime_metadata(
            &execution.runtime,
            execution.observation.runtime_report.as_ref(),
        ),
        "metrics": {
            "ttft_ms": execution.observation.ttft_ms.unwrap_or_default() as f64,
            "prefill_tok_s": execution.observation.prefill_tok_s(),
            "decode_tok_s": execution.observation.decode_tok_s(),
            "e2e_latency_ms": execution.observation.e2e_latency_ms as f64,
            "memory_peak_mb": execution.observation.memory_peak_mb,
            "cpu_time_per_token_us": execution.observation.cpu_time_per_token_us(),
            "runner_time_per_token_us": execution.observation.runner_time_per_token_us(),
            "runner_time_share_pct": execution.observation.runner_time_share_pct(),
            "prefix_hit_rate": execution.observation.prefix_hit_rate(),
            "kv_block_usage": execution.observation.kv_peak_blocks as f64,
            "evictions": execution.observation.evictions as f64
        },
        "memory_peak_estimate": memory_peak_estimate_json(
            &execution.runtime,
            &execution.observation,
        ),
        "token_accounting": token_accounting_json(&execution.observation),
        "correctness": {
            "passed": execution.correctness.passed,
            "reason": execution.correctness.reason
        },
        "determinism": {
            "passed": execution.determinism.passed,
            "reason": execution.determinism.reason
        },
        "step_count": execution.observation.step_count,
        "scheduled_tokens_per_step": execution.observation.scheduled_tokens_per_step(),
        "memory_blocked_steps": execution.observation.memory_blocked_steps,
        "memory_blocked_request_events": execution.observation.memory_blocked_request_events,
        "replay_status": execution.observation.replay_status(),
        "churn_status": execution.observation.churn_status()
    });

    if execution.runtime.llama_cpp_preset.is_some() {
        metrics_json
            .get_mut("metrics")
            .and_then(Value::as_object_mut)
            .expect("metrics artifact should contain an object metrics field")
            .insert(
                "delegated_llama_cpp".to_string(),
                json!({
                    "kv_usage_blocks": execution.observation.kv_peak_blocks,
                    "requests_processing_events": execution.observation.llama_cpp_processing_request_events,
                    "requests_deferred_events": execution.observation.llama_cpp_deferred_request_events,
                    "backend_reported_cached_prompt_tokens": backend_reported_cached_prompt_tokens(
                        &execution.observation.route_metadata
                    ).unwrap_or(0),
                    "cache_reuse_observed": backend_reported_cached_prompt_tokens(
                        &execution.observation.route_metadata
                    ).is_some()
                }),
            );
    }

    metrics_json
}

pub(crate) fn memory_peak_estimate_json(
    runtime: &RuntimeConfig,
    observation: &RuntimeObservation,
) -> Value {
    json!({
        "kind": "kv_cache_legacy_estimate",
        "metric": "memory_peak_mb",
        "formula": "kv_peak_blocks * block_size_tokens * estimated_bytes_per_token / 1024",
        "kv_peak_blocks": observation.kv_peak_blocks,
        "block_size_tokens": runtime.block_size_tokens,
        "estimated_bytes_per_token": LEGACY_KV_CACHE_ESTIMATED_BYTES_PER_TOKEN,
        "model_architecture_adjusted": false
    })
}

pub(crate) fn token_accounting_json(observation: &RuntimeObservation) -> Value {
    json!({
        "sources": observation.token_accounting_sources.clone(),
        "contains_estimates": observation
            .token_accounting_sources
            .keys()
            .any(|source| source.contains("estimate") || source.contains("synthetic")),
    })
}

pub(crate) fn build_routes_json(run_id: &str, execution: &RuntimeResult) -> Value {
    let mut route_json = serialize_route_metadata(&execution.observation.route_metadata);
    annotate_route_json_with_decode_batching_opportunity(
        &mut route_json,
        &execution.observation.step_trace,
    );
    json!({
        "schema_version": "ax.engine_bench.routes.v1",
        "run_id": run_id,
        "runtime": serialize_runtime_metadata(
            &execution.runtime,
            execution.observation.runtime_report.as_ref(),
        ),
        "route": route_json,
        "mode": execution.tool_mode
    })
}

pub(crate) fn build_trace_json(run_id: &str, execution: &RuntimeResult) -> Value {
    json!({
        "schema_version": "ax.engine_bench.trace.v1",
        "run_id": run_id,
        "mode": execution.tool_mode,
        "runtime": serialize_runtime_metadata(
            &execution.runtime,
            execution.observation.runtime_report.as_ref(),
        ),
        "steps": execution
            .observation
            .step_trace
            .iter()
            .map(StepTraceEntry::json)
            .collect::<Vec<_>>(),
        "observation": execution.observation.digest
    })
}

pub(crate) fn build_execution_summary_markdown(
    run_id: &str,
    command: &str,
    manifest_path: &Path,
    execution: &RuntimeResult,
) -> String {
    let correctness = if execution.correctness.passed {
        "pass"
    } else {
        "fail"
    };
    let determinism = if execution.determinism.passed {
        "pass"
    } else {
        "fail"
    };
    let fallback_reason = execution
        .runtime
        .resolved_backend
        .fallback_reason
        .as_deref()
        .unwrap_or("none");
    let selected_backend = json_string_label(execution.runtime.resolved_backend.selected_backend);
    let support_tier = json_string_label(execution.runtime.resolved_backend.support_tier);
    let resolution_policy = json_string_label(execution.runtime.backend_policy.resolution_policy);
    let backend_adapter = execution
        .runtime
        .backend_adapter
        .as_ref()
        .map(|adapter| serde_json::to_string(adapter).unwrap_or_else(|_| "unknown".to_string()))
        .unwrap_or_else(|| "none".to_string());
    let prefix_reuse_provenance =
        prefix_reuse_provenance_label(&execution.observation.route_metadata);
    let execution_semantics =
        route_execution_semantics_label(&execution.observation.route_metadata);
    let metal_numeric_scaffold_only = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_numeric_scaffold_only",
    );
    let metal_complete_model_forward_supported = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_complete_model_forward_supported",
    ) || route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_runtime_complete_model_forward_supported",
    );
    let metal_real_model_forward = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_real_model_forward",
    );
    let metal_model_conditioned_inputs = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_model_conditioned_inputs",
    ) || route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_runtime_model_conditioned_inputs",
    );
    let mlx_metal_prefix_layers_attention = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_layers_native_attention",
    );
    let metal_prefix_layers_cpu_reference = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_layers_cpu_reference",
    );
    let mlx_metal_prefix_dispatch_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_dispatch_count",
    );
    let metal_prefix_cpu_reference_dispatch_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_reference_dispatch_count",
    );
    let mlx_metal_prefix_projection_row_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_projection_row_count",
    );
    let metal_prefix_cpu_projection_row_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_projection_row_count",
    );
    let mlx_metal_prefix_rms_norm_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_rms_norm_element_count",
    );
    let metal_prefix_cpu_rms_norm_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_rms_norm_element_count",
    );
    let mlx_metal_prefix_ffn_activation_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_ffn_activation_element_count",
    );
    let metal_prefix_cpu_ffn_activation_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_ffn_activation_element_count",
    );
    let mlx_metal_prefix_residual_add_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_residual_add_element_count",
    );
    let metal_prefix_cpu_residual_add_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_residual_add_element_count",
    );
    let mlx_metal_prefix_scale_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_scale_element_count",
    );
    let metal_prefix_cpu_scale_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_scale_element_count",
    );
    let metal_direct_decode_tokens = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_tokens",
    );
    let metal_direct_decode_model_bound_ffn = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_model_bound_ffn",
    );
    let metal_direct_decode_checksum_lo = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_checksum_lo",
    );
    let metal_real_model_tensor_inputs = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_real_model_tensor_inputs",
    ) || route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_runtime_real_model_tensor_inputs",
    );
    let metal_model_artifacts_validated = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_model_artifacts_validated",
    );
    let mlx_metal_projection_f32_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_f32_binding_count",
    );
    let mlx_metal_projection_f16_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_f16_binding_count",
    );
    let mlx_metal_projection_bf16_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_bf16_binding_count",
    );
    let mlx_metal_projection_unsupported_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_unsupported_binding_count",
    );
    let mlx_metal_projection_source_quantized_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_source_quantized_binding_count",
    );
    let mlx_metal_rms_norm_f32_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_f32_binding_count",
    );
    let mlx_metal_rms_norm_f16_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_f16_binding_count",
    );
    let mlx_metal_rms_norm_bf16_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_bf16_binding_count",
    );
    let mlx_metal_rms_norm_unsupported_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_unsupported_binding_count",
    );
    let mlx_metal_rms_norm_source_quantized_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_source_quantized_binding_count",
    );
    let mlx_metal_direct_decode_projection_row_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_projection_row_count",
    );
    let metal_direct_decode_cpu_projection_row_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_projection_row_count",
    );
    let mlx_metal_direct_decode_rms_norm_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_rms_norm_element_count",
    );
    let metal_direct_decode_cpu_rms_norm_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_rms_norm_element_count",
    );
    let mlx_metal_direct_decode_ffn_activation_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_ffn_activation_element_count",
    );
    let metal_direct_decode_cpu_ffn_activation_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_ffn_activation_element_count",
    );
    let mlx_metal_direct_decode_residual_add_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_residual_add_element_count",
    );
    let metal_direct_decode_cpu_residual_add_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_residual_add_element_count",
    );
    let mlx_metal_direct_decode_scale_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_scale_element_count",
    );
    let metal_direct_decode_cpu_scale_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_scale_element_count",
    );
    let metal_direct_decode_batched_logits_group_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_batched_logits_group_count",
    );
    let metal_direct_decode_batched_logits_token_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_batched_logits_token_count",
    );
    let metal_direct_decode_batched_group_fallback_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_batched_group_fallback_count",
    );
    let metal_direct_decode_batched_group_fallback_token_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_batched_group_fallback_token_count",
    );
    let backend_reported_cached_prompt_tokens =
        backend_reported_cached_prompt_tokens(&execution.observation.route_metadata)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string());
    let llama_cpp_preset = execution
        .runtime
        .llama_cpp_preset
        .as_ref()
        .map(|preset| serde_json::to_string(preset).unwrap_or_else(|_| "unknown".to_string()))
        .unwrap_or_else(|| "none".to_string());
    let llama_cpp_processing_request_events =
        execution.observation.llama_cpp_processing_request_events;
    let llama_cpp_deferred_request_events = execution.observation.llama_cpp_deferred_request_events;
    let prefix_projection_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_projection_row_count,
        metal_prefix_cpu_projection_row_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let prefix_rms_norm_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_rms_norm_element_count,
        metal_prefix_cpu_rms_norm_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let prefix_ffn_activation_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_ffn_activation_element_count,
        metal_prefix_cpu_ffn_activation_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let prefix_residual_add_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_residual_add_element_count,
        metal_prefix_cpu_residual_add_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let prefix_scale_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_scale_element_count,
        metal_prefix_cpu_scale_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_projection_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_projection_row_count,
        metal_direct_decode_cpu_projection_row_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_rms_norm_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_rms_norm_element_count,
        metal_direct_decode_cpu_rms_norm_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_ffn_activation_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_ffn_activation_element_count,
        metal_direct_decode_cpu_ffn_activation_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_residual_add_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_residual_add_element_count,
        metal_direct_decode_cpu_residual_add_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_scale_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_scale_element_count,
        metal_direct_decode_cpu_scale_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let metal_direct_decode_batching_opportunity_observed =
        direct_decode_batching_opportunity_observed(&execution.observation.step_trace);
    let native_dense_dequantized_source = execution
        .runtime
        .native_model_report()
        .is_some_and(|report| native_model_report_has_dense_dequantized_source(&report));
    let mlx_metal_readiness = mlx_metal_readiness(MlxMetalReadinessInputs {
        tool_mode: execution.tool_mode,
        selected_backend: &selected_backend,
        native_dense_dequantized_source,
        native_quantized_projection_binding_count:
            mlx_metal_projection_source_quantized_binding_count,
        direct_decode_batching_opportunity_observed:
            metal_direct_decode_batching_opportunity_observed,
        metal_complete_model_forward_supported,
        metal_real_model_forward,
        metal_model_artifacts_validated,
        mlx_metal_prefix_layers_attention,
        metal_prefix_layers_cpu_reference,
        metal_prefix_cpu_reference_dispatch_count,
        mlx_metal_prefix_projection_row_count,
        metal_prefix_cpu_projection_row_count,
        mlx_metal_prefix_rms_norm_element_count,
        metal_prefix_cpu_rms_norm_element_count,
        mlx_metal_prefix_ffn_activation_element_count,
        metal_prefix_cpu_ffn_activation_element_count,
        mlx_metal_prefix_residual_add_element_count,
        metal_prefix_cpu_residual_add_element_count,
        mlx_metal_prefix_scale_element_count,
        metal_prefix_cpu_scale_element_count,
        metal_direct_decode_tokens,
        metal_direct_decode_model_bound_ffn,
        metal_direct_decode_batched_logits_group_count,
        metal_direct_decode_batched_logits_token_count,
        metal_direct_decode_batched_group_fallback_count,
        metal_direct_decode_batched_group_fallback_token_count,
        mlx_metal_direct_decode_projection_row_count,
        metal_direct_decode_cpu_projection_row_count,
        mlx_metal_direct_decode_rms_norm_element_count,
        metal_direct_decode_cpu_rms_norm_element_count,
        mlx_metal_direct_decode_ffn_activation_element_count,
        metal_direct_decode_cpu_ffn_activation_element_count,
        mlx_metal_direct_decode_residual_add_element_count,
        metal_direct_decode_cpu_residual_add_element_count,
        mlx_metal_direct_decode_scale_element_count,
        metal_direct_decode_cpu_scale_element_count,
    });
    let mlx_metal_readiness_status = mlx_metal_readiness.status;
    let mlx_metal_hot_path_cpu_fallback_free = mlx_metal_readiness.hot_path_cpu_fallback_free;
    let mlx_metal_batched_direct_decode_logits_ready =
        mlx_metal_readiness.batched_direct_decode_logits_ready;
    let mlx_metal_prefix_min_dispatch_share =
        mlx_metal_dispatch_share_label(mlx_metal_readiness.prefix_min_mlx_metal_dispatch_share);
    let mlx_metal_direct_decode_min_dispatch_share = mlx_metal_dispatch_share_label(
        mlx_metal_readiness.direct_decode_min_mlx_metal_dispatch_share,
    );
    let mlx_metal_readiness_blockers = mlx_metal_readiness.blockers_label();
    let summary_note = if execution.tool_mode == "llama_cpp_blocking_runtime" {
        "\nLlama.cpp benchmarking currently runs through the supported llama.cpp route. TTFT, prefill throughput, and decode throughput should be read as llama.cpp wall-time proxies rather than MLX-mode stepwise measurements.\n"
    } else if execution.tool_mode == "llama_cpp_stepwise_runtime" {
        "\nLlama.cpp benchmarking currently runs through the delegated SDK stepwise lifecycle over `llama.cpp /completion`. TTFT and decode throughput reflect delegated step cadence, and any reported prefix reuse is backend-managed prompt-cache evidence such as `delegated_prompt_cache`, not AX Engine scheduler / KV reuse. Prefill throughput, KV metrics, and runner metrics therefore remain exploratory.\n"
    } else if execution_semantics == "metal_real_model_forward" {
        "\nThis run reports a Metal path that marked real model forward execution. Compare it only against runs with the same execution semantics and validated model provenance.\n"
    } else if execution_semantics == "metal_multilayer_mixed_prefix_attention" {
        "\nThis run loaded validated multi-layer model artifacts and executed part of prefix-layer attention through MLX Metal dispatch, but some multilayer prefix work still fell back to CPU reference. Read it as partial MLX dense-forward progress, not as full MLX prefix coverage or final inference evidence.\n"
    } else if execution_semantics == "metal_multilayer_native_prefix_attention" {
        "\nThis run loaded validated multi-layer model artifacts and already executed at least part of prefix-layer attention through MLX Metal dispatch, but it still did not report a complete dense real-model forward. Read it as a stronger MLX Metal bring-up milestone than CPU-reference multilayer staging, not as final inference evidence.\n"
    } else if execution_semantics == "metal_multilayer_model_incomplete" {
        "\nThis run loaded validated multi-layer model artifacts into the Metal path, but it did not report a complete real model forward. Read it as partial MLX Metal bring-up progress, not final dense inference evidence.\n"
    } else if execution_semantics == "metal_model_bound_ffn_decode" {
        "\nThis run executed Metal dispatch with real model tensor inputs and reported model-bound FFN direct decode continuation. It is a stronger MLX Metal bring-up milestone than tensor-input-only evidence, but it still does not prove a full release-grade dense forward path.\n"
    } else if execution_semantics == "metal_real_model_tensor_inputs" {
        "\nThis run executed Metal dispatch with real model tensor inputs present, but it did not mark a real model forward pass. Treat it as model-bound bring-up evidence rather than final inference performance.\n"
    } else if execution_semantics == "metal_model_conditioned_numeric_scaffold" {
        "\nThis run executed Metal dispatch with model-conditioned staged inputs, but it still remains a numeric scaffold path rather than a full model forward. Do not compare it directly with release-grade inference results.\n"
    } else if execution_semantics == "metal_numeric_scaffold_only" {
        "\nThis run submitted real Metal command buffers, but the data path remained numeric scaffold only. It validates kernel plumbing and runtime bring-up, not model-bound inference quality or throughput.\n"
    } else {
        "\nThis run executed the current engine bring-up path through the preview SDK session, request manager, scheduler, KV manager, deterministic runner, and deterministic sampler.\n"
    };

    format!(
        "# Benchmark Run\n\n- run_id: `{run_id}`\n- command: `ax-engine-bench {command}`\n- manifest: `{}`\n- status: `{}`\n- tool_mode: `{}`\n- selected_backend: `{selected_backend}`\n- support_tier: `{support_tier}`\n- resolution_policy: `{resolution_policy}`\n- backend_adapter: `{backend_adapter}`\n- llama_cpp_preset: `{llama_cpp_preset}`\n- fallback_reason: `{fallback_reason}`\n- execution_semantics: `{execution_semantics}`\n- mlx_metal_readiness: `{mlx_metal_readiness_status}`\n- mlx_metal_hot_path_cpu_fallback_free: `{mlx_metal_hot_path_cpu_fallback_free}`\n- mlx_metal_batched_direct_decode_logits_ready: `{mlx_metal_batched_direct_decode_logits_ready}`\n- mlx_metal_prefix_min_dispatch_share: `{mlx_metal_prefix_min_dispatch_share}`\n- mlx_metal_direct_decode_min_dispatch_share: `{mlx_metal_direct_decode_min_dispatch_share}`\n- mlx_metal_readiness_blockers: `{mlx_metal_readiness_blockers}`\n- metal_numeric_scaffold_only: `{metal_numeric_scaffold_only}`\n- metal_complete_model_forward_supported: `{metal_complete_model_forward_supported}`\n- metal_model_conditioned_inputs: `{metal_model_conditioned_inputs}`\n- mlx_metal_prefix_layers_attention: `{mlx_metal_prefix_layers_attention}`\n- metal_prefix_layers_cpu_reference: `{metal_prefix_layers_cpu_reference}`\n- mlx_metal_prefix_dispatch_count: `{mlx_metal_prefix_dispatch_count}`\n- metal_prefix_cpu_reference_dispatch_count: `{metal_prefix_cpu_reference_dispatch_count}`\n- mlx_metal_prefix_projection_row_count: `{mlx_metal_prefix_projection_row_count}`\n- metal_prefix_cpu_projection_row_count: `{metal_prefix_cpu_projection_row_count}`\n- metal_prefix_projection_mlx_metal_dispatch_share: `{prefix_projection_mlx_metal_dispatch_share}`\n- mlx_metal_prefix_rms_norm_element_count: `{mlx_metal_prefix_rms_norm_element_count}`\n- metal_prefix_cpu_rms_norm_element_count: `{metal_prefix_cpu_rms_norm_element_count}`\n- metal_prefix_rms_norm_mlx_metal_dispatch_share: `{prefix_rms_norm_mlx_metal_dispatch_share}`\n- mlx_metal_prefix_ffn_activation_element_count: `{mlx_metal_prefix_ffn_activation_element_count}`\n- metal_prefix_cpu_ffn_activation_element_count: `{metal_prefix_cpu_ffn_activation_element_count}`\n- metal_prefix_ffn_activation_mlx_metal_dispatch_share: `{prefix_ffn_activation_mlx_metal_dispatch_share}`\n- mlx_metal_prefix_residual_add_element_count: `{mlx_metal_prefix_residual_add_element_count}`\n- metal_prefix_cpu_residual_add_element_count: `{metal_prefix_cpu_residual_add_element_count}`\n- metal_prefix_residual_add_mlx_metal_dispatch_share: `{prefix_residual_add_mlx_metal_dispatch_share}`\n- mlx_metal_prefix_scale_element_count: `{mlx_metal_prefix_scale_element_count}`\n- metal_prefix_cpu_scale_element_count: `{metal_prefix_cpu_scale_element_count}`\n- metal_prefix_scale_mlx_metal_dispatch_share: `{prefix_scale_mlx_metal_dispatch_share}`\n- metal_direct_decode_tokens: `{metal_direct_decode_tokens}`\n- metal_direct_decode_batching_opportunity_observed: `{metal_direct_decode_batching_opportunity_observed}`\n- metal_direct_decode_model_bound_ffn: `{metal_direct_decode_model_bound_ffn}`\n- metal_direct_decode_checksum_lo: `{metal_direct_decode_checksum_lo}`\n- metal_direct_decode_batched_logits_group_count: `{metal_direct_decode_batched_logits_group_count}`\n- metal_direct_decode_batched_logits_token_count: `{metal_direct_decode_batched_logits_token_count}`\n- metal_direct_decode_batched_group_fallback_count: `{metal_direct_decode_batched_group_fallback_count}`\n- metal_direct_decode_batched_group_fallback_token_count: `{metal_direct_decode_batched_group_fallback_token_count}`\n- metal_real_model_tensor_inputs: `{metal_real_model_tensor_inputs}`\n- metal_real_model_forward: `{metal_real_model_forward}`\n- metal_model_artifacts_validated: `{metal_model_artifacts_validated}`\n- mlx_metal_projection_f32_binding_count: `{mlx_metal_projection_f32_binding_count}`\n- mlx_metal_projection_f16_binding_count: `{mlx_metal_projection_f16_binding_count}`\n- mlx_metal_projection_bf16_binding_count: `{mlx_metal_projection_bf16_binding_count}`\n- mlx_metal_projection_unsupported_binding_count: `{mlx_metal_projection_unsupported_binding_count}`\n- mlx_metal_projection_source_quantized_binding_count: `{mlx_metal_projection_source_quantized_binding_count}`\n- mlx_metal_rms_norm_f32_binding_count: `{mlx_metal_rms_norm_f32_binding_count}`\n- mlx_metal_rms_norm_f16_binding_count: `{mlx_metal_rms_norm_f16_binding_count}`\n- mlx_metal_rms_norm_bf16_binding_count: `{mlx_metal_rms_norm_bf16_binding_count}`\n- mlx_metal_rms_norm_unsupported_binding_count: `{mlx_metal_rms_norm_unsupported_binding_count}`\n- mlx_metal_rms_norm_source_quantized_binding_count: `{mlx_metal_rms_norm_source_quantized_binding_count}`\n- mlx_metal_direct_decode_projection_row_count: `{mlx_metal_direct_decode_projection_row_count}`\n- metal_direct_decode_cpu_projection_row_count: `{metal_direct_decode_cpu_projection_row_count}`\n- metal_direct_decode_projection_mlx_metal_dispatch_share: `{direct_decode_projection_mlx_metal_dispatch_share}`\n- mlx_metal_direct_decode_rms_norm_element_count: `{mlx_metal_direct_decode_rms_norm_element_count}`\n- metal_direct_decode_cpu_rms_norm_element_count: `{metal_direct_decode_cpu_rms_norm_element_count}`\n- metal_direct_decode_rms_norm_mlx_metal_dispatch_share: `{direct_decode_rms_norm_mlx_metal_dispatch_share}`\n- mlx_metal_direct_decode_ffn_activation_element_count: `{mlx_metal_direct_decode_ffn_activation_element_count}`\n- metal_direct_decode_cpu_ffn_activation_element_count: `{metal_direct_decode_cpu_ffn_activation_element_count}`\n- metal_direct_decode_ffn_activation_mlx_metal_dispatch_share: `{direct_decode_ffn_activation_mlx_metal_dispatch_share}`\n- mlx_metal_direct_decode_residual_add_element_count: `{mlx_metal_direct_decode_residual_add_element_count}`\n- metal_direct_decode_cpu_residual_add_element_count: `{metal_direct_decode_cpu_residual_add_element_count}`\n- metal_direct_decode_residual_add_mlx_metal_dispatch_share: `{direct_decode_residual_add_mlx_metal_dispatch_share}`\n- mlx_metal_direct_decode_scale_element_count: `{mlx_metal_direct_decode_scale_element_count}`\n- metal_direct_decode_cpu_scale_element_count: `{metal_direct_decode_cpu_scale_element_count}`\n- metal_direct_decode_scale_mlx_metal_dispatch_share: `{direct_decode_scale_mlx_metal_dispatch_share}`\n- correctness: `{}`\n- determinism: `{}`\n- ttft_ms: `{:.2}`\n- prefill_tok_s: `{:.2}`\n- decode_tok_s: `{:.2}`\n- e2e_latency_ms: `{:.2}`\n- cpu_time_per_token_us: `{:.2}`\n- runner_time_per_token_us: `{:.2}`\n- runner_time_share_pct: `{:.2}`\n- prefix_hit_rate: `{:.2}`\n- prefix_reuse_provenance: `{prefix_reuse_provenance}`\n- backend_reported_cached_prompt_tokens: `{backend_reported_cached_prompt_tokens}`\n- llama_cpp_processing_request_events: `{llama_cpp_processing_request_events}`\n- llama_cpp_deferred_request_events: `{llama_cpp_deferred_request_events}`\n- kv_peak_blocks: `{}`\n{}",
        manifest_path.display(),
        execution.status_label(),
        execution.tool_mode,
        correctness,
        determinism,
        execution.observation.ttft_ms.unwrap_or_default() as f64,
        execution.observation.prefill_tok_s(),
        execution.observation.decode_tok_s(),
        execution.observation.e2e_latency_ms as f64,
        execution.observation.cpu_time_per_token_us(),
        execution.observation.runner_time_per_token_us(),
        execution.observation.runner_time_share_pct(),
        execution.observation.prefix_hit_rate(),
        execution.observation.kv_peak_blocks,
        summary_note
    )
}

pub(crate) fn build_contract_failure_json(
    run_id: &str,
    command: &str,
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    output_root: &Path,
    started_at_unix_s: u64,
    message: &str,
) -> Result<Value, CliError> {
    let completed_at_unix_s = unix_timestamp_secs().unwrap_or(started_at_unix_s);
    let cwd = env::current_dir().map_err(|error| {
        CliError::Runtime(format!(
            "failed to resolve current working directory for contract failure artifact: {error}"
        ))
    })?;
    let failure = classify_contract_failure(manifest, message);

    Ok(json!({
        "schema_version": "ax.engine_bench.contract_failure.v1",
        "run_id": run_id,
        "status": "contract_failure",
        "command": command,
        "tool_mode": contract_failure_tool_mode(manifest),
        "manifest": {
            "id": manifest.id,
            "class": manifest.class,
            "scenario": manifest.scenario
        },
        "runtime": serialize_contract_failure_runtime(manifest),
        "failure": {
            "class": "contract",
            "code": failure.code,
            "message": message,
            "recommended_action": failure.recommended_action
        },
        "benchmark": {
            "started_at_unix_s": started_at_unix_s,
            "completed_at_unix_s": completed_at_unix_s,
            "manifest_path": manifest_path.display().to_string(),
            "output_root": output_root.display().to_string(),
            "cwd": cwd.display().to_string()
        }
    }))
}

pub(crate) fn build_contract_failure_summary_markdown(
    run_id: &str,
    command: &str,
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    message: &str,
) -> String {
    let failure = classify_contract_failure(manifest, message);
    let runtime = serialize_contract_failure_runtime(manifest);
    let selected_backend = nested_value(&runtime, &["selected_backend"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let support_tier = nested_value(&runtime, &["support_tier"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let resolution_policy = nested_value(&runtime, &["resolution_policy"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let backend_adapter = nested_value(&runtime, &["backend_adapter"])
        .cloned()
        .unwrap_or(Value::String("none".to_string()));
    format!(
        "# Benchmark Contract Failure\n\n- run_id: `{run_id}`\n- command: `ax-engine-bench {command}`\n- manifest: `{}`\n- status: `contract_failure`\n- code: `{}`\n- selected_backend: `{selected_backend}`\n- support_tier: `{support_tier}`\n- resolution_policy: `{resolution_policy}`\n- backend_adapter: `{}`\n- scenario: `{}`\n\nFailure reason:\n\n{}\n\nRecommended action:\n\n{}\n",
        manifest_path.display(),
        failure.code,
        json_value_label(&backend_adapter),
        manifest.scenario,
        message,
        failure.recommended_action
    )
}

#[allow(
    clippy::expect_used,
    reason = "BenchmarkManifest has a controlled JSON-compatible Serialize implementation"
)]
pub(crate) fn serialize_contract_failure_runtime(manifest: &BenchmarkManifest) -> Value {
    match runtime_config_from_manifest(manifest) {
        Ok(runtime) => serialize_runtime_metadata(&runtime, None),
        Err(_) => serde_json::to_value(&manifest.runtime)
            .expect("runtime manifest should serialize for contract failure artifact"),
    }
}

pub(crate) struct ContractFailureClassification {
    pub(crate) code: &'static str,
    pub(crate) recommended_action: &'static str,
}

pub(crate) fn classify_contract_failure(
    manifest: &BenchmarkManifest,
    message: &str,
) -> ContractFailureClassification {
    if message.contains("runtime.backend_adapter is required") {
        return ContractFailureClassification {
            code: "llama_backend_adapter_required",
            recommended_action: "Set runtime.backend_adapter to a supported llama.cpp adapter before running delegated benchmark manifests.",
        };
    }

    if message.contains("runtime.backend_adapter kind does not match selected_backend") {
        return ContractFailureClassification {
            code: "backend_adapter_kind_mismatch",
            recommended_action: "Align runtime.selected_backend and runtime.backend_adapter.kind so they describe the same delegated backend.",
        };
    }

    if message.contains("runtime.backend_adapter must be omitted when selected_backend is mlx") {
        return ContractFailureClassification {
            code: "mlx_backend_adapter_forbidden",
            recommended_action: "Remove runtime.backend_adapter when selected_backend is mlx.",
        };
    }

    if message.contains("invalid benchmark runtime backend-resolution contract") {
        return ContractFailureClassification {
            code: "runtime_backend_resolution_contract_invalid",
            recommended_action: "Fix selected_backend, support_tier, resolution_policy, and fallback_reason so they satisfy the benchmark runtime contract.",
        };
    }

    if manifest.runtime.support_tier == SupportTier::LlamaCpp
        && matches!(
            manifest.runtime.backend_adapter,
            Some(BackendAdapterManifest::LlamaCppCli { .. })
        )
    {
        return ContractFailureClassification {
            code: "llama_cpp_cli_token_workload_unsupported",
            recommended_action: "Use runtime.backend_adapter.kind=llama_cpp_server_completion for token-based llama.cpp benchmark manifests.",
        };
    }

    if message
        .contains("blocking llama.cpp benchmark adapters currently support scenario manifests only")
    {
        return ContractFailureClassification {
            code: "llama_cpp_blocking_replay_unsupported",
            recommended_action: "Use the llama.cpp server adapter for non-MLX llama.cpp workloads, or switch to an MLX-mode manifest for ax-engine MLX inference.",
        };
    }

    if message
        .contains("blocking llama.cpp benchmark adapters currently require shape.concurrency=1")
    {
        return ContractFailureClassification {
            code: "llama_cpp_blocking_concurrency_unsupported",
            recommended_action: "Reduce shape.concurrency to 1 for blocking llama.cpp benchmark adapters, or use the stepwise llama.cpp server adapter for multi-request delegated workloads.",
        };
    }

    if message.contains("blocking llama.cpp benchmark adapters do not support require_prefix_reuse")
    {
        return ContractFailureClassification {
            code: "llama_cpp_blocking_prefix_reuse_unsupported",
            recommended_action: "Disable require_prefix_reuse for blocking delegated adapters, or use the stepwise llama.cpp server adapter for delegated prompt-cache benchmark coverage.",
        };
    }

    if message.contains("scenario manifest must contain a shape object") {
        return ContractFailureClassification {
            code: "scenario_shape_required",
            recommended_action: "Add a valid shape object to the scenario manifest before rerunning ax-engine-bench scenario.",
        };
    }

    if message.contains("submit event missing request_id")
        || message.contains("cancel event missing request_id")
        || message.contains("submit event missing prompt_ref")
    {
        return ContractFailureClassification {
            code: "replay_event_schema_invalid",
            recommended_action: "Repair the replay events so submit/cancel entries include the required request identifiers and prompt references.",
        };
    }

    let _ = message;
    ContractFailureClassification {
        code: "contract_validation_failed",
        recommended_action: "Review the manifest/runtime contract and rerun with a supported workload shape or runtime configuration.",
    }
}

pub(crate) fn contract_failure_tool_mode(manifest: &BenchmarkManifest) -> &'static str {
    if let Ok(runtime) = runtime_config_from_manifest(manifest) {
        runtime.tool_mode()
    } else if matches!(
        manifest.runtime.backend_adapter.as_ref(),
        Some(adapter) if adapter.supports_stepwise_benchmark()
    ) {
        "llama_cpp_stepwise_runtime"
    } else {
        "llama_cpp_blocking_runtime"
    }
}

#[allow(
    clippy::expect_used,
    reason = "the mutated fields are object literals created in this function"
)]
pub(crate) fn build_regression_json(
    baseline_metrics: &Value,
    candidate_metrics: &Value,
    baseline_environment: &Value,
    candidate_environment: &Value,
    trusted_baseline: Option<&Value>,
) -> Result<Value, CliError> {
    let baseline_run_id = baseline_metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("baseline");
    let candidate_run_id = candidate_metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("candidate");
    let baseline_prefix_cache_path =
        nested_value(baseline_environment, &["route", "prefix_cache_path"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let candidate_prefix_cache_path =
        nested_value(candidate_environment, &["route", "prefix_cache_path"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let baseline_prefix_cache_evidence =
        nested_value(baseline_environment, &["route", "prefix_cache_evidence"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let candidate_prefix_cache_evidence =
        nested_value(candidate_environment, &["route", "prefix_cache_evidence"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let baseline_prefix_reuse_provenance =
        nested_value(baseline_environment, &["route", "prefix_reuse_provenance"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let candidate_prefix_reuse_provenance =
        nested_value(candidate_environment, &["route", "prefix_reuse_provenance"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let baseline_route = baseline_environment
        .get("route")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let candidate_route = candidate_environment
        .get("route")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let baseline_execution_semantics =
        route_execution_semantics_from_environment_json(baseline_environment);
    let candidate_execution_semantics =
        route_execution_semantics_from_environment_json(candidate_environment);
    let baseline_metal_numeric_scaffold_only = route_flag_from_json(
        &baseline_route,
        "metal_numeric_scaffold_only",
        &["metal_dispatch_numeric_scaffold_only"],
    );
    let candidate_metal_numeric_scaffold_only = route_flag_from_json(
        &candidate_route,
        "metal_numeric_scaffold_only",
        &["metal_dispatch_numeric_scaffold_only"],
    );
    let baseline_metal_real_model_forward = route_flag_from_json(
        &baseline_route,
        "metal_real_model_forward",
        &["metal_dispatch_real_model_forward"],
    );
    let candidate_metal_real_model_forward = route_flag_from_json(
        &candidate_route,
        "metal_real_model_forward",
        &["metal_dispatch_real_model_forward"],
    );
    let baseline_metal_complete_model_forward_supported = route_flag_from_json(
        &baseline_route,
        "metal_complete_model_forward_supported",
        &[
            "metal_dispatch_complete_model_forward_supported",
            "metal_dispatch_runtime_complete_model_forward_supported",
        ],
    );
    let candidate_metal_complete_model_forward_supported = route_flag_from_json(
        &candidate_route,
        "metal_complete_model_forward_supported",
        &[
            "metal_dispatch_complete_model_forward_supported",
            "metal_dispatch_runtime_complete_model_forward_supported",
        ],
    );
    let baseline_metal_model_conditioned_inputs = route_flag_from_json(
        &baseline_route,
        "metal_model_conditioned_inputs",
        &[
            "metal_dispatch_model_conditioned_inputs",
            "metal_dispatch_runtime_model_conditioned_inputs",
        ],
    );
    let candidate_metal_model_conditioned_inputs = route_flag_from_json(
        &candidate_route,
        "metal_model_conditioned_inputs",
        &[
            "metal_dispatch_model_conditioned_inputs",
            "metal_dispatch_runtime_model_conditioned_inputs",
        ],
    );
    let baseline_mlx_metal_prefix_layers_attention = route_flag_from_json(
        &baseline_route,
        "mlx_metal_prefix_layers_attention",
        &["metal_dispatch_prefix_layers_native_attention"],
    );
    let candidate_mlx_metal_prefix_layers_attention = route_flag_from_json(
        &candidate_route,
        "mlx_metal_prefix_layers_attention",
        &["metal_dispatch_prefix_layers_native_attention"],
    );
    let baseline_metal_prefix_layers_cpu_reference = route_flag_from_json(
        &baseline_route,
        "metal_prefix_layers_cpu_reference",
        &["metal_dispatch_prefix_layers_cpu_reference"],
    );
    let candidate_metal_prefix_layers_cpu_reference = route_flag_from_json(
        &candidate_route,
        "metal_prefix_layers_cpu_reference",
        &["metal_dispatch_prefix_layers_cpu_reference"],
    );
    let baseline_mlx_metal_prefix_dispatch_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_prefix_dispatch_count",
        &["metal_dispatch_prefix_native_dispatch_count"],
    );
    let candidate_mlx_metal_prefix_dispatch_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_prefix_dispatch_count",
        &["metal_dispatch_prefix_native_dispatch_count"],
    );
    let baseline_metal_prefix_cpu_reference_dispatch_count = route_count_from_json(
        &baseline_route,
        "metal_prefix_cpu_reference_dispatch_count",
        &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
    );
    let candidate_metal_prefix_cpu_reference_dispatch_count = route_count_from_json(
        &candidate_route,
        "metal_prefix_cpu_reference_dispatch_count",
        &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
    );
    let baseline_mlx_metal_projection_f32_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_projection_f32_binding_count",
        &["metal_dispatch_native_projection_f32_binding_count"],
    );
    let candidate_mlx_metal_projection_f32_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_projection_f32_binding_count",
        &["metal_dispatch_native_projection_f32_binding_count"],
    );
    let baseline_mlx_metal_projection_f16_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_projection_f16_binding_count",
        &["metal_dispatch_native_projection_f16_binding_count"],
    );
    let candidate_mlx_metal_projection_f16_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_projection_f16_binding_count",
        &["metal_dispatch_native_projection_f16_binding_count"],
    );
    let baseline_mlx_metal_projection_bf16_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_projection_bf16_binding_count",
        &["metal_dispatch_native_projection_bf16_binding_count"],
    );
    let candidate_mlx_metal_projection_bf16_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_projection_bf16_binding_count",
        &["metal_dispatch_native_projection_bf16_binding_count"],
    );
    let baseline_mlx_metal_projection_unsupported_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_projection_unsupported_binding_count",
        &["metal_dispatch_native_projection_unsupported_binding_count"],
    );
    let candidate_mlx_metal_projection_unsupported_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_projection_unsupported_binding_count",
        &["metal_dispatch_native_projection_unsupported_binding_count"],
    );
    let baseline_mlx_metal_rms_norm_f32_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_rms_norm_f32_binding_count",
        &["metal_dispatch_native_rms_norm_f32_binding_count"],
    );
    let candidate_mlx_metal_rms_norm_f32_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_rms_norm_f32_binding_count",
        &["metal_dispatch_native_rms_norm_f32_binding_count"],
    );
    let baseline_mlx_metal_rms_norm_f16_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_rms_norm_f16_binding_count",
        &["metal_dispatch_native_rms_norm_f16_binding_count"],
    );
    let candidate_mlx_metal_rms_norm_f16_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_rms_norm_f16_binding_count",
        &["metal_dispatch_native_rms_norm_f16_binding_count"],
    );
    let baseline_mlx_metal_rms_norm_bf16_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_rms_norm_bf16_binding_count",
        &["metal_dispatch_native_rms_norm_bf16_binding_count"],
    );
    let candidate_mlx_metal_rms_norm_bf16_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_rms_norm_bf16_binding_count",
        &["metal_dispatch_native_rms_norm_bf16_binding_count"],
    );
    let baseline_mlx_metal_rms_norm_unsupported_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_rms_norm_unsupported_binding_count",
        &["metal_dispatch_native_rms_norm_unsupported_binding_count"],
    );
    let candidate_mlx_metal_rms_norm_unsupported_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_rms_norm_unsupported_binding_count",
        &["metal_dispatch_native_rms_norm_unsupported_binding_count"],
    );
    let baseline_metal_direct_decode_tokens = route_flag_from_json(
        &baseline_route,
        "metal_direct_decode_tokens",
        &["metal_dispatch_direct_decode_tokens"],
    );
    let candidate_metal_direct_decode_tokens = route_flag_from_json(
        &candidate_route,
        "metal_direct_decode_tokens",
        &["metal_dispatch_direct_decode_tokens"],
    );
    let baseline_metal_direct_decode_batching_opportunity_observed = route_flag_from_json(
        &baseline_route,
        "metal_direct_decode_batching_opportunity_observed",
        &["metal_direct_decode_batching_opportunity_observed"],
    );
    let candidate_metal_direct_decode_batching_opportunity_observed = route_flag_from_json(
        &candidate_route,
        "metal_direct_decode_batching_opportunity_observed",
        &["metal_direct_decode_batching_opportunity_observed"],
    );
    let baseline_metal_direct_decode_model_bound_ffn = route_flag_from_json(
        &baseline_route,
        "metal_direct_decode_model_bound_ffn",
        &["metal_dispatch_direct_decode_model_bound_ffn"],
    );
    let candidate_metal_direct_decode_model_bound_ffn = route_flag_from_json(
        &candidate_route,
        "metal_direct_decode_model_bound_ffn",
        &["metal_dispatch_direct_decode_model_bound_ffn"],
    );
    let baseline_metal_direct_decode_batched_logits_group_count = route_count_from_json(
        &baseline_route,
        "metal_direct_decode_batched_logits_group_count",
        &["metal_dispatch_direct_decode_batched_logits_group_count"],
    );
    let candidate_metal_direct_decode_batched_logits_group_count = route_count_from_json(
        &candidate_route,
        "metal_direct_decode_batched_logits_group_count",
        &["metal_dispatch_direct_decode_batched_logits_group_count"],
    );
    let baseline_metal_direct_decode_batched_logits_token_count = route_count_from_json(
        &baseline_route,
        "metal_direct_decode_batched_logits_token_count",
        &["metal_dispatch_direct_decode_batched_logits_token_count"],
    );
    let candidate_metal_direct_decode_batched_logits_token_count = route_count_from_json(
        &candidate_route,
        "metal_direct_decode_batched_logits_token_count",
        &["metal_dispatch_direct_decode_batched_logits_token_count"],
    );
    let baseline_metal_direct_decode_batched_group_fallback_count = route_count_from_json(
        &baseline_route,
        "metal_direct_decode_batched_group_fallback_count",
        &["metal_dispatch_direct_decode_batched_group_fallback_count"],
    );
    let candidate_metal_direct_decode_batched_group_fallback_count = route_count_from_json(
        &candidate_route,
        "metal_direct_decode_batched_group_fallback_count",
        &["metal_dispatch_direct_decode_batched_group_fallback_count"],
    );
    let baseline_metal_direct_decode_batched_group_fallback_token_count = route_count_from_json(
        &baseline_route,
        "metal_direct_decode_batched_group_fallback_token_count",
        &["metal_dispatch_direct_decode_batched_group_fallback_token_count"],
    );
    let candidate_metal_direct_decode_batched_group_fallback_token_count = route_count_from_json(
        &candidate_route,
        "metal_direct_decode_batched_group_fallback_token_count",
        &["metal_dispatch_direct_decode_batched_group_fallback_token_count"],
    );
    let baseline_metal_real_model_tensor_inputs = route_flag_from_json(
        &baseline_route,
        "metal_real_model_tensor_inputs",
        &[
            "metal_dispatch_real_model_tensor_inputs",
            "metal_dispatch_runtime_real_model_tensor_inputs",
        ],
    );
    let candidate_metal_real_model_tensor_inputs = route_flag_from_json(
        &candidate_route,
        "metal_real_model_tensor_inputs",
        &[
            "metal_dispatch_real_model_tensor_inputs",
            "metal_dispatch_runtime_real_model_tensor_inputs",
        ],
    );
    let baseline_metal_model_artifacts_validated = route_flag_from_json(
        &baseline_route,
        "metal_model_artifacts_validated",
        &["metal_dispatch_model_artifacts_validated"],
    );
    let candidate_metal_model_artifacts_validated = route_flag_from_json(
        &candidate_route,
        "metal_model_artifacts_validated",
        &["metal_dispatch_model_artifacts_validated"],
    );
    let baseline_backend_reported_cached_prompt_tokens = nested_value(
        baseline_environment,
        &["route", "backend_reported_cached_prompt_tokens"],
    )
    .cloned();
    let candidate_backend_reported_cached_prompt_tokens = nested_value(
        candidate_environment,
        &["route", "backend_reported_cached_prompt_tokens"],
    )
    .cloned();
    let baseline_tool_mode =
        string_at_path_or_unknown(baseline_environment, &["software", "tool_mode"]);
    let candidate_tool_mode =
        string_at_path_or_unknown(candidate_environment, &["software", "tool_mode"]);
    let baseline_selected_backend =
        string_at_path_or_unknown(baseline_environment, &["runtime", "selected_backend"]);
    let candidate_selected_backend =
        string_at_path_or_unknown(candidate_environment, &["runtime", "selected_backend"]);
    let baseline_support_tier =
        string_at_path_or_unknown(baseline_environment, &["runtime", "support_tier"]);
    let candidate_support_tier =
        string_at_path_or_unknown(candidate_environment, &["runtime", "support_tier"]);
    let baseline_resolution_policy =
        string_at_path_or_unknown(baseline_environment, &["runtime", "resolution_policy"]);
    let candidate_resolution_policy =
        string_at_path_or_unknown(candidate_environment, &["runtime", "resolution_policy"]);
    let baseline_backend_adapter =
        nested_value(baseline_environment, &["runtime", "backend_adapter"]).cloned();
    let candidate_backend_adapter =
        nested_value(candidate_environment, &["runtime", "backend_adapter"]).cloned();
    let baseline_llama_cpp_preset =
        nested_value(baseline_environment, &["runtime", "llama_cpp_preset"]).cloned();
    let candidate_llama_cpp_preset =
        nested_value(candidate_environment, &["runtime", "llama_cpp_preset"]).cloned();
    let baseline_mlx_dense_dequantized_source = nested_value(baseline_environment, &["runtime"])
        .is_some_and(native_dense_dequantized_source_from_runtime_json);
    let candidate_mlx_dense_dequantized_source = nested_value(candidate_environment, &["runtime"])
        .is_some_and(native_dense_dequantized_source_from_runtime_json);
    let baseline_mlx_metal_readiness = mlx_metal_readiness_from_route_json(
        &baseline_tool_mode,
        &baseline_selected_backend,
        &baseline_route,
        baseline_mlx_dense_dequantized_source,
    );
    let candidate_mlx_metal_readiness = mlx_metal_readiness_from_route_json(
        &candidate_tool_mode,
        &candidate_selected_backend,
        &candidate_route,
        candidate_mlx_dense_dequantized_source,
    );
    let compare_result = compare_result_label(&baseline_tool_mode, &baseline_execution_semantics);
    let mut runtime = json!({
        "tool_mode": baseline_tool_mode,
        "selected_backend": baseline_selected_backend,
        "support_tier": baseline_support_tier,
        "resolution_policy": baseline_resolution_policy
    });
    if let Some(backend_adapter) = baseline_backend_adapter.clone() {
        runtime
            .as_object_mut()
            .expect("regression runtime should serialize as object")
            .insert("backend_adapter".to_string(), backend_adapter);
    }
    if let Some(llama_cpp_preset) = baseline_llama_cpp_preset.clone() {
        runtime
            .as_object_mut()
            .expect("regression runtime should serialize as object")
            .insert("llama_cpp_preset".to_string(), llama_cpp_preset);
    }
    let mut contract_runtime = json!({
        "selected_backend": {
            "baseline": baseline_selected_backend,
            "candidate": candidate_selected_backend
        },
        "support_tier": {
            "baseline": baseline_support_tier,
            "candidate": candidate_support_tier
        },
        "resolution_policy": {
            "baseline": baseline_resolution_policy,
            "candidate": candidate_resolution_policy
        }
    });
    if baseline_backend_adapter.is_some() || candidate_backend_adapter.is_some() {
        contract_runtime
            .as_object_mut()
            .expect("regression contract runtime should serialize as object")
            .insert(
                "backend_adapter".to_string(),
                json!({
                    "baseline": baseline_backend_adapter,
                    "candidate": candidate_backend_adapter
                }),
            );
    }
    if baseline_llama_cpp_preset.is_some() || candidate_llama_cpp_preset.is_some() {
        contract_runtime
            .as_object_mut()
            .expect("regression contract runtime should serialize as object")
            .insert(
                "llama_cpp_preset".to_string(),
                json!({
                    "baseline": baseline_llama_cpp_preset,
                    "candidate": candidate_llama_cpp_preset
                }),
            );
    }

    let mut regression = json!({
        "schema_version": "ax.engine_bench.regression.v1",
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "runtime": runtime,
        "informational_diff": {
            "runtime_tuning": runtime_tuning_informational_diff_json(
                baseline_environment,
                candidate_environment
            )
        },
        "comparison": {
            "ttft_ms_pct": percentage_delta(metric_number(baseline_metrics, "ttft_ms")?, metric_number(candidate_metrics, "ttft_ms")?),
            "decode_tok_s_pct": percentage_delta(metric_number(baseline_metrics, "decode_tok_s")?, metric_number(candidate_metrics, "decode_tok_s")?),
            "memory_peak_mb_pct": percentage_delta(metric_number(baseline_metrics, "memory_peak_mb")?, metric_number(candidate_metrics, "memory_peak_mb")?),
            "prefix_hit_rate_pct": percentage_delta(metric_number(baseline_metrics, "prefix_hit_rate")?, metric_number(candidate_metrics, "prefix_hit_rate")?)
        },
        "gates": {
            "correctness": "inspect",
            "determinism": "inspect",
            "benchmark_contract": "pass",
            "performance": "inspect"
        },
        "summary": {
            "result": compare_result,
            "requires_human_review": true,
            "tool_mode": baseline_tool_mode,
            "selected_backend": baseline_selected_backend,
            "support_tier": baseline_support_tier,
            "resolution_policy": baseline_resolution_policy,
            "execution_semantics": baseline_execution_semantics,
            "metal_numeric_scaffold_only": baseline_metal_numeric_scaffold_only,
            "metal_complete_model_forward_supported": baseline_metal_complete_model_forward_supported,
            "metal_real_model_forward": baseline_metal_real_model_forward,
            "metal_model_conditioned_inputs": baseline_metal_model_conditioned_inputs,
            "mlx_metal_prefix_layers_attention": baseline_mlx_metal_prefix_layers_attention,
            "metal_prefix_layers_cpu_reference": baseline_metal_prefix_layers_cpu_reference,
            "mlx_metal_prefix_dispatch_count": baseline_mlx_metal_prefix_dispatch_count,
            "metal_prefix_cpu_reference_dispatch_count": baseline_metal_prefix_cpu_reference_dispatch_count,
            "mlx_metal_projection_f32_binding_count": baseline_mlx_metal_projection_f32_binding_count,
            "mlx_metal_projection_f16_binding_count": baseline_mlx_metal_projection_f16_binding_count,
            "mlx_metal_projection_bf16_binding_count": baseline_mlx_metal_projection_bf16_binding_count,
            "mlx_metal_projection_unsupported_binding_count": baseline_mlx_metal_projection_unsupported_binding_count,
            "metal_direct_decode_tokens": baseline_metal_direct_decode_tokens,
            "metal_direct_decode_batching_opportunity_observed": baseline_metal_direct_decode_batching_opportunity_observed,
            "metal_direct_decode_model_bound_ffn": baseline_metal_direct_decode_model_bound_ffn,
            "metal_direct_decode_batched_logits_group_count": baseline_metal_direct_decode_batched_logits_group_count,
            "metal_direct_decode_batched_logits_token_count": baseline_metal_direct_decode_batched_logits_token_count,
            "metal_direct_decode_batched_group_fallback_count": baseline_metal_direct_decode_batched_group_fallback_count,
            "metal_direct_decode_batched_group_fallback_token_count": baseline_metal_direct_decode_batched_group_fallback_token_count,
            "metal_real_model_tensor_inputs": baseline_metal_real_model_tensor_inputs,
            "metal_model_artifacts_validated": baseline_metal_model_artifacts_validated,
            "mlx_metal_rms_norm_f32_binding_count": baseline_mlx_metal_rms_norm_f32_binding_count,
            "mlx_metal_rms_norm_f16_binding_count": baseline_mlx_metal_rms_norm_f16_binding_count,
            "mlx_metal_rms_norm_bf16_binding_count": baseline_mlx_metal_rms_norm_bf16_binding_count,
            "mlx_metal_rms_norm_unsupported_binding_count": baseline_mlx_metal_rms_norm_unsupported_binding_count,
            "mlx_metal_readiness": baseline_mlx_metal_readiness.status,
            "mlx_metal_hot_path_cpu_fallback_free": baseline_mlx_metal_readiness.hot_path_cpu_fallback_free,
            "mlx_metal_batched_direct_decode_logits_ready": baseline_mlx_metal_readiness.batched_direct_decode_logits_ready,
            "mlx_metal_prefix_min_dispatch_share": baseline_mlx_metal_readiness.prefix_min_mlx_metal_dispatch_share,
            "mlx_metal_direct_decode_min_dispatch_share": baseline_mlx_metal_readiness.direct_decode_min_mlx_metal_dispatch_share,
            "mlx_metal_readiness_blockers": baseline_mlx_metal_readiness.blockers,
            "prefix_cache_path": baseline_prefix_cache_path,
            "prefix_cache_evidence": baseline_prefix_cache_evidence,
            "prefix_reuse_provenance": baseline_prefix_reuse_provenance
        },
        "contract": {
            "tool_mode": {
                "baseline": baseline_tool_mode,
                "candidate": candidate_tool_mode
            },
            "runtime": contract_runtime,
            "execution_semantics": {
                "baseline": baseline_execution_semantics,
                "candidate": candidate_execution_semantics
            },
            "metal_numeric_scaffold_only": {
                "baseline": baseline_metal_numeric_scaffold_only,
                "candidate": candidate_metal_numeric_scaffold_only
            },
            "metal_complete_model_forward_supported": {
                "baseline": baseline_metal_complete_model_forward_supported,
                "candidate": candidate_metal_complete_model_forward_supported
            },
            "metal_real_model_forward": {
                "baseline": baseline_metal_real_model_forward,
                "candidate": candidate_metal_real_model_forward
            },
            "metal_model_conditioned_inputs": {
                "baseline": baseline_metal_model_conditioned_inputs,
                "candidate": candidate_metal_model_conditioned_inputs
            },
            "mlx_metal_prefix_layers_attention": {
                "baseline": baseline_mlx_metal_prefix_layers_attention,
                "candidate": candidate_mlx_metal_prefix_layers_attention
            },
            "metal_prefix_layers_cpu_reference": {
                "baseline": baseline_metal_prefix_layers_cpu_reference,
                "candidate": candidate_metal_prefix_layers_cpu_reference
            },
            "mlx_metal_prefix_dispatch_count": {
                "baseline": baseline_mlx_metal_prefix_dispatch_count,
                "candidate": candidate_mlx_metal_prefix_dispatch_count
            },
            "metal_prefix_cpu_reference_dispatch_count": {
                "baseline": baseline_metal_prefix_cpu_reference_dispatch_count,
                "candidate": candidate_metal_prefix_cpu_reference_dispatch_count
            },
            "mlx_metal_projection_f32_binding_count": {
                "baseline": baseline_mlx_metal_projection_f32_binding_count,
                "candidate": candidate_mlx_metal_projection_f32_binding_count
            },
            "mlx_metal_projection_f16_binding_count": {
                "baseline": baseline_mlx_metal_projection_f16_binding_count,
                "candidate": candidate_mlx_metal_projection_f16_binding_count
            },
            "mlx_metal_projection_bf16_binding_count": {
                "baseline": baseline_mlx_metal_projection_bf16_binding_count,
                "candidate": candidate_mlx_metal_projection_bf16_binding_count
            },
            "mlx_metal_projection_unsupported_binding_count": {
                "baseline": baseline_mlx_metal_projection_unsupported_binding_count,
                "candidate": candidate_mlx_metal_projection_unsupported_binding_count
            },
            "metal_direct_decode_tokens": {
                "baseline": baseline_metal_direct_decode_tokens,
                "candidate": candidate_metal_direct_decode_tokens
            },
            "metal_direct_decode_batching_opportunity_observed": {
                "baseline": baseline_metal_direct_decode_batching_opportunity_observed,
                "candidate": candidate_metal_direct_decode_batching_opportunity_observed
            },
            "metal_direct_decode_model_bound_ffn": {
                "baseline": baseline_metal_direct_decode_model_bound_ffn,
                "candidate": candidate_metal_direct_decode_model_bound_ffn
            },
            "metal_direct_decode_batched_logits_group_count": {
                "baseline": baseline_metal_direct_decode_batched_logits_group_count,
                "candidate": candidate_metal_direct_decode_batched_logits_group_count
            },
            "metal_direct_decode_batched_logits_token_count": {
                "baseline": baseline_metal_direct_decode_batched_logits_token_count,
                "candidate": candidate_metal_direct_decode_batched_logits_token_count
            },
            "metal_direct_decode_batched_group_fallback_count": {
                "baseline": baseline_metal_direct_decode_batched_group_fallback_count,
                "candidate": candidate_metal_direct_decode_batched_group_fallback_count
            },
            "metal_direct_decode_batched_group_fallback_token_count": {
                "baseline": baseline_metal_direct_decode_batched_group_fallback_token_count,
                "candidate": candidate_metal_direct_decode_batched_group_fallback_token_count
            },
            "metal_real_model_tensor_inputs": {
                "baseline": baseline_metal_real_model_tensor_inputs,
                "candidate": candidate_metal_real_model_tensor_inputs
            },
            "metal_model_artifacts_validated": {
                "baseline": baseline_metal_model_artifacts_validated,
                "candidate": candidate_metal_model_artifacts_validated
            },
            "mlx_metal_rms_norm_f32_binding_count": {
                "baseline": baseline_mlx_metal_rms_norm_f32_binding_count,
                "candidate": candidate_mlx_metal_rms_norm_f32_binding_count
            },
            "mlx_metal_rms_norm_f16_binding_count": {
                "baseline": baseline_mlx_metal_rms_norm_f16_binding_count,
                "candidate": candidate_mlx_metal_rms_norm_f16_binding_count
            },
            "mlx_metal_rms_norm_bf16_binding_count": {
                "baseline": baseline_mlx_metal_rms_norm_bf16_binding_count,
                "candidate": candidate_mlx_metal_rms_norm_bf16_binding_count
            },
            "mlx_metal_rms_norm_unsupported_binding_count": {
                "baseline": baseline_mlx_metal_rms_norm_unsupported_binding_count,
                "candidate": candidate_mlx_metal_rms_norm_unsupported_binding_count
            },
            "prefix_cache_path": {
                "baseline": baseline_prefix_cache_path,
                "candidate": candidate_prefix_cache_path
            },
            "prefix_cache_evidence": {
                "baseline": baseline_prefix_cache_evidence,
                "candidate": candidate_prefix_cache_evidence
            },
            "prefix_reuse_provenance": {
                "baseline": baseline_prefix_reuse_provenance,
                "candidate": candidate_prefix_reuse_provenance
            }
        }
    });

    regression
        .as_object_mut()
        .expect("regression artifact should serialize as object")
        .insert(
            "readiness".to_string(),
            json!({
                "baseline": baseline_mlx_metal_readiness.to_json(),
                "candidate": candidate_mlx_metal_readiness.to_json()
            }),
        );

    for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
        let baseline_value = route_counter_from_json(&baseline_route, key);
        let candidate_value = route_counter_from_json(&candidate_route, key);
        regression
            .get_mut("summary")
            .and_then(Value::as_object_mut)
            .expect("regression summary should serialize as object")
            .insert(key.to_string(), json!(baseline_value));
        regression
            .get_mut("contract")
            .and_then(Value::as_object_mut)
            .expect("regression contract should serialize as object")
            .insert(
                key.to_string(),
                json!({
                    "baseline": baseline_value,
                    "candidate": candidate_value
                }),
            );
    }

    if let Some(value) = baseline_backend_reported_cached_prompt_tokens.clone() {
        regression
            .get_mut("summary")
            .and_then(Value::as_object_mut)
            .expect("regression summary should serialize as object")
            .insert("backend_reported_cached_prompt_tokens".to_string(), value);
    }

    if baseline_backend_reported_cached_prompt_tokens.is_some()
        || candidate_backend_reported_cached_prompt_tokens.is_some()
    {
        regression
            .get_mut("contract")
            .and_then(Value::as_object_mut)
            .expect("regression contract should serialize as object")
            .insert(
                "backend_reported_cached_prompt_tokens".to_string(),
                json!({
                    "baseline": baseline_backend_reported_cached_prompt_tokens,
                    "candidate": candidate_backend_reported_cached_prompt_tokens
                }),
            );
    }

    if let Some(trusted_baseline) = trusted_baseline {
        regression
            .as_object_mut()
            .expect("regression artifact should serialize as object")
            .insert(
                "trusted_baseline".to_string(),
                json!({
                    "name": trusted_baseline.get("name").cloned().unwrap_or(Value::String("unknown".to_string())),
                    "slug": trusted_baseline.get("slug").cloned().unwrap_or(Value::String("unknown".to_string())),
                    "created_at_unix_s": trusted_baseline
                        .get("created_at_unix_s")
                        .cloned()
                        .unwrap_or(Value::Null),
                    "source_result_dir": trusted_baseline
                        .get("source_result_dir")
                        .cloned()
                        .unwrap_or(Value::String("unknown".to_string())),
                    "source_run_id": trusted_baseline
                        .get("source_run_id")
                        .cloned()
                        .unwrap_or(Value::String("unknown".to_string()))
                }),
            );
    }

    Ok(regression)
}

pub(crate) fn build_compare_summary_markdown(
    baseline_metrics: &Value,
    candidate_metrics: &Value,
    regression_json: &Value,
    trusted_baseline: Option<&Value>,
) -> String {
    let baseline_id = baseline_metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("baseline");
    let candidate_id = candidate_metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("candidate");
    let ttft = regression_json
        .get("comparison")
        .and_then(|comparison| comparison.get("ttft_ms_pct"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let decode = regression_json
        .get("comparison")
        .and_then(|comparison| comparison.get("decode_tok_s_pct"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let memory = regression_json
        .get("comparison")
        .and_then(|comparison| comparison.get("memory_peak_mb_pct"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let prefix = regression_json
        .get("comparison")
        .and_then(|comparison| comparison.get("prefix_hit_rate_pct"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let prefix_reuse_provenance = regression_json
        .get("summary")
        .and_then(|summary| summary.get("prefix_reuse_provenance"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_path = regression_json
        .get("summary")
        .and_then(|summary| summary.get("prefix_cache_path"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_evidence = regression_json
        .get("summary")
        .and_then(|summary| summary.get("prefix_cache_evidence"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let execution_semantics = regression_json
        .get("summary")
        .and_then(|summary| summary.get("execution_semantics"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let metal_numeric_scaffold_only = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_numeric_scaffold_only"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_complete_model_forward_supported = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_complete_model_forward_supported"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_real_model_forward = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_real_model_forward"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_model_conditioned_inputs = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_model_conditioned_inputs"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mlx_metal_prefix_layers_attention = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_prefix_layers_attention"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_prefix_layers_cpu_reference = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_prefix_layers_cpu_reference"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mlx_metal_prefix_dispatch_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_prefix_dispatch_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_prefix_cpu_reference_dispatch_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_prefix_cpu_reference_dispatch_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_projection_f32_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_projection_f32_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_projection_f16_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_projection_f16_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_projection_bf16_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_projection_bf16_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_projection_unsupported_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_projection_unsupported_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_direct_decode_tokens = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_tokens"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_direct_decode_batching_opportunity_observed = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batching_opportunity_observed"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_direct_decode_model_bound_ffn = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_model_bound_ffn"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_direct_decode_batched_logits_group_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batched_logits_group_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_direct_decode_batched_logits_token_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batched_logits_token_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_direct_decode_batched_group_fallback_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batched_group_fallback_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_direct_decode_batched_group_fallback_token_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batched_group_fallback_token_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_real_model_tensor_inputs = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_real_model_tensor_inputs"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_model_artifacts_validated = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_model_artifacts_validated"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mlx_metal_rms_norm_f32_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_rms_norm_f32_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_rms_norm_f16_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_rms_norm_f16_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_rms_norm_bf16_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_rms_norm_bf16_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_rms_norm_unsupported_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_rms_norm_unsupported_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let baseline_mlx_metal_readiness = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let baseline_mlx_metal_hot_path_cpu_fallback_free = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("hot_path_cpu_fallback_free"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let baseline_mlx_metal_batched_direct_decode_logits_ready = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("batched_direct_decode_logits_ready"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let baseline_mlx_metal_prefix_min_dispatch_share = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("prefix_min_mlx_metal_dispatch_share"))
        .and_then(Value::as_f64);
    let baseline_mlx_metal_direct_decode_min_dispatch_share = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("direct_decode_min_mlx_metal_dispatch_share"))
        .and_then(Value::as_f64);
    let baseline_mlx_metal_readiness_blockers = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("blockers"))
        .map(json_value_label)
        .unwrap_or_else(|| "[]".to_string());
    let candidate_mlx_metal_readiness = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let candidate_mlx_metal_hot_path_cpu_fallback_free = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("hot_path_cpu_fallback_free"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let candidate_mlx_metal_batched_direct_decode_logits_ready = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("batched_direct_decode_logits_ready"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let candidate_mlx_metal_prefix_min_dispatch_share = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("prefix_min_mlx_metal_dispatch_share"))
        .and_then(Value::as_f64);
    let candidate_mlx_metal_direct_decode_min_dispatch_share = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("direct_decode_min_mlx_metal_dispatch_share"))
        .and_then(Value::as_f64);
    let candidate_mlx_metal_readiness_blockers = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("blockers"))
        .map(json_value_label)
        .unwrap_or_else(|| "[]".to_string());
    let backend_reported_cached_prompt_tokens = regression_json
        .get("summary")
        .and_then(|summary| summary.get("backend_reported_cached_prompt_tokens"))
        .map(json_value_label)
        .unwrap_or_else(|| "none".to_string());
    let mlx_kv_cache_lines = mlx_kv_cache_regression_markdown_lines(regression_json);
    let compare_mode = explicit_or_inferred_compare_mode(regression_json);
    let tool_mode = explicit_or_inferred_tool_mode(
        regression_json,
        &["summary", "tool_mode"],
        "engine_bringup_runtime",
    );
    let selected_backend = regression_json
        .get("summary")
        .and_then(|summary| summary.get("selected_backend"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let support_tier = regression_json
        .get("summary")
        .and_then(|summary| summary.get("support_tier"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let resolution_policy = regression_json
        .get("summary")
        .and_then(|summary| summary.get("resolution_policy"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let backend_adapter = regression_json
        .get("runtime")
        .and_then(|runtime| runtime.get("backend_adapter"))
        .map(json_value_label)
        .unwrap_or_else(|| "none".to_string());
    let trusted_baseline_line = trusted_baseline
        .and_then(|baseline| baseline.get("name"))
        .and_then(Value::as_str)
        .map(|name| format!("- trusted_baseline: `{name}`\n"))
        .unwrap_or_default();
    let summary_note = compare_summary_note(&tool_mode, execution_semantics);
    let baseline_mlx_metal_prefix_min_dispatch_share =
        mlx_metal_dispatch_share_label(baseline_mlx_metal_prefix_min_dispatch_share);
    let baseline_mlx_metal_direct_decode_min_dispatch_share =
        mlx_metal_dispatch_share_label(baseline_mlx_metal_direct_decode_min_dispatch_share);
    let candidate_mlx_metal_prefix_min_dispatch_share =
        mlx_metal_dispatch_share_label(candidate_mlx_metal_prefix_min_dispatch_share);
    let candidate_mlx_metal_direct_decode_min_dispatch_share =
        mlx_metal_dispatch_share_label(candidate_mlx_metal_direct_decode_min_dispatch_share);

    format!(
        "# Benchmark Compare\n\n- baseline: `{baseline_id}`\n{trusted_baseline_line}- candidate: `{candidate_id}`\n- mode: `{compare_mode}`\n- tool_mode: `{tool_mode}`\n- selected_backend: `{selected_backend}`\n- support_tier: `{support_tier}`\n- resolution_policy: `{resolution_policy}`\n- backend_adapter: `{backend_adapter}`\n- execution_semantics: `{execution_semantics}`\n- baseline_mlx_metal_readiness: `{baseline_mlx_metal_readiness}`\n- baseline_mlx_metal_hot_path_cpu_fallback_free: `{baseline_mlx_metal_hot_path_cpu_fallback_free}`\n- baseline_mlx_metal_batched_direct_decode_logits_ready: `{baseline_mlx_metal_batched_direct_decode_logits_ready}`\n- baseline_prefix_min_mlx_metal_dispatch_share: `{baseline_mlx_metal_prefix_min_dispatch_share}`\n- baseline_direct_decode_min_mlx_metal_dispatch_share: `{baseline_mlx_metal_direct_decode_min_dispatch_share}`\n- baseline_mlx_metal_readiness_blockers: `{baseline_mlx_metal_readiness_blockers}`\n- candidate_mlx_metal_readiness: `{candidate_mlx_metal_readiness}`\n- candidate_mlx_metal_hot_path_cpu_fallback_free: `{candidate_mlx_metal_hot_path_cpu_fallback_free}`\n- candidate_mlx_metal_batched_direct_decode_logits_ready: `{candidate_mlx_metal_batched_direct_decode_logits_ready}`\n- candidate_prefix_min_mlx_metal_dispatch_share: `{candidate_mlx_metal_prefix_min_dispatch_share}`\n- candidate_direct_decode_min_mlx_metal_dispatch_share: `{candidate_mlx_metal_direct_decode_min_dispatch_share}`\n- candidate_mlx_metal_readiness_blockers: `{candidate_mlx_metal_readiness_blockers}`\n- metal_numeric_scaffold_only: `{metal_numeric_scaffold_only}`\n- metal_complete_model_forward_supported: `{metal_complete_model_forward_supported}`\n- metal_model_conditioned_inputs: `{metal_model_conditioned_inputs}`\n- mlx_metal_prefix_layers_attention: `{mlx_metal_prefix_layers_attention}`\n- metal_prefix_layers_cpu_reference: `{metal_prefix_layers_cpu_reference}`\n- mlx_metal_prefix_dispatch_count: `{mlx_metal_prefix_dispatch_count}`\n- metal_prefix_cpu_reference_dispatch_count: `{metal_prefix_cpu_reference_dispatch_count}`\n- mlx_metal_projection_f32_binding_count: `{mlx_metal_projection_f32_binding_count}`\n- mlx_metal_projection_f16_binding_count: `{mlx_metal_projection_f16_binding_count}`\n- mlx_metal_projection_bf16_binding_count: `{mlx_metal_projection_bf16_binding_count}`\n- mlx_metal_projection_unsupported_binding_count: `{mlx_metal_projection_unsupported_binding_count}`\n- metal_direct_decode_tokens: `{metal_direct_decode_tokens}`\n- metal_direct_decode_batching_opportunity_observed: `{metal_direct_decode_batching_opportunity_observed}`\n- metal_direct_decode_model_bound_ffn: `{metal_direct_decode_model_bound_ffn}`\n- metal_direct_decode_batched_logits_group_count: `{metal_direct_decode_batched_logits_group_count}`\n- metal_direct_decode_batched_logits_token_count: `{metal_direct_decode_batched_logits_token_count}`\n- metal_direct_decode_batched_group_fallback_count: `{metal_direct_decode_batched_group_fallback_count}`\n- metal_direct_decode_batched_group_fallback_token_count: `{metal_direct_decode_batched_group_fallback_token_count}`\n- metal_real_model_tensor_inputs: `{metal_real_model_tensor_inputs}`\n- metal_real_model_forward: `{metal_real_model_forward}`\n- metal_model_artifacts_validated: `{metal_model_artifacts_validated}`\n- mlx_metal_rms_norm_f32_binding_count: `{mlx_metal_rms_norm_f32_binding_count}`\n- mlx_metal_rms_norm_f16_binding_count: `{mlx_metal_rms_norm_f16_binding_count}`\n- mlx_metal_rms_norm_bf16_binding_count: `{mlx_metal_rms_norm_bf16_binding_count}`\n- mlx_metal_rms_norm_unsupported_binding_count: `{mlx_metal_rms_norm_unsupported_binding_count}`\n{mlx_kv_cache_lines}- prefix_cache_path: `{prefix_cache_path}`\n- prefix_cache_evidence: `{prefix_cache_evidence}`\n- prefix_reuse_provenance: `{prefix_reuse_provenance}`\n- backend_reported_cached_prompt_tokens: `{backend_reported_cached_prompt_tokens}`\n\n| Metric | Delta % |\n| --- | ---: |\n| TTFT | {ttft:.2} |\n| Decode tok/s | {decode:.2} |\n| Memory peak MB | {memory:.2} |\n| Prefix hit rate | {prefix:.2} |\n{}",
        summary_note
    )
}
