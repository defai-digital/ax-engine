use super::*;

pub(crate) fn build_trusted_baseline_json(
    name: &str,
    slug: &str,
    source_dir: &Path,
    manifest: &Value,
    environment: &Value,
    metrics: &Value,
) -> Result<Value, CliError> {
    let created_at_unix_s = unix_timestamp_secs()?;
    let source_run_id = metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let manifest_id = manifest
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let manifest_class = manifest
        .get("class")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let scenario = manifest
        .get("scenario")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model_family = nested_value(manifest, &["model", "family"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let tool_mode =
        explicit_or_inferred_tool_mode(environment, &["software", "tool_mode"], "unknown");
    let selected_backend = nested_value(environment, &["runtime", "selected_backend"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let support_tier = nested_value(environment, &["runtime", "support_tier"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let resolution_policy = nested_value(environment, &["runtime", "resolution_policy"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_path = nested_value(environment, &["route", "prefix_cache_path"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_evidence = nested_value(environment, &["route", "prefix_cache_evidence"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_reuse_provenance = nested_value(environment, &["route", "prefix_reuse_provenance"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let route_json = environment
        .get("route")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let execution_semantics = route_execution_semantics_from_environment_json(environment);
    let metal_numeric_scaffold_only = route_flag_from_json(
        &route_json,
        "metal_numeric_scaffold_only",
        &["metal_dispatch_numeric_scaffold_only"],
    );
    let metal_real_model_forward = route_flag_from_json(
        &route_json,
        "metal_real_model_forward",
        &["metal_dispatch_real_model_forward"],
    );
    let metal_complete_model_forward_supported = route_flag_from_json(
        &route_json,
        "metal_complete_model_forward_supported",
        &[
            "metal_dispatch_complete_model_forward_supported",
            "metal_dispatch_runtime_complete_model_forward_supported",
        ],
    );
    let metal_model_conditioned_inputs = route_flag_from_json(
        &route_json,
        "metal_model_conditioned_inputs",
        &[
            "metal_dispatch_model_conditioned_inputs",
            "metal_dispatch_runtime_model_conditioned_inputs",
        ],
    );
    let mlx_metal_prefix_dispatch_count = route_count_from_json(
        &route_json,
        "mlx_metal_prefix_dispatch_count",
        &["metal_dispatch_prefix_native_dispatch_count"],
    );
    let metal_prefix_cpu_reference_dispatch_count = route_count_from_json(
        &route_json,
        "metal_prefix_cpu_reference_dispatch_count",
        &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
    );
    let metal_direct_decode_tokens = route_flag_from_json(
        &route_json,
        "metal_direct_decode_tokens",
        &["metal_dispatch_direct_decode_tokens"],
    );
    let metal_direct_decode_model_bound_ffn = route_flag_from_json(
        &route_json,
        "metal_direct_decode_model_bound_ffn",
        &["metal_dispatch_direct_decode_model_bound_ffn"],
    );
    let metal_direct_decode_checksum_lo =
        nested_value(&route_json, &["metal_direct_decode_checksum_lo"])
            .cloned()
            .unwrap_or_else(|| json!(0));
    let metal_real_model_tensor_inputs = route_flag_from_json(
        &route_json,
        "metal_real_model_tensor_inputs",
        &[
            "metal_dispatch_real_model_tensor_inputs",
            "metal_dispatch_runtime_real_model_tensor_inputs",
        ],
    );
    let metal_model_artifacts_validated = route_flag_from_json(
        &route_json,
        "metal_model_artifacts_validated",
        &["metal_dispatch_model_artifacts_validated"],
    );
    let mlx_metal_projection_f32_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_f32_binding_count",
        &["metal_dispatch_native_projection_f32_binding_count"],
    );
    let mlx_metal_projection_f16_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_f16_binding_count",
        &["metal_dispatch_native_projection_f16_binding_count"],
    );
    let mlx_metal_projection_bf16_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_bf16_binding_count",
        &["metal_dispatch_native_projection_bf16_binding_count"],
    );
    let mlx_metal_projection_unsupported_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_unsupported_binding_count",
        &["metal_dispatch_native_projection_unsupported_binding_count"],
    );
    let mlx_metal_projection_source_quantized_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_source_quantized_binding_count",
        &["metal_dispatch_native_projection_source_quantized_binding_count"],
    );
    let mlx_metal_rms_norm_f32_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_f32_binding_count",
        &["metal_dispatch_native_rms_norm_f32_binding_count"],
    );
    let mlx_metal_rms_norm_f16_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_f16_binding_count",
        &["metal_dispatch_native_rms_norm_f16_binding_count"],
    );
    let mlx_metal_rms_norm_bf16_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_bf16_binding_count",
        &["metal_dispatch_native_rms_norm_bf16_binding_count"],
    );
    let mlx_metal_rms_norm_unsupported_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_unsupported_binding_count",
        &["metal_dispatch_native_rms_norm_unsupported_binding_count"],
    );
    let mlx_metal_rms_norm_source_quantized_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_source_quantized_binding_count",
        &["metal_dispatch_native_rms_norm_source_quantized_binding_count"],
    );
    let backend_reported_cached_prompt_tokens = nested_value(
        environment,
        &["route", "backend_reported_cached_prompt_tokens"],
    )
    .cloned();
    let machine = environment
        .get("machine")
        .cloned()
        .unwrap_or_else(|| json!({ "device": "unknown" }));
    let software = environment
        .get("software")
        .cloned()
        .unwrap_or_else(|| json!({ "tool_mode": tool_mode.clone() }));

    let mut baseline = json!({
        "schema_version": "ax.engine_bench.trusted_baseline.v1",
        "name": name,
        "slug": slug,
        "created_at_unix_s": created_at_unix_s,
        "source_result_dir": source_dir.display().to_string(),
        "source_run_id": source_run_id,
        "manifest": {
            "id": manifest_id,
            "class": manifest_class,
            "scenario": scenario,
            "model_family": model_family
        },
        "runtime": {
            "tool_mode": tool_mode,
            "selected_backend": selected_backend,
            "support_tier": support_tier,
            "resolution_policy": resolution_policy
        },
        "route": {
            "execution_semantics": execution_semantics,
            "prefix_cache_path": prefix_cache_path,
            "prefix_cache_evidence": prefix_cache_evidence,
            "prefix_reuse_provenance": prefix_reuse_provenance,
            "metal_numeric_scaffold_only": metal_numeric_scaffold_only,
            "metal_complete_model_forward_supported": metal_complete_model_forward_supported,
            "metal_real_model_forward": metal_real_model_forward,
            "metal_model_conditioned_inputs": metal_model_conditioned_inputs,
            "mlx_metal_prefix_dispatch_count": mlx_metal_prefix_dispatch_count,
            "metal_prefix_cpu_reference_dispatch_count": metal_prefix_cpu_reference_dispatch_count,
            "metal_direct_decode_tokens": metal_direct_decode_tokens,
            "metal_direct_decode_model_bound_ffn": metal_direct_decode_model_bound_ffn,
            "metal_direct_decode_checksum_lo": metal_direct_decode_checksum_lo,
            "metal_real_model_tensor_inputs": metal_real_model_tensor_inputs,
            "metal_model_artifacts_validated": metal_model_artifacts_validated,
            "mlx_metal_projection_f32_binding_count": mlx_metal_projection_f32_binding_count,
            "mlx_metal_projection_f16_binding_count": mlx_metal_projection_f16_binding_count,
            "mlx_metal_projection_bf16_binding_count": mlx_metal_projection_bf16_binding_count,
            "mlx_metal_projection_unsupported_binding_count": mlx_metal_projection_unsupported_binding_count,
            "mlx_metal_projection_source_quantized_binding_count": mlx_metal_projection_source_quantized_binding_count,
            "mlx_metal_rms_norm_f32_binding_count": mlx_metal_rms_norm_f32_binding_count,
            "mlx_metal_rms_norm_f16_binding_count": mlx_metal_rms_norm_f16_binding_count,
            "mlx_metal_rms_norm_bf16_binding_count": mlx_metal_rms_norm_bf16_binding_count,
            "mlx_metal_rms_norm_unsupported_binding_count": mlx_metal_rms_norm_unsupported_binding_count,
            "mlx_metal_rms_norm_source_quantized_binding_count": mlx_metal_rms_norm_source_quantized_binding_count
        },
        "machine": machine,
        "software": software,
        "metrics": {
            "ttft_ms": metric_number(metrics, "ttft_ms")?,
            "decode_tok_s": metric_number(metrics, "decode_tok_s")?,
            "memory_peak_mb": metric_number(metrics, "memory_peak_mb")?,
            "prefix_hit_rate": metric_number(metrics, "prefix_hit_rate")?
        }
    });

    if let Some(value) = backend_reported_cached_prompt_tokens {
        baseline
            .get_mut("route")
            .and_then(Value::as_object_mut)
            .expect("trusted baseline route should serialize as object")
            .insert("backend_reported_cached_prompt_tokens".to_string(), value);
    }

    if let Some(adapter) = nested_value(environment, &["runtime", "backend_adapter"]).cloned() {
        baseline
            .get_mut("runtime")
            .and_then(Value::as_object_mut)
            .expect("trusted baseline runtime should serialize as object")
            .insert("backend_adapter".to_string(), adapter);
    }

    if let Some(mlx_runtime) = nested_value(environment, &["runtime", "mlx_runtime"]).cloned() {
        baseline
            .get_mut("runtime")
            .and_then(Value::as_object_mut)
            .expect("trusted baseline runtime should serialize as object")
            .insert("mlx_runtime".to_string(), mlx_runtime);
    }

    if let Some(mlx_model) = nested_value(environment, &["runtime", "mlx_model"]).cloned() {
        baseline
            .get_mut("runtime")
            .and_then(Value::as_object_mut)
            .expect("trusted baseline runtime should serialize as object")
            .insert("mlx_model".to_string(), mlx_model);
    }

    Ok(baseline)
}

pub(crate) fn build_trusted_baseline_summary_markdown(trusted_baseline: &Value) -> String {
    let name = trusted_baseline
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let source_run_id = trusted_baseline
        .get("source_run_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let source_result_dir = trusted_baseline
        .get("source_result_dir")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let manifest_id = nested_value(trusted_baseline, &["manifest", "id"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let manifest_class = nested_value(trusted_baseline, &["manifest", "class"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let scenario = nested_value(trusted_baseline, &["manifest", "scenario"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model_family = nested_value(trusted_baseline, &["manifest", "model_family"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let tool_mode = nested_value(trusted_baseline, &["runtime", "tool_mode"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let selected_backend = nested_value(trusted_baseline, &["runtime", "selected_backend"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let support_tier = nested_value(trusted_baseline, &["runtime", "support_tier"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let resolution_policy = nested_value(trusted_baseline, &["runtime", "resolution_policy"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let backend_adapter = nested_value(trusted_baseline, &["runtime", "backend_adapter"])
        .map(json_value_label)
        .unwrap_or_else(|| "none".to_string());
    let prefix_cache_path = nested_value(trusted_baseline, &["route", "prefix_cache_path"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_evidence = nested_value(trusted_baseline, &["route", "prefix_cache_evidence"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_reuse_provenance =
        nested_value(trusted_baseline, &["route", "prefix_reuse_provenance"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let execution_semantics = nested_value(trusted_baseline, &["route", "execution_semantics"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let metal_numeric_scaffold_only =
        nested_value(trusted_baseline, &["route", "metal_numeric_scaffold_only"])
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let metal_complete_model_forward_supported = nested_value(
        trusted_baseline,
        &["route", "metal_complete_model_forward_supported"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let metal_real_model_forward =
        nested_value(trusted_baseline, &["route", "metal_real_model_forward"])
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let metal_model_conditioned_inputs = nested_value(
        trusted_baseline,
        &["route", "metal_model_conditioned_inputs"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let mlx_metal_prefix_layers_attention = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_prefix_layers_attention"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let metal_prefix_layers_cpu_reference = nested_value(
        trusted_baseline,
        &["route", "metal_prefix_layers_cpu_reference"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let mlx_metal_prefix_dispatch_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_prefix_dispatch_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_prefix_cpu_reference_dispatch_count = nested_value(
        trusted_baseline,
        &["route", "metal_prefix_cpu_reference_dispatch_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_tokens =
        nested_value(trusted_baseline, &["route", "metal_direct_decode_tokens"])
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let metal_direct_decode_model_bound_ffn = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_model_bound_ffn"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let metal_direct_decode_checksum_lo = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_checksum_lo"],
    )
    .map(json_value_label)
    .unwrap_or_else(|| "0".to_string());
    let metal_real_model_tensor_inputs = nested_value(
        trusted_baseline,
        &["route", "metal_real_model_tensor_inputs"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let metal_model_artifacts_validated = nested_value(
        trusted_baseline,
        &["route", "metal_model_artifacts_validated"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let mlx_metal_projection_f32_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_projection_f32_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_projection_f16_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_projection_f16_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_projection_bf16_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_projection_bf16_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_projection_unsupported_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_projection_unsupported_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_rms_norm_f32_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_rms_norm_f32_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_rms_norm_f16_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_rms_norm_f16_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_rms_norm_bf16_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_rms_norm_bf16_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_rms_norm_unsupported_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_rms_norm_unsupported_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batched_logits_group_count = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_batched_logits_group_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batched_logits_token_count = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_batched_logits_token_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batched_group_fallback_count = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_batched_group_fallback_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batched_group_fallback_token_count = nested_value(
        trusted_baseline,
        &[
            "route",
            "metal_direct_decode_batched_group_fallback_token_count",
        ],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batching_opportunity_observed = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_batching_opportunity_observed"],
    )
    .map(json_value_label)
    .unwrap_or_else(|| "unknown".to_string());
    let backend_reported_cached_prompt_tokens = nested_value(
        trusted_baseline,
        &["route", "backend_reported_cached_prompt_tokens"],
    )
    .map(json_value_label)
    .unwrap_or_else(|| "none".to_string());
    let route_json = trusted_baseline
        .get("route")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mlx_kv_cache_lines = mlx_kv_cache_route_markdown_lines(&route_json, "");
    let ttft_ms = nested_value(trusted_baseline, &["metrics", "ttft_ms"])
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let decode_tok_s = nested_value(trusted_baseline, &["metrics", "decode_tok_s"])
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let memory_peak_mb = nested_value(trusted_baseline, &["metrics", "memory_peak_mb"])
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let prefix_hit_rate = nested_value(trusted_baseline, &["metrics", "prefix_hit_rate"])
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    format!(
        "# Trusted Benchmark Baseline\n\n- name: `{name}`\n- source_run_id: `{source_run_id}`\n- source_result_dir: `{source_result_dir}`\n- manifest_id: `{manifest_id}`\n- manifest_class: `{manifest_class}`\n- scenario: `{scenario}`\n- model_family: `{model_family}`\n- tool_mode: `{tool_mode}`\n- selected_backend: `{selected_backend}`\n- support_tier: `{support_tier}`\n- resolution_policy: `{resolution_policy}`\n- backend_adapter: `{backend_adapter}`\n- execution_semantics: `{execution_semantics}`\n- metal_numeric_scaffold_only: `{metal_numeric_scaffold_only}`\n- metal_complete_model_forward_supported: `{metal_complete_model_forward_supported}`\n- metal_model_conditioned_inputs: `{metal_model_conditioned_inputs}`\n- mlx_metal_prefix_layers_attention: `{mlx_metal_prefix_layers_attention}`\n- metal_prefix_layers_cpu_reference: `{metal_prefix_layers_cpu_reference}`\n- mlx_metal_prefix_dispatch_count: `{mlx_metal_prefix_dispatch_count}`\n- metal_prefix_cpu_reference_dispatch_count: `{metal_prefix_cpu_reference_dispatch_count}`\n- metal_direct_decode_tokens: `{metal_direct_decode_tokens}`\n- metal_direct_decode_batching_opportunity_observed: `{metal_direct_decode_batching_opportunity_observed}`\n- metal_direct_decode_model_bound_ffn: `{metal_direct_decode_model_bound_ffn}`\n- metal_direct_decode_checksum_lo: `{metal_direct_decode_checksum_lo}`\n- metal_direct_decode_batched_logits_group_count: `{metal_direct_decode_batched_logits_group_count}`\n- metal_direct_decode_batched_logits_token_count: `{metal_direct_decode_batched_logits_token_count}`\n- metal_direct_decode_batched_group_fallback_count: `{metal_direct_decode_batched_group_fallback_count}`\n- metal_direct_decode_batched_group_fallback_token_count: `{metal_direct_decode_batched_group_fallback_token_count}`\n- metal_real_model_tensor_inputs: `{metal_real_model_tensor_inputs}`\n- metal_real_model_forward: `{metal_real_model_forward}`\n- metal_model_artifacts_validated: `{metal_model_artifacts_validated}`\n- mlx_metal_projection_f32_binding_count: `{mlx_metal_projection_f32_binding_count}`\n- mlx_metal_projection_f16_binding_count: `{mlx_metal_projection_f16_binding_count}`\n- mlx_metal_projection_bf16_binding_count: `{mlx_metal_projection_bf16_binding_count}`\n- mlx_metal_projection_unsupported_binding_count: `{mlx_metal_projection_unsupported_binding_count}`\n- mlx_metal_rms_norm_f32_binding_count: `{mlx_metal_rms_norm_f32_binding_count}`\n- mlx_metal_rms_norm_f16_binding_count: `{mlx_metal_rms_norm_f16_binding_count}`\n- mlx_metal_rms_norm_bf16_binding_count: `{mlx_metal_rms_norm_bf16_binding_count}`\n- mlx_metal_rms_norm_unsupported_binding_count: `{mlx_metal_rms_norm_unsupported_binding_count}`\n{mlx_kv_cache_lines}- prefix_cache_path: `{prefix_cache_path}`\n- prefix_cache_evidence: `{prefix_cache_evidence}`\n- prefix_reuse_provenance: `{prefix_reuse_provenance}`\n- backend_reported_cached_prompt_tokens: `{backend_reported_cached_prompt_tokens}`\n- ttft_ms: `{ttft_ms:.2}`\n- decode_tok_s: `{decode_tok_s:.2}`\n- memory_peak_mb: `{memory_peak_mb:.2}`\n- prefix_hit_rate: `{prefix_hit_rate:.2}`\n\nThis directory is a named trusted baseline snapshot. Compare future benchmark runs against this directory instead of overwriting it in place.\n"
    )
}

pub(crate) fn validate_comparable_manifests(
    baseline: &Value,
    candidate: &Value,
) -> Result<(), CliError> {
    let fields = [
        ["schema_version"].as_slice(),
        ["id"].as_slice(),
        ["class"].as_slice(),
        ["scenario"].as_slice(),
        ["model"].as_slice(),
        ["sampling"].as_slice(),
        ["checks"].as_slice(),
    ];

    for field in fields {
        validate_matching_json_field(baseline, candidate, field)?;
    }

    validate_matching_optional_json_field(baseline, candidate, &["source"])?;
    validate_matching_json_field(baseline, candidate, &["runtime", "selected_backend"])?;
    validate_matching_json_field(baseline, candidate, &["runtime", "support_tier"])?;
    validate_matching_json_field(baseline, candidate, &["runtime", "resolution_policy"])?;
    validate_matching_json_field(baseline, candidate, &["runtime", "deterministic"])?;
    validate_matching_optional_json_field(baseline, candidate, &["runtime", "fallback_reason"])?;
    validate_matching_optional_json_field(baseline, candidate, &["runtime", "backend_adapter"])?;
    validate_matching_optional_json_field(baseline, candidate, &["runtime", "llama_cpp_preset"])?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model_artifacts_dir"],
    )?;

    match nested_string(baseline, &["class"])? {
        "scenario" => validate_matching_json_field(baseline, candidate, &["shape"])?,
        "replay" => validate_matching_json_field(baseline, candidate, &["events"])?,
        other => {
            return Err(CliError::Contract(format!(
                "unsupported manifest class for compare validation: {other}"
            )));
        }
    }

    Ok(())
}

pub(crate) fn validate_comparable_environments(
    baseline: &Value,
    candidate: &Value,
) -> Result<(), CliError> {
    let fields = [
        ["schema_version"].as_slice(),
        ["machine", "arch"].as_slice(),
        ["machine", "system_model"].as_slice(),
        ["machine", "soc"].as_slice(),
        ["machine", "memory_bytes"].as_slice(),
        ["software", "os_family"].as_slice(),
        ["software", "os_version"].as_slice(),
        ["software", "os_build"].as_slice(),
        ["software", "kernel_release"].as_slice(),
        ["software", "metal_driver"].as_slice(),
        ["software", "tool_mode"].as_slice(),
        ["runtime", "selected_backend"].as_slice(),
        ["runtime", "support_tier"].as_slice(),
        ["runtime", "resolution_policy"].as_slice(),
        ["runtime", "host", "os"].as_slice(),
        ["runtime", "host", "arch"].as_slice(),
        ["runtime", "host", "supported_mlx_runtime"].as_slice(),
        ["runtime", "host", "unsupported_host_override_active"].as_slice(),
        ["runtime", "metal_toolchain", "fully_available"].as_slice(),
        ["runtime", "metal_toolchain", "metal", "available"].as_slice(),
        ["runtime", "metal_toolchain", "metallib", "available"].as_slice(),
        ["runtime", "metal_toolchain", "metal_ar", "available"].as_slice(),
        ["route", "prefix_cache_path"].as_slice(),
        ["route", "prefix_cache_evidence"].as_slice(),
        ["route", "prefix_reuse_provenance"].as_slice(),
        ["benchmark", "schema_family"].as_slice(),
        ["benchmark", "subcommand"].as_slice(),
    ];

    for field in fields {
        validate_matching_json_field(baseline, candidate, field)?;
    }

    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "host", "detected_soc"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "metal_toolchain", "metal", "version"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "metal_toolchain", "metallib", "version"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "metal_toolchain", "metal_ar", "version"],
    )?;
    validate_matching_optional_json_field(baseline, candidate, &["runtime", "fallback_reason"])?;
    validate_matching_optional_json_field(baseline, candidate, &["runtime", "backend_adapter"])?;
    validate_matching_optional_json_field(baseline, candidate, &["runtime", "llama_cpp_preset"])?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_runtime", "runner"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_runtime", "artifacts_source"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "artifacts_source"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "model_family"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "tensor_format"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "layer_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "tensor_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "tie_word_embeddings"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "bindings_prepared"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "buffers_bound"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "buffer_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "buffer_bytes"],
    )?;
    validate_matching_optional_json_field(baseline, candidate, &["route", "execution_semantics"])?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_numeric_scaffold_only"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_complete_model_forward_supported"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_real_model_forward"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_model_conditioned_inputs"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_prefix_layers_attention"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_prefix_layers_cpu_reference"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_prefix_dispatch_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_prefix_cpu_reference_dispatch_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_direct_decode_tokens"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_direct_decode_model_bound_ffn"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_real_model_tensor_inputs"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_model_artifacts_validated"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_projection_f32_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_projection_f16_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_projection_bf16_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_projection_unsupported_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_rms_norm_f32_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_rms_norm_f16_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_rms_norm_bf16_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_rms_norm_unsupported_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "backend_reported_cached_prompt_tokens"],
    )?;

    Ok(())
}
