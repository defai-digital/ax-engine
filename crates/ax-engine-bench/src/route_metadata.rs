use ax_engine_core::RouteMetadata;
use ax_engine_sdk::GenerateRouteReport;
use serde_json::{Map, Value, json};

pub(crate) fn route_metadata_from_generate_route(route: &GenerateRouteReport) -> RouteMetadata {
    RouteMetadata {
        execution_plan: route.execution_plan.clone(),
        attention_route: route.attention_route.clone(),
        kv_mode: route.kv_mode.clone(),
        prefix_cache_path: route.prefix_cache_path.clone(),
        barrier_mode: route.barrier_mode.clone(),
        crossover_decisions: route
            .crossover_decisions
            .iter()
            .map(|(key, value)| (key.clone(), *value))
            .collect(),
    }
}

pub(crate) fn route_decision_value(route: &RouteMetadata, key: &str) -> u32 {
    route
        .crossover_decisions
        .iter()
        .find(|(decision, _)| decision == key)
        .map(|(_, value)| *value)
        .unwrap_or(0)
}

pub(crate) fn route_decision_flag(route: &RouteMetadata, key: &str) -> bool {
    route_decision_value(route, key) > 0
}

pub(crate) fn mlx_metal_coverage_ratio(native: u32, cpu: u32) -> Option<f64> {
    let total = native.checked_add(cpu)?;
    (total > 0).then_some(native as f64 / total as f64)
}

pub(crate) fn mlx_metal_dispatch_share_label(share: Option<f64>) -> String {
    share
        .map(|value| format!("{:.2}%", value * 100.0))
        .unwrap_or_else(|| "n/a".to_string())
}

pub(crate) fn route_model_layer_count(route: &RouteMetadata) -> u32 {
    route_decision_value(route, "metal_dispatch_model_layer_count")
}

pub(crate) fn route_prefix_native_dispatch_count(route: &RouteMetadata) -> u32 {
    route_decision_value(route, "metal_dispatch_prefix_native_dispatch_count")
}

pub(crate) fn route_prefix_cpu_reference_dispatch_count(route: &RouteMetadata) -> u32 {
    route_decision_value(route, "metal_dispatch_prefix_cpu_reference_dispatch_count")
}

pub(crate) fn route_execution_semantics_label(route: &RouteMetadata) -> &'static str {
    let metal_real_model_forward = route_decision_flag(route, "metal_dispatch_real_model_forward");
    let metal_direct_decode_model_bound_ffn =
        route_decision_flag(route, "metal_dispatch_direct_decode_model_bound_ffn");
    let metal_real_model_tensor_inputs =
        route_decision_flag(route, "metal_dispatch_real_model_tensor_inputs")
            || route_decision_flag(route, "metal_dispatch_runtime_real_model_tensor_inputs");
    let metal_model_conditioned_inputs =
        route_decision_flag(route, "metal_dispatch_model_conditioned_inputs")
            || route_decision_flag(route, "metal_dispatch_runtime_model_conditioned_inputs");
    let mlx_metal_prefix_layers_attention =
        route_decision_flag(route, "metal_dispatch_prefix_layers_native_attention");
    let metal_prefix_layers_cpu_reference =
        route_decision_flag(route, "metal_dispatch_prefix_layers_cpu_reference");
    let mlx_metal_prefix_dispatch_count = route_prefix_native_dispatch_count(route);
    let metal_prefix_cpu_reference_dispatch_count =
        route_prefix_cpu_reference_dispatch_count(route);
    let metal_multilayer_mixed_prefix_attention = mlx_metal_prefix_dispatch_count > 0
        && metal_prefix_cpu_reference_dispatch_count > 0
        || (mlx_metal_prefix_layers_attention && metal_prefix_layers_cpu_reference);
    let metal_numeric_scaffold_only =
        route_decision_flag(route, "metal_dispatch_numeric_scaffold_only");
    let metal_model_layer_count = route_model_layer_count(route);
    let metal_multilayer_model_incomplete =
        metal_model_layer_count > 1 && !metal_real_model_forward;
    let delegated_cached_tokens = route_decision_flag(route, "delegated_cached_tokens")
        || route.prefix_cache_path.as_deref() == Some("delegated_prompt_cache");

    if metal_real_model_forward {
        "metal_real_model_forward"
    } else if metal_multilayer_model_incomplete && metal_multilayer_mixed_prefix_attention {
        "metal_multilayer_mixed_prefix_attention"
    } else if metal_multilayer_model_incomplete && mlx_metal_prefix_layers_attention {
        "metal_multilayer_native_prefix_attention"
    } else if metal_multilayer_model_incomplete {
        "metal_multilayer_model_incomplete"
    } else if metal_real_model_tensor_inputs && metal_direct_decode_model_bound_ffn {
        "metal_model_bound_ffn_decode"
    } else if metal_real_model_tensor_inputs {
        "metal_real_model_tensor_inputs"
    } else if metal_model_conditioned_inputs {
        "metal_model_conditioned_numeric_scaffold"
    } else if metal_numeric_scaffold_only {
        "metal_numeric_scaffold_only"
    } else if delegated_cached_tokens {
        "delegated_backend_runtime"
    } else {
        "none_observed"
    }
}

pub(crate) fn prefix_cache_evidence_label(route: &RouteMetadata) -> String {
    match route.prefix_cache_path.as_deref() {
        Some("live_request_share") => "engine_live_request_scheduler_reuse".to_string(),
        Some("retained_prompt_prefix_cache") => "engine_retained_block_cache_metadata".to_string(),
        Some("mixed_live_and_retained") => "engine_mixed_live_and_retained_reuse".to_string(),
        Some("delegated_prompt_cache") => "backend_reported_cached_prompt_tokens".to_string(),
        Some("mixed_prefix_cache_paths") => "mixed_prefix_cache_evidence".to_string(),
        Some("metadata_lookup") | None => "none_observed".to_string(),
        Some(other) => format!("route_label:{other}"),
    }
}

pub(crate) fn prefix_reuse_provenance_label(route: &RouteMetadata) -> String {
    match route.prefix_cache_path.as_deref() {
        Some("live_request_share") => "mlx_live_request_share".to_string(),
        Some("retained_prompt_prefix_cache") => "mlx_retained_prompt_prefix_cache".to_string(),
        Some("mixed_live_and_retained") => "mlx_mixed_live_and_retained".to_string(),
        Some("delegated_prompt_cache") => "delegated_backend_prompt_cache".to_string(),
        Some("mixed_prefix_cache_paths") => "mixed_prefix_reuse_provenance".to_string(),
        Some("metadata_lookup") | None => "none_observed".to_string(),
        Some(other) => format!("route_label:{other}"),
    }
}

pub(crate) fn backend_reported_cached_prompt_tokens(route: &RouteMetadata) -> Option<u32> {
    route
        .crossover_decisions
        .iter()
        .find(|(key, _)| key == "delegated_cached_tokens")
        .map(|(_, value)| *value)
        .filter(|value| *value > 0)
}

pub(crate) fn serialize_route_metadata(route: &RouteMetadata) -> Value {
    let mut crossover_decisions = Map::new();
    for (key, value) in &route.crossover_decisions {
        crossover_decisions.insert(key.clone(), json!(value));
    }

    let mut route_json = Map::new();
    route_json.insert("execution_plan".to_string(), json!(route.execution_plan));
    route_json.insert(
        "execution_semantics".to_string(),
        json!(route_execution_semantics_label(route)),
    );
    route_json.insert("attention_route".to_string(), json!(route.attention_route));
    route_json.insert("kv_mode".to_string(), json!(route.kv_mode));
    route_json.insert(
        "prefix_cache_path".to_string(),
        json!(route.prefix_cache_path),
    );
    route_json.insert(
        "prefix_cache_evidence".to_string(),
        json!(prefix_cache_evidence_label(route)),
    );
    route_json.insert(
        "prefix_reuse_provenance".to_string(),
        json!(prefix_reuse_provenance_label(route)),
    );
    route_json.insert("barrier_mode".to_string(), json!(route.barrier_mode));
    route_json.insert(
        "metal_numeric_scaffold_only".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_numeric_scaffold_only"
        )),
    );
    route_json.insert(
        "metal_model_layer_count".to_string(),
        json!(route_model_layer_count(route)),
    );
    for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
        if let Some(value) = crossover_decisions.get(key).cloned() {
            route_json.insert(key.to_string(), value);
        }
    }
    route_json.insert(
        "metal_complete_model_forward_supported".to_string(),
        json!(
            route_decision_flag(route, "metal_dispatch_complete_model_forward_supported")
                || route_decision_flag(
                    route,
                    "metal_dispatch_runtime_complete_model_forward_supported"
                )
        ),
    );
    route_json.insert(
        "metal_multilayer_model_incomplete".to_string(),
        json!(
            route_model_layer_count(route) > 1
                && !route_decision_flag(route, "metal_dispatch_real_model_forward")
                && !(route_decision_flag(route, "metal_dispatch_complete_model_forward_supported")
                    || route_decision_flag(
                        route,
                        "metal_dispatch_runtime_complete_model_forward_supported"
                    ))
        ),
    );
    route_json.insert(
        "metal_real_model_forward".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_real_model_forward"
        )),
    );
    route_json.insert(
        "metal_model_conditioned_inputs".to_string(),
        json!(
            route_decision_flag(route, "metal_dispatch_model_conditioned_inputs")
                || route_decision_flag(route, "metal_dispatch_runtime_model_conditioned_inputs")
        ),
    );
    route_json.insert(
        "mlx_metal_prefix_layers_attention".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_prefix_layers_native_attention"
        )),
    );
    route_json.insert(
        "metal_prefix_layers_cpu_reference".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_prefix_layers_cpu_reference"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_dispatch_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_dispatch_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_reference_dispatch_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_reference_dispatch_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_projection_row_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_projection_row_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_projection_row_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_projection_row_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_rms_norm_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_rms_norm_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_rms_norm_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_rms_norm_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_ffn_activation_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_ffn_activation_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_ffn_activation_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_ffn_activation_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_residual_add_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_residual_add_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_residual_add_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_residual_add_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_scale_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_scale_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_scale_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_scale_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_projection_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(route, "metal_dispatch_prefix_native_projection_row_count"),
            route_decision_value(route, "metal_dispatch_prefix_cpu_projection_row_count"),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_prefix_rms_norm_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(route, "metal_dispatch_prefix_native_rms_norm_element_count"),
            route_decision_value(route, "metal_dispatch_prefix_cpu_rms_norm_element_count"),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_prefix_ffn_activation_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_prefix_native_ffn_activation_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_prefix_cpu_ffn_activation_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_prefix_residual_add_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_prefix_native_residual_add_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_prefix_cpu_residual_add_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_prefix_scale_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(route, "metal_dispatch_prefix_native_scale_element_count"),
            route_decision_value(route, "metal_dispatch_prefix_cpu_scale_element_count"),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "mlx_metal_projection_f32_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_projection_f32_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_projection_f16_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_projection_f16_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_projection_bf16_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_projection_bf16_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_projection_unsupported_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_projection_unsupported_binding_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_tokens".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_direct_decode_tokens"
        )),
    );
    route_json.insert(
        "metal_direct_decode_model_bound_ffn".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_direct_decode_model_bound_ffn"
        )),
    );
    route_json.insert(
        "metal_direct_decode_checksum_lo".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_checksum_lo"
        )),
    );
    route_json.insert(
        "metal_real_model_tensor_inputs".to_string(),
        json!(
            route_decision_flag(route, "metal_dispatch_real_model_tensor_inputs")
                || route_decision_flag(route, "metal_dispatch_runtime_real_model_tensor_inputs")
        ),
    );
    route_json.insert(
        "metal_model_artifacts_validated".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_model_artifacts_validated"
        )),
    );
    route_json.insert(
        "mlx_metal_rms_norm_f32_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_rms_norm_f32_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_rms_norm_f16_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_rms_norm_f16_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_rms_norm_bf16_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_rms_norm_bf16_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_rms_norm_unsupported_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_rms_norm_unsupported_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_projection_row_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_projection_row_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_projection_row_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_projection_row_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_rms_norm_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_rms_norm_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_rms_norm_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_rms_norm_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_ffn_activation_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_ffn_activation_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_ffn_activation_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_ffn_activation_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_residual_add_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_residual_add_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_residual_add_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_residual_add_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_scale_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_scale_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_scale_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_scale_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_batched_logits_group_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_logits_group_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_batched_logits_token_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_logits_token_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_batched_group_fallback_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_group_fallback_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_batched_group_fallback_token_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_group_fallback_token_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_projection_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_projection_row_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_projection_row_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_direct_decode_rms_norm_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_rms_norm_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_rms_norm_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_direct_decode_ffn_activation_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_ffn_activation_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_ffn_activation_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_direct_decode_residual_add_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_residual_add_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_residual_add_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_direct_decode_scale_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_scale_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_scale_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "crossover_decisions".to_string(),
        Value::Object(crossover_decisions),
    );

    if let Some(value) = backend_reported_cached_prompt_tokens(route) {
        route_json.insert(
            "backend_reported_cached_prompt_tokens".to_string(),
            json!(value),
        );
    }

    Value::Object(route_json)
}
