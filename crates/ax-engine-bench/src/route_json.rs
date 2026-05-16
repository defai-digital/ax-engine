use serde_json::Value;

use crate::json_io::nested_value;

pub(crate) fn route_flag_from_json(
    route_json: &Value,
    field: &str,
    decision_keys: &[&str],
) -> bool {
    if let Some(value) = nested_value(route_json, &[field]).and_then(Value::as_bool) {
        return value;
    }

    decision_keys.iter().any(|key| {
        nested_value(route_json, &["crossover_decisions", key])
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
    })
}

pub(crate) fn route_count_from_json(
    route_json: &Value,
    field: &str,
    decision_keys: &[&str],
) -> u32 {
    if let Some(value) = nested_value(route_json, &[field]).and_then(Value::as_u64) {
        return value.min(u64::from(u32::MAX)) as u32;
    }

    decision_keys
        .iter()
        .find_map(|key| {
            nested_value(route_json, &["crossover_decisions", key])
                .and_then(Value::as_u64)
                .map(|value| value.min(u64::from(u32::MAX)) as u32)
        })
        .unwrap_or(0)
}

pub(crate) fn route_counter_from_json(route_json: &Value, key: &str) -> u32 {
    route_count_from_json(route_json, key, &[key])
}

pub(crate) fn mlx_kv_cache_route_markdown_lines(route_json: &Value, prefix: &str) -> String {
    let mut lines = String::new();
    for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
        lines.push_str(&format!(
            "- {prefix}{key}: `{}`\n",
            route_counter_from_json(route_json, key)
        ));
    }
    lines
}

pub(crate) fn mlx_kv_cache_regression_markdown_lines(regression_json: &Value) -> String {
    let mut lines = String::new();
    for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
        let baseline = nested_value(regression_json, &["contract", key, "baseline"])
            .or_else(|| nested_value(regression_json, &["summary", key]))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let candidate = nested_value(regression_json, &["contract", key, "candidate"])
            .and_then(Value::as_u64)
            .unwrap_or(0);
        lines.push_str(&format!("- baseline_{key}: `{baseline}`\n"));
        lines.push_str(&format!("- candidate_{key}: `{candidate}`\n"));
    }
    lines
}

pub(crate) fn route_execution_semantics_from_json(route_json: &Value) -> String {
    if let Some(value) = nested_value(route_json, &["execution_semantics"]).and_then(Value::as_str)
    {
        return value.to_string();
    }

    let metal_real_model_forward = route_flag_from_json(
        route_json,
        "metal_real_model_forward",
        &["metal_dispatch_real_model_forward"],
    );
    let metal_direct_decode_model_bound_ffn = route_flag_from_json(
        route_json,
        "metal_direct_decode_model_bound_ffn",
        &["metal_dispatch_direct_decode_model_bound_ffn"],
    );
    let metal_real_model_tensor_inputs = route_flag_from_json(
        route_json,
        "metal_real_model_tensor_inputs",
        &[
            "metal_dispatch_real_model_tensor_inputs",
            "metal_dispatch_runtime_real_model_tensor_inputs",
        ],
    );
    let metal_model_conditioned_inputs = route_flag_from_json(
        route_json,
        "metal_model_conditioned_inputs",
        &[
            "metal_dispatch_model_conditioned_inputs",
            "metal_dispatch_runtime_model_conditioned_inputs",
        ],
    );
    let mlx_metal_prefix_layers_attention = route_flag_from_json(
        route_json,
        "mlx_metal_prefix_layers_attention",
        &["metal_dispatch_prefix_layers_native_attention"],
    );
    let metal_prefix_layers_cpu_reference = route_flag_from_json(
        route_json,
        "metal_prefix_layers_cpu_reference",
        &["metal_dispatch_prefix_layers_cpu_reference"],
    );
    let mlx_metal_prefix_dispatch_count = route_count_from_json(
        route_json,
        "mlx_metal_prefix_dispatch_count",
        &["metal_dispatch_prefix_native_dispatch_count"],
    );
    let metal_prefix_cpu_reference_dispatch_count = route_count_from_json(
        route_json,
        "metal_prefix_cpu_reference_dispatch_count",
        &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
    );
    let metal_multilayer_mixed_prefix_attention = mlx_metal_prefix_dispatch_count > 0
        && metal_prefix_cpu_reference_dispatch_count > 0
        || (mlx_metal_prefix_layers_attention && metal_prefix_layers_cpu_reference);
    let metal_numeric_scaffold_only = route_flag_from_json(
        route_json,
        "metal_numeric_scaffold_only",
        &["metal_dispatch_numeric_scaffold_only"],
    );
    let metal_model_layer_count = nested_value(route_json, &["metal_model_layer_count"])
        .and_then(Value::as_u64)
        .map(|value| value.min(u64::from(u32::MAX)) as u32)
        .unwrap_or_else(|| {
            nested_value(
                route_json,
                &["crossover_decisions", "metal_dispatch_model_layer_count"],
            )
            .and_then(Value::as_u64)
            .map(|value| value.min(u64::from(u32::MAX)) as u32)
            .unwrap_or(0)
        });
    let metal_multilayer_model_incomplete =
        metal_model_layer_count > 1 && !metal_real_model_forward;
    let delegated_backend_runtime = nested_value(route_json, &["prefix_cache_path"])
        .and_then(Value::as_str)
        == Some("delegated_prompt_cache")
        || nested_value(
            route_json,
            &["crossover_decisions", "delegated_cached_tokens"],
        )
        .and_then(Value::as_u64)
        .unwrap_or(0)
            > 0;

    if metal_real_model_forward {
        "metal_real_model_forward".to_string()
    } else if metal_multilayer_model_incomplete && metal_multilayer_mixed_prefix_attention {
        "metal_multilayer_mixed_prefix_attention".to_string()
    } else if metal_multilayer_model_incomplete && mlx_metal_prefix_layers_attention {
        "metal_multilayer_native_prefix_attention".to_string()
    } else if metal_multilayer_model_incomplete {
        "metal_multilayer_model_incomplete".to_string()
    } else if metal_real_model_tensor_inputs && metal_direct_decode_model_bound_ffn {
        "metal_model_bound_ffn_decode".to_string()
    } else if metal_real_model_tensor_inputs {
        "metal_real_model_tensor_inputs".to_string()
    } else if metal_model_conditioned_inputs {
        "metal_model_conditioned_numeric_scaffold".to_string()
    } else if metal_numeric_scaffold_only {
        "metal_numeric_scaffold_only".to_string()
    } else if delegated_backend_runtime {
        "delegated_backend_runtime".to_string()
    } else {
        "none_observed".to_string()
    }
}

pub(crate) fn route_execution_semantics_from_environment_json(environment: &Value) -> String {
    let route_json = environment.get("route").unwrap_or(&Value::Null);
    let semantics = route_execution_semantics_from_json(route_json);

    if semantics != "none_observed" {
        return semantics;
    }

    semantics
}
