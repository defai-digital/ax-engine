use serde_json::{Value, json};

use crate::json_io::nested_value;
use crate::labels::compare_result_label;

pub(crate) fn inferred_tool_mode_from_runtime_json(runtime: &Value) -> Option<&'static str> {
    match nested_value(runtime, &["mlx_runtime", "runner"]).and_then(Value::as_str) {
        Some("metal_bringup") => return Some("engine_bringup_runtime"),
        _ => {}
    }

    match nested_value(runtime, &["backend_adapter", "kind"]).and_then(Value::as_str) {
        Some("llama_cpp_server_completion") => return Some("llama_cpp_stepwise_runtime"),
        Some(_) => return Some("llama_cpp_blocking_runtime"),
        None => {}
    }

    match runtime.get("selected_backend").and_then(Value::as_str) {
        Some("mlx") => None,
        Some(_) => Some("llama_cpp_blocking_runtime"),
        None => None,
    }
}

pub(crate) fn explicit_or_inferred_tool_mode(
    artifact: &Value,
    explicit_path: &[&str],
    fallback: &'static str,
) -> String {
    nested_value(artifact, explicit_path)
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            artifact
                .get("runtime")
                .and_then(inferred_tool_mode_from_runtime_json)
                .map(str::to_string)
        })
        .unwrap_or_else(|| fallback.to_string())
}

pub(crate) fn explicit_or_inferred_compare_mode(regression: &Value) -> String {
    nested_value(regression, &["summary", "result"])
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            let tool_mode =
                explicit_or_inferred_tool_mode(regression, &["summary", "tool_mode"], "unknown");
            let execution_semantics = nested_value(regression, &["summary", "execution_semantics"])
                .and_then(Value::as_str)
                .unwrap_or("unknown");

            if tool_mode != "unknown" {
                return Some(compare_result_label(&tool_mode, execution_semantics).to_string());
            }

            regression
                .get("runtime")
                .and_then(inferred_tool_mode_from_runtime_json)
                .map(|tool_mode| compare_result_label(tool_mode, "unknown"))
                .map(str::to_string)
        })
        .unwrap_or_else(|| "engine_bringup_compare".to_string())
}

pub(crate) fn string_at_path_or_unknown(json: &Value, path: &[&str]) -> String {
    nested_value(json, path)
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string()
}

fn informational_diff_entry(baseline: &Value, candidate: &Value, path: &[&str]) -> Value {
    let baseline_value = nested_value(baseline, path).cloned().unwrap_or(Value::Null);
    let candidate_value = nested_value(candidate, path)
        .cloned()
        .unwrap_or(Value::Null);

    json!({
        "baseline": baseline_value,
        "candidate": candidate_value,
        "changed": baseline_value != candidate_value
    })
}

pub(crate) fn runtime_tuning_informational_diff_json(
    baseline_environment: &Value,
    candidate_environment: &Value,
) -> Value {
    json!({
        "max_batch_tokens": informational_diff_entry(
            baseline_environment,
            candidate_environment,
            &["runtime", "max_batch_tokens"],
        ),
        "kv_total_blocks": informational_diff_entry(
            baseline_environment,
            candidate_environment,
            &["runtime", "kv_total_blocks"],
        ),
        "prefix_cache": informational_diff_entry(
            baseline_environment,
            candidate_environment,
            &["runtime", "flags", "prefix_cache"],
        )
    })
}
