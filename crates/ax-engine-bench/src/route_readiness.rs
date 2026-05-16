use ax_engine_sdk::NativeModelReport;
use serde_json::{Value, json};

use crate::json_io::nested_value;
use crate::route_json::{route_count_from_json, route_flag_from_json};
use crate::route_metadata::mlx_metal_coverage_ratio;

const NATIVE_DENSE_DEQUANTIZED_EXPORT_NOTE: &str =
    "source_quantization_dequantized_for_dense_native_export";
pub(crate) const NATIVE_DENSE_DEQUANTIZED_SOURCE_BLOCKER: &str =
    "source_quantization_dequantized_dense_native_artifact";

pub(crate) fn native_model_report_has_dense_dequantized_source(report: &NativeModelReport) -> bool {
    serde_json::to_value(report)
        .ok()
        .is_some_and(|native_model| {
            native_dense_dequantized_source_from_native_model_json(&native_model)
        })
}

pub(crate) fn native_dense_dequantized_source_from_runtime_json(runtime_json: &Value) -> bool {
    nested_value(runtime_json, &["mlx_model"])
        .is_some_and(native_dense_dequantized_source_from_native_model_json)
}

fn native_dense_dequantized_source_from_native_model_json(native_model: &Value) -> bool {
    let contains_quantized_tensors = nested_value(
        native_model,
        &["source_quantization", "contains_quantized_tensors"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let has_dense_dequantized_note = nested_value(native_model, &["runtime_status", "notes"])
        .and_then(Value::as_array)
        .is_some_and(|notes| {
            notes
                .iter()
                .any(|note| note.as_str() == Some(NATIVE_DENSE_DEQUANTIZED_EXPORT_NOTE))
        });

    contains_quantized_tensors && has_dense_dequantized_note
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MlxMetalReadiness {
    pub(crate) status: &'static str,
    pub(crate) hot_path_cpu_fallback_free: bool,
    pub(crate) batched_direct_decode_logits_ready: bool,
    pub(crate) prefix_min_mlx_metal_dispatch_share: Option<f64>,
    pub(crate) direct_decode_min_mlx_metal_dispatch_share: Option<f64>,
    pub(crate) blockers: Vec<&'static str>,
}

impl MlxMetalReadiness {
    pub(crate) fn to_json(&self) -> Value {
        json!({
            "status": self.status,
            "hot_path_cpu_fallback_free": self.hot_path_cpu_fallback_free,
            "batched_direct_decode_logits_ready": self.batched_direct_decode_logits_ready,
            "prefix_min_mlx_metal_dispatch_share": self.prefix_min_mlx_metal_dispatch_share,
            "direct_decode_min_mlx_metal_dispatch_share": self.direct_decode_min_mlx_metal_dispatch_share,
            "blockers": self.blockers
        })
    }

    pub(crate) fn blockers_label(&self) -> String {
        if self.blockers.is_empty() {
            "none".to_string()
        } else {
            self.blockers.join(",")
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MlxMetalReadinessInputs<'a> {
    pub(crate) tool_mode: &'a str,
    pub(crate) selected_backend: &'a str,
    pub(crate) native_dense_dequantized_source: bool,
    pub(crate) native_quantized_projection_binding_count: u32,
    pub(crate) direct_decode_batching_opportunity_observed: bool,
    pub(crate) metal_complete_model_forward_supported: bool,
    pub(crate) metal_real_model_forward: bool,
    pub(crate) metal_model_artifacts_validated: bool,
    pub(crate) mlx_metal_prefix_layers_attention: bool,
    pub(crate) metal_prefix_layers_cpu_reference: bool,
    pub(crate) metal_prefix_cpu_reference_dispatch_count: u32,
    pub(crate) mlx_metal_prefix_projection_row_count: u32,
    pub(crate) metal_prefix_cpu_projection_row_count: u32,
    pub(crate) mlx_metal_prefix_rms_norm_element_count: u32,
    pub(crate) metal_prefix_cpu_rms_norm_element_count: u32,
    pub(crate) mlx_metal_prefix_ffn_activation_element_count: u32,
    pub(crate) metal_prefix_cpu_ffn_activation_element_count: u32,
    pub(crate) mlx_metal_prefix_residual_add_element_count: u32,
    pub(crate) metal_prefix_cpu_residual_add_element_count: u32,
    pub(crate) mlx_metal_prefix_scale_element_count: u32,
    pub(crate) metal_prefix_cpu_scale_element_count: u32,
    pub(crate) metal_direct_decode_tokens: bool,
    pub(crate) metal_direct_decode_model_bound_ffn: bool,
    pub(crate) metal_direct_decode_batched_logits_group_count: u32,
    pub(crate) metal_direct_decode_batched_logits_token_count: u32,
    pub(crate) metal_direct_decode_batched_group_fallback_count: u32,
    pub(crate) metal_direct_decode_batched_group_fallback_token_count: u32,
    pub(crate) mlx_metal_direct_decode_projection_row_count: u32,
    pub(crate) metal_direct_decode_cpu_projection_row_count: u32,
    pub(crate) mlx_metal_direct_decode_rms_norm_element_count: u32,
    pub(crate) metal_direct_decode_cpu_rms_norm_element_count: u32,
    pub(crate) mlx_metal_direct_decode_ffn_activation_element_count: u32,
    pub(crate) metal_direct_decode_cpu_ffn_activation_element_count: u32,
    pub(crate) mlx_metal_direct_decode_residual_add_element_count: u32,
    pub(crate) metal_direct_decode_cpu_residual_add_element_count: u32,
    pub(crate) mlx_metal_direct_decode_scale_element_count: u32,
    pub(crate) metal_direct_decode_cpu_scale_element_count: u32,
}

fn min_mlx_metal_dispatch_share(shares: &[Option<f64>]) -> Option<f64> {
    shares.iter().flatten().copied().reduce(f64::min)
}

pub(crate) fn mlx_metal_readiness(inputs: MlxMetalReadinessInputs<'_>) -> MlxMetalReadiness {
    if inputs.tool_mode.starts_with("llama_cpp_") || inputs.selected_backend != "mlx" {
        return MlxMetalReadiness {
            status: "delegated_runtime",
            hot_path_cpu_fallback_free: false,
            batched_direct_decode_logits_ready: false,
            prefix_min_mlx_metal_dispatch_share: None,
            direct_decode_min_mlx_metal_dispatch_share: None,
            blockers: vec!["delegated_runtime_not_mlx"],
        };
    }

    let prefix_projection_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_projection_row_count,
        inputs.metal_prefix_cpu_projection_row_count,
    );
    let prefix_rms_norm_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_rms_norm_element_count,
        inputs.metal_prefix_cpu_rms_norm_element_count,
    );
    let prefix_ffn_activation_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_ffn_activation_element_count,
        inputs.metal_prefix_cpu_ffn_activation_element_count,
    );
    let prefix_residual_add_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_residual_add_element_count,
        inputs.metal_prefix_cpu_residual_add_element_count,
    );
    let prefix_scale_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_scale_element_count,
        inputs.metal_prefix_cpu_scale_element_count,
    );
    let direct_decode_projection_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_projection_row_count,
        inputs.metal_direct_decode_cpu_projection_row_count,
    );
    let direct_decode_rms_norm_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_rms_norm_element_count,
        inputs.metal_direct_decode_cpu_rms_norm_element_count,
    );
    let direct_decode_ffn_activation_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_ffn_activation_element_count,
        inputs.metal_direct_decode_cpu_ffn_activation_element_count,
    );
    let direct_decode_residual_add_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_residual_add_element_count,
        inputs.metal_direct_decode_cpu_residual_add_element_count,
    );
    let direct_decode_scale_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_scale_element_count,
        inputs.metal_direct_decode_cpu_scale_element_count,
    );
    let prefix_min_mlx_metal_dispatch_share = min_mlx_metal_dispatch_share(&[
        prefix_projection_mlx_metal_dispatch_share,
        prefix_rms_norm_mlx_metal_dispatch_share,
        prefix_ffn_activation_mlx_metal_dispatch_share,
        prefix_residual_add_mlx_metal_dispatch_share,
        prefix_scale_mlx_metal_dispatch_share,
    ]);
    let direct_decode_min_mlx_metal_dispatch_share = min_mlx_metal_dispatch_share(&[
        direct_decode_projection_mlx_metal_dispatch_share,
        direct_decode_rms_norm_mlx_metal_dispatch_share,
        direct_decode_ffn_activation_mlx_metal_dispatch_share,
        direct_decode_residual_add_mlx_metal_dispatch_share,
        direct_decode_scale_mlx_metal_dispatch_share,
    ]);
    let hot_path_cpu_fallback_free = inputs.metal_prefix_cpu_reference_dispatch_count == 0
        && inputs.metal_prefix_cpu_projection_row_count == 0
        && inputs.metal_prefix_cpu_rms_norm_element_count == 0
        && inputs.metal_prefix_cpu_ffn_activation_element_count == 0
        && inputs.metal_prefix_cpu_residual_add_element_count == 0
        && inputs.metal_prefix_cpu_scale_element_count == 0
        && inputs.metal_direct_decode_cpu_projection_row_count == 0
        && inputs.metal_direct_decode_cpu_rms_norm_element_count == 0
        && inputs.metal_direct_decode_cpu_ffn_activation_element_count == 0
        && inputs.metal_direct_decode_cpu_residual_add_element_count == 0
        && inputs.metal_direct_decode_cpu_scale_element_count == 0;
    let batched_direct_decode_logits_ready = !inputs.direct_decode_batching_opportunity_observed
        || (inputs.metal_direct_decode_tokens
            && inputs.metal_direct_decode_batched_logits_group_count > 0
            && inputs.metal_direct_decode_batched_logits_token_count > 0
            && inputs.metal_direct_decode_batched_group_fallback_count == 0
            && inputs.metal_direct_decode_batched_group_fallback_token_count == 0);

    let mut blockers = Vec::new();
    if !inputs.metal_model_artifacts_validated {
        blockers.push("model_artifacts_unvalidated");
    }
    if inputs.native_dense_dequantized_source {
        blockers.push(NATIVE_DENSE_DEQUANTIZED_SOURCE_BLOCKER);
    }
    if inputs.native_quantized_projection_binding_count > 0 {
        blockers.push("native_quantized_projection_kernel_missing");
    }
    if !inputs.metal_complete_model_forward_supported {
        blockers.push("complete_model_forward_not_reported");
    }
    if !inputs.metal_real_model_forward {
        blockers.push("real_model_forward_not_reported");
    }
    if !inputs.mlx_metal_prefix_layers_attention {
        blockers.push("prefix_native_attention_not_observed");
    }
    if inputs.metal_prefix_layers_cpu_reference
        || inputs.metal_prefix_cpu_reference_dispatch_count > 0
    {
        blockers.push("prefix_attention_cpu_reference_remaining");
    }
    if inputs.metal_prefix_cpu_projection_row_count > 0 {
        blockers.push("prefix_projection_cpu_fallback_remaining");
    }
    if inputs.metal_prefix_cpu_rms_norm_element_count > 0 {
        blockers.push("prefix_rms_norm_cpu_fallback_remaining");
    }
    if inputs.metal_prefix_cpu_ffn_activation_element_count > 0 {
        blockers.push("prefix_ffn_cpu_fallback_remaining");
    }
    if inputs.metal_prefix_cpu_residual_add_element_count > 0 {
        blockers.push("prefix_residual_add_cpu_fallback_remaining");
    }
    if inputs.metal_prefix_cpu_scale_element_count > 0 {
        blockers.push("prefix_scale_cpu_fallback_remaining");
    }
    if !inputs.metal_direct_decode_tokens {
        blockers.push("direct_decode_tokens_not_observed");
    } else {
        if !inputs.metal_direct_decode_model_bound_ffn {
            blockers.push("direct_decode_model_bound_ffn_not_reported");
        }
        if inputs.direct_decode_batching_opportunity_observed && !batched_direct_decode_logits_ready
        {
            blockers.push("batched_direct_decode_logits_not_observed");
        }
        if inputs.metal_direct_decode_batched_group_fallback_count > 0
            || inputs.metal_direct_decode_batched_group_fallback_token_count > 0
        {
            blockers.push("batched_direct_decode_group_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_projection_row_count > 0 {
            blockers.push("direct_decode_projection_cpu_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_rms_norm_element_count > 0 {
            blockers.push("direct_decode_rms_norm_cpu_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_ffn_activation_element_count > 0 {
            blockers.push("direct_decode_ffn_cpu_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_residual_add_element_count > 0 {
            blockers.push("direct_decode_residual_add_cpu_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_scale_element_count > 0 {
            blockers.push("direct_decode_scale_cpu_fallback_remaining");
        }
    }

    MlxMetalReadiness {
        status: if blockers.is_empty() {
            "ready"
        } else {
            "coverage_gap"
        },
        hot_path_cpu_fallback_free,
        batched_direct_decode_logits_ready,
        prefix_min_mlx_metal_dispatch_share,
        direct_decode_min_mlx_metal_dispatch_share,
        blockers,
    }
}

pub(crate) fn mlx_metal_readiness_from_route_json(
    tool_mode: &str,
    selected_backend: &str,
    route_json: &Value,
    native_dense_dequantized_source: bool,
) -> MlxMetalReadiness {
    mlx_metal_readiness(MlxMetalReadinessInputs {
        tool_mode,
        selected_backend,
        native_dense_dequantized_source,
        native_quantized_projection_binding_count: route_count_from_json(
            route_json,
            "mlx_metal_projection_source_quantized_binding_count",
            &["metal_dispatch_native_projection_source_quantized_binding_count"],
        ),
        direct_decode_batching_opportunity_observed: route_flag_from_json(
            route_json,
            "metal_direct_decode_batching_opportunity_observed",
            &["metal_direct_decode_batching_opportunity_observed"],
        ) || route_count_from_json(
            route_json,
            "metal_direct_decode_batched_logits_group_count",
            &["metal_dispatch_direct_decode_batched_logits_group_count"],
        ) > 0
            || route_count_from_json(
                route_json,
                "metal_direct_decode_batched_group_fallback_count",
                &["metal_dispatch_direct_decode_batched_group_fallback_count"],
            ) > 0,
        metal_complete_model_forward_supported: route_flag_from_json(
            route_json,
            "metal_complete_model_forward_supported",
            &[
                "metal_dispatch_complete_model_forward_supported",
                "metal_dispatch_runtime_complete_model_forward_supported",
            ],
        ),
        metal_real_model_forward: route_flag_from_json(
            route_json,
            "metal_real_model_forward",
            &["metal_dispatch_real_model_forward"],
        ),
        metal_model_artifacts_validated: route_flag_from_json(
            route_json,
            "metal_model_artifacts_validated",
            &["metal_dispatch_model_artifacts_validated"],
        ),
        mlx_metal_prefix_layers_attention: route_flag_from_json(
            route_json,
            "mlx_metal_prefix_layers_attention",
            &["metal_dispatch_prefix_layers_native_attention"],
        ),
        metal_prefix_layers_cpu_reference: route_flag_from_json(
            route_json,
            "metal_prefix_layers_cpu_reference",
            &["metal_dispatch_prefix_layers_cpu_reference"],
        ),
        metal_prefix_cpu_reference_dispatch_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_reference_dispatch_count",
            &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
        ),
        mlx_metal_prefix_projection_row_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_projection_row_count",
            &["metal_dispatch_prefix_native_projection_row_count"],
        ),
        metal_prefix_cpu_projection_row_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_projection_row_count",
            &["metal_dispatch_prefix_cpu_projection_row_count"],
        ),
        mlx_metal_prefix_rms_norm_element_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_rms_norm_element_count",
            &["metal_dispatch_prefix_native_rms_norm_element_count"],
        ),
        metal_prefix_cpu_rms_norm_element_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_rms_norm_element_count",
            &["metal_dispatch_prefix_cpu_rms_norm_element_count"],
        ),
        mlx_metal_prefix_ffn_activation_element_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_ffn_activation_element_count",
            &["metal_dispatch_prefix_native_ffn_activation_element_count"],
        ),
        metal_prefix_cpu_ffn_activation_element_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_ffn_activation_element_count",
            &["metal_dispatch_prefix_cpu_ffn_activation_element_count"],
        ),
        mlx_metal_prefix_residual_add_element_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_residual_add_element_count",
            &["metal_dispatch_prefix_native_residual_add_element_count"],
        ),
        metal_prefix_cpu_residual_add_element_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_residual_add_element_count",
            &["metal_dispatch_prefix_cpu_residual_add_element_count"],
        ),
        mlx_metal_prefix_scale_element_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_scale_element_count",
            &["metal_dispatch_prefix_native_scale_element_count"],
        ),
        metal_prefix_cpu_scale_element_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_scale_element_count",
            &["metal_dispatch_prefix_cpu_scale_element_count"],
        ),
        metal_direct_decode_tokens: route_flag_from_json(
            route_json,
            "metal_direct_decode_tokens",
            &["metal_dispatch_direct_decode_tokens"],
        ),
        metal_direct_decode_model_bound_ffn: route_flag_from_json(
            route_json,
            "metal_direct_decode_model_bound_ffn",
            &["metal_dispatch_direct_decode_model_bound_ffn"],
        ),
        metal_direct_decode_batched_logits_group_count: route_count_from_json(
            route_json,
            "metal_direct_decode_batched_logits_group_count",
            &["metal_dispatch_direct_decode_batched_logits_group_count"],
        ),
        metal_direct_decode_batched_logits_token_count: route_count_from_json(
            route_json,
            "metal_direct_decode_batched_logits_token_count",
            &["metal_dispatch_direct_decode_batched_logits_token_count"],
        ),
        metal_direct_decode_batched_group_fallback_count: route_count_from_json(
            route_json,
            "metal_direct_decode_batched_group_fallback_count",
            &["metal_dispatch_direct_decode_batched_group_fallback_count"],
        ),
        metal_direct_decode_batched_group_fallback_token_count: route_count_from_json(
            route_json,
            "metal_direct_decode_batched_group_fallback_token_count",
            &["metal_dispatch_direct_decode_batched_group_fallback_token_count"],
        ),
        mlx_metal_direct_decode_projection_row_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_projection_row_count",
            &["metal_dispatch_direct_decode_native_projection_row_count"],
        ),
        metal_direct_decode_cpu_projection_row_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_projection_row_count",
            &["metal_dispatch_direct_decode_cpu_projection_row_count"],
        ),
        mlx_metal_direct_decode_rms_norm_element_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_rms_norm_element_count",
            &["metal_dispatch_direct_decode_native_rms_norm_element_count"],
        ),
        metal_direct_decode_cpu_rms_norm_element_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_rms_norm_element_count",
            &["metal_dispatch_direct_decode_cpu_rms_norm_element_count"],
        ),
        mlx_metal_direct_decode_ffn_activation_element_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_ffn_activation_element_count",
            &["metal_dispatch_direct_decode_native_ffn_activation_element_count"],
        ),
        metal_direct_decode_cpu_ffn_activation_element_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_ffn_activation_element_count",
            &["metal_dispatch_direct_decode_cpu_ffn_activation_element_count"],
        ),
        mlx_metal_direct_decode_residual_add_element_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_residual_add_element_count",
            &["metal_dispatch_direct_decode_native_residual_add_element_count"],
        ),
        metal_direct_decode_cpu_residual_add_element_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_residual_add_element_count",
            &["metal_dispatch_direct_decode_cpu_residual_add_element_count"],
        ),
        mlx_metal_direct_decode_scale_element_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_scale_element_count",
            &["metal_dispatch_direct_decode_native_scale_element_count"],
        ),
        metal_direct_decode_cpu_scale_element_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_scale_element_count",
            &["metal_dispatch_direct_decode_cpu_scale_element_count"],
        ),
    })
}
