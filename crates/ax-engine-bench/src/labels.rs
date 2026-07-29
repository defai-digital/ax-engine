use ax_engine_core::StopReason;
use ax_engine_sdk::{
    GenerateFinishReason, GenerateStatus, SelectedBackend, SessionRequestState, SupportTier,
};

pub(crate) fn selected_backend_label(backend: SelectedBackend) -> &'static str {
    match backend {
        SelectedBackend::Mlx => "mlx",
        SelectedBackend::MlxLmDelegated => "mlx_lm_delegated",
        SelectedBackend::LlamaCpp => "llama_cpp",
    }
}

pub(crate) fn support_tier_label(support_tier: SupportTier) -> &'static str {
    match support_tier {
        SupportTier::MlxCertified => "mlx_certified",
        SupportTier::MlxPreview => "mlx_preview",
        SupportTier::MlxLmDelegated => "mlx_lm_delegated",
        SupportTier::LlamaCpp => "llama_cpp",
        SupportTier::Unsupported => "unsupported",
    }
}

pub(crate) fn request_state_label(state: SessionRequestState) -> &'static str {
    match state {
        SessionRequestState::Waiting => "waiting",
        SessionRequestState::Runnable => "runnable",
        SessionRequestState::Running => "running",
        SessionRequestState::BlockedOnMemory => "blocked_on_memory",
        SessionRequestState::Finished => "finished",
        SessionRequestState::Cancelled => "cancelled",
        SessionRequestState::Failed => "failed",
    }
}

pub(crate) fn llama_cpp_request_is_terminal(state: SessionRequestState) -> bool {
    matches!(
        state,
        SessionRequestState::Finished
            | SessionRequestState::Cancelled
            | SessionRequestState::Failed
    )
}

pub(crate) fn llama_cpp_final_request_state_label(state: SessionRequestState) -> &'static str {
    match state {
        SessionRequestState::Finished => "Finished",
        SessionRequestState::Cancelled => "Cancelled",
        SessionRequestState::Failed => "Failed",
        SessionRequestState::Waiting
        | SessionRequestState::Runnable
        | SessionRequestState::Running
        | SessionRequestState::BlockedOnMemory => "Running",
    }
}

pub(crate) fn generate_status_label(status: GenerateStatus) -> &'static str {
    match status {
        GenerateStatus::Pending => "pending",
        GenerateStatus::Finished => "finished",
        GenerateStatus::Cancelled => "cancelled",
        GenerateStatus::Failed => "failed",
    }
}

pub(crate) fn generate_finish_reason_label(finish_reason: GenerateFinishReason) -> &'static str {
    match finish_reason {
        GenerateFinishReason::Stop => "stop",
        GenerateFinishReason::MaxOutputTokens => "max_output_tokens",
        GenerateFinishReason::ContentFilter => "content_filter",
        GenerateFinishReason::Cancelled => "cancelled",
        GenerateFinishReason::Error => "error",
    }
}

pub(crate) fn optional_route_label(value: Option<&str>) -> &str {
    value.unwrap_or("none")
}

#[cfg(test)]
pub(crate) fn optional_u32_label(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

pub(crate) fn stop_reason_from_generate_finish_reason(
    finish_reason: Option<GenerateFinishReason>,
) -> Option<StopReason> {
    match finish_reason {
        Some(GenerateFinishReason::Stop) => Some(StopReason::EosToken),
        Some(GenerateFinishReason::MaxOutputTokens) => Some(StopReason::MaxOutputTokens),
        Some(GenerateFinishReason::ContentFilter) => Some(StopReason::Error),
        Some(GenerateFinishReason::Cancelled) => Some(StopReason::Cancelled),
        Some(GenerateFinishReason::Error) => Some(StopReason::Error),
        None => None,
    }
}

pub(crate) fn compare_result_label(tool_mode: &str, execution_semantics: &str) -> &'static str {
    match tool_mode {
        "llama_cpp_stepwise_runtime" => "llama_cpp_stepwise_compare",
        "llama_cpp_blocking_runtime" => "llama_cpp_blocking_compare",
        _ => match execution_semantics {
            "metal_real_model_forward" => "metal_real_model_forward_compare",
            "metal_multilayer_mixed_prefix_attention" => {
                "metal_multilayer_mixed_prefix_attention_compare"
            }
            "metal_multilayer_native_prefix_attention" => {
                "metal_multilayer_native_prefix_attention_compare"
            }
            "metal_multilayer_model_incomplete" => "metal_multilayer_model_incomplete_compare",
            "metal_model_bound_ffn_decode" => "metal_model_bound_ffn_decode_compare",
            "metal_real_model_tensor_inputs" => "metal_real_model_tensor_inputs_compare",
            "metal_model_conditioned_numeric_scaffold" => {
                "metal_model_conditioned_scaffold_compare"
            }
            "metal_numeric_scaffold_only" => "metal_numeric_scaffold_compare",
            _ => "engine_bringup_compare",
        },
    }
}

pub(crate) fn compare_summary_note(tool_mode: &str, execution_semantics: &str) -> &'static str {
    match tool_mode {
        "llama_cpp_blocking_runtime" => {
            "\nThis comparison reflects the current llama.cpp llama.cpp benchmark path. TTFT, prefill throughput, and decode throughput should be read as delegated llama.cpp wall-time proxies rather than MLX-mode scheduler measurements.\n"
        }
        "llama_cpp_stepwise_runtime" => {
            "\nThis comparison reflects the delegated llama.cpp benchmark path over the stepwise `llama.cpp /completion` adapter. Throughput and prefix-reuse deltas should be read as delegated request-cadence and backend-managed prompt-cache evidence, not as AX Engine scheduler, KV, or runner evidence.\n"
        }
        _ if execution_semantics == "metal_real_model_forward" => {
            "\nThis comparison reflects Metal runs that explicitly marked real model forward execution. Compare only against the same execution semantics and validated model provenance.\n"
        }
        _ if execution_semantics == "metal_multilayer_mixed_prefix_attention" => {
            "\nThis comparison reflects multi-layer Metal runs where some prefix-layer attention already executed through MLX Metal dispatch while part of the prefix path still fell back to CPU reference. Read it as partial MLX dense-forward progress, not as full MLX prefix coverage or release-grade inference.\n"
        }
        _ if execution_semantics == "metal_multilayer_native_prefix_attention" => {
            "\nThis comparison reflects Metal runs where validated multi-layer model artifacts were present and at least part of prefix-layer attention already executed through MLX Metal dispatch, but the full dense forward path still remained incomplete. Treat these runs as stronger bring-up evidence than CPU-reference multilayer staging, but not as release-grade MLX inference.\n"
        }
        _ if execution_semantics == "metal_multilayer_model_incomplete" => {
            "\nThis comparison reflects Metal runs that loaded validated multi-layer model artifacts, but did not report a complete real model forward path. Treat these runs as partial MLX Metal bring-up milestones rather than final dense inference performance.\n"
        }
        _ if execution_semantics == "metal_model_bound_ffn_decode" => {
            "\nThis comparison reflects Metal runs that consumed real model tensor inputs and reported model-bound FFN direct decode continuation. It is stronger evidence than tensor-input bring-up alone, but it is still not a full release-grade native forward path.\n"
        }
        _ if execution_semantics == "metal_real_model_tensor_inputs" => {
            "\nThis comparison reflects Metal runs with real model tensor inputs present, but without a reported real model forward pass. Treat deltas as model-bound bring-up progress rather than final inference performance.\n"
        }
        _ if execution_semantics == "metal_model_conditioned_numeric_scaffold" => {
            "\nThis comparison reflects Metal runs with model-conditioned staged inputs that still remain numeric scaffold bring-up paths. Do not use these deltas as release-grade inference conclusions.\n"
        }
        _ if execution_semantics == "metal_numeric_scaffold_only" => {
            "\nThis comparison reflects Metal numeric scaffold bring-up runs. The command queue, kernels, and runtime may be real, but the workload is still not a full model forward.\n"
        }
        _ => {
            "\nThis comparison reflects the current engine bring-up benchmark path and still requires human review before performance conclusions are used for release decisions.\n"
        }
    }
}
