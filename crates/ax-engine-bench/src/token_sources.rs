use ax_engine_sdk::GenerateResponse;

pub(crate) fn prompt_token_count_source(
    response: &GenerateResponse,
    used_manifest_target: bool,
) -> &'static str {
    if response.prompt_token_count.is_some() {
        "backend_reported_usage"
    } else if !response.prompt_tokens.is_empty() {
        "token_array"
    } else if used_manifest_target {
        "manifest_synthetic_target"
    } else {
        "unknown"
    }
}

pub(crate) fn output_token_count_source(
    response: &GenerateResponse,
    used_synthetic_text_estimate: bool,
) -> &'static str {
    if response.output_token_count.is_some() {
        "backend_reported_usage"
    } else if !response.output_tokens.is_empty() {
        "token_array"
    } else if used_synthetic_text_estimate {
        "synthetic_text_estimate"
    } else {
        "unknown"
    }
}
