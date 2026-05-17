use ax_engine_sdk::SelectedBackend;
use axum::Json;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::errors::{ErrorResponse, error_response};

pub(crate) fn validate_openai_request(
    state: &AppState,
    model: Option<&str>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    validate_openai_text_backend(state)?;
    validate_model(state, model)
}

pub(crate) fn validate_openai_text_backend(
    state: &AppState,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if !matches!(
        state.runtime_report.selected_backend,
        SelectedBackend::LlamaCpp | SelectedBackend::MlxLmDelegated
    ) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "OpenAI-compatible text endpoints require a llama.cpp or mlx_lm_delegated backend; use /v1/generate for repo-owned MLX preview".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_model(
    state: &AppState,
    request_model: Option<&str>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(model) = request_model {
        if model != state.model_id.as_ref() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "model_mismatch",
                format!(
                    "requested model_id {model} does not match configured preview model {}",
                    state.model_id.as_ref()
                ),
            ));
        }
    }

    Ok(())
}
