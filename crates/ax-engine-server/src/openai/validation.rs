use ax_engine_sdk::SelectedBackend;
use axum::Json;
use axum::http::StatusCode;

use crate::app_state::LiveState;
use crate::errors::{ErrorResponse, error_response};

pub(crate) fn validate_openai_request(
    live: &LiveState,
    model: Option<&str>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    validate_openai_text_backend(live)?;
    validate_model(live, model)
}

pub(crate) fn validate_openai_text_backend(
    live: &LiveState,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if !matches!(
        live.runtime_report.selected_backend,
        SelectedBackend::LlamaCpp | SelectedBackend::MlxLmDelegated | SelectedBackend::Mlx
    ) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "OpenAI-compatible text endpoints require a text-capable backend".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_model(
    live: &LiveState,
    request_model: Option<&str>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(model) = request_model {
        if model != live.model_id.as_ref() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "model_mismatch",
                format!(
                    "requested model_id {model} does not match configured preview model {}",
                    live.model_id.as_ref()
                ),
            ));
        }
    }

    Ok(())
}
