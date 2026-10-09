use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::SelectedBackend;
use axum::Json;
use axum::http::StatusCode;
use serde_json::Value;

use crate::app_state::{AppState, LiveState};
use crate::errors::{ErrorResponse, error_response};

#[cfg(test)]
pub(crate) fn validate_openai_request(
    live: &LiveState,
    model: Option<&str>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    validate_openai_text_backend(live)?;
    validate_model(live, model)
}

pub(crate) fn select_model(
    state: &AppState,
    requested_model: Option<&str>,
) -> Result<LiveState, (StatusCode, Json<ErrorResponse>)> {
    state.snapshot_for_model(requested_model).ok_or_else(|| {
        let requested = requested_model.unwrap_or_default();
        let loaded = state.model_ids().join(", ");
        error_response(
            StatusCode::BAD_REQUEST,
            "model_not_found",
            format!("requested model_id {requested} is not loaded; loaded models: {loaded}"),
        )
    })
}

pub(crate) fn select_openai_model(
    state: &AppState,
    requested_model: Option<&str>,
) -> Result<LiveState, (StatusCode, Json<ErrorResponse>)> {
    let live = select_model(state, requested_model)?;
    validate_openai_text_backend(&live)?;
    Ok(live)
}

pub(crate) fn validate_openai_text_backend(
    live: &LiveState,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if crate::metadata::model_family_from_artifacts(live).as_deref() == Some("whisper") {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Whisper is a speech-to-text model; use /v1/audio/transcriptions or /v1/audio/translations"
                .to_string(),
        ));
    }
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

#[cfg(test)]
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

pub(crate) fn openai_value_is_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(object) => !object.is_empty(),
        Value::Number(value) => json_number_is_nonzero(value),
    }
}

pub(crate) fn json_number_is_nonzero(value: &serde_json::Number) -> bool {
    value
        .as_i64()
        .map(|value| value != 0)
        .or_else(|| value.as_u64().map(|value| value != 0))
        .or_else(|| value.as_f64().map(|value| value != 0.0))
        .unwrap_or(true)
}

pub(crate) fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) fn openai_tool_choice_enables_tool_call(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "auto" | "none" | "false" | "off")
        }
        Value::Array(values) => !values.is_empty(),
        Value::Object(object) => !object.is_empty(),
        Value::Number(value) => json_number_is_nonzero(value),
    }
}
