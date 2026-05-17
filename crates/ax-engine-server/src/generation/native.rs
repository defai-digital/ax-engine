use std::sync::Arc;

use ax_engine_sdk::{GenerateRequest, GenerateResponse};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::errors::ErrorResponse;
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::validate_model;

pub(crate) async fn generate(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_model(&state, request.model.as_deref())?;

    let request = build_generate_request(&state, request);
    let (_, response) = run_stateless_generate_request(&state, request).await?;

    Ok(Json(response))
}

pub(crate) async fn run_stateless_generate_request(
    state: &AppState,
    request: GenerateRequest,
) -> Result<(u64, GenerateResponse), (StatusCode, Json<ErrorResponse>)> {
    let request_id = state.allocate_request_id();
    let context = Arc::clone(&state.stateless_generate_context);
    let response = crate::run_blocking_session_task(move || {
        context.generate_with_request_id(request_id, request)
    })
    .await?;

    Ok((request_id, response))
}
