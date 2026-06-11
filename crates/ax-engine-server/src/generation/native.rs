use std::sync::Arc;

use ax_engine_sdk::{GenerateRequest, GenerateResponse};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::app_state::{AppState, LiveState};
use crate::errors::ErrorResponse;
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::validate_model;
use crate::tasks::run_blocking_session_task;

pub(crate) async fn generate(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    validate_model(&live, request.model.as_deref())?;

    let request = build_generate_request(&live, request);
    let (_, response) = run_stateless_generate_request(&state, &live, request).await?;

    Ok(Json(response))
}

/// Runs a generate request against the caller's `LiveState` snapshot, so the
/// model that validated/tokenized the request is the one that executes it
/// even if a hot-swap lands mid-request.
pub(crate) async fn run_stateless_generate_request(
    state: &AppState,
    live: &LiveState,
    request: GenerateRequest,
) -> Result<(u64, GenerateResponse), (StatusCode, Json<ErrorResponse>)> {
    let request_id = state.allocate_request_id();
    let context = Arc::clone(&live.stateless_generate_context);
    let response =
        run_blocking_session_task(move || context.generate_with_request_id(request_id, request))
            .await?;

    Ok((request_id, response))
}
