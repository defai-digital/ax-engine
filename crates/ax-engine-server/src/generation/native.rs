use std::sync::Arc;

use ax_engine_sdk::{GenerateRequest, GenerateResponse};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::errors::ErrorResponse;
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::validate_model;
use crate::tasks::run_blocking_session_task;

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
    if state.uses_native_mlx_backend() {
        let (session_config, mlx_prefix_cache) = state.request_session_parts();
        let response = run_blocking_session_task(move || {
            let mut session =
                AppState::build_request_session_from_parts(session_config, mlx_prefix_cache)?;
            session.generate_with_request_id(request_id, request)
        })
        .await?;

        return Ok((request_id, response));
    }

    let context = Arc::clone(&state.stateless_generate_context);
    let response =
        run_blocking_session_task(move || context.generate_with_request_id(request_id, request))
            .await?;

    Ok((request_id, response))
}
