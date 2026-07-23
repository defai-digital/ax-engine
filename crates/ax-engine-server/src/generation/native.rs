use std::sync::Arc;

use ax_engine_sdk::{GenerateRequest, GenerateResponse};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::app_state::{AppState, LiveState};
use crate::errors::{ErrorResponse, admission_error_response, map_generation_service_error};
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::select_model;
use crate::tasks::run_blocking_session_task;

pub(crate) async fn generate(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let live = select_model(&state, request.model.as_deref())?;
    request.reject_video_inputs(&live)?;

    let request = build_generate_request(&live, request);
    let (_, response) = run_stateless_generate_request(&state, &live, request).await?;

    Ok(Json(response))
}

/// Run a request against the caller's model generation.
/// Admission keeps an admitted generation alive through completion and rejects
/// a stale snapshot when a hot-swap finished while the request was prepared.
pub(crate) async fn run_stateless_generate_request(
    state: &AppState,
    live: &LiveState,
    request: GenerateRequest,
) -> Result<(u64, GenerateResponse), (StatusCode, Json<ErrorResponse>)> {
    let permit = state.try_admit(live).map_err(admission_error_response)?;
    let request_id = state.allocate_request_id();
    if live.runtime_report.selected_backend.is_mlx() {
        let generation_service = live.generation_service.clone();
        let response = generation_service
            .generate(request_id, request, permit)
            .await
            .map_err(map_generation_service_error)?;
        return Ok((request_id, response));
    }

    let context = Arc::clone(&live.stateless_generate_context);
    let response = run_blocking_session_task(move || {
        let _permit = permit;
        context.generate_with_request_id(request_id, request)
    })
    .await?;

    Ok((request_id, response))
}
