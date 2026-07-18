use std::sync::atomic::Ordering;

use ax_engine_sdk::{EngineStepReport, SessionRequestReport};
use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use serde::Deserialize;

use crate::app_state::AppState;
use crate::errors::{
    ErrorResponse, admission_error_response, error_response, map_generation_service_error,
    request_not_found_response,
};
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::select_model;

pub(crate) async fn submit_request(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<(StatusCode, Json<SessionRequestReport>), (StatusCode, Json<ErrorResponse>)> {
    request.reject_video_inputs()?;
    // Return the lifecycle-specific conflict before request construction;
    // generation-bound admission below remains the final race guard.
    if state.loading.load(Ordering::Acquire) {
        return Err(error_response(
            StatusCode::CONFLICT,
            "model_loading",
            "a model load is in progress; new stepwise requests are rejected until it \
             completes to avoid submitting into a session that is about to be replaced"
                .to_string(),
        ));
    }

    let live = select_model(&state, request.model.as_deref())?;

    let request_id = state.allocate_request_id();
    let request = build_generate_request(&live, request);
    let generation_service = live.generation_service.clone();
    let permit = state.try_admit(&live).map_err(admission_error_response)?;
    let report = generation_service
        .submit_stepwise(request_id, request, permit)
        .await
        .map_err(map_generation_service_error)?;

    Ok((StatusCode::CREATED, Json(report)))
}

pub(crate) async fn request_snapshot(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    for live in state.snapshots() {
        match live.generation_service.request_snapshot(request_id).await {
            Ok(report) => return Ok(Json(report)),
            Err(crate::generation::service::GenerationServiceError::Engine(
                ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation { .. },
            )) => {}
            Err(error) => return Err(map_generation_service_error(error)),
        }
    }
    Err(request_not_found_response(request_id))
}

pub(crate) async fn cancel_request(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    for live in state.snapshots() {
        match live.generation_service.request_snapshot(request_id).await {
            Ok(_) => {
                let report = live
                    .generation_service
                    .cancel_stepwise(request_id)
                    .await
                    .map_err(map_generation_service_error)?;
                return Ok(Json(report));
            }
            Err(crate::generation::service::GenerationServiceError::Engine(
                ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation { .. },
            )) => {}
            Err(error) => return Err(map_generation_service_error(error)),
        }
    }
    Err(request_not_found_response(request_id))
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct StepQuery {
    model: Option<String>,
}

pub(crate) async fn step_request(
    State(state): State<AppState>,
    Query(query): Query<StepQuery>,
) -> Result<Json<EngineStepReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = select_model(&state, query.model.as_deref())?;
    let report = live
        .generation_service
        .advance()
        .await
        .map_err(map_generation_service_error)?;
    Ok(Json(report))
}
