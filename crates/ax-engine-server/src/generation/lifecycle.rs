use std::sync::atomic::Ordering;

use ax_engine_sdk::{EngineStepReport, SessionRequestReport};
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::errors::{
    ErrorResponse, admission_error_response, error_response, map_generation_service_error,
    request_not_found_response,
};
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::validate_model;

pub(crate) async fn submit_request(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<(StatusCode, Json<SessionRequestReport>), (StatusCode, Json<ErrorResponse>)> {
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

    let live = state.snapshot();
    validate_model(&live, request.model.as_deref())?;

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
    let live = state.snapshot();
    let report = live
        .generation_service
        .request_snapshot(request_id)
        .await
        .map_err(|error| match error {
            crate::generation::service::GenerationServiceError::Engine(
                ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation { .. },
            ) => request_not_found_response(request_id),
            error => map_generation_service_error(error),
        })?;
    Ok(Json(report))
}

pub(crate) async fn cancel_request(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    let report = live
        .generation_service
        .cancel_stepwise(request_id)
        .await
        .map_err(|error| match error {
            crate::generation::service::GenerationServiceError::Engine(
                ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation { .. },
            ) => request_not_found_response(request_id),
            error => map_generation_service_error(error),
        })?;
    Ok(Json(report))
}

pub(crate) async fn step_request(
    State(state): State<AppState>,
) -> Result<Json<EngineStepReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    let report = live
        .generation_service
        .advance()
        .await
        .map_err(map_generation_service_error)?;
    Ok(Json(report))
}
