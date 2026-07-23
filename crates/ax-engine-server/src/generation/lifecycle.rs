use ax_engine_sdk::{EngineStepReport, SessionRequestReport};
use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use serde::Deserialize;

use crate::app_state::{AppState, LiveState};
use crate::errors::{
    ErrorResponse, admission_error_response, map_generation_service_error,
    request_not_found_response,
};
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::select_model;

pub(crate) async fn submit_request(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<(StatusCode, Json<SessionRequestReport>), (StatusCode, Json<ErrorResponse>)> {
    let live = select_model(&state, request.model.as_deref())?;
    request.reject_video_inputs(&live)?;

    let request_id = state.allocate_request_id();
    let request = build_generate_request(&live, request);
    let generation_service = live.generation_service.clone();
    let permit = state.try_admit(&live).map_err(admission_error_response)?;
    let mut owner = RequestOwnerRegistration::new(state.clone(), request_id, &live);
    let report = generation_service
        .submit_stepwise(request_id, request, permit)
        .await
        .map_err(map_generation_service_error)?;
    owner.commit();

    Ok((StatusCode::CREATED, Json(report)))
}

struct RequestOwnerRegistration {
    state: AppState,
    request_id: u64,
    committed: bool,
}

impl RequestOwnerRegistration {
    fn new(state: AppState, request_id: u64, live: &LiveState) -> Self {
        state.register_request_owner(request_id, live);
        Self {
            state,
            request_id,
            committed: false,
        }
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for RequestOwnerRegistration {
    fn drop(&mut self) {
        if !self.committed {
            self.state.remove_request_owner(self.request_id);
        }
    }
}

pub(crate) async fn request_snapshot(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = state
        .snapshot_for_request(request_id)
        .ok_or_else(|| request_not_found_response(request_id))?;
    live.generation_service
        .request_snapshot(request_id)
        .await
        .map(Json)
        .map_err(|error| map_owned_request_error(&state, request_id, error))
}

pub(crate) async fn cancel_request(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = state
        .snapshot_for_request(request_id)
        .ok_or_else(|| request_not_found_response(request_id))?;
    live.generation_service
        .cancel_stepwise(request_id)
        .await
        .map(Json)
        .map_err(|error| map_owned_request_error(&state, request_id, error))
}

fn map_owned_request_error(
    state: &AppState,
    request_id: u64,
    error: crate::generation::service::GenerationServiceError,
) -> (StatusCode, Json<ErrorResponse>) {
    if matches!(
        &error,
        crate::generation::service::GenerationServiceError::Engine(
            ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation { .. }
        )
    ) {
        state.remove_request_owner(request_id);
        request_not_found_response(request_id)
    } else {
        map_generation_service_error(error)
    }
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
