use std::sync::atomic::Ordering;

use ax_engine_sdk::{EngineStepReport, SessionRequestReport, SessionRequestState};
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::errors::{
    ErrorResponse, admission_error_response, error_response, map_generation_service_error,
    map_session_error, request_not_found_response,
};
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::validate_model;

pub(crate) async fn submit_request(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<(StatusCode, Json<SessionRequestReport>), (StatusCode, Json<ErrorResponse>)> {
    // `load_model` only checks for in-flight stepwise requests once, before
    // starting a load that can take tens of seconds; it does not hold any
    // lock across that window. Without this check, a request submitted
    // after that snapshot but before the swap lands in the session that's
    // about to be replaced, and is silently orphaned: the client's next
    // /v1/requests/:id or /v1/step call 404s instead of surfacing a real
    // terminal state. Reject up front instead, mirroring load_model's own
    // concurrent-load 409.
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
    let permit = state
        .admission
        .try_admit()
        .map_err(admission_error_response)?;
    let stepwise_admission = state.stepwise_admission.clone();
    let report = generation_service
        .execute(move |session| {
            let request_id = session.submit_generate_with_request_id(request_id, request)?;
            let report = session.request_report(request_id).ok_or(
                ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation {
                    request_id,
                    message: "request missing immediately after submission",
                },
            )?;
            let replaced = stepwise_admission.lock().insert(request_id, permit);
            debug_assert!(replaced.is_none(), "request IDs are process-unique");
            Ok(report)
        })
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
        .execute(move |session| {
            session.request_report(request_id).ok_or(
                ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation {
                    request_id,
                    message: "request missing from preview session state",
                },
            )
        })
        .await
        .map_err(|error| match error {
            crate::generation::service::GenerationServiceError::Engine(
                ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation { .. },
            ) => request_not_found_response(request_id),
            error => map_generation_service_error(error),
        })?;
    release_terminal_stepwise_permit(&state, &report);

    Ok(Json(report))
}

pub(crate) async fn cancel_request(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    let report = live
        .generation_service
        .execute(move |session| {
            if session.request_report(request_id).is_none() {
                return Err(
                    ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation {
                        request_id,
                        message: "request missing from preview session state",
                    },
                );
            }
            session.cancel_request(request_id)?;
            session.request_report(request_id).ok_or(
                ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation {
                    request_id,
                    message: "request missing after cancellation",
                },
            )
        })
        .await
        .map_err(|error| match error {
            crate::generation::service::GenerationServiceError::Engine(
                ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation { .. },
            ) => request_not_found_response(request_id),
            error => map_generation_service_error(error),
        })?;
    release_terminal_stepwise_permit(&state, &report);

    Ok(Json(report))
}

pub(crate) async fn step_request(
    State(state): State<AppState>,
) -> Result<Json<EngineStepReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    let generation_service = live.generation_service.clone();
    let request_ids = state
        .stepwise_admission
        .lock()
        .keys()
        .copied()
        .collect::<Vec<_>>();
    let (report, terminal_request_ids) = generation_service
        .execute(move |session| {
            let report = session.step_report();
            let terminal_request_ids: Vec<u64> = request_ids
                .into_iter()
                .filter(|request_id| {
                    session
                        .request_report(*request_id)
                        .is_some_and(|request| request_state_is_terminal(request.state))
                })
                .collect();
            Ok((report, terminal_request_ids))
        })
        .await
        .map_err(map_generation_service_error)?;
    let mut permits = state.stepwise_admission.lock();
    for request_id in terminal_request_ids {
        permits.remove(&request_id);
    }
    drop(permits);
    let report = report.map_err(map_session_error)?;
    state.metrics.record_step_report(&report);
    Ok(Json(report))
}

fn release_terminal_stepwise_permit(state: &AppState, report: &SessionRequestReport) {
    if request_state_is_terminal(report.state) {
        state.stepwise_admission.lock().remove(&report.request_id);
    }
}

fn request_state_is_terminal(state: SessionRequestState) -> bool {
    matches!(
        state,
        SessionRequestState::Finished
            | SessionRequestState::Cancelled
            | SessionRequestState::Failed
    )
}
