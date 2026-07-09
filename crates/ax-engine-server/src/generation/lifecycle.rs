use std::sync::atomic::Ordering;

use ax_engine_sdk::{EngineStepReport, SessionRequestReport};
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::errors::{ErrorResponse, error_response, map_session_error, request_not_found_response};
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::validate_model;
use crate::tasks::run_blocking_session_task;

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
    let request_session = live.request_session.clone();
    let report = run_blocking_session_task(move || {
        let mut session = request_session.blocking_lock();
        let request_id = session.submit_generate_with_request_id(request_id, request)?;
        session.request_report(request_id).ok_or(
            ax_engine_sdk::EngineSessionError::RequestReportInvariantViolation {
                request_id,
                message: "request missing immediately after submission",
            },
        )
    })
    .await?;

    Ok((StatusCode::CREATED, Json(report)))
}

pub(crate) async fn request_snapshot(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    let session = live.request_session.lock().await;
    let report = session
        .request_report(request_id)
        .ok_or_else(|| request_not_found_response(request_id))?;

    Ok(Json(report))
}

pub(crate) async fn cancel_request(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    let mut session = live.request_session.lock().await;
    if session.request_report(request_id).is_none() {
        return Err(request_not_found_response(request_id));
    }

    session
        .cancel_request(request_id)
        .map_err(map_session_error)?;
    let report = session
        .request_report(request_id)
        .ok_or_else(|| request_not_found_response(request_id))?;

    Ok(Json(report))
}

pub(crate) async fn step_request(
    State(state): State<AppState>,
) -> Result<Json<EngineStepReport>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    let request_session = live.request_session.clone();
    let report = run_blocking_session_task(move || {
        let mut session = request_session.blocking_lock();
        session.step_report()
    })
    .await?;
    state.metrics.record_step_report(&report);
    Ok(Json(report))
}
