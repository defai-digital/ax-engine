use ax_engine_sdk::RuntimeReport;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Serialize;
use serde_json::json;

use crate::app_state::AppState;
use crate::errors::{ErrorResponse, error_response};

#[derive(Debug, Serialize)]
pub(crate) struct ServerInfoResponse {
    service: &'static str,
    model_id: String,
    deterministic: bool,
    max_batch_tokens: u32,
    block_size_tokens: u32,
    runtime: RuntimeResponse,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    runtime: RuntimeResponse,
}

pub(crate) type RuntimeResponse = RuntimeReport;

pub(crate) async fn health(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    // `/health` is the readiness probe most callers (bench harness,
    // k8s, load balancers) poll while a server starts. Returning 200
    // when the server has bound a port but the inference session is
    // wedged (deadlocked on another in-flight call, runtime panicked,
    // weights not loadable on this device, etc.) sends those callers
    // into the failure pattern below. A `try_lock` is a sub-us probe
    // that confirms the session mutex is grabbable, which is the
    // strongest "ready" signal we can give without doing real work.
    let session_lock = state.request_session.try_lock();
    if session_lock.is_err() {
        return Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "session_busy",
            "ax-engine-server has not finished initialising its inference session".into(),
        ));
    }
    drop(session_lock);
    Ok(Json(json!({
        "status": "ok",
        "service": "ax-engine-server",
        "model_id": state.model_id.as_ref(),
        "runtime": runtime_response(&state),
    })))
}

pub(crate) async fn runtime_info(State(state): State<AppState>) -> Json<ServerInfoResponse> {
    Json(server_info_response(&state))
}

pub(crate) async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: state.model_id.to_string(),
            object: "model",
            owned_by: "ax-engine-v4",
            runtime: runtime_response(&state),
        }],
    })
}

fn server_info_response(state: &AppState) -> ServerInfoResponse {
    ServerInfoResponse {
        service: "ax-engine-server",
        model_id: state.model_id.to_string(),
        deterministic: state.session_config.deterministic,
        max_batch_tokens: state.session_config.max_batch_tokens,
        block_size_tokens: state.session_config.kv_config.block_size_tokens,
        runtime: runtime_response(state),
    }
}

fn runtime_response(state: &AppState) -> RuntimeResponse {
    state.runtime_report.clone()
}
