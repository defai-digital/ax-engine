use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use ax_engine_sdk::{EngineSession, EngineSessionConfig, EngineSessionError};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::app_state::{AppState, build_live_state};
use crate::errors::{ErrorResponse, error_response};

type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);

#[derive(Debug, Deserialize)]
pub(crate) struct LoadModelRequest {
    /// Identifier the server will advertise for this model.
    pub model_id: String,
    /// Path to the MLX model artifacts directory (must contain model-manifest.json).
    pub model_path: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct LoadModelResponse {
    pub model_id: String,
    pub state: &'static str,
    pub context_length: u32,
}

pub(crate) async fn load_model(
    State(state): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> Result<Json<LoadModelResponse>, HttpErrorResponse> {
    if request.model_id.trim().is_empty() {
        return Err(error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "model_id must not be empty".to_string(),
        ));
    }

    let model_path = PathBuf::from(&request.model_path);
    if !model_path.exists() {
        return Err(error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            format!("model_path does not exist: {}", request.model_path),
        ));
    }

    // Prevent concurrent loads — 409 if already in progress (mirrors ax-serving).
    if state
        .loading
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(error_response(
            StatusCode::CONFLICT,
            "model_loading",
            "a model load is already in progress".to_string(),
        ));
    }

    // Reject the swap while the current session has non-terminal stepwise
    // (/v1/requests, /v1/step) work. Request state lives entirely inside the
    // EngineSession instance with no cross-session registry, so replacing it
    // mid-flight would silently orphan those requests: the client's next
    // /v1/requests/:id or /v1/step call would find nothing (a bare "not
    // found" instead of a real terminal state), and the request's GPU/KV
    // resources would only be reclaimed once the old session's last Arc
    // reference drops. Fail closed instead — mirrors the concurrent-load 409
    // above.
    if state
        .snapshot()
        .request_session
        .lock()
        .await
        .has_active_stepwise_requests()
    {
        state.loading.store(false, Ordering::Release);
        return Err(error_response(
            StatusCode::CONFLICT,
            "requests_in_flight",
            "the current session has non-terminal /v1/requests work; drain or cancel it before \
             loading a new model"
                .to_string(),
        ));
    }

    // Clone the current config and swap in the new model artifacts dir.
    // All other KV / backend settings are inherited from the running config.
    // Only the MLX-native backend reads mlx_model_artifacts_dir — on a
    // delegated backend (mlx-lm, llama.cpp) the rebuilt session would silently
    // keep serving the old model under the new model_id, so reject up front.
    let new_config = {
        let live = state.snapshot();
        if !live
            .session_config
            .resolved_backend
            .selected_backend
            .is_mlx()
        {
            state.loading.store(false, Ordering::Release);
            return Err(error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_backend",
                "model load is only supported on the MLX-native backend".to_string(),
            ));
        }
        Arc::unwrap_or_clone(live.session_config).with_mlx_model_artifacts_dir(model_path.clone())
    };

    let model_id = request.model_id.clone();

    // Run load + swap in a detached task: axum drops handler futures when the
    // client disconnects, and the load must still complete (or fail) and clear
    // the loading flag either way.
    let state_clone = state.clone();
    let load_task = tokio::spawn(async move {
        // Load the session on the blocking thread pool — weight loading can take
        // tens of seconds; blocking the async runtime would stall all other requests.
        let result =
            tokio::task::spawn_blocking(move || build_new_session(model_id, new_config)).await;
        let response = match result {
            Ok(Ok(live)) => {
                let ctx_len = crate::metadata::context_length(&live);
                state_clone.swap_live(live);
                Ok(ctx_len)
            }
            Ok(Err(e)) => Err(error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "load_failed",
                format!("failed to load model: {e}"),
            )),
            Err(e) => Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                format!("load task panicked: {e}"),
            )),
        };
        // Always clear the loading flag, even on failure.
        state_clone.loading.store(false, Ordering::Release);
        response
    });

    match load_task.await {
        Ok(Ok(ctx_len)) => Ok(Json(LoadModelResponse {
            model_id: request.model_id,
            state: "loaded",
            context_length: ctx_len,
        })),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("load task panicked: {e}"),
        )),
    }
}

fn build_new_session(
    model_id: String,
    config: EngineSessionConfig,
) -> Result<crate::app_state::LiveState, EngineSessionError> {
    let session = EngineSession::new(config)?;
    build_live_state(model_id, session)
}
