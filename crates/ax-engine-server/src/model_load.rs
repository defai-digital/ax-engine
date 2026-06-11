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

    // Clone the current config and swap in the new model artifacts dir.
    // All other KV / backend settings are inherited from the running config.
    let new_config = {
        let live = state.snapshot();
        Arc::unwrap_or_clone(live.session_config).with_mlx_model_artifacts_dir(model_path.clone())
    };

    let model_id = request.model_id.clone();
    let state_clone = state.clone();

    // Load the session on the blocking thread pool — weight loading can take
    // tens of seconds; blocking the async runtime would stall all other requests.
    let result = tokio::task::spawn_blocking(move || build_new_session(model_id, new_config)).await;

    // Always clear the loading flag, even on failure.
    state.loading.store(false, Ordering::Release);

    match result {
        Ok(Ok(live)) => {
            let ctx_len = {
                // Compute context_length from the new config before swapping.
                let kv = &live.session_config.kv_config;
                kv.block_size_tokens.saturating_mul(kv.total_blocks)
            };
            state_clone.swap_live(live);
            Ok(Json(LoadModelResponse {
                model_id: request.model_id,
                state: "loaded",
                context_length: ctx_len,
            }))
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
    }
}

fn build_new_session(
    model_id: String,
    config: EngineSessionConfig,
) -> Result<crate::app_state::LiveState, EngineSessionError> {
    let session = EngineSession::new(config)?;
    build_live_state(model_id, session)
}
