use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use ax_engine_sdk::{EngineSession, EngineSessionConfig, EngineSessionError};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::app_state::{AppState, build_live_state};
use crate::errors::{ErrorResponse, error_response, map_generation_service_error};

type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);

#[derive(Debug, Deserialize)]
pub(crate) struct LoadModelRequest {
    /// Identifier the server will advertise for this model.
    pub model_id: String,
    /// Path to the MLX model artifacts directory (must contain model-manifest.json).
    pub model_path: String,
    #[serde(default)]
    pub load_policy: LoadModelPolicy,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LoadModelPolicy {
    #[default]
    AvailabilityFirst,
    MemoryConstrained,
}

#[derive(Debug, Serialize)]
pub(crate) struct LoadModelResponse {
    pub model_id: String,
    pub state: &'static str,
    pub context_length: u32,
    pub load_policy: LoadModelPolicy,
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
    let drain_guard = state.admission.begin_drain();

    let model_id = request.model_id.clone();
    let load_policy = request.load_policy;
    let generation_service = state.snapshot().generation_service;

    // Run load + swap in a detached task: axum drops handler futures when the
    // client disconnects, and the load must still complete (or fail) and clear
    // the loading flag either way.
    let state_clone = state.clone();
    let load_task = tokio::spawn(async move {
        let _drain_guard = drain_guard;
        let _loading_guard = LoadingFlagGuard(state_clone.clone());
        if generation_service
            .has_active_stepwise()
            .await
            .map_err(map_generation_service_error)?
        {
            return Err(active_stepwise_load_error());
        }
        // Clone the current config and swap in the new model artifacts dir.
        // Only MLX-native consumes this field; delegated backends would keep
        // serving the old model under a new identifier.
        let live = state_clone.snapshot();
        if !live
            .session_config
            .resolved_backend
            .selected_backend
            .is_mlx()
        {
            return Err(error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_backend",
                "model load is only supported on the MLX-native backend".to_string(),
            ));
        }
        let new_config =
            Arc::unwrap_or_clone(live.session_config).with_mlx_model_artifacts_dir(model_path);
        wait_for_idle_without_stepwise(&state_clone, &generation_service).await?;
        if load_policy == LoadModelPolicy::MemoryConstrained {
            generation_service
                .shutdown()
                .await
                .map_err(map_generation_service_error)?;
        }

        // Load the session on the blocking thread pool — weight loading can take
        // tens of seconds; blocking the async runtime would stall all other requests.
        let result =
            tokio::task::spawn_blocking(move || build_new_session(model_id, new_config)).await;
        match result {
            Ok(Ok(live)) => {
                let ctx_len = crate::metadata::context_length(&live);
                let previous = state_clone.swap_live(live);
                previous.retire().await.map_err(|error| {
                    error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "server_error",
                        format!(
                            "new model loaded but previous generation failed to retire: {error}"
                        ),
                    )
                })?;
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
        }
    });

    match load_task.await {
        Ok(Ok(ctx_len)) => Ok(Json(LoadModelResponse {
            model_id: request.model_id,
            state: "loaded",
            context_length: ctx_len,
            load_policy,
        })),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("load task panicked: {e}"),
        )),
    }
}

async fn wait_for_idle_without_stepwise(
    state: &AppState,
    generation_service: &crate::generation::service::NativeGenerationService,
) -> Result<(), HttpErrorResponse> {
    loop {
        let has_active_stepwise_requests = generation_service
            .has_active_stepwise()
            .await
            .map_err(map_generation_service_error)?;
        if has_active_stepwise_requests {
            return Err(active_stepwise_load_error());
        }
        if state.admission.active_jobs() == 0 {
            return Ok(());
        }
        if tokio::time::timeout(Duration::from_millis(10), state.admission.wait_for_idle())
            .await
            .is_ok()
        {
            return Ok(());
        }
    }
}

fn active_stepwise_load_error() -> HttpErrorResponse {
    error_response(
        StatusCode::CONFLICT,
        "requests_in_flight",
        "the current session has non-terminal /v1/requests work; drain or cancel it before \
         loading a new model"
            .to_string(),
    )
}

struct LoadingFlagGuard(AppState);

impl Drop for LoadingFlagGuard {
    fn drop(&mut self) {
        self.0.loading.store(false, Ordering::Release);
    }
}

fn build_new_session(
    model_id: String,
    config: EngineSessionConfig,
) -> Result<crate::app_state::LiveState, EngineSessionError> {
    EngineSession::clear_native_model_compile_caches();
    let session = EngineSession::new(config)?;
    build_live_state(model_id, session)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_policy_defaults_to_availability_first() {
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model_id": "next",
            "model_path": "/tmp/model"
        }))
        .expect("request should parse");

        assert_eq!(request.load_policy, LoadModelPolicy::AvailabilityFirst);
    }

    #[test]
    fn memory_constrained_policy_is_explicit() {
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model_id": "next",
            "model_path": "/tmp/model",
            "load_policy": "memory_constrained"
        }))
        .expect("request should parse");

        assert_eq!(request.load_policy, LoadModelPolicy::MemoryConstrained);
    }
}
