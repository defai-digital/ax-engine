use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::app_state::{AppState, build_replacement_live_state};
use crate::errors::{ErrorResponse, error_response, map_generation_service_error};
use crate::generation::service::{GenerationServiceError, NativeGenerationService};

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
        ensure_no_active_stepwise(&generation_service).await?;
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

        // Start the replacement from the blocking pool and wait for readiness there.
        // The service constructs the session on its dedicated owner worker; weight
        // loading can take tens of seconds and must not stall the async runtime.
        let result =
            tokio::task::spawn_blocking(move || build_replacement_live_state(model_id, new_config))
                .await;
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
    generation_service: &NativeGenerationService,
) -> Result<(), HttpErrorResponse> {
    loop {
        ensure_no_active_stepwise(generation_service).await?;
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

async fn ensure_no_active_stepwise(
    generation_service: &NativeGenerationService,
) -> Result<(), HttpErrorResponse> {
    if !generation_service.is_ready() {
        return Ok(());
    }

    check_active_stepwise_status(generation_service.has_active_stepwise().await)
}

fn check_active_stepwise_status(
    status: Result<bool, GenerationServiceError>,
) -> Result<(), HttpErrorResponse> {
    match status {
        Ok(false) | Err(GenerationServiceError::Unavailable) => Ok(()),
        Ok(true) => Err(active_stepwise_load_error()),
        Err(error) => Err(map_generation_service_error(error)),
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ax_engine_sdk::{
        EngineSessionConfig, PreviewBackendRequest, PreviewSessionConfigRequest, SupportTier,
    };

    use crate::app_state::build_app_state;

    use super::*;

    fn delegated_config() -> EngineSessionConfig {
        EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::LlamaCpp,
                llama_model_path: Some(PathBuf::from("fake-model.gguf")),
                ..PreviewBackendRequest::default()
            },
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview session config should build")
    }

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

    #[test]
    fn worker_stopping_during_stepwise_check_does_not_block_recovery() {
        check_active_stepwise_status(Err(GenerationServiceError::Unavailable))
            .expect("an unavailable worker has no live stepwise state");
    }

    #[tokio::test]
    async fn stopped_worker_does_not_block_recovery_load() {
        let state = build_app_state("old".to_string(), delegated_config())
            .expect("test app state should build");
        let generation_service = state.snapshot().generation_service;
        generation_service
            .shutdown()
            .await
            .expect("worker should stop");

        ensure_no_active_stepwise(&generation_service)
            .await
            .expect("a stopped worker has no live stepwise state");
        wait_for_idle_without_stepwise(&state, &generation_service)
            .await
            .expect("recovery load should proceed after the worker stopped");
    }
}
