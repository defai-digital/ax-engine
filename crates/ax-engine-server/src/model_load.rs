use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::app_state::{AppState, build_live_state, build_replacement_live_state};
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
    /// `replace` preserves the historical hot-swap behavior. `add` retains
    /// existing models and publishes this model into the process registry.
    #[serde(default)]
    pub load_mode: LoadModelMode,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LoadModelPolicy {
    #[default]
    AvailabilityFirst,
    MemoryConstrained,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LoadModelMode {
    #[default]
    Replace,
    Add,
}

#[derive(Debug, Serialize)]
pub(crate) struct LoadModelResponse {
    pub model_id: String,
    pub state: &'static str,
    pub context_length: u32,
    pub load_policy: LoadModelPolicy,
    pub load_mode: LoadModelMode,
}

#[derive(Debug, Deserialize)]
pub(crate) struct UnloadModelRequest {
    pub model_id: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct UnloadModelResponse {
    pub model_id: String,
    pub state: &'static str,
}

pub(crate) async fn load_model(
    State(state): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> Result<Json<LoadModelResponse>, HttpErrorResponse> {
    let model_id = request.model_id.trim().to_string();
    if model_id.is_empty() {
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

    let load_policy = request.load_policy;
    let load_mode = request.load_mode;
    if load_mode == LoadModelMode::Add && load_policy == LoadModelPolicy::MemoryConstrained {
        return Err(error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "load_mode=add retains existing models and is incompatible with load_policy=memory_constrained"
                .to_string(),
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
    // Hold the loading flag across synchronous validation and move the guard
    // into the detached task only after every request error has been ruled out.
    // Invalid loads must not close admission or wait for unrelated traffic.
    let loading_guard = LoadingFlagGuard(state.clone());
    // Preserve the lifecycle contract: live stepwise work takes precedence
    // over backend/artifact validation because replacing its owner would
    // orphan request IDs. The loading flag already blocks new submissions.
    ensure_no_active_stepwise_for_all(&state).await?;
    let live = validate_load_preflight(&state, &model_id, &model_path, load_mode)?;
    let drain_guard = state.admission.begin_drain();
    let response_model_id = model_id.clone();

    // Run load + swap in a detached task: axum drops handler futures when the
    // client disconnects, and the load must still complete (or fail) and clear
    // the loading flag either way.
    let state_clone = state.clone();
    let load_task = tokio::spawn(async move {
        let _drain_guard = drain_guard;
        let _loading_guard = loading_guard;
        ensure_no_active_stepwise_for_all(&state_clone).await?;
        // Clone the current config and swap in the new model artifacts dir.
        // Only MLX-native consumes this field; delegated backends would keep
        // serving the old model under a new identifier.
        let new_config =
            Arc::unwrap_or_clone(live.session_config).with_mlx_model_artifacts_dir(model_path);
        wait_for_idle_without_stepwise(&state_clone).await?;
        if load_policy == LoadModelPolicy::MemoryConstrained {
            live.generation_service
                .shutdown()
                .await
                .map_err(map_generation_service_error)?;
        }

        // Start the replacement from the blocking pool and wait for readiness there.
        // The service constructs the session on its dedicated owner worker; weight
        // loading can take tens of seconds and must not stall the async runtime.
        let result = tokio::task::spawn_blocking(move || match load_mode {
            LoadModelMode::Replace => build_replacement_live_state(model_id, new_config),
            LoadModelMode::Add => build_live_state(model_id, new_config),
        })
        .await;
        match result {
            Ok(Ok(live)) => {
                let ctx_len = crate::metadata::context_length(&live);
                let previous = match load_mode {
                    LoadModelMode::Replace => Some(state_clone.swap_live(live)),
                    LoadModelMode::Add => state_clone.publish_live(live, true),
                };
                if let Some(previous) = previous {
                    previous.retire().await.map_err(|error| {
                        error_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "server_error",
                            format!(
                                "new model loaded but previous generation failed to retire: {error}"
                            ),
                        )
                    })?;
                }
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
            model_id: response_model_id,
            state: "loaded",
            context_length: ctx_len,
            load_policy,
            load_mode,
        })),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("load task panicked: {e}"),
        )),
    }
}

pub(crate) async fn unload_model(
    State(state): State<AppState>,
    Json(request): Json<UnloadModelRequest>,
) -> Result<Json<UnloadModelResponse>, HttpErrorResponse> {
    let model_id = request.model_id.trim().to_string();
    if model_id.is_empty() {
        return Err(error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "model_id must not be empty".to_string(),
        ));
    }
    if state
        .loading
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(error_response(
            StatusCode::CONFLICT,
            "model_loading",
            "a model load or unload is already in progress".to_string(),
        ));
    }
    // The loading flag serializes registry mutations, so these preconditions
    // remain true until remove_live runs. Reject without draining active work.
    let loading_guard = LoadingFlagGuard(state.clone());
    validate_unload_preflight(&state, &model_id)?;
    let drain_guard = state.admission.begin_drain();
    let state_clone = state.clone();
    let unload_model_id = model_id.clone();
    let task = tokio::spawn(async move {
        let _drain_guard = drain_guard;
        let _loading_guard = loading_guard;
        ensure_no_active_stepwise_for_all(&state_clone).await?;
        wait_for_idle_without_stepwise(&state_clone).await?;
        let live = state_clone
            .remove_live(&unload_model_id)
            .map_err(|reason| match reason {
                "last_model" => error_response(
                    StatusCode::CONFLICT,
                    "last_model",
                    "cannot unload the last model; load another model first".to_string(),
                ),
                _ => error_response(
                    StatusCode::BAD_REQUEST,
                    "model_not_found",
                    format!("model {unload_model_id} is not loaded"),
                ),
            })?;
        live.retire().await.map_err(map_generation_service_error)
    });

    match task.await {
        Ok(Ok(())) => Ok(Json(UnloadModelResponse {
            model_id,
            state: "unloaded",
        })),
        Ok(Err(error)) => Err(error),
        Err(error) => Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("unload task panicked: {error}"),
        )),
    }
}

fn validate_load_preflight(
    state: &AppState,
    model_id: &str,
    model_path: &std::path::Path,
    load_mode: LoadModelMode,
) -> Result<crate::app_state::LiveState, HttpErrorResponse> {
    let live = state.snapshot();
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

    let loaded_model_ids = state.model_ids();
    match load_mode {
        LoadModelMode::Add => {
            if loaded_model_ids.iter().any(|loaded| loaded == model_id) {
                return Err(error_response(
                    StatusCode::CONFLICT,
                    "model_already_loaded",
                    format!(
                        "model {model_id} is already loaded; use load_mode=replace to replace it"
                    ),
                ));
            }
            validate_multi_model_target(model_id, model_path)?;
            validate_retained_multi_model_ids(&loaded_model_ids)?;
        }
        LoadModelMode::Replace => {
            if live.model_id.as_ref() != model_id
                && loaded_model_ids.iter().any(|loaded| loaded == model_id)
            {
                return Err(error_response(
                    StatusCode::CONFLICT,
                    "model_already_loaded",
                    format!(
                        "model {model_id} is already loaded; unload it before replacing the default model with the same id"
                    ),
                ));
            }

            // Replacing the sole model preserves the historical unrestricted
            // hot-swap contract. Once another model is retained, the result is
            // multi-model serving and every surviving/new model must remain in
            // the five-model product scope.
            if loaded_model_ids.len() > 1 {
                validate_multi_model_target(model_id, model_path)?;
                let retained = loaded_model_ids
                    .into_iter()
                    .filter(|loaded| loaded != live.model_id.as_ref())
                    .collect::<Vec<_>>();
                validate_retained_multi_model_ids(&retained)?;
            }
        }
    }
    Ok(live)
}

fn validate_unload_preflight(state: &AppState, model_id: &str) -> Result<(), HttpErrorResponse> {
    let loaded_model_ids = state.model_ids();
    if !loaded_model_ids.iter().any(|loaded| loaded == model_id) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "model_not_found",
            format!("model {model_id} is not loaded"),
        ));
    }
    if loaded_model_ids.len() == 1 {
        return Err(error_response(
            StatusCode::CONFLICT,
            "last_model",
            "cannot unload the last model; load another model first".to_string(),
        ));
    }
    Ok(())
}

async fn wait_for_idle_without_stepwise(state: &AppState) -> Result<(), HttpErrorResponse> {
    loop {
        ensure_no_active_stepwise_for_all(state).await?;
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

async fn ensure_no_active_stepwise_for_all(state: &AppState) -> Result<(), HttpErrorResponse> {
    for live in state.snapshots() {
        ensure_no_active_stepwise(&live.generation_service).await?;
    }
    Ok(())
}

fn validate_multi_model_target(
    model_id: &str,
    model_path: &std::path::Path,
) -> Result<(), HttpErrorResponse> {
    let inferred = crate::args::infer_model_id_from_artifacts(model_path).map_err(|message| {
        error_response(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", message)
    })?;
    let requested_target = multi_model_target_key(model_id);
    let inferred_target = inferred.as_deref().and_then(multi_model_target_key);
    if requested_target.is_some()
        && inferred
            .as_deref()
            .is_none_or(|_| inferred_target == requested_target)
    {
        return Ok(());
    }
    Err(error_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        "unsupported_model",
        format!(
            "multi-model loading is limited to Qwen 3.6 35B/27B and Gemma 4 12B/26B/31B; requested model_id={model_id}, inferred_artifact_model={}",
            inferred.as_deref().unwrap_or("unknown")
        ),
    ))
}

fn validate_retained_multi_model_ids(model_ids: &[String]) -> Result<(), HttpErrorResponse> {
    if let Some(model_id) = model_ids
        .iter()
        .find(|model_id| multi_model_target_key(model_id).is_none())
    {
        return Err(error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_model",
            format!(
                "multi-model loading is limited to Qwen 3.6 35B/27B and Gemma 4 12B/26B/31B; retained model_id={model_id} is outside that scope"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
fn is_supported_multi_model_id(model_id: &str) -> bool {
    multi_model_target_key(model_id).is_some()
}

fn multi_model_target_key(model_id: &str) -> Option<&'static str> {
    let normalized = model_id
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>();
    let qwen36 = normalized.contains("qwen3-6") || normalized.contains("qwen36");
    let gemma4 = normalized.contains("gemma-4") || normalized.contains("gemma4");
    if qwen36 && normalized.contains("35b") {
        Some("qwen3.6-35b")
    } else if qwen36 && normalized.contains("27b") {
        Some("qwen3.6-27b")
    } else if gemma4 && normalized.contains("12b") {
        Some("gemma-4-12b")
    } else if gemma4 && normalized.contains("26b") {
        Some("gemma-4-26b")
    } else if gemma4 && normalized.contains("31b") {
        Some("gemma-4-31b")
    } else {
        None
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
        wait_for_idle_without_stepwise(&state)
            .await
            .expect("recovery load should proceed after the worker stopped");
    }

    #[test]
    fn multi_model_target_allowlist_is_exact_to_product_scope() {
        for model_id in [
            "qwen3.6-35b-a3b",
            "mlx-community/Qwen3.6-27B-6bit",
            "gemma-4-12b-it",
            "gemma-4-26b-a4b-it",
            "gemma-4-31b-it",
        ] {
            assert!(is_supported_multi_model_id(model_id), "{model_id}");
        }
        for model_id in [
            "qwen3.5-9b",
            "qwen3-coder-next",
            "gemma-4-e2b-it",
            "gemma-4-e4b-it",
            "llama3.3-70b",
        ] {
            assert!(!is_supported_multi_model_id(model_id), "{model_id}");
        }
    }

    #[test]
    fn retained_models_cannot_bypass_multi_model_product_scope() {
        let supported = vec!["qwen3.6-27b".to_string(), "gemma-4-31b-it".to_string()];
        validate_retained_multi_model_ids(&supported)
            .expect("targeted multi-model ids should be accepted");

        let unsupported = vec!["qwen3.6-27b".to_string(), "llama3.3-70b".to_string()];
        let error = validate_retained_multi_model_ids(&unsupported)
            .expect_err("an out-of-scope retained model must be rejected");
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(error.1.0.error.code.as_deref(), Some("unsupported_model"));
    }
}
