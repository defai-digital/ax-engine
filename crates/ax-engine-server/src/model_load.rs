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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MultiModelTarget {
    Qwen36_35b,
    Qwen36_27b,
    Gemma4_12b,
    Gemma4_26b,
    Gemma4_31b,
}

impl MultiModelTarget {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen36_35b => "qwen3.6-35b",
            Self::Qwen36_27b => "qwen3.6-27b",
            Self::Gemma4_12b => "gemma-4-12b",
            Self::Gemma4_26b => "gemma-4-26b",
            Self::Gemma4_31b => "gemma-4-31b",
        }
    }
}

#[derive(Debug, Deserialize)]
struct MultiModelManifestIdentity {
    model_family: String,
    layer_count: u32,
    hidden_size: u32,
    #[serde(default)]
    intermediate_size: u32,
    attention_head_count: u32,
    attention_head_dim: u32,
    kv_head_count: u32,
    vocab_size: u32,
    #[serde(default)]
    moe: MultiModelMoeIdentity,
}

#[derive(Debug, Default, Deserialize)]
struct MultiModelMoeIdentity {
    expert_count: Option<u32>,
    experts_per_token: Option<u32>,
    expert_intermediate_size: Option<u32>,
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
    let live = validate_load_preflight(&state, &model_id, &model_path, load_mode, load_policy)?;
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
    perform_unload(&state, model_id.clone()).await?;
    Ok(Json(UnloadModelResponse {
        model_id,
        state: "unloaded",
    }))
}

/// Shared unload flow used by the HTTP handler and the idle evictor: takes
/// the loading flag, validates, drains admission, and retires the model on a
/// detached task so a caller disconnect cannot abort cleanup.
async fn perform_unload(state: &AppState, model_id: String) -> Result<(), HttpErrorResponse> {
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
    validate_unload_preflight(state, &model_id)?;
    let drain_guard = state.admission.begin_drain();
    let state_clone = state.clone();
    let unload_model_id = model_id;
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
        Ok(result) => result,
        Err(error) => Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("unload task panicked: {error}"),
        )),
    }
}

/// Background idle evictor (opt-in via `--model-idle-timeout-secs`): retires
/// non-default resident models that have not admitted a request within the
/// timeout. The default model is never evicted, and a sweep only runs while
/// the server is otherwise idle — the unload flow drains global admission, so
/// evicting during active traffic would stall unrelated requests.
pub(crate) fn spawn_model_idle_evictor(state: AppState, idle_timeout: Duration) {
    let tick = Duration::from_secs((idle_timeout.as_secs() / 4).clamp(10, 60));
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tick).await;
            if state.admission.active_jobs() != 0 {
                continue;
            }
            let default_model_id = state.snapshot().model_id;
            let now = crate::app_state::unix_now_secs();
            let idle_candidates = state
                .snapshots()
                .into_iter()
                .filter(|live| live.model_id.as_ref() != default_model_id.as_ref())
                .filter(|live| {
                    let last_used = live.last_used.load(Ordering::Acquire);
                    now.saturating_sub(last_used) >= idle_timeout.as_secs()
                })
                .map(|live| live.model_id.as_ref().clone())
                .collect::<Vec<_>>();
            for model_id in idle_candidates {
                match perform_unload(&state, model_id.clone()).await {
                    Ok(()) => {
                        tracing::info!(model_id, "idle-evicted resident model");
                    }
                    Err((_, error)) => {
                        // Busy (load in progress, active work) or already
                        // gone: skip this sweep and retry on a later tick.
                        tracing::debug!(
                            model_id,
                            error = %error.0.error.message,
                            "idle eviction skipped"
                        );
                    }
                }
            }
        }
    });
}

fn validate_load_preflight(
    state: &AppState,
    model_id: &str,
    model_path: &std::path::Path,
    load_mode: LoadModelMode,
    load_policy: LoadModelPolicy,
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
    let loaded_models = state.snapshots();
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
            validate_retained_multi_model_artifacts(&loaded_models)?;
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
                let retained_models = loaded_models
                    .iter()
                    .filter(|loaded| loaded.model_id.as_ref() != live.model_id.as_ref())
                    .cloned()
                    .collect::<Vec<_>>();
                validate_retained_multi_model_artifacts(&retained_models)?;
            }
        }
    }
    validate_load_memory_preflight(state, model_path, load_mode, load_policy)?;
    Ok(live)
}

/// Runtime factor + fixed floor over on-disk weight bytes (ADR-040 D4):
/// quantized weights land in memory at ~disk size; the extra 1/8 covers
/// compiled graphs, speculative-decoding state, and allocator slack; the
/// floor covers KV baseline and runtime buffers. Conservative by design —
/// the estimator's job is to fail a load that cannot fit, loudly and early.
const LOAD_FOOTPRINT_FIXED_FLOOR_BYTES: u64 = 768 * 1024 * 1024;

fn estimated_footprint_bytes(weight_bytes: u64) -> u64 {
    weight_bytes
        .saturating_add(weight_bytes / 8)
        .saturating_add(LOAD_FOOTPRINT_FIXED_FLOOR_BYTES)
}

/// Sum of `*.safetensors` file sizes directly in the artifacts dir; `None`
/// when there are none (unknown layout — the preflight then skips rather
/// than guessing).
fn model_weight_bytes(dir: &std::path::Path) -> Option<u64> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut total = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
            && let Ok(metadata) = entry.metadata()
            && metadata.is_file()
        {
            total = total.saturating_add(metadata.len());
        }
    }
    (total > 0).then_some(total)
}

/// Peak resident estimate for the load: `add` keeps every resident model;
/// `replace` under `availability_first` keeps the outgoing model resident
/// until the replacement is ready (the overlap is that policy's point), while
/// `memory_constrained` shuts it down first.
fn projected_peak_bytes(
    resident_total: u64,
    outgoing: u64,
    incoming: u64,
    load_mode: LoadModelMode,
    load_policy: LoadModelPolicy,
) -> u64 {
    let resident = match (load_mode, load_policy) {
        (LoadModelMode::Replace, LoadModelPolicy::MemoryConstrained) => {
            resident_total.saturating_sub(outgoing)
        }
        _ => resident_total,
    };
    resident.saturating_add(incoming)
}

fn memory_preflight_disabled() -> bool {
    std::env::var("AX_SERVER_LOAD_MEMORY_PREFLIGHT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "off" | "0" | "false")
        })
        .unwrap_or(false)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Memory admission for model loads (ADR-040 D4). Runs synchronously before
/// drain, preserving the no-side-effects-on-reject preflight contract. Skips
/// (never blocks) when the device budget or the incoming weight layout is
/// unknowable, and can be disabled with `AX_SERVER_LOAD_MEMORY_PREFLIGHT=off`.
fn validate_load_memory_preflight(
    state: &AppState,
    model_path: &std::path::Path,
    load_mode: LoadModelMode,
    load_policy: LoadModelPolicy,
) -> Result<(), HttpErrorResponse> {
    if memory_preflight_disabled() {
        return Ok(());
    }
    let budget = mlx_sys::max_recommended_working_set_size() as u64;
    if budget == 0 {
        return Ok(());
    }
    let Some(incoming_weights) = model_weight_bytes(model_path) else {
        return Ok(());
    };
    let incoming = estimated_footprint_bytes(incoming_weights);

    let outgoing_model_id = state.snapshot().model_id;
    let mut resident_total = 0u64;
    let mut outgoing = 0u64;
    for live in state.snapshots() {
        let Some(dir) = live.session_config.mlx_model_artifacts_dir() else {
            continue;
        };
        let Some(weights) = model_weight_bytes(dir) else {
            continue;
        };
        let footprint = estimated_footprint_bytes(weights);
        resident_total = resident_total.saturating_add(footprint);
        if live.model_id.as_ref() == outgoing_model_id.as_ref() {
            outgoing = footprint;
        }
    }

    let peak = projected_peak_bytes(resident_total, outgoing, incoming, load_mode, load_policy);
    if peak <= budget {
        return Ok(());
    }
    Err(error_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        "insufficient_memory",
        format!(
            "projected peak resident set {:.1} GiB (resident models {:.1} GiB + incoming model \
             {:.1} GiB estimated) exceeds the Metal working-set budget {:.1} GiB; unload a model \
             first, use load_policy=memory_constrained for replace, or set \
             AX_SERVER_LOAD_MEMORY_PREFLIGHT=off to override",
            gib(peak),
            gib(resident_total),
            gib(incoming),
            gib(budget)
        ),
    ))
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
    let requested_target = multi_model_target_key(model_id);
    let artifact_target = multi_model_artifact_target(model_path)?;
    if requested_target.is_some() && requested_target == artifact_target {
        return Ok(());
    }
    Err(error_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        "unsupported_model",
        format!(
            "multi-model loading is limited to Qwen 3.6 35B/27B and Gemma 4 12B/26B/31B; requested model_id={model_id}, inferred_artifact_model={}",
            artifact_target.map_or("unknown", MultiModelTarget::as_str)
        ),
    ))
}

fn validate_retained_multi_model_artifacts(
    lives: &[crate::app_state::LiveState],
) -> Result<(), HttpErrorResponse> {
    for live in lives {
        let model_path = live
            .session_config
            .mlx_model_artifacts_dir()
            .ok_or_else(|| {
                error_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "unsupported_model",
                    format!(
                        "retained model {} has no native MLX artifact identity",
                        live.model_id
                    ),
                )
            })?;
        validate_multi_model_target(live.model_id.as_ref(), model_path)?;
    }
    Ok(())
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

fn multi_model_target_key(model_id: &str) -> Option<MultiModelTarget> {
    let normalized = normalize_model_label(model_id);
    if target_label_matches(&normalized, &["qwen3-6-35b", "qwen36-35b"]) {
        Some(MultiModelTarget::Qwen36_35b)
    } else if target_label_matches(&normalized, &["qwen3-6-27b", "qwen36-27b"]) {
        Some(MultiModelTarget::Qwen36_27b)
    } else if target_label_matches(&normalized, &["gemma-4-12b", "gemma4-12b"]) {
        Some(MultiModelTarget::Gemma4_12b)
    } else if target_label_matches(&normalized, &["gemma-4-26b", "gemma4-26b"]) {
        Some(MultiModelTarget::Gemma4_26b)
    } else if target_label_matches(&normalized, &["gemma-4-31b", "gemma4-31b"]) {
        Some(MultiModelTarget::Gemma4_31b)
    } else {
        None
    }
}

fn normalize_model_label(model_id: &str) -> String {
    let label = model_id.rsplit(['/', '\\']).next().unwrap_or(model_id);
    let mut normalized = String::with_capacity(label.len());
    let mut separated = false;
    for ch in label.to_ascii_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch);
            separated = false;
        } else if !separated && !normalized.is_empty() {
            normalized.push('-');
            separated = true;
        }
    }
    normalized.trim_end_matches('-').to_string()
}

fn target_label_matches(label: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| {
        label == *prefix
            || label
                .strip_prefix(prefix)
                .is_some_and(|suffix| suffix.starts_with('-'))
    })
}

fn multi_model_artifact_target(
    model_path: &std::path::Path,
) -> Result<Option<MultiModelTarget>, HttpErrorResponse> {
    let manifest_path = model_path.join("model-manifest.json");
    let bytes = std::fs::read(&manifest_path).map_err(|error| {
        error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            format!("cannot read {}: {error}", manifest_path.display()),
        )
    })?;
    let identity =
        serde_json::from_slice::<MultiModelManifestIdentity>(&bytes).map_err(|error| {
            error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request",
                format!("cannot parse {}: {error}", manifest_path.display()),
            )
        })?;
    Ok(multi_model_target_from_manifest(&identity))
}

fn multi_model_target_from_manifest(
    identity: &MultiModelManifestIdentity,
) -> Option<MultiModelTarget> {
    let signature = (
        identity.model_family.as_str(),
        identity.layer_count,
        identity.hidden_size,
        identity.intermediate_size,
        identity.attention_head_count,
        identity.attention_head_dim,
        identity.kv_head_count,
        identity.vocab_size,
        identity.moe.expert_count,
        identity.moe.experts_per_token,
        identity.moe.expert_intermediate_size,
    );
    match signature {
        ("qwen3_5", 64, 5120, 17408, 24, 256, 4, 248320, None, None, None) => {
            Some(MultiModelTarget::Qwen36_27b)
        }
        ("qwen3_5", 40, 2048, 0, 16, 256, 2, 248320, Some(256), Some(8), Some(512)) => {
            Some(MultiModelTarget::Qwen36_35b)
        }
        ("gemma4", 48, 3840, 15360, 16, 256, 8, 262144, None, None, None) => {
            Some(MultiModelTarget::Gemma4_12b)
        }
        ("gemma4", 30, 2816, 2112, 16, 256, 8, 262144, Some(128), Some(8), Some(704)) => {
            Some(MultiModelTarget::Gemma4_26b)
        }
        ("gemma4", 60, 5376, 21504, 32, 256, 16, 262144, None, None, None) => {
            Some(MultiModelTarget::Gemma4_31b)
        }
        _ => None,
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
            "prefix-qwen3.6-35b",
            "qwen3.65-35b",
            "gemma-4-e2b-it",
            "gemma-4-e4b-it",
            "gemma-40-12b-it",
            "llama3.3-70b",
        ] {
            assert!(!is_supported_multi_model_id(model_id), "{model_id}");
        }
    }

    #[test]
    fn multi_model_artifact_identity_is_manifest_authoritative() {
        let dir = std::env::temp_dir().join(format!(
            "ax-multi-model-identity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time should be valid")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("test dir should create");
        std::fs::write(
            dir.join("model-manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_family": "qwen3_5",
                "layer_count": 64,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "attention_head_count": 24,
                "attention_head_dim": 256,
                "kv_head_count": 4,
                "vocab_size": 248320
            }))
            .expect("manifest should serialize"),
        )
        .expect("manifest should write");

        validate_multi_model_target("qwen3.6-27b-6bit", &dir)
            .expect("the exact manifest signature should pass without config.json");
        let mismatch = validate_multi_model_target("gemma-4-12b-it", &dir)
            .expect_err("a mismatched public label must fail closed");
        assert_eq!(
            mismatch.1.0.error.code.as_deref(),
            Some("unsupported_model")
        );

        std::fs::write(
            dir.join("model-manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_family": "qwen3_5",
                "layer_count": 63,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "attention_head_count": 24,
                "attention_head_dim": 256,
                "kv_head_count": 4,
                "vocab_size": 248320
            }))
            .expect("manifest should serialize"),
        )
        .expect("manifest should write");
        let unknown = validate_multi_model_target("qwen3.6-27b-6bit", &dir)
            .expect_err("an unknown architecture signature must fail closed");
        assert_eq!(unknown.1.0.error.code.as_deref(), Some("unsupported_model"));
        std::fs::remove_dir_all(dir).expect("test dir should clean up");
    }

    #[test]
    fn all_supported_manifest_signatures_are_distinct() {
        let identities = [
            (
                serde_json::json!({
                    "model_family": "qwen3_5", "layer_count": 64, "hidden_size": 5120,
                    "intermediate_size": 17408, "attention_head_count": 24,
                    "attention_head_dim": 256, "kv_head_count": 4, "vocab_size": 248320
                }),
                MultiModelTarget::Qwen36_27b,
            ),
            (
                serde_json::json!({
                    "model_family": "qwen3_5", "layer_count": 40, "hidden_size": 2048,
                    "intermediate_size": 0, "attention_head_count": 16,
                    "attention_head_dim": 256, "kv_head_count": 2, "vocab_size": 248320,
                    "moe": {"expert_count": 256, "experts_per_token": 8,
                            "expert_intermediate_size": 512}
                }),
                MultiModelTarget::Qwen36_35b,
            ),
            (
                serde_json::json!({
                    "model_family": "gemma4", "layer_count": 48, "hidden_size": 3840,
                    "intermediate_size": 15360, "attention_head_count": 16,
                    "attention_head_dim": 256, "kv_head_count": 8, "vocab_size": 262144
                }),
                MultiModelTarget::Gemma4_12b,
            ),
            (
                serde_json::json!({
                    "model_family": "gemma4", "layer_count": 30, "hidden_size": 2816,
                    "intermediate_size": 2112, "attention_head_count": 16,
                    "attention_head_dim": 256, "kv_head_count": 8, "vocab_size": 262144,
                    "moe": {"expert_count": 128, "experts_per_token": 8,
                            "expert_intermediate_size": 704}
                }),
                MultiModelTarget::Gemma4_26b,
            ),
            (
                serde_json::json!({
                    "model_family": "gemma4", "layer_count": 60, "hidden_size": 5376,
                    "intermediate_size": 21504, "attention_head_count": 32,
                    "attention_head_dim": 256, "kv_head_count": 16, "vocab_size": 262144
                }),
                MultiModelTarget::Gemma4_31b,
            ),
        ];
        for (identity, expected) in identities {
            let identity = serde_json::from_value(identity).expect("identity should parse");
            assert_eq!(multi_model_target_from_manifest(&identity), Some(expected));
        }
    }

    #[test]
    fn footprint_estimate_adds_runtime_factor_and_floor() {
        let weights = 16 * 1024 * 1024 * 1024u64; // 16 GiB on disk
        let footprint = estimated_footprint_bytes(weights);
        assert_eq!(
            footprint,
            weights + weights / 8 + LOAD_FOOTPRINT_FIXED_FLOOR_BYTES
        );
    }

    #[test]
    fn projected_peak_counts_overlap_per_mode_and_policy() {
        let resident_total = 30u64;
        let outgoing = 20u64;
        let incoming = 25u64;
        // add: every resident model stays.
        assert_eq!(
            projected_peak_bytes(
                resident_total,
                outgoing,
                incoming,
                LoadModelMode::Add,
                LoadModelPolicy::AvailabilityFirst
            ),
            55
        );
        // replace + availability_first: outgoing stays resident until the
        // replacement is ready, so the peak includes it.
        assert_eq!(
            projected_peak_bytes(
                resident_total,
                outgoing,
                incoming,
                LoadModelMode::Replace,
                LoadModelPolicy::AvailabilityFirst
            ),
            55
        );
        // replace + memory_constrained: outgoing is shut down first.
        assert_eq!(
            projected_peak_bytes(
                resident_total,
                outgoing,
                incoming,
                LoadModelMode::Replace,
                LoadModelPolicy::MemoryConstrained
            ),
            35
        );
    }

    #[test]
    fn weight_bytes_sums_safetensors_only() {
        let dir =
            std::env::temp_dir().join(format!("ax-model-weight-bytes-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("test dir should create");
        std::fs::write(dir.join("model-00001.safetensors"), vec![0u8; 1024])
            .expect("weight shard should write");
        std::fs::write(dir.join("model-00002.safetensors"), vec![0u8; 512])
            .expect("weight shard should write");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("tokenizer should write");
        assert_eq!(model_weight_bytes(&dir), Some(1536));
        std::fs::remove_dir_all(&dir).expect("test dir should clean up");
    }

    #[test]
    fn weight_bytes_is_unknown_without_safetensors() {
        let dir = std::env::temp_dir().join(format!(
            "ax-model-weight-bytes-empty-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("test dir should create");
        assert_eq!(model_weight_bytes(&dir), None);
        std::fs::remove_dir_all(&dir).expect("test dir should clean up");
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
