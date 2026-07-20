use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use ax_engine_sdk::{
    KvManagerConfig, NativeDiffusionConfig, NativeLinearAttentionConfig, NativeMlaAttentionConfig,
};
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
    /// Whether the loaded model becomes the default for requests that omit
    /// `model`. Defaults to `true` (the historical behavior). Only
    /// meaningful for `load_mode=add`; `replace` swaps the default model in
    /// place and rejects `false`.
    #[serde(default)]
    pub make_default: Option<bool>,
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
    Qwen35_9b,
    Qwen36_35b,
    Qwen36_27b,
    Qwen3CoderNext,
    Gemma4_12b,
    Gemma4_26b,
    Gemma4_31b,
    EmbeddingGemma300m,
    Qwen3Embedding0_6b,
    Qwen3Embedding4b,
    Qwen3Embedding8b,
}

/// One-line product scope used by every multi-model rejection message.
const MULTI_MODEL_SCOPE: &str = "Qwen 3.5 9B, Qwen 3.6 35B/27B, Qwen3-Coder-Next, \
     Gemma 4 12B/26B/31B, EmbeddingGemma 300M, and Qwen3-Embedding 0.6B/4B/8B";

impl MultiModelTarget {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen35_9b => "qwen3.5-9b",
            Self::Qwen36_35b => "qwen3.6-35b",
            Self::Qwen36_27b => "qwen3.6-27b",
            Self::Qwen3CoderNext => "qwen3-coder-next",
            Self::Gemma4_12b => "gemma-4-12b",
            Self::Gemma4_26b => "gemma-4-26b",
            Self::Gemma4_31b => "gemma-4-31b",
            Self::EmbeddingGemma300m => "embeddinggemma-300m",
            Self::Qwen3Embedding0_6b => "qwen3-embedding-0.6b",
            Self::Qwen3Embedding4b => "qwen3-embedding-4b",
            Self::Qwen3Embedding8b => "qwen3-embedding-8b",
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
    /// Default model after the load — differs from `model_id` for
    /// `load_mode=add` with `make_default:false`.
    pub default_model_id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct UnloadModelRequest {
    pub model_id: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct UnloadModelResponse {
    pub model_id: String,
    pub state: &'static str,
    /// Default model after the unload — reports the reassignment when the
    /// unloaded model was the default.
    pub default_model_id: String,
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
    // Canonicalize to resolve symlinks and `..` components, preventing path
    // traversal from escaping an intended model directory. The canonical form
    // is used for all downstream validation and loading.
    let model_path = match model_path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            return Err(error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request",
                format!(
                    "model_path does not exist or cannot be resolved: {}",
                    request.model_path
                ),
            ));
        }
    };
    if !model_path.is_dir() {
        return Err(error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            format!("model_path must be a directory: {}", model_path.display()),
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
    if load_mode == LoadModelMode::Replace && request.make_default == Some(false) {
        return Err(error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            "load_mode=replace swaps the default model in place and cannot honor make_default=false; \
             use load_mode=add to load a non-default model"
                .to_string(),
        ));
    }
    let make_default = request.make_default.unwrap_or(true);

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
                    LoadModelMode::Add => state_clone.publish_live(live, make_default),
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
            default_model_id: state.snapshot().model_id.as_ref().clone(),
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
        default_model_id: state.snapshot().model_id.as_ref().clone(),
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
            // the multi-model product scope.
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

/// Floor applied when the KV pool is charged explicitly (ADR-041 D1): the
/// remainder covers runtime buffers, linear-attention per-request state,
/// MTP state, and allocator slack — everything the old 768 MiB floor
/// bundled together with "KV baseline".
const LOAD_FOOTPRINT_RUNTIME_FLOOR_BYTES: u64 = 512 * 1024 * 1024;

/// K + V, at fp16/bf16 element width. Quantized-KV serving would shrink
/// this; the admission bound stays at the unquantized worst case.
const KV_CACHE_BYTES_PER_HEAD_ELEMENT: u64 = 2 * 2;

fn estimated_footprint_bytes(weight_bytes: u64, kv_pool_bytes: Option<u64>) -> u64 {
    let base = weight_bytes.saturating_add(weight_bytes / 8);
    match kv_pool_bytes {
        Some(kv_bytes) => base
            .saturating_add(kv_bytes)
            .saturating_add(LOAD_FOOTPRINT_RUNTIME_FLOOR_BYTES),
        None => base.saturating_add(LOAD_FOOTPRINT_FIXED_FLOOR_BYTES),
    }
}

/// Tolerant projection of `model-manifest.json` for KV-geometry estimation
/// (ADR-041 D1). Only the fields the estimator needs; identity checks live
/// in `MultiModelManifestIdentity` and stay independent. Unknown manifest
/// fields never fail this parse.
#[derive(Debug, Deserialize)]
struct ManifestKvGeometry {
    #[serde(default)]
    model_family: String,
    layer_count: u32,
    attention_head_dim: u32,
    kv_head_count: u32,
    /// ISWA full-attention layers use this head dim; sliding layers use
    /// `attention_head_dim`.
    #[serde(default)]
    global_head_dim: Option<u32>,
    /// Per-layer annotations ("sliding_attention" / "full_attention");
    /// empty for homogeneous models.
    #[serde(default)]
    layer_types: Vec<String>,
    /// Layers that read another layer's K/V and allocate none of their own.
    #[serde(default)]
    kv_shared_source_layers: BTreeMap<u32, u32>,
    #[serde(default)]
    linear_attention: NativeLinearAttentionConfig,
    #[serde(default)]
    mla_attention: NativeMlaAttentionConfig,
    #[serde(default)]
    diffusion: NativeDiffusionConfig,
}

fn manifest_kv_geometry(dir: &std::path::Path) -> Option<ManifestKvGeometry> {
    let bytes = std::fs::read(dir.join("model-manifest.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Tokens the engine can ever hand this model's KV manager: the logical
/// block pool is the admission bound the scheduler enforces per model.
fn kv_pool_tokens(kv_config: &KvManagerConfig) -> u64 {
    u64::from(kv_config.total_blocks).saturating_mul(u64::from(kv_config.block_size_tokens))
}

/// Worst-case KV-cache bytes for one model at its configured pool, from the
/// manifest's attention geometry. `None` when the geometry is unknowable —
/// MLA latent caches and diffusion follow different math, and a zero dim or
/// an unresolvable hybrid interval means the manifest predates this
/// estimator — in which case the caller falls back to the flat floor
/// (ADR-040 D4 behavior) rather than guessing.
fn estimated_kv_pool_bytes(geometry: &ManifestKvGeometry, pool_tokens: u64) -> Option<u64> {
    if geometry.mla_attention.is_enabled() || geometry.diffusion.is_enabled() {
        return None;
    }
    if geometry.layer_count == 0 || geometry.kv_head_count == 0 || geometry.attention_head_dim == 0
    {
        return None;
    }
    let linear_interval = if geometry.linear_attention.is_enabled() {
        match geometry
            .linear_attention
            .resolved_full_attention_interval(&geometry.model_family)
        {
            Some(interval) if interval > 0 => Some(u64::from(interval)),
            _ => return None,
        }
    } else {
        None
    };
    let kv_heads = u64::from(geometry.kv_head_count);
    let sliding_bytes_per_token = kv_heads
        .saturating_mul(u64::from(geometry.attention_head_dim))
        .saturating_mul(KV_CACHE_BYTES_PER_HEAD_ELEMENT);
    let full_head_dim = geometry
        .global_head_dim
        .unwrap_or(geometry.attention_head_dim);
    let full_bytes_per_token = kv_heads
        .saturating_mul(u64::from(full_head_dim))
        .saturating_mul(KV_CACHE_BYTES_PER_HEAD_ELEMENT);

    let mut total = 0u64;
    for layer_idx in 0..geometry.layer_count {
        if geometry.kv_shared_source_layers.contains_key(&layer_idx) {
            continue;
        }
        if let Some(interval) = linear_interval {
            // Hybrid linear-attention layers keep per-request state, not a
            // per-token cache; the runtime floor covers them.
            if u64::from(layer_idx) % interval != interval - 1 {
                continue;
            }
            total = total.saturating_add(pool_tokens.saturating_mul(full_bytes_per_token));
            continue;
        }
        let layer_type = geometry.layer_types.get(layer_idx as usize);
        if layer_type.is_some_and(|kind| kind == "sliding_attention") {
            // Sliding rings bound KV per REQUEST, not per pool: many
            // concurrent ≤window-length requests each own window-sized
            // rings, and their sum legally reaches the pool. Admission
            // charges the pool-wide worst case, so sliding differs from
            // full attention only in head dim.
            total = total.saturating_add(pool_tokens.saturating_mul(sliding_bytes_per_token));
        } else {
            total = total.saturating_add(pool_tokens.saturating_mul(full_bytes_per_token));
        }
    }
    Some(total)
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
    // The incoming session inherits the default model's config (see the
    // load task), so its KV pool is sized from that config.
    let default_live = state.snapshot();
    let incoming_pool = kv_pool_tokens(&default_live.session_config.kv_config);
    let incoming_kv = manifest_kv_geometry(model_path)
        .and_then(|geometry| estimated_kv_pool_bytes(&geometry, incoming_pool));
    let incoming = estimated_footprint_bytes(incoming_weights, incoming_kv);

    let outgoing_model_id = default_live.model_id;
    let mut resident_total = 0u64;
    let mut outgoing = 0u64;
    for live in state.snapshots() {
        let Some(dir) = live.session_config.mlx_model_artifacts_dir() else {
            continue;
        };
        let Some(weights) = model_weight_bytes(dir) else {
            continue;
        };
        // A resident manifest that cannot be read keeps the legacy flat
        // floor — someone else's load must never fail on it.
        let resident_kv = manifest_kv_geometry(dir).and_then(|geometry| {
            estimated_kv_pool_bytes(&geometry, kv_pool_tokens(&live.session_config.kv_config))
        });
        let footprint = estimated_footprint_bytes(weights, resident_kv);
        resident_total = resident_total.saturating_add(footprint);
        if live.model_id.as_ref() == outgoing_model_id.as_ref() {
            outgoing = footprint;
        }
    }

    let peak = projected_peak_bytes(resident_total, outgoing, incoming, load_mode, load_policy);
    if peak <= budget {
        return Ok(());
    }
    let incoming_detail = match incoming_kv {
        Some(kv_bytes) => format!(
            "{:.1} GiB estimated (incl. {:.1} GiB KV pool)",
            gib(incoming),
            gib(kv_bytes)
        ),
        None => format!("{:.1} GiB estimated", gib(incoming)),
    };
    Err(error_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        "insufficient_memory",
        format!(
            "projected peak resident set {:.1} GiB (resident models {:.1} GiB + incoming model \
             {incoming_detail}) exceeds the Metal working-set budget {:.1} GiB; unload a model \
             first, use load_policy=memory_constrained for replace, or set \
             AX_SERVER_LOAD_MEMORY_PREFLIGHT=off to override",
            gib(peak),
            gib(resident_total),
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

/// Maximum time to wait for in-flight requests to drain before model load
/// proceeds with a warning. Prevents indefinite blocking if a job is stuck.
const DRAIN_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

async fn wait_for_idle_without_stepwise(state: &AppState) -> Result<(), HttpErrorResponse> {
    let deadline = tokio::time::Instant::now() + DRAIN_IDLE_TIMEOUT;
    loop {
        ensure_no_active_stepwise_for_all(state).await?;
        if state.admission.active_jobs() == 0 {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            tracing::warn!(
                active_jobs = state.admission.active_jobs(),
                timeout_secs = DRAIN_IDLE_TIMEOUT.as_secs(),
                "drain idle timeout reached; proceeding with model load while jobs may still be active"
            );
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
            "multi-model loading is limited to {MULTI_MODEL_SCOPE}; requested model_id={model_id}, inferred_artifact_model={}",
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
                "multi-model loading is limited to {MULTI_MODEL_SCOPE}; retained model_id={model_id} is outside that scope"
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
    // AutomatosX publishes the product models under an `AX-` brand prefix
    // (e.g. `AutomatosX/AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP`); the branded id
    // names the same target as the bare family label, so match both forms.
    // The manifest signature check stays authoritative either way, and any
    // other prefix keeps failing closed.
    std::iter::once(normalized.as_str())
        .chain(normalized.strip_prefix("ax-"))
        .find_map(multi_model_target_for_label)
}

fn multi_model_target_for_label(label: &str) -> Option<MultiModelTarget> {
    if target_label_matches(label, &["qwen3-5-9b", "qwen35-9b"]) {
        Some(MultiModelTarget::Qwen35_9b)
    } else if target_label_matches(label, &["qwen3-6-35b", "qwen36-35b"]) {
        Some(MultiModelTarget::Qwen36_35b)
    } else if target_label_matches(label, &["qwen3-6-27b", "qwen36-27b"]) {
        Some(MultiModelTarget::Qwen36_27b)
    } else if target_label_matches(label, &["qwen3-coder-next", "qwen3-next-80b"]) {
        Some(MultiModelTarget::Qwen3CoderNext)
    } else if target_label_matches(label, &["gemma-4-12b", "gemma4-12b"]) {
        Some(MultiModelTarget::Gemma4_12b)
    } else if target_label_matches(label, &["gemma-4-26b", "gemma4-26b"]) {
        Some(MultiModelTarget::Gemma4_26b)
    } else if target_label_matches(label, &["gemma-4-31b", "gemma4-31b"]) {
        Some(MultiModelTarget::Gemma4_31b)
    } else if target_label_matches(label, &["embeddinggemma", "embedding-gemma"]) {
        Some(MultiModelTarget::EmbeddingGemma300m)
    } else if target_label_matches(label, &["qwen3-embedding-0-6b", "qwen3-embedding-06b"]) {
        Some(MultiModelTarget::Qwen3Embedding0_6b)
    } else if target_label_matches(label, &["qwen3-embedding-4b"]) {
        Some(MultiModelTarget::Qwen3Embedding4b)
    } else if target_label_matches(label, &["qwen3-embedding-8b"]) {
        Some(MultiModelTarget::Qwen3Embedding8b)
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
        // Signature read from the converted 9B artifact's manifest
        // (dense hybrid GatedDeltaNet; vision tower stripped at
        // conversion, so the text-decoder identity is authoritative).
        ("qwen3_5", 32, 4096, 12288, 16, 256, 4, 248320, None, None, None) => {
            Some(MultiModelTarget::Qwen35_9b)
        }
        ("qwen3_5", 64, 5120, 17408, 24, 256, 4, 248320, None, None, None) => {
            Some(MultiModelTarget::Qwen36_27b)
        }
        ("qwen3_5", 40, 2048, 0, 16, 256, 2, 248320, Some(256), Some(8), Some(512)) => {
            Some(MultiModelTarget::Qwen36_35b)
        }
        // Qwen3-Coder-Next (hybrid GatedDeltaNet + sparse MoE on the
        // Qwen3-Next 80B-A3B geometry); signature read from the published
        // AutomatosX pack's config (AX-Qwen3-Coder-Next-MLX-4bit).
        ("qwen3_next", 48, 2048, 5120, 16, 256, 2, 151936, Some(512), Some(10), Some(512)) => {
            Some(MultiModelTarget::Qwen3CoderNext)
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
        // Embedding co-residency targets: signatures read from the shipped
        // AutomatosX pack manifests (AX-EmbeddingGemma-300M-MLX-8bit,
        // AX-Qwen3-Embedding-{0.6B,4B,8B}). Qwen3-Embedding serves on the
        // standard qwen3 decoder with last-token pooling; EmbeddingGemma is
        // the encoder+mean-pooling family.
        ("embeddinggemma", 24, 768, 1152, 3, 256, 1, 262144, None, None, None) => {
            Some(MultiModelTarget::EmbeddingGemma300m)
        }
        ("qwen3", 28, 1024, 3072, 16, 128, 8, 151669, None, None, None) => {
            Some(MultiModelTarget::Qwen3Embedding0_6b)
        }
        ("qwen3", 36, 2560, 9728, 32, 128, 8, 151665, None, None, None) => {
            Some(MultiModelTarget::Qwen3Embedding4b)
        }
        ("qwen3", 36, 4096, 12288, 32, 128, 8, 151665, None, None, None) => {
            Some(MultiModelTarget::Qwen3Embedding8b)
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
            "qwen3.5-9b",
            "qwen35-9b",
            "mlx-community/Qwen3.5-9B-MLX-4bit",
            "qwen3.6-35b-a3b",
            "mlx-community/Qwen3.6-27B-6bit",
            "gemma-4-12b-it",
            "gemma-4-26b-a4b-it",
            "gemma-4-31b-it",
            "qwen3-coder-next",
            "qwen3-next-80b-a3b",
            "embeddinggemma-300m",
            "embeddinggemma",
            "qwen3-embedding-0.6b",
            "qwen3-embedding-4b",
            "qwen3-embedding-8b",
            // AutomatosX org packages (https://huggingface.co/AutomatosX):
            // the AX- brand prefix names the same product targets.
            "AutomatosX/AX-Qwen3.5-9B-MLX-OptiQ-4bit-MTP",
            "AX-Qwen3.5-9B-MLX-6bit-MTP",
            "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP",
            "AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP",
            "AutomatosX/AX-Qwen3-Coder-Next-MLX-6bit",
            "AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP",
            "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-OptiQ-4bit-Assistant-MTP",
            "AutomatosX/AX-Gemma-4-31B-IT-MLX-6bit-Assistant-MTP",
            "AutomatosX/AX-EmbeddingGemma-300M-MLX-8bit",
            "AX-Qwen3-Embedding-0.6B-MLX-8bit",
            "AutomatosX/AX-Qwen3-Embedding-4B-MLX-4bit-DWQ",
            "AutomatosX/AX-Qwen3-Embedding-8B-MLX-4bit-DWQ",
        ] {
            assert!(is_supported_multi_model_id(model_id), "{model_id}");
        }
        for model_id in [
            "qwen3.5-0.8b",
            "qwen3.55-9b",
            "prefix-qwen3.5-9b",
            "prefix-qwen3.6-35b",
            "qwen3.65-35b",
            "qwen3-coder-nextgen",
            "qwen3-embedding-1b",
            "embeddinggemma2-300m",
            "gemma-4-e2b-it",
            "gemma-4-e4b-it",
            "gemma-40-12b-it",
            "llama3.3-70b",
            // The brand strip is exact: only a leading `ax-` label segment
            // may be dropped, and the remainder must still be a product id.
            "AX-DiffusionGemma-26B-A4B-IT-MLX-4bit",
            "axe-qwen3.5-9b",
            "max-qwen3.5-9b",
            "ax-llama3.3-70b",
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
                    "model_family": "qwen3_5", "layer_count": 32, "hidden_size": 4096,
                    "intermediate_size": 12288, "attention_head_count": 16,
                    "attention_head_dim": 256, "kv_head_count": 4, "vocab_size": 248320
                }),
                MultiModelTarget::Qwen35_9b,
            ),
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
            (
                serde_json::json!({
                    "model_family": "qwen3_next", "layer_count": 48, "hidden_size": 2048,
                    "intermediate_size": 5120, "attention_head_count": 16,
                    "attention_head_dim": 256, "kv_head_count": 2, "vocab_size": 151936,
                    "moe": {"expert_count": 512, "experts_per_token": 10,
                            "expert_intermediate_size": 512}
                }),
                MultiModelTarget::Qwen3CoderNext,
            ),
            (
                serde_json::json!({
                    "model_family": "embeddinggemma", "layer_count": 24, "hidden_size": 768,
                    "intermediate_size": 1152, "attention_head_count": 3,
                    "attention_head_dim": 256, "kv_head_count": 1, "vocab_size": 262144
                }),
                MultiModelTarget::EmbeddingGemma300m,
            ),
            (
                serde_json::json!({
                    "model_family": "qwen3", "layer_count": 28, "hidden_size": 1024,
                    "intermediate_size": 3072, "attention_head_count": 16,
                    "attention_head_dim": 128, "kv_head_count": 8, "vocab_size": 151669
                }),
                MultiModelTarget::Qwen3Embedding0_6b,
            ),
            (
                serde_json::json!({
                    "model_family": "qwen3", "layer_count": 36, "hidden_size": 2560,
                    "intermediate_size": 9728, "attention_head_count": 32,
                    "attention_head_dim": 128, "kv_head_count": 8, "vocab_size": 151665
                }),
                MultiModelTarget::Qwen3Embedding4b,
            ),
            (
                serde_json::json!({
                    "model_family": "qwen3", "layer_count": 36, "hidden_size": 4096,
                    "intermediate_size": 12288, "attention_head_count": 32,
                    "attention_head_dim": 128, "kv_head_count": 8, "vocab_size": 151665
                }),
                MultiModelTarget::Qwen3Embedding8b,
            ),
        ];
        for (identity, expected) in identities {
            let identity = serde_json::from_value(identity).expect("identity should parse");
            assert_eq!(multi_model_target_from_manifest(&identity), Some(expected));
        }
    }

    /// The four AutomatosX pack manifests that ship with the org models
    /// (https://huggingface.co/AutomatosX) resolve to the label the org
    /// publishes them under, id-side and artifact-side agreeing.
    #[test]
    fn automatosx_pack_ids_agree_with_shipped_manifest_identities() {
        let cases = [
            (
                "AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-8bit",
                serde_json::json!({
                    "model_family": "qwen3", "layer_count": 28, "hidden_size": 1024,
                    "intermediate_size": 3072, "attention_head_count": 16,
                    "attention_head_dim": 128, "kv_head_count": 8, "vocab_size": 151669
                }),
            ),
            (
                "AutomatosX/AX-EmbeddingGemma-300M-MLX-8bit",
                serde_json::json!({
                    "model_family": "embeddinggemma", "layer_count": 24, "hidden_size": 768,
                    "intermediate_size": 1152, "attention_head_count": 3,
                    "attention_head_dim": 256, "kv_head_count": 1, "vocab_size": 262144
                }),
            ),
            (
                "AutomatosX/AX-Qwen3-Coder-Next-MLX-4bit",
                serde_json::json!({
                    "model_family": "qwen3_next", "layer_count": 48, "hidden_size": 2048,
                    "intermediate_size": 5120, "attention_head_count": 16,
                    "attention_head_dim": 256, "kv_head_count": 2, "vocab_size": 151936,
                    "moe": {"expert_count": 512, "experts_per_token": 10,
                            "expert_intermediate_size": 512}
                }),
            ),
            (
                "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP",
                serde_json::json!({
                    "model_family": "qwen3_5", "layer_count": 40, "hidden_size": 2048,
                    "intermediate_size": 0, "attention_head_count": 16,
                    "attention_head_dim": 256, "kv_head_count": 2, "vocab_size": 248320,
                    "moe": {"expert_count": 256, "experts_per_token": 8,
                            "expert_intermediate_size": 512}
                }),
            ),
        ];
        for (model_id, identity) in cases {
            let identity = serde_json::from_value(identity).expect("identity should parse");
            assert_eq!(
                multi_model_target_key(model_id),
                multi_model_target_from_manifest(&identity),
                "{model_id}"
            );
            assert!(multi_model_target_key(model_id).is_some(), "{model_id}");
        }
    }

    #[test]
    fn footprint_estimate_adds_runtime_factor_and_floor() {
        let weights = 16 * 1024 * 1024 * 1024u64; // 16 GiB on disk
        let legacy = estimated_footprint_bytes(weights, None);
        assert_eq!(
            legacy,
            weights + weights / 8 + LOAD_FOOTPRINT_FIXED_FLOOR_BYTES
        );

        let kv_bytes = 3 * 1024 * 1024 * 1024u64;
        let with_kv = estimated_footprint_bytes(weights, Some(kv_bytes));
        assert_eq!(
            with_kv,
            weights + weights / 8 + kv_bytes + LOAD_FOOTPRINT_RUNTIME_FLOOR_BYTES
        );
    }

    fn kv_geometry(value: serde_json::Value) -> ManifestKvGeometry {
        serde_json::from_value(value).expect("geometry should parse")
    }

    #[test]
    fn kv_pool_bytes_charges_dense_layers_at_pool() {
        let geometry = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2
        }));
        let per_token_layer = 2 * 128 * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        assert_eq!(
            estimated_kv_pool_bytes(&geometry, 16384),
            Some(4 * 16384 * per_token_layer)
        );
    }

    #[test]
    fn kv_pool_bytes_charges_sliding_layers_at_pool_and_skips_shared() {
        let geometry = kv_geometry(serde_json::json!({
            "layer_count": 6, "attention_head_dim": 128, "kv_head_count": 2,
            "global_head_dim": 256, "sliding_window_size": 512,
            "layer_types": [
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
            "kv_shared_source_layers": {"3": 0}
        }));
        let pool = 16384u64;
        let sliding_per_token = 2 * 128 * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        let full_per_token = 2 * 256 * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        // Layers 0, 1, 4 are sliding: rings bound KV per request, so the
        // pool-wide worst case (many concurrent ≤window requests) is the
        // pool at the sliding head dim — the window never caps admission.
        // Layer 3 shares layer 0's KV and charges nothing; layers 2 and 5
        // are full attention at the pool, using the ISWA global head dim.
        let expected = 3 * pool * sliding_per_token + 2 * pool * full_per_token;
        assert_eq!(estimated_kv_pool_bytes(&geometry, pool), Some(expected));
    }

    #[test]
    fn kv_pool_bytes_charges_only_hybrid_full_attention_layers() {
        let explicit = kv_geometry(serde_json::json!({
            "model_family": "qwen3_5",
            "layer_count": 8, "attention_head_dim": 256, "kv_head_count": 4,
            "linear_attention": {"full_attention_interval": 4}
        }));
        let per_token_layer = 4 * 256 * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        // Layers 3 and 7 (every 4th) keep KV; linear layers hold per-request
        // state covered by the runtime floor.
        assert_eq!(
            estimated_kv_pool_bytes(&explicit, 1000),
            Some(2 * 1000 * per_token_layer)
        );

        let family_default = kv_geometry(serde_json::json!({
            "model_family": "qwen3_5",
            "layer_count": 8, "attention_head_dim": 256, "kv_head_count": 4,
            "linear_attention": {"num_key_heads": 16}
        }));
        assert_eq!(
            estimated_kv_pool_bytes(&family_default, 1000),
            Some(2 * 1000 * per_token_layer)
        );
    }

    #[test]
    fn kv_pool_bytes_fails_open_on_unknown_geometry() {
        let mla = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2,
            "mla_attention": {"kv_lora_rank": 512}
        }));
        assert_eq!(estimated_kv_pool_bytes(&mla, 1000), None);

        let diffusion = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2,
            "diffusion": {"canvas_size": 64}
        }));
        assert_eq!(estimated_kv_pool_bytes(&diffusion, 1000), None);

        let zero_layers = kv_geometry(serde_json::json!({
            "layer_count": 0, "attention_head_dim": 128, "kv_head_count": 2
        }));
        assert_eq!(estimated_kv_pool_bytes(&zero_layers, 1000), None);

        // Linear attention enabled on a family with no default interval and
        // no explicit interval: geometry is unknowable.
        let unresolved_hybrid = kv_geometry(serde_json::json!({
            "model_family": "not_a_hybrid_family",
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2,
            "linear_attention": {"num_key_heads": 16}
        }));
        assert_eq!(estimated_kv_pool_bytes(&unresolved_hybrid, 1000), None);

        // A sliding annotation charges the pool at the sliding head dim
        // whether or not the manifest carries a window (rings are a
        // per-request bound, so the window never lowers admission).
        let sliding_without_window = kv_geometry(serde_json::json!({
            "layer_count": 1, "attention_head_dim": 128, "kv_head_count": 2,
            "layer_types": ["sliding_attention"]
        }));
        assert_eq!(
            estimated_kv_pool_bytes(&sliding_without_window, 1000),
            Some(1000 * 2 * 128 * KV_CACHE_BYTES_PER_HEAD_ELEMENT)
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
        let supported = vec![
            "qwen3.5-9b".to_string(),
            "qwen3.6-27b".to_string(),
            "gemma-4-31b-it".to_string(),
        ];
        validate_retained_multi_model_ids(&supported)
            .expect("targeted multi-model ids should be accepted");

        let unsupported = vec!["qwen3.6-27b".to_string(), "llama3.3-70b".to_string()];
        let error = validate_retained_multi_model_ids(&unsupported)
            .expect_err("an out-of-scope retained model must be rejected");
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(error.1.0.error.code.as_deref(), Some("unsupported_model"));
    }
}
