use std::convert::Infallible;
use std::path::{Component, Path as FsPath, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use ax_engine_core::chat::{
    ChatMessage as CoreChatMessage, ChatRenderOptions, ChatRole as CoreChatRole,
    render_chat_messages,
};
use ax_engine_sdk::{
    BackendKind, FinishReason, GenerationOptions, LoadOptions, Model, PromptCacheStats, Session,
    SessionOptions, SessionSnapshot, TokenPiece,
};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

mod args;

use args::ServerArgs;

struct AppState {
    model: Model,
    model_path: String,
    model_alias: String,
    created: u64,
    model_size: Option<u64>,
    api_key: Option<String>,
    defaults: ServerDefaults,
    status_args: Vec<String>,
    no_slots: bool,
    metrics_enabled: bool,
    slot_save_path: Option<PathBuf>,
    slot_cache: Mutex<Option<SessionSnapshot>>,
    slot_state: Mutex<SlotState>,
    next_task_id: AtomicU64,
    metrics: MetricsState,
    inference_slot: Arc<Semaphore>,
}

#[derive(Debug, Clone)]
struct ServerDefaults {
    n_ctx: usize,
    n_predict: i32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: i32,
    frequency_penalty: f32,
    presence_penalty: f32,
    seed: Option<u64>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

#[derive(Debug, Clone)]
struct SlotState {
    id_task: u64,
    is_processing: bool,
    params: SlotParams,
    next_token: SlotNextToken,
}

#[derive(Debug, Clone)]
struct SlotParams {
    max_tokens: usize,
    seed: Option<u64>,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: i32,
    frequency_penalty: f32,
    presence_penalty: f32,
    stop: Vec<String>,
    stream: bool,
}

#[derive(Debug, Clone, Copy)]
struct SlotNextToken {
    has_next_token: bool,
    has_new_line: bool,
    n_remain: i64,
    n_decoded: usize,
}

#[derive(Debug, Default)]
struct MetricsState {
    prompt_tokens_total: AtomicU64,
    predicted_tokens_total: AtomicU64,
    predicted_micros_total: AtomicU64,
    requests_processing: AtomicU64,
    n_tokens_max: AtomicU64,
}

type ApiResult<T> = Result<T, ApiError>;

#[derive(Debug, Deserialize)]
struct TokenizeRequest {
    content: String,
    #[serde(default)]
    add_special: bool,
    #[serde(default = "default_true")]
    parse_special: bool,
    #[serde(default)]
    with_pieces: bool,
}

#[derive(Debug, Deserialize)]
struct SlotsQuery {
    fail_on_no_slot: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SlotActionQuery {
    action: String,
}

#[derive(Debug, Deserialize)]
struct SlotPersistenceRequest {
    filename: String,
}

#[derive(Debug, Deserialize)]
struct DetokenizeRequest {
    tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct ApplyTemplateRequest {
    messages: Vec<ChatInputMessage>,
    #[serde(default)]
    add_generation_prompt: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct LegacyCompletionRequest {
    prompt: Value,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    id_slot: Option<u32>,
    #[serde(default = "default_true")]
    cache_prompt: bool,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    n_predict: Option<i32>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    repeat_last_n: Option<i32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    return_tokens: bool,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: Value,
    #[serde(default)]
    id_slot: Option<u32>,
    #[serde(default = "default_true")]
    cache_prompt: bool,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    n_predict: Option<i32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    repeat_last_n: Option<i32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    seed: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatInputMessage>,
    #[serde(default)]
    id_slot: Option<u32>,
    #[serde(default = "default_true")]
    cache_prompt: bool,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    n_predict: Option<i32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    repeat_last_n: Option<i32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    response_format: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ResponsesRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    id_slot: Option<u32>,
    #[serde(default = "default_true")]
    cache_prompt: bool,
    #[serde(default)]
    instructions: Option<Value>,
    #[serde(default)]
    input: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_output_tokens: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    repeat_last_n: Option<i32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize, Clone)]
struct ChatInputMessage {
    role: String,
    #[serde(default)]
    content: Option<MessageContent>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<MessagePart>),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MessagePart {
    Text {
        text: String,
    },
    InputText {
        text: String,
    },
    OutputText {
        text: String,
    },
    Refusal {
        refusal: String,
    },
    ImageUrl {
        #[serde(rename = "image_url")]
        _image_url: Value,
    },
    InputImage {
        #[serde(rename = "image_url")]
        _image_url: Option<String>,
        #[serde(rename = "file_id")]
        _file_id: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopInput {
    Single(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone)]
struct PreparedPrompt {
    tokens: Vec<u32>,
}

#[derive(Debug, Clone)]
struct RenderedMessage {
    role: CoreChatRole,
    content: String,
}

#[derive(Debug, Clone)]
struct GenerationOverrides {
    max_tokens: Option<usize>,
    n_predict: Option<i32>,
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    stop: Option<Vec<String>>,
    seed: Option<i64>,
}

#[derive(Debug, Clone)]
struct RunResult {
    output: ax_engine_sdk::GenerationOutput,
    elapsed_ms: f64,
    cache_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct SlotSnapshotFile {
    format: String,
    model_id: String,
    architecture: String,
    vocab_size: usize,
    bos_token_id: u32,
    eos_token_id: u32,
    context_length: usize,
    tokens: Vec<u32>,
}

const SLOT_SNAPSHOT_FORMAT: &str = "ax-engine.slot.tokens.v1";

#[derive(Debug, Serialize)]
struct TokenPieceResponse {
    id: u32,
    piece: Value,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();

    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("ax_engine_core=info,ax_engine_server=debug")
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("ax_engine_core=warn,ax_engine_server=info")
            .init();
    }

    for warning in args.compatibility_warnings() {
        tracing::warn!("{warning}");
    }

    if args.threads > 0 {
        ax_engine_core::scheduler::init_global_threadpool_with_count(args.threads as usize);
    } else {
        ax_engine_core::scheduler::init_global_threadpool();
    }

    let load_options = LoadOptions {
        backend: BackendKind::Auto,
        context_length: (args.ctx_size > 0).then_some(args.ctx_size),
    };
    let model = Model::load(&args.model, load_options)
        .with_context(|| format!("failed to load model: {}", args.model))?;
    let model_info = model.info();
    let slot_save_path = match args.slot_save_path.as_deref() {
        Some(path) => {
            let path = PathBuf::from(path);
            if path.exists() && !path.is_dir() {
                anyhow::bail!("--slot-save-path must point to a directory");
            }
            Some(path)
        }
        None => None,
    };
    let default_seed = normalize_seed(Some(args.seed));
    let default_max_tokens = resolve_max_tokens(args.n_predict, None, None);
    let state = Arc::new(AppState {
        model,
        model_path: args.model.clone(),
        model_alias: args.model_alias(),
        created: unix_timestamp(),
        model_size: std::fs::metadata(&args.model)
            .ok()
            .map(|metadata| metadata.len()),
        api_key: args.api_key.clone(),
        defaults: ServerDefaults {
            n_ctx: model_info.context_length,
            n_predict: args.n_predict,
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            min_p: args.min_p,
            repeat_penalty: args.repeat_penalty,
            repeat_last_n: args.repeat_last_n,
            frequency_penalty: args.frequency_penalty,
            presence_penalty: args.presence_penalty,
            seed: default_seed,
        },
        status_args: args.status_args(),
        no_slots: args.no_slots,
        metrics_enabled: args.metrics,
        slot_save_path,
        slot_cache: Mutex::new(None),
        slot_state: Mutex::new(SlotState {
            id_task: 0,
            is_processing: false,
            params: SlotParams {
                max_tokens: default_max_tokens,
                seed: default_seed,
                temperature: args.temperature,
                top_k: args.top_k,
                top_p: args.top_p,
                min_p: args.min_p,
                repeat_penalty: args.repeat_penalty,
                repeat_last_n: args.repeat_last_n,
                frequency_penalty: args.frequency_penalty,
                presence_penalty: args.presence_penalty,
                stop: Vec::new(),
                stream: false,
            },
            next_token: SlotNextToken {
                has_next_token: false,
                has_new_line: false,
                n_remain: max_tokens_to_slots_remaining(default_max_tokens),
                n_decoded: 0,
            },
        }),
        next_task_id: AtomicU64::new(1),
        metrics: MetricsState::default(),
        inference_slot: Arc::new(Semaphore::new(1)),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/v1/health", get(health))
        .route("/models", get(list_models))
        .route("/v1/models", get(list_models_v1))
        .route("/props", get(get_props))
        .route("/completion", post(legacy_completion))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/responses", post(openai_responses))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/apply-template", post(apply_template))
        .route("/slots", get(get_slots))
        .route("/slots/{id_slot}", post(slot_action))
        .route("/metrics", get(metrics))
        .route("/embedding", post(unsupported_embedding))
        .route("/v1/embeddings", post(unsupported_embedding))
        .route("/reranking", post(unsupported_reranking))
        .route("/rerank", post(unsupported_reranking))
        .route("/v1/rerank", post(unsupported_reranking))
        .route("/v1/reranking", post(unsupported_reranking))
        .route("/infill", post(unsupported_infill))
        .fallback(not_found)
        .with_state(state.clone());

    let bind_addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("failed to bind {bind_addr}"))?;

    tracing::info!(
        model = %state.model_alias,
        path = %state.model_path,
        address = %bind_addr,
        "ax-engine-server ready"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server failed")?;

    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    require_api_key(&state, &headers)?;
    let info = state.model.info();
    Ok(Json(json!({
        "data": [{
            "id": state.model_alias,
            "in_cache": false,
            "path": state.model_path,
            "status": {
                "value": "loaded",
                "args": state.status_args,
            },
            "meta": {
                "architecture": info.architecture,
                "n_vocab": info.vocab_size,
                "n_ctx_train": info.context_length,
                "size": state.model_size,
            },
            "capabilities": {
                "completion": true,
                "chat_completion": true,
                "tokenize": true,
                "apply_template": true,
                "multimodal": false,
                "embedding": false,
                "reranking": false
            }
        }]
    })))
}

async fn list_models_v1(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    require_api_key(&state, &headers)?;
    let info = state.model.info();
    Ok(Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_alias,
            "object": "model",
            "created": state.created,
            "owned_by": "ax-engine",
            "meta": {
                "architecture": info.architecture,
                "n_vocab": info.vocab_size,
                "n_ctx_train": info.context_length,
                "size": state.model_size,
                "model_name": info.model_name,
                "support_note": info.support_note
            }
        }]
    })))
}

async fn get_props(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> ApiResult<Json<Value>> {
    require_api_key(&state, &headers)?;
    Ok(Json(json!({
        "default_generation_settings": {
            "id": 0,
            "id_task": -1,
            "n_ctx": state.defaults.n_ctx,
            "speculative": false,
            "is_processing": false,
            "params": generation_settings_json(&state, &build_generation_options(&state.defaults, GenerationOverrides {
                max_tokens: None,
                n_predict: None,
                temperature: None,
                top_k: None,
                top_p: None,
                min_p: None,
                repeat_penalty: None,
                repeat_last_n: None,
                frequency_penalty: None,
                presence_penalty: None,
                stop: None,
                seed: None,
            }))
        },
        "model_path": state.model_path,
        "model_alias": state.model_alias,
        "chat_template": state.model.chat_template(),
        "total_slots": 1
    })))
}

async fn get_slots(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<SlotsQuery>,
) -> ApiResult<Json<Value>> {
    require_api_key(&state, &headers)?;
    ensure_slots_enabled(&state)?;

    if query.fail_on_no_slot.as_deref().is_some_and(is_truthy)
        && state.inference_slot.available_permits() == 0
    {
        return Err(ApiError::service_unavailable("no available slots"));
    }

    let slot_state = state
        .slot_state
        .lock()
        .map_err(|_| ApiError::internal("slot state lock poisoned"))?
        .clone();

    Ok(Json(json!([slot_state_json(&state, &slot_state)])))
}

async fn slot_action(
    State(state): State<Arc<AppState>>,
    Path(id_slot): Path<u32>,
    Query(query): Query<SlotActionQuery>,
    headers: HeaderMap,
    body: String,
) -> ApiResult<Json<Value>> {
    require_api_key(&state, &headers)?;
    ensure_slots_enabled(&state)?;
    ensure_slot_id(id_slot)?;

    match query.action.as_str() {
        "erase" => {
            ensure_slot_idle(&state)?;
            replace_slot_snapshot(&state, None)?;
            let mut slot_state = state
                .slot_state
                .lock()
                .map_err(|_| ApiError::internal("slot state lock poisoned"))?;
            let n_erased = slot_state.next_token.n_decoded;
            slot_state.id_task = 0;
            slot_state.is_processing = false;
            slot_state.next_token = SlotNextToken {
                has_next_token: false,
                has_new_line: false,
                n_remain: max_tokens_to_slots_remaining(slot_state.params.max_tokens),
                n_decoded: 0,
            };
            Ok(Json(json!({
                "id_slot": id_slot,
                "n_erased": n_erased
            })))
        }
        "save" => {
            ensure_slot_idle(&state)?;
            let request = parse_slot_persistence_request(&body)?;
            let output_path = resolve_slot_snapshot_path(&state, &request.filename)?;
            let tokens = cloned_slot_snapshot(&state)?
                .map(|snapshot| snapshot.history_tokens().to_vec())
                .unwrap_or_default();
            let model_info = state.model.info();
            let snapshot = SlotSnapshotFile {
                format: SLOT_SNAPSHOT_FORMAT.to_string(),
                model_id: state.model.id().to_string(),
                architecture: state.model.architecture().to_string(),
                vocab_size: model_info.vocab_size,
                bos_token_id: model_info.bos_token_id,
                eos_token_id: model_info.eos_token_id,
                context_length: state.defaults.n_ctx,
                tokens,
            };
            let started = Instant::now();
            let encoded = serde_json::to_vec(&snapshot).map_err(|error| {
                ApiError::internal(format!("failed to encode slot snapshot: {error}"))
            })?;
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| {
                    ApiError::internal(format!("failed to create slot snapshot directory: {error}"))
                })?;
            }
            std::fs::write(&output_path, &encoded).map_err(|error| {
                ApiError::internal(format!("failed to write slot snapshot: {error}"))
            })?;
            Ok(Json(json!({
                "id_slot": id_slot,
                "filename": request.filename,
                "n_saved": snapshot.tokens.len(),
                "n_written": encoded.len(),
                "timings": {
                    "save_ms": started.elapsed().as_secs_f64() * 1_000.0
                }
            })))
        }
        "restore" => {
            ensure_slot_idle(&state)?;
            let request = parse_slot_persistence_request(&body)?;
            let input_path = resolve_slot_snapshot_path(&state, &request.filename)?;
            let started = Instant::now();
            let encoded = std::fs::read(&input_path).map_err(|error| match error.kind() {
                std::io::ErrorKind::NotFound => ApiError::not_found(format!(
                    "slot snapshot '{}' was not found",
                    request.filename
                )),
                _ => ApiError::internal(format!("failed to read slot snapshot: {error}")),
            })?;
            let snapshot: SlotSnapshotFile = serde_json::from_slice(&encoded)
                .map_err(|_| ApiError::bad_request("slot snapshot file is not valid JSON"))?;
            validate_slot_snapshot(&state, &snapshot)?;
            let restored_snapshot = build_slot_snapshot_from_tokens(&state, &snapshot.tokens)?;
            replace_slot_snapshot(&state, restored_snapshot)?;
            reset_slot_tracking_state(&state)?;
            Ok(Json(json!({
                "id_slot": id_slot,
                "filename": request.filename,
                "n_restored": snapshot.tokens.len(),
                "n_read": encoded.len(),
                "timings": {
                    "restore_ms": started.elapsed().as_secs_f64() * 1_000.0
                }
            })))
        }
        other => Err(ApiError::bad_request(format!(
            "unsupported slot action '{other}'"
        ))),
    }
}

async fn metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> ApiResult<Response> {
    require_api_key(&state, &headers)?;
    if !state.metrics_enabled {
        return Err(ApiError::not_supported(
            "This server does not support metrics endpoint.",
        ));
    }

    let prompt_tokens_total = state.metrics.prompt_tokens_total.load(Ordering::Relaxed);
    let predicted_tokens_total = state.metrics.predicted_tokens_total.load(Ordering::Relaxed);
    let predicted_micros_total = state.metrics.predicted_micros_total.load(Ordering::Relaxed);
    let requests_processing = state.metrics.requests_processing.load(Ordering::Relaxed);
    let n_tokens_max = state.metrics.n_tokens_max.load(Ordering::Relaxed);
    let predicted_tokens_seconds = if predicted_micros_total > 0 {
        predicted_tokens_total as f64 / (predicted_micros_total as f64 / 1_000_000.0)
    } else {
        0.0
    };

    let body = format!(
        concat!(
            "# TYPE llamacpp:prompt_tokens_total counter\n",
            "llamacpp:prompt_tokens_total {prompt_tokens_total}\n",
            "# TYPE llamacpp:tokens_predicted_total counter\n",
            "llamacpp:tokens_predicted_total {predicted_tokens_total}\n",
            "# TYPE llamacpp:prompt_tokens_seconds gauge\n",
            "llamacpp:prompt_tokens_seconds 0\n",
            "# TYPE llamacpp:predicted_tokens_seconds gauge\n",
            "llamacpp:predicted_tokens_seconds {predicted_tokens_seconds}\n",
            "# TYPE llamacpp:kv_cache_usage_ratio gauge\n",
            "llamacpp:kv_cache_usage_ratio 0\n",
            "# TYPE llamacpp:kv_cache_tokens gauge\n",
            "llamacpp:kv_cache_tokens 0\n",
            "# TYPE llamacpp:requests_processing gauge\n",
            "llamacpp:requests_processing {requests_processing}\n",
            "# TYPE llamacpp:requests_deferred gauge\n",
            "llamacpp:requests_deferred 0\n",
            "# TYPE llamacpp:n_tokens_max gauge\n",
            "llamacpp:n_tokens_max {n_tokens_max}\n"
        ),
        prompt_tokens_total = prompt_tokens_total,
        predicted_tokens_total = predicted_tokens_total,
        predicted_tokens_seconds = predicted_tokens_seconds,
        requests_processing = requests_processing,
        n_tokens_max = n_tokens_max,
    );

    Ok((
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response())
}

async fn tokenize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<TokenizeRequest>,
) -> ApiResult<Json<Value>> {
    require_api_key(&state, &headers)?;
    let token_ids = state.model.tokenize_with_options(
        &request.content,
        request.add_special,
        request.parse_special,
    );
    if request.with_pieces {
        let tokens = token_ids
            .iter()
            .map(|&token_id| TokenPieceResponse {
                id: token_id,
                piece: token_piece_json(state.model.token_piece(token_id)),
            })
            .collect::<Vec<_>>();
        return Ok(Json(json!({ "tokens": tokens })));
    }

    Ok(Json(json!({ "tokens": token_ids })))
}

async fn detokenize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<DetokenizeRequest>,
) -> ApiResult<Json<Value>> {
    require_api_key(&state, &headers)?;
    Ok(Json(
        json!({ "content": state.model.decode(&request.tokens) }),
    ))
}

async fn apply_template(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ApplyTemplateRequest>,
) -> ApiResult<Json<Value>> {
    require_api_key(&state, &headers)?;
    let messages = normalize_chat_messages(&request.messages)?;
    let prompt = render_chat_prompt(
        &messages,
        state.model.architecture(),
        request.add_generation_prompt.unwrap_or(true),
    );
    Ok(Json(json!({ "prompt": prompt })))
}

async fn legacy_completion(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<LegacyCompletionRequest>,
) -> ApiResult<Response> {
    require_api_key(&state, &headers)?;
    ensure_requested_model(&state, request.model.as_deref())?;
    let prompts = normalize_completion_input(&state.model, &request.prompt)?;
    let use_slot_cache =
        use_slot_prompt_cache(request.id_slot, request.cache_prompt, prompts.len())?;
    let overrides = GenerationOverrides {
        max_tokens: request.max_tokens,
        n_predict: request.n_predict,
        temperature: request.temperature,
        top_k: request.top_k,
        top_p: request.top_p,
        min_p: request.min_p,
        repeat_penalty: request.repeat_penalty,
        repeat_last_n: request.repeat_last_n,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        stop: request.stop.map(StopInput::into_vec),
        seed: request.seed,
    };
    let generation_options = build_generation_options(&state.defaults, overrides);

    if request.stream {
        if prompts.len() != 1 {
            return Err(ApiError::bad_request(
                "streaming /completion only supports a single prompt",
            ));
        }
        let permit = acquire_inference_slot(&state).await?;
        return Ok(legacy_completion_stream(
            state,
            prompts[0].clone(),
            generation_options,
            permit,
            use_slot_cache,
        ));
    }

    let mut results = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let run = if use_slot_cache {
            run_cached_generation(
                state.clone(),
                prompt.tokens.clone(),
                generation_options.clone(),
                false,
            )
            .await?
        } else {
            run_generation(
                state.clone(),
                prompt.tokens.clone(),
                generation_options.clone(),
                false,
            )
            .await?
        };
        results.push(legacy_completion_body(
            &state,
            &prompt.tokens,
            &run,
            &generation_options,
            request.return_tokens,
        ));
    }

    if results.len() == 1 {
        Ok(Json(results.pop().unwrap_or_else(|| json!({}))).into_response())
    } else {
        Ok(Json(Value::Array(results)).into_response())
    }
}

async fn openai_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> ApiResult<Response> {
    require_api_key(&state, &headers)?;
    ensure_requested_model(&state, request.model.as_deref())?;
    let prompts = normalize_completion_input(&state.model, &request.prompt)?;
    let use_slot_cache =
        use_slot_prompt_cache(request.id_slot, request.cache_prompt, prompts.len())?;
    let generation_options = build_generation_options(
        &state.defaults,
        GenerationOverrides {
            max_tokens: request.max_tokens,
            n_predict: request.n_predict,
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            min_p: request.min_p,
            repeat_penalty: request.repeat_penalty,
            repeat_last_n: request.repeat_last_n,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop.map(StopInput::into_vec),
            seed: request.seed,
        },
    );

    if request.stream {
        if prompts.len() != 1 {
            return Err(ApiError::bad_request(
                "streaming /v1/completions only supports a single prompt",
            ));
        }
        let permit = acquire_inference_slot(&state).await?;
        return Ok(openai_completion_stream(
            state,
            prompts[0].clone(),
            generation_options,
            permit,
            use_slot_cache,
        ));
    }

    let created = unix_timestamp();
    let request_id = format!("cmpl-{}", Uuid::new_v4().simple());
    let mut choices = Vec::with_capacity(prompts.len());
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;

    for (index, prompt) in prompts.iter().enumerate() {
        let run = if use_slot_cache {
            run_cached_generation(
                state.clone(),
                prompt.tokens.clone(),
                generation_options.clone(),
                false,
            )
            .await?
        } else {
            run_generation(
                state.clone(),
                prompt.tokens.clone(),
                generation_options.clone(),
                false,
            )
            .await?
        };
        prompt_tokens += run.output.usage.prompt_tokens;
        completion_tokens += run.output.usage.completion_tokens;
        choices.push(json!({
            "text": run.output.text,
            "index": index,
            "logprobs": Value::Null,
            "finish_reason": finish_reason_name(run.output.finish_reason),
        }));
    }

    Ok(Json(json!({
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": state.model_alias,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }))
    .into_response())
}

async fn openai_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> ApiResult<Response> {
    require_api_key(&state, &headers)?;
    ensure_requested_model(&state, request.model.as_deref())?;
    ensure_requested_slot(request.id_slot)?;
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(ApiError::not_implemented(
            "tool calling is not implemented yet",
        ));
    }
    if request.response_format.is_some() {
        return Err(ApiError::not_implemented(
            "response_format is not implemented yet",
        ));
    }

    let normalized_messages = normalize_chat_messages(&request.messages)?;
    let prompt = render_chat_prompt(&normalized_messages, state.model.architecture(), true);
    let prompt_tokens = state.model.tokenize_with_options(&prompt, true, true);
    let use_slot_cache = request.cache_prompt;
    let generation_options = build_generation_options(
        &state.defaults,
        GenerationOverrides {
            max_tokens: request.max_tokens,
            n_predict: request.n_predict,
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            min_p: request.min_p,
            repeat_penalty: request.repeat_penalty,
            repeat_last_n: request.repeat_last_n,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop.map(StopInput::into_vec),
            seed: request.seed,
        },
    );

    if request.stream {
        let permit = acquire_inference_slot(&state).await?;
        return Ok(openai_chat_stream(
            state,
            PreparedPrompt {
                tokens: prompt_tokens,
            },
            generation_options,
            permit,
            use_slot_cache,
        ));
    }

    let created = unix_timestamp();
    let request_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let run = if use_slot_cache {
        run_cached_generation(state.clone(), prompt_tokens, generation_options, false).await?
    } else {
        run_generation(state.clone(), prompt_tokens, generation_options, false).await?
    };
    Ok(Json(json!({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": state.model_alias,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": run.output.text,
            },
            "finish_reason": finish_reason_name(run.output.finish_reason),
        }],
        "usage": {
            "prompt_tokens": run.output.usage.prompt_tokens,
            "completion_tokens": run.output.usage.completion_tokens,
            "total_tokens": run.output.usage.total_tokens
        },
        "timings": timings_json(&run)
    }))
    .into_response())
}

async fn openai_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ResponsesRequest>,
) -> ApiResult<Response> {
    require_api_key(&state, &headers)?;
    ensure_requested_model(&state, request.model.as_deref())?;
    ensure_requested_slot(request.id_slot)?;
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(ApiError::not_implemented(
            "responses.tools is not implemented yet",
        ));
    }

    let messages =
        normalize_responses_messages(request.instructions.as_ref(), request.input.as_ref())?;
    if messages.is_empty() {
        return Err(ApiError::bad_request("responses.input must not be empty"));
    }

    let prompt = render_chat_prompt(&messages, state.model.architecture(), true);
    let prompt_tokens = state.model.tokenize_with_options(&prompt, true, true);
    let use_slot_cache = request.cache_prompt;
    let generation_options = build_generation_options(
        &state.defaults,
        GenerationOverrides {
            max_tokens: request.max_output_tokens.or(request.max_tokens),
            n_predict: None,
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            min_p: request.min_p,
            repeat_penalty: request.repeat_penalty,
            repeat_last_n: request.repeat_last_n,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop.map(StopInput::into_vec),
            seed: request.seed,
        },
    );

    if request.stream {
        let permit = acquire_inference_slot(&state).await?;
        return Ok(openai_responses_stream(
            state,
            PreparedPrompt {
                tokens: prompt_tokens,
            },
            generation_options,
            permit,
            use_slot_cache,
        ));
    }

    let run = if use_slot_cache {
        run_cached_generation(state.clone(), prompt_tokens, generation_options, false).await?
    } else {
        run_generation(state.clone(), prompt_tokens, generation_options, false).await?
    };
    Ok(Json(response_object_json(
        &format!("resp_{}", Uuid::new_v4().simple()),
        unix_timestamp(),
        &state.model_alias,
        "completed",
        &run.output.text,
        Some(&run.output.usage),
        Some(run.output.finish_reason),
    ))
    .into_response())
}

async fn unsupported_embedding() -> ApiError {
    ApiError::not_implemented("embedding routes are not implemented in ax-engine-server yet")
}

async fn unsupported_reranking() -> ApiError {
    ApiError::not_implemented("reranking routes are not implemented in ax-engine-server yet")
}

async fn unsupported_infill() -> ApiError {
    ApiError::not_implemented("infill is not implemented in ax-engine-server yet")
}

async fn not_found() -> ApiError {
    ApiError::not_found("route not found")
}

fn require_api_key(state: &AppState, headers: &HeaderMap) -> ApiResult<()> {
    let Some(expected_key) = state.api_key.as_deref() else {
        return Ok(());
    };

    let provided = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
        .or_else(|| {
            headers
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.strip_prefix("Bearer "))
                .map(str::to_owned)
        });

    match provided {
        Some(provided) if provided == expected_key => Ok(()),
        _ => Err(ApiError::unauthorized("invalid or missing api key")),
    }
}

fn ensure_requested_model(state: &AppState, requested_model: Option<&str>) -> ApiResult<()> {
    let Some(requested_model) = requested_model.filter(|value| !value.trim().is_empty()) else {
        return Ok(());
    };
    if requested_model == state.model_alias
        || requested_model == state.model_path
        || requested_model == state.model.id()
    {
        return Ok(());
    }

    Err(ApiError::not_found(format!(
        "model '{requested_model}' is not loaded"
    )))
}

fn normalize_completion_input(model: &Model, value: &Value) -> ApiResult<Vec<PreparedPrompt>> {
    match value {
        Value::Array(items) if !items.is_empty() => {
            let all_strings = items.iter().all(Value::is_string);
            let all_numbers = items.iter().all(Value::is_number);
            let all_primitives = items
                .iter()
                .all(|item| item.is_string() || item.is_number());

            if all_strings {
                items
                    .iter()
                    .map(|item| normalize_single_prompt(model, item))
                    .collect()
            } else if all_numbers || all_primitives {
                Ok(vec![normalize_single_prompt(model, value)?])
            } else {
                items
                    .iter()
                    .map(|item| normalize_single_prompt(model, item))
                    .collect()
            }
        }
        _ => Ok(vec![normalize_single_prompt(model, value)?]),
    }
}

fn normalize_single_prompt(model: &Model, value: &Value) -> ApiResult<PreparedPrompt> {
    let tokens = match value {
        Value::String(text) => model.tokenize_with_options(text, true, true),
        Value::Array(items) => normalize_prompt_array(model, items)?,
        Value::Object(object) => normalize_prompt_object(model, object)?,
        _ => {
            return Err(ApiError::bad_request(
                "prompt must be a string, token array, prompt object, or batch array",
            ));
        }
    };

    if tokens.is_empty() {
        return Err(ApiError::bad_request("prompt produced no tokens"));
    }

    Ok(PreparedPrompt { tokens })
}

fn normalize_prompt_array(model: &Model, items: &[Value]) -> ApiResult<Vec<u32>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    if items.iter().all(Value::is_number) {
        return items.iter().map(json_value_to_u32).collect();
    }

    if items
        .iter()
        .all(|item| item.is_string() || item.is_number())
    {
        let mut tokens = Vec::new();
        for (index, item) in items.iter().enumerate() {
            match item {
                Value::String(text) => {
                    tokens.extend(model.tokenize_with_options(text, index == 0, true));
                }
                Value::Number(_) => tokens.push(json_value_to_u32(item)?),
                _ => unreachable!("validated primitive mixed prompt"),
            }
        }
        return Ok(tokens);
    }

    Err(ApiError::bad_request(
        "nested prompt arrays must contain only strings and token ids",
    ))
}

fn normalize_prompt_object(
    model: &Model,
    object: &serde_json::Map<String, Value>,
) -> ApiResult<Vec<u32>> {
    if object
        .get("multimodal_data")
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty())
    {
        return Err(ApiError::not_implemented(
            "multimodal prompts are not implemented yet",
        ));
    }

    let prompt_string = object
        .get("prompt_string")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::bad_request("prompt object requires string field 'prompt_string'")
        })?;
    Ok(model.tokenize_with_options(prompt_string, true, true))
}

fn normalize_chat_messages(messages: &[ChatInputMessage]) -> ApiResult<Vec<RenderedMessage>> {
    if messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    messages
        .iter()
        .map(|message| {
            let role = match message.role.trim().to_ascii_lowercase().as_str() {
                "system" | "developer" => CoreChatRole::System,
                "user" => CoreChatRole::User,
                "assistant" => CoreChatRole::Assistant,
                "tool" | "function" => {
                    return Err(ApiError::not_implemented(
                        "tool/function messages are not implemented yet",
                    ));
                }
                other => {
                    return Err(ApiError::bad_request(format!(
                        "unsupported chat role '{other}'"
                    )));
                }
            };

            Ok(RenderedMessage {
                role,
                content: flatten_message_content(message.content.as_ref())?,
            })
        })
        .collect()
}

fn normalize_responses_messages(
    instructions: Option<&Value>,
    input: Option<&Value>,
) -> ApiResult<Vec<RenderedMessage>> {
    let mut messages = Vec::new();

    if let Some(instructions) = instructions {
        let content = normalize_response_content(instructions)?;
        if !content.is_empty() {
            messages.push(RenderedMessage {
                role: CoreChatRole::System,
                content,
            });
        }
    }

    match input {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) => messages.push(RenderedMessage {
            role: CoreChatRole::User,
            content: text.clone(),
        }),
        Some(Value::Array(items))
            if items
                .iter()
                .all(|item| item.is_object() && item.get("role").is_some()) =>
        {
            for item in items {
                messages.push(normalize_response_message(item)?);
            }
        }
        Some(Value::Array(_)) => messages.push(RenderedMessage {
            role: CoreChatRole::User,
            content: normalize_response_content(input.unwrap_or(&Value::Null))?,
        }),
        Some(Value::Object(object)) if object.get("role").is_some() => {
            messages.push(normalize_response_message(input.unwrap_or(&Value::Null))?);
        }
        Some(_) => {
            return Err(ApiError::bad_request("unsupported responses.input shape"));
        }
    }

    Ok(messages)
}

fn normalize_response_message(value: &Value) -> ApiResult<RenderedMessage> {
    let message: ChatInputMessage = serde_json::from_value(value.clone())
        .map_err(|_| ApiError::bad_request("invalid responses input message shape"))?;
    let mut normalized = normalize_chat_messages(&[message])?;
    normalized
        .pop()
        .ok_or_else(|| ApiError::bad_request("responses input message was empty"))
}

fn normalize_response_content(value: &Value) -> ApiResult<String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut combined = String::new();
            for part in parts {
                combined.push_str(&normalize_response_content_part(part)?);
            }
            Ok(combined)
        }
        Value::Object(object) if object.get("type").is_some() => {
            normalize_response_content_part(value)
        }
        _ => Err(ApiError::bad_request("unsupported responses content shape")),
    }
}

fn normalize_response_content_part(value: &Value) -> ApiResult<String> {
    let object = value
        .as_object()
        .ok_or_else(|| ApiError::bad_request("responses content parts must be objects"))?;
    let part_type = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();

    match part_type.as_str() {
        "text" | "input_text" | "output_text" => Ok(object
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()),
        other => Err(ApiError::bad_request(format!(
            "unsupported responses content type '{other}'"
        ))),
    }
}

fn flatten_message_content(content: Option<&MessageContent>) -> ApiResult<String> {
    let Some(content) = content else {
        return Ok(String::new());
    };
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(parts) => {
            let mut flattened = String::new();
            for part in parts {
                match part {
                    MessagePart::Text { text }
                    | MessagePart::InputText { text }
                    | MessagePart::OutputText { text } => {
                        flattened.push_str(text);
                    }
                    MessagePart::Refusal { refusal } => flattened.push_str(refusal),
                    MessagePart::ImageUrl { .. } | MessagePart::InputImage { .. } => {
                        return Err(ApiError::not_implemented(
                            "multimodal chat content is not implemented yet",
                        ));
                    }
                }
            }
            Ok(flattened)
        }
    }
}

fn render_chat_prompt(
    messages: &[RenderedMessage],
    architecture: &str,
    add_generation_prompt: bool,
) -> String {
    if add_generation_prompt
        && let Some(last) = messages.last()
        && last.role == CoreChatRole::Assistant
        && !last.content.is_empty()
    {
        let prefix = messages[..messages.len().saturating_sub(1)]
            .iter()
            .map(|message| CoreChatMessage::new(message.role, message.content.as_str()))
            .collect::<Vec<_>>();
        let mut rendered = render_chat_messages(
            &prefix,
            architecture,
            ChatRenderOptions {
                add_generation_prompt: true,
            },
        );
        rendered.push_str(&last.content);
        return rendered;
    }

    let rendered_messages = messages
        .iter()
        .map(|message| CoreChatMessage::new(message.role, message.content.as_str()))
        .collect::<Vec<_>>();
    render_chat_messages(
        &rendered_messages,
        architecture,
        ChatRenderOptions {
            add_generation_prompt,
        },
    )
}

fn build_generation_options(
    defaults: &ServerDefaults,
    overrides: GenerationOverrides,
) -> GenerationOptions {
    let mut options = GenerationOptions {
        max_tokens: resolve_max_tokens(
            defaults.n_predict,
            overrides.max_tokens,
            overrides.n_predict,
        ),
        temperature: overrides.temperature.unwrap_or(defaults.temperature),
        top_k: overrides.top_k.unwrap_or(defaults.top_k),
        top_p: overrides.top_p.unwrap_or(defaults.top_p),
        min_p: overrides.min_p.unwrap_or(defaults.min_p),
        repeat_penalty: overrides.repeat_penalty.unwrap_or(defaults.repeat_penalty),
        repeat_last_n: overrides.repeat_last_n.unwrap_or(defaults.repeat_last_n),
        frequency_penalty: overrides
            .frequency_penalty
            .unwrap_or(defaults.frequency_penalty),
        presence_penalty: overrides
            .presence_penalty
            .unwrap_or(defaults.presence_penalty),
        stop_strings: overrides.stop.unwrap_or_default(),
        seed: resolve_seed(defaults.seed, overrides.seed),
    };
    options.stop_strings.retain(|stop| !stop.is_empty());
    options
}

fn resolve_max_tokens(
    default_n_predict: i32,
    max_tokens: Option<usize>,
    n_predict: Option<i32>,
) -> usize {
    if let Some(max_tokens) = max_tokens {
        return max_tokens;
    }

    match n_predict.unwrap_or(default_n_predict) {
        value if value < 0 => usize::MAX,
        value => value as usize,
    }
}

fn normalize_seed(seed: Option<i64>) -> Option<u64> {
    seed.filter(|seed| *seed >= 0).map(|seed| seed as u64)
}

fn resolve_seed(default_seed: Option<u64>, request_seed: Option<i64>) -> Option<u64> {
    match request_seed {
        Some(seed) if seed >= 0 => Some(seed as u64),
        Some(_) => None,
        None => default_seed,
    }
}

fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn ensure_slots_enabled(state: &AppState) -> ApiResult<()> {
    if state.no_slots {
        return Err(ApiError::not_supported(
            "This server does not support slots endpoint.",
        ));
    }
    Ok(())
}

fn ensure_slot_id(id_slot: u32) -> ApiResult<()> {
    if id_slot == 0 {
        Ok(())
    } else {
        Err(ApiError::not_found(format!(
            "slot '{id_slot}' was not found"
        )))
    }
}

fn ensure_requested_slot(id_slot: Option<u32>) -> ApiResult<()> {
    if let Some(id_slot) = id_slot {
        ensure_slot_id(id_slot)?;
    }
    Ok(())
}

fn ensure_slot_idle(state: &AppState) -> ApiResult<()> {
    let is_processing = state
        .slot_state
        .lock()
        .map_err(|_| ApiError::internal("slot state lock poisoned"))?
        .is_processing;
    if is_processing || state.inference_slot.available_permits() == 0 {
        return Err(ApiError::service_unavailable("slot is busy"));
    }
    Ok(())
}

fn reset_slot_tracking_state(state: &AppState) -> ApiResult<()> {
    let mut slot_state = state
        .slot_state
        .lock()
        .map_err(|_| ApiError::internal("slot state lock poisoned"))?;
    slot_state.id_task = 0;
    slot_state.is_processing = false;
    slot_state.next_token = SlotNextToken {
        has_next_token: false,
        has_new_line: false,
        n_remain: max_tokens_to_slots_remaining(slot_state.params.max_tokens),
        n_decoded: 0,
    };
    Ok(())
}

fn parse_slot_persistence_request(body: &str) -> ApiResult<SlotPersistenceRequest> {
    if body.trim().is_empty() {
        return Err(ApiError::bad_request(
            "slot persistence request requires JSON body with 'filename'",
        ));
    }
    let request: SlotPersistenceRequest = serde_json::from_str(body)
        .map_err(|_| ApiError::bad_request("invalid slot persistence request body"))?;
    if request.filename.trim().is_empty() {
        return Err(ApiError::bad_request(
            "slot persistence filename must not be empty",
        ));
    }
    Ok(request)
}

fn resolve_slot_snapshot_path(state: &AppState, filename: &str) -> ApiResult<PathBuf> {
    let base_dir = state
        .slot_save_path
        .as_deref()
        .ok_or_else(|| ApiError::not_supported("slot persistence requires --slot-save-path"))?;
    let relative = FsPath::new(filename);
    if relative.is_absolute() {
        return Err(ApiError::bad_request(
            "slot persistence filename must be relative to --slot-save-path",
        ));
    }
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ApiError::bad_request(
            "slot persistence filename must not contain '.', '..', or root components",
        ));
    }
    Ok(base_dir.join(relative))
}

fn validate_slot_snapshot(state: &AppState, snapshot: &SlotSnapshotFile) -> ApiResult<()> {
    if snapshot.format != SLOT_SNAPSHOT_FORMAT {
        return Err(ApiError::bad_request("unsupported slot snapshot format"));
    }
    let info = state.model.info();
    if snapshot.architecture != state.model.architecture()
        || snapshot.vocab_size != info.vocab_size
        || snapshot.bos_token_id != info.bos_token_id
        || snapshot.eos_token_id != info.eos_token_id
    {
        return Err(ApiError::bad_request(
            "slot snapshot is not compatible with the loaded model tokenizer",
        ));
    }
    if snapshot.context_length > state.defaults.n_ctx {
        return Err(ApiError::bad_request(
            "slot snapshot exceeds the configured context length",
        ));
    }
    if snapshot.model_id != state.model.id() {
        tracing::warn!(
            snapshot_model_id = %snapshot.model_id,
            loaded_model_id = %state.model.id(),
            "restoring slot snapshot from a different model id with compatible tokenizer metadata"
        );
    }
    Ok(())
}

fn use_slot_prompt_cache(
    id_slot: Option<u32>,
    cache_prompt: bool,
    prompt_count: usize,
) -> ApiResult<bool> {
    ensure_requested_slot(id_slot)?;
    Ok(cache_prompt && prompt_count == 1)
}

fn cloned_slot_snapshot(state: &AppState) -> ApiResult<Option<SessionSnapshot>> {
    Ok(state
        .slot_cache
        .lock()
        .map_err(|_| ApiError::internal("slot cache lock poisoned"))?
        .clone())
}

fn replace_slot_snapshot(state: &AppState, snapshot: Option<SessionSnapshot>) -> ApiResult<()> {
    *state
        .slot_cache
        .lock()
        .map_err(|_| ApiError::internal("slot cache lock poisoned"))? = snapshot;
    Ok(())
}

fn build_slot_snapshot_from_tokens(
    state: &AppState,
    tokens: &[u32],
) -> ApiResult<Option<SessionSnapshot>> {
    if tokens.is_empty() {
        return Ok(None);
    }
    let session = state
        .model
        .session(SessionOptions::default())
        .map_err(ApiError::from_anyhow)?;
    session
        .load_prompt_tokens(tokens)
        .map_err(ApiError::from_anyhow)?;
    session.snapshot().map(Some).map_err(ApiError::from_anyhow)
}

async fn acquire_inference_slot(state: &Arc<AppState>) -> ApiResult<OwnedSemaphorePermit> {
    state
        .inference_slot
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("inference slot semaphore closed"))
}

fn begin_request_tracking(state: &AppState, options: &GenerationOptions, stream: bool) -> u64 {
    let task_id = state.next_task_id.fetch_add(1, Ordering::Relaxed);
    state
        .metrics
        .requests_processing
        .fetch_add(1, Ordering::Relaxed);

    if let Ok(mut slot_state) = state.slot_state.lock() {
        slot_state.id_task = task_id;
        slot_state.is_processing = true;
        slot_state.params = slot_params_from_generation_options(options, stream);
        slot_state.next_token = SlotNextToken {
            has_next_token: true,
            has_new_line: false,
            n_remain: max_tokens_to_slots_remaining(options.max_tokens),
            n_decoded: 0,
        };
    }

    task_id
}

fn record_stream_chunk(state: &AppState, task_id: u64, chunk: &str) {
    if let Ok(mut slot_state) = state.slot_state.lock()
        && slot_state.id_task == task_id
    {
        slot_state.next_token.has_new_line |= chunk.contains('\n');
    }
}

fn finish_request_tracking(
    state: &AppState,
    task_id: u64,
    options: &GenerationOptions,
    output: &ax_engine_sdk::GenerationOutput,
    elapsed_ms: f64,
) {
    state
        .metrics
        .requests_processing
        .fetch_sub(1, Ordering::Relaxed);
    state
        .metrics
        .prompt_tokens_total
        .fetch_add(output.usage.prompt_tokens as u64, Ordering::Relaxed);
    state
        .metrics
        .predicted_tokens_total
        .fetch_add(output.usage.completion_tokens as u64, Ordering::Relaxed);
    state.metrics.predicted_micros_total.fetch_add(
        (elapsed_ms.max(0.0) * 1_000.0).round() as u64,
        Ordering::Relaxed,
    );
    let total_tokens = output.usage.total_tokens as u64;
    let _ =
        state
            .metrics
            .n_tokens_max
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (total_tokens > current).then_some(total_tokens)
            });

    if let Ok(mut slot_state) = state.slot_state.lock()
        && slot_state.id_task == task_id
    {
        let n_remain = if options.max_tokens == usize::MAX {
            -1
        } else {
            options
                .max_tokens
                .saturating_sub(output.usage.completion_tokens) as i64
        };
        slot_state.is_processing = false;
        slot_state.next_token.has_next_token = false;
        slot_state.next_token.n_decoded = output.usage.completion_tokens;
        slot_state.next_token.n_remain = n_remain;
    }
}

fn fail_request_tracking(state: &AppState, task_id: u64) {
    state
        .metrics
        .requests_processing
        .fetch_sub(1, Ordering::Relaxed);
    if let Ok(mut slot_state) = state.slot_state.lock()
        && slot_state.id_task == task_id
    {
        slot_state.is_processing = false;
        slot_state.next_token.has_next_token = false;
    }
}

fn slot_params_from_generation_options(options: &GenerationOptions, stream: bool) -> SlotParams {
    SlotParams {
        max_tokens: options.max_tokens,
        seed: options.seed,
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        min_p: options.min_p,
        repeat_penalty: options.repeat_penalty,
        repeat_last_n: options.repeat_last_n,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        stop: options.stop_strings.clone(),
        stream,
    }
}

fn max_tokens_to_slots_remaining(max_tokens: usize) -> i64 {
    if max_tokens == usize::MAX {
        -1
    } else {
        i64::try_from(max_tokens).unwrap_or(i64::MAX)
    }
}

fn slot_state_json(state: &AppState, slot_state: &SlotState) -> Value {
    json!({
        "id": 0,
        "id_task": slot_state.id_task,
        "n_ctx": state.defaults.n_ctx,
        "speculative": false,
        "is_processing": slot_state.is_processing,
        "params": {
            "n_predict": if slot_state.params.max_tokens == usize::MAX { -1 } else { slot_state.params.max_tokens as i64 },
            "seed": slot_state.params.seed.unwrap_or(u64::MAX),
            "temperature": slot_state.params.temperature,
            "top_k": slot_state.params.top_k,
            "top_p": slot_state.params.top_p,
            "min_p": slot_state.params.min_p,
            "repeat_last_n": slot_state.params.repeat_last_n,
            "repeat_penalty": slot_state.params.repeat_penalty,
            "presence_penalty": slot_state.params.presence_penalty,
            "frequency_penalty": slot_state.params.frequency_penalty,
            "max_tokens": if slot_state.params.max_tokens == usize::MAX { -1 } else { slot_state.params.max_tokens as i64 },
            "stream": slot_state.params.stream,
            "stop": slot_state.params.stop,
        },
        "next_token": {
            "has_next_token": slot_state.next_token.has_next_token,
            "has_new_line": slot_state.next_token.has_new_line,
            "n_remain": slot_state.next_token.n_remain,
            "n_decoded": slot_state.next_token.n_decoded,
        }
    })
}

fn response_object_json(
    id: &str,
    created_at: u64,
    model: &str,
    status: &str,
    text: &str,
    usage: Option<&ax_engine_sdk::Usage>,
    finish_reason: Option<FinishReason>,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": [{
            "id": format!("{id}:output:0"),
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": []
            }]
        }],
        "output_text": text,
        "usage": usage.map(|usage| json!({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        })),
        "finish_reason": finish_reason.map(finish_reason_name),
    })
}

async fn run_generation(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    options: GenerationOptions,
    stream: bool,
) -> ApiResult<RunResult> {
    let permit = acquire_inference_slot(&state).await?;
    let task_id = begin_request_tracking(state.as_ref(), &options, stream);
    let tracked_state = state.clone();
    let tracked_options = options.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let session = state
            .model
            .session(SessionOptions::default())
            .context("failed to create generation session")?;
        let started = Instant::now();
        let output = session
            .generate_tokens(&prompt_tokens, options)
            .context("generation failed")?;
        Ok::<_, anyhow::Error>(RunResult {
            output,
            elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
            cache_tokens: 0,
        })
    })
    .await
    .map_err(|error| ApiError::internal(format!("generation worker failed: {error}")))?;

    match result {
        Ok(run) => {
            finish_request_tracking(
                tracked_state.as_ref(),
                task_id,
                &tracked_options,
                &run.output,
                run.elapsed_ms,
            );
            Ok(run)
        }
        Err(error) => {
            fail_request_tracking(tracked_state.as_ref(), task_id);
            Err(ApiError::from_anyhow(error))
        }
    }
}

async fn run_cached_generation(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    options: GenerationOptions,
    stream: bool,
) -> ApiResult<RunResult> {
    let permit = acquire_inference_slot(&state).await?;
    let task_id = begin_request_tracking(state.as_ref(), &options, stream);
    let tracked_state = state.clone();
    let tracked_options = options.clone();
    let cached_snapshot = cloned_slot_snapshot(state.as_ref())?;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let session = state
            .model
            .session(SessionOptions::default())
            .context("failed to create cached generation session")?;
        if let Some(snapshot) = cached_snapshot.as_ref() {
            session
                .restore_snapshot(snapshot)
                .context("failed to restore cached session snapshot")?;
        }
        let started = Instant::now();
        let (output, cache_stats) = session
            .generate_with_prefix_reuse(&prompt_tokens, options)
            .context("cached generation failed")?;
        let next_snapshot = session
            .snapshot()
            .context("failed to capture cached session snapshot")?;
        Ok::<_, anyhow::Error>((
            RunResult {
                output,
                elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
                cache_tokens: cache_stats.cached_tokens,
            },
            next_snapshot,
        ))
    })
    .await
    .map_err(|error| ApiError::internal(format!("generation worker failed: {error}")))?;

    match result {
        Ok((run, next_snapshot)) => {
            replace_slot_snapshot(tracked_state.as_ref(), Some(next_snapshot))?;
            finish_request_tracking(
                tracked_state.as_ref(),
                task_id,
                &tracked_options,
                &run.output,
                run.elapsed_ms,
            );
            Ok(run)
        }
        Err(error) => {
            fail_request_tracking(tracked_state.as_ref(), task_id);
            Err(ApiError::from_anyhow(error))
        }
    }
}

fn build_stream_for_prompt(
    state: &AppState,
    prompt_tokens: &[u32],
    options: GenerationOptions,
    use_slot_cache: bool,
) -> anyhow::Result<(Session, ax_engine_sdk::TextStream, PromptCacheStats)> {
    if use_slot_cache {
        let session = state
            .model
            .session(SessionOptions::default())
            .context("failed to create cached generation session")?;
        if let Some(snapshot) =
            cloned_slot_snapshot(state).map_err(|error| anyhow::anyhow!(error.message))?
        {
            session
                .restore_snapshot(&snapshot)
                .context("failed to restore cached session snapshot")?;
        }
        let (stream, cache_stats) = session
            .stream_with_prefix_reuse(prompt_tokens, options)
            .context("failed to start cached stream")?;
        Ok((session, stream, cache_stats))
    } else {
        let session = state
            .model
            .session(SessionOptions::default())
            .context("failed to create generation session")?;
        let stream = session
            .stream_tokens(prompt_tokens, options)
            .context("failed to start stream")?;
        Ok((session, stream, PromptCacheStats::default()))
    }
}

fn legacy_completion_stream(
    state: Arc<AppState>,
    prompt: PreparedPrompt,
    options: GenerationOptions,
    permit: OwnedSemaphorePermit,
    use_slot_cache: bool,
) -> Response {
    let task_id = begin_request_tracking(state.as_ref(), &options, true);
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::task::spawn_blocking(move || {
        let result: anyhow::Result<()> = (|| {
            let _permit = permit;
            let started = Instant::now();
            let (session, mut stream, _cache_stats) = build_stream_for_prompt(
                state.as_ref(),
                &prompt.tokens,
                options.clone(),
                use_slot_cache,
            )
            .context("failed to start stream")?;

            while let Some(chunk) = stream.next_chunk()? {
                record_stream_chunk(state.as_ref(), task_id, &chunk);
                if tx
                    .blocking_send(Ok(json_event(&json!({
                        "content": chunk,
                        "tokens": [],
                        "stop": false
                    }))?))
                    .is_err()
                {
                    return Ok(());
                }
            }

            let output = stream
                .output()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("stream ended without output"))?;
            if use_slot_cache {
                let snapshot = session
                    .snapshot()
                    .context("failed to capture cached session snapshot")?;
                replace_slot_snapshot(state.as_ref(), Some(snapshot))
                    .map_err(|error| anyhow::anyhow!(error.message))?;
            }
            let final_event = json!({
                "content": "",
                "tokens": [],
                "stop": true,
                "stop_type": legacy_stop_type(output.finish_reason, !options.stop_strings.is_empty()),
                "stopping_word": "",
            });
            finish_request_tracking(
                state.as_ref(),
                task_id,
                &options,
                &output,
                started.elapsed().as_secs_f64() * 1_000.0,
            );
            let _ = tx.blocking_send(Ok(json_event(&final_event)?));
            Ok(())
        })();

        if let Err(error) = result {
            fail_request_tracking(state.as_ref(), task_id);
            let _ = tx.blocking_send(Ok(json_event(&json!({
                "error": {
                    "code": 500,
                    "message": error.to_string(),
                    "type": "internal_error"
                }
            })).unwrap_or_else(|_| Event::default().data("{\"error\":{\"code\":500,\"message\":\"internal error\",\"type\":\"internal_error\"}}"))));
        }
    });

    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn openai_completion_stream(
    state: Arc<AppState>,
    prompt: PreparedPrompt,
    options: GenerationOptions,
    permit: OwnedSemaphorePermit,
    use_slot_cache: bool,
) -> Response {
    let request_id = format!("cmpl-{}", Uuid::new_v4().simple());
    let created = unix_timestamp();
    let task_id = begin_request_tracking(state.as_ref(), &options, true);
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::task::spawn_blocking(move || {
        let result: anyhow::Result<()> = (|| {
            let _permit = permit;
            let started = Instant::now();
            let (session, mut stream, _cache_stats) = build_stream_for_prompt(
                state.as_ref(),
                &prompt.tokens,
                options.clone(),
                use_slot_cache,
            )
            .context("failed to start stream")?;

            while let Some(chunk) = stream.next_chunk()? {
                record_stream_chunk(state.as_ref(), task_id, &chunk);
                let chunk_event = json!({
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": state.model_alias,
                    "choices": [{
                        "text": chunk,
                        "index": 0,
                        "logprobs": Value::Null,
                        "finish_reason": Value::Null
                    }]
                });
                if tx.blocking_send(Ok(json_event(&chunk_event)?)).is_err() {
                    return Ok(());
                }
            }

            let output = stream
                .output()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("stream ended without output"))?;
            if use_slot_cache {
                let snapshot = session
                    .snapshot()
                    .context("failed to capture cached session snapshot")?;
                replace_slot_snapshot(state.as_ref(), Some(snapshot))
                    .map_err(|error| anyhow::anyhow!(error.message))?;
            }
            let final_event = json!({
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": state.model_alias,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": Value::Null,
                    "finish_reason": finish_reason_name(output.finish_reason)
                }]
            });
            finish_request_tracking(
                state.as_ref(),
                task_id,
                &options,
                &output,
                started.elapsed().as_secs_f64() * 1_000.0,
            );
            let _ = tx.blocking_send(Ok(json_event(&final_event)?));
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            Ok(())
        })();

        if let Err(error) = result {
            fail_request_tracking(state.as_ref(), task_id);
            let _ = tx.blocking_send(Ok(json_event(&json!({
                "error": {
                    "code": 500,
                    "message": error.to_string(),
                    "type": "internal_error"
                }
            })).unwrap_or_else(|_| Event::default().data("{\"error\":{\"code\":500,\"message\":\"internal error\",\"type\":\"internal_error\"}}"))));
        }
    });

    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn openai_chat_stream(
    state: Arc<AppState>,
    prompt: PreparedPrompt,
    options: GenerationOptions,
    permit: OwnedSemaphorePermit,
    use_slot_cache: bool,
) -> Response {
    let request_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = unix_timestamp();
    let task_id = begin_request_tracking(state.as_ref(), &options, true);
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::task::spawn_blocking(move || {
        let result: anyhow::Result<()> = (|| {
            let _permit = permit;
            let started = Instant::now();
            let (session, mut stream, _cache_stats) = build_stream_for_prompt(
                state.as_ref(),
                &prompt.tokens,
                options.clone(),
                use_slot_cache,
            )
            .context("failed to start stream")?;
            let mut first_chunk = true;

            while let Some(chunk) = stream.next_chunk()? {
                record_stream_chunk(state.as_ref(), task_id, &chunk);
                let delta = if first_chunk {
                    first_chunk = false;
                    json!({
                        "role": "assistant",
                        "content": chunk
                    })
                } else {
                    json!({
                        "content": chunk
                    })
                };

                let event = json!({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.model_alias,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": Value::Null
                    }]
                });
                if tx.blocking_send(Ok(json_event(&event)?)).is_err() {
                    return Ok(());
                }
            }

            let output = stream
                .output()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("stream ended without output"))?;
            if use_slot_cache {
                let snapshot = session
                    .snapshot()
                    .context("failed to capture cached session snapshot")?;
                replace_slot_snapshot(state.as_ref(), Some(snapshot))
                    .map_err(|error| anyhow::anyhow!(error.message))?;
            }
            let final_event = json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": state.model_alias,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason_name(output.finish_reason)
                }]
            });
            finish_request_tracking(
                state.as_ref(),
                task_id,
                &options,
                &output,
                started.elapsed().as_secs_f64() * 1_000.0,
            );
            let _ = tx.blocking_send(Ok(json_event(&final_event)?));
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            Ok(())
        })();

        if let Err(error) = result {
            fail_request_tracking(state.as_ref(), task_id);
            let _ = tx.blocking_send(Ok(json_event(&json!({
                "error": {
                    "code": 500,
                    "message": error.to_string(),
                    "type": "internal_error"
                }
            })).unwrap_or_else(|_| Event::default().data("{\"error\":{\"code\":500,\"message\":\"internal error\",\"type\":\"internal_error\"}}"))));
        }
    });

    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn openai_responses_stream(
    state: Arc<AppState>,
    prompt: PreparedPrompt,
    options: GenerationOptions,
    permit: OwnedSemaphorePermit,
    use_slot_cache: bool,
) -> Response {
    let response_id = format!("resp_{}", Uuid::new_v4().simple());
    let created = unix_timestamp();
    let task_id = begin_request_tracking(state.as_ref(), &options, true);
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::task::spawn_blocking(move || {
        let result: anyhow::Result<()> = (|| {
            let _permit = permit;
            let started = Instant::now();
            let (session, mut stream, _cache_stats) = build_stream_for_prompt(
                state.as_ref(),
                &prompt.tokens,
                options.clone(),
                use_slot_cache,
            )
            .context("failed to start responses stream")?;
            let created_event = json!({
                "type": "response.created",
                "response": response_object_json(
                    &response_id,
                    created,
                    &state.model_alias,
                    "in_progress",
                    "",
                    None,
                    None,
                )
            });
            if tx.blocking_send(Ok(json_event(&created_event)?)).is_err() {
                return Ok(());
            }

            let mut text = String::new();
            while let Some(chunk) = stream.next_chunk()? {
                record_stream_chunk(state.as_ref(), task_id, &chunk);
                text.push_str(&chunk);
                let delta_event = json!({
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": chunk
                });
                if tx.blocking_send(Ok(json_event(&delta_event)?)).is_err() {
                    return Ok(());
                }
            }

            let output = stream
                .output()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("stream ended without output"))?;
            if use_slot_cache {
                let snapshot = session
                    .snapshot()
                    .context("failed to capture cached session snapshot")?;
                replace_slot_snapshot(state.as_ref(), Some(snapshot))
                    .map_err(|error| anyhow::anyhow!(error.message))?;
            }
            finish_request_tracking(
                state.as_ref(),
                task_id,
                &options,
                &output,
                started.elapsed().as_secs_f64() * 1_000.0,
            );
            let done_event = json!({
                "type": "response.output_text.done",
                "response_id": response_id,
                "output_index": 0,
                "content_index": 0,
                "text": text
            });
            let completed_event = json!({
                "type": "response.completed",
                "response": response_object_json(
                    &response_id,
                    created,
                    &state.model_alias,
                    "completed",
                    &text,
                    Some(&output.usage),
                    Some(output.finish_reason),
                )
            });
            let _ = tx.blocking_send(Ok(json_event(&done_event)?));
            let _ = tx.blocking_send(Ok(json_event(&completed_event)?));
            Ok(())
        })();

        if let Err(error) = result {
            fail_request_tracking(state.as_ref(), task_id);
            let _ = tx.blocking_send(Ok(json_event(&json!({
                "error": {
                    "code": 500,
                    "message": error.to_string(),
                    "type": "internal_error"
                }
            })).unwrap_or_else(|_| Event::default().data("{\"error\":{\"code\":500,\"message\":\"internal error\",\"type\":\"internal_error\"}}"))));
        }
    });

    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn legacy_completion_body(
    state: &AppState,
    prompt_tokens: &[u32],
    run: &RunResult,
    options: &GenerationOptions,
    return_tokens: bool,
) -> Value {
    json!({
        "content": run.output.text,
        "tokens": if return_tokens { json!([]) } else { json!([]) },
        "stop": true,
        "generation_settings": generation_settings_json(state, options),
        "model": state.model_alias,
        "prompt": prompt_tokens,
        "stop_type": legacy_stop_type(run.output.finish_reason, !options.stop_strings.is_empty()),
        "stopping_word": "",
        "timings": timings_json(run),
        "tokens_cached": run.cache_tokens,
        "tokens_evaluated": run.output.usage.prompt_tokens,
        "truncated": false
    })
}

fn generation_settings_json(state: &AppState, options: &GenerationOptions) -> Value {
    json!({
        "n_ctx": state.defaults.n_ctx,
        "n_predict": options.max_tokens,
        "temperature": options.temperature,
        "top_k": options.top_k,
        "top_p": options.top_p,
        "min_p": options.min_p,
        "repeat_penalty": options.repeat_penalty,
        "repeat_last_n": options.repeat_last_n,
        "frequency_penalty": options.frequency_penalty,
        "presence_penalty": options.presence_penalty,
        "stop": options.stop_strings,
        "seed": options.seed.unwrap_or(u64::MAX),
        "model": state.model_alias
    })
}

fn timings_json(run: &RunResult) -> Value {
    let prompt_n = run.output.usage.prompt_tokens as f64;
    let predicted_n = run.output.usage.completion_tokens as f64;
    let predicted_ms = run.elapsed_ms;
    let predicted_per_token_ms = if predicted_n > 0.0 {
        predicted_ms / predicted_n
    } else {
        0.0
    };
    let predicted_per_second = if predicted_ms > 0.0 {
        predicted_n * 1000.0 / predicted_ms
    } else {
        0.0
    };
    json!({
        "cache_n": run.cache_tokens,
        "prompt_n": prompt_n as usize,
        "prompt_ms": 0.0,
        "prompt_per_token_ms": 0.0,
        "prompt_per_second": 0.0,
        "predicted_n": predicted_n as usize,
        "predicted_ms": predicted_ms,
        "predicted_per_token_ms": predicted_per_token_ms,
        "predicted_per_second": predicted_per_second
    })
}

fn finish_reason_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn legacy_stop_type(reason: FinishReason, has_stop_strings: bool) -> &'static str {
    match reason {
        FinishReason::Length => "limit",
        FinishReason::Stop if has_stop_strings => "word",
        FinishReason::Stop => "eos",
    }
}

fn token_piece_json(piece: Option<TokenPiece>) -> Value {
    match piece {
        Some(TokenPiece::Text(text)) => json!(text),
        Some(TokenPiece::Bytes(bytes)) => json!(bytes),
        None => Value::Null,
    }
}

fn json_value_to_u32(value: &Value) -> ApiResult<u32> {
    let raw = value
        .as_u64()
        .ok_or_else(|| ApiError::bad_request("token ids must be unsigned integers"))?;
    u32::try_from(raw).map_err(|_| ApiError::bad_request("token id exceeds u32 range"))
}

fn json_event(value: &Value) -> anyhow::Result<Event> {
    Ok(Event::default().data(serde_json::to_string(value)?))
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn default_true() -> bool {
    true
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

impl StopInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            kind: "authentication_error",
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "not_found_error",
            message: message.into(),
        }
    }

    fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            kind: "not_implemented_error",
            message: message.into(),
        }
    }

    fn not_supported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            kind: "not_supported_error",
            message: message.into(),
        }
    }

    fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "server_error",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "internal_error",
            message: message.into(),
        }
    }

    fn from_anyhow(error: anyhow::Error) -> Self {
        Self::internal(error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "code": self.status.as_u16(),
                    "message": self.message,
                    "type": self.kind
                }
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_message_content_text_parts() {
        let content = MessageContent::Parts(vec![
            MessagePart::Text {
                text: "Hello".to_string(),
            },
            MessagePart::InputText {
                text: " world".to_string(),
            },
        ]);
        assert_eq!(
            flatten_message_content(Some(&content)).unwrap(),
            "Hello world"
        );
    }

    #[test]
    fn test_resolve_max_tokens_prefers_explicit_max_tokens() {
        assert_eq!(resolve_max_tokens(-1, Some(123), Some(456)), 123);
    }

    #[test]
    fn test_normalize_seed_ignores_negative_values() {
        assert_eq!(normalize_seed(Some(-1)), None);
        assert_eq!(normalize_seed(Some(42)), Some(42));
    }

    #[test]
    fn test_resolve_seed_disables_default_on_negative_request() {
        assert_eq!(resolve_seed(Some(7), Some(-1)), None);
        assert_eq!(resolve_seed(Some(7), Some(9)), Some(9));
        assert_eq!(resolve_seed(Some(7), None), Some(7));
    }

    #[test]
    fn test_normalize_response_content_concatenates_parts() {
        let value = json!([
            { "type": "input_text", "text": "Hello" },
            { "type": "output_text", "text": " world" }
        ]);
        assert_eq!(normalize_response_content(&value).unwrap(), "Hello world");
    }

    #[test]
    fn test_normalize_responses_messages_maps_instructions_and_input() {
        let messages = normalize_responses_messages(
            Some(&json!("Answer briefly.")),
            Some(&json!("Say hello.")),
        )
        .unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, CoreChatRole::System);
        assert_eq!(messages[0].content, "Answer briefly.");
        assert_eq!(messages[1].role, CoreChatRole::User);
        assert_eq!(messages[1].content, "Say hello.");
    }

    #[test]
    fn test_response_object_json_contains_output_text() {
        let usage = ax_engine_sdk::Usage {
            prompt_tokens: 3,
            completion_tokens: 4,
            total_tokens: 7,
        };
        let value = response_object_json(
            "resp_test",
            1,
            "test-model",
            "completed",
            "Hello",
            Some(&usage),
            Some(FinishReason::Stop),
        );
        assert_eq!(value["object"], "response");
        assert_eq!(value["output_text"], "Hello");
        assert_eq!(value["usage"]["total_tokens"], 7);
    }

    #[test]
    fn test_legacy_stop_type_maps_length() {
        assert_eq!(legacy_stop_type(FinishReason::Length, false), "limit");
    }

    #[test]
    fn test_parse_slot_persistence_request_rejects_empty_filename() {
        let err = parse_slot_persistence_request(r#"{"filename":""}"#)
            .unwrap_err()
            .message;
        assert!(err.contains("must not be empty"));
    }

    #[test]
    fn test_use_slot_prompt_cache_defaults_to_single_prompt_only() {
        assert!(use_slot_prompt_cache(None, true, 1).unwrap());
        assert!(!use_slot_prompt_cache(None, false, 1).unwrap());
        assert!(!use_slot_prompt_cache(None, true, 2).unwrap());
    }

    #[test]
    fn test_use_slot_prompt_cache_rejects_unknown_slot() {
        let err = use_slot_prompt_cache(Some(1), true, 1).unwrap_err();
        assert_eq!(err.status, StatusCode::NOT_FOUND);
    }
}
