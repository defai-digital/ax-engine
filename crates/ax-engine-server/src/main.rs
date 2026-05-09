#![allow(clippy::collapsible_if)]

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use ax_engine_sdk::{
    EmbeddingPooling, EngineSession, EngineSessionConfig, EngineSessionError, EngineStepReport,
    GenerateFinishReason, GenerateRequest, GenerateResponse, GenerateSampling, GenerateStreamEvent,
    GenerateStreamState, LlamaCppBackendError, MlxLmBackendError, RuntimeReport, SelectedBackend,
    SessionRequestReport, StatelessGenerateContext,
};
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::time::{Instant, sleep_until};
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;
use tracing_subscriber::EnvFilter;

mod args;

use args::{ServerArgs, render_presets};

const MAX_REQUEST_BODY_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_EMBEDDING_MICROBATCH_WINDOW_MS: u64 = 2;
const DEFAULT_EMBEDDING_MICROBATCH_MAX_BATCH: usize = 32;
const STREAM_CHANNEL_CAPACITY: usize = 128;

type StreamEvent = Result<Event, Infallible>;
type StreamEventSender = mpsc::Sender<StreamEvent>;
type StreamEventReceiver = mpsc::Receiver<StreamEvent>;

#[derive(Clone)]
struct AppState {
    model_id: Arc<String>,
    session_config: Arc<EngineSessionConfig>,
    stateless_generate_context: Arc<StatelessGenerateContext>,
    runtime_report: RuntimeReport,
    request_session: Arc<Mutex<EngineSession>>,
    embedding_batcher: Arc<EmbeddingMicroBatcher>,
    next_request_id: Arc<AtomicU64>,
}

#[derive(Clone)]
struct EmbeddingMicroBatcher {
    sender: mpsc::UnboundedSender<EmbeddingBatchItem>,
}

struct EmbeddingBatchItem {
    input: Vec<u32>,
    pooling: EmbeddingPooling,
    normalize: bool,
    response_tx: oneshot::Sender<Result<Vec<f32>, EngineSessionError>>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct EmbeddingBatchKey {
    pooling_code: u8,
    normalize: bool,
}

#[derive(Clone, Copy, Debug)]
struct EmbeddingBatchRequestOptions {
    pooling: EmbeddingPooling,
    normalize: bool,
}

#[derive(Clone)]
struct EmbeddingBatchRunItem {
    input: Vec<u32>,
    pooling: EmbeddingPooling,
    normalize: bool,
}

#[derive(Debug, Deserialize)]
struct GenerateHttpRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    input_tokens: Vec<u32>,
    #[serde(default)]
    input_text: Option<String>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    sampling: Option<GenerateSampling>,
    #[serde(default)]
    metadata: Option<String>,
}

#[derive(Debug, Serialize)]
struct ServerInfoResponse {
    service: &'static str,
    model_id: String,
    deterministic: bool,
    max_batch_tokens: u32,
    block_size_tokens: u32,
    runtime: RuntimeResponse,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    runtime: RuntimeResponse,
}

type RuntimeResponse = RuntimeReport;

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: Option<String>,
    param: Option<String>,
    message: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingRequest {
    #[serde(default)]
    model: Option<String>,
    /// Pre-tokenized input — a single sequence of token IDs.
    input: Vec<u32>,
    /// Ignored — always returns float32. Present for OpenAI API compatibility.
    #[serde(default)]
    #[allow(dead_code)]
    encoding_format: Option<String>,
    /// Pooling strategy: "last" (default), "mean", or "cls".
    ///
    /// "last" takes the final token's hidden state, which is the standard for
    /// decoder-only embedding models (Qwen3-Embedding, Gemma Embedding, etc.).
    /// The caller is responsible for appending an EOS token to the input when
    /// the model expects it.  "mean" averages all positions; "cls" takes the
    /// first token.
    #[serde(default)]
    pooling: Option<String>,
    /// Whether to L2-normalize the output vector to unit length (default true).
    ///
    /// Normalized embeddings allow cosine similarity to be computed as a simple
    /// dot product.  All major embedding APIs return normalized vectors.
    #[serde(default)]
    normalize: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OpenAiEmbeddingResponse {
    object: &'static str,
    data: Vec<OpenAiEmbeddingObject>,
    model: String,
    usage: OpenAiEmbeddingUsage,
}

#[derive(Debug, Serialize)]
struct OpenAiEmbeddingObject {
    object: &'static str,
    embedding: Vec<f32>,
    index: u32,
}

#[derive(Debug, Serialize)]
struct OpenAiEmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAiCompletionHttpRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: OpenAiPromptInput,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    repetition_context_size: Option<u32>,
    #[serde(default)]
    stop: Option<OpenAiStopInput>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    metadata: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatCompletionHttpRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<OpenAiChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    repetition_context_size: Option<u32>,
    #[serde(default)]
    stop: Option<OpenAiStopInput>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    metadata: Option<String>,
}

/// OpenAI `stop` field: either a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiStopInput {
    Single(String),
    Multiple(Vec<String>),
}

impl OpenAiStopInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            OpenAiStopInput::Single(s) => vec![s],
            OpenAiStopInput::Multiple(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiPromptInput {
    Text(String),
    TextBatch(Vec<String>),
    Tokens(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct OpenAiChatMessage {
    role: String,
    content: OpenAiChatContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiChatContent {
    Text(String),
    Parts(Vec<OpenAiChatContentPart>),
}

#[derive(Debug, Deserialize)]
struct OpenAiChatContentPart {
    #[serde(rename = "type")]
    part_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiCompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionChoice {
    index: u32,
    text: String,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChatCompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionChoice {
    index: u32,
    message: OpenAiChatMessageResponse,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct OpenAiChatMessageResponse {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionChunkChoice {
    index: u32,
    text: String,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChatCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionChunkChoice {
    index: u32,
    delta: OpenAiChatDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
struct OpenAiChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Clone, Copy, Debug)]
enum OpenAiStreamKind {
    Completion,
    ChatCompletion,
}

impl OpenAiStreamKind {
    fn response_id(self, request_id: u64) -> String {
        match self {
            OpenAiStreamKind::Completion => format!("cmpl-{request_id}"),
            OpenAiStreamKind::ChatCompletion => format!("chatcmpl-{request_id}"),
        }
    }

    fn stream_chunk_object(self) -> &'static str {
        match self {
            OpenAiStreamKind::Completion => "text_completion.chunk",
            OpenAiStreamKind::ChatCompletion => "chat.completion.chunk",
        }
    }

    fn build_non_stream_response(
        self,
        response: &GenerateResponse,
        request_id: u64,
    ) -> axum::response::Response {
        let id = self.response_id(request_id);
        match self {
            OpenAiStreamKind::Completion => {
                Json(openai_completion_response(response, id)).into_response()
            }
            OpenAiStreamKind::ChatCompletion => {
                Json(openai_chat_completion_response(response, id)).into_response()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChatPromptTemplate {
    QwenChatMl,
    Llama3,
    PlainRolePrefix,
}

impl ChatPromptTemplate {
    fn for_model_id(model_id: &str) -> Self {
        let normalized = model_id.to_ascii_lowercase();
        if normalized.contains("qwen") {
            Self::QwenChatMl
        } else if normalized.contains("llama-3")
            || normalized.contains("llama3")
            || normalized.contains("llama_3")
        {
            Self::Llama3
        } else {
            Self::PlainRolePrefix
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tracing_enabled = init_tracing();

    let args = ServerArgs::parse();
    if args.list_presets {
        println!("{}", render_presets());
        return Ok(());
    }
    let bind_address = args.bind_address();
    let model_id = args.effective_model_id().to_string();
    let support_tier = args.effective_support_tier();
    let session_config = args
        .session_config()
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    let session = EngineSession::new(session_config.clone())?;
    let state = build_app_state(model_id.clone(), session)?;
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;

    if tracing_enabled {
        info!(
            bind_address = %bind_address,
            model_id = %model_id,
            support_tier = ?support_tier,
            "ax-engine-server preview listening"
        );
    } else {
        eprintln!(
            "ax-engine-server preview listening on http://{} model_id={} support_tier={:?}",
            bind_address, model_id, support_tier
        );
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

fn build_app_state(
    model_id: String,
    session: EngineSession,
) -> Result<AppState, EngineSessionError> {
    let session_config = session.config().clone();
    let stateless_generate_context =
        StatelessGenerateContext::new(session_config.clone()).map(Arc::new)?;
    let runtime_report = session.runtime_report();
    let request_session = Arc::new(Mutex::new(session));
    let embedding_batcher = EmbeddingMicroBatcher::spawn(request_session.clone());

    Ok(AppState {
        model_id: Arc::new(model_id),
        session_config: Arc::new(session_config),
        stateless_generate_context,
        runtime_report,
        request_session,
        embedding_batcher,
        next_request_id: Arc::new(AtomicU64::new(1)),
    })
}

fn init_tracing() -> bool {
    let filter = env::var("AX_ENGINE_SERVER_LOG")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("RUST_LOG")
                .ok()
                .filter(|value| !value.trim().is_empty())
        });
    let Some(filter) = filter else {
        return false;
    };
    let Ok(env_filter) = EnvFilter::try_new(filter) else {
        return false;
    };

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_ansi(false)
        .compact()
        .try_init()
        .is_ok()
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/v1/runtime", get(runtime_info))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(openai_embeddings))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/step", post(step_request))
        .route("/v1/requests", post(submit_request))
        .route("/v1/requests/:request_id", get(request_snapshot))
        .route("/v1/requests/:request_id/cancel", post(cancel_request))
        .route("/v1/generate/stream", post(generate_stream))
        .route("/v1/generate", post(generate))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "service": "ax-engine-server",
        "model_id": state.model_id.as_ref(),
        "runtime": runtime_response(&state),
    }))
}

async fn runtime_info(State(state): State<AppState>) -> Json<ServerInfoResponse> {
    Json(server_info_response(&state))
}

async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
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

async fn generate(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<Json<ax_engine_sdk::GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_model(&state, request.model.as_deref())?;

    let request = build_generate_request(&state, request);
    let (_, response) = run_stateless_generate_request(&state, request).await?;

    Ok(Json(response))
}

async fn openai_embeddings(
    State(state): State<AppState>,
    Json(request): Json<OpenAiEmbeddingRequest>,
) -> Result<Json<OpenAiEmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_model(&state, request.model.as_deref())?;

    if request.input.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "input must not be empty".into(),
        ));
    }

    let pooling = parse_embedding_pooling(request.pooling.as_deref())
        .map_err(|message| error_response(StatusCode::BAD_REQUEST, "invalid_request", message))?;
    let normalize = request.normalize.unwrap_or(true);

    let token_count = request.input.len();
    let model_id = state.model_id.as_ref().clone();
    let embedding = state
        .embedding_batcher
        .embed(request.input, pooling, normalize)
        .await
        .map_err(map_session_error)?;

    Ok(Json(OpenAiEmbeddingResponse {
        object: "list",
        data: vec![OpenAiEmbeddingObject {
            object: "embedding",
            embedding,
            index: 0,
        }],
        model: model_id,
        usage: OpenAiEmbeddingUsage {
            prompt_tokens: token_count,
            total_tokens: token_count,
        },
    }))
}

fn parse_embedding_pooling(pooling: Option<&str>) -> Result<EmbeddingPooling, String> {
    match pooling.unwrap_or("last") {
        "last" => Ok(EmbeddingPooling::Last),
        "mean" => Ok(EmbeddingPooling::Mean),
        "cls" => Ok(EmbeddingPooling::Cls),
        other => Err(format!(
            "unknown pooling strategy {other:?}; expected \"last\", \"mean\", or \"cls\""
        )),
    }
}

impl EmbeddingMicroBatcher {
    fn spawn(request_session: Arc<Mutex<EngineSession>>) -> Arc<Self> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let batch_window = embedding_microbatch_window();
        let max_batch = embedding_microbatch_max_batch();
        tokio::spawn(run_embedding_microbatch_worker(
            receiver,
            request_session,
            batch_window,
            max_batch,
        ));
        Arc::new(Self { sender })
    }

    async fn embed(
        &self,
        input: Vec<u32>,
        pooling: EmbeddingPooling,
        normalize: bool,
    ) -> Result<Vec<f32>, EngineSessionError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.sender
            .send(EmbeddingBatchItem {
                input,
                pooling,
                normalize,
                response_tx,
            })
            .map_err(|_| EngineSessionError::EmbeddingFailed {
                message: "embedding microbatch queue is unavailable",
            })?;

        response_rx
            .await
            .map_err(|_| EngineSessionError::EmbeddingFailed {
                message: "embedding microbatch response channel closed",
            })?
    }
}

async fn run_embedding_microbatch_worker(
    mut receiver: mpsc::UnboundedReceiver<EmbeddingBatchItem>,
    request_session: Arc<Mutex<EngineSession>>,
    batch_window: Duration,
    max_batch: usize,
) {
    while let Some(first) = receiver.recv().await {
        let batch =
            collect_embedding_microbatch(&mut receiver, first, batch_window, max_batch).await;
        let run_items: Vec<EmbeddingBatchRunItem> = batch
            .iter()
            .map(|item| EmbeddingBatchRunItem {
                input: item.input.clone(),
                pooling: item.pooling,
                normalize: item.normalize,
            })
            .collect();
        let responses = run_embedding_batch(request_session.clone(), run_items).await;
        for (item, response) in batch.into_iter().zip(responses) {
            let _ = item.response_tx.send(response);
        }
    }
}

async fn collect_embedding_microbatch(
    receiver: &mut mpsc::UnboundedReceiver<EmbeddingBatchItem>,
    first: EmbeddingBatchItem,
    batch_window: Duration,
    max_batch: usize,
) -> Vec<EmbeddingBatchItem> {
    let target = max_batch.max(1);
    let mut batch = Vec::with_capacity(target);
    batch.push(first);
    if target == 1 {
        return batch;
    }

    if batch_window.is_zero() {
        while batch.len() < target {
            match receiver.try_recv() {
                Ok(item) => batch.push(item),
                Err(_) => break,
            }
        }
        return batch;
    }

    let deadline = Instant::now() + batch_window;
    while batch.len() < target {
        tokio::select! {
            maybe = receiver.recv() => {
                match maybe {
                    Some(item) => batch.push(item),
                    None => break,
                }
            }
            _ = sleep_until(deadline) => break,
        }
    }

    batch
}

async fn run_embedding_batch(
    request_session: Arc<Mutex<EngineSession>>,
    items: Vec<EmbeddingBatchRunItem>,
) -> Vec<Result<Vec<f32>, EngineSessionError>> {
    let groups = collect_embedding_batch_groups(
        &items
            .iter()
            .map(|item| EmbeddingBatchRequestOptions {
                pooling: item.pooling,
                normalize: item.normalize,
            })
            .collect::<Vec<_>>(),
    );

    let item_count = items.len();
    match tokio::task::spawn_blocking(move || {
        let session = request_session.blocking_lock();
        let mut outputs: Vec<Option<Result<Vec<f32>, EngineSessionError>>> =
            (0..item_count).map(|_| None).collect();

        for (key, indices) in groups {
            let pooling = pooling_from_code(key.pooling_code);
            let batch_inputs: Vec<Vec<u32>> = indices
                .iter()
                .map(|index| items[*index].input.clone())
                .collect();

            match session.embed_batch(&batch_inputs, pooling, key.normalize) {
                Ok(vectors) if vectors.len() == indices.len() => {
                    for (index, vector) in indices.iter().copied().zip(vectors) {
                        outputs[index] = Some(Ok(vector));
                    }
                }
                Ok(_) | Err(_) => {
                    // Fallback preserves per-request error behavior even if grouped execution fails.
                    for index in indices {
                        outputs[index] = Some(session.embed(
                            &items[index].input,
                            items[index].pooling,
                            items[index].normalize,
                        ));
                    }
                }
            }
        }

        outputs
            .into_iter()
            .map(|value| {
                value.unwrap_or(Err(EngineSessionError::EmbeddingFailed {
                    message: "embedding microbatch output missing",
                }))
            })
            .collect::<Vec<_>>()
    })
    .await
    {
        Ok(outputs) => outputs,
        Err(_) => (0..item_count)
            .map(|_| {
                Err(EngineSessionError::EmbeddingFailed {
                    message: "embedding microbatch worker failed",
                })
            })
            .collect(),
    }
}

fn collect_embedding_batch_groups(
    options: &[EmbeddingBatchRequestOptions],
) -> Vec<(EmbeddingBatchKey, Vec<usize>)> {
    let mut grouped: BTreeMap<EmbeddingBatchKey, Vec<usize>> = BTreeMap::new();
    for (index, option) in options.iter().enumerate() {
        grouped
            .entry(EmbeddingBatchKey {
                pooling_code: pooling_code(option.pooling),
                normalize: option.normalize,
            })
            .or_default()
            .push(index);
    }
    grouped.into_iter().collect()
}

fn pooling_code(pooling: EmbeddingPooling) -> u8 {
    match pooling {
        EmbeddingPooling::Mean => 0,
        EmbeddingPooling::Last => 1,
        EmbeddingPooling::Cls => 2,
    }
}

fn pooling_from_code(code: u8) -> EmbeddingPooling {
    match code {
        0 => EmbeddingPooling::Mean,
        1 => EmbeddingPooling::Last,
        _ => EmbeddingPooling::Cls,
    }
}

fn embedding_microbatch_window() -> Duration {
    let millis = env::var("AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(DEFAULT_EMBEDDING_MICROBATCH_WINDOW_MS)
        .min(100);
    Duration::from_millis(millis)
}

fn embedding_microbatch_max_batch() -> usize {
    env::var("AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_EMBEDDING_MICROBATCH_MAX_BATCH)
        .clamp(1, 512)
}

async fn openai_completions(
    State(state): State<AppState>,
    Json(request): Json<OpenAiCompletionHttpRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    validate_openai_request(&state, request.model.as_deref())?;
    let request = build_openai_completion_request(&state, request)?;

    run_openai_text_generation(state, request, OpenAiStreamKind::Completion).await
}

async fn openai_chat_completions(
    State(state): State<AppState>,
    Json(request): Json<OpenAiChatCompletionHttpRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    validate_openai_request(&state, request.model.as_deref())?;
    let request = build_openai_chat_request(&state, request)?;

    run_openai_text_generation(state, request, OpenAiStreamKind::ChatCompletion).await
}

async fn run_openai_text_generation(
    state: AppState,
    request: OpenAiBuiltRequest,
    kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    if request.stream {
        return stream_openai_request(state, request.generate_request, kind).await;
    }

    let (request_id, response) =
        run_stateless_generate_request(&state, request.generate_request).await?;

    Ok(kind.build_non_stream_response(&response, request_id))
}

async fn generate_stream(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<Sse<ReceiverStream<StreamEvent>>, (StatusCode, Json<ErrorResponse>)> {
    validate_model(&state, request.model.as_deref())?;

    let request = build_generate_request(&state, request);
    let (stream_state, stream_context) = build_stream_state(&state, request).await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_stream_task(
        tx,
        stream_state,
        move |stream_state, tx| match stream_context {
            StreamStateSource::Stateless(context) => {
                drive_stateless_generate_stream_state(context, stream_state, tx);
            }
            StreamStateSource::Stateful(mut session) => {
                drive_generate_stream_state(&mut session, stream_state, tx);
            }
        },
    );

    Ok(build_keep_alive_stream(rx))
}

async fn stream_openai_request(
    state: AppState,
    request: GenerateRequest,
    stream_kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let (stream_state, stream_context) = build_stream_state(&state, request).await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_stream_task(
        tx,
        stream_state,
        move |stream_state, tx| match stream_context {
            StreamStateSource::Stateless(context) => {
                drive_stateless_openai_stream_state(context, stream_state, tx, stream_kind);
            }
            StreamStateSource::Stateful(mut session) => {
                drive_openai_stream_state(&mut session, stream_state, tx, stream_kind);
            }
        },
    );

    Ok(build_keep_alive_stream(rx).into_response())
}

enum StreamStateSource {
    Stateless(Arc<StatelessGenerateContext>),
    Stateful(Box<EngineSession>),
}

async fn build_stream_state(
    state: &AppState,
    request: GenerateRequest,
) -> Result<(GenerateStreamState, StreamStateSource), (StatusCode, Json<ErrorResponse>)> {
    let request_id = allocate_request_id(state);
    if state
        .stateless_generate_context
        .supports_stateless_streaming()
    {
        let stateless_generate_context = Arc::clone(&state.stateless_generate_context);
        let stream_context = Arc::clone(&stateless_generate_context);
        let stream_state = run_blocking_session_task(move || {
            stream_context.stream_state_with_request_id(request_id, request)
        })
        .await?;

        return Ok((
            stream_state,
            StreamStateSource::Stateless(stateless_generate_context),
        ));
    }

    let session_config = state.session_config.as_ref().clone();
    let (session, stream_state) = run_blocking_session_task(move || {
        let mut session = EngineSession::new(session_config)?;
        let stream_state = session.stream_generate_state_with_request_id(request_id, request)?;
        Ok((session, stream_state))
    })
    .await?;

    Ok((stream_state, StreamStateSource::Stateful(Box::new(session))))
}

async fn submit_request(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<(StatusCode, Json<SessionRequestReport>), (StatusCode, Json<ErrorResponse>)> {
    validate_model(&state, request.model.as_deref())?;

    let request_id = allocate_request_id(&state);
    let request = build_generate_request(&state, request);
    let request_session = state.request_session.clone();
    let report = run_blocking_session_task(move || {
        let mut session = request_session.blocking_lock();
        let request_id = session.submit_generate_with_request_id(request_id, request)?;
        session.request_report(request_id).ok_or(
            EngineSessionError::RequestReportInvariantViolation {
                request_id,
                message: "request missing immediately after submission",
            },
        )
    })
    .await?;

    Ok((StatusCode::CREATED, Json(report)))
}

fn allocate_request_id(state: &AppState) -> u64 {
    state.next_request_id.fetch_add(1, Ordering::AcqRel)
}

async fn run_stateless_generate_request(
    state: &AppState,
    request: GenerateRequest,
) -> Result<(u64, GenerateResponse), (StatusCode, Json<ErrorResponse>)> {
    let request_id = allocate_request_id(state);
    let context = Arc::clone(&state.stateless_generate_context);
    let response =
        run_blocking_session_task(move || context.generate_with_request_id(request_id, request))
            .await?;

    Ok((request_id, response))
}

fn spawn_stream_task<F>(tx: StreamEventSender, stream_state: GenerateStreamState, driver: F)
where
    F: FnOnce(&mut GenerateStreamState, StreamEventSender) + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let mut stream_state = stream_state;
        driver(&mut stream_state, tx);
    });
}

fn validate_openai_request(
    state: &AppState,
    model: Option<&str>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    validate_openai_text_backend(state)?;
    validate_model(state, model)
}

fn drive_generate_stream_state(
    session: &mut EngineSession,
    state: &mut GenerateStreamState,
    tx: StreamEventSender,
) {
    drive_stream_events(
        state,
        &tx,
        |state| session.next_stream_event(state),
        |event| send_sdk_stream_event(&tx, event),
        || {},
    );
}

fn drive_stateless_generate_stream_state(
    context: Arc<StatelessGenerateContext>,
    state: &mut GenerateStreamState,
    tx: StreamEventSender,
) {
    drive_stream_events(
        state,
        &tx,
        |state| context.next_stream_event(state),
        |event| send_sdk_stream_event(&tx, event),
        || {},
    );
}

fn drive_openai_stream_state(
    session: &mut EngineSession,
    state: &mut GenerateStreamState,
    tx: StreamEventSender,
    stream_kind: OpenAiStreamKind,
) {
    let mut chat_role_emitted = false;

    drive_stream_events(
        state,
        &tx,
        |state| session.next_stream_event(state),
        |event| send_openai_stream_event(&tx, event, stream_kind, &mut chat_role_emitted),
        || {
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        },
    );
}

fn drive_stateless_openai_stream_state(
    context: Arc<StatelessGenerateContext>,
    state: &mut GenerateStreamState,
    tx: StreamEventSender,
    stream_kind: OpenAiStreamKind,
) {
    let mut chat_role_emitted = false;

    drive_stream_events(
        state,
        &tx,
        |state| context.next_stream_event(state),
        |event| send_openai_stream_event(&tx, event, stream_kind, &mut chat_role_emitted),
        || {
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        },
    );
}

fn drive_stream_events<N, E, D>(
    state: &mut GenerateStreamState,
    tx: &StreamEventSender,
    mut next_event: N,
    mut emit_event: E,
    mut on_done: D,
) where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
    E: FnMut(GenerateStreamEvent) -> bool,
    D: FnMut(),
{
    loop {
        match next_event(state) {
            Ok(Some(event)) => {
                if !emit_event(event) {
                    return;
                }
            }
            Ok(None) => {
                on_done();
                return;
            }
            Err(error) => {
                let (_, Json(error)) = map_session_error(error);
                send_stream_error(tx, error);
                return;
            }
        }
    }
}

fn send_sdk_stream_event(tx: &StreamEventSender, event: GenerateStreamEvent) -> bool {
    let event_name = event.event_name();

    match event {
        GenerateStreamEvent::Request(payload) => send_stream_event(tx, event_name, &payload),
        GenerateStreamEvent::Step(payload) => send_stream_event(tx, event_name, &payload),
        GenerateStreamEvent::Response(payload) => send_stream_event(tx, event_name, &payload),
    }
}

fn send_openai_stream_event(
    tx: &StreamEventSender,
    event: GenerateStreamEvent,
    stream_kind: OpenAiStreamKind,
    chat_role_emitted: &mut bool,
) -> bool {
    match event {
        GenerateStreamEvent::Request(_) => true,
        GenerateStreamEvent::Step(payload) => match stream_kind {
            OpenAiStreamKind::Completion => {
                let Some(delta_text) = payload.delta_text else {
                    return true;
                };
                if delta_text.is_empty() {
                    return true;
                }
                let chunk = OpenAiCompletionChunk {
                    id: stream_kind.response_id(payload.request.request_id),
                    object: stream_kind.stream_chunk_object(),
                    created: unix_timestamp_secs(),
                    model: payload.request.model_id,
                    choices: vec![OpenAiCompletionChunkChoice {
                        index: 0,
                        text: delta_text,
                        finish_reason: None,
                    }],
                };
                send_openai_stream_chunk(tx, &chunk)
            }
            OpenAiStreamKind::ChatCompletion => {
                let Some(delta_text) = payload.delta_text else {
                    return true;
                };
                if delta_text.is_empty() {
                    return true;
                }
                let chunk = OpenAiChatCompletionChunk {
                    id: stream_kind.response_id(payload.request.request_id),
                    object: stream_kind.stream_chunk_object(),
                    created: unix_timestamp_secs(),
                    model: payload.request.model_id,
                    choices: vec![OpenAiChatCompletionChunkChoice {
                        index: 0,
                        delta: OpenAiChatDelta {
                            role: if *chat_role_emitted {
                                None
                            } else {
                                *chat_role_emitted = true;
                                Some("assistant")
                            },
                            content: Some(delta_text),
                        },
                        finish_reason: None,
                    }],
                };
                send_openai_stream_chunk(tx, &chunk)
            }
        },
        GenerateStreamEvent::Response(payload) => match stream_kind {
            OpenAiStreamKind::Completion => {
                let chunk = OpenAiCompletionChunk {
                    id: stream_kind.response_id(payload.response.request_id),
                    object: stream_kind.stream_chunk_object(),
                    created: unix_timestamp_secs(),
                    model: payload.response.model_id,
                    choices: vec![OpenAiCompletionChunkChoice {
                        index: 0,
                        text: String::new(),
                        finish_reason: openai_finish_reason(payload.response.finish_reason),
                    }],
                };
                send_openai_stream_chunk(tx, &chunk)
            }
            OpenAiStreamKind::ChatCompletion => {
                let chunk = OpenAiChatCompletionChunk {
                    id: stream_kind.response_id(payload.response.request_id),
                    object: stream_kind.stream_chunk_object(),
                    created: unix_timestamp_secs(),
                    model: payload.response.model_id,
                    choices: vec![OpenAiChatCompletionChunkChoice {
                        index: 0,
                        delta: OpenAiChatDelta::default(),
                        finish_reason: openai_finish_reason(payload.response.finish_reason),
                    }],
                };
                send_openai_stream_chunk(tx, &chunk)
            }
        },
    }
}

fn send_stream_event<T: Serialize>(tx: &StreamEventSender, event_name: &str, payload: &T) -> bool {
    match serde_json::to_string(payload) {
        Ok(data) => tx
            .blocking_send(Ok(Event::default().event(event_name).data(data)))
            .is_ok(),
        Err(error) => {
            send_stream_error(
                tx,
                ErrorResponse {
                    error: ErrorBody {
                        error_type: "server_error",
                        code: Some("engine_error".to_string()),
                        param: None,
                        message: format!("failed to serialize {event_name} event: {error}"),
                    },
                },
            );
            false
        }
    }
}

fn build_keep_alive_stream(rx: StreamEventReceiver) -> Sse<ReceiverStream<StreamEvent>> {
    Sse::new(ReceiverStream::new(rx)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keep-alive"),
    )
}

fn send_stream_error(tx: &StreamEventSender, error: ErrorResponse) {
    let payload = serde_json::to_string(&error).unwrap_or_else(|_| {
        "{\"error\":{\"type\":\"server_error\",\"code\":\"engine_error\",\"param\":null,\"message\":\"failed to serialize stream error\"}}".to_string()
    });
    let _ = tx.blocking_send(Ok(Event::default().event("error").data(payload)));
    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
}

fn send_openai_stream_chunk<T: Serialize>(tx: &StreamEventSender, payload: &T) -> bool {
    match serde_json::to_string(payload) {
        Ok(data) => tx.blocking_send(Ok(Event::default().data(data))).is_ok(),
        Err(error) => {
            send_stream_error(
                tx,
                ErrorResponse {
                    error: ErrorBody {
                        error_type: "server_error",
                        code: Some("engine_error".to_string()),
                        param: None,
                        message: format!("failed to serialize OpenAI stream chunk: {error}"),
                    },
                },
            );
            false
        }
    }
}

async fn request_snapshot(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    let session = state.request_session.lock().await;
    let report = session
        .request_report(request_id)
        .ok_or_else(|| request_not_found_response(request_id))?;

    Ok(Json(report))
}

async fn cancel_request(
    State(state): State<AppState>,
    Path(request_id): Path<u64>,
) -> Result<Json<SessionRequestReport>, (StatusCode, Json<ErrorResponse>)> {
    let mut session = state.request_session.lock().await;
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

async fn step_request(
    State(state): State<AppState>,
) -> Result<Json<EngineStepReport>, (StatusCode, Json<ErrorResponse>)> {
    let request_session = state.request_session.clone();
    let report = run_blocking_session_task(move || {
        let mut session = request_session.blocking_lock();
        session.step_report()
    })
    .await?;
    Ok(Json(report))
}

async fn run_blocking_session_task<T, F>(
    operation: F,
) -> Result<T, (StatusCode, Json<ErrorResponse>)>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, EngineSessionError> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(map_blocking_task_error)?
        .map_err(map_session_error)
}

fn map_blocking_task_error(error: tokio::task::JoinError) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "engine_error",
        format!("blocking server task failed: {error}"),
    )
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

struct OpenAiBuiltRequest {
    generate_request: GenerateRequest,
    stream: bool,
}

fn build_generate_request(state: &AppState, request: GenerateHttpRequest) -> GenerateRequest {
    build_generate_request_internal(
        state,
        request.input_tokens,
        request.input_text,
        request.max_output_tokens.unwrap_or(256),
        request.sampling.unwrap_or_default(),
        Vec::new(),
        request.metadata,
    )
}

fn build_openai_completion_request(
    state: &AppState,
    request: OpenAiCompletionHttpRequest,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = require_openai_max_tokens(request.max_tokens)?;
    let (input_tokens, input_text) = match request.prompt {
        OpenAiPromptInput::Text(text) => (Vec::new(), Some(text)),
        OpenAiPromptInput::TextBatch(texts) => (Vec::new(), Some(texts.join("\n"))),
        OpenAiPromptInput::Tokens(tokens) => (tokens, None),
    };

    Ok(OpenAiBuiltRequest {
        generate_request: build_openai_generate_request(
            state,
            input_tokens,
            input_text,
            max_output_tokens,
            build_openai_sampling(
                request.temperature,
                request.top_p,
                request.top_k,
                request.min_p,
                request.repetition_penalty,
                request.repetition_context_size,
                request.seed,
            ),
            request.stop,
            request.metadata,
        ),
        stream: request.stream,
    })
}

fn build_openai_chat_request(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = require_openai_max_tokens(request.max_tokens)?;
    let input_text = render_openai_chat_prompt(state.model_id.as_ref(), &request.messages)?;

    Ok(OpenAiBuiltRequest {
        generate_request: build_openai_generate_request(
            state,
            Vec::new(),
            Some(input_text),
            max_output_tokens,
            build_openai_sampling(
                request.temperature,
                request.top_p,
                request.top_k,
                request.min_p,
                request.repetition_penalty,
                request.repetition_context_size,
                request.seed,
            ),
            request.stop,
            request.metadata,
        ),
        stream: request.stream,
    })
}

fn build_openai_generate_request(
    state: &AppState,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    max_output_tokens: u32,
    sampling: GenerateSampling,
    stop: Option<OpenAiStopInput>,
    metadata: Option<String>,
) -> GenerateRequest {
    build_generate_request_internal(
        state,
        input_tokens,
        input_text,
        max_output_tokens,
        sampling,
        stop.map(OpenAiStopInput::into_vec).unwrap_or_default(),
        metadata,
    )
}

fn build_generate_request_internal(
    state: &AppState,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    max_output_tokens: u32,
    sampling: GenerateSampling,
    stop_sequences: Vec<String>,
    metadata: Option<String>,
) -> GenerateRequest {
    GenerateRequest {
        model_id: state.model_id.to_string(),
        input_tokens,
        input_text,
        max_output_tokens,
        sampling,
        stop_sequences,
        metadata,
    }
}

fn build_openai_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    repetition_context_size: Option<u32>,
    seed: Option<u64>,
) -> GenerateSampling {
    GenerateSampling {
        temperature: temperature.unwrap_or(0.0),
        top_p: top_p.unwrap_or(1.0),
        top_k: top_k.unwrap_or(0),
        min_p,
        repetition_penalty: repetition_penalty.unwrap_or(1.0),
        repetition_context_size,
        seed: seed.unwrap_or(0),
        deterministic: None,
    }
}

fn require_openai_max_tokens(
    max_tokens: Option<u32>,
) -> Result<u32, (StatusCode, Json<ErrorResponse>)> {
    max_tokens.ok_or_else(|| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "OpenAI-compatible preview endpoints require max_tokens; refusing to apply a hidden default".to_string(),
        )
    })
}

fn render_openai_chat_prompt(
    model_id: &str,
    messages: &[OpenAiChatMessage],
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    if messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "chat.completions requires at least one message".to_string(),
        ));
    }

    render_openai_chat_prompt_with_template(ChatPromptTemplate::for_model_id(model_id), messages)
}

fn render_openai_chat_prompt_with_template(
    template: ChatPromptTemplate,
    messages: &[OpenAiChatMessage],
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    let mut prompt = String::new();
    if template == ChatPromptTemplate::Llama3 {
        prompt.push_str("<|begin_of_text|>");
    }
    for message in messages {
        let role = normalize_openai_chat_role(&message.role)?;
        let content = render_openai_chat_content(&message.content)?;
        match template {
            ChatPromptTemplate::QwenChatMl => {
                prompt.push_str("<|im_start|>");
                prompt.push_str(role);
                prompt.push('\n');
                prompt.push_str(&content);
                prompt.push_str("<|im_end|>\n");
            }
            ChatPromptTemplate::Llama3 => {
                prompt.push_str("<|start_header_id|>");
                prompt.push_str(role);
                prompt.push_str("<|end_header_id|>\n\n");
                prompt.push_str(&content);
                prompt.push_str("<|eot_id|>");
            }
            ChatPromptTemplate::PlainRolePrefix => {
                prompt.push_str(role);
                prompt.push_str(": ");
                prompt.push_str(&content);
                prompt.push('\n');
            }
        }
    }
    match template {
        ChatPromptTemplate::QwenChatMl => prompt.push_str("<|im_start|>assistant\n"),
        ChatPromptTemplate::Llama3 => {
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        ChatPromptTemplate::PlainRolePrefix => prompt.push_str("assistant:"),
    }
    Ok(prompt)
}

fn normalize_openai_chat_role(
    role: &str,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    match role.trim() {
        "system" => Ok("system"),
        "user" => Ok("user"),
        "assistant" => Ok("assistant"),
        "tool" => Ok("tool"),
        "function" => Ok("function"),
        _ => Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "unsupported chat role; expected one of system, user, assistant, tool, function"
                .to_string(),
        )),
    }
}

fn render_openai_chat_content(
    content: &OpenAiChatContent,
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    match content {
        OpenAiChatContent::Text(text) => Ok(text.clone()),
        OpenAiChatContent::Parts(parts) => {
            let mut rendered = String::new();
            for part in parts {
                if part.part_type != "text" {
                    return Err(error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request",
                        format!(
                            "unsupported chat content part type {}; AX preview currently accepts text-only chat messages",
                            part.part_type
                        ),
                    ));
                }
                let text = part.text.as_deref().ok_or_else(|| {
                    error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request",
                        "text chat content parts require a text field".to_string(),
                    )
                })?;
                rendered.push_str(text);
            }
            Ok(rendered)
        }
    }
}

fn openai_completion_response(response: &GenerateResponse, id: String) -> OpenAiCompletionResponse {
    OpenAiCompletionResponse {
        id,
        object: "text_completion",
        created: unix_timestamp_secs(),
        model: response.model_id.clone(),
        choices: vec![OpenAiCompletionChoice {
            index: 0,
            text: response.output_text.clone().unwrap_or_default(),
            finish_reason: openai_finish_reason(response.finish_reason),
        }],
        usage: openai_usage(response),
    }
}

fn openai_chat_completion_response(
    response: &GenerateResponse,
    id: String,
) -> OpenAiChatCompletionResponse {
    OpenAiChatCompletionResponse {
        id,
        object: "chat.completion",
        created: unix_timestamp_secs(),
        model: response.model_id.clone(),
        choices: vec![OpenAiChatCompletionChoice {
            index: 0,
            message: OpenAiChatMessageResponse {
                role: "assistant",
                content: response.output_text.clone().unwrap_or_default(),
            },
            finish_reason: openai_finish_reason(response.finish_reason),
        }],
        usage: openai_usage(response),
    }
}

fn openai_usage(response: &GenerateResponse) -> Option<OpenAiUsage> {
    let (prompt_tokens, completion_tokens) = response.known_usage()?;
    Some(OpenAiUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    })
}

fn openai_finish_reason(finish_reason: Option<GenerateFinishReason>) -> Option<&'static str> {
    match finish_reason {
        Some(GenerateFinishReason::MaxOutputTokens) => Some("length"),
        Some(GenerateFinishReason::Stop) => Some("stop"),
        Some(GenerateFinishReason::ContentFilter) => Some("content_filter"),
        Some(GenerateFinishReason::Cancelled) | Some(GenerateFinishReason::Error) | None => None,
    }
}

fn validate_openai_text_backend(state: &AppState) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if !matches!(
        state.runtime_report.selected_backend,
        SelectedBackend::LlamaCpp | SelectedBackend::MlxLmDelegated
    ) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "OpenAI-compatible text endpoints require a llama.cpp or mlx_lm_delegated backend; use /v1/generate for repo-owned MLX preview".to_string(),
        ));
    }
    Ok(())
}

fn unix_timestamp_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn validate_model(
    state: &AppState,
    request_model: Option<&str>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(model) = request_model {
        if model != state.model_id.as_ref() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "model_mismatch",
                format!(
                    "requested model_id {model} does not match configured preview model {}",
                    state.model_id.as_ref()
                ),
            ));
        }
    }

    Ok(())
}

fn map_session_error(error: EngineSessionError) -> (StatusCode, Json<ErrorResponse>) {
    match error {
        EngineSessionError::EmptyInputTokens
        | EngineSessionError::InvalidMaxOutputTokens
        | EngineSessionError::MlxBackendRequiresTokenizedInput
        | EngineSessionError::InvalidMaxBatchTokens
        | EngineSessionError::InvalidRequestId
        | EngineSessionError::UnsupportedSupportTier
        | EngineSessionError::LlamaCppDoesNotSupportLifecycle { .. }
        | EngineSessionError::MlxLmDoesNotSupportLifecycle { .. }
        | EngineSessionError::MlxLmDoesNotSupportStreaming
        | EngineSessionError::NativeBackendStatelessStreamNotSupported { .. }
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::StreamingNotSupported { .. })
        | EngineSessionError::RequestDidNotTerminate { .. }
        | EngineSessionError::MissingRequestSnapshot { .. } => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
        ),
        EngineSessionError::LlamaCpp(LlamaCppBackendError::MissingInputText { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::MissingPromptInput { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::UnsupportedTokenPrompt { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::AmbiguousPromptInput { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::BackendConfigMismatch { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingInputText)
        | EngineSessionError::MlxLm(MlxLmBackendError::UnsupportedTokenPrompt)
        | EngineSessionError::MlxLm(MlxLmBackendError::BackendConfigMismatch { .. }) => {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                error.to_string(),
            )
        }
        EngineSessionError::MlxLm(MlxLmBackendError::MissingCompletionChoice { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingStreamChoice { .. }) => {
            error_response(StatusCode::BAD_GATEWAY, "backend_error", error.to_string())
        }
        EngineSessionError::BackendContract(_)
        | EngineSessionError::MissingLlamaCppConfig { .. }
        | EngineSessionError::MissingMlxLmConfig
        | EngineSessionError::MissingDelegatedRuntime { .. }
        | EngineSessionError::LlamaCppStreamEndedBeforeStop { .. }
        | EngineSessionError::MlxRuntimeArtifactsRequired
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::CommandLaunch { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::CommandFailed { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::CommandTimedOut { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::NonUtf8Output { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::SerializeRequestJson { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::HttpRequest { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::HttpStatus { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::HttpResponseRead { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::InvalidResponseJson { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::SerializeRequestJson { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::HttpRequest { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::HttpStatus { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::InvalidResponseJson { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::SseRead { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::InvalidStreamChunk { .. })
        | EngineSessionError::UnsupportedHostHardware { .. } => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "unsupported_host",
            error.to_string(),
        ),
        EngineSessionError::EmbeddingNotSupported => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
        ),
        EngineSessionError::RequestReportInvariantViolation { .. }
        | EngineSessionError::StreamEndedWithoutResponse { .. }
        | EngineSessionError::EmbeddingFailed { .. }
        | EngineSessionError::Core(_)
        | EngineSessionError::MetalRuntime(_)
        | EngineSessionError::MlxRuntimeUnavailable => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "engine_error",
            error.to_string(),
        ),
    }
}

fn error_response(
    status: StatusCode,
    code: &'static str,
    message: String,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: ErrorBody {
                error_type: openai_error_type(status),
                code: Some(code.to_string()),
                param: None,
                message,
            },
        }),
    )
}

fn openai_error_type(status: StatusCode) -> &'static str {
    if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    }
}

fn request_not_found_response(request_id: u64) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::NOT_FOUND,
        "request_not_found",
        format!("request {request_id} is missing from preview session state"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, header};
    use serde_json::Value;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    const TEST_MODEL_ID: &str = "qwen3_dense";

    fn sample_http_request(input_tokens: &[u32], max_output_tokens: u32) -> Value {
        json!({
            "model": TEST_MODEL_ID,
            "input_tokens": input_tokens,
            "max_output_tokens": max_output_tokens
        })
    }

    fn sample_sdk_request(input_tokens: &[u32], max_output_tokens: u32) -> GenerateRequest {
        GenerateRequest {
            model_id: TEST_MODEL_ID.to_string(),
            input_tokens: input_tokens.to_vec(),
            input_text: None,
            max_output_tokens,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        }
    }

    fn sample_text_http_request(input_text: &str, max_output_tokens: u32) -> Value {
        json!({
            "model": TEST_MODEL_ID,
            "input_text": input_text,
            "max_output_tokens": max_output_tokens
        })
    }

    fn sample_openai_request_base(
        max_tokens: Option<u32>,
        stream: bool,
    ) -> serde_json::Map<String, Value> {
        let mut request = serde_json::Map::new();
        request.insert("model".to_string(), json!(TEST_MODEL_ID));
        if let Some(max_tokens) = max_tokens {
            request.insert("max_tokens".to_string(), json!(max_tokens));
        }
        request.insert("stream".to_string(), json!(stream));
        request
    }

    fn sample_openai_completion_request(
        prompt: &str,
        max_tokens: Option<u32>,
        stream: bool,
    ) -> Value {
        let mut request = sample_openai_request_base(max_tokens, stream);
        request.insert("prompt".to_string(), json!(prompt));
        Value::Object(request)
    }

    fn sample_openai_chat_request(message: &str, max_tokens: Option<u32>, stream: bool) -> Value {
        sample_openai_chat_request_with_role(message, "user", max_tokens, stream)
    }

    fn expected_qwen_chatml_user_prompt(message: &str) -> String {
        format!("<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn sample_openai_chat_request_with_role(
        message: &str,
        role: &str,
        max_tokens: Option<u32>,
        stream: bool,
    ) -> Value {
        let mut request = sample_openai_request_base(max_tokens, stream);
        request.insert(
            "messages".to_string(),
            json!([{
                "role": role,
                "content": message,
            }]),
        );
        Value::Object(request)
    }

    fn sample_openai_embedding_request(input: &[u32], pooling: Option<&str>) -> Value {
        let mut request = serde_json::Map::new();
        request.insert("model".to_string(), json!(TEST_MODEL_ID));
        request.insert("input".to_string(), json!(input));
        if let Some(pooling) = pooling {
            request.insert("pooling".to_string(), json!(pooling));
        }
        Value::Object(request)
    }

    fn sdk_session_for_state(state: &AppState) -> EngineSession {
        EngineSession::new(state.session_config.as_ref().clone()).expect("sdk session should build")
    }

    fn sdk_stream_payload(event: GenerateStreamEvent) -> (String, Value) {
        match event {
            GenerateStreamEvent::Request(payload) => (
                "request".to_string(),
                serde_json::to_value(payload).expect("request payload should serialize"),
            ),
            GenerateStreamEvent::Step(payload) => (
                "step".to_string(),
                serde_json::to_value(payload).expect("step payload should serialize"),
            ),
            GenerateStreamEvent::Response(payload) => (
                "response".to_string(),
                serde_json::to_value(payload).expect("response payload should serialize"),
            ),
        }
    }

    fn parse_sse_events(body: &str) -> Vec<(String, Value)> {
        let mut events = Vec::new();
        let mut current_name: Option<String> = None;
        let mut current_data = String::new();

        for line in body.lines() {
            if line.is_empty() {
                if let Some(name) = current_name.take() {
                    let payload =
                        serde_json::from_str(&current_data).expect("sse payload should be json");
                    events.push((name, payload));
                    current_data.clear();
                }
                continue;
            }

            if let Some(value) = line.strip_prefix("event: ") {
                current_name = Some(value.to_string());
            } else if let Some(value) = line.strip_prefix("data: ") {
                if !current_data.is_empty() {
                    current_data.push('\n');
                }
                current_data.push_str(value);
            }
        }

        if let Some(name) = current_name {
            let payload = serde_json::from_str(&current_data).expect("sse payload should be json");
            events.push((name, payload));
        }

        events
    }

    fn parse_openai_sse_payloads(body: &str) -> Vec<Value> {
        let mut payloads = Vec::new();
        let mut current_data = String::new();

        for line in body.lines() {
            if let Some(value) = line.strip_prefix("data: ") {
                if value == "[DONE]" {
                    break;
                }
                if !current_data.is_empty() {
                    current_data.push('\n');
                }
                current_data.push_str(value);
                continue;
            }

            if line.is_empty() && !current_data.is_empty() {
                payloads.push(
                    serde_json::from_str(&current_data).expect("openai sse payload should parse"),
                );
                current_data.clear();
            }
        }

        if !current_data.is_empty() {
            payloads.push(
                serde_json::from_str(&current_data).expect("openai sse payload should parse"),
            );
        }

        payloads
    }

    fn json_request_body<T: serde::Serialize>(value: &T) -> Vec<u8> {
        serde_json::to_vec(value).expect("request body should serialize")
    }

    fn normalize_measurement_fields(value: &mut Value) {
        match value {
            Value::Object(map) => {
                map.remove("cpu_time_us");
                map.remove("runner_time_us");
                for value in map.values_mut() {
                    normalize_measurement_fields(value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    normalize_measurement_fields(value);
                }
            }
            _ => {}
        }
    }

    fn base_server_args() -> ServerArgs {
        ServerArgs {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model_id: TEST_MODEL_ID.to_string(),
            preset: None,
            list_presets: false,
            deterministic: true,
            max_batch_tokens: 2048,
            cache_group_id: 0,
            block_size_tokens: 16,
            total_blocks: 1024,
            mlx: false,
            support_tier: args::PreviewSupportTier::LlamaCpp,
            llama_cli_path: "llama-cli".to_string(),
            llama_model_path: None,
            llama_server_url: None,
            mlx_lm_server_url: None,
            delegated_http_connect_timeout_secs:
                ax_engine_sdk::DelegatedHttpTimeouts::default_connect_secs(),
            delegated_http_read_timeout_secs: ax_engine_sdk::DelegatedHttpTimeouts::default_io_secs(
            ),
            delegated_http_write_timeout_secs:
                ax_engine_sdk::DelegatedHttpTimeouts::default_io_secs(),
            mlx_model_artifacts_dir: None,
            resolve_model_artifacts: args::ModelArtifactResolution::ExplicitOnly,
            hf_cache_root: None,
            disable_ngram_acceleration: false,
            experimental_mlx_kv_compression: args::PreviewMlxKvCompression::Disabled,
            experimental_mlx_kv_compression_hot_window_tokens:
                ax_engine_sdk::MlxKvCompressionConfig::DEFAULT_HOT_WINDOW_TOKENS,
            experimental_mlx_kv_compression_min_context_tokens:
                ax_engine_sdk::MlxKvCompressionConfig::DEFAULT_MIN_CONTEXT_TOKENS,
        }
    }

    fn fake_llama_cpp_script() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-engine-server-llama-cpp-{unique}.py"));
        let script = r#"#!/usr/bin/env python3
from __future__ import annotations

import sys

args = sys.argv[1:]
prompt = args[args.index("--prompt") + 1]
sys.stdout.write(f"server::{prompt}")
"#;

        fs::write(&path, script).expect("fake script should be written");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = fs::metadata(&path)
                .expect("script metadata should exist")
                .permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&path, permissions).expect("script should be executable");
        }
        path
    }

    fn test_app_state<F>(customize: F) -> AppState
    where
        F: FnOnce(&mut ServerArgs),
    {
        let mut args = base_server_args();
        customize(&mut args);

        let session_config = args.session_config().expect("session config should build");
        let session = EngineSession::new(session_config.clone()).expect("session should build");
        build_app_state(args.model_id.clone(), session).expect("app state should build")
    }

    fn llama_cpp_state() -> AppState {
        let script_path = fake_llama_cpp_script();
        let model_path = std::env::temp_dir().join("ax-engine-server-llama-cpp-model.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");
        test_app_state(|args| {
            args.llama_cli_path = script_path.display().to_string();
            args.llama_model_path = Some(model_path);
        })
    }

    fn llama_cpp_server_state(server_url: String) -> AppState {
        test_app_state(|args| {
            args.llama_server_url = Some(server_url);
        })
    }

    fn mlx_lm_delegated_state(server_url: String) -> AppState {
        test_app_state(|args| {
            args.support_tier = args::PreviewSupportTier::MlxLmDelegated;
            args.mlx_lm_server_url = Some(server_url);
        })
    }

    fn spawn_llama_cpp_completion_server(
        response_body: String,
        assert_request: impl FnOnce(Value) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("request should arrive");
            let payload = read_http_request_payload(&mut stream);
            assert_request(payload);

            write_http_response(&mut stream, "application/json", &response_body, None);
        });

        (format!("http://{address}"), handle)
    }

    fn spawn_llama_cpp_completion_stream_server(
        expected_requests: usize,
        chunks: Vec<Value>,
        assert_request: impl Fn(Value) + Send + Sync + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept().expect("request should arrive");
                let payload = read_http_request_payload(&mut stream);
                assert_request(payload);

                let mut body = String::new();
                for chunk in &chunks {
                    body.push_str("data: ");
                    body.push_str(
                        &serde_json::to_string(chunk).expect("chunk payload should serialize"),
                    );
                    body.push_str("\n\n");
                }
                body.push_str("data: [DONE]\n\n");

                write_http_response(
                    &mut stream,
                    "text/event-stream",
                    &body,
                    Some("Cache-Control: no-cache"),
                );
            }
        });

        (format!("http://{address}"), handle)
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        let mut header_end = None;
        let mut content_length = None;

        loop {
            let bytes_read = stream.read(&mut buffer).expect("request should read");
            assert!(
                bytes_read > 0,
                "client closed connection before request completed"
            );
            request.extend_from_slice(&buffer[..bytes_read]);

            if header_end.is_none() {
                header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4);
                if let Some(end) = header_end {
                    let headers =
                        String::from_utf8(request[..end].to_vec()).expect("headers should be utf8");
                    content_length = Some(parse_content_length(&headers));
                }
            }

            if let (Some(end), Some(length)) = (header_end, content_length) {
                if request.len() >= end + length {
                    request.truncate(end + length);
                    return request;
                }
            }
        }
    }

    fn read_http_request_payload(stream: &mut std::net::TcpStream) -> Value {
        parse_http_request_payload(&read_http_request(stream))
    }

    fn parse_http_request_payload(request: &[u8]) -> Value {
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
            .expect("request should include header terminator");
        let headers =
            String::from_utf8(request[..header_end].to_vec()).expect("headers should be utf8");
        let _ = parse_content_length(&headers);
        let body = String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
        serde_json::from_str(&body).expect("request body should be json")
    }

    fn write_http_response(
        stream: &mut std::net::TcpStream,
        content_type: &str,
        body: &str,
        extra_header: Option<&str>,
    ) {
        let mut headers = format!(
            "Content-Type: {}\r\nContent-Length: {}\r\nConnection: close",
            content_type,
            body.len()
        );
        if let Some(extra_header) = extra_header {
            headers.push_str(&format!("\r\n{extra_header}"));
        }

        let response = format!("HTTP/1.1 200 OK\r\n{headers}\r\n\r\n{body}");
        stream
            .write_all(response.as_bytes())
            .expect("response should write");
    }

    fn parse_content_length(headers: &str) -> usize {
        headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    Some(
                        value
                            .trim()
                            .parse::<usize>()
                            .expect("content-length should parse"),
                    )
                } else {
                    None
                }
            })
            .expect("content-length header should exist")
    }

    async fn json_response(app: &Router, request: Request<Body>) -> (StatusCode, Value) {
        let response = app
            .clone()
            .oneshot(request)
            .await
            .expect("request should succeed");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        let json = serde_json::from_slice(&bytes).expect("response should be json");
        (status, json)
    }

    async fn text_response(app: &Router, request: Request<Body>) -> (StatusCode, String, String) {
        let response = app
            .clone()
            .oneshot(request)
            .await
            .expect("request should succeed");
        let status = response.status();
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        let body = String::from_utf8(bytes.to_vec()).expect("body should be utf8");
        (status, content_type, body)
    }

    #[tokio::test]
    async fn llama_cpp_generate_endpoint_runs_text_request_through_sdk() {
        let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
            serde_json::json!({
                "content": "server::hello from server",
                "tokens": [4, 5],
                "stop": true,
                "stop_type": "limit"
            })
            .to_string(),
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello from server".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );
        let state = llama_cpp_server_state(llama_server_url);
        let app = build_router(state.clone());
        let request_body = sample_text_http_request("hello from server", 2);
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&request_body)))
                .unwrap(),
        )
        .await;
        llama_cpp_server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json.get("prompt_text").and_then(|value| value.as_str()),
            Some("hello from server")
        );
        assert_eq!(
            json.get("output_text").and_then(|value| value.as_str()),
            Some("server::hello from server")
        );
        assert_eq!(
            json.get("runtime")
                .and_then(|runtime| runtime.get("selected_backend"))
                .and_then(|value| value.as_str()),
            Some("llama_cpp")
        );
    }

    #[tokio::test]
    async fn openai_completions_endpoint_translates_llama_cpp_response() {
        let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
            serde_json::json!({
                "content": "server::hello openai",
                "tokens": [4, 5],
                "stop": true,
                "stop_type": "limit"
            })
            .to_string(),
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello openai".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            },
        );
        let app = build_router(llama_cpp_server_state(llama_server_url));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(
                    &sample_openai_completion_request("hello openai", Some(2), false),
                )))
                .unwrap(),
        )
        .await;
        llama_cpp_server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json.get("object").and_then(Value::as_str),
            Some("text_completion")
        );
        assert_eq!(
            json.get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("text"))
                .and_then(Value::as_str),
            Some("server::hello openai")
        );
        assert_eq!(
            json.get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(Value::as_str),
            Some("length")
        );
    }

    #[tokio::test]
    async fn openai_completions_endpoint_translates_mlx_lm_delegated_response() {
        let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_server(
            r#"{"choices":[{"text":"mlx-lm::hello","finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#.to_string(),
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello mlx-lm".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
                assert_eq!(payload.get("top_k"), Some(&json!(0)));
                assert_eq!(payload.get("repetition_penalty"), Some(&json!(1.0)));
            },
        );
        let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(
                    &sample_openai_completion_request("hello mlx-lm", Some(2), false),
                )))
                .unwrap(),
        )
        .await;
        mlx_lm_server_handle
            .join()
            .expect("mlx-lm server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json.get("object").and_then(Value::as_str),
            Some("text_completion")
        );
        assert_eq!(
            json.get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("text"))
                .and_then(Value::as_str),
            Some("mlx-lm::hello")
        );
        assert_eq!(
            json.get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(Value::as_str),
            Some("stop")
        );
    }

    #[tokio::test]
    async fn mlx_lm_delegated_generate_uses_text_and_usage_count_contract() {
        let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_server(
            r#"{"choices":[{"text":"mlx-lm::hello","finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#.to_string(),
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello mlx-lm".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            },
        );
        let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_text_http_request(
                    "hello mlx-lm",
                    2,
                ))))
                .unwrap(),
        )
        .await;
        mlx_lm_server_handle
            .join()
            .expect("mlx-lm server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json.get("output_text").and_then(Value::as_str),
            Some("mlx-lm::hello")
        );
        assert!(
            json.get("output_tokens")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
        );
        assert_eq!(json.get("output_token_count"), Some(&json!(3)));
        assert_eq!(
            json.get("finish_reason").and_then(Value::as_str),
            Some("stop")
        );
    }

    #[tokio::test]
    async fn openai_completions_endpoint_requires_max_tokens() {
        let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(
                    &sample_openai_completion_request("hello openai", None, false),
                )))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            json.get("error")
                .and_then(|error| error.get("code"))
                .and_then(Value::as_str),
            Some("invalid_request")
        );
        assert!(
            json.get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("require max_tokens")
        );
    }

    #[tokio::test]
    async fn openai_embeddings_endpoint_rejects_empty_input() {
        let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(
                    &sample_openai_embedding_request(&[], None),
                )))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            json.get("error")
                .and_then(|error| error.get("code"))
                .and_then(Value::as_str),
            Some("invalid_request")
        );
        assert!(
            json.get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("input must not be empty")
        );
    }

    #[tokio::test]
    async fn openai_embeddings_endpoint_rejects_unknown_pooling() {
        let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(
                    &sample_openai_embedding_request(&[1, 2, 3], Some("max")),
                )))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            json.get("error")
                .and_then(|error| error.get("code"))
                .and_then(Value::as_str),
            Some("invalid_request")
        );
        assert!(
            json.get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("unknown pooling strategy")
        );
    }

    #[test]
    fn embedding_microbatch_groups_requests_by_options() {
        let groups = collect_embedding_batch_groups(&[
            EmbeddingBatchRequestOptions {
                pooling: EmbeddingPooling::Last,
                normalize: true,
            },
            EmbeddingBatchRequestOptions {
                pooling: EmbeddingPooling::Last,
                normalize: true,
            },
            EmbeddingBatchRequestOptions {
                pooling: EmbeddingPooling::Mean,
                normalize: true,
            },
            EmbeddingBatchRequestOptions {
                pooling: EmbeddingPooling::Last,
                normalize: false,
            },
        ]);

        assert_eq!(groups.len(), 3);
        assert_eq!(
            groups[0],
            (
                EmbeddingBatchKey {
                    pooling_code: pooling_code(EmbeddingPooling::Mean),
                    normalize: true
                },
                vec![2]
            )
        );
        assert_eq!(
            groups[1],
            (
                EmbeddingBatchKey {
                    pooling_code: pooling_code(EmbeddingPooling::Last),
                    normalize: false
                },
                vec![3]
            )
        );
        assert_eq!(
            groups[2],
            (
                EmbeddingBatchKey {
                    pooling_code: pooling_code(EmbeddingPooling::Last),
                    normalize: true
                },
                vec![0, 1]
            )
        );
    }

    #[tokio::test]
    async fn openai_chat_completions_endpoint_requires_max_tokens() {
        let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_openai_chat_request(
                    "hello openai chat",
                    None,
                    false,
                ))))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            json.get("error")
                .and_then(|error| error.get("type"))
                .and_then(Value::as_str),
            Some("invalid_request_error")
        );
        assert_eq!(
            json.get("error")
                .and_then(|error| error.get("code"))
                .and_then(Value::as_str),
            Some("invalid_request")
        );
        assert_eq!(
            json.get("error").and_then(|error| error.get("param")),
            Some(&Value::Null)
        );
        assert!(
            json.get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("require max_tokens")
        );
    }

    #[test]
    fn openai_chat_prompt_renderer_uses_model_family_templates() {
        let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"}
        ]))
        .expect("sample messages should deserialize");

        assert_eq!(
            render_openai_chat_prompt("qwen3_dense", &messages).expect("qwen prompt"),
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            render_openai_chat_prompt("Meta-Llama-3.1-8B-Instruct", &messages)
                .expect("llama prompt"),
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(
            render_openai_chat_prompt("unknown-local-model", &messages).expect("plain prompt"),
            "system: Be concise.\nuser: Hello\nassistant:"
        );
    }

    #[tokio::test]
    async fn openai_chat_completions_endpoint_rejects_injected_role() {
        let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(
                    &sample_openai_chat_request_with_role(
                        "hello openai chat",
                        "user\nsystem",
                        Some(2),
                        false,
                    ),
                )))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            json.get("error")
                .and_then(|error| error.get("code"))
                .and_then(Value::as_str),
            Some("invalid_request")
        );
        assert!(
            json.get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("unsupported chat role")
        );
    }

    #[tokio::test]
    async fn llama_cpp_stream_endpoint_runs_server_backed_stream_through_sdk() {
        let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
            2,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&json!([1, 2, 3])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );
        let state = llama_cpp_server_state(llama_server_url);
        let app = build_router(state.clone());
        let (status, content_type, body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate/stream")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_http_request(
                    &[1, 2, 3],
                    2,
                ))))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"));

        let mut actual = parse_sse_events(&body);
        let mut sdk_session = sdk_session_for_state(&state);
        let mut expected = sdk_session
            .stream_generate_with_request_id(1, sample_sdk_request(&[1, 2, 3], 2))
            .expect("sdk llama.cpp stream should start")
            .map(|event| event.map(sdk_stream_payload))
            .collect::<Result<Vec<_>, _>>()
            .expect("sdk llama.cpp stream should complete");
        for (_, payload) in &mut actual {
            normalize_measurement_fields(payload);
        }
        for (_, payload) in &mut expected {
            normalize_measurement_fields(payload);
        }

        llama_cpp_server_handle
            .join()
            .expect("llama.cpp server thread should finish");
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn mlx_lm_delegated_generate_stream_endpoint_emits_sse_chunks() {
        let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_stream_server(
            1,
            vec![
                serde_json::json!({
                    "choices": [{"index": 0, "text": " hello", "finish_reason": null}],
                    "usage": null
                }),
                serde_json::json!({
                    "choices": [{"index": 0, "text": " world", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
                }),
            ],
            |payload| {
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello delegated stream".to_string()))
                );
            },
        );
        let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
        let (status, content_type, body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate/stream")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_text_http_request(
                    "hello delegated stream",
                    2,
                ))))
                .unwrap(),
        )
        .await;

        mlx_lm_server_handle
            .join()
            .expect("mlx-lm server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"));

        let events = parse_sse_events(&body);
        let response_event = events
            .iter()
            .find(|(name, _)| name == "response")
            .expect("SSE stream should contain a response event");
        assert_eq!(
            response_event
                .1
                .get("response")
                .and_then(|r| r.get("output_text"))
                .and_then(Value::as_str),
            Some(" hello world")
        );
    }

    #[tokio::test]
    async fn openai_completions_stream_endpoint_emits_openai_sse_chunks() {
        let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
            1,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello stream".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            },
        );
        let app = build_router(llama_cpp_server_state(llama_server_url));
        let (status, content_type, body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(
                    &sample_openai_completion_request("hello stream", Some(2), true),
                )))
                .unwrap(),
        )
        .await;
        llama_cpp_server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"));
        let payloads = parse_openai_sse_payloads(&body);
        assert_eq!(payloads.len(), 3);
        assert_eq!(
            payloads[0]
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("text"))
                .and_then(Value::as_str),
            Some("hello")
        );
        assert_eq!(
            payloads[1]
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("text"))
                .and_then(Value::as_str),
            Some(" world")
        );
        assert_eq!(
            payloads[2]
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(Value::as_str),
            Some("length")
        );
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn openai_chat_completions_stream_endpoint_emits_openai_sse_chunks() {
        let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
            1,
            vec![
                serde_json::json!({
                    "content": "chat",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " stream",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String(expected_qwen_chatml_user_prompt(
                        "hello chat stream"
                    )))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            },
        );
        let app = build_router(llama_cpp_server_state(llama_server_url));
        let (status, content_type, body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_openai_chat_request(
                    "hello chat stream",
                    Some(2),
                    true,
                ))))
                .unwrap(),
        )
        .await;
        llama_cpp_server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"));
        let payloads = parse_openai_sse_payloads(&body);
        assert_eq!(payloads.len(), 3);
        assert_eq!(
            payloads[0]
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("delta"))
                .and_then(|delta| delta.get("role"))
                .and_then(Value::as_str),
            Some("assistant")
        );
        assert_eq!(
            payloads[0]
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("delta"))
                .and_then(|delta| delta.get("content"))
                .and_then(Value::as_str),
            Some("chat")
        );
        assert_eq!(
            payloads[1]
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("delta"))
                .and_then(|delta| delta.get("content"))
                .and_then(Value::as_str),
            Some(" stream")
        );
        assert_eq!(
            payloads[2]
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(Value::as_str),
            Some("length")
        );
    }

    #[tokio::test]
    async fn openai_chat_stream_endpoint_emits_sse_for_mlx_lm_delegated() {
        let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_stream_server(
            1,
            vec![
                serde_json::json!({
                    "choices": [{"index": 0, "text": " hello", "finish_reason": null}],
                    "usage": null
                }),
                serde_json::json!({
                    "choices": [{"index": 0, "text": " world", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
                }),
            ],
            |payload| {
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String(expected_qwen_chatml_user_prompt(
                        "hello mlx-lm chat stream"
                    )))
                );
            },
        );
        let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
        let (status, content_type, body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_openai_chat_request(
                    "hello mlx-lm chat stream",
                    Some(2),
                    true,
                ))))
                .unwrap(),
        )
        .await;

        mlx_lm_server_handle
            .join()
            .expect("mlx-lm server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"));

        let payloads = parse_openai_sse_payloads(&body);
        let text_chunks: Vec<&str> = payloads
            .iter()
            .filter_map(|p| {
                p.get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
            })
            .collect();
        assert!(
            !text_chunks.is_empty(),
            "streaming response should have text chunks"
        );
    }

    #[tokio::test]
    async fn llama_cpp_cli_stream_endpoint_fails_closed() {
        let app = build_router(llama_cpp_state());
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate/stream")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_text_http_request(
                    "hello from stream",
                    2,
                ))))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            json.get("error")
                .and_then(|value| value.get("code"))
                .and_then(|value| value.as_str()),
            Some("invalid_request")
        );
    }

    #[tokio::test]
    async fn llama_cpp_stepwise_request_endpoints_share_sdk_lifecycle() {
        let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
            2,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&json!([1, 2, 3])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );
        let state = llama_cpp_server_state(llama_server_url);
        let app = build_router(state.clone());
        let request_body = sample_http_request(&[1, 2, 3], 2);
        let (status, submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&request_body)))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);
        let request_id = 1u64;

        let mut sdk_session = sdk_session_for_state(&state);
        sdk_session
            .submit_generate_with_request_id(request_id, sample_sdk_request(&[1, 2, 3], 2))
            .expect("sdk llama.cpp submit should succeed");
        let expected_submit = serde_json::to_value(
            sdk_session
                .request_report(request_id)
                .expect("sdk llama.cpp report should exist"),
        )
        .expect("sdk llama.cpp submit report should serialize");
        assert_eq!(submit_json, expected_submit);

        for _ in 0..4 {
            let expected_snapshot = serde_json::to_value(
                sdk_session
                    .request_report(request_id)
                    .expect("sdk llama.cpp request report should still exist"),
            )
            .expect("sdk llama.cpp snapshot should serialize");
            let (snapshot_status, snapshot_json) = json_response(
                &app,
                Request::builder()
                    .uri(format!("/v1/requests/{request_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(snapshot_status, StatusCode::OK);
            assert_eq!(snapshot_json, expected_snapshot);

            if expected_snapshot
                .get("state")
                .and_then(|value| value.as_str())
                == Some("finished")
            {
                break;
            }

            let expected_step = serde_json::to_value(
                sdk_session
                    .step_report()
                    .expect("sdk llama.cpp step should succeed"),
            )
            .expect("sdk llama.cpp step should serialize");
            let (step_status, step_json) = json_response(
                &app,
                Request::builder()
                    .method("POST")
                    .uri("/v1/step")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(step_status, StatusCode::OK);
            let mut step_json = step_json;
            let mut expected_step = expected_step;
            normalize_measurement_fields(&mut step_json);
            normalize_measurement_fields(&mut expected_step);
            assert_eq!(step_json, expected_step);
        }

        let expected_terminal = serde_json::to_value(
            sdk_session
                .request_report(request_id)
                .expect("sdk llama.cpp terminal request should exist"),
        )
        .expect("sdk llama.cpp terminal snapshot should serialize");
        let (terminal_status, terminal_json) = json_response(
            &app,
            Request::builder()
                .uri(format!("/v1/requests/{request_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(terminal_status, StatusCode::OK);
        assert_eq!(terminal_json, expected_terminal);

        llama_cpp_server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }

    #[test]
    fn openai_finish_reason_preserves_only_real_openai_terminal_labels() {
        assert_eq!(
            openai_finish_reason(Some(GenerateFinishReason::Stop)),
            Some("stop")
        );
        assert_eq!(
            openai_finish_reason(Some(GenerateFinishReason::MaxOutputTokens)),
            Some("length")
        );
        assert_eq!(
            openai_finish_reason(Some(GenerateFinishReason::ContentFilter)),
            Some("content_filter")
        );
        assert_eq!(
            openai_finish_reason(Some(GenerateFinishReason::Cancelled)),
            None
        );
        assert_eq!(
            openai_finish_reason(Some(GenerateFinishReason::Error)),
            None
        );
        assert_eq!(openai_finish_reason(None), None);
    }

    #[tokio::test]
    async fn llama_cpp_stepwise_request_endpoints_aggregate_multiple_active_requests() {
        let expected_prompts = vec![json!([1, 2, 3]), json!([7, 8, 9])];
        let expected_prompts_for_request = expected_prompts.clone();
        let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
            4,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            move |payload| {
                let prompt = payload.get("prompt").expect("prompt should be present");
                assert!(
                    expected_prompts_for_request
                        .iter()
                        .any(|candidate| prompt == candidate)
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );
        let state = llama_cpp_server_state(llama_server_url);
        let app = build_router(state.clone());

        let (first_submit_status, first_submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_http_request(
                    &[1, 2, 3],
                    2,
                ))))
                .unwrap(),
        )
        .await;
        let (second_submit_status, second_submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_http_request(
                    &[7, 8, 9],
                    2,
                ))))
                .unwrap(),
        )
        .await;
        assert_eq!(first_submit_status, StatusCode::CREATED);
        assert_eq!(second_submit_status, StatusCode::CREATED);

        let first_request_id = first_submit_json
            .get("request_id")
            .and_then(|value| value.as_u64())
            .expect("first request_id should exist");
        let second_request_id = second_submit_json
            .get("request_id")
            .and_then(|value| value.as_u64())
            .expect("second request_id should exist");

        let mut sdk_session = sdk_session_for_state(&state);
        sdk_session
            .submit_generate_with_request_id(first_request_id, sample_sdk_request(&[1, 2, 3], 2))
            .expect("first sdk llama.cpp submit should succeed");
        sdk_session
            .submit_generate_with_request_id(second_request_id, sample_sdk_request(&[7, 8, 9], 2))
            .expect("second sdk llama.cpp submit should succeed");

        for _ in 0..2 {
            let expected_step = serde_json::to_value(
                sdk_session
                    .step_report()
                    .expect("sdk llama.cpp aggregated step should succeed"),
            )
            .expect("sdk llama.cpp step should serialize");
            let (step_status, step_json) = json_response(
                &app,
                Request::builder()
                    .method("POST")
                    .uri("/v1/step")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(step_status, StatusCode::OK);
            let mut step_json = step_json;
            let mut expected_step = expected_step;
            normalize_measurement_fields(&mut step_json);
            normalize_measurement_fields(&mut expected_step);
            assert_eq!(step_json, expected_step);

            for request_id in [first_request_id, second_request_id] {
                let expected_snapshot = serde_json::to_value(
                    sdk_session
                        .request_report(request_id)
                        .expect("sdk llama.cpp snapshot should exist"),
                )
                .expect("sdk llama.cpp snapshot should serialize");
                let (snapshot_status, snapshot_json) = json_response(
                    &app,
                    Request::builder()
                        .uri(format!("/v1/requests/{request_id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await;
                assert_eq!(snapshot_status, StatusCode::OK);
                assert_eq!(snapshot_json, expected_snapshot);
            }
        }

        llama_cpp_server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }

    #[tokio::test]
    async fn llama_cpp_cancel_endpoint_surfaces_cancelled_snapshot() {
        let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
            1,
            vec![serde_json::json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            })],
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&json!([7, 8, 9])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            },
        );
        let app = build_router(llama_cpp_server_state(llama_server_url));
        let (_, submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&sample_http_request(
                    &[7, 8, 9],
                    2,
                ))))
                .unwrap(),
        )
        .await;
        let request_id = submit_json
            .get("request_id")
            .and_then(|value| value.as_u64())
            .expect("request_id should exist");

        let (status, cancel_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri(format!("/v1/requests/{request_id}/cancel"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            cancel_json.get("state").and_then(|value| value.as_str()),
            Some("cancelled")
        );
        assert_eq!(
            cancel_json
                .get("cancel_requested")
                .and_then(|value| value.as_bool()),
            Some(true)
        );

        llama_cpp_server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }
}
