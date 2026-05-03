use std::convert::Infallible;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ax_engine_sdk::{
    classify_native_gguf_export_failure_message, CompatibilityBackendError, EngineSession,
    EngineSessionConfig, EngineSessionError, EngineStepReport, GenerateFinishReason,
    GenerateRequest, GenerateResponse, GenerateSampling, GenerateStreamEvent, GenerateStreamState,
    NativeGgufExportFailureKind, RuntimeReport, SessionRequestReport,
    StatelessGenerateContext,
};
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::info;
use tracing_subscriber::EnvFilter;

mod args;

use args::ServerArgs;

const MAX_REQUEST_BODY_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone)]
struct AppState {
    model_id: Arc<String>,
    session_config: Arc<EngineSessionConfig>,
    stateless_generate_context: Arc<StatelessGenerateContext>,
    runtime_report: RuntimeReport,
    request_session: Arc<Mutex<EngineSession>>,
    next_request_id: Arc<AtomicU64>,
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
    code: &'static str,
    message: String,
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
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    metadata: Option<String>,
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tracing_enabled = init_tracing();

    let args = ServerArgs::parse();
    let bind_address = args.bind_address();
    let session_config = args
        .session_config()
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    let session = EngineSession::new(session_config.clone())?;
    let state = build_app_state(args.model_id.clone(), session)?;
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;

    if tracing_enabled {
        info!(
            bind_address = %bind_address,
            model_id = %args.model_id,
            support_tier = ?args.support_tier,
            "ax-engine-server preview listening"
        );
    } else {
        eprintln!(
            "ax-engine-server preview listening on http://{} model_id={} support_tier={:?}",
            bind_address, args.model_id, args.support_tier
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

    Ok(AppState {
        model_id: Arc::new(model_id),
        session_config: Arc::new(session_config),
        stateless_generate_context,
        runtime_report,
        request_session: Arc::new(Mutex::new(session)),
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

    let request_id = allocate_request_id(&state);
    let stateless_generate_context = Arc::clone(&state.stateless_generate_context);
    let request = build_generate_request(&state, request);
    let response = run_blocking_session_task(move || {
        stateless_generate_context.generate_with_request_id(request_id, request)
    })
    .await?;

    Ok(Json(response))
}

async fn openai_completions(
    State(state): State<AppState>,
    Json(request): Json<OpenAiCompletionHttpRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    validate_openai_compatibility_backend(&state)?;
    validate_model(&state, request.model.as_deref())?;
    let request = build_openai_completion_request(&state, request)?;

    if request.stream {
        return stream_openai_request(
            state,
            request.generate_request,
            OpenAiStreamKind::Completion,
        )
        .await;
    }

    let request_id = allocate_request_id(&state);
    let stateless_generate_context = Arc::clone(&state.stateless_generate_context);
    let response = run_blocking_session_task(move || {
        stateless_generate_context.generate_with_request_id(request_id, request.generate_request)
    })
    .await?;
    let payload = openai_completion_response(&response, openai_completion_id(request_id));

    Ok(Json(payload).into_response())
}

async fn openai_chat_completions(
    State(state): State<AppState>,
    Json(request): Json<OpenAiChatCompletionHttpRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    validate_openai_compatibility_backend(&state)?;
    validate_model(&state, request.model.as_deref())?;
    let request = build_openai_chat_request(&state, request)?;

    if request.stream {
        return stream_openai_request(
            state,
            request.generate_request,
            OpenAiStreamKind::ChatCompletion,
        )
        .await;
    }

    let request_id = allocate_request_id(&state);
    let stateless_generate_context = Arc::clone(&state.stateless_generate_context);
    let response = run_blocking_session_task(move || {
        stateless_generate_context.generate_with_request_id(request_id, request.generate_request)
    })
    .await?;
    let payload = openai_chat_completion_response(&response, openai_chat_completion_id(request_id));

    Ok(Json(payload).into_response())
}

async fn generate_stream(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<
    Sse<UnboundedReceiverStream<Result<Event, Infallible>>>,
    (StatusCode, Json<ErrorResponse>),
> {
    validate_model(&state, request.model.as_deref())?;

    let request_id = allocate_request_id(&state);
    let request = build_generate_request(&state, request);
    if state
        .stateless_generate_context
        .supports_compatibility_streaming()
    {
        let stateless_generate_context = Arc::clone(&state.stateless_generate_context);
        let drive_context = Arc::clone(&stateless_generate_context);
        let stream_state = run_blocking_session_task(move || {
            stateless_generate_context.stream_state_with_request_id(request_id, request)
        })
        .await?;

        let (tx, rx) = mpsc::unbounded_channel();
        tokio::task::spawn_blocking(move || {
            drive_stateless_generate_stream_state(drive_context, stream_state, tx);
        });

        return Ok(Sse::new(UnboundedReceiverStream::new(rx)).keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(5))
                .text("keep-alive"),
        ));
    }

    let session_config = state.session_config.as_ref().clone();
    let (session, stream_state) = run_blocking_session_task(move || {
        let mut session = EngineSession::new(session_config)?;
        let stream_state = session.stream_generate_state_with_request_id(request_id, request)?;
        Ok((session, stream_state))
    })
    .await?;

    let (tx, rx) = mpsc::unbounded_channel();

    tokio::task::spawn_blocking(move || {
        let mut session = session;
        drive_generate_stream_state(&mut session, stream_state, tx);
    });

    Ok(Sse::new(UnboundedReceiverStream::new(rx)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keep-alive"),
    ))
}

async fn stream_openai_request(
    state: AppState,
    request: GenerateRequest,
    stream_kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let request_id = allocate_request_id(&state);
    if state
        .stateless_generate_context
        .supports_compatibility_streaming()
    {
        let stateless_generate_context = Arc::clone(&state.stateless_generate_context);
        let drive_context = Arc::clone(&stateless_generate_context);
        let stream_state = run_blocking_session_task(move || {
            stateless_generate_context.stream_state_with_request_id(request_id, request)
        })
        .await?;

        let (tx, rx) = mpsc::unbounded_channel();
        tokio::task::spawn_blocking(move || {
            drive_stateless_openai_stream_state(drive_context, stream_state, tx, stream_kind);
        });

        return Ok(Sse::new(UnboundedReceiverStream::new(rx))
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(5))
                    .text("keep-alive"),
            )
            .into_response());
    }

    let session_config = state.session_config.as_ref().clone();
    let (session, stream_state) = run_blocking_session_task(move || {
        let mut session = EngineSession::new(session_config)?;
        let stream_state = session.stream_generate_state_with_request_id(request_id, request)?;
        Ok((session, stream_state))
    })
    .await?;

    let (tx, rx) = mpsc::unbounded_channel();
    tokio::task::spawn_blocking(move || {
        let mut session = session;
        drive_openai_stream_state(&mut session, stream_state, tx, stream_kind);
    });

    Ok(Sse::new(UnboundedReceiverStream::new(rx))
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(5))
                .text("keep-alive"),
        )
        .into_response())
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

fn drive_generate_stream_state(
    session: &mut EngineSession,
    mut state: GenerateStreamState,
    tx: mpsc::UnboundedSender<Result<Event, Infallible>>,
) {
    loop {
        match session.next_stream_event(&mut state) {
            Ok(Some(event)) => {
                if !send_sdk_stream_event(&tx, event) {
                    return;
                }
            }
            Ok(None) => return,
            Err(error) => {
                let (_, Json(error)) = map_session_error(error);
                send_stream_error(&tx, error);
                return;
            }
        }
    }
}

fn drive_stateless_generate_stream_state(
    context: Arc<StatelessGenerateContext>,
    mut state: GenerateStreamState,
    tx: mpsc::UnboundedSender<Result<Event, Infallible>>,
) {
    loop {
        match context.next_stream_event(&mut state) {
            Ok(Some(event)) => {
                if !send_sdk_stream_event(&tx, event) {
                    return;
                }
            }
            Ok(None) => return,
            Err(error) => {
                let (_, Json(error)) = map_session_error(error);
                send_stream_error(&tx, error);
                return;
            }
        }
    }
}

fn drive_openai_stream_state(
    session: &mut EngineSession,
    mut state: GenerateStreamState,
    tx: mpsc::UnboundedSender<Result<Event, Infallible>>,
    stream_kind: OpenAiStreamKind,
) {
    let mut chat_role_emitted = false;

    loop {
        match session.next_stream_event(&mut state) {
            Ok(Some(event)) => {
                if !send_openai_stream_event(&tx, event, stream_kind, &mut chat_role_emitted) {
                    return;
                }
            }
            Ok(None) => {
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
                return;
            }
            Err(error) => {
                let (_, Json(error)) = map_session_error(error);
                send_openai_stream_error(&tx, error);
                return;
            }
        }
    }
}

fn drive_stateless_openai_stream_state(
    context: Arc<StatelessGenerateContext>,
    mut state: GenerateStreamState,
    tx: mpsc::UnboundedSender<Result<Event, Infallible>>,
    stream_kind: OpenAiStreamKind,
) {
    let mut chat_role_emitted = false;

    loop {
        match context.next_stream_event(&mut state) {
            Ok(Some(event)) => {
                if !send_openai_stream_event(&tx, event, stream_kind, &mut chat_role_emitted) {
                    return;
                }
            }
            Ok(None) => {
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
                return;
            }
            Err(error) => {
                let (_, Json(error)) = map_session_error(error);
                send_openai_stream_error(&tx, error);
                return;
            }
        }
    }
}

fn send_sdk_stream_event(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    event: GenerateStreamEvent,
) -> bool {
    let event_name = event.event_name();

    match event {
        GenerateStreamEvent::Request(payload) => send_stream_event(tx, event_name, &payload),
        GenerateStreamEvent::Step(payload) => send_stream_event(tx, event_name, &payload),
        GenerateStreamEvent::Response(payload) => send_stream_event(tx, event_name, &payload),
    }
}

fn send_openai_stream_event(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
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
                    id: openai_completion_id(payload.request.request_id),
                    object: "text_completion.chunk",
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
                    id: openai_chat_completion_id(payload.request.request_id),
                    object: "chat.completion.chunk",
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
                    id: openai_completion_id(payload.response.request_id),
                    object: "text_completion.chunk",
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
                    id: openai_chat_completion_id(payload.response.request_id),
                    object: "chat.completion.chunk",
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

fn send_stream_event<T: Serialize>(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    event_name: &str,
    payload: &T,
) -> bool {
    match serde_json::to_string(payload) {
        Ok(data) => tx
            .send(Ok(Event::default().event(event_name).data(data)))
            .is_ok(),
        Err(error) => {
            send_stream_error(
                tx,
                ErrorResponse {
                    error: ErrorBody {
                        code: "engine_error",
                        message: format!("failed to serialize {event_name} event: {error}"),
                    },
                },
            );
            false
        }
    }
}

fn send_stream_error(tx: &mpsc::UnboundedSender<Result<Event, Infallible>>, error: ErrorResponse) {
    let payload = serde_json::to_string(&error).unwrap_or_else(|_| {
        "{\"error\":{\"code\":\"engine_error\",\"message\":\"failed to serialize stream error\"}}"
            .to_string()
    });
    let _ = tx.send(Ok(Event::default().event("error").data(payload)));
    let _ = tx.send(Ok(Event::default().data("[DONE]")));
}

fn send_openai_stream_chunk<T: Serialize>(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    payload: &T,
) -> bool {
    match serde_json::to_string(payload) {
        Ok(data) => tx.send(Ok(Event::default().data(data))).is_ok(),
        Err(error) => {
            send_openai_stream_error(
                tx,
                ErrorResponse {
                    error: ErrorBody {
                        code: "engine_error",
                        message: format!("failed to serialize OpenAI stream chunk: {error}"),
                    },
                },
            );
            false
        }
    }
}

fn send_openai_stream_error(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    error: ErrorResponse,
) {
    let payload = serde_json::to_string(&error).unwrap_or_else(|_| {
        "{\"error\":{\"code\":\"engine_error\",\"message\":\"failed to serialize stream error\"}}"
            .to_string()
    });
    let _ = tx.send(Ok(Event::default().event("error").data(payload)));
    let _ = tx.send(Ok(Event::default().data("[DONE]")));
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
    GenerateRequest {
        model_id: state.model_id.to_string(),
        input_tokens: request.input_tokens,
        input_text: request.input_text,
        max_output_tokens: request.max_output_tokens.unwrap_or(256),
        sampling: request.sampling.unwrap_or_default(),
        metadata: request.metadata,
    }
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
        generate_request: GenerateRequest {
            model_id: state.model_id.to_string(),
            input_tokens,
            input_text,
            max_output_tokens,
            sampling: GenerateSampling {
                temperature: request.temperature.unwrap_or(0.0),
                top_p: request.top_p.unwrap_or(1.0),
                top_k: 0,
                repetition_penalty: 1.0,
                seed: request.seed.unwrap_or(0),
                deterministic: None,
            },
            metadata: request.metadata,
        },
        stream: request.stream,
    })
}

fn build_openai_chat_request(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = require_openai_max_tokens(request.max_tokens)?;
    let input_text = render_openai_chat_prompt(&request.messages)?;

    Ok(OpenAiBuiltRequest {
        generate_request: GenerateRequest {
            model_id: state.model_id.to_string(),
            input_tokens: Vec::new(),
            input_text: Some(input_text),
            max_output_tokens,
            sampling: GenerateSampling {
                temperature: request.temperature.unwrap_or(0.0),
                top_p: request.top_p.unwrap_or(1.0),
                top_k: 0,
                repetition_penalty: 1.0,
                seed: request.seed.unwrap_or(0),
                deterministic: None,
            },
            metadata: request.metadata,
        },
        stream: request.stream,
    })
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
    messages: &[OpenAiChatMessage],
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    if messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "chat.completions requires at least one message".to_string(),
        ));
    }

    let mut prompt = String::new();
    for message in messages {
        let content = render_openai_chat_content(&message.content)?;
        prompt.push_str(message.role.trim());
        prompt.push_str(": ");
        prompt.push_str(&content);
        prompt.push('\n');
    }
    prompt.push_str("assistant:");
    Ok(prompt)
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
        Some(GenerateFinishReason::Cancelled) | Some(GenerateFinishReason::Error) | None => None,
    }
}

fn validate_openai_compatibility_backend(
    state: &AppState,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if state.runtime_report.selected_backend.is_native() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "OpenAI-compatible endpoints require a compatibility backend; native backends use tokenized input".to_string(),
        ));
    }
    Ok(())
}

fn openai_completion_id(request_id: u64) -> String {
    format!("cmpl-{request_id}")
}

fn openai_chat_completion_id(request_id: u64) -> String {
    format!("chatcmpl-{request_id}")
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
        | EngineSessionError::NativeBackendRequiresTokenizedInput
        | EngineSessionError::InvalidMaxBatchTokens
        | EngineSessionError::InvalidRequestId
        | EngineSessionError::UnsupportedSupportTier
        | EngineSessionError::CompatibilityBackendDoesNotSupportLifecycle { .. }
        | EngineSessionError::StatelessStreamRequiresCompatibilityBackend { .. }
        | EngineSessionError::Compatibility(CompatibilityBackendError::StreamingNotSupported {
            ..
        })
        | EngineSessionError::RequestDidNotTerminate { .. }
        | EngineSessionError::MissingRequestSnapshot { .. } => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
        ),
        EngineSessionError::Compatibility(CompatibilityBackendError::MissingInputText {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::MissingPromptInput {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::UnsupportedTokenPrompt {
            ..
        })
        | EngineSessionError::Compatibility(
            CompatibilityBackendError::UnsupportedSamplingOption { .. },
        )
        | EngineSessionError::Compatibility(CompatibilityBackendError::AmbiguousPromptInput {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::BackendConfigMismatch {
            ..
        }) => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
        ),
        EngineSessionError::BackendContract(_)
        | EngineSessionError::MissingCompatibilityBackendConfig { .. }
        | EngineSessionError::CompatibilityBackendConfigMismatch { .. }
        | EngineSessionError::CompatibilityFallbackMustUseLlamaCpp { .. }
        | EngineSessionError::CompatibilityFallbackRequiresNonStrictPolicy
        | EngineSessionError::CompatibilityStreamEndedBeforeStop { .. }
        | EngineSessionError::CompatibilityFallbackFailed { .. }
        | EngineSessionError::NativeStartupFallbackFailed { .. }
        | EngineSessionError::NativeRuntimeArtifactsRequired
        | EngineSessionError::Compatibility(CompatibilityBackendError::CommandLaunch { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::CommandFailed { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::CommandTimedOut {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::NonUtf8Output { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::SerializeRequestJson {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::HttpRequest { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::HttpStatus { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::HttpResponseRead {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::InvalidResponseJson {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::EmptyChoicesInResponse {
            ..
        })
        | EngineSessionError::UnsupportedHostHardware { .. }
        | EngineSessionError::NativeModelAutoConvert { .. } => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "unsupported_host",
            error.to_string(),
        ),
        EngineSessionError::NativeModelGgufExportFailed { message, .. } => {
            map_native_gguf_export_failed_response(message)
        }
        EngineSessionError::RequestReportInvariantViolation { .. }
        | EngineSessionError::StreamEndedWithoutResponse { .. }
        | EngineSessionError::Core(_)
        | EngineSessionError::MetalRuntime(_)
        | EngineSessionError::AxNativeNotSupported
        | EngineSessionError::NativeModelGgufExportLaunch { .. } => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "engine_error",
            error.to_string(),
        ),
    }
}

fn map_native_gguf_export_failed_response(message: String) -> (StatusCode, Json<ErrorResponse>) {
    match classify_native_gguf_export_failure_message(&message) {
        NativeGgufExportFailureKind::UnsupportedModel => {
            error_response(StatusCode::BAD_REQUEST, "invalid_request", message)
        }
        NativeGgufExportFailureKind::MissingPythonDependency
        | NativeGgufExportFailureKind::ExportFailed => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "engine_error", message)
        }
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
            error: ErrorBody { code, message },
        }),
    )
}

fn request_not_found_response(request_id: u64) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::NOT_FOUND,
        "request_not_found",
        format!("request {request_id} is missing from preview session state"),
    )
}

fn missing_request_after_submit(request_id: u64) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "engine_error",
        format!("request {request_id} is missing immediately after submission"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{header, Request};
    use serde_json::Value;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    fn sample_http_request(input_tokens: &[u32], max_output_tokens: u32) -> Value {
        json!({
            "model": "qwen3_dense",
            "input_tokens": input_tokens,
            "max_output_tokens": max_output_tokens
        })
    }

    fn sample_sdk_request(input_tokens: &[u32], max_output_tokens: u32) -> GenerateRequest {
        GenerateRequest {
            model_id: "qwen3_dense".to_string(),
            input_tokens: input_tokens.to_vec(),
            input_text: None,
            max_output_tokens,
            sampling: GenerateSampling::default(),
            metadata: None,
        }
    }

    fn sample_text_http_request(input_text: &str, max_output_tokens: u32) -> Value {
        json!({
            "model": "qwen3_dense",
            "input_text": input_text,
            "max_output_tokens": max_output_tokens
        })
    }

    fn sample_openai_completion_request(prompt: &str, max_tokens: u32, stream: bool) -> Value {
        json!({
            "model": "qwen3_dense",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream
        })
    }

    #[test]
    fn map_session_error_treats_unsupported_native_gguf_export_as_bad_request() {
        let (status, Json(response)) =
            map_session_error(EngineSessionError::NativeModelGgufExportFailed {
                gguf_path: "/tmp/model.gguf".into(),
                message: "status=unsupported blockers=moe_expert_tensors_present".to_string(),
            });

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response.error.code, "invalid_request");
        assert!(response
            .error
            .message
            .contains("moe_expert_tensors_present"));
    }

    #[test]
    fn map_session_error_treats_missing_gguf_python_dependency_as_engine_error() {
        let (status, Json(response)) =
            map_session_error(EngineSessionError::NativeModelGgufExportFailed {
                gguf_path: "/tmp/model.gguf".into(),
                message: "status=error reason=missing_python_dependency".to_string(),
            });

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.error.code, "engine_error");
        assert!(response.error.message.contains("missing_python_dependency"));
    }

    fn sample_openai_chat_request(message: &str, max_tokens: u32, stream: bool) -> Value {
        json!({
            "model": "qwen3_dense",
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "max_tokens": max_tokens,
            "stream": stream
        })
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
            model_id: "qwen3_dense".to_string(),
            deterministic: true,
            max_batch_tokens: 2048,
            cache_group_id: 0,
            block_size_tokens: 16,
            total_blocks: 1024,
            mlx: false,
            mlx_native: false,
            support_tier: args::PreviewSupportTier::Compatibility,
            compat_backend: args::PreviewCompatibilityBackend::LlamaCpp,
            compat_cli_path: "llama-cli".to_string(),
            compat_model_path: None,
            compat_server_url: None,
            llama_fallback_cli_path: "llama-cli".to_string(),
            llama_fallback_model_path: None,
            llama_fallback_server_url: None,
            native_runtime_artifacts_dir: None,
            native_model_artifacts_dir: None,
        }
    }

    fn test_state() -> AppState {
        let args = ServerArgs {
            support_tier: args::PreviewSupportTier::NativePreview,
            ..base_server_args()
        };
        let mut session_config = args.session_config().expect("session config should build");
        session_config.allow_deterministic_native_fallback = true;
        let session = EngineSession::new(session_config.clone()).expect("session should build");
        build_app_state(args.model_id.clone(), session).expect("app state should build")
    }

    fn fake_compat_script() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-engine-server-compat-{unique}.py"));
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

    fn compatibility_state() -> AppState {
        let script_path = fake_compat_script();
        let model_path = std::env::temp_dir().join("ax-engine-server-compat-model.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");
        let args = ServerArgs {
            compat_cli_path: script_path.display().to_string(),
            compat_model_path: Some(model_path),
            ..base_server_args()
        };

        let session_config = args.session_config().expect("session config should build");
        let session = EngineSession::new(session_config.clone()).expect("session should build");
        build_app_state(args.model_id.clone(), session).expect("app state should build")
    }

    fn compatibility_server_state(server_url: String) -> AppState {
        compatibility_server_state_for_backend(
            args::PreviewCompatibilityBackend::LlamaCpp,
            server_url,
        )
    }

    fn compatibility_server_state_for_backend(
        compat_backend: args::PreviewCompatibilityBackend,
        server_url: String,
    ) -> AppState {
        let args = ServerArgs {
            compat_backend,
            compat_server_url: Some(server_url),
            ..base_server_args()
        };

        let session_config = args.session_config().expect("session config should build");
        let session = EngineSession::new(session_config.clone()).expect("session should build");
        build_app_state(args.model_id.clone(), session).expect("app state should build")
    }

    fn fake_python_mlx_cli_script() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("python3-ax-engine-server-mlx-{unique}.py"));
        let script = r#"#!/usr/bin/env python3
from __future__ import annotations

import sys

args = sys.argv[1:]
assert args[:2] == ["-m", "mlx_lm.generate"], args
model = args[args.index("--model") + 1]
prompt = args[args.index("--prompt") + 1]
assert args[args.index("--max-tokens") + 1] == "2"
assert args[args.index("--temp") + 1] == "0"
assert args[args.index("--top-p") + 1] == "1"
assert args[args.index("--top-k") + 1] == "0"
assert args[args.index("--seed") + 1] == "0"
assert args[args.index("--verbose") + 1] == "false"
assert "--ignore-chat-template" in args
sys.stdout.write(f"mlx::{model}::{prompt}")
"#;

        fs::write(&path, script).expect("fake mlx script should be written");
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

    fn compatibility_mlx_cli_state() -> (AppState, std::path::PathBuf) {
        let script_path = fake_python_mlx_cli_script();
        let model_path = std::env::temp_dir().join("ax-engine-server-compat-model-mlx");
        fs::write(&model_path, "fake mlx model").expect("fake mlx model should be written");
        let args = ServerArgs {
            mlx: true,
            compat_backend: args::PreviewCompatibilityBackend::Mlx,
            compat_cli_path: script_path.display().to_string(),
            compat_model_path: Some(model_path.clone()),
            llama_fallback_model_path: Some(
                std::env::temp_dir().join("ax-engine-server-compat-fallback.gguf"),
            ),
            ..base_server_args()
        };

        let session_config = args.session_config().expect("session config should build");
        let session = EngineSession::new(session_config.clone()).expect("session should build");
        (
            build_app_state(args.model_id.clone(), session).expect("app state should build"),
            model_path,
        )
    }

    fn spawn_compat_completion_server(
        response_body: String,
        assert_request: impl FnOnce(Value) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("request should arrive");
            let request = read_http_request(&mut stream);
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
                .expect("request should include header terminator");
            let body =
                String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
            let payload: Value = serde_json::from_str(&body).expect("request body should be json");
            assert_request(payload);

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream
                .write_all(response.as_bytes())
                .expect("response should write");
        });

        (format!("http://{address}"), handle)
    }

    fn spawn_compat_completion_stream_server(
        expected_requests: usize,
        chunks: Vec<Value>,
        assert_request: impl Fn(Value) + Send + Sync + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept().expect("request should arrive");
                let request = read_http_request(&mut stream);
                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4)
                    .expect("request should include header terminator");
                let body =
                    String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
                let payload: Value =
                    serde_json::from_str(&body).expect("request body should be json");
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

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("response should write");
            }
        });

        (format!("http://{address}"), handle)
    }

    fn spawn_single_json_completion_server(
        endpoint_path: &'static str,
        response_body: String,
        assert_request: impl FnOnce(Value) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("request should arrive");
            let request = read_http_request(&mut stream);
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
                .expect("request should include header terminator");
            let headers =
                String::from_utf8(request[..header_end].to_vec()).expect("headers should be utf8");
            let request_line = headers.lines().next().expect("request line should exist");
            assert!(
                request_line.contains(endpoint_path),
                "request path should target {endpoint_path}, got {request_line}"
            );
            let body =
                String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
            let payload: Value = serde_json::from_str(&body).expect("request body should be json");
            assert_request(payload);

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream
                .write_all(response.as_bytes())
                .expect("response should write");
        });

        (format!("http://{address}"), handle)
    }

    fn spawn_json_completion_stream_server(
        expected_requests: usize,
        chunks: Vec<Value>,
        assert_request: impl Fn(Value) + Send + Sync + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept().expect("request should arrive");
                let request = read_http_request(&mut stream);
                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4)
                    .expect("request should include header terminator");
                let headers = String::from_utf8(request[..header_end].to_vec())
                    .expect("headers should be utf8");
                let request_line = headers.lines().next().expect("request line should exist");
                assert!(
                    request_line.contains("/v1/completions"),
                    "request path should target /v1/completions, got {request_line}"
                );
                let body =
                    String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
                let payload: Value =
                    serde_json::from_str(&body).expect("request body should be json");
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

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("response should write");
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
    async fn health_endpoint_reports_preview_runtime() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("health request should succeed");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn runtime_endpoint_surfaces_backend_metadata() {
        let state = test_state();
        let expected_runtime =
            serde_json::to_value(state.runtime_report.clone()).expect("runtime should serialize");
        let app = build_router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/runtime")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .expect("runtime request should succeed");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("runtime response should be json");

        assert_eq!(
            json.get("model_id").and_then(|v| v.as_str()),
            Some("qwen3_dense")
        );
        assert_eq!(json.get("runtime"), Some(&expected_runtime));
    }

    #[tokio::test]
    async fn generate_endpoint_rejects_oversized_request_body() {
        let app = build_router(test_state());
        let oversized_body = serde_json::to_vec(&json!({
            "input_text": "x".repeat(MAX_REQUEST_BODY_BYTES),
            "max_output_tokens": 1
        }))
        .expect("request should serialize");
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/generate")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(oversized_body))
                    .unwrap(),
            )
            .await
            .expect("oversized request should receive a response");

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn generate_endpoint_runs_request_through_sdk() {
        let state = test_state();
        let app = build_router(state.clone());
        let request_body = sample_http_request(&[1, 2, 3], 2);
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::OK);

        let mut sdk_session = sdk_session_for_state(&state);
        let request_id = 1;
        sdk_session
            .submit_generate_with_request_id(request_id, sample_sdk_request(&[1, 2, 3], 2))
            .expect("sdk submit should succeed");
        let expected = serde_json::to_value(
            sdk_session
                .run_to_completion(request_id)
                .expect("sdk generate should complete"),
        )
        .expect("sdk response should serialize");

        assert_eq!(json, expected);
    }

    #[tokio::test]
    async fn compatibility_generate_endpoint_runs_text_request_through_sdk() {
        let (compat_server_url, compat_server_handle) = spawn_compat_completion_server(
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
        let state = compatibility_server_state(compat_server_url);
        let app = build_router(state.clone());
        let request_body = sample_text_http_request("hello from server", 2);
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await;
        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");

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
    async fn openai_completions_endpoint_translates_compatibility_response() {
        let (compat_server_url, compat_server_handle) = spawn_compat_completion_server(
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
        let app = build_router(compatibility_server_state(compat_server_url));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_openai_completion_request("hello openai", 2, false))
                        .unwrap(),
                ))
                .unwrap(),
        )
        .await;
        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");

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
    async fn openai_completions_endpoint_requires_max_tokens() {
        let app = build_router(compatibility_server_state("http://127.0.0.1:1".to_string()));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "qwen3_dense",
                        "prompt": "hello openai",
                        "stream": false
                    }))
                    .unwrap(),
                ))
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
        assert!(json
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("require max_tokens"));
    }

    #[tokio::test]
    async fn openai_chat_completions_endpoint_translates_compatibility_response() {
        let (compat_server_url, compat_server_handle) = spawn_single_json_completion_server(
            "/v1/completions",
            serde_json::json!({
                "choices": [{
                    "text": "chat::hello openai chat",
                    "finish_reason": "length"
                }]
            })
            .to_string(),
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String(
                        "user: hello openai chat\nassistant:".to_string()
                    ))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            },
        );
        let app = build_router(compatibility_server_state_for_backend(
            args::PreviewCompatibilityBackend::Vllm,
            compat_server_url,
        ));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_openai_chat_request("hello openai chat", 2, false))
                        .unwrap(),
                ))
                .unwrap(),
        )
        .await;
        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json.get("object").and_then(Value::as_str),
            Some("chat.completion")
        );
        assert_eq!(
            json.get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("role"))
                .and_then(Value::as_str),
            Some("assistant")
        );
        assert_eq!(
            json.get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str),
            Some("chat::hello openai chat")
        );
    }

    #[tokio::test]
    async fn openai_chat_completions_endpoint_requires_max_tokens() {
        let app = build_router(compatibility_server_state_for_backend(
            args::PreviewCompatibilityBackend::Vllm,
            "http://127.0.0.1:1".to_string(),
        ));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "qwen3_dense",
                        "messages": [
                            {
                                "role": "user",
                                "content": "hello openai chat"
                            }
                        ],
                        "stream": false
                    }))
                    .unwrap(),
                ))
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
        assert!(json
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("require max_tokens"));
    }

    #[tokio::test]
    async fn openai_completions_endpoint_reports_usage_from_backend_response() {
        // Verifies that when the upstream OpenAI-compatible backend returns a usage object,
        // the /v1/completions response carries accurate token counts rather than zeros.
        let (compat_server_url, compat_server_handle) = spawn_single_json_completion_server(
            "/v1/completions",
            serde_json::json!({
                "choices": [{"text": "hello usage", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10
                }
            })
            .to_string(),
            |_| {},
        );
        let app = build_router(compatibility_server_state_for_backend(
            args::PreviewCompatibilityBackend::Vllm,
            compat_server_url,
        ));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_openai_completion_request("prompt text", 4, false))
                        .unwrap(),
                ))
                .unwrap(),
        )
        .await;
        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json.get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(Value::as_u64),
            Some(7),
            "prompt_tokens should come from the backend usage object"
        );
        assert_eq!(
            json.get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(Value::as_u64),
            Some(3),
            "completion_tokens should come from the backend usage object"
        );
        assert_eq!(
            json.get("usage")
                .and_then(|u| u.get("total_tokens"))
                .and_then(Value::as_u64),
            Some(10),
            "total_tokens should be the sum"
        );
    }

    #[tokio::test]
    async fn compatibility_mlx_cli_generate_endpoint_runs_text_request_through_sdk() {
        let (state, model_path) = compatibility_mlx_cli_state();
        let app = build_router(state);
        let request_body = sample_text_http_request("hello mlx", 2);
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json.get("prompt_text").and_then(|value| value.as_str()),
            Some("hello mlx")
        );
        assert_eq!(
            json.get("output_text").and_then(|value| value.as_str()),
            Some(format!("mlx::{}::hello mlx", model_path.display()).as_str())
        );
        assert_eq!(
            json.get("route")
                .and_then(|route| route.get("execution_plan"))
                .and_then(|value| value.as_str()),
            Some("compatibility.mlx.blocking_cli")
        );
        assert_eq!(
            json.get("runtime")
                .and_then(|runtime| runtime.get("selected_backend"))
                .and_then(|value| value.as_str()),
            Some("mlx")
        );
    }

    #[tokio::test]
    async fn openai_completions_endpoint_omits_usage_when_token_counts_are_unknown() {
        let (state, _model_path) = compatibility_mlx_cli_state();
        let app = build_router(state);
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_openai_completion_request("hello mlx", 2, false))
                        .unwrap(),
                ))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert!(
            json.get("usage").is_none(),
            "usage should be omitted when AX has no authoritative token counts"
        );
    }

    #[tokio::test]
    async fn compatibility_stream_endpoint_runs_server_backed_stream_through_sdk() {
        let (compat_server_url, compat_server_handle) = spawn_compat_completion_stream_server(
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
        let state = compatibility_server_state(compat_server_url);
        let app = build_router(state.clone());
        let (status, content_type, body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_http_request(&[1, 2, 3], 2)).unwrap(),
                ))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"));

        let mut actual = parse_sse_events(&body);
        let mut sdk_session = sdk_session_for_state(&state);
        let mut expected = sdk_session
            .stream_generate_with_request_id(1, sample_sdk_request(&[1, 2, 3], 2))
            .expect("sdk compatibility stream should start")
            .map(|event| event.map(sdk_stream_payload))
            .collect::<Result<Vec<_>, _>>()
            .expect("sdk compatibility stream should complete");
        for (_, payload) in &mut actual {
            normalize_measurement_fields(payload);
        }
        for (_, payload) in &mut expected {
            normalize_measurement_fields(payload);
        }

        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn openai_completions_stream_endpoint_emits_openai_sse_chunks() {
        let (compat_server_url, compat_server_handle) = spawn_compat_completion_stream_server(
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
        let app = build_router(compatibility_server_state(compat_server_url));
        let (status, content_type, body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_openai_completion_request("hello stream", 2, true))
                        .unwrap(),
                ))
                .unwrap(),
        )
        .await;
        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");

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
        let (compat_server_url, compat_server_handle) = spawn_compat_completion_stream_server(
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
                    Some(&Value::String(
                        "user: hello chat stream\nassistant:".to_string()
                    ))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            },
        );
        let app = build_router(compatibility_server_state(compat_server_url));
        let (status, content_type, body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_openai_chat_request("hello chat stream", 2, true))
                        .unwrap(),
                ))
                .unwrap(),
        )
        .await;
        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");

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
    async fn openai_chat_completions_fail_closed_on_native_preview() {
        let app = build_router(test_state());
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_openai_chat_request("native should fail", 2, false))
                        .unwrap(),
                ))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            json.get("error")
                .and_then(|value| value.get("message"))
                .and_then(Value::as_str),
            Some("OpenAI-compatible endpoints require a compatibility backend; native backends use tokenized input")
        );
    }

    #[tokio::test]
    async fn compatibility_cli_stream_endpoint_fails_closed() {
        let app = build_router(compatibility_state());
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_text_http_request("hello from stream", 2)).unwrap(),
                ))
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
    async fn server_allocates_unique_request_ids_across_paths() {
        let app = build_router(test_state());
        let (submit_status, submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "qwen3_dense",
                        "input_tokens": [1, 2, 3],
                        "max_output_tokens": 2
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await;
        assert_eq!(submit_status, StatusCode::CREATED);
        let submit_id = submit_json
            .get("request_id")
            .and_then(|value| value.as_u64())
            .expect("submit request_id should exist");

        let (stream_status, content_type, stream_body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "qwen3_dense",
                        "input_tokens": [4, 5, 6],
                        "max_output_tokens": 2
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await;
        assert_eq!(stream_status, StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"));
        assert!(stream_body.contains("\"request_id\":2"));

        let (generate_status, generate_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "qwen3_dense",
                        "input_tokens": [7, 8, 9],
                        "max_output_tokens": 2
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await;
        assert_eq!(generate_status, StatusCode::OK);
        let generate_id = generate_json
            .get("request_id")
            .and_then(|value| value.as_u64())
            .expect("generate request_id should exist");

        assert_eq!(submit_id, 1);
        assert_eq!(generate_id, 3);
    }

    #[tokio::test]
    async fn generate_stream_endpoint_emits_sse_lifecycle_events() {
        let state = test_state();
        let app = build_router(state.clone());
        let request_body = sample_http_request(&[1, 2, 3], 2);
        let (status, content_type, body) = text_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/generate/stream")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"));

        let actual = parse_sse_events(&body);
        let mut sdk_session = sdk_session_for_state(&state);
        let expected = sdk_session
            .stream_generate_with_request_id(1, sample_sdk_request(&[1, 2, 3], 2))
            .expect("sdk stream should start")
            .map(|event| event.map(sdk_stream_payload))
            .collect::<Result<Vec<_>, _>>()
            .expect("sdk stream should complete");
        let mut actual = actual;
        let mut expected = expected;
        for (_, payload) in &mut actual {
            normalize_measurement_fields(payload);
        }
        for (_, payload) in &mut expected {
            normalize_measurement_fields(payload);
        }

        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn stepwise_request_endpoints_share_sdk_lifecycle() {
        let state = test_state();
        let app = build_router(state.clone());
        let request_body = sample_http_request(&[1, 2, 3], 2);
        let (status, submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);
        let request_id = 1u64;

        let mut sdk_session = sdk_session_for_state(&state);
        sdk_session
            .submit_generate_with_request_id(request_id, sample_sdk_request(&[1, 2, 3], 2))
            .expect("sdk submit should succeed");
        let expected_submit = serde_json::to_value(
            sdk_session
                .request_report(request_id)
                .expect("sdk report should exist"),
        )
        .expect("sdk submit report should serialize");
        assert_eq!(submit_json, expected_submit);

        for _ in 0..8 {
            let expected_snapshot = serde_json::to_value(
                sdk_session
                    .request_report(request_id)
                    .expect("sdk request report should still exist"),
            )
            .expect("sdk snapshot should serialize");
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

            let expected_step =
                serde_json::to_value(sdk_session.step_report().expect("sdk step should succeed"))
                    .expect("sdk step should serialize");
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
                .expect("sdk terminal request should exist"),
        )
        .expect("sdk terminal snapshot should serialize");
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
    }

    #[tokio::test]
    async fn compatibility_stepwise_request_endpoints_share_sdk_lifecycle() {
        let (compat_server_url, compat_server_handle) = spawn_compat_completion_stream_server(
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
        let state = compatibility_server_state(compat_server_url);
        let app = build_router(state.clone());
        let request_body = sample_http_request(&[1, 2, 3], 2);
        let (status, submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);
        let request_id = 1u64;

        let mut sdk_session = sdk_session_for_state(&state);
        sdk_session
            .submit_generate_with_request_id(request_id, sample_sdk_request(&[1, 2, 3], 2))
            .expect("sdk compatibility submit should succeed");
        let expected_submit = serde_json::to_value(
            sdk_session
                .request_report(request_id)
                .expect("sdk compatibility report should exist"),
        )
        .expect("sdk compatibility submit report should serialize");
        assert_eq!(submit_json, expected_submit);

        for _ in 0..4 {
            let expected_snapshot = serde_json::to_value(
                sdk_session
                    .request_report(request_id)
                    .expect("sdk compatibility request report should still exist"),
            )
            .expect("sdk compatibility snapshot should serialize");
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
                    .expect("sdk compatibility step should succeed"),
            )
            .expect("sdk compatibility step should serialize");
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
                .expect("sdk compatibility terminal request should exist"),
        )
        .expect("sdk compatibility terminal snapshot should serialize");
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

        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[tokio::test]
    async fn compatibility_openai_server_stepwise_request_endpoints_share_sdk_lifecycle() {
        let (compat_server_url, compat_server_handle) = spawn_json_completion_stream_server(
            2,
            vec![
                serde_json::json!({
                    "choices": [{
                        "text": "compat",
                        "finish_reason": null
                    }]
                }),
                serde_json::json!({
                    "choices": [{
                        "text": " stream",
                        "finish_reason": "length"
                    }]
                }),
            ],
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello compatibility".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            },
        );
        let state = compatibility_server_state_for_backend(
            args::PreviewCompatibilityBackend::Vllm,
            compat_server_url,
        );
        let app = build_router(state.clone());
        let request_body = sample_text_http_request("hello compatibility", 2);
        let (status, submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);
        let request_id = 1u64;

        let mut sdk_session = sdk_session_for_state(&state);
        sdk_session
            .submit_generate_with_request_id(
                request_id,
                GenerateRequest {
                    model_id: "qwen3_dense".to_string(),
                    input_tokens: Vec::new(),
                    input_text: Some("hello compatibility".to_string()),
                    max_output_tokens: 2,
                    sampling: GenerateSampling {
                        temperature: 0.0,
                        top_p: 1.0,
                        top_k: 0,
                        repetition_penalty: 1.0,
                        seed: 1234,
                        deterministic: Some(true),
                    },
                    metadata: None,
                },
            )
            .expect("sdk openai-compatible submit should succeed");
        let expected_submit = serde_json::to_value(
            sdk_session
                .request_report(request_id)
                .expect("sdk compatibility report should exist"),
        )
        .expect("sdk compatibility submit report should serialize");
        assert_eq!(submit_json, expected_submit);

        for _ in 0..4 {
            let expected_snapshot = serde_json::to_value(
                sdk_session
                    .request_report(request_id)
                    .expect("sdk compatibility request report should still exist"),
            )
            .expect("sdk compatibility snapshot should serialize");
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
                    .expect("sdk compatibility step should succeed"),
            )
            .expect("sdk compatibility step should serialize");
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
                .expect("sdk compatibility terminal request should exist"),
        )
        .expect("sdk compatibility terminal snapshot should serialize");
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

        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");
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
    async fn compatibility_stepwise_request_endpoints_aggregate_multiple_active_requests() {
        let expected_prompts = vec![json!([1, 2, 3]), json!([7, 8, 9])];
        let expected_prompts_for_request = expected_prompts.clone();
        let (compat_server_url, compat_server_handle) = spawn_compat_completion_stream_server(
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
                assert!(expected_prompts_for_request
                    .iter()
                    .any(|candidate| prompt == candidate));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );
        let state = compatibility_server_state(compat_server_url);
        let app = build_router(state.clone());

        let (first_submit_status, first_submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_http_request(&[1, 2, 3], 2)).unwrap(),
                ))
                .unwrap(),
        )
        .await;
        let (second_submit_status, second_submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&sample_http_request(&[7, 8, 9], 2)).unwrap(),
                ))
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
            .expect("first sdk compatibility submit should succeed");
        sdk_session
            .submit_generate_with_request_id(second_request_id, sample_sdk_request(&[7, 8, 9], 2))
            .expect("second sdk compatibility submit should succeed");

        for _ in 0..2 {
            let expected_step = serde_json::to_value(
                sdk_session
                    .step_report()
                    .expect("sdk compatibility aggregated step should succeed"),
            )
            .expect("sdk compatibility step should serialize");
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
                        .expect("sdk compatibility snapshot should exist"),
                )
                .expect("sdk compatibility snapshot should serialize");
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

        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[tokio::test]
    async fn compatibility_cancel_endpoint_surfaces_cancelled_snapshot() {
        let (compat_server_url, compat_server_handle) = spawn_compat_completion_stream_server(
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
        let app = build_router(compatibility_server_state(compat_server_url));
        let (_, submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "qwen3_dense",
                        "input_tokens": [7, 8, 9],
                        "max_output_tokens": 2
                    }))
                    .unwrap(),
                ))
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

        compat_server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[tokio::test]
    async fn cancel_endpoint_surfaces_cancelled_snapshot() {
        let app = build_router(test_state());
        let (_, submit_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "qwen3_dense",
                        "input_tokens": [7, 8, 9],
                        "max_output_tokens": 2
                    }))
                    .unwrap(),
                ))
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
    }

    #[tokio::test]
    async fn missing_request_snapshot_returns_not_found() {
        let app = build_router(test_state());
        let (status, json) = json_response(
            &app,
            Request::builder()
                .uri("/v1/requests/999")
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(
            json.get("error")
                .and_then(|error| error.get("code"))
                .and_then(|value| value.as_str()),
            Some("request_not_found")
        );
    }
}
