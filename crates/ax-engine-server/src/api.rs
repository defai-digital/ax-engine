use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::anyhow;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::{
    FinishReason, GenerateOutput, GenerateRequest, ModelInfo, PromptInput, RenderedChatMessage,
    RenderedChatRole, ServerEngine,
};

#[derive(Clone)]
pub(crate) struct AppState {
    engine: Arc<Mutex<ServerEngine>>,
    info: Arc<ModelInfo>,
    request_counter: Arc<AtomicU64>,
}

pub(crate) fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(create_completion))
        .route("/v1/chat/completions", post(create_chat_completion))
        .with_state(state)
}

impl AppState {
    pub(crate) fn new(engine: ServerEngine) -> Self {
        let info = Arc::new(engine.info());
        Self {
            engine: Arc::new(Mutex::new(engine)),
            info,
            request_counter: Arc::new(AtomicU64::new(1)),
        }
    }

    fn next_request_id(&self, prefix: &str) -> String {
        let counter = self.request_counter.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{counter:016x}")
    }
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    model: String,
    architecture: String,
    backend: String,
    context_length: usize,
    vocab_size: usize,
    support_note: Option<String>,
    routing: Option<String>,
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
    created: i64,
    owned_by: &'static str,
    root: String,
    backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    routing: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
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
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopSequences>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatCompletionMessage>,
    #[serde(default, alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
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
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopSequences>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    role: String,
    content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
}

#[derive(Debug, Deserialize)]
struct MessageContentPart {
    #[serde(rename = "type")]
    part_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChoice {
    index: usize,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize)]
struct ChatChunkChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct CompletionChunk {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
struct CompletionChunkChoice {
    index: usize,
    text: String,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
    kind: &'static str,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
        }
    }

    fn internal(err: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
            kind: "server_error",
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorEnvelope {
                error: ErrorBody {
                    message: self.message,
                    kind: self.kind,
                },
            }),
        )
            .into_response()
    }
}

async fn healthz(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model: state.info.id.clone(),
        architecture: state.info.architecture.clone(),
        backend: state.info.backend.clone(),
        context_length: state.info.context_length,
        vocab_size: state.info.vocab_size,
        support_note: state.info.support_note.clone(),
        routing: state.info.routing.clone(),
    })
}

async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: state.info.id.clone(),
            object: "model",
            created: state.info.created,
            owned_by: "ax-engine",
            root: state.info.id.clone(),
            backend: state.info.backend.clone(),
            routing: state.info.routing.clone(),
        }],
    })
}

async fn create_completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, AppError> {
    validate_model(request.model.as_deref(), &state.info)?;
    validate_choice_count(request.n)?;

    let request_id = state.next_request_id("cmpl");
    let created = unix_timestamp();
    let model_id = state.info.id.clone();
    let generate_request = GenerateRequest {
        prompt: PromptInput::Completion(request.prompt),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_k: request.top_k,
        top_p: request.top_p,
        min_p: request.min_p,
        repeat_penalty: request.repeat_penalty,
        repeat_last_n: request.repeat_last_n,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        stop_strings: normalize_stop_sequences(request.stop),
        seed: request.seed,
    };

    if request.stream {
        let response = completion_stream_response(
            state.engine.clone(),
            request_id,
            created,
            model_id,
            generate_request,
        );
        return Ok(response.into_response());
    }

    let engine = state.engine.clone();
    let output = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
        engine.generate(generate_request)
    })
    .await
    .map_err(|err| AppError::internal(anyhow!(err)))?
    .map_err(map_generation_error)?;

    Ok(Json(build_completion_response(
        request_id, created, model_id, output,
    ))
    .into_response())
}

async fn create_chat_completion(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    validate_model(request.model.as_deref(), &state.info)?;
    validate_choice_count(request.n)?;
    if request.messages.is_empty() {
        return Err(AppError::bad_request("messages must not be empty"));
    }

    let messages = normalize_chat_messages(&request.messages)?;
    let request_id = state.next_request_id("chatcmpl");
    let created = unix_timestamp();
    let model_id = state.info.id.clone();
    let generate_request = GenerateRequest {
        prompt: PromptInput::Chat(messages),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_k: request.top_k,
        top_p: request.top_p,
        min_p: request.min_p,
        repeat_penalty: request.repeat_penalty,
        repeat_last_n: request.repeat_last_n,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        stop_strings: normalize_stop_sequences(request.stop),
        seed: request.seed,
    };

    if request.stream {
        let response = chat_stream_response(
            state.engine.clone(),
            request_id,
            created,
            model_id,
            generate_request,
        );
        return Ok(response.into_response());
    }

    let engine = state.engine.clone();
    let output = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
        engine.generate(generate_request)
    })
    .await
    .map_err(|err| AppError::internal(anyhow!(err)))?
    .map_err(map_generation_error)?;

    Ok(Json(build_chat_completion_response(
        request_id, created, model_id, output,
    ))
    .into_response())
}

fn completion_stream_response(
    engine: Arc<Mutex<ServerEngine>>,
    request_id: String,
    created: i64,
    model_id: String,
    request: GenerateRequest,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<String>(32);
    tokio::task::spawn_blocking(move || {
        let generation = || -> anyhow::Result<GenerateOutput> {
            let mut engine = engine.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
            engine.stream(request, |chunk| {
                let payload = serde_json::to_string(&CompletionChunk {
                    id: request_id.clone(),
                    object: "text_completion",
                    created,
                    model: model_id.clone(),
                    choices: vec![CompletionChunkChoice {
                        index: 0,
                        text: chunk.to_string(),
                        finish_reason: None,
                    }],
                })?;
                tx.blocking_send(payload)
                    .map_err(|_| anyhow!("stream receiver dropped"))?;
                Ok(())
            })
        };

        match generation() {
            Ok(output) => {
                let payload = serde_json::to_string(&CompletionChunk {
                    id: request_id.clone(),
                    object: "text_completion",
                    created,
                    model: model_id.clone(),
                    choices: vec![CompletionChunkChoice {
                        index: 0,
                        text: String::new(),
                        finish_reason: Some(finish_reason_label(output.finish_reason)),
                    }],
                });
                if let Ok(payload) = payload {
                    let _ = tx.blocking_send(payload);
                }
                let _ = tx.blocking_send("[DONE]".to_string());
            }
            Err(err) => {
                let _ = tx.blocking_send(stream_error_payload(err));
                let _ = tx.blocking_send("[DONE]".to_string());
            }
        }
    });

    sse_from_channel(rx)
}

fn chat_stream_response(
    engine: Arc<Mutex<ServerEngine>>,
    request_id: String,
    created: i64,
    model_id: String,
    request: GenerateRequest,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<String>(32);
    tokio::task::spawn_blocking(move || {
        let role_payload = serde_json::to_string(&ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
        });
        if let Ok(payload) = role_payload {
            let _ = tx.blocking_send(payload);
        }

        let generation = || -> anyhow::Result<GenerateOutput> {
            let mut engine = engine.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
            engine.stream(request, |chunk| {
                let payload = serde_json::to_string(&ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_id.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(chunk.to_string()),
                        },
                        finish_reason: None,
                    }],
                })?;
                tx.blocking_send(payload)
                    .map_err(|_| anyhow!("stream receiver dropped"))?;
                Ok(())
            })
        };

        match generation() {
            Ok(output) => {
                let payload = serde_json::to_string(&ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_id.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some(finish_reason_label(output.finish_reason)),
                    }],
                });
                if let Ok(payload) = payload {
                    let _ = tx.blocking_send(payload);
                }
                let _ = tx.blocking_send("[DONE]".to_string());
            }
            Err(err) => {
                let _ = tx.blocking_send(stream_error_payload(err));
                let _ = tx.blocking_send("[DONE]".to_string());
            }
        }
    });

    sse_from_channel(rx)
}

fn sse_from_channel(
    rx: mpsc::Receiver<String>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = ReceiverStream::new(rx).map(|payload| Ok(Event::default().data(payload)));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn build_completion_response(
    request_id: String,
    created: i64,
    model_id: String,
    output: GenerateOutput,
) -> CompletionResponse {
    CompletionResponse {
        id: request_id,
        object: "text_completion",
        created,
        model: model_id,
        choices: vec![CompletionChoice {
            index: 0,
            text: output.text,
            finish_reason: finish_reason_label(output.finish_reason),
        }],
        usage: Usage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    }
}

fn build_chat_completion_response(
    request_id: String,
    created: i64,
    model_id: String,
    output: GenerateOutput,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model: model_id,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: output.text,
            },
            finish_reason: finish_reason_label(output.finish_reason),
        }],
        usage: Usage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    }
}

fn normalize_chat_messages(
    messages: &[ChatCompletionMessage],
) -> Result<Vec<RenderedChatMessage>, AppError> {
    messages
        .iter()
        .map(|message| {
            let role = match message.role.trim().to_ascii_lowercase().as_str() {
                "system" | "developer" => RenderedChatRole::System,
                "user" => RenderedChatRole::User,
                "assistant" => RenderedChatRole::Assistant,
                other => {
                    return Err(AppError::bad_request(format!(
                        "unsupported chat role '{other}'"
                    )));
                }
            };
            let content = message.content.to_text()?;
            Ok(RenderedChatMessage { role, content })
        })
        .collect()
}

impl MessageContent {
    fn to_text(&self) -> Result<String, AppError> {
        match self {
            Self::Text(text) => Ok(text.clone()),
            Self::Parts(parts) => {
                let mut rendered = String::new();
                for part in parts {
                    match part.part_type.as_str() {
                        "text" => {
                            rendered.push_str(part.text.as_deref().unwrap_or_default());
                        }
                        other => {
                            return Err(AppError::bad_request(format!(
                                "unsupported content part type '{other}'"
                            )));
                        }
                    }
                }
                Ok(rendered)
            }
        }
    }
}

fn normalize_stop_sequences(stop: Option<StopSequences>) -> Vec<String> {
    match stop {
        Some(StopSequences::One(stop)) => vec![stop],
        Some(StopSequences::Many(stops)) => stops,
        None => Vec::new(),
    }
}

fn validate_model(requested: Option<&str>, info: &ModelInfo) -> Result<(), AppError> {
    if let Some(requested) = requested {
        let matches_loaded_id = requested == info.id;
        let matches_name = info
            .model_name
            .as_deref()
            .is_some_and(|name| name == requested);
        if !matches_loaded_id && !matches_name {
            return Err(AppError::bad_request(format!(
                "model '{requested}' is not loaded; available model is '{}'",
                info.id
            )));
        }
    }
    Ok(())
}

fn validate_choice_count(n: Option<u32>) -> Result<(), AppError> {
    if let Some(n) = n
        && n != 1
    {
        return Err(AppError::bad_request(
            "AX basic server currently supports n=1 only",
        ));
    }
    Ok(())
}

fn finish_reason_label(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn map_generation_error(err: anyhow::Error) -> AppError {
    let message = err.to_string();
    if is_validation_error(&message) {
        AppError::bad_request(message)
    } else {
        AppError::internal(err)
    }
}

fn is_validation_error(message: &str) -> bool {
    message.contains("must be")
        || message.contains("does not fit")
        || message.contains("unsupported")
        || message.contains("provide a non-empty prompt")
}

fn stream_error_payload(err: anyhow::Error) -> String {
    serde_json::to_string(&ErrorEnvelope {
        error: ErrorBody {
            message: err.to_string(),
            kind: if is_validation_error(&err.to_string()) {
                "invalid_request_error"
            } else {
                "server_error"
            },
        },
    })
    .unwrap_or_else(|_| {
        "{\"error\":{\"message\":\"stream failed\",\"type\":\"server_error\"}}".to_string()
    })
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_stop_sequences_accepts_string() {
        let normalized = normalize_stop_sequences(Some(StopSequences::One("END".to_string())));
        assert_eq!(normalized, vec!["END".to_string()]);
    }

    #[test]
    fn test_message_content_parts_concatenate_text() {
        let content = MessageContent::Parts(vec![
            MessageContentPart {
                part_type: "text".to_string(),
                text: Some("hello".to_string()),
            },
            MessageContentPart {
                part_type: "text".to_string(),
                text: Some(" world".to_string()),
            },
        ]);
        assert_eq!(content.to_text().unwrap(), "hello world");
    }
}
