//! Minimal OpenAI-compatible greedy completions frontend for a pipeline chain.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::EngineTokenizer;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::TokenStepRequest;
use crate::client::{PipelineChainClient, PipelineClientError};

#[derive(Clone)]
pub struct GatewayState {
    client: Arc<PipelineChainClient>,
    tokenizer: EngineTokenizer,
    model_id: Arc<str>,
    api_key: Arc<str>,
    permits: Arc<Semaphore>,
    next_request_id: Arc<AtomicU64>,
    request_timeout: Duration,
}

impl GatewayState {
    pub fn new(
        client: PipelineChainClient,
        tokenizer: EngineTokenizer,
        model_id: String,
        api_key: String,
        maximum_concurrent_requests: usize,
        request_timeout: Duration,
    ) -> Result<Self, GatewayConfigError> {
        if model_id.trim().is_empty() {
            return Err(GatewayConfigError::EmptyModelId);
        }
        if api_key.len() < 16 {
            return Err(GatewayConfigError::WeakApiKey);
        }
        if request_timeout.is_zero() {
            return Err(GatewayConfigError::ZeroRequestTimeout);
        }
        Ok(Self {
            client: Arc::new(client),
            tokenizer,
            model_id: Arc::from(model_id),
            api_key: Arc::from(api_key),
            permits: Arc::new(Semaphore::new(maximum_concurrent_requests.max(1))),
            next_request_id: Arc::new(AtomicU64::new(seed_request_id())),
            request_timeout,
        })
    }

    fn request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed).max(1)
    }
}

pub fn router(state: GatewayState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { ready: true })
}

async fn models(State(state): State<GatewayState>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state.api_key) {
        return unauthorized();
    }
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: state.model_id.to_string(),
            object: "model",
        }],
    })
    .into_response()
}

async fn completions(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> Response {
    if !authorized(&headers, &state.api_key) {
        return unauthorized();
    }
    if request.model != state.model_id.as_ref() {
        return api_error(StatusCode::NOT_FOUND, "requested model is not loaded");
    }
    if request.prompt.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "prompt must not be empty");
    }
    if request
        .temperature
        .is_some_and(|temperature| temperature != 0.0)
    {
        return api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "initial distributed pipeline supports greedy temperature=0 only",
        );
    }
    if request.stop.is_some() {
        return api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "text stop sequences are not yet supported; EOS is honored",
        );
    }
    let maximum_tokens = request.max_tokens.unwrap_or(128);
    if maximum_tokens == 0 || maximum_tokens > 4096 {
        return api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "max_tokens must be between 1 and 4096",
        );
    }
    let prompt_tokens = match state
        .tokenizer
        .encode_with_special_tokens(&request.prompt, true)
    {
        Ok(tokens) if !tokens.is_empty() => tokens,
        Ok(_) => return api_error(StatusCode::BAD_REQUEST, "prompt encoded to zero tokens"),
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let permit = match Arc::clone(&state.permits).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "pipeline gateway concurrency limit reached",
            );
        }
    };
    let request_id = state.request_id();
    let completion_id = format!("cmpl-{request_id}");
    let created = unix_seconds();

    if request.stream {
        let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(16);
        tokio::spawn(async move {
            let _permit = permit;
            let result = generate(
                &state,
                request_id,
                &prompt_tokens,
                maximum_tokens,
                |delta| {
                    let sender = sender.clone();
                    let chunk = CompletionChunk {
                        id: completion_id.clone(),
                        object: "text_completion",
                        created,
                        model: state.model_id.to_string(),
                        choices: vec![CompletionChunkChoice {
                            text: delta,
                            index: 0,
                            finish_reason: None,
                        }],
                    };
                    async move {
                        let data = serde_json::to_string(&chunk)
                            .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.into());
                        sender.send(Ok(Event::default().data(data))).await.is_ok()
                    }
                },
            )
            .await;
            match result {
                Ok(outcome) => {
                    let final_chunk = CompletionChunk {
                        id: completion_id,
                        object: "text_completion",
                        created,
                        model: state.model_id.to_string(),
                        choices: vec![CompletionChunkChoice {
                            text: String::new(),
                            index: 0,
                            finish_reason: Some(outcome.finish_reason),
                        }],
                    };
                    if let Ok(data) = serde_json::to_string(&final_chunk) {
                        let _ = sender.send(Ok(Event::default().data(data))).await;
                    }
                }
                Err(error) => {
                    let data = serde_json::json!({
                        "error": {"message": error.to_string(), "type": "pipeline_error"}
                    })
                    .to_string();
                    let _ = sender.send(Ok(Event::default().data(data))).await;
                }
            }
            let _ = sender.send(Ok(Event::default().data("[DONE]"))).await;
        });
        return Sse::new(ReceiverStream::new(receiver))
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let _permit = permit;
    let result = generate(
        &state,
        request_id,
        &prompt_tokens,
        maximum_tokens,
        |_| async { true },
    )
    .await;
    match result {
        Ok(outcome) => {
            let text = match state.tokenizer.decode(&outcome.tokens, true) {
                Ok(text) => text,
                Err(error) => {
                    return api_error(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string());
                }
            };
            Json(CompletionResponse {
                id: completion_id,
                object: "text_completion",
                created,
                model: state.model_id.to_string(),
                choices: vec![CompletionChoice {
                    text,
                    index: 0,
                    finish_reason: outcome.finish_reason,
                }],
                usage: Usage {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens: outcome.tokens.len(),
                    total_tokens: prompt_tokens.len() + outcome.tokens.len(),
                },
            })
            .into_response()
        }
        Err(error) => generate_failure_response(&error),
    }
}

async fn chat_completions(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if !authorized(&headers, &state.api_key) {
        return unauthorized();
    }
    if request.model != state.model_id.as_ref() {
        return api_error(StatusCode::NOT_FOUND, "requested model is not loaded");
    }
    if request.messages.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    if request
        .temperature
        .is_some_and(|temperature| temperature != 0.0)
    {
        return api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "initial distributed pipeline supports greedy temperature=0 only",
        );
    }
    if request.stop.is_some() {
        return api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "text stop sequences are not yet supported; EOS is honored",
        );
    }
    let maximum_tokens = request.max_tokens.unwrap_or(128);
    if maximum_tokens == 0 || maximum_tokens > 4096 {
        return api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "max_tokens must be between 1 and 4096",
        );
    }
    let prompt = match format_llama3_chat(&request.messages) {
        Ok(prompt) => prompt,
        Err(message) => return api_error(StatusCode::UNPROCESSABLE_ENTITY, message),
    };
    let prompt_tokens = match state.tokenizer.encode_with_special_tokens(&prompt, false) {
        Ok(tokens) if !tokens.is_empty() => tokens,
        Ok(_) => return api_error(StatusCode::BAD_REQUEST, "messages encoded to zero tokens"),
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let permit = match Arc::clone(&state.permits).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "pipeline gateway concurrency limit reached",
            );
        }
    };
    let request_id = state.request_id();
    let completion_id = format!("chatcmpl-{request_id}");
    let created = unix_seconds();

    if request.stream {
        let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(16);
        tokio::spawn(async move {
            let _permit = permit;
            let initial = ChatCompletionChunk {
                id: completion_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: state.model_id.to_string(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: Some("assistant"),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            if let Ok(data) = serde_json::to_string(&initial) {
                let _ = sender.send(Ok(Event::default().data(data))).await;
            }
            let result = generate(
                &state,
                request_id,
                &prompt_tokens,
                maximum_tokens,
                |delta| {
                    let sender = sender.clone();
                    let chunk = ChatCompletionChunk {
                        id: completion_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: state.model_id.to_string(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: Some(delta),
                            },
                            finish_reason: None,
                        }],
                    };
                    async move {
                        let data = serde_json::to_string(&chunk)
                            .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.into());
                        sender.send(Ok(Event::default().data(data))).await.is_ok()
                    }
                },
            )
            .await;
            match result {
                Ok(outcome) => {
                    let final_chunk = ChatCompletionChunk {
                        id: completion_id,
                        object: "chat.completion.chunk",
                        created,
                        model: state.model_id.to_string(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(outcome.finish_reason),
                        }],
                    };
                    if let Ok(data) = serde_json::to_string(&final_chunk) {
                        let _ = sender.send(Ok(Event::default().data(data))).await;
                    }
                }
                Err(error) => {
                    let data = serde_json::json!({
                        "error": {"message": error.to_string(), "type": "pipeline_error"}
                    })
                    .to_string();
                    let _ = sender.send(Ok(Event::default().data(data))).await;
                }
            }
            let _ = sender.send(Ok(Event::default().data("[DONE]"))).await;
        });
        return Sse::new(ReceiverStream::new(receiver))
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let _permit = permit;
    match generate(
        &state,
        request_id,
        &prompt_tokens,
        maximum_tokens,
        |_| async { true },
    )
    .await
    {
        Ok(outcome) => {
            let content = match state.tokenizer.decode(&outcome.tokens, true) {
                Ok(content) => content,
                Err(error) => {
                    return api_error(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string());
                }
            };
            Json(ChatCompletionResponse {
                id: completion_id,
                object: "chat.completion",
                created,
                model: state.model_id.to_string(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessageResponse {
                        role: "assistant",
                        content,
                    },
                    finish_reason: outcome.finish_reason,
                }],
                usage: Usage {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens: outcome.tokens.len(),
                    total_tokens: prompt_tokens.len() + outcome.tokens.len(),
                },
            })
            .into_response()
        }
        Err(error) => generate_failure_response(&error),
    }
}

fn format_llama3_chat(messages: &[ChatMessage]) -> Result<String, &'static str> {
    let mut prompt = String::from("<|begin_of_text|>");
    for message in messages {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            return Err("initial Llama 3 chat accepts system, user, and assistant roles only");
        }
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(&message.role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(&message.content);
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    Ok(prompt)
}

struct GenerateOutcome {
    tokens: Vec<u32>,
    /// OpenAI finish reason: `"stop"` on EOS/disconnect, `"length"` when the
    /// max_tokens budget is exhausted without EOS.
    finish_reason: &'static str,
}

async fn generate<F, Fut>(
    state: &GatewayState,
    request_id: u64,
    prompt_tokens: &[u32],
    maximum_tokens: usize,
    mut on_delta: F,
) -> Result<GenerateOutcome, PipelineClientError>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let mut generated = Vec::with_capacity(maximum_tokens);
    // Codepoint-complete text already emitted to the client. Incomplete
    // multi-byte tails (decoded as U+FFFD) are held back until later tokens
    // finish them — never streamed as � and never re-emitted wholesale.
    let mut emitted = String::new();
    let mut sequence = 1_u64;
    let mut token_offset = 0_u64;
    let mut input = prompt_tokens.to_vec();
    // Default: budget exhausted. EOS or client disconnect overrides to stop.
    let mut finish_reason = "length";
    let deadline = tokio::time::Instant::now() + state.request_timeout;
    let generation_result = async {
        while generated.len() < maximum_tokens {
            let token = tokio::time::timeout_at(
                deadline,
                state.client.step(TokenStepRequest {
                    request_id,
                    request_sequence: sequence,
                    token_offset,
                    token_ids: input.clone(),
                }),
            )
            .await
            .map_err(|_| PipelineClientError::DeadlineExceeded)??
            .token_id;
            generated.push(token);
            if let Ok(next_rendered) = state.tokenizer.decode(&generated, true)
                && let Some(delta) = stream_delta(&emitted, &next_rendered)
            {
                // Advance the emitted cursor only over complete text so a
                // later completed codepoint still prefixes cleanly.
                emitted = complete_decode_prefix(&next_rendered).to_string();
                if !on_delta(delta).await {
                    // Consumer gone: treat as a clean client stop.
                    finish_reason = "stop";
                    break;
                }
            }
            if state
                .tokenizer
                .eos_token_id()
                .is_some_and(|eos| eos == token)
            {
                finish_reason = "stop";
                break;
            }
            token_offset = token_offset
                .checked_add(if sequence == 1 {
                    prompt_tokens.len() as u64
                } else {
                    1
                })
                .ok_or(PipelineClientError::TokenOffsetOverflow)?;
            sequence = sequence
                .checked_add(1)
                .ok_or(PipelineClientError::TokenOffsetOverflow)?;
            input.clear();
            input.push(token);
        }
        Ok(GenerateOutcome {
            tokens: generated,
            finish_reason,
        })
    }
    .await;
    let close_result = match tokio::time::timeout(
        Duration::from_secs(10),
        state.client.close_request(request_id),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => Err(PipelineClientError::CloseDeadlineExceeded),
    };
    match generation_result {
        Ok(outcome) => {
            // Generation already produced tokens (and may have streamed deltas).
            // A rank close/timeout must not discard a successful model result —
            // log and return the outcome; ops can reclaim stranded KV.
            if let Err(error) = close_result {
                tracing::warn!(
                    request_id,
                    %error,
                    "pipeline close_request failed after successful generation"
                );
            }
            Ok(outcome)
        }
        Err(error) => {
            let _ = close_result;
            Err(error)
        }
    }
}

/// Decode text with a trailing incomplete multi-byte codepoint stripped.
///
/// Byte-level BPE (Qwen/Gemma/etc.) can leave a partial UTF-8 sequence at the
/// end of a token window; HuggingFace-style decode renders that as U+FFFD.
fn complete_decode_prefix(decoded: &str) -> &str {
    decoded.strip_suffix('\u{FFFD}').unwrap_or(decoded)
}

/// Diff consecutive full-sequence decodes for streaming SSE deltas.
///
/// Holds back a trailing U+FFFD (incomplete multi-byte codepoint) and never
/// falls back to re-emitting the entire string when prefix strip fails — that
/// fallback was the source of CJK/emoji corruption and full-text re-sends.
fn stream_delta(already_emitted: &str, next_full_decode: &str) -> Option<String> {
    let complete = complete_decode_prefix(next_full_decode);
    if complete.len() <= already_emitted.len() {
        return None;
    }
    if !complete.starts_with(already_emitted) {
        // Tokenizer non-monotonic decode or corrupted cursor: skip rather than
        // re-emit the whole string. The final non-stream decode remains correct.
        return None;
    }
    if !complete.is_char_boundary(already_emitted.len()) {
        return None;
    }
    let delta = &complete[already_emitted.len()..];
    if delta.is_empty() {
        None
    } else {
        Some(delta.to_string())
    }
}

fn authorized(headers: &HeaderMap, expected: &str) -> bool {
    let provided = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .unwrap_or("");
    constant_time_eq(provided, expected)
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.as_bytes()
        .iter()
        .zip(right.as_bytes())
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        "missing or invalid bearer token",
    )
        .into_response()
}

fn api_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {"message": message, "type": "invalid_request_error"}
        })),
    )
        .into_response()
}

/// An expired generation deadline is a gateway timeout (504), not a bad
/// upstream response (502): retry policies and error-rate classifiers
/// distinguish the two.
fn generate_failure_response(error: &PipelineClientError) -> Response {
    let status = match error {
        PipelineClientError::DeadlineExceeded | PipelineClientError::CloseDeadlineExceeded => {
            StatusCode::GATEWAY_TIMEOUT
        }
        _ => StatusCode::BAD_GATEWAY,
    };
    api_error(status, &error.to_string())
}

fn seed_request_id() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(1)
        .max(1)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct HealthResponse {
    ready: bool,
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct CompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChunkChoice>,
}

#[derive(Serialize)]
struct CompletionChunkChoice {
    text: String,
    index: usize,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessageResponse,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatMessageResponse {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChunkChoice>,
}

#[derive(Serialize)]
struct ChatChunkChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum GatewayConfigError {
    #[error("pipeline gateway model_id must not be empty")]
    EmptyModelId,
    #[error("pipeline gateway API key must contain at least 16 bytes")]
    WeakApiKey,
    #[error("pipeline gateway request timeout must be greater than zero")]
    ZeroRequestTimeout,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn llama3_chat_template_preserves_order_and_adds_assistant_header() {
        let prompt = format_llama3_chat(&[
            ChatMessage {
                role: "system".into(),
                content: "Be concise.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
        ])
        .expect("valid roles");
        assert!(prompt.starts_with("<|begin_of_text|><|start_header_id|>system"));
        assert!(prompt.contains("Be concise.<|eot_id|><|start_header_id|>user"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn llama3_chat_template_rejects_tool_role_until_tool_contract_exists() {
        assert!(
            format_llama3_chat(&[ChatMessage {
                role: "tool".into(),
                content: "result".into(),
            }])
            .is_err()
        );
    }

    #[test]
    fn stream_delta_emits_ascii_suffix_only() {
        assert_eq!(stream_delta("", "hello"), Some("hello".to_string()));
        assert_eq!(stream_delta("hel", "hello"), Some("lo".to_string()));
        assert_eq!(stream_delta("hello", "hello"), None);
    }

    #[test]
    fn generate_failure_maps_deadline_to_gateway_timeout() {
        let response = generate_failure_response(&PipelineClientError::DeadlineExceeded);
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);

        let response = generate_failure_response(&PipelineClientError::CloseDeadlineExceeded);
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);

        let response = generate_failure_response(&PipelineClientError::EmptyTokenStep);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn stream_delta_holds_back_incomplete_multibyte_tail() {
        // Partial UTF-8 from byte-level BPE decodes as trailing U+FFFD.
        // Complete leading text is emitted; only the incomplete tail is held.
        assert_eq!(stream_delta("", "ab\u{FFFD}"), Some("ab".to_string()));
        assert_eq!(stream_delta("ab", "ab\u{FFFD}"), None);
        // Completing the codepoint emits only the new character.
        assert_eq!(stream_delta("ab", "ab你"), Some("你".to_string()));
        assert_eq!(stream_delta("", "🚀"), Some("🚀".to_string()));
        // Mixed complete CJK + incomplete tail: emit the complete codepoints.
        assert_eq!(stream_delta("你", "你好\u{FFFD}"), Some("好".to_string()));
        assert_eq!(stream_delta("你好", "你好世"), Some("世".to_string()));
    }

    #[test]
    fn stream_delta_never_reemits_full_string_on_prefix_mismatch() {
        // Historical bug: strip_prefix failure fell back to the whole decode,
        // re-sending prior content after a corrupted � prefix.
        assert_eq!(stream_delta("ab\u{FFFD}", "ab你"), None);
        assert_eq!(stream_delta("xy", "hello"), None);
    }

    #[test]
    fn complete_decode_prefix_strips_only_trailing_replacement() {
        assert_eq!(complete_decode_prefix("hello"), "hello");
        assert_eq!(complete_decode_prefix("hello\u{FFFD}"), "hello");
        assert_eq!(complete_decode_prefix("\u{FFFD}"), "");
        // Mid-string replacement (corrupt data) is left alone.
        assert_eq!(complete_decode_prefix("a\u{FFFD}b"), "a\u{FFFD}b");
    }
}
