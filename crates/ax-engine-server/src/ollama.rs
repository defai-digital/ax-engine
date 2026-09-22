use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::{
    EngineSessionError, EngineTokenizer, GenerateResponse, GenerateStreamEvent, SelectedBackend,
};
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::app_state::{AppState, LiveState};
use crate::backends::{llama_cpp, mlx_lm};
use crate::chat::ChatPromptTemplate;
use crate::errors::{ErrorResponse, admission_error_response, error_response, map_session_error};
use crate::generation::native::run_stateless_generate_request;
use crate::generation::streaming::{StreamStateSource, build_stream_state};
use crate::metadata::{
    MODEL_OWNER, context_length, model_supports_image, model_supports_reasoning,
};
use crate::model_load::UnloadWaitPolicy;
use crate::openai::chat_requests::MAX_INLINE_IMAGES_PER_REQUEST;
use crate::openai::generation::{populate_native_mlx_output_text, validate_openai_response_format};
use crate::openai::requests::{
    OpenAiBuiltLlamaCppChatRequest, OpenAiBuiltMlxLmChatRequest, OpenAiBuiltRequest,
    build_openai_chat_request_offloading_media, build_openai_completion_request,
    build_openai_llama_cpp_chat_request, build_openai_mlx_lm_chat_request,
    openai_chat_prompt_render_options_for_live,
};
use crate::openai::responses::{
    openai_chat_completion_response, openai_finish_reason, response_decode_tokenizer,
};
use crate::openai::schema::{
    OpenAiChatCompletionHttpRequest, OpenAiChatCompletionResponse, OpenAiChatContent,
    OpenAiChatContentPart, OpenAiChatMessage, OpenAiChatTemplateKwargs,
    OpenAiCompletionHttpRequest, OpenAiPromptInput, OpenAiStopInput, OpenAiStreamKind,
    OpenAiToolCall,
};
use crate::openai::streaming::{ChatChannelStreamFilter, IncrementalDecoder};
use crate::openai::validation::select_openai_model;
use crate::tasks::run_blocking_session_task;

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<OllamaMessage>,
    #[serde(default = "default_ollama_stream")]
    stream: bool,
    #[serde(default)]
    options: OllamaOptions,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    format: Option<Value>,
    #[serde(default)]
    keep_alive: Option<Value>,
    /// Ollama's own field is `think`; OpenClaw model `params` commonly carry
    /// the switch as `thinking`, so both spellings are accepted.
    #[serde(default, alias = "thinking")]
    think: Option<Value>,
    #[serde(default, flatten)]
    unsupported: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
    #[serde(default, flatten, skip_serializing_if = "BTreeMap::is_empty")]
    unsupported: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct OllamaToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    function: OllamaFunctionCall,
    #[serde(default, flatten, skip_serializing_if = "BTreeMap::is_empty")]
    unsupported: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct OllamaFunctionCall {
    name: String,
    #[serde(default)]
    arguments: Value,
    #[serde(default, flatten, skip_serializing_if = "BTreeMap::is_empty")]
    unsupported: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct OllamaOptions {
    /// Signed to accept Ollama's documented sentinels: `-1` (no fixed budget)
    /// and `-2` (fill the context). Resolved against the session's advertised
    /// output limit in [`resolve_ollama_num_predict`].
    #[serde(default)]
    num_predict: Option<i64>,
    /// Signed so negative values (unset/default sentinel) deserialize
    /// instead of failing the whole request; negative maps to `None` in
    /// [`validate_ollama_num_ctx`]. Zero remains a rejection.
    #[serde(default)]
    num_ctx: Option<i64>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// Signed to accept negative values real Ollama clients send
    /// (`-1` = backend default); negatives map to `None` like `seed`.
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    /// Signed to accept negative values real Ollama clients send
    /// (`-1` = backend default); negatives map to `None` like `seed`.
    #[serde(default)]
    repeat_last_n: Option<i64>,
    /// Ollama accepts negative seeds (`-1` = unseeded); they map to `None`.
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    stop: Option<OllamaStopInput>,
    #[serde(default, flatten)]
    unsupported: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum OllamaStopInput {
    Single(String),
    Multiple(Vec<String>),
}

impl OllamaStopInput {
    fn into_openai_stop(self) -> OpenAiStopInput {
        match self {
            Self::Single(value) => OpenAiStopInput::Single(value),
            Self::Multiple(values) => OpenAiStopInput::Multiple(values),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaGenerateRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: String,
    #[serde(default = "default_ollama_stream")]
    stream: bool,
    #[serde(default)]
    options: OllamaOptions,
    #[serde(default)]
    format: Option<Value>,
    #[serde(default)]
    images: Option<Vec<String>>,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    template: Option<String>,
    #[serde(default)]
    context: Option<Vec<u32>>,
    #[serde(default)]
    raw: Option<bool>,
    #[serde(default)]
    keep_alive: Option<Value>,
    #[serde(default, flatten)]
    unsupported: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaShowRequest {
    #[serde(default, alias = "name")]
    model: Option<String>,
    #[serde(default)]
    verbose: Option<bool>,
    #[serde(default, flatten)]
    unsupported: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaTagsResponse {
    models: Vec<OllamaModelTag>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OllamaModelTag {
    name: String,
    model: String,
    modified_at: String,
    size: u64,
    digest: String,
    details: OllamaModelDetails,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OllamaModelDetails {
    parent_model: String,
    format: String,
    family: String,
    families: Vec<String>,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaChatResponse {
    model: String,
    created_at: String,
    message: OllamaMessage,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<&'static str>,
    total_duration: u64,
    load_duration: u64,
    prompt_eval_count: u32,
    prompt_eval_duration: u64,
    eval_count: u32,
    eval_duration: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaGenerateResponse {
    model: String,
    created_at: String,
    response: String,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    context: Vec<u32>,
    total_duration: u64,
    load_duration: u64,
    prompt_eval_count: u32,
    prompt_eval_duration: u64,
    eval_count: u32,
    eval_duration: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaShowResponse {
    license: String,
    modelfile: String,
    parameters: String,
    template: String,
    modified_at: String,
    details: OllamaModelDetails,
    model_info: Value,
    capabilities: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaPsResponse {
    models: Vec<OllamaRunningModel>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaRunningModel {
    name: String,
    model: String,
    size: u64,
    digest: String,
    details: OllamaModelDetails,
    expires_at: String,
    size_vram: u64,
    context_length: u32,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaVersionResponse {
    version: &'static str,
}

pub(crate) async fn ollama_tags(State(state): State<AppState>) -> Json<OllamaTagsResponse> {
    Json(OllamaTagsResponse {
        models: state.snapshots().iter().map(ollama_model_tag).collect(),
    })
}

/// Convert an internal OpenAI-shaped error into the body real Ollama clients
/// expect: `{"error": "<message>"}` with the same HTTP status. ollama-js,
/// ollama-python, and OpenClaw's Ollama provider all read `error` as a plain
/// string; the structured OpenAI envelope belongs to `/v1/*` only.
fn ollama_error_http_response(error: (StatusCode, Json<ErrorResponse>)) -> Response {
    let (status, Json(body)) = error;
    (status, Json(json!({ "error": body.error.message }))).into_response()
}

/// Extract an Ollama request body, folding axum's JSON rejections (missing
/// content type, malformed JSON, missing required fields) into the Ollama
/// `{"error": "..."}` envelope with status 400. The default `Json<T>`
/// extractor answers those with plain text and non-Ollama statuses (415/422),
/// which Ollama clients cannot parse.
fn ollama_json_request<T>(
    request: Result<Json<T>, JsonRejection>,
) -> Result<T, (StatusCode, Json<ErrorResponse>)> {
    match request {
        Ok(Json(request)) => Ok(request),
        Err(rejection) => Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("invalid request body: {}", rejection.body_text()),
        )),
    }
}

pub(crate) async fn ollama_show(
    state: State<AppState>,
    request: Result<Json<OllamaShowRequest>, JsonRejection>,
) -> Response {
    let request = match ollama_json_request(request) {
        Ok(request) => request,
        Err(error) => return ollama_error_http_response(error),
    };
    match ollama_show_inner(state, request).await {
        Ok(response) => response.into_response(),
        Err(error) => ollama_error_http_response(error),
    }
}

async fn ollama_show_inner(
    State(state): State<AppState>,
    request: OllamaShowRequest,
) -> Result<Json<OllamaShowResponse>, (StatusCode, Json<ErrorResponse>)> {
    reject_unsupported_fields(&request.unsupported, "request")?;
    if request.verbose == Some(true) {
        reject_unused_field(request.verbose, "verbose")?;
    }
    let live =
        select_openai_model(&state, request.model.as_deref()).map_err(ollama_model_status)?;
    let tag = ollama_model_tag(&live);
    Ok(Json(OllamaShowResponse {
        license: String::new(),
        modelfile: ollama_modelfile(&live),
        parameters: ollama_parameters(&live),
        template: ollama_template_hint(&live),
        modified_at: rfc3339_now(),
        details: tag.details,
        model_info: ollama_model_info(&live),
        capabilities: ollama_capabilities(&live),
    }))
}

pub(crate) async fn ollama_ps(State(state): State<AppState>) -> Json<OllamaPsResponse> {
    Json(OllamaPsResponse {
        models: state
            .snapshots()
            .iter()
            .map(|live| {
                let tag = ollama_model_tag(live);
                OllamaRunningModel {
                    name: tag.name,
                    model: tag.model,
                    size: tag.size,
                    digest: tag.digest,
                    details: tag.details,
                    expires_at: rfc3339_now(),
                    size_vram: 0,
                    context_length: context_length(live),
                }
            })
            .collect(),
    })
}

pub(crate) async fn ollama_version() -> Json<OllamaVersionResponse> {
    Json(OllamaVersionResponse {
        version: env!("CARGO_PKG_VERSION"),
    })
}

pub(crate) async fn ollama_chat(
    state: State<AppState>,
    request: Result<Json<OllamaChatRequest>, JsonRejection>,
) -> Response {
    let request = match ollama_json_request(request) {
        Ok(request) => request,
        Err(error) => return ollama_error_http_response(error),
    };
    match ollama_chat_inner(state, request).await {
        Ok(response) => response,
        Err(error) => ollama_error_http_response(error),
    }
}

/// Ollama clients branch on 404 for an unknown model (pull-on-miss); the
/// OpenAI surface keeps its 400.
fn ollama_model_status(
    (status, body): (StatusCode, Json<ErrorResponse>),
) -> (StatusCode, Json<ErrorResponse>) {
    if body.error.code.as_deref() == Some("model_not_found") {
        (StatusCode::NOT_FOUND, body)
    } else {
        (status, body)
    }
}

async fn ollama_chat_inner(
    State(state): State<AppState>,
    request: OllamaChatRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let live =
        select_openai_model(&state, request.model.as_deref()).map_err(ollama_model_status)?;
    reject_ollama_tools_without_support(&live, request.tools.as_ref())?;
    let num_ctx = validate_ollama_num_ctx(&live, request.options.num_ctx)?;
    let keep_alive_unload = keep_alive_requests_unload(request.keep_alive.as_ref());
    let thinking = resolve_ollama_thinking(request.think.as_ref())?;
    if thinking == Some(true) && !model_supports_reasoning(&live) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "Ollama-compatible field `think` is enabled, but the selected AX Engine model does not advertise native reasoning support"
                .to_string(),
        ));
    }
    let stream = request.stream;
    // True per-token NDJSON streaming for the native MLX backend. Tool and
    // format requests keep the buffered two-chunk emulation because their
    // responses require whole-output post-processing (tool-call extraction,
    // JSON-object validation); delegated backends keep it because their
    // Ollama adapters run through blocking chat completion.
    let tools_requested = request.tools.is_some();
    let format_requested = request.format.is_some();
    let num_predict = resolve_ollama_num_predict(&live, request.options.num_predict)?;
    let openai_request = ollama_chat_to_openai_request(request, thinking, num_predict)?;
    // Thinking may be on by model default (DeepSeek-R1, Ornith) even when
    // `think` is omitted; the buffered path is the only one that splits
    // reasoning from content, so gate on the effective state.
    let effective_thinking =
        openai_chat_prompt_render_options_for_live(&openai_request, &live).enable_thinking;
    let can_true_stream = stream
        && live.runtime_report.selected_backend == SelectedBackend::Mlx
        && !tools_requested
        && !format_requested
        && !effective_thinking;
    if can_true_stream {
        let OpenAiBuiltRequest {
            generate_request,
            response_options,
            ..
        } = build_openai_chat_request_offloading_media(&live, &state.media, openai_request).await?;
        enforce_ollama_num_ctx_prompt_budget(num_ctx, &generate_request.input_tokens)?;
        let response = stream_ollama_native(
            state.clone(),
            live.clone(),
            generate_request,
            OllamaNativeStreamKind::Chat,
            response_options.client_stop_sequences,
        )
        .await?;
        if keep_alive_unload {
            // Generation is still running inside the body; the unload flow
            // drains admission and waits for model idle, so parking happens
            // only after the stream terminates. Run it detached.
            spawn_keep_alive_unload(state, live.model_id.as_ref().clone());
        }
        return Ok(response);
    }
    let started = std::time::Instant::now();
    let response =
        run_ollama_chat_completion(state.clone(), live.clone(), openai_request, num_ctx).await?;
    let mut ollama = ollama_chat_response_from_openai(response)?;
    // Wall time of the completion: clients derive throughput as
    // `eval_count / eval_duration`, and a zero duration divides by zero.
    let elapsed_ns = duration_nanos(started.elapsed());
    ollama.total_duration = elapsed_ns;
    ollama.eval_duration = elapsed_ns;
    if keep_alive_unload {
        // The completion has been produced (the two-chunk stream emulation is
        // fully buffered), so unloading now honors `keep_alive: 0` without
        // cutting off an in-flight body.
        unload_model_after_keep_alive(&state, live.model_id.as_ref()).await;
    }
    if stream {
        return ollama_ndjson_response(vec![
            ollama_chat_stream_chunk(&ollama),
            ollama_chat_final_chunk(&ollama),
        ]);
    }
    Ok(Json(ollama).into_response())
}

pub(crate) async fn ollama_generate(
    state: State<AppState>,
    request: Result<Json<OllamaGenerateRequest>, JsonRejection>,
) -> Response {
    let request = match ollama_json_request(request) {
        Ok(request) => request,
        Err(error) => return ollama_error_http_response(error),
    };
    match ollama_generate_inner(state, request).await {
        Ok(response) => response,
        Err(error) => ollama_error_http_response(error),
    }
}

async fn ollama_generate_inner(
    State(state): State<AppState>,
    request: OllamaGenerateRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let live =
        select_openai_model(&state, request.model.as_deref()).map_err(ollama_model_status)?;
    let num_ctx = validate_ollama_num_ctx(&live, request.options.num_ctx)?;
    let keep_alive_unload = keep_alive_requests_unload(request.keep_alive.as_ref());
    if let Some(response) = ollama_generate_lifecycle_response(&live, &request) {
        // Pure unload form: Ollama unloads the model before answering, so a
        // failed unload must surface instead of claiming `done_reason:
        // "unload"` while the model stays resident.
        if keep_alive_unload {
            crate::model_load::perform_unload(
                &state,
                live.model_id.as_ref().to_string(),
                crate::model_load::UnloadWaitPolicy::WaitForIdle,
            )
            .await
            .map_err(ollama_model_status)?;
        }
        return Ok(Json(response).into_response());
    }
    let stream = request.stream;
    let can_true_stream = stream
        && live.runtime_report.selected_backend == SelectedBackend::Mlx
        && request.format.is_none();
    let num_predict = resolve_ollama_num_predict(&live, request.options.num_predict)?;
    let openai_request = ollama_generate_to_openai_request(request, num_predict)?;
    if can_true_stream {
        let OpenAiBuiltRequest {
            generate_request,
            response_options,
            ..
        } = build_openai_completion_request(&live, openai_request)?;
        enforce_ollama_num_ctx_prompt_budget(num_ctx, &generate_request.input_tokens)?;
        let response = stream_ollama_native(
            state.clone(),
            live.clone(),
            generate_request,
            OllamaNativeStreamKind::Generate,
            response_options.client_stop_sequences,
        )
        .await?;
        if keep_alive_unload {
            // Generation is still running inside the body; the unload flow
            // drains admission and waits for model idle, so parking happens
            // only after the stream terminates. Run it detached.
            spawn_keep_alive_unload(state, live.model_id.as_ref().clone());
        }
        return Ok(response);
    }
    let ollama =
        run_ollama_completion(state.clone(), live.clone(), openai_request, num_ctx).await?;
    if keep_alive_unload {
        unload_model_after_keep_alive(&state, live.model_id.as_ref()).await;
    }
    if stream {
        return ollama_ndjson_response(vec![
            ollama_generate_stream_chunk(&ollama),
            ollama_generate_final_chunk(&ollama),
        ]);
    }
    Ok(Json(ollama).into_response())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OllamaNativeStreamKind {
    Chat,
    Generate,
}

/// Drive a native MLX generation as a true per-token NDJSON stream.
///
/// Client disconnect flips the cancel flag (channel closed), which stops the
/// drive loop between events; dropping the per-request session then frees its
/// KV state — the buffered emulation could not abort a running generation.
async fn stream_ollama_native(
    state: AppState,
    live: LiveState,
    generate_request: ax_engine_sdk::GenerateRequest,
    kind: OllamaNativeStreamKind,
    client_stop_sequences: Vec<String>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let stop_scanner = crate::openai::stop::StopSequenceScanner::new(client_stop_sequences);
    let Some(model_dir) = live.session_config.mlx_model_artifacts_dir() else {
        return Err(server_error(
            "native MLX Ollama streaming requires mlx_model_artifacts_dir with tokenizer.json"
                .to_string(),
        ));
    };
    let tokenizer = EngineTokenizer::from_model_dir_cached(model_dir).map_err(|error| {
        server_error(format!(
            "failed to load tokenizer for native MLX Ollama stream decode: {error}"
        ))
    })?;
    let stream_context = build_stream_state(&state, &live, generate_request).await?;

    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(128);
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_monitor = Arc::clone(&cancel);
    // Mirror the SSE path: a client hangup must reach the generation worker
    // (not just the blocking adapter) so it cancels at the next scheduler
    // boundary instead of finishing the prefill for a dead connection.
    let service_disconnect = match &stream_context {
        StreamStateSource::Service(events) => Some(events.disconnect_flag()),
        StreamStateSource::Stateless { .. } | StreamStateSource::Stateful { .. } => None,
    };
    let monitor_tx = tx.clone();
    let error_tx = tx.clone();
    tokio::spawn(async move {
        monitor_tx.closed().await;
        cancel_monitor.store(true, Ordering::Relaxed);
        if let Some(disconnected) = service_disconnect {
            disconnected.store(true, Ordering::Release);
        }
    });
    let handle = match stream_context {
        StreamStateSource::Service(mut events) => tokio::task::spawn_blocking(move || {
            drive_ollama_native_events(kind, &tx, &cancel, tokenizer, stop_scanner, || {
                events.blocking_recv().transpose()
            });
        }),
        StreamStateSource::Stateless {
            mut state,
            context,
            permit,
        } => tokio::task::spawn_blocking(move || {
            let _permit = permit;
            drive_ollama_native_events(kind, &tx, &cancel, tokenizer, stop_scanner, || {
                context.next_stream_event(&mut state)
            });
        }),
        StreamStateSource::Stateful {
            mut state,
            mut session,
            permit,
        } => tokio::task::spawn_blocking(move || {
            let _permit = permit;
            drive_ollama_native_events(kind, &tx, &cancel, tokenizer, stop_scanner, || {
                session.next_stream_event(&mut state)
            });
        }),
    };
    // A panic in the drive task would otherwise drop the sender and end the
    // NDJSON body with a bare EOF that clients cannot tell from success.
    tokio::spawn(async move {
        if let Err(error) = handle.await {
            tracing::error!(%error, "Ollama stream task failed");
            let _ = error_tx
                .send(Ok(format!(
                    "{}\n",
                    json!({ "error": format!("generation stream task failed: {error}") })
                )))
                .await;
        }
    });

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        Body::from_stream(ReceiverStream::new(rx)),
    )
        .into_response())
}

fn drive_ollama_native_events<N>(
    kind: OllamaNativeStreamKind,
    tx: &mpsc::Sender<Result<String, std::io::Error>>,
    cancel: &AtomicBool,
    tokenizer: EngineTokenizer,
    mut stop_scanner: Option<crate::openai::stop::StopSequenceScanner>,
    mut next: N,
) where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    // Chat streams strip model channel framing (Gemma 4 / GPT-OSS Harmony;
    // mirrors the OpenAI SSE chat path); raw generate streams keep the
    // verbatim decode.
    let mut channel_filter = match kind {
        OllamaNativeStreamKind::Chat => ChatChannelStreamFilter::from_tokenizer(&tokenizer),
        OllamaNativeStreamKind::Generate => None,
    };
    let mut decoder = IncrementalDecoder::new(tokenizer);
    loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::debug!("Ollama stream cancelled: client disconnected");
            return;
        }
        match next() {
            Ok(None) => {
                // The event channel closed without a terminal `Response`
                // (worker retired or the stream was cancelled server-side):
                // name it, so the client does not read a truncated body as
                // a complete answer.
                let _ = send_ollama_ndjson_line(
                    tx,
                    &json!({ "error": "generation stream ended before a terminal frame" }),
                );
                return;
            }
            Err(error) => {
                // Ollama surfaces mid-stream failures as an NDJSON error line.
                let _ = send_ollama_ndjson_line(tx, &json!({ "error": error.to_string() }));
                return;
            }
            Ok(Some(GenerateStreamEvent::Request(_))) => {}
            Ok(Some(GenerateStreamEvent::Step(payload))) => {
                let filtered;
                let delta_tokens = if payload.delta_text.is_none()
                    && let Some(filter) = channel_filter.as_mut()
                {
                    filtered = filter.filter(&payload.delta_tokens);
                    if filtered.is_empty() {
                        continue;
                    }
                    filtered.as_slice()
                } else {
                    payload.delta_tokens.as_slice()
                };
                let delta_text = if let Some(delta_text) = payload.delta_text {
                    delta_text
                } else if delta_tokens.is_empty() {
                    continue;
                } else {
                    match decoder.push(delta_tokens) {
                        Ok(text) => text,
                        Err(error) => {
                            let _ = send_ollama_ndjson_line(
                                tx,
                                &json!({ "error": format!(
                                    "failed to decode native MLX Ollama stream tokens: {error}"
                                ) }),
                            );
                            return;
                        }
                    }
                };
                if delta_text.is_empty() {
                    continue;
                }
                if let Some(filter) = channel_filter.as_mut() {
                    filter.mark_kept_output();
                }
                // Client stops are enforced server-side on the native backend
                // (ADR-040 D2): withheld candidate text never streams, and a
                // match ends the stream (receiver drop cancels the request).
                let (emit, stop_matched) = match stop_scanner.as_mut() {
                    Some(scanner) => {
                        let step = scanner.push(&delta_text);
                        (step.emit, step.matched)
                    }
                    None => (delta_text, false),
                };
                if !emit.is_empty() {
                    let chunk = ollama_native_delta_chunk(kind, &payload.request.model_id, emit);
                    if send_ollama_ndjson_line(tx, &chunk).is_err() {
                        return;
                    }
                }
                if stop_matched {
                    let final_chunk =
                        ollama_native_stop_final_chunk(kind, &payload.request.model_id);
                    let _ = send_ollama_ndjson_line(tx, &final_chunk);
                    return;
                }
            }
            Ok(Some(GenerateStreamEvent::Response(payload))) => {
                // The model can leave its entire answer inside an unclosed
                // thinking channel; serve that body before the final chunk.
                if let Some(filter) = channel_filter.as_mut()
                    && let Some(body_text) = filter.take_fallback_text(&mut decoder)
                {
                    let (emit, stop_matched) = match stop_scanner.as_mut() {
                        Some(scanner) => {
                            let step = scanner.push(&body_text);
                            (step.emit, step.matched)
                        }
                        None => (body_text, false),
                    };
                    if !emit.is_empty() {
                        let chunk =
                            ollama_native_delta_chunk(kind, &payload.response.model_id, emit);
                        if send_ollama_ndjson_line(tx, &chunk).is_err() {
                            return;
                        }
                    }
                    if stop_matched {
                        let final_chunk =
                            ollama_native_stop_final_chunk(kind, &payload.response.model_id);
                        let _ = send_ollama_ndjson_line(tx, &final_chunk);
                        return;
                    }
                }
                // Release the stop scanner's withheld tail (no match fired).
                if let Some(scanner) = stop_scanner.as_mut() {
                    let tail = scanner.finish();
                    if !tail.is_empty() {
                        let chunk =
                            ollama_native_delta_chunk(kind, &payload.response.model_id, tail);
                        if send_ollama_ndjson_line(tx, &chunk).is_err() {
                            return;
                        }
                    }
                }
                let summary = ollama_generate_response_from_generate(payload.response);
                let final_chunk = match kind {
                    OllamaNativeStreamKind::Chat => with_done_reason(
                        json!({
                            "model": summary.model,
                            "created_at": summary.created_at,
                            "message": {"role": "assistant", "content": ""},
                            "done": true,
                            "total_duration": summary.total_duration,
                            "load_duration": summary.load_duration,
                            "prompt_eval_count": summary.prompt_eval_count,
                            "prompt_eval_duration": summary.prompt_eval_duration,
                            "eval_count": summary.eval_count,
                            "eval_duration": summary.eval_duration,
                        }),
                        summary.done_reason,
                    ),
                    OllamaNativeStreamKind::Generate => {
                        let mut summary = summary;
                        summary.response = String::new();
                        ollama_generate_final_chunk(&summary)
                    }
                };
                let _ = send_ollama_ndjson_line(tx, &final_chunk);
                return;
            }
        }
    }
}

fn ollama_native_delta_chunk(
    kind: OllamaNativeStreamKind,
    model: &str,
    delta_text: String,
) -> Value {
    match kind {
        OllamaNativeStreamKind::Chat => json!({
            "model": model,
            "created_at": rfc3339_now(),
            "message": { "role": "assistant", "content": delta_text },
            "done": false,
        }),
        OllamaNativeStreamKind::Generate => json!({
            "model": model,
            "created_at": rfc3339_now(),
            "response": delta_text,
            "done": false,
        }),
    }
}

fn send_ollama_ndjson_line(
    tx: &mpsc::Sender<Result<String, std::io::Error>>,
    chunk: &Value,
) -> Result<(), ()> {
    let Ok(mut line) = serde_json::to_string(chunk) else {
        return Err(());
    };
    line.push('\n');
    tx.blocking_send(Ok(line)).map_err(|_| ())
}

fn ollama_chat_to_openai_request(
    request: OllamaChatRequest,
    thinking: Option<bool>,
    num_predict: OllamaNumPredict,
) -> Result<OpenAiChatCompletionHttpRequest, (StatusCode, Json<ErrorResponse>)> {
    reject_unsupported_fields(&request.unsupported, "request")?;
    reject_unsupported_fields(&request.options.unsupported, "options")?;
    let _ = request.keep_alive;
    let mut inline_images = 0usize;
    for message in &request.messages {
        let added = message.images.as_ref().map_or(0, Vec::len);
        inline_images = inline_images
            .checked_add(added)
            .filter(|&total| total <= MAX_INLINE_IMAGES_PER_REQUEST)
            .ok_or_else(too_many_inline_images)?;
    }
    let messages = request
        .messages
        .into_iter()
        .map(ollama_message_to_openai_message)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OpenAiChatCompletionHttpRequest {
        model: request.model,
        messages,
        input_tokens: Vec::new(),
        max_tokens: num_predict.max_tokens,
        max_completion_tokens: None,
        temperature: request.options.temperature,
        top_p: request.options.top_p,
        top_k: request
            .options
            .top_k
            .and_then(|top_k| u32::try_from(top_k).ok()),
        min_p: request.options.min_p,
        repetition_penalty: request.options.repeat_penalty,
        repetition_context_size: request
            .options
            .repeat_last_n
            .and_then(|repeat_last_n| u32::try_from(repeat_last_n).ok()),
        skip_special_tokens: None,
        vllm_xargs: None,
        stop: request.options.stop.map(OllamaStopInput::into_openai_stop),
        seed: request
            .options
            .seed
            .and_then(|seed| u64::try_from(seed).ok()),
        stream: false,
        stream_options: Default::default(),
        n: None,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: false,
        top_logprobs: None,
        reasoning: thinking.map(Value::Bool),
        ax_max_think_tokens: None,
        ax_answer_reserve_tokens: None,
        // Replay prior `thinking` history only when thinking is on for this
        // turn. With `think: false` the Qwen templates strip reasoning from
        // history; replaying it into a no-think prompt would contradict the
        // requested mode.
        chat_template_kwargs: thinking.map(|enable_thinking| OpenAiChatTemplateKwargs {
            enable_thinking: Some(enable_thinking),
            preserve_thinking: Some(enable_thinking),
        }),
        metadata: Some("ollama:/api/chat".to_string()),
        multimodal_inputs: Default::default(),
        response_format: ollama_format_to_openai(request.format)?,
        tools: request.tools,
        tool_choice: None,
        parallel_tool_calls: None,
        fit_max_tokens_to_context: num_predict.fill_context,
    })
}

fn ollama_generate_to_openai_request(
    request: OllamaGenerateRequest,
    num_predict: OllamaNumPredict,
) -> Result<OpenAiCompletionHttpRequest, (StatusCode, Json<ErrorResponse>)> {
    reject_unsupported_fields(&request.unsupported, "request")?;
    reject_unsupported_fields(&request.options.unsupported, "options")?;
    reject_unused_field(request.images, "images")?;
    reject_unused_field(request.template, "template")?;
    reject_unused_field(request.context, "context")?;
    let prompt = match request.system {
        Some(system) if request.raw != Some(true) && !system.trim().is_empty() => {
            format!("{}\n\n{}", system.trim(), request.prompt)
        }
        _ => request.prompt,
    };
    Ok(OpenAiCompletionHttpRequest {
        model: request.model,
        prompt: OpenAiPromptInput::Text(prompt),
        max_tokens: num_predict.max_tokens,
        max_completion_tokens: None,
        temperature: request.options.temperature,
        top_p: request.options.top_p,
        top_k: request
            .options
            .top_k
            .and_then(|top_k| u32::try_from(top_k).ok()),
        min_p: request.options.min_p,
        repetition_penalty: request.options.repeat_penalty,
        repetition_context_size: request
            .options
            .repeat_last_n
            .and_then(|repeat_last_n| u32::try_from(repeat_last_n).ok()),
        skip_special_tokens: None,
        vllm_xargs: None,
        stop: request.options.stop.map(OllamaStopInput::into_openai_stop),
        seed: request
            .options
            .seed
            .and_then(|seed| u64::try_from(seed).ok()),
        stream: false,
        stream_options: Default::default(),
        n: None,
        best_of: None,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        metadata: Some("ollama:/api/generate".to_string()),
        multimodal_inputs: Default::default(),
        response_format: ollama_format_to_openai(request.format)?,
        fit_max_tokens_to_context: num_predict.fill_context,
    })
}

fn ollama_generate_lifecycle_response(
    live: &LiveState,
    request: &OllamaGenerateRequest,
) -> Option<OllamaGenerateResponse> {
    if !request.prompt.is_empty()
        || request
            .system
            .as_deref()
            .is_some_and(|system| !system.trim().is_empty())
        || request.images.is_some()
        || request.template.is_some()
        || request.context.is_some()
        || !request.unsupported.is_empty()
        || !request.options.unsupported.is_empty()
    {
        return None;
    }
    Some(OllamaGenerateResponse {
        model: live.model_id.to_string(),
        created_at: rfc3339_now(),
        response: String::new(),
        done: true,
        done_reason: keep_alive_requests_unload(request.keep_alive.as_ref()).then_some("unload"),
        context: Vec::new(),
        total_duration: 0,
        load_duration: 0,
        prompt_eval_count: 0,
        prompt_eval_duration: 0,
        eval_count: 0,
        eval_duration: 0,
    })
}

fn keep_alive_requests_unload(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Number(number)) => json_number_is_zero(number),
        Some(Value::String(value)) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "0s" | "0m" | "0h"
        ),
        _ => false,
    }
}

/// Unload (soft-park) the model through the exact `/v1/model/unload` flow so
/// Ollama `keep_alive: 0` actually frees the registry entry a later load
/// depends on. Ollama clients send this before loading a different model and
/// expect the memory to be released; answering `done_reason: "unload"`
/// without running the unload would leave the model resident.
async fn unload_model_after_keep_alive(state: &AppState, model_id: &str) {
    let result = crate::model_load::perform_unload(
        state,
        model_id.to_string(),
        UnloadWaitPolicy::WaitForIdle,
    )
    .await;
    if let Err((status, Json(body))) = result {
        tracing::warn!(
            model_id,
            status = %status,
            "keep_alive unload after Ollama response failed: {}",
            body.error.message
        );
    }
}

fn json_number_is_zero(value: &serde_json::Number) -> bool {
    value
        .as_i64()
        .map(|value| value == 0)
        .or_else(|| value.as_u64().map(|value| value == 0))
        .or_else(|| value.as_f64().map(|value| value == 0.0))
        .unwrap_or(false)
}

fn reject_ollama_tools_without_support(
    live: &LiveState,
    tools: Option<&Value>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if !ollama_value_is_present(tools) || ollama_tools_supported(live) {
        return Ok(());
    }
    Err(error_response(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        "Ollama-compatible field `tools` is not supported by the selected AX Engine backend/model yet"
            .to_string(),
    ))
}

/// Validate `options.num_ctx` against the session window and return the
/// resolved positive value (negative means unset/default, mirroring Ollama's
/// integer sentinels). The value is a client-side budget only: AX has no
/// OpenAI-schema context-hint field to forward it into, so a fitting value is
/// accepted and the session's configured window keeps governing generation.
fn validate_ollama_num_ctx(
    live: &LiveState,
    requested: Option<i64>,
) -> Result<Option<u32>, (StatusCode, Json<ErrorResponse>)> {
    let Some(requested) = requested.and_then(|value| u32::try_from(value).ok()) else {
        return Ok(None);
    };
    let available = context_length(live);
    if requested == 0 || requested > available {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            format!(
                "Ollama options.num_ctx requested {requested} tokens, but this AX Engine session is configured for {available}; restart AX with a matching --total-blocks value or lower OpenClaw's contextTokens/contextWindow"
            ),
        ));
    }
    Ok(Some(requested))
}

/// Fail closed when a rendered prompt cannot fit the client's
/// `options.num_ctx` budget: real Ollama truncates the prompt to fit, and AX
/// never truncates silently, so the request is rejected with an actionable
/// message instead. Checked wherever the token count first exists (after the
/// OpenAI request is built and tokenized); delegated backends forward raw
/// text and keep no server-side count, so the check is vacuous there.
fn enforce_ollama_num_ctx_prompt_budget(
    num_ctx: Option<u32>,
    input_tokens: &[u32],
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let Some(num_ctx) = num_ctx else {
        return Ok(());
    };
    let prompt_tokens = input_tokens.len();
    if prompt_tokens > num_ctx as usize {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            format!(
                "prompt has {prompt_tokens} tokens but options.num_ctx is {num_ctx}; AX does not truncate prompts, raise num_ctx or shorten the prompt"
            ),
        ));
    }
    Ok(())
}

/// Detached `keep_alive: 0` unload for true-stream responses: the body still
/// owns the generation, and `perform_unload` drains admission and waits for
/// model idle, so parking lands after the stream terminates.
fn spawn_keep_alive_unload(state: AppState, model_id: String) {
    tokio::spawn(async move {
        unload_model_after_keep_alive(&state, &model_id).await;
    });
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OllamaNumPredict {
    max_tokens: Option<u32>,
    fill_context: bool,
}

/// Resolve Ollama `options.num_predict` into an OpenAI `max_tokens` value.
/// Positive budgets pass through. Ollama's documented sentinels `-1` (no
/// fixed budget) and `-2` (fill the context) map to the session's advertised
/// `max_output_tokens` instead of the conservative OpenAI default; other
/// non-positive values are rejected.
fn resolve_ollama_num_predict(
    live: &LiveState,
    num_predict: Option<i64>,
) -> Result<OllamaNumPredict, (StatusCode, Json<ErrorResponse>)> {
    match num_predict {
        None => Ok(OllamaNumPredict::default()),
        Some(value) if value > 0 => Ok(OllamaNumPredict {
            max_tokens: Some(u32::try_from(value).unwrap_or(u32::MAX)),
            fill_context: false,
        }),
        // The sentinels mean "as many tokens as still fit": the budget is
        // clamped to the remaining context once the prompt is tokenized,
        // never rejected as context_length_exceeded.
        Some(-1) | Some(-2) => Ok(OllamaNumPredict {
            max_tokens: Some(crate::metadata::max_output_tokens_live(
                live,
                context_length(live),
            )),
            fill_context: true,
        }),
        Some(value) => Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!(
                "Ollama options.num_predict must be a positive token budget or the \
sentinels -1/-2 (received {value})"
            ),
        )),
    }
}

fn resolve_ollama_thinking(
    value: Option<&Value>,
) -> Result<Option<bool>, (StatusCode, Json<ErrorResponse>)> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        Value::Bool(enabled) => Ok(Some(*enabled)),
        Value::String(level) => match level.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "false" | "disabled" => Ok(Some(false)),
            "low" | "medium" | "high" | "minimal" | "xhigh" | "adaptive" | "max" => Ok(Some(true)),
            _ => Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Ollama-compatible field `think` must be false, true, off, low, medium, or high"
                    .to_string(),
            )),
        },
        _ => Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Ollama-compatible field `think` must be a boolean or thinking level string"
                .to_string(),
        )),
    }
}

fn ollama_value_is_present(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::String(value)) => !value.trim().is_empty(),
        Some(Value::Array(values)) => !values.is_empty(),
        Some(Value::Object(object)) => !object.is_empty(),
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(number)) => json_number_is_nonzero(number),
    }
}

fn json_number_is_nonzero(value: &serde_json::Number) -> bool {
    value
        .as_i64()
        .map(|value| value != 0)
        .or_else(|| value.as_u64().map(|value| value != 0))
        .or_else(|| value.as_f64().map(|value| value != 0.0))
        .unwrap_or(true)
}

fn ollama_message_to_openai_message(
    message: OllamaMessage,
) -> Result<OpenAiChatMessage, (StatusCode, Json<ErrorResponse>)> {
    reject_unsupported_fields(&message.unsupported, "messages[]")?;
    let content = ollama_message_content(message.content, message.images)?;
    let tool_calls = match message.tool_calls {
        Some(calls) => {
            for call in &calls {
                reject_unsupported_fields(&call.unsupported, "messages[].tool_calls[]")?;
                reject_unsupported_fields(
                    &call.function.unsupported,
                    "messages[].tool_calls[].function",
                )?;
            }
            Some(serde_json::to_value(calls).unwrap_or(Value::Null))
        }
        None => None,
    };
    Ok(OpenAiChatMessage {
        role: message.role,
        content: Some(content),
        tool_calls,
        reasoning_content: message.thinking,
        _tool_call_id: None,
        _name: message.tool_name,
    })
}

fn too_many_inline_images() -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!(
            "request carries more than {MAX_INLINE_IMAGES_PER_REQUEST} \
inline images; split the conversation or drop older images"
        ),
    )
}

fn ollama_message_content(
    text: String,
    images: Option<Vec<String>>,
) -> Result<OpenAiChatContent, (StatusCode, Json<ErrorResponse>)> {
    let Some(images) = images.filter(|images| !images.is_empty()) else {
        return Ok(OpenAiChatContent::Text(text));
    };
    let added = images.len();
    if added > MAX_INLINE_IMAGES_PER_REQUEST {
        return Err(too_many_inline_images());
    }
    let mut parts = Vec::with_capacity(added + usize::from(!text.is_empty()));
    if !text.is_empty() {
        parts.push(OpenAiChatContentPart {
            part_type: "text".to_string(),
            text: Some(text),
            image_url: None,
            video_url: None,
            input_audio: None,
            audio_url: None,
        });
    }
    for image in images {
        let url = ollama_image_data_uri(&image)?;
        parts.push(OpenAiChatContentPart {
            part_type: "image_url".to_string(),
            text: None,
            image_url: Some(json!({ "url": url })),
            video_url: None,
            input_audio: None,
            audio_url: None,
        });
    }
    Ok(OpenAiChatContent::Parts(parts))
}

fn ollama_image_data_uri(encoded: &str) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    // Ollama's contract is raw base64, but some clients send a full data URI.
    // Accept it by unwrapping to the payload, then validate exactly like raw
    // base64 — a `data:` prefix must not bypass decoding or magic-byte checks.
    let encoded = if let Some(rest) = encoded.trim().strip_prefix("data:") {
        let Some((meta, payload)) = rest.split_once(',') else {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "messages[].images data: URI is malformed (missing comma)".to_string(),
            ));
        };
        if !meta.contains("base64") {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "messages[].images data: URI must be base64-encoded".to_string(),
            ));
        }
        payload.trim()
    } else {
        encoded.trim()
    };
    if encoded.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "messages[].images contains an empty image payload".to_string(),
        ));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("messages[].images contains invalid base64: {error}"),
            )
        })?;
    let mime = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        "image/gif"
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        "image/webp"
    } else if bytes.starts_with(b"BM") {
        "image/bmp"
    } else {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_modality",
            "messages[].images must contain base64 PNG, JPEG, GIF, WebP, or BMP image data"
                .to_string(),
        ));
    };
    Ok(format!("data:{mime};base64,{encoded}"))
}

/// Map Ollama `format` onto an OpenAI `response_format`. Ollama accepts the
/// string `"json"` or a bare JSON Schema object; the OpenAI layer only
/// enforces `json_object` / `json_schema` wrappers, so a bare schema must be
/// wrapped (an unwrapped one would pass validation-free) and anything else
/// fails closed instead of silently producing unconstrained output.
fn ollama_format_to_openai(
    format: Option<Value>,
) -> Result<Option<Value>, (StatusCode, Json<ErrorResponse>)> {
    let Some(format) = format else {
        return Ok(None);
    };
    match format {
        Value::Null => Ok(None),
        Value::String(value)
            if value.eq_ignore_ascii_case("json") || value.eq_ignore_ascii_case("json_object") =>
        {
            Ok(Some(json!({"type": "json_object"})))
        }
        Value::Object(ref object)
            if matches!(
                object.get("type").and_then(Value::as_str),
                Some("json_object" | "json_schema" | "text")
            ) =>
        {
            Ok(Some(format))
        }
        Value::Object(_) => Ok(Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "ollama_format", "schema": format}
        }))),
        other => Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!(
                "Ollama-compatible field `format` must be \"json\" or a JSON Schema object (received {other})"
            ),
        )),
    }
}

/// Real Ollama omits `done_reason` rather than emitting `null`; keep the
/// streaming terminal line shaped like the non-streaming response struct.
fn with_done_reason(mut chunk: Value, done_reason: Option<&'static str>) -> Value {
    if let (Some(reason), Some(object)) = (done_reason, chunk.as_object_mut()) {
        object.insert("done_reason".to_string(), Value::String(reason.to_string()));
    }
    chunk
}

async fn run_ollama_chat_completion(
    state: AppState,
    live: LiveState,
    request: OpenAiChatCompletionHttpRequest,
    num_ctx: Option<u32>,
) -> Result<OpenAiChatCompletionResponse, (StatusCode, Json<ErrorResponse>)> {
    if mlx_lm::is_selected(&live) {
        // Delegated backend: no server-side token count exists to check
        // against options.num_ctx; the upstream server owns its window.
        let OpenAiBuiltMlxLmChatRequest {
            chat_request,
            response_options,
            ..
        } = build_openai_mlx_lm_chat_request(&live, request)?;
        let request_id = state.allocate_request_id();
        let runtime = live.runtime_report.clone();
        let mlx_lm_backend = mlx_lm::config(&live).map_err(map_session_error)?;
        let permit = state.try_admit(&live).map_err(admission_error_response)?;
        let response = run_blocking_session_task(move || {
            let _permit = permit;
            mlx_lm::run_chat_generate(request_id, &runtime, &mlx_lm_backend, &chat_request)
        })
        .await?;
        validate_openai_response_format(&response, &response_options)?;
        return Ok(openai_chat_completion_response(
            &response,
            OpenAiStreamKind::ChatCompletion.response_id(request_id),
            response_options,
            None,
            response_decode_tokenizer(&live).as_ref(),
        ));
    }

    if llama_cpp::supports_server_chat(&live) {
        // Delegated backend: no server-side token count exists to check
        // against options.num_ctx; the upstream server owns its window.
        let OpenAiBuiltLlamaCppChatRequest {
            chat_request,
            response_options,
            ..
        } = build_openai_llama_cpp_chat_request(&live, request)?;
        let request_id = state.allocate_request_id();
        let runtime = live.runtime_report.clone();
        let llama_backend = llama_cpp::config(&live).map_err(map_session_error)?;
        let permit = state.try_admit(&live).map_err(admission_error_response)?;
        let response = run_blocking_session_task(move || {
            let _permit = permit;
            llama_cpp::run_chat_generate(request_id, &runtime, &llama_backend, &chat_request)
        })
        .await?;
        validate_openai_response_format(&response, &response_options)?;
        return Ok(openai_chat_completion_response(
            &response,
            OpenAiStreamKind::ChatCompletion.response_id(request_id),
            response_options,
            None,
            response_decode_tokenizer(&live).as_ref(),
        ));
    }

    let OpenAiBuiltRequest {
        generate_request,
        response_options,
        ..
    } = build_openai_chat_request_offloading_media(&live, &state.media, request).await?;
    enforce_ollama_num_ctx_prompt_budget(num_ctx, &generate_request.input_tokens)?;
    let (request_id, mut response) =
        run_stateless_generate_request(&state, &live, generate_request).await?;
    let native_reasoning = populate_native_mlx_output_text(
        &live,
        &mut response,
        OpenAiStreamKind::ChatCompletion,
        response_options.include_reasoning,
    )?;
    validate_openai_response_format(&response, &response_options)?;
    Ok(openai_chat_completion_response(
        &response,
        OpenAiStreamKind::ChatCompletion.response_id(request_id),
        response_options,
        native_reasoning,
        response_decode_tokenizer(&live).as_ref(),
    ))
}

async fn run_ollama_completion(
    state: AppState,
    live: LiveState,
    request: OpenAiCompletionHttpRequest,
    num_ctx: Option<u32>,
) -> Result<OllamaGenerateResponse, (StatusCode, Json<ErrorResponse>)> {
    let started = std::time::Instant::now();
    let OpenAiBuiltRequest {
        generate_request,
        response_options,
        ..
    } = build_openai_completion_request(&live, request)?;
    enforce_ollama_num_ctx_prompt_budget(num_ctx, &generate_request.input_tokens)?;
    let (_request_id, mut response) =
        run_stateless_generate_request(&state, &live, generate_request).await?;
    let native_reasoning = populate_native_mlx_output_text(
        &live,
        &mut response,
        OpenAiStreamKind::Completion,
        response_options.include_reasoning,
    )?;
    validate_openai_response_format(&response, &response_options)?;
    debug_assert!(native_reasoning.is_none());
    // Native client stops are enforced server-side (ADR-040 D2); the Ollama
    // generate surface is built directly from GenerateResponse, so truncate
    // here rather than in the OpenAI response builders.
    let _ = crate::openai::stop::apply_client_stops_to_generate_response(
        &mut response,
        &response_options.client_stop_sequences,
    );
    let mut ollama = ollama_generate_response_from_generate(response);
    let elapsed_ns = duration_nanos(started.elapsed());
    ollama.total_duration = elapsed_ns;
    ollama.eval_duration = elapsed_ns;
    Ok(ollama)
}

fn duration_nanos(elapsed: std::time::Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

fn ollama_chat_response_from_openai(
    response: OpenAiChatCompletionResponse,
) -> Result<OllamaChatResponse, (StatusCode, Json<ErrorResponse>)> {
    let usage = response.usage;
    let Some(choice) = response.choices.into_iter().next() else {
        return Err(server_error(
            "OpenAI chat response did not contain a choice",
        ));
    };
    let tool_calls = choice
        .message
        .tool_calls
        .map(|calls| calls.into_iter().map(ollama_tool_call).collect());
    Ok(OllamaChatResponse {
        model: response.model,
        created_at: rfc3339_from_unix(response.created),
        message: OllamaMessage {
            role: "assistant".to_string(),
            content: choice.message.content,
            images: None,
            thinking: choice.message.reasoning_content,
            tool_calls,
            tool_name: None,
            unsupported: BTreeMap::new(),
        },
        done: true,
        done_reason: choice.finish_reason.and_then(ollama_done_reason),
        total_duration: 0,
        load_duration: 0,
        prompt_eval_count: usage.as_ref().map(|usage| usage.prompt_tokens).unwrap_or(0),
        prompt_eval_duration: 0,
        eval_count: usage
            .as_ref()
            .map(|usage| usage.completion_tokens)
            .unwrap_or(0),
        eval_duration: 0,
    })
}

fn ollama_generate_response_from_generate(response: GenerateResponse) -> OllamaGenerateResponse {
    let prompt_eval_count = response.prompt_tokens_used().unwrap_or(0);
    let eval_count = response.known_output_token_count().unwrap_or(0);
    OllamaGenerateResponse {
        model: response.model_id,
        created_at: rfc3339_now(),
        response: response.output_text.unwrap_or_default(),
        done: true,
        done_reason: openai_finish_reason(response.finish_reason).and_then(ollama_done_reason),
        context: Vec::new(),
        total_duration: 0,
        load_duration: 0,
        prompt_eval_count,
        prompt_eval_duration: 0,
        eval_count,
        eval_duration: 0,
    }
}

fn ollama_tool_call(call: OpenAiToolCall) -> OllamaToolCall {
    let arguments = serde_json::from_str::<Value>(&call.function.arguments)
        .unwrap_or(Value::String(call.function.arguments));
    OllamaToolCall {
        id: Some(call.id),
        function: OllamaFunctionCall {
            name: call.function.name,
            arguments,
            unsupported: BTreeMap::new(),
        },
        unsupported: BTreeMap::new(),
    }
}

/// Final NDJSON chunk when a client stop sequence ends the stream early: the
/// generation was cancelled, so the usual end-of-stream summary payload (with
/// timing/usage counts) never arrives.
fn ollama_native_stop_final_chunk(kind: OllamaNativeStreamKind, model_id: &str) -> Value {
    match kind {
        OllamaNativeStreamKind::Chat => json!({
            "model": model_id,
            "created_at": rfc3339_now(),
            "message": {"role": "assistant", "content": ""},
            "done": true,
            "done_reason": "stop",
        }),
        OllamaNativeStreamKind::Generate => json!({
            "model": model_id,
            "created_at": rfc3339_now(),
            "response": "",
            "done": true,
            "done_reason": "stop",
        }),
    }
}

fn ollama_chat_stream_chunk(response: &OllamaChatResponse) -> Value {
    json!({
        "model": response.model,
        "created_at": response.created_at,
        "message": response.message,
        "done": false,
    })
}

fn ollama_chat_final_chunk(response: &OllamaChatResponse) -> Value {
    with_done_reason(
        json!({
            "model": response.model,
            "created_at": response.created_at,
            // Real Ollama always includes `message` (with empty content) on the
            // terminal `done: true` line; a client parsing every NDJSON line
            // with a uniform schema would otherwise fail on the last line.
            "message": {"role": "assistant", "content": ""},
            "done": true,
            "total_duration": response.total_duration,
            "load_duration": response.load_duration,
            "prompt_eval_count": response.prompt_eval_count,
            "prompt_eval_duration": response.prompt_eval_duration,
            "eval_count": response.eval_count,
            "eval_duration": response.eval_duration,
        }),
        response.done_reason,
    )
}

fn ollama_generate_stream_chunk(response: &OllamaGenerateResponse) -> Value {
    json!({
        "model": response.model,
        "created_at": response.created_at,
        "response": response.response,
        "done": false,
    })
}

fn ollama_generate_final_chunk(response: &OllamaGenerateResponse) -> Value {
    with_done_reason(
        json!({
            "model": response.model,
            "created_at": response.created_at,
            // Real Ollama always includes `response` (empty string) on the
            // terminal `done: true` line; a client parsing every NDJSON line
            // with a uniform schema would otherwise fail on the last line.
            "response": "",
            "done": true,
            "total_duration": response.total_duration,
            "load_duration": response.load_duration,
            "prompt_eval_count": response.prompt_eval_count,
            "prompt_eval_duration": response.prompt_eval_duration,
            "eval_count": response.eval_count,
            "eval_duration": response.eval_duration,
        }),
        response.done_reason,
    )
}

fn ollama_ndjson_response(
    chunks: Vec<Value>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let mut body = String::new();
    for chunk in chunks {
        let line = serde_json::to_string(&chunk)
            .map_err(|error| server_error(format!("failed to serialize Ollama chunk: {error}")))?;
        body.push_str(&line);
        body.push('\n');
    }
    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        Body::from(body),
    )
        .into_response())
}

fn reject_unused_field<T>(
    value: Option<T>,
    field: &'static str,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if value.is_none() {
        return Ok(());
    }
    Err(error_response(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        format!(
            "Ollama-compatible field `{field}` is not supported by this AX Engine endpoint yet"
        ),
    ))
}

fn reject_unsupported_fields(
    fields: &BTreeMap<String, Value>,
    scope: &'static str,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let Some(field) = fields.keys().next() else {
        return Ok(());
    };
    let name = if scope == "request" {
        field.to_string()
    } else {
        format!("{scope}.{field}")
    };
    Err(error_response(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        format!("Ollama-compatible field `{name}` is not supported by this AX Engine endpoint yet"),
    ))
}

fn server_error(message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        message.into(),
    )
}

fn ollama_done_reason(reason: &str) -> Option<&'static str> {
    match reason {
        "stop" | "tool_calls" => Some("stop"),
        "length" => Some("length"),
        _ => None,
    }
}

fn ollama_model_family(model_id: &str) -> String {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("qwen") {
        "qwen".to_string()
    } else if lower.contains("gemma") {
        "gemma".to_string()
    } else if lower.contains("glm") {
        "glm".to_string()
    } else {
        "unknown".to_string()
    }
}

fn ollama_model_tag(live: &LiveState) -> OllamaModelTag {
    let model = live.model_id.to_string();
    OllamaModelTag {
        name: model.clone(),
        model,
        modified_at: rfc3339_now(),
        size: 0,
        digest: ollama_model_digest(live),
        details: ollama_model_details(live),
    }
}

fn ollama_model_digest(live: &LiveState) -> String {
    format!("ax-engine:{:?}", live.runtime_report.selected_backend)
}

fn ollama_model_details(live: &LiveState) -> OllamaModelDetails {
    let family = ollama_model_family(live.model_id.as_ref());
    OllamaModelDetails {
        parent_model: String::new(),
        format: MODEL_OWNER.to_string(),
        family: family.clone(),
        families: vec![family],
        parameter_size: "unknown".to_string(),
        quantization_level: "unknown".to_string(),
    }
}

fn ollama_model_info(live: &LiveState) -> Value {
    let family = ollama_model_family(live.model_id.as_ref());
    let mut info = json!({
        "general.architecture": family,
        "general.name": live.model_id.as_ref(),
        "ax_engine.backend": format!("{:?}", live.runtime_report.selected_backend),
        "ax_engine.context_length": context_length(live),
        "ax_engine.max_batch_tokens": live.session_config.max_batch_tokens,
    });
    // Real Ollama exposes the window as `<architecture>.context_length`;
    // Ollama clients (including OpenClaw) read that key for the context-size
    // default, so publish it alongside the AX-native keys.
    info[format!("{family}.context_length")] = json!(context_length(live));
    info
}

fn ollama_capabilities(live: &LiveState) -> Vec<&'static str> {
    let mut capabilities = vec!["completion"];
    if model_supports_image(live) {
        capabilities.push("vision");
    }
    if model_supports_reasoning(live) {
        capabilities.push("thinking");
    }
    if ollama_tools_supported(live) {
        capabilities.push("tools");
    }
    capabilities
}

fn ollama_tools_supported(live: &LiveState) -> bool {
    // Delegate to the shared capability probe so `/api/show` and `/api/chat`
    // gating always agree with `/v1/models` `capabilities.toolcall`.
    crate::metadata::model_supports_tool_calling(live)
}

fn ollama_parameters(live: &LiveState) -> String {
    format!(
        "num_ctx {}\nnum_batch {}",
        context_length(live),
        live.session_config.max_batch_tokens
    )
}

fn ollama_modelfile(live: &LiveState) -> String {
    format!(
        "# AX Engine Ollama-compatible view for {}\nFROM {}",
        live.model_id.as_ref(),
        live.model_id.as_ref()
    )
}

fn ollama_template_hint(live: &LiveState) -> String {
    match crate::chat::resolve_chat_template(
        live.model_id.as_ref(),
        crate::metadata::model_family_from_artifacts(live).as_deref(),
    ) {
        ChatPromptTemplate::QwenChatMl => "qwen-chatml".to_string(),
        ChatPromptTemplate::Gemma4 => "gemma4".to_string(),
        ChatPromptTemplate::Llama3 => "llama3".to_string(),
        ChatPromptTemplate::Llama4 => "llama4".to_string(),
        ChatPromptTemplate::Glm47 => "glm".to_string(),
        ChatPromptTemplate::MistralInstruct => "mistral".to_string(),
        ChatPromptTemplate::MinistralInstruct => "ministral".to_string(),
        ChatPromptTemplate::GptOssHarmony => "gpt-oss-harmony".to_string(),
        ChatPromptTemplate::MuseGlimmerAtem => "muse-glimmer-atem".to_string(),
        ChatPromptTemplate::DeepSeekChat => "deepseek".to_string(),
        ChatPromptTemplate::MiniMaxM3 => "minimax-m3".to_string(),
        ChatPromptTemplate::Unsupported(family) => family.label().to_string(),
        ChatPromptTemplate::PlainRolePrefix => "plain".to_string(),
    }
}

fn default_ollama_stream() -> bool {
    true
}

fn rfc3339_now() -> String {
    rfc3339_from_unix(unix_timestamp_secs())
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn rfc3339_from_unix(timestamp: u64) -> String {
    let days = (timestamp / 86_400) as i64;
    let seconds_of_day = (timestamp % 86_400) as u32;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3600;
    let minute = (seconds_of_day % 3600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

// Howard Hinnant's civil-from-days conversion for proleptic Gregorian UTC dates.
fn civil_from_days(days_since_unix_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year as i32, m as u32, d as u32)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn negative_seed_is_unseeded_and_model_not_found_maps_to_404() {
        let request: OllamaGenerateRequest = serde_json::from_value(json!({
            "model": "m", "prompt": "hi", "options": {"seed": -1}
        }))
        .expect("negative seed parses");
        assert_eq!(request.options.seed, Some(-1));
        assert_eq!(
            request
                .options
                .seed
                .and_then(|seed| u64::try_from(seed).ok()),
            None
        );
        let request: OllamaGenerateRequest = serde_json::from_value(json!({
            "model": "m", "prompt": "hi", "options": {"seed": 7}
        }))
        .expect("positive seed parses");
        assert_eq!(
            request
                .options
                .seed
                .and_then(|seed| u64::try_from(seed).ok()),
            Some(7)
        );

        let (status, _) = ollama_model_status(error_response(
            StatusCode::BAD_REQUEST,
            "model_not_found",
            "missing".to_string(),
        ));
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = ollama_model_status(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "bad".to_string(),
        ));
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
    use crate::openai::schema::{
        OpenAiChatCompletionChoice, OpenAiChatMessageResponse, OpenAiFunctionCall, OpenAiToolCall,
        OpenAiUsage,
    };

    #[test]
    fn ollama_format_wraps_bare_json_schema_and_rejects_unknown_values() {
        assert_eq!(ollama_format_to_openai(None).unwrap(), None);
        assert_eq!(ollama_format_to_openai(Some(Value::Null)).unwrap(), None);
        assert_eq!(
            ollama_format_to_openai(Some(json!("JSON"))).unwrap(),
            Some(json!({"type": "json_object"}))
        );
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let wrapped = ollama_format_to_openai(Some(schema.clone()))
            .unwrap()
            .unwrap();
        assert_eq!(wrapped["type"], "json_schema");
        assert_eq!(wrapped["json_schema"]["schema"], schema);
        // Already OpenAI-shaped wrappers pass through untouched.
        let wrapper = json!({"type": "json_object"});
        assert_eq!(
            ollama_format_to_openai(Some(wrapper.clone())).unwrap(),
            Some(wrapper)
        );
        let (status, _) = ollama_format_to_openai(Some(json!("yaml"))).unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn ollama_final_chunks_omit_done_reason_when_unknown() {
        let mut response = OllamaChatResponse {
            model: "m".to_string(),
            created_at: "now".to_string(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                images: None,
                thinking: None,
                tool_calls: None,
                tool_name: None,
                unsupported: BTreeMap::new(),
            },
            done: true,
            done_reason: None,
            total_duration: 0,
            load_duration: 0,
            prompt_eval_count: 0,
            prompt_eval_duration: 0,
            eval_count: 0,
            eval_duration: 0,
        };
        let chunk = ollama_chat_final_chunk(&response);
        assert!(chunk.get("done_reason").is_none(), "{chunk}");
        response.done_reason = Some("stop");
        assert_eq!(ollama_chat_final_chunk(&response)["done_reason"], "stop");
    }

    #[test]
    fn ollama_chat_request_maps_to_openai_tool_request() {
        let request = OllamaChatRequest {
            model: Some("qwen3".to_string()),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
                images: None,
                thinking: None,
                tool_calls: None,
                tool_name: None,
                unsupported: BTreeMap::new(),
            }],
            stream: true,
            options: OllamaOptions {
                num_predict: Some(16),
                temperature: Some(0.2),
                top_p: Some(0.9),
                stop: Some(OllamaStopInput::Multiple(vec!["</tool_call>".to_string()])),
                ..Default::default()
            },
            tools: Some(
                json!([{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]),
            ),
            format: Some(json!("json")),
            keep_alive: None,
            think: None,
            unsupported: BTreeMap::new(),
        };

        let openai = ollama_chat_to_openai_request(
            request,
            None,
            OllamaNumPredict {
                max_tokens: Some(16),
                fill_context: false,
            },
        )
        .expect("request should map");

        assert_eq!(openai.model.as_deref(), Some("qwen3"));
        assert_eq!(openai.max_tokens, Some(16));
        assert_eq!(openai.temperature, Some(0.2));
        assert_eq!(openai.top_p, Some(0.9));
        assert!(openai.tools.is_some());
        assert_eq!(openai.response_format, Some(json!({"type": "json_object"})));
        assert!(!openai.stream);
        assert_eq!(openai.messages.len(), 1);
    }

    #[test]
    fn ollama_chat_request_rejects_more_than_inline_image_budget() {
        let images = vec!["x"; MAX_INLINE_IMAGES_PER_REQUEST + 1];
        let request: OllamaChatRequest = serde_json::from_value(json!({
            "model": "qwen3",
            "messages": [{
                "role": "user",
                "content": "describe",
                "images": images
            }]
        }))
        .expect("over-budget image request should deserialize");

        let error = ollama_chat_to_openai_request(request, None, OllamaNumPredict::default())
            .expect_err("requests above the inline image budget must be rejected");
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(
            error
                .1
                .0
                .error
                .message
                .contains("more than 40 inline images"),
            "unexpected message: {}",
            error.1.0.error.message
        );
    }

    #[test]
    fn ollama_chat_request_counts_inline_images_across_messages() {
        let first = vec!["x"; 20];
        let second = vec!["x"; 21];
        let request: OllamaChatRequest = serde_json::from_value(json!({
            "model": "qwen3",
            "messages": [
                {
                    "role": "user",
                    "content": "one",
                    "images": first
                },
                {
                    "role": "user",
                    "content": "two",
                    "images": second
                }
            ]
        }))
        .expect("split over-budget image request should deserialize");

        let error = ollama_chat_to_openai_request(request, None, OllamaNumPredict::default())
            .expect_err("the per-request budget applies across messages");
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(
            error
                .1
                .0
                .error
                .message
                .contains("more than 40 inline images"),
            "unexpected message: {}",
            error.1.0.error.message
        );
    }

    #[test]
    fn openclaw_native_ollama_request_maps_images_thinking_and_tool_ids() {
        let request: OllamaChatRequest = serde_json::from_value(json!({
            "model": "Qwen3.6-27B-4bit",
            "stream": true,
            "think": "low",
            "options": {"num_ctx": 16384},
            "messages": [
                {
                    "role": "user",
                    "content": "describe",
                    "images": ["iVBORw0KGgo="]
                },
                {
                    "role": "assistant",
                    "content": "",
                    "thinking": "inspect the image",
                    "tool_calls": [{
                        "id": "call_openclaw_1",
                        "function": {
                            "name": "record",
                            "arguments": {"finding": "square"}
                        }
                    }]
                },
                {
                    "role": "tool",
                    "content": "{\"saved\":true}",
                    "tool_name": "record"
                }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "record",
                    "description": "Record a finding",
                    "parameters": {"type": "object"}
                }
            }]
        }))
        .expect("current OpenClaw Ollama request should deserialize");

        assert_eq!(request.options.num_ctx, Some(16_384));
        let thinking =
            resolve_ollama_thinking(request.think.as_ref()).expect("thinking level should map");
        assert_eq!(thinking, Some(true));
        let openai = ollama_chat_to_openai_request(request, thinking, OllamaNumPredict::default())
            .expect("request should map");

        assert_eq!(
            openai
                .chat_template_kwargs
                .map(|kwargs| kwargs.enable_thinking),
            Some(Some(true))
        );
        let OpenAiChatContent::Parts(parts) = openai.messages[0]
            .content
            .as_ref()
            .expect("image message content")
        else {
            panic!("Ollama images must become OpenAI multipart content");
        };
        assert_eq!(parts[0].text.as_deref(), Some("describe"));
        assert_eq!(
            parts[1]
                .image_url
                .as_ref()
                .and_then(|value| value.get("url"))
                .and_then(Value::as_str),
            Some("data:image/png;base64,iVBORw0KGgo=")
        );
        assert_eq!(
            openai.messages[1]
                .tool_calls
                .as_ref()
                .and_then(|calls| calls[0].get("id"))
                .and_then(Value::as_str),
            Some("call_openclaw_1")
        );
        assert_eq!(
            openai.messages[1].reasoning_content.as_deref(),
            Some("inspect the image")
        );
        assert_eq!(openai.messages[2]._name.as_deref(), Some("record"));
    }

    #[test]
    fn ollama_think_true_replays_history_and_think_false_does_not() {
        let request = |think: Value| -> OllamaChatRequest {
            serde_json::from_value(json!({
                "model": "Qwen3.6-27B-4bit",
                "think": think,
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .expect("think request should deserialize")
        };

        let on = ollama_chat_to_openai_request(
            request(json!(true)),
            Some(true),
            OllamaNumPredict::default(),
        )
        .expect("think=true should map")
        .chat_template_kwargs
        .expect("kwargs should be set");
        assert_eq!(on.enable_thinking, Some(true));
        assert_eq!(on.preserve_thinking, Some(true));

        let off = ollama_chat_to_openai_request(
            request(json!(false)),
            Some(false),
            OllamaNumPredict::default(),
        )
        .expect("think=false should map")
        .chat_template_kwargs
        .expect("kwargs should be set");
        assert_eq!(off.enable_thinking, Some(false));
        assert_eq!(
            off.preserve_thinking,
            Some(false),
            "think=false must not replay prior reasoning into a no-think prompt"
        );
    }

    #[test]
    fn ollama_chat_response_converts_openai_tool_arguments_to_object() {
        let openai = OpenAiChatCompletionResponse {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion",
            created: 0,
            model: "qwen3".to_string(),
            system_fingerprint: None,
            choices: vec![OpenAiChatCompletionChoice {
                index: 0,
                message: OpenAiChatMessageResponse {
                    role: "assistant",
                    content: String::new(),
                    reasoning_content: Some("reasoning body".to_string()),
                    tool_calls: Some(vec![OpenAiToolCall {
                        id: "call_0".to_string(),
                        tool_type: "function",
                        function: OpenAiFunctionCall {
                            name: "lookup".to_string(),
                            arguments: r#"{"query":"weather"}"#.to_string(),
                        },
                    }]),
                },
                logprobs: None,
                finish_reason: Some("tool_calls"),
            }],
            usage: Some(OpenAiUsage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
                prompt_tokens_details: None,
            }),
        };

        let ollama = ollama_chat_response_from_openai(openai).expect("response should convert");

        assert_eq!(ollama.created_at, "1970-01-01T00:00:00Z");
        assert_eq!(ollama.done_reason, Some("stop"));
        assert_eq!(ollama.prompt_eval_count, 3);
        assert_eq!(ollama.eval_count, 2);
        assert_eq!(ollama.message.thinking.as_deref(), Some("reasoning body"));
        let calls = ollama
            .message
            .tool_calls
            .expect("tool calls should be present");
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(calls[0].id.as_deref(), Some("call_0"));
        assert_eq!(calls[0].function.arguments, json!({"query": "weather"}));
    }

    #[test]
    fn rfc3339_formatter_handles_epoch_and_leap_day() {
        assert_eq!(rfc3339_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_from_unix(1_582_934_400), "2020-02-29T00:00:00Z");
    }

    #[test]
    fn ollama_model_family_detects_supported_families() {
        assert_eq!(ollama_model_family("ax-engine/qwen3-coder-next"), "qwen");
        assert_eq!(ollama_model_family("google-gemma-4-it"), "gemma");
        assert_eq!(ollama_model_family("glm-4.7"), "glm");
    }

    #[test]
    fn ollama_value_presence_treats_numeric_zero_as_false() {
        assert!(!ollama_value_is_present(Some(&json!(0))));
        assert!(!ollama_value_is_present(Some(&json!(0.0))));
        assert!(ollama_value_is_present(Some(&json!(1))));
        assert!(ollama_value_is_present(Some(&json!(-1))));
        assert!(ollama_value_is_present(Some(&json!(0.5))));
    }

    #[test]
    fn keep_alive_unload_treats_numeric_zero_as_unload() {
        assert!(keep_alive_requests_unload(Some(&json!(0))));
        assert!(keep_alive_requests_unload(Some(&json!(0.0))));
        assert!(keep_alive_requests_unload(Some(&json!("0s"))));
        assert!(!keep_alive_requests_unload(Some(&json!(1))));
        assert!(!keep_alive_requests_unload(Some(&json!(0.5))));
        // A positive duration keeps the model resident (no unload).
        assert!(!keep_alive_requests_unload(Some(&json!("5m"))));
    }

    #[test]
    fn num_ctx_budget_rejects_oversized_prompt_and_allows_fit() {
        // No num_ctx: the budget check is vacuous.
        assert!(enforce_ollama_num_ctx_prompt_budget(None, &[1, 2, 3]).is_ok());
        // Prompt fits exactly within the budget.
        assert!(enforce_ollama_num_ctx_prompt_budget(Some(3), &[1, 2, 3]).is_ok());
        // Prompt longer than the budget must fail closed (AX never truncates).
        let (status, Json(body)) =
            enforce_ollama_num_ctx_prompt_budget(Some(2), &[1, 2, 3]).expect_err("oversized");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body.error.message.contains("does not truncate prompts"),
            "unexpected message: {}",
            body.error.message
        );
    }

    #[test]
    fn negative_num_ctx_is_unset_sentinel() {
        let request: OllamaGenerateRequest = serde_json::from_value(json!({
            "model": "m", "prompt": "hi", "options": {"num_ctx": -1}
        }))
        .expect("negative num_ctx parses as the unset sentinel");
        assert_eq!(request.options.num_ctx, Some(-1));
        // Negative maps to `None` (unset) in validate_ollama_num_ctx, the same
        // mapping used for the signed `seed` field.
        assert_eq!(
            request.options.num_ctx.and_then(|v| u32::try_from(v).ok()),
            None
        );
        let request: OllamaGenerateRequest = serde_json::from_value(json!({
            "model": "m", "prompt": "hi", "options": {"num_ctx": 2048}
        }))
        .expect("positive num_ctx parses");
        assert_eq!(
            request.options.num_ctx.and_then(|v| u32::try_from(v).ok()),
            Some(2048)
        );
    }

    #[test]
    fn generate_stream_final_chunk_omits_context() {
        // Per the Ollama API spec, the context field is only present in
        // non-streaming responses, not in streaming final chunks.
        let response = OllamaGenerateResponse {
            model: "test".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            response: String::new(),
            done: true,
            done_reason: Some("stop"),
            context: vec![1, 2, 3],
            total_duration: 100,
            load_duration: 10,
            prompt_eval_count: 3,
            prompt_eval_duration: 20,
            eval_count: 5,
            eval_duration: 70,
        };
        let chunk = ollama_generate_final_chunk(&response);
        assert!(
            chunk.get("context").is_none(),
            "streaming final chunk must not include context field"
        );
        assert_eq!(chunk["done"], json!(true));
    }
}
