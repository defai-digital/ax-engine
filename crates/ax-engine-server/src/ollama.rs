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
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::app_state::{AppState, LiveState};
use crate::backends::{llama_cpp, mlx_lm};
use crate::chat::{ChatPromptTemplate, Gemma4ChannelIds, strip_gemma4_channel_name_header};
use crate::errors::{ErrorResponse, admission_error_response, error_response, map_session_error};
use crate::generation::native::run_stateless_generate_request;
use crate::generation::streaming::{StreamStateSource, build_stream_state};
use crate::metadata::{MODEL_OWNER, context_length};
use crate::openai::generation::{
    populate_native_mlx_output_text, validate_openai_json_object_response,
};
use crate::openai::requests::{
    OpenAiBuiltLlamaCppChatRequest, OpenAiBuiltMlxLmChatRequest, OpenAiBuiltRequest,
    build_openai_chat_request_offloading_media, build_openai_completion_request,
    build_openai_llama_cpp_chat_request, build_openai_mlx_lm_chat_request,
};
use crate::openai::responses::{openai_chat_completion_response, openai_finish_reason};
use crate::openai::schema::{
    OpenAiChatCompletionHttpRequest, OpenAiChatCompletionResponse, OpenAiChatContent,
    OpenAiChatMessage, OpenAiCompletionHttpRequest, OpenAiPromptInput, OpenAiStopInput,
    OpenAiStreamKind, OpenAiToolCall,
};
use crate::openai::streaming::{Gemma4ChannelStreamFilter, IncrementalDecoder};
use crate::openai::validation::validate_openai_request;
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
    tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(default, flatten, skip_serializing_if = "BTreeMap::is_empty")]
    unsupported: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct OllamaToolCall {
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
    #[serde(default)]
    num_predict: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    repeat_last_n: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
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
    #[serde(default)]
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
    let live = state.snapshot();
    Json(OllamaTagsResponse {
        models: vec![ollama_model_tag(&live)],
    })
}

pub(crate) async fn ollama_show(
    State(state): State<AppState>,
    Json(request): Json<OllamaShowRequest>,
) -> Result<Json<OllamaShowResponse>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    reject_unsupported_fields(&request.unsupported, "request")?;
    if request.verbose == Some(true) {
        reject_unused_field(request.verbose, "verbose")?;
    }
    validate_openai_request(&live, request.model.as_deref())?;
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
    let live = state.snapshot();
    let tag = ollama_model_tag(&live);
    Json(OllamaPsResponse {
        models: vec![OllamaRunningModel {
            name: tag.name,
            model: tag.model,
            size: tag.size,
            digest: tag.digest,
            details: tag.details,
            expires_at: rfc3339_now(),
            size_vram: 0,
            context_length: context_length(&live),
        }],
    })
}

pub(crate) async fn ollama_version() -> Json<OllamaVersionResponse> {
    Json(OllamaVersionResponse {
        version: env!("CARGO_PKG_VERSION"),
    })
}

pub(crate) async fn ollama_chat(
    State(state): State<AppState>,
    Json(request): Json<OllamaChatRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    validate_openai_request(&live, request.model.as_deref())?;
    reject_ollama_tools_without_support(&live, request.tools.as_ref())?;
    let stream = request.stream;
    // True per-token NDJSON streaming for the native MLX backend. Tool and
    // format requests keep the buffered two-chunk emulation because their
    // responses require whole-output post-processing (tool-call extraction,
    // JSON-object validation); delegated backends keep it because their
    // Ollama adapters run through blocking chat completion.
    let can_true_stream = stream
        && live.runtime_report.selected_backend == SelectedBackend::Mlx
        && request.tools.is_none()
        && request.format.is_none();
    let openai_request = ollama_chat_to_openai_request(request)?;
    if can_true_stream {
        let OpenAiBuiltRequest {
            generate_request, ..
        } = build_openai_chat_request_offloading_media(&live, openai_request).await?;
        return stream_ollama_native(state, live, generate_request, OllamaNativeStreamKind::Chat)
            .await;
    }
    let response = run_ollama_chat_completion(state, live, openai_request).await?;
    let ollama = ollama_chat_response_from_openai(response)?;
    if stream {
        return ollama_ndjson_response(vec![
            ollama_chat_stream_chunk(&ollama),
            ollama_chat_final_chunk(&ollama),
        ]);
    }
    Ok(Json(ollama).into_response())
}

pub(crate) async fn ollama_generate(
    State(state): State<AppState>,
    Json(request): Json<OllamaGenerateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    validate_openai_request(&live, request.model.as_deref())?;
    if let Some(response) = ollama_generate_lifecycle_response(&live, &request) {
        return Ok(Json(response).into_response());
    }
    let stream = request.stream;
    let can_true_stream = stream
        && live.runtime_report.selected_backend == SelectedBackend::Mlx
        && request.format.is_none();
    let openai_request = ollama_generate_to_openai_request(request)?;
    if can_true_stream {
        let OpenAiBuiltRequest {
            generate_request, ..
        } = build_openai_completion_request(&live, openai_request)?;
        return stream_ollama_native(
            state,
            live,
            generate_request,
            OllamaNativeStreamKind::Generate,
        )
        .await;
    }
    let ollama = run_ollama_completion(state, live, openai_request).await?;
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
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
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
    let monitor_tx = tx.clone();
    tokio::spawn(async move {
        monitor_tx.closed().await;
        cancel_monitor.store(true, Ordering::Relaxed);
    });
    match stream_context {
        StreamStateSource::Service(mut events) => {
            tokio::task::spawn_blocking(move || {
                drive_ollama_native_events(kind, &tx, &cancel, tokenizer, || {
                    events.blocking_recv().transpose()
                });
            });
        }
        StreamStateSource::Stateless {
            mut state,
            context,
            permit,
        } => {
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                drive_ollama_native_events(kind, &tx, &cancel, tokenizer, || {
                    context.next_stream_event(&mut state)
                });
            });
        }
        StreamStateSource::Stateful {
            mut state,
            mut session,
            permit,
        } => {
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                drive_ollama_native_events(kind, &tx, &cancel, tokenizer, || {
                    session.next_stream_event(&mut state)
                });
            });
        }
    }

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
    mut next: N,
) where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    // Chat streams strip Gemma 4 thinking-channel framing (mirrors the OpenAI
    // SSE chat path); raw generate streams keep the verbatim decode.
    let mut channel_filter = match kind {
        OllamaNativeStreamKind::Chat => Gemma4ChannelIds::from_tokenizer(&tokenizer)
            .map(|ids| Gemma4ChannelStreamFilter::new(ids, tokenizer.token_to_id("thought"))),
        OllamaNativeStreamKind::Generate => None,
    };
    let mut decoder = IncrementalDecoder::new(tokenizer);
    loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::debug!("Ollama stream cancelled: client disconnected");
            return;
        }
        match next() {
            Ok(None) => return,
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
                    filter.kept_output = true;
                }
                let chunk = ollama_native_delta_chunk(kind, &payload.request.model_id, delta_text);
                if send_ollama_ndjson_line(tx, &chunk).is_err() {
                    return;
                }
            }
            Ok(Some(GenerateStreamEvent::Response(payload))) => {
                // The model can leave its entire answer inside an unclosed
                // thinking channel; serve that body before the final chunk.
                if let Some(filter) = channel_filter.as_mut()
                    && let Some(body_tokens) = filter.take_fallback_tokens()
                    && let Ok(body_text) = decoder.push(&body_tokens)
                {
                    let body_text = strip_gemma4_channel_name_header(&body_text);
                    if !body_text.is_empty() {
                        let chunk = ollama_native_delta_chunk(
                            kind,
                            &payload.response.model_id,
                            body_text.to_string(),
                        );
                        if send_ollama_ndjson_line(tx, &chunk).is_err() {
                            return;
                        }
                    }
                }
                let summary = ollama_generate_response_from_generate(payload.response);
                let final_chunk = match kind {
                    OllamaNativeStreamKind::Chat => json!({
                        "model": summary.model,
                        "created_at": summary.created_at,
                        "message": {"role": "assistant", "content": ""},
                        "done": true,
                        "done_reason": summary.done_reason,
                        "total_duration": summary.total_duration,
                        "load_duration": summary.load_duration,
                        "prompt_eval_count": summary.prompt_eval_count,
                        "prompt_eval_duration": summary.prompt_eval_duration,
                        "eval_count": summary.eval_count,
                        "eval_duration": summary.eval_duration,
                    }),
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
) -> Result<OpenAiChatCompletionHttpRequest, (StatusCode, Json<ErrorResponse>)> {
    reject_unsupported_fields(&request.unsupported, "request")?;
    reject_unsupported_fields(&request.options.unsupported, "options")?;
    let _ = request.keep_alive;
    let messages = request
        .messages
        .into_iter()
        .map(ollama_message_to_openai_message)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OpenAiChatCompletionHttpRequest {
        model: request.model,
        messages,
        input_tokens: Vec::new(),
        max_tokens: request.options.num_predict,
        temperature: request.options.temperature,
        top_p: request.options.top_p,
        top_k: request.options.top_k,
        min_p: request.options.min_p,
        repetition_penalty: request.options.repeat_penalty,
        repetition_context_size: request.options.repeat_last_n,
        stop: request.options.stop.map(OllamaStopInput::into_openai_stop),
        seed: request.options.seed,
        stream: false,
        n: None,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: false,
        top_logprobs: None,
        reasoning: None,
        metadata: Some("ollama:/api/chat".to_string()),
        multimodal_inputs: Default::default(),
        response_format: ollama_format_to_openai(request.format),
        tools: request.tools,
        tool_choice: None,
    })
}

fn ollama_generate_to_openai_request(
    request: OllamaGenerateRequest,
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
        max_tokens: request.options.num_predict,
        temperature: request.options.temperature,
        top_p: request.options.top_p,
        top_k: request.options.top_k,
        min_p: request.options.min_p,
        repetition_penalty: request.options.repeat_penalty,
        repetition_context_size: request.options.repeat_last_n,
        stop: request.options.stop.map(OllamaStopInput::into_openai_stop),
        seed: request.options.seed,
        stream: false,
        n: None,
        best_of: None,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        metadata: Some("ollama:/api/generate".to_string()),
        multimodal_inputs: Default::default(),
        response_format: ollama_format_to_openai(request.format),
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
    reject_unused_field(message.images, "messages[].images")?;
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
        content: Some(OpenAiChatContent::Text(message.content)),
        tool_calls,
        _tool_call_id: None,
        _name: None,
    })
}

fn ollama_format_to_openai(format: Option<Value>) -> Option<Value> {
    let format = format?;
    match format {
        Value::String(value)
            if value.eq_ignore_ascii_case("json") || value.eq_ignore_ascii_case("json_object") =>
        {
            Some(json!({"type": "json_object"}))
        }
        Value::Null => None,
        value => Some(value),
    }
}

async fn run_ollama_chat_completion(
    state: AppState,
    live: LiveState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiChatCompletionResponse, (StatusCode, Json<ErrorResponse>)> {
    if mlx_lm::is_selected(&live) {
        let OpenAiBuiltMlxLmChatRequest {
            chat_request,
            response_options,
            ..
        } = build_openai_mlx_lm_chat_request(&live, request)?;
        let request_id = state.allocate_request_id();
        let runtime = live.runtime_report.clone();
        let mlx_lm_backend = mlx_lm::config(&live).map_err(map_session_error)?;
        let permit = state
            .admission
            .try_admit()
            .map_err(admission_error_response)?;
        let response = run_blocking_session_task(move || {
            let _permit = permit;
            mlx_lm::run_chat_generate(request_id, &runtime, &mlx_lm_backend, &chat_request)
        })
        .await?;
        validate_openai_json_object_response(&response, &response_options)?;
        return Ok(openai_chat_completion_response(
            &response,
            OpenAiStreamKind::ChatCompletion.response_id(request_id),
            response_options,
            None,
        ));
    }

    if llama_cpp::supports_server_chat(&live) {
        let OpenAiBuiltLlamaCppChatRequest {
            chat_request,
            response_options,
            ..
        } = build_openai_llama_cpp_chat_request(&live, request)?;
        let request_id = state.allocate_request_id();
        let runtime = live.runtime_report.clone();
        let llama_backend = llama_cpp::config(&live).map_err(map_session_error)?;
        let permit = state
            .admission
            .try_admit()
            .map_err(admission_error_response)?;
        let response = run_blocking_session_task(move || {
            let _permit = permit;
            llama_cpp::run_chat_generate(request_id, &runtime, &llama_backend, &chat_request)
        })
        .await?;
        validate_openai_json_object_response(&response, &response_options)?;
        return Ok(openai_chat_completion_response(
            &response,
            OpenAiStreamKind::ChatCompletion.response_id(request_id),
            response_options,
            None,
        ));
    }

    let OpenAiBuiltRequest {
        generate_request,
        response_options,
        ..
    } = build_openai_chat_request_offloading_media(&live, request).await?;
    let (request_id, mut response) =
        run_stateless_generate_request(&state, &live, generate_request).await?;
    let native_reasoning = populate_native_mlx_output_text(
        &live,
        &mut response,
        OpenAiStreamKind::ChatCompletion,
        response_options.include_reasoning,
    )?;
    validate_openai_json_object_response(&response, &response_options)?;
    Ok(openai_chat_completion_response(
        &response,
        OpenAiStreamKind::ChatCompletion.response_id(request_id),
        response_options,
        native_reasoning,
    ))
}

async fn run_ollama_completion(
    state: AppState,
    live: LiveState,
    request: OpenAiCompletionHttpRequest,
) -> Result<OllamaGenerateResponse, (StatusCode, Json<ErrorResponse>)> {
    let OpenAiBuiltRequest {
        generate_request,
        response_options,
        ..
    } = build_openai_completion_request(&live, request)?;
    let (_request_id, mut response) =
        run_stateless_generate_request(&state, &live, generate_request).await?;
    let native_reasoning = populate_native_mlx_output_text(
        &live,
        &mut response,
        OpenAiStreamKind::Completion,
        response_options.include_reasoning,
    )?;
    validate_openai_json_object_response(&response, &response_options)?;
    debug_assert!(native_reasoning.is_none());
    Ok(ollama_generate_response_from_generate(response))
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
            tool_calls,
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
        function: OllamaFunctionCall {
            name: call.function.name,
            arguments,
            unsupported: BTreeMap::new(),
        },
        unsupported: BTreeMap::new(),
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
    json!({
        "model": response.model,
        "created_at": response.created_at,
        // Real Ollama always includes `message` (with empty content) on the
        // terminal `done: true` line; a client parsing every NDJSON line
        // with a uniform schema would otherwise fail on the last line.
        "message": {"role": "assistant", "content": ""},
        "done": true,
        "done_reason": response.done_reason,
        "total_duration": response.total_duration,
        "load_duration": response.load_duration,
        "prompt_eval_count": response.prompt_eval_count,
        "prompt_eval_duration": response.prompt_eval_duration,
        "eval_count": response.eval_count,
        "eval_duration": response.eval_duration,
    })
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
    json!({
        "model": response.model,
        "created_at": response.created_at,
        // Real Ollama always includes `response` (empty string) on the
        // terminal `done: true` line; a client parsing every NDJSON line
        // with a uniform schema would otherwise fail on the last line.
        "response": "",
        "done": true,
        "done_reason": response.done_reason,
        "total_duration": response.total_duration,
        "load_duration": response.load_duration,
        "prompt_eval_count": response.prompt_eval_count,
        "prompt_eval_duration": response.prompt_eval_duration,
        "eval_count": response.eval_count,
        "eval_duration": response.eval_duration,
    })
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
    json!({
        "general.architecture": ollama_model_family(live.model_id.as_ref()),
        "general.name": live.model_id.as_ref(),
        "ax_engine.backend": format!("{:?}", live.runtime_report.selected_backend),
        "ax_engine.context_length": context_length(live),
        "ax_engine.max_batch_tokens": live.session_config.max_batch_tokens,
    })
}

fn ollama_capabilities(live: &LiveState) -> Vec<&'static str> {
    let mut capabilities = vec!["completion"];
    if ollama_tools_supported(live) {
        capabilities.push("tools");
    }
    capabilities
}

fn ollama_tools_supported(live: &LiveState) -> bool {
    live.runtime_report.selected_backend == SelectedBackend::Mlx
        && matches!(
            ChatPromptTemplate::for_model_id(live.model_id.as_ref()),
            ChatPromptTemplate::QwenChatMl | ChatPromptTemplate::Gemma4
        )
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
    match ChatPromptTemplate::for_model_id(live.model_id.as_ref()) {
        ChatPromptTemplate::QwenChatMl => "qwen-chatml".to_string(),
        ChatPromptTemplate::Gemma4 => "gemma4".to_string(),
        ChatPromptTemplate::Llama3 => "llama3".to_string(),
        ChatPromptTemplate::Glm47 => "glm".to_string(),
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
mod tests {
    use super::*;
    use crate::openai::schema::{
        OpenAiChatCompletionChoice, OpenAiChatMessageResponse, OpenAiFunctionCall, OpenAiToolCall,
        OpenAiUsage,
    };

    #[test]
    fn ollama_chat_request_maps_to_openai_tool_request() {
        let request = OllamaChatRequest {
            model: Some("qwen3".to_string()),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
                images: None,
                tool_calls: None,
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
            unsupported: BTreeMap::new(),
        };

        let openai = ollama_chat_to_openai_request(request).expect("request should map");

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
                    reasoning_content: None,
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
            }),
        };

        let ollama = ollama_chat_response_from_openai(openai).expect("response should convert");

        assert_eq!(ollama.created_at, "1970-01-01T00:00:00Z");
        assert_eq!(ollama.done_reason, Some("stop"));
        assert_eq!(ollama.prompt_eval_count, 3);
        assert_eq!(ollama.eval_count, 2);
        let calls = ollama
            .message
            .tool_calls
            .expect("tool calls should be present");
        assert_eq!(calls[0].function.name, "lookup");
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
