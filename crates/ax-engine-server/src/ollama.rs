use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::GenerateResponse;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::app_state::{AppState, LiveState};
use crate::backends::{llama_cpp, mlx_lm};
use crate::errors::{ErrorResponse, error_response, map_session_error};
use crate::generation::native::run_stateless_generate_request;
use crate::metadata::MODEL_OWNER;
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
    _keep_alive: Option<Value>,
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
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct OllamaFunctionCall {
    name: String,
    #[serde(default)]
    arguments: Value,
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
    _context: Option<Vec<u32>>,
    #[serde(default)]
    _raw: Option<bool>,
    #[serde(default)]
    _keep_alive: Option<Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaTagsResponse {
    models: Vec<OllamaModelTag>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OllamaModelTag {
    name: String,
    model: String,
    modified_at: String,
    size: u64,
    digest: String,
    details: OllamaModelDetails,
}

#[derive(Debug, Serialize)]
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
    context: Vec<u32>,
    total_duration: u64,
    load_duration: u64,
    prompt_eval_count: u32,
    prompt_eval_duration: u64,
    eval_count: u32,
    eval_duration: u64,
}

pub(crate) async fn ollama_tags(State(state): State<AppState>) -> Json<OllamaTagsResponse> {
    let live = state.snapshot();
    let model = live.model_id.to_string();
    Json(OllamaTagsResponse {
        models: vec![OllamaModelTag {
            name: model.clone(),
            model,
            modified_at: rfc3339_now(),
            size: 0,
            digest: format!("ax-engine:{:?}", live.runtime_report.selected_backend),
            details: OllamaModelDetails {
                parent_model: String::new(),
                format: MODEL_OWNER.to_string(),
                family: ollama_model_family(live.model_id.as_ref()),
                families: vec![ollama_model_family(live.model_id.as_ref())],
                parameter_size: "unknown".to_string(),
                quantization_level: "unknown".to_string(),
            },
        }],
    })
}

pub(crate) async fn ollama_chat(
    State(state): State<AppState>,
    Json(request): Json<OllamaChatRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    validate_openai_request(&live, request.model.as_deref())?;
    let stream = request.stream;
    let openai_request = ollama_chat_to_openai_request(request)?;
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
    let stream = request.stream;
    let openai_request = ollama_generate_to_openai_request(request)?;
    let ollama = run_ollama_completion(state, live, openai_request).await?;
    if stream {
        return ollama_ndjson_response(vec![
            ollama_generate_stream_chunk(&ollama),
            ollama_generate_final_chunk(&ollama),
        ]);
    }
    Ok(Json(ollama).into_response())
}

fn ollama_chat_to_openai_request(
    request: OllamaChatRequest,
) -> Result<OpenAiChatCompletionHttpRequest, (StatusCode, Json<ErrorResponse>)> {
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
    reject_unused_field(request.images, "images")?;
    reject_unused_field(request.template, "template")?;
    let prompt = match request.system {
        Some(system) if !system.trim().is_empty() => {
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
        logprobs: None,
        top_logprobs: None,
        metadata: Some("ollama:/api/generate".to_string()),
        multimodal_inputs: Default::default(),
        response_format: ollama_format_to_openai(request.format),
    })
}

fn ollama_message_to_openai_message(
    message: OllamaMessage,
) -> Result<OpenAiChatMessage, (StatusCode, Json<ErrorResponse>)> {
    reject_unused_field(message.images, "messages[].images")?;
    let tool_calls = message
        .tool_calls
        .map(|calls| serde_json::to_value(calls).unwrap_or(Value::Null));
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
        let response = run_blocking_session_task(move || {
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
        let response = run_blocking_session_task(move || {
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
        },
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
        "done": true,
        "done_reason": response.done_reason,
        "context": response.context,
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
            _keep_alive: None,
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
}
