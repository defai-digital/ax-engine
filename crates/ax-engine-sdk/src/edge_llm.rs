use std::io::{BufRead, BufReader, Read};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::backend::{RuntimeReport, SelectedBackend};
use crate::delegated_http::{
    DelegatedHttpPostError, DelegatedHttpTimeouts, normalize_base_url, parse_json_response,
    send_json_post_request,
};
use crate::generate::{
    GenerateFinishReason, GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateStatus,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EdgeLlmConfig {
    ServerCompletion(EdgeLlmServerCompletionConfig),
}

impl EdgeLlmConfig {
    pub fn server_completion(base_url: impl Into<String>) -> Self {
        Self::ServerCompletion(EdgeLlmServerCompletionConfig::new(base_url))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeLlmServerCompletionConfig {
    pub base_url: String,
    pub timeouts: DelegatedHttpTimeouts,
}

impl EdgeLlmServerCompletionConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: normalize_base_url(base_url.into()),
            timeouts: DelegatedHttpTimeouts::default(),
        }
    }

    pub fn with_timeouts(mut self, timeouts: DelegatedHttpTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    pub fn completions_url(&self) -> String {
        format!("{}/v1/completions", self.base_url)
    }

    pub fn chat_completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }
}

#[derive(Debug, Error)]
pub enum EdgeLlmBackendError {
    #[error("tensorrt-edge-llm backend expected {configured_backend:?}, got {resolved_backend:?}")]
    BackendConfigMismatch {
        configured_backend: SelectedBackend,
        resolved_backend: SelectedBackend,
    },
    #[error("tensorrt-edge-llm delegated backend requires input_text; token-array prompts need AX MLX mode")]
    MissingInputText,
    #[error("tensorrt-edge-llm delegated backend does not accept input_tokens in this preview contract")]
    UnsupportedTokenPrompt,
    #[error("failed to serialize tensorrt-edge-llm request JSON for {endpoint}: {source}")]
    SerializeRequestJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("tensorrt-edge-llm backend HTTP request to {endpoint} failed: {source}")]
    HttpRequest {
        endpoint: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error("tensorrt-edge-llm backend HTTP response from {endpoint} returned status {status}: {body}")]
    HttpStatus {
        endpoint: String,
        status: u16,
        body: String,
    },
    #[error("tensorrt-edge-llm backend HTTP response from {endpoint} was not valid JSON: {source}")]
    InvalidResponseJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("tensorrt-edge-llm backend HTTP response from {endpoint} did not include a completion choice")]
    MissingCompletionChoice { endpoint: String },
    #[error("tensorrt-edge-llm backend SSE read from {endpoint} failed: {source}")]
    SseRead {
        endpoint: String,
        #[source]
        source: std::io::Error,
    },
    #[error("tensorrt-edge-llm backend SSE chunk from {endpoint} was not valid JSON: {source}")]
    InvalidStreamChunk {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("tensorrt-edge-llm backend SSE stream from {endpoint} did not include a choice in chunk")]
    MissingStreamChoice { endpoint: String },
}

#[derive(Debug)]
pub struct EdgeLlmStreamChunkResult {
    pub text: String,
    pub finish_reason: Option<String>,
    pub prompt_token_count: Option<u32>,
    pub output_token_count: Option<u32>,
}

pub struct EdgeLlmStreamHandle {
    endpoint: String,
    reader: BufReader<Box<dyn Read + Send>>,
}

impl EdgeLlmStreamHandle {
    pub(crate) fn new(endpoint: String, reader: Box<dyn Read + Send>) -> Self {
        Self {
            endpoint,
            reader: BufReader::new(reader),
        }
    }

    pub fn next_chunk(&mut self) -> Result<Option<EdgeLlmStreamChunkResult>, EdgeLlmBackendError> {
        loop {
            let mut line = String::new();
            let bytes_read =
                self.reader
                    .read_line(&mut line)
                    .map_err(|source| EdgeLlmBackendError::SseRead {
                        endpoint: self.endpoint.clone(),
                        source,
                    })?;

            if bytes_read == 0 {
                return Ok(None);
            }

            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                continue;
            }

            // SSE allows optional whitespace after the colon (`data:` or `data: `).
            let data = match line.strip_prefix("data:") {
                Some(rest) => rest.strip_prefix(' ').unwrap_or(rest),
                None => continue,
            };

            if data == "[DONE]" {
                return Ok(None);
            }

            let chunk: EdgeLlmStreamChunk = serde_json::from_str(data).map_err(|source| {
                EdgeLlmBackendError::InvalidStreamChunk {
                    endpoint: self.endpoint.clone(),
                    source,
                }
            })?;

            let has_usage = chunk.usage.is_some();
            let choice = chunk.choices.into_iter().next();
            if choice.is_none() && !has_usage {
                return Err(EdgeLlmBackendError::MissingStreamChoice {
                    endpoint: self.endpoint.clone(),
                });
            }
            let choice = choice.unwrap_or_default();

            return Ok(Some(EdgeLlmStreamChunkResult {
                text: choice
                    .delta
                    .and_then(|delta| delta.content)
                    .unwrap_or(choice.text),
                finish_reason: choice.finish_reason,
                prompt_token_count: chunk.usage.as_ref().map(|u| u.prompt_tokens),
                output_token_count: chunk.usage.as_ref().map(|u| u.completion_tokens),
            }));
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct EdgeLlmChatMessage {
    pub role: String,
    pub content: String,
}

impl EdgeLlmChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EdgeLlmChatGenerateRequest {
    pub model_id: String,
    pub messages: Vec<EdgeLlmChatMessage>,
    pub max_output_tokens: u32,
    pub sampling: crate::generate::GenerateSampling,
    pub stop_sequences: Vec<String>,
    pub metadata: Option<String>,
    pub chat_template_kwargs: Option<serde_json::Value>,
}

impl std::fmt::Debug for EdgeLlmStreamHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeLlmStreamHandle")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

pub fn start_streaming_generate(
    runtime: &RuntimeReport,
    config: &EdgeLlmConfig,
    request: &GenerateRequest,
) -> Result<EdgeLlmStreamHandle, EdgeLlmBackendError> {
    ensure_edge_llm_backend(runtime)?;

    match config {
        EdgeLlmConfig::ServerCompletion(config) => {
            start_edge_llm_server_completion_stream(config, request, runtime.selected_backend)
        }
    }
}

pub fn start_streaming_chat_generate(
    runtime: &RuntimeReport,
    config: &EdgeLlmConfig,
    request: &EdgeLlmChatGenerateRequest,
) -> Result<EdgeLlmStreamHandle, EdgeLlmBackendError> {
    ensure_edge_llm_backend(runtime)?;

    match config {
        EdgeLlmConfig::ServerCompletion(config) => {
            start_edge_llm_server_chat_completion_stream(config, request, runtime.selected_backend)
        }
    }
}

fn start_edge_llm_server_completion_stream(
    config: &EdgeLlmServerCompletionConfig,
    request: &GenerateRequest,
    selected_backend: SelectedBackend,
) -> Result<EdgeLlmStreamHandle, EdgeLlmBackendError> {
    let endpoint = config.completions_url();
    let prompt = completion_prompt_text(request)?;
    let payload = build_edge_llm_completion_request(request, &prompt, true, selected_backend);

    let response = send_edge_llm_json_post_request(&endpoint, &payload, None, config.timeouts)?;
    let reader: Box<dyn Read + Send> = Box::new(response.into_reader());
    Ok(EdgeLlmStreamHandle::new(endpoint, reader))
}

fn ensure_edge_llm_backend(runtime: &RuntimeReport) -> Result<(), EdgeLlmBackendError> {
    // Shared OpenAI-compatible L2 adapter for both TensorRT Edge-LLM (Thor)
    // and TensorRT-LLM (`trtllm-serve` on desktop/datacenter CUDA).
    if !matches!(
        runtime.selected_backend,
        SelectedBackend::TensorRtEdgeLlm | SelectedBackend::TensorRtLlm
    ) {
        return Err(EdgeLlmBackendError::BackendConfigMismatch {
            configured_backend: SelectedBackend::TensorRtEdgeLlm,
            resolved_backend: runtime.selected_backend,
        });
    }

    Ok(())
}

fn trt_openai_execution_plan(backend: SelectedBackend, kind: &str) -> String {
    let prefix = match backend {
        SelectedBackend::TensorRtLlm => "tensor_rt_llm",
        _ => "tensor_rt_edge_llm",
    };
    format!("{prefix}.{kind}")
}

fn should_coerce_edge_sampler(backend: SelectedBackend) -> bool {
    // Edge-LLM C++ sampler rejects top_k==0 && top_p>=1.0; TRT-LLM OpenAI
    // server accepts standard OpenAI disabled defaults.
    matches!(backend, SelectedBackend::TensorRtEdgeLlm)
}

fn start_edge_llm_server_chat_completion_stream(
    config: &EdgeLlmServerCompletionConfig,
    request: &EdgeLlmChatGenerateRequest,
    selected_backend: SelectedBackend,
) -> Result<EdgeLlmStreamHandle, EdgeLlmBackendError> {
    let endpoint = config.chat_completions_url();
    let payload = build_edge_llm_chat_completion_request(request, true, selected_backend);

    let response = send_edge_llm_json_post_request(&endpoint, &payload, None, config.timeouts)?;
    let reader: Box<dyn Read + Send> = Box::new(response.into_reader());
    Ok(EdgeLlmStreamHandle::new(endpoint, reader))
}

fn send_edge_llm_json_post_request<T>(
    endpoint: &str,
    payload: &T,
    accept: Option<&str>,
    timeouts: DelegatedHttpTimeouts,
) -> Result<ureq::Response, EdgeLlmBackendError>
where
    T: Serialize + ?Sized,
{
    send_json_post_request(endpoint, payload, accept, timeouts, |error| match error {
        DelegatedHttpPostError::Serialize(source) => EdgeLlmBackendError::SerializeRequestJson {
            endpoint: endpoint.to_string(),
            source,
        },
        DelegatedHttpPostError::Status { status, body } => EdgeLlmBackendError::HttpStatus {
            endpoint: endpoint.to_string(),
            status,
            body,
        },
        DelegatedHttpPostError::Request(source) => EdgeLlmBackendError::HttpRequest {
            endpoint: endpoint.to_string(),
            source,
        },
    })
}

fn parse_edge_llm_json_response<T>(
    response: ureq::Response,
    endpoint: &str,
) -> Result<T, EdgeLlmBackendError>
where
    T: DeserializeOwned,
{
    parse_json_response(response, |source| EdgeLlmBackendError::InvalidResponseJson {
        endpoint: endpoint.to_string(),
        source,
    })
}

fn first_choice_for_completion<T>(endpoint: &str, choices: Vec<T>) -> Result<T, EdgeLlmBackendError> {
    choices
        .into_iter()
        .next()
        .ok_or_else(|| EdgeLlmBackendError::MissingCompletionChoice {
            endpoint: endpoint.to_string(),
        })
}

pub fn run_blocking_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &EdgeLlmConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, EdgeLlmBackendError> {
    ensure_edge_llm_backend(runtime)?;

    match config {
        EdgeLlmConfig::ServerCompletion(config) => {
            run_edge_llm_server_completion_generate(request_id, runtime, config, request)
        }
    }
}

pub fn run_blocking_chat_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &EdgeLlmConfig,
    request: &EdgeLlmChatGenerateRequest,
) -> Result<GenerateResponse, EdgeLlmBackendError> {
    ensure_edge_llm_backend(runtime)?;

    match config {
        EdgeLlmConfig::ServerCompletion(config) => {
            run_edge_llm_server_chat_completion_generate(request_id, runtime, config, request)
        }
    }
}

fn run_edge_llm_server_completion_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &EdgeLlmServerCompletionConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, EdgeLlmBackendError> {
    let endpoint = config.completions_url();
    let prompt = completion_prompt_text(request)?;
    let payload =
        build_edge_llm_completion_request(request, &prompt, false, runtime.selected_backend);

    let response = send_edge_llm_json_post_request(&endpoint, &payload, None, config.timeouts)?;
    let response: EdgeLlmCompletionResponse = parse_edge_llm_json_response(response, &endpoint)?;
    let choice = first_choice_for_completion(&endpoint, response.choices)?;

    Ok(build_edge_llm_delegated_response(
        request_id,
        &request.model_id,
        runtime,
        Some(prompt),
        choice.text,
        response.usage.as_ref().map(|usage| usage.prompt_tokens),
        response.usage.as_ref().map(|usage| usage.completion_tokens),
        finish_reason_from_edge_llm(choice.finish_reason.as_deref()),
        &trt_openai_execution_plan(runtime.selected_backend, "server_completion"),
    ))
}

fn run_edge_llm_server_chat_completion_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &EdgeLlmServerCompletionConfig,
    request: &EdgeLlmChatGenerateRequest,
) -> Result<GenerateResponse, EdgeLlmBackendError> {
    let endpoint = config.chat_completions_url();
    let payload = build_edge_llm_chat_completion_request(request, false, runtime.selected_backend);

    let response = send_edge_llm_json_post_request(&endpoint, &payload, None, config.timeouts)?;
    let response: EdgeLlmChatCompletionResponse = parse_edge_llm_json_response(response, &endpoint)?;
    let choice = first_choice_for_completion(&endpoint, response.choices)?;

    Ok(build_edge_llm_delegated_response(
        request_id,
        &request.model_id,
        runtime,
        None,
        choice.message.content,
        response.usage.as_ref().map(|usage| usage.prompt_tokens),
        response.usage.as_ref().map(|usage| usage.completion_tokens),
        finish_reason_from_edge_llm(choice.finish_reason.as_deref()),
        &trt_openai_execution_plan(runtime.selected_backend, "server_chat_completion"),
    ))
}

fn build_edge_llm_delegated_response(
    request_id: u64,
    model_id: &str,
    runtime: &RuntimeReport,
    prompt_text: Option<String>,
    output_text: String,
    prompt_token_count: Option<u32>,
    output_token_count: Option<u32>,
    finish_reason: Option<GenerateFinishReason>,
    execution_plan: &str,
) -> GenerateResponse {
    GenerateResponse {
        request_id,
        model_id: model_id.to_string(),
        prompt_tokens: Vec::new(),
        prompt_text,
        output_tokens: Vec::new(),
        output_token_logprobs: Vec::new(),
        output_text: Some(output_text),
        prompt_token_count,
        output_token_count,
        status: GenerateStatus::Finished,
        finish_reason,
        step_count: 0,
        ttft_step: None,
        route: GenerateRouteReport::with_execution_plan(execution_plan),
        runtime: runtime.clone(),
        performance: crate::generate::GeneratePerformanceReport::default(),
    }
}

fn build_edge_llm_completion_request<'a>(
    request: &'a GenerateRequest,
    prompt: &'a str,
    stream: bool,
    selected_backend: SelectedBackend,
) -> EdgeLlmCompletionRequest<'a> {
    let (top_k, top_p) =
        edge_llm_sampling_topk_topp(&request.sampling, should_coerce_edge_sampler(selected_backend));
    EdgeLlmCompletionRequest {
        model: None,
        prompt,
        max_tokens: request.max_output_tokens,
        temperature: request.sampling.temperature,
        top_p,
        top_k,
        min_p: request.sampling.min_p,
        repetition_penalty: request.sampling.repetition_penalty,
        repetition_context_size: request.sampling.repetition_context_size,
        seed: request.sampling.seed,
        stream,
        stop: request.stop_sequences.clone(),
        stream_options: stream.then_some(EdgeLlmStreamOptions {
            include_usage: true,
        }),
    }
}

fn completion_prompt_text(request: &GenerateRequest) -> Result<String, EdgeLlmBackendError> {
    if !request.input_tokens.is_empty() {
        return Err(EdgeLlmBackendError::UnsupportedTokenPrompt);
    }
    request
        .input_text
        .clone()
        .ok_or(EdgeLlmBackendError::MissingInputText)
}

/// Edge-LLM C++ sampler requires `top_k > 0` **or** `top_p < 1.0`. OpenAI-style
/// defaults (`top_k=0`, `top_p=1.0`) therefore fail closed unless coerced.
/// TensorRT-LLM OpenAI servers do not need this coercion.
fn edge_llm_sampling_topk_topp(
    sampling: &crate::generate::GenerateSampling,
    coerce_for_edge_sampler: bool,
) -> (u32, f32) {
    let mut top_k = sampling.top_k;
    let mut top_p = sampling.top_p;
    if coerce_for_edge_sampler && top_k == 0 && top_p >= 1.0 {
        // Match experimental Edge-LLM API defaults when both are "disabled".
        top_k = 50;
        top_p = 0.9;
    }
    (top_k, top_p)
}

fn build_edge_llm_chat_completion_request(
    request: &EdgeLlmChatGenerateRequest,
    stream: bool,
    selected_backend: SelectedBackend,
) -> EdgeLlmChatCompletionRequest<'_> {
    let (top_k, top_p) =
        edge_llm_sampling_topk_topp(&request.sampling, should_coerce_edge_sampler(selected_backend));
    EdgeLlmChatCompletionRequest {
        // Edge-LLM servers typically require an explicit model id.
        model: Some(request.model_id.as_str()),
        messages: &request.messages,
        max_tokens: request.max_output_tokens,
        temperature: request.sampling.temperature,
        top_p,
        top_k,
        min_p: request.sampling.min_p,
        repetition_penalty: request.sampling.repetition_penalty,
        repetition_context_size: request.sampling.repetition_context_size,
        seed: request.sampling.seed,
        stream,
        stop: request.stop_sequences.clone(),
        metadata: request.metadata.as_deref(),
        chat_template_kwargs: request.chat_template_kwargs.as_ref(),
        stream_options: stream.then_some(EdgeLlmStreamOptions {
            include_usage: true,
        }),
    }
}

pub fn finish_reason_from_edge_llm(value: Option<&str>) -> Option<GenerateFinishReason> {
    match value {
        Some("stop") => Some(GenerateFinishReason::Stop),
        Some("length") => Some(GenerateFinishReason::MaxOutputTokens),
        Some("content_filter") => Some(GenerateFinishReason::ContentFilter),
        Some(_) | None => None,
    }
}

#[derive(Debug, Deserialize, Default)]
struct EdgeLlmStreamChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    delta: Option<EdgeLlmStreamDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct EdgeLlmStreamDelta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EdgeLlmStreamChunk {
    #[serde(default)]
    choices: Vec<EdgeLlmStreamChoice>,
    #[serde(default)]
    usage: Option<EdgeLlmCompletionUsage>,
}

#[derive(Debug, Serialize)]
struct EdgeLlmCompletionRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    repetition_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_context_size: Option<u32>,
    seed: u64,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<EdgeLlmStreamOptions>,
}

#[derive(Debug, Serialize)]
struct EdgeLlmChatCompletionRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    messages: &'a [EdgeLlmChatMessage],
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    repetition_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_context_size: Option<u32>,
    seed: u64,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<EdgeLlmStreamOptions>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct EdgeLlmStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Deserialize, Default)]
struct EdgeLlmCompletionChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EdgeLlmCompletionUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct EdgeLlmCompletionResponse {
    #[serde(default)]
    choices: Vec<EdgeLlmCompletionChoice>,
    #[serde(default)]
    usage: Option<EdgeLlmCompletionUsage>,
}

#[derive(Debug, Deserialize, Default)]
struct EdgeLlmChatCompletionMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize, Default)]
struct EdgeLlmChatCompletionChoice {
    #[serde(default)]
    message: EdgeLlmChatCompletionMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EdgeLlmChatCompletionResponse {
    #[serde(default)]
    choices: Vec<EdgeLlmChatCompletionChoice>,
    #[serde(default)]
    usage: Option<EdgeLlmCompletionUsage>,
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use serde_json::Value;

    use super::*;
    use crate::backend::{BackendPolicy, ResolvedBackend, RuntimeReport};
    use crate::generate::{GenerateRequest, GenerateSampling};

    #[test]
    fn server_completion_url_normalizes_trailing_slashes() {
        let config = EdgeLlmConfig::server_completion("http://127.0.0.1:8090///");
        let EdgeLlmConfig::ServerCompletion(config) = config;

        assert_eq!(
            config.completions_url(),
            "http://127.0.0.1:8090/v1/completions"
        );
        assert_eq!(
            config.chat_completions_url(),
            "http://127.0.0.1:8090/v1/chat/completions"
        );
    }

    #[test]
    fn blocking_generate_calls_edge_llm_completions_contract() {
        let response_body = r#"{"choices":[{"text":" world","finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}"#.to_string();
        let (server_url, handle) = spawn_completion_server(response_body, |payload| {
            // Completions path may omit model; chat path sends model_id explicitly.
            assert_eq!(payload["prompt"], "hello");
            assert_eq!(payload["max_tokens"], 3);
            assert_eq!(payload["temperature"], 0.25);
            assert_eq!(payload["top_p"], 0.75);
            assert_eq!(payload["top_k"], 40);
            assert_eq!(payload["repetition_penalty"], 1.1);
            assert_eq!(payload["seed"], 42);
            assert_eq!(payload["stream"], false);
        });

        let mut request = text_request("hello");
        request.sampling.temperature = 0.25;
        request.sampling.top_p = 0.75;
        request.sampling.top_k = 40;
        request.sampling.repetition_penalty = 1.1;
        request.sampling.seed = 42;
        let response = run_blocking_generate(
            7,
            &runtime_report(),
            &EdgeLlmConfig::server_completion(server_url),
            &request,
        )
        .expect("tensorrt-edge-llm delegated completion should succeed");

        assert_eq!(response.request_id, 7);
        assert_eq!(response.prompt_text.as_deref(), Some("hello"));
        assert_eq!(response.output_text.as_deref(), Some(" world"));
        assert_eq!(response.prompt_token_count, Some(2));
        assert_eq!(response.output_token_count, Some(1));
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("tensor_rt_edge_llm.server_completion")
        );
        handle.join().expect("server thread should finish");
    }

    #[test]
    fn blocking_chat_generate_calls_edge_llm_chat_completions_contract() {
        let response_body = r#"{"choices":[{"message":{"content":"bonjour"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":1}}"#.to_string();
        let (server_url, handle) = spawn_completion_server(response_body, |payload| {
            assert_eq!(
                payload["model"], "qwen3",
                "edge-llm chat requests must forward the AX model id"
            );
            assert_eq!(payload["messages"][0]["role"], "system");
            assert_eq!(payload["messages"][0]["content"], "Be concise.");
            assert_eq!(payload["messages"][1]["role"], "user");
            assert_eq!(payload["messages"][1]["content"], "hello");
            assert_eq!(payload["max_tokens"], 3);
            assert_eq!(payload["stream"], false);
            assert_eq!(
                payload["chat_template_kwargs"],
                serde_json::json!({"enable_thinking": false})
            );
            assert!(
                payload.get("prompt").is_none(),
                "chat forwarding must not flatten messages into a completion prompt"
            );
        });

        let response = run_blocking_chat_generate(
            9,
            &runtime_report(),
            &EdgeLlmConfig::server_completion(server_url),
            &chat_request(),
        )
        .expect("tensorrt-edge-llm delegated chat completion should succeed");

        assert_eq!(response.request_id, 9);
        assert_eq!(response.prompt_text, None);
        assert_eq!(response.output_text.as_deref(), Some("bonjour"));
        assert_eq!(response.prompt_token_count, Some(8));
        assert_eq!(response.output_token_count, Some(1));
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("tensor_rt_edge_llm.server_chat_completion")
        );
        handle.join().expect("server thread should finish");
    }

    #[test]
    fn blocking_generate_rejects_token_prompts() {
        let mut request = text_request("hello");
        request.input_tokens = vec![1, 2, 3];

        let error = run_blocking_generate(
            7,
            &runtime_report(),
            &EdgeLlmConfig::server_completion("http://127.0.0.1:8090"),
            &request,
        )
        .expect_err("token prompts should fail closed");

        assert!(matches!(error, EdgeLlmBackendError::UnsupportedTokenPrompt));
    }

    #[test]
    fn chat_payload_coerces_openai_disabled_topk_topp_for_edge_llm_sampler() {
        // Edge-LLM C++ rejects top_k==0 && top_p>=1.0; adapter must coerce.
        let response_body = r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#.to_string();
        let (server_url, handle) = spawn_completion_server(response_body, |payload| {
            assert_eq!(payload["top_k"], 50);
            assert_eq!(payload["top_p"], 0.9);
        });

        let mut request = chat_request();
        request.sampling.top_k = 0;
        request.sampling.top_p = 1.0;

        run_blocking_chat_generate(
            11,
            &runtime_report(),
            &EdgeLlmConfig::server_completion(server_url),
            &request,
        )
        .expect("coerced sampling must be accepted by Edge-LLM contract");
        handle.join().expect("server thread should finish");
    }

    #[test]
    fn blocking_generate_rejects_missing_completion_choice() {
        let (server_url, handle) = spawn_completion_server(
            r#"{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}"#.to_string(),
            |_| {},
        );

        let error = run_blocking_generate(
            7,
            &runtime_report(),
            &EdgeLlmConfig::server_completion(server_url),
            &text_request("hello"),
        )
        .expect_err("missing choices should fail closed");

        assert!(matches!(
            error,
            EdgeLlmBackendError::MissingCompletionChoice { .. }
        ));
        handle.join().expect("server thread should finish");
    }

    #[test]
    fn stream_handle_parses_openai_sse_chunks_and_usage() {
        let mut stream = edge_llm_stream(
            "event: ignored\n\
             \n\
             data: {\"choices\":[{\"text\":\" hello\",\"finish_reason\":null}],\"usage\":null}\n\
             \n\
             data: {\"choices\":[{\"text\":\" world\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2,\"total_tokens\":4}}\n\
             \n\
             data: [DONE]\n\n",
        );

        let first = stream
            .next_chunk()
            .expect("first chunk should parse")
            .expect("first chunk should exist");
        assert_eq!(first.text, " hello");
        assert_eq!(first.finish_reason, None);
        assert_eq!(first.prompt_token_count, None);
        assert_eq!(first.output_token_count, None);

        let second = stream
            .next_chunk()
            .expect("second chunk should parse")
            .expect("second chunk should exist");
        assert_eq!(second.text, " world");
        assert_eq!(second.finish_reason.as_deref(), Some("stop"));
        assert_eq!(second.prompt_token_count, Some(2));
        assert_eq!(second.output_token_count, Some(2));

        assert!(stream.next_chunk().expect("DONE should parse").is_none());
    }

    #[test]
    fn stream_handle_parses_chat_completion_delta_chunks() {
        let mut stream = edge_llm_stream(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}],\"usage\":null}\n\
             \n\
             data: {\"choices\":[{\"delta\":{\"content\":\" hello\"},\"finish_reason\":null}],\"usage\":null}\n\
             \n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\
             \n\
             data: [DONE]\n\n",
        );

        let role_only = stream
            .next_chunk()
            .expect("role chunk should parse")
            .expect("role chunk should exist");
        assert_eq!(role_only.text, "");
        assert_eq!(role_only.finish_reason, None);

        let first = stream
            .next_chunk()
            .expect("first delta chunk should parse")
            .expect("first delta chunk should exist");
        assert_eq!(first.text, " hello");
        assert_eq!(first.finish_reason, None);

        let second = stream
            .next_chunk()
            .expect("second delta chunk should parse")
            .expect("second delta chunk should exist");
        assert_eq!(second.text, " world");
        assert_eq!(second.finish_reason.as_deref(), Some("stop"));
        assert_eq!(second.prompt_token_count, Some(5));
        assert_eq!(second.output_token_count, Some(2));
    }

    #[test]
    fn stream_handle_accepts_data_prefix_without_space() {
        let mut stream = edge_llm_stream(
            "data:{\"choices\":[{\"text\":\"ok\",\"finish_reason\":\"stop\"}],\"usage\":null}\n\n\
             data:[DONE]\n\n",
        );
        let chunk = stream
            .next_chunk()
            .expect("chunk without space after data: should parse")
            .expect("chunk should exist");
        assert_eq!(chunk.text, "ok");
        assert_eq!(chunk.finish_reason.as_deref(), Some("stop"));
        assert!(
            stream
                .next_chunk()
                .expect("[DONE] should end stream")
                .is_none()
        );
    }

    #[test]
    fn stream_handle_rejects_missing_stream_choice() {
        let mut stream = edge_llm_stream("data: {\"choices\":[]}\n\n");

        let error = stream
            .next_chunk()
            .expect_err("missing stream choices should fail closed");
        assert!(matches!(
            error,
            EdgeLlmBackendError::MissingStreamChoice { .. }
        ));
    }

    #[test]
    fn stream_handle_rejects_invalid_stream_json() {
        let mut stream = edge_llm_stream("data: {not-json}\n\n");

        let error = stream
            .next_chunk()
            .expect_err("invalid stream JSON should fail closed");
        assert!(matches!(
            error,
            EdgeLlmBackendError::InvalidStreamChunk { .. }
        ));
    }

    fn runtime_report() -> RuntimeReport {
        RuntimeReport::from_resolution(
            &BackendPolicy::allow_tensor_rt_edge_llm(),
            &ResolvedBackend::tensor_rt_edge_llm("test delegated route"),
        )
    }

    fn tensor_rt_llm_runtime_report() -> RuntimeReport {
        RuntimeReport::from_resolution(
            &BackendPolicy::allow_tensor_rt_llm(),
            &ResolvedBackend::tensor_rt_llm("test tensorrt-llm delegated route"),
        )
    }

    #[test]
    fn blocking_chat_generate_labels_tensor_rt_llm_execution_plan() {
        let response_body = r#"{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}"#.to_string();
        let (server_url, handle) = spawn_completion_server(response_body, |payload| {
            assert_eq!(payload["model"], "qwen3");
            assert_eq!(payload["stream"], false);
        });

        let response = run_blocking_chat_generate(
            21,
            &tensor_rt_llm_runtime_report(),
            &EdgeLlmConfig::server_completion(server_url),
            &chat_request(),
        )
        .expect("tensorrt-llm delegated chat completion should succeed");

        assert_eq!(response.output_text.as_deref(), Some("hi"));
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("tensor_rt_llm.server_chat_completion")
        );
        assert_eq!(
            response.runtime.selected_backend,
            SelectedBackend::TensorRtLlm
        );
        handle.join().expect("server thread should finish");
    }

    #[test]
    fn tensor_rt_llm_chat_payload_preserves_openai_disabled_topk_topp() {
        // Unlike Edge-LLM C++ sampler, TRT-LLM OpenAI path keeps top_k=0 / top_p=1.0.
        let response_body = r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#.to_string();
        let (server_url, handle) = spawn_completion_server(response_body, |payload| {
            assert_eq!(payload["top_k"], 0);
            assert_eq!(payload["top_p"], 1.0);
        });

        let mut request = chat_request();
        request.sampling.top_k = 0;
        request.sampling.top_p = 1.0;

        run_blocking_chat_generate(
            22,
            &tensor_rt_llm_runtime_report(),
            &EdgeLlmConfig::server_completion(server_url),
            &request,
        )
        .expect("tensorrt-llm must accept OpenAI-style disabled sampling");
        handle.join().expect("server thread should finish");
    }

    fn text_request(prompt: &str) -> GenerateRequest {
        GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: Vec::new(),
            input_text: Some(prompt.to_string()),
            multimodal_inputs: Default::default(),
            max_output_tokens: 3,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        }
    }

    fn chat_request() -> EdgeLlmChatGenerateRequest {
        EdgeLlmChatGenerateRequest {
            model_id: "qwen3".to_string(),
            messages: vec![
                EdgeLlmChatMessage::new("system", "Be concise."),
                EdgeLlmChatMessage::new("user", "hello"),
            ],
            max_output_tokens: 3,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: None,
            chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        }
    }

    fn edge_llm_stream(body: &str) -> EdgeLlmStreamHandle {
        EdgeLlmStreamHandle::new(
            "http://127.0.0.1:8090/v1/completions".to_string(),
            Box::new(Cursor::new(body.as_bytes().to_vec())),
        )
    }

    fn spawn_completion_server(
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
}
