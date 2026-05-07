use std::io::{BufRead, BufReader, Read};
use std::sync::OnceLock;
use std::time::Duration;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::backend::{RuntimeReport, SelectedBackend};
use crate::generate::{
    GenerateFinishReason, GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateStatus,
};

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_IO_TIMEOUT: Duration = Duration::from_secs(300);
static MLX_LM_HTTP_AGENT: OnceLock<ureq::Agent> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MlxLmConfig {
    ServerCompletion(MlxLmServerCompletionConfig),
}

impl MlxLmConfig {
    pub fn server_completion(base_url: impl Into<String>) -> Self {
        Self::ServerCompletion(MlxLmServerCompletionConfig::new(base_url))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxLmServerCompletionConfig {
    pub base_url: String,
}

impl MlxLmServerCompletionConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: normalize_base_url(base_url.into()),
        }
    }

    pub fn completions_url(&self) -> String {
        format!("{}/v1/completions", self.base_url)
    }
}

#[derive(Debug, Error)]
pub enum MlxLmBackendError {
    #[error("mlx-lm backend expected {configured_backend:?}, got {resolved_backend:?}")]
    BackendConfigMismatch {
        configured_backend: SelectedBackend,
        resolved_backend: SelectedBackend,
    },
    #[error("mlx-lm delegated backend requires input_text; token-array prompts need AX MLX mode")]
    MissingInputText,
    #[error("mlx-lm delegated backend does not accept input_tokens in this preview contract")]
    UnsupportedTokenPrompt,
    #[error("failed to serialize mlx-lm request JSON for {endpoint}: {source}")]
    SerializeRequestJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("mlx-lm backend HTTP request to {endpoint} failed: {source}")]
    HttpRequest {
        endpoint: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error("mlx-lm backend HTTP response from {endpoint} returned status {status}: {body}")]
    HttpStatus {
        endpoint: String,
        status: u16,
        body: String,
    },
    #[error("mlx-lm backend HTTP response from {endpoint} was not valid JSON: {source}")]
    InvalidResponseJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("mlx-lm backend HTTP response from {endpoint} did not include a completion choice")]
    MissingCompletionChoice { endpoint: String },
    #[error("mlx-lm backend SSE read from {endpoint} failed: {source}")]
    SseRead {
        endpoint: String,
        #[source]
        source: std::io::Error,
    },
    #[error("mlx-lm backend SSE chunk from {endpoint} was not valid JSON: {source}")]
    InvalidStreamChunk {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("mlx-lm backend SSE stream from {endpoint} did not include a choice in chunk")]
    MissingStreamChoice { endpoint: String },
}

pub(crate) struct MlxLmStreamChunkResult {
    pub text: String,
    pub finish_reason: Option<String>,
    pub prompt_token_count: Option<u32>,
    pub output_token_count: Option<u32>,
}

pub(crate) struct MlxLmStreamHandle {
    endpoint: String,
    reader: BufReader<Box<dyn Read + Send>>,
}

impl MlxLmStreamHandle {
    fn new(endpoint: String, reader: Box<dyn Read + Send>) -> Self {
        Self {
            endpoint,
            reader: BufReader::new(reader),
        }
    }

    pub(crate) fn next_chunk(
        &mut self,
    ) -> Result<Option<MlxLmStreamChunkResult>, MlxLmBackendError> {
        loop {
            let mut line = String::new();
            let bytes_read =
                self.reader
                    .read_line(&mut line)
                    .map_err(|source| MlxLmBackendError::SseRead {
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

            let data = match line.strip_prefix("data: ") {
                Some(data) => data,
                None => continue,
            };

            if data == "[DONE]" {
                return Ok(None);
            }

            let chunk: MlxLmStreamChunk = serde_json::from_str(data).map_err(|source| {
                MlxLmBackendError::InvalidStreamChunk {
                    endpoint: self.endpoint.clone(),
                    source,
                }
            })?;

            let choice = chunk.choices.into_iter().next().ok_or_else(|| {
                MlxLmBackendError::MissingStreamChoice {
                    endpoint: self.endpoint.clone(),
                }
            })?;

            return Ok(Some(MlxLmStreamChunkResult {
                text: choice.text,
                finish_reason: choice.finish_reason,
                prompt_token_count: chunk.usage.as_ref().map(|u| u.prompt_tokens),
                output_token_count: chunk.usage.as_ref().map(|u| u.completion_tokens),
            }));
        }
    }
}

impl std::fmt::Debug for MlxLmStreamHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxLmStreamHandle")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

pub(crate) fn start_streaming_generate(
    runtime: &RuntimeReport,
    config: &MlxLmConfig,
    request: &GenerateRequest,
) -> Result<MlxLmStreamHandle, MlxLmBackendError> {
    if runtime.selected_backend != SelectedBackend::MlxLmDelegated {
        return Err(MlxLmBackendError::BackendConfigMismatch {
            configured_backend: SelectedBackend::MlxLmDelegated,
            resolved_backend: runtime.selected_backend,
        });
    }

    match config {
        MlxLmConfig::ServerCompletion(config) => {
            start_mlx_lm_server_completion_stream(config, request)
        }
    }
}

fn start_mlx_lm_server_completion_stream(
    config: &MlxLmServerCompletionConfig,
    request: &GenerateRequest,
) -> Result<MlxLmStreamHandle, MlxLmBackendError> {
    if !request.input_tokens.is_empty() {
        return Err(MlxLmBackendError::UnsupportedTokenPrompt);
    }
    let prompt = request
        .input_text
        .clone()
        .ok_or(MlxLmBackendError::MissingInputText)?;

    let endpoint = config.completions_url();
    let payload = MlxLmCompletionRequest {
        model: &request.model_id,
        prompt: &prompt,
        max_tokens: request.max_output_tokens,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: request.sampling.top_k,
        min_p: request.sampling.min_p,
        repetition_penalty: request.sampling.repetition_penalty,
        seed: request.sampling.seed,
        stream: true,
        stop: request.stop_sequences.clone(),
    };

    let response = send_json_post_request(&endpoint, &payload)?;
    let reader: Box<dyn Read + Send> = Box::new(response.into_reader());
    Ok(MlxLmStreamHandle::new(endpoint, reader))
}

pub fn run_blocking_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &MlxLmConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, MlxLmBackendError> {
    if runtime.selected_backend != SelectedBackend::MlxLmDelegated {
        return Err(MlxLmBackendError::BackendConfigMismatch {
            configured_backend: SelectedBackend::MlxLmDelegated,
            resolved_backend: runtime.selected_backend,
        });
    }

    match config {
        MlxLmConfig::ServerCompletion(config) => {
            run_mlx_lm_server_completion_generate(request_id, runtime, config, request)
        }
    }
}

fn run_mlx_lm_server_completion_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &MlxLmServerCompletionConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, MlxLmBackendError> {
    if !request.input_tokens.is_empty() {
        return Err(MlxLmBackendError::UnsupportedTokenPrompt);
    }
    let prompt = request
        .input_text
        .clone()
        .ok_or(MlxLmBackendError::MissingInputText)?;

    let endpoint = config.completions_url();
    let payload = MlxLmCompletionRequest {
        model: &request.model_id,
        prompt: &prompt,
        max_tokens: request.max_output_tokens,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: request.sampling.top_k,
        min_p: request.sampling.min_p,
        repetition_penalty: request.sampling.repetition_penalty,
        seed: request.sampling.seed,
        stream: false,
        stop: request.stop_sequences.clone(),
    };

    let response = send_json_post_request(&endpoint, &payload)?;
    let response: MlxLmCompletionResponse = parse_json_response(&endpoint, response)?;
    let choice = response.choices.into_iter().next().ok_or_else(|| {
        MlxLmBackendError::MissingCompletionChoice {
            endpoint: endpoint.clone(),
        }
    })?;

    Ok(GenerateResponse {
        request_id,
        model_id: request.model_id.clone(),
        prompt_tokens: Vec::new(),
        prompt_text: Some(prompt),
        output_tokens: Vec::new(),
        output_token_logprobs: Vec::new(),
        output_text: Some(choice.text),
        prompt_token_count: response.usage.as_ref().map(|usage| usage.prompt_tokens),
        output_token_count: response.usage.as_ref().map(|usage| usage.completion_tokens),
        status: GenerateStatus::Finished,
        finish_reason: finish_reason_from_mlx_lm(choice.finish_reason.as_deref()),
        step_count: 0,
        ttft_step: None,
        route: GenerateRouteReport {
            execution_plan: Some("mlx_lm_delegated.server_completion".to_string()),
            attention_route: None,
            kv_mode: None,
            prefix_cache_path: None,
            barrier_mode: None,
            crossover_decisions: Default::default(),
        },
        runtime: runtime.clone(),
    })
}

fn mlx_lm_http_agent() -> &'static ureq::Agent {
    MLX_LM_HTTP_AGENT.get_or_init(|| {
        ureq::AgentBuilder::new()
            .timeout_connect(HTTP_CONNECT_TIMEOUT)
            .timeout_read(HTTP_IO_TIMEOUT)
            .timeout_write(HTTP_IO_TIMEOUT)
            .build()
    })
}

fn send_json_post_request<T>(
    endpoint: &str,
    payload: &T,
) -> Result<ureq::Response, MlxLmBackendError>
where
    T: Serialize + ?Sized,
{
    let body =
        serde_json::to_vec(payload).map_err(|source| MlxLmBackendError::SerializeRequestJson {
            endpoint: endpoint.to_string(),
            source,
        })?;

    match mlx_lm_http_agent()
        .post(endpoint)
        .set("Content-Type", "application/json")
        .send_bytes(&body)
    {
        Ok(response) => Ok(response),
        Err(ureq::Error::Status(status, response)) => {
            let body = response
                .into_string()
                .unwrap_or_else(|_| "<failed to read response body>".to_string());
            Err(MlxLmBackendError::HttpStatus {
                endpoint: endpoint.to_string(),
                status,
                body: body.trim().to_string(),
            })
        }
        Err(source) => Err(MlxLmBackendError::HttpRequest {
            endpoint: endpoint.to_string(),
            source: Box::new(source),
        }),
    }
}

fn parse_json_response<T>(endpoint: &str, response: ureq::Response) -> Result<T, MlxLmBackendError>
where
    T: DeserializeOwned,
{
    serde_json::from_reader(response.into_reader()).map_err(|source| {
        MlxLmBackendError::InvalidResponseJson {
            endpoint: endpoint.to_string(),
            source,
        }
    })
}

fn normalize_base_url(mut value: String) -> String {
    while value.ends_with('/') {
        value.pop();
    }
    value
}

pub(crate) fn finish_reason_from_mlx_lm(value: Option<&str>) -> Option<GenerateFinishReason> {
    match value {
        Some("stop") => Some(GenerateFinishReason::Stop),
        Some("length") => Some(GenerateFinishReason::MaxOutputTokens),
        Some(_) | None => None,
    }
}

#[derive(Debug, Deserialize, Default)]
struct MlxLmStreamChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlxLmStreamChunk {
    #[serde(default)]
    choices: Vec<MlxLmStreamChoice>,
    #[serde(default)]
    usage: Option<MlxLmCompletionUsage>,
}

#[derive(Debug, Serialize)]
struct MlxLmCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    repetition_penalty: f32,
    seed: u64,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
struct MlxLmCompletionChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlxLmCompletionUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct MlxLmCompletionResponse {
    #[serde(default)]
    choices: Vec<MlxLmCompletionChoice>,
    #[serde(default)]
    usage: Option<MlxLmCompletionUsage>,
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use serde_json::Value;

    use super::*;
    use crate::backend::{BackendPolicy, ResolvedBackend, RuntimeReport};
    use crate::generate::{GenerateRequest, GenerateSampling};

    #[test]
    fn server_completion_url_normalizes_trailing_slashes() {
        let config = MlxLmConfig::server_completion("http://127.0.0.1:8090///");
        let MlxLmConfig::ServerCompletion(config) = config;

        assert_eq!(
            config.completions_url(),
            "http://127.0.0.1:8090/v1/completions"
        );
    }

    #[test]
    fn blocking_generate_calls_mlx_lm_completions_contract() {
        let response_body = r#"{"choices":[{"text":" world","finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}"#.to_string();
        let (server_url, handle) = spawn_completion_server(response_body, |payload| {
            assert_eq!(payload["model"], "qwen3_dense");
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
            &MlxLmConfig::server_completion(server_url),
            &request,
        )
        .expect("mlx-lm delegated completion should succeed");

        assert_eq!(response.request_id, 7);
        assert_eq!(response.prompt_text.as_deref(), Some("hello"));
        assert_eq!(response.output_text.as_deref(), Some(" world"));
        assert_eq!(response.prompt_token_count, Some(2));
        assert_eq!(response.output_token_count, Some(1));
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("mlx_lm_delegated.server_completion")
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
            &MlxLmConfig::server_completion("http://127.0.0.1:8090"),
            &request,
        )
        .expect_err("token prompts should fail closed");

        assert!(matches!(error, MlxLmBackendError::UnsupportedTokenPrompt));
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
            &MlxLmConfig::server_completion(server_url),
            &text_request("hello"),
        )
        .expect_err("missing choices should fail closed");

        assert!(matches!(
            error,
            MlxLmBackendError::MissingCompletionChoice { .. }
        ));
        handle.join().expect("server thread should finish");
    }

    fn runtime_report() -> RuntimeReport {
        RuntimeReport::from_resolution(
            &BackendPolicy::allow_mlx_lm_delegated(),
            &ResolvedBackend::mlx_lm_delegated("test delegated route"),
        )
    }

    fn text_request(prompt: &str) -> GenerateRequest {
        GenerateRequest {
            model_id: "qwen3_dense".to_string(),
            input_tokens: Vec::new(),
            input_text: Some(prompt.to_string()),
            max_output_tokens: 3,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        }
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
