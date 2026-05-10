use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use wait_timeout::ChildExt;

use crate::backend::{RuntimeReport, SelectedBackend};
use crate::delegated_http::{
    DelegatedHttpPostError, DelegatedHttpTimeouts, normalize_base_url, parse_json_response,
    send_json_post_request,
};
use crate::generate::{
    GenerateFinishReason, GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateSampling,
    GenerateStatus,
};

const LLAMA_CPP_CLI_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaCppConfig {
    Cli(LlamaCppCliConfig),
    ServerCompletion(LlamaCppServerCompletionConfig),
}

impl LlamaCppConfig {
    pub fn new(cli_path: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        Self::cli(cli_path, model_path)
    }

    pub fn cli(cli_path: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        Self::Cli(LlamaCppCliConfig::new(cli_path, model_path))
    }

    pub fn server_completion(base_url: impl Into<String>) -> Self {
        Self::ServerCompletion(LlamaCppServerCompletionConfig::new(base_url))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaCppCliConfig {
    pub cli_path: PathBuf,
    pub model_path: PathBuf,
    pub extra_args: Vec<String>,
}

impl LlamaCppCliConfig {
    pub fn new(cli_path: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        Self {
            cli_path: cli_path.into(),
            model_path: model_path.into(),
            extra_args: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaCppServerCompletionConfig {
    pub base_url: String,
    pub timeouts: DelegatedHttpTimeouts,
}

impl LlamaCppServerCompletionConfig {
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

    pub fn completion_url(&self) -> String {
        format!("{}/completion", self.base_url)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LlamaCppStreamFormat {
    LlamaCppCompletion,
}

pub(crate) struct LlamaCppStreamHandle {
    endpoint: String,
    format: LlamaCppStreamFormat,
    reader: BufReader<Box<dyn Read + Send>>,
    bos_space_stripped: bool,
}

impl LlamaCppStreamHandle {
    fn new(endpoint: String, format: LlamaCppStreamFormat, reader: Box<dyn Read + Send>) -> Self {
        Self {
            endpoint,
            format,
            reader: BufReader::new(reader),
            bos_space_stripped: false,
        }
    }

    pub(crate) fn next_chunk(
        &mut self,
    ) -> Result<Option<LlamaCppStreamChunk>, LlamaCppBackendError> {
        let mut payload = String::new();

        loop {
            let mut line = String::new();
            let bytes_read = self.reader.read_line(&mut line).map_err(|source| {
                LlamaCppBackendError::HttpResponseRead {
                    endpoint: self.endpoint.clone(),
                    source,
                }
            })?;

            if bytes_read == 0 {
                if payload.is_empty() {
                    return Ok(None);
                }
                break;
            }

            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                if !payload.is_empty() {
                    break;
                }
                continue;
            }

            if let Some(value) = line.strip_prefix("data: ") {
                if value == "[DONE]" {
                    return Ok(None);
                }
                if !payload.is_empty() {
                    payload.push('\n');
                }
                payload.push_str(value);
            }
        }

        let mut chunk: LlamaCppStreamChunk = match self.format {
            LlamaCppStreamFormat::LlamaCppCompletion => serde_json::from_str(&payload),
        }
        .map_err(|source| LlamaCppBackendError::InvalidResponseJson {
            endpoint: self.endpoint.clone(),
            source,
        })?;

        if !self.bos_space_stripped && !chunk.content.is_empty() {
            self.bos_space_stripped = true;
            if chunk.content.starts_with(' ') {
                chunk.content.remove(0);
            }
        }

        Ok(Some(chunk))
    }
}

impl std::fmt::Debug for LlamaCppStreamHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppStreamHandle")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct LlamaCppStreamChunk {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tokens: Vec<u32>,
    #[serde(default)]
    pub stop: bool,
    #[serde(default)]
    pub stop_type: Option<String>,
    #[serde(default)]
    pub prompt_progress: Option<LlamaCppPromptProgress>,
    #[serde(default)]
    pub prompt_token_count: Option<u32>,
    #[serde(default)]
    pub output_token_count: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub(crate) struct LlamaCppPromptProgress {
    #[serde(default)]
    pub total: u32,
    #[serde(default)]
    pub cache: u32,
    #[serde(default)]
    pub processed: u32,
}

#[derive(Debug, Error)]
pub enum LlamaCppBackendError {
    #[error("llama.cpp backend {selected_backend:?} requires input_text")]
    MissingInputText { selected_backend: SelectedBackend },
    #[error("llama.cpp backend {selected_backend:?} requires input_tokens or input_text")]
    MissingPromptInput { selected_backend: SelectedBackend },
    #[error(
        "llama.cpp backend {selected_backend:?} CLI path does not support tokenized prompt input yet"
    )]
    UnsupportedTokenPrompt { selected_backend: SelectedBackend },
    #[error(
        "llama.cpp backend {selected_backend:?} received both input_tokens and input_text, but this preview request shape cannot preserve mixed prompt ordering"
    )]
    AmbiguousPromptInput { selected_backend: SelectedBackend },
    #[error(
        "llama.cpp backend config targets {configured_backend:?}, but session resolved {resolved_backend:?}"
    )]
    BackendConfigMismatch {
        configured_backend: SelectedBackend,
        resolved_backend: SelectedBackend,
    },
    #[error("failed to launch llama.cpp backend command {command}: {source}")]
    CommandLaunch {
        command: String,
        #[source]
        source: std::io::Error,
    },
    #[error("llama.cpp backend command {command} exited with status {status}: {stderr}")]
    CommandFailed {
        command: String,
        status: String,
        stderr: String,
    },
    #[error("llama.cpp backend command {command} timed out after {timeout_seconds}s")]
    CommandTimedOut {
        command: String,
        timeout_seconds: u64,
    },
    #[error("llama.cpp backend command {command} returned non-utf8 output: {source}")]
    NonUtf8Output {
        command: String,
        #[source]
        source: std::string::FromUtf8Error,
    },
    #[error("llama.cpp backend HTTP request to {endpoint} failed: {source}")]
    HttpRequest {
        endpoint: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error("llama.cpp backend HTTP request to {endpoint} could not be serialized: {source}")]
    SerializeRequestJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("llama.cpp backend HTTP request to {endpoint} returned status {status}: {body}")]
    HttpStatus {
        endpoint: String,
        status: u16,
        body: String,
    },
    #[error("llama.cpp backend HTTP response from {endpoint} could not be read: {source}")]
    HttpResponseRead {
        endpoint: String,
        #[source]
        source: std::io::Error,
    },
    #[error("llama.cpp backend HTTP response from {endpoint} was not valid JSON: {source}")]
    InvalidResponseJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "llama.cpp backend {selected_backend:?} does not support streaming generate in this preview contract"
    )]
    StreamingNotSupported { selected_backend: SelectedBackend },
}

pub fn run_blocking_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &LlamaCppConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, LlamaCppBackendError> {
    if runtime.selected_backend != SelectedBackend::LlamaCpp {
        return Err(LlamaCppBackendError::BackendConfigMismatch {
            configured_backend: SelectedBackend::LlamaCpp,
            resolved_backend: runtime.selected_backend,
        });
    }
    run_llama_cpp_generate(request_id, runtime, config, request)
}

pub(crate) fn start_streaming_generate(
    runtime: &RuntimeReport,
    config: &LlamaCppConfig,
    request: &GenerateRequest,
) -> Result<LlamaCppStreamHandle, LlamaCppBackendError> {
    if runtime.selected_backend != SelectedBackend::LlamaCpp {
        return Err(LlamaCppBackendError::BackendConfigMismatch {
            configured_backend: SelectedBackend::LlamaCpp,
            resolved_backend: runtime.selected_backend,
        });
    }
    start_llama_cpp_streaming_generate(config, request)
}

fn run_llama_cpp_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &LlamaCppConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, LlamaCppBackendError> {
    match config {
        LlamaCppConfig::Cli(config) => {
            run_llama_cpp_cli_generate(request_id, runtime, config, request)
        }
        LlamaCppConfig::ServerCompletion(config) => {
            run_llama_cpp_server_completion_generate(request_id, runtime, config, request)
        }
    }
}

fn start_llama_cpp_streaming_generate(
    config: &LlamaCppConfig,
    request: &GenerateRequest,
) -> Result<LlamaCppStreamHandle, LlamaCppBackendError> {
    match config {
        LlamaCppConfig::Cli(_) => Err(LlamaCppBackendError::StreamingNotSupported {
            selected_backend: SelectedBackend::LlamaCpp,
        }),
        LlamaCppConfig::ServerCompletion(config) => {
            start_llama_cpp_server_completion_stream(config, request)
        }
    }
}

fn run_llama_cpp_cli_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &LlamaCppCliConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, LlamaCppBackendError> {
    if !request.input_tokens.is_empty() {
        return Err(LlamaCppBackendError::UnsupportedTokenPrompt {
            selected_backend: SelectedBackend::LlamaCpp,
        });
    }

    let prompt_text = request
        .input_text
        .clone()
        .ok_or(LlamaCppBackendError::MissingInputText {
            selected_backend: SelectedBackend::LlamaCpp,
        })?;

    let command_display = config.cli_path.display().to_string();
    let mut command = Command::new(&config.cli_path);
    command
        .arg("--simple-io")
        .arg("--no-display-prompt")
        .arg("--single-turn")
        .arg("--log-disable")
        .arg("--model")
        .arg(&config.model_path)
        .arg("--prompt")
        .arg(&prompt_text)
        .arg("--n-predict")
        .arg(request.max_output_tokens.to_string());

    append_sampling_args(&mut command, &request.sampling);
    append_stop_sequence_args(&mut command, &request.stop_sequences);
    command.args(&config.extra_args);

    let output = run_command_with_timeout(command, command_display.clone(), LLAMA_CPP_CLI_TIMEOUT)?;

    if !output.status.success() {
        return Err(LlamaCppBackendError::CommandFailed {
            command: command_display,
            status: output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "terminated by signal".to_string()),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    let output_text =
        String::from_utf8(output.stdout).map_err(|source| LlamaCppBackendError::NonUtf8Output {
            command: config.cli_path.display().to_string(),
            source,
        })?;
    let output_text = extract_cli_response(output_text);

    Ok(GenerateResponse {
        request_id,
        model_id: request.model_id.clone(),
        prompt_tokens: Vec::new(),
        prompt_text: Some(prompt_text),
        output_tokens: Vec::new(),
        output_token_logprobs: Vec::new(),
        output_text: Some(output_text),
        prompt_token_count: None,
        output_token_count: None,
        status: GenerateStatus::Finished,
        finish_reason: None,
        step_count: 0,
        ttft_step: None,
        route: GenerateRouteReport {
            execution_plan: Some("llama_cpp.blocking_cli".to_string()),
            attention_route: None,
            kv_mode: None,
            prefix_cache_path: None,
            barrier_mode: None,
            crossover_decisions: Default::default(),
        },
        runtime: runtime.clone(),
    })
}

fn run_llama_cpp_server_completion_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &LlamaCppServerCompletionConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, LlamaCppBackendError> {
    let prompt = build_llama_cpp_prompt(request)?;
    let endpoint = config.completion_url();
    let payload = LlamaCppCompletionRequest {
        prompt,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: request.sampling.top_k,
        min_p: request.sampling.min_p,
        repeat_penalty: request.sampling.repetition_penalty,
        repeat_last_n: request.sampling.repetition_context_size,
        seed: request.sampling.seed,
        n_predict: request.max_output_tokens,
        stream: false,
        return_tokens: true,
        return_progress: false,
        stop: request.stop_sequences.clone(),
    };

    let response = send_llama_cpp_json_post_request(&endpoint, &payload, None, config.timeouts)?;
    let response: LlamaCppCompletionResponse = parse_llama_cpp_json_response(response, &endpoint)?;

    let output_tokens = response.tokens;
    let output_token_logprobs = vec![None; output_tokens.len()];
    let prompt_token_count = if request.input_tokens.is_empty() && response.tokens_evaluated > 0 {
        Some(response.tokens_evaluated)
    } else {
        None
    };

    Ok(GenerateResponse {
        request_id,
        model_id: request.model_id.clone(),
        prompt_tokens: request.input_tokens.clone(),
        prompt_text: request.input_text.clone(),
        output_tokens,
        output_token_logprobs,
        output_text: Some(strip_bos_leading_space(response.content)),
        prompt_token_count,
        output_token_count: None,
        status: GenerateStatus::Finished,
        finish_reason: finish_reason_from_stop_type(response.stop, response.stop_type.as_deref()),
        step_count: 0,
        ttft_step: None,
        route: llama_cpp_server_completion_route(
            "llama_cpp.server_completion",
            response.tokens_cached,
        ),
        runtime: runtime.clone(),
    })
}

fn start_llama_cpp_server_completion_stream(
    config: &LlamaCppServerCompletionConfig,
    request: &GenerateRequest,
) -> Result<LlamaCppStreamHandle, LlamaCppBackendError> {
    let prompt = build_llama_cpp_prompt(request)?;
    let endpoint = config.completion_url();
    let payload = LlamaCppCompletionRequest {
        prompt,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: request.sampling.top_k,
        min_p: request.sampling.min_p,
        repeat_penalty: request.sampling.repetition_penalty,
        repeat_last_n: request.sampling.repetition_context_size,
        seed: request.sampling.seed,
        n_predict: request.max_output_tokens,
        stream: true,
        return_tokens: true,
        return_progress: true,
        stop: request.stop_sequences.clone(),
    };

    let response = send_llama_cpp_json_post_request(
        &endpoint,
        &payload,
        Some("text/event-stream"),
        config.timeouts,
    )?;

    let reader: Box<dyn Read + Send> = Box::new(response.into_reader());
    Ok(LlamaCppStreamHandle::new(
        endpoint,
        LlamaCppStreamFormat::LlamaCppCompletion,
        reader,
    ))
}

fn send_llama_cpp_json_post_request<T>(
    endpoint: &str,
    payload: &T,
    accept: Option<&str>,
    timeouts: DelegatedHttpTimeouts,
) -> Result<ureq::Response, LlamaCppBackendError>
where
    T: Serialize + ?Sized,
{
    send_json_post_request(endpoint, payload, accept, timeouts, |error| match error {
        DelegatedHttpPostError::Serialize(source) => LlamaCppBackendError::SerializeRequestJson {
            endpoint: endpoint.to_string(),
            source,
        },
        DelegatedHttpPostError::Status { status, body } => LlamaCppBackendError::HttpStatus {
            endpoint: endpoint.to_string(),
            status,
            body,
        },
        DelegatedHttpPostError::Request(source) => LlamaCppBackendError::HttpRequest {
            endpoint: endpoint.to_string(),
            source,
        },
    })
}

fn parse_llama_cpp_json_response<T>(
    response: ureq::Response,
    endpoint: &str,
) -> Result<T, LlamaCppBackendError>
where
    T: DeserializeOwned,
{
    parse_json_response(response, |source| {
        LlamaCppBackendError::InvalidResponseJson {
            endpoint: endpoint.to_string(),
            source,
        }
    })
}

fn build_llama_cpp_prompt(
    request: &GenerateRequest,
) -> Result<LlamaCppPrompt<'_>, LlamaCppBackendError> {
    match (
        request.input_tokens.is_empty(),
        request.input_text.as_deref(),
    ) {
        (false, Some(_)) => Err(LlamaCppBackendError::AmbiguousPromptInput {
            selected_backend: SelectedBackend::LlamaCpp,
        }),
        (false, None) => Ok(LlamaCppPrompt::Tokens(&request.input_tokens)),
        (true, Some(text)) => Ok(LlamaCppPrompt::Text(text)),
        (true, None) => Err(LlamaCppBackendError::MissingPromptInput {
            selected_backend: SelectedBackend::LlamaCpp,
        }),
    }
}

fn finish_reason_from_stop_type(
    stop: bool,
    stop_type: Option<&str>,
) -> Option<GenerateFinishReason> {
    if !stop {
        return None;
    }

    match stop_type {
        Some("limit") => Some(GenerateFinishReason::MaxOutputTokens),
        Some("content_filter") => Some(GenerateFinishReason::ContentFilter),
        // "eos" = natural end-of-sequence, "word" = stop string match — both are Stop.
        Some("eos" | "word") | None => Some(GenerateFinishReason::Stop),
        Some(unknown) => {
            tracing::warn!(
                stop_type = unknown,
                "llama.cpp returned unknown stop_type; reporting generic stop"
            );
            Some(GenerateFinishReason::Stop)
        }
    }
}

fn append_sampling_args(command: &mut Command, sampling: &GenerateSampling) {
    command
        .arg("--seed")
        .arg(sampling.seed.to_string())
        .arg("--temp")
        .arg(sampling.temperature.to_string())
        .arg("--top-p")
        .arg(sampling.top_p.to_string())
        .arg("--top-k")
        .arg(sampling.top_k.to_string())
        .arg("--repeat-penalty")
        .arg(sampling.repetition_penalty.to_string());
    if let Some(min_p) = sampling.min_p {
        command.arg("--min-p").arg(min_p.to_string());
    }
    if let Some(ctx) = sampling.repetition_context_size {
        command.arg("--repeat-last-n").arg(ctx.to_string());
    }
}

fn append_stop_sequence_args(command: &mut Command, stop_sequences: &[String]) {
    for seq in stop_sequences {
        command.arg("--stopping-string").arg(seq);
    }
}

fn trim_single_trailing_newline(mut value: String) -> String {
    if value.ends_with('\n') {
        value.pop();
        if value.ends_with('\r') {
            value.pop();
        }
    }
    value
}

// llama.cpp server prepends a BOS-space to the first decoded token; strip it.
fn strip_bos_leading_space(mut s: String) -> String {
    if s.starts_with(' ') {
        s.remove(0);
    }
    s
}

// llama-cli with --simple-io emits: [preamble] "assistant:\n\n" {RESPONSE} "\n[ Prompt: …" "\nExiting…"
// Strip everything outside the assistant body.
fn extract_cli_response(raw: String) -> String {
    const ASSISTANT_MARKER: &str = "assistant:\n\n";
    if let Some(start) = raw.find(ASSISTANT_MARKER) {
        let body = &raw[start + ASSISTANT_MARKER.len()..];
        let end = body
            .find("\n[ Prompt:")
            .or_else(|| body.find("\nExiting..."))
            .unwrap_or(body.len());
        return body[..end].trim_end().to_string();
    }
    if looks_like_llama_cli_wrapped_output(&raw) {
        tracing::warn!(
            "llama-cli output did not contain expected assistant marker; returning trimmed stdout"
        );
    }
    trim_single_trailing_newline(raw)
}

fn looks_like_llama_cli_wrapped_output(raw: &str) -> bool {
    raw.contains("[ Prompt:") || raw.contains("llama-cli") || raw.contains("Exiting...")
}

fn run_command_with_timeout(
    mut command: Command,
    command_display: String,
    timeout: Duration,
) -> Result<Output, LlamaCppBackendError> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| LlamaCppBackendError::CommandLaunch {
            command: command_display.clone(),
            source,
        })?;

    match child
        .wait_timeout(timeout)
        .map_err(|source| LlamaCppBackendError::CommandLaunch {
            command: command_display.clone(),
            source,
        })? {
        Some(_) => child
            .wait_with_output()
            .map_err(|source| LlamaCppBackendError::CommandLaunch {
                command: command_display,
                source,
            }),
        None => {
            let _ = child.kill();
            let _ = child.wait();
            Err(LlamaCppBackendError::CommandTimedOut {
                command: command_display,
                timeout_seconds: timeout.as_secs(),
            })
        }
    }
}

#[derive(Debug, Serialize)]
struct LlamaCppCompletionRequest<'a> {
    prompt: LlamaCppPrompt<'a>,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    repeat_penalty: f32,
    /// Number of tokens to look back when applying repeat penalty (llama.cpp: repeat_last_n).
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_last_n: Option<u32>,
    seed: u64,
    n_predict: u32,
    stream: bool,
    return_tokens: bool,
    return_progress: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum LlamaCppPrompt<'a> {
    Text(&'a str),
    Tokens(&'a [u32]),
}

#[derive(Debug, Deserialize)]
struct LlamaCppCompletionResponse {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tokens: Vec<u32>,
    #[serde(default)]
    stop: bool,
    #[serde(default)]
    stop_type: Option<String>,
    #[serde(default)]
    tokens_cached: u32,
    #[serde(default)]
    tokens_evaluated: u32,
}

fn llama_cpp_server_completion_route(
    execution_plan: &str,
    tokens_cached: u32,
) -> GenerateRouteReport {
    let mut route = GenerateRouteReport {
        execution_plan: Some(execution_plan.to_string()),
        attention_route: None,
        kv_mode: None,
        prefix_cache_path: None,
        barrier_mode: None,
        crossover_decisions: Default::default(),
    };

    if tokens_cached > 0 {
        route.prefix_cache_path = Some("delegated_prompt_cache".to_string());
        route
            .crossover_decisions
            .insert("delegated_cached_tokens".to_string(), tokens_cached);
    }

    route
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_cli_response_strips_preamble_and_stats() {
        let raw = "ggml backend loaded\n\
                   llama-cli v1.0 ...\n\
                   > user: hello\n\
                   assistant:\n\
                   \n\
                   Hello there!\n\
                   \n\
                   [ Prompt: 14.2 t/s | Generation: 25.8 t/s ]\n\
                   \n\
                   Exiting...\n"
            .to_string();
        assert_eq!(extract_cli_response(raw), "Hello there!");
    }

    #[test]
    fn extract_cli_response_no_marker_falls_back_to_trim() {
        let raw = "direct output\n".to_string();
        assert_eq!(extract_cli_response(raw), "direct output");
    }

    #[test]
    fn extract_cli_response_no_marker_keeps_wrapped_output_for_diagnostics() {
        let raw = "llama-cli v1.0\nhello\n[ Prompt: 1 t/s ]\nExiting...\n".to_string();
        assert_eq!(
            extract_cli_response(raw),
            "llama-cli v1.0\nhello\n[ Prompt: 1 t/s ]\nExiting..."
        );
        assert!(looks_like_llama_cli_wrapped_output(
            "llama-cli v1.0\nhello\n[ Prompt: 1 t/s ]\nExiting..."
        ));
    }

    #[test]
    #[cfg(unix)]
    fn run_command_with_timeout_collects_output_without_polling() {
        let mut command = Command::new("sh");
        command.arg("-c").arg("printf ready");

        let output = run_command_with_timeout(
            command,
            "sh -c 'printf ready'".to_string(),
            Duration::from_secs(1),
        )
        .expect("quick command should complete before timeout");

        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout), "ready");
    }

    #[test]
    #[cfg(unix)]
    fn run_command_with_timeout_kills_command_after_deadline() {
        let mut command = Command::new("sh");
        command.arg("-c").arg("sleep 2");

        let error = run_command_with_timeout(
            command,
            "sh -c 'sleep 2'".to_string(),
            Duration::from_millis(20),
        )
        .expect_err("slow command should time out");

        assert!(matches!(
            error,
            LlamaCppBackendError::CommandTimedOut { .. }
        ));
    }

    #[test]
    fn strip_bos_leading_space_removes_single_space() {
        assert_eq!(strip_bos_leading_space(" hello".to_string()), "hello");
        assert_eq!(strip_bos_leading_space("hello".to_string()), "hello");
        assert_eq!(strip_bos_leading_space("  hi".to_string()), " hi");
    }
}
