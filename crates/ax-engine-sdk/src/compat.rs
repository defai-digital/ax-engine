use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

use crate::backend::{RuntimeReport, SelectedBackend};
use crate::generate::{
    GenerateFinishReason, GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateSampling,
    GenerateStatus,
};

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_IO_TIMEOUT: Duration = Duration::from_secs(300);
const COMPATIBILITY_CLI_TIMEOUT: Duration = Duration::from_secs(300);
static COMPATIBILITY_HTTP_AGENT: OnceLock<ureq::Agent> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompatibilityBackendConfig {
    LlamaCpp(LlamaCppConfig),
    Vllm(OpenAiCompatibleServerConfig),
    MistralRs(OpenAiCompatibleServerConfig),
    Mlx(MlxConfig),
}

impl CompatibilityBackendConfig {
    pub fn selected_backend(&self) -> SelectedBackend {
        match self {
            Self::LlamaCpp(_) => SelectedBackend::LlamaCpp,
            Self::Vllm(_) => SelectedBackend::Vllm,
            Self::MistralRs(_) => SelectedBackend::MistralRs,
            Self::Mlx(_) => SelectedBackend::Mlx,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MlxConfig {
    Cli(MlxCliConfig),
    ServerCompletions(OpenAiCompatibleServerConfig),
}

impl MlxConfig {
    pub fn cli(cli_path: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        Self::Cli(MlxCliConfig::new(cli_path, model_path))
    }

    pub fn server_completions(base_url: impl Into<String>) -> Self {
        Self::ServerCompletions(OpenAiCompatibleServerConfig::new(base_url))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxCliConfig {
    pub cli_path: PathBuf,
    pub model_path: PathBuf,
    pub extra_args: Vec<String>,
    pub invoke_as_python_module: bool,
}

impl MlxCliConfig {
    pub fn new(cli_path: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        let cli_path = cli_path.into();
        let invoke_as_python_module = cli_path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("python"));
        Self {
            cli_path,
            model_path: model_path.into(),
            extra_args: Vec::new(),
            invoke_as_python_module,
        }
    }
}

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
}

impl LlamaCppServerCompletionConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: normalize_base_url(base_url.into()),
        }
    }

    pub fn completion_url(&self) -> String {
        format!("{}/completion", self.base_url)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiCompatibleServerConfig {
    pub base_url: String,
}

impl OpenAiCompatibleServerConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: normalize_base_url(base_url.into()),
        }
    }

    pub fn completions_url(&self) -> String {
        format!("{}/v1/completions", self.base_url)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompatibilityStreamFormat {
    LlamaCppCompletion,
    OpenAiCompletion,
}

pub(crate) struct CompatibilityStreamHandle {
    endpoint: String,
    format: CompatibilityStreamFormat,
    reader: BufReader<Box<dyn Read + Send>>,
}

impl CompatibilityStreamHandle {
    fn new(
        endpoint: String,
        format: CompatibilityStreamFormat,
        reader: Box<dyn Read + Send>,
    ) -> Self {
        Self {
            endpoint,
            format,
            reader: BufReader::new(reader),
        }
    }

    pub(crate) fn next_chunk(
        &mut self,
    ) -> Result<Option<CompatibilityStreamChunk>, CompatibilityBackendError> {
        let mut payload = String::new();

        loop {
            let mut line = String::new();
            let bytes_read = self.reader.read_line(&mut line).map_err(|source| {
                CompatibilityBackendError::HttpResponseRead {
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

        let chunk = match self.format {
            CompatibilityStreamFormat::LlamaCppCompletion => serde_json::from_str(&payload),
            CompatibilityStreamFormat::OpenAiCompletion => {
                parse_openai_completion_stream_chunk(&payload)
            }
        }
        .map_err(|source| CompatibilityBackendError::InvalidResponseJson {
            endpoint: self.endpoint.clone(),
            source,
        })?;

        Ok(Some(chunk))
    }
}

impl std::fmt::Debug for CompatibilityStreamHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompatibilityStreamHandle")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct CompatibilityStreamChunk {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tokens: Vec<u32>,
    #[serde(default)]
    pub stop: bool,
    #[serde(default)]
    pub stop_type: Option<String>,
    #[serde(default)]
    pub prompt_progress: Option<CompatibilityPromptProgress>,
    #[serde(default)]
    pub prompt_token_count: Option<u32>,
    #[serde(default)]
    pub output_token_count: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub(crate) struct CompatibilityPromptProgress {
    #[serde(default)]
    pub total: u32,
    #[serde(default)]
    pub cache: u32,
    #[serde(default)]
    pub processed: u32,
}

#[derive(Debug, Error)]
pub enum CompatibilityBackendError {
    #[error("compatibility backend {selected_backend:?} requires input_text")]
    MissingInputText { selected_backend: SelectedBackend },
    #[error("compatibility backend {selected_backend:?} requires input_tokens or input_text")]
    MissingPromptInput { selected_backend: SelectedBackend },
    #[error("compatibility backend {selected_backend:?} CLI path does not support tokenized prompt input yet")]
    UnsupportedTokenPrompt { selected_backend: SelectedBackend },
    #[error(
        "compatibility backend {selected_backend:?} received both input_tokens and input_text, but this preview request shape cannot preserve mixed prompt ordering"
    )]
    AmbiguousPromptInput { selected_backend: SelectedBackend },
    #[error(
        "compatibility backend config targets {configured_backend:?}, but session resolved {resolved_backend:?}"
    )]
    BackendConfigMismatch {
        configured_backend: SelectedBackend,
        resolved_backend: SelectedBackend,
    },
    #[error("failed to launch compatibility backend command {command}: {source}")]
    CommandLaunch {
        command: String,
        #[source]
        source: std::io::Error,
    },
    #[error("compatibility backend command {command} exited with status {status}: {stderr}")]
    CommandFailed {
        command: String,
        status: String,
        stderr: String,
    },
    #[error("compatibility backend command {command} timed out after {timeout_seconds}s")]
    CommandTimedOut {
        command: String,
        timeout_seconds: u64,
    },
    #[error("compatibility backend command {command} returned non-utf8 output: {source}")]
    NonUtf8Output {
        command: String,
        #[source]
        source: std::string::FromUtf8Error,
    },
    #[error("compatibility backend HTTP request to {endpoint} failed: {source}")]
    HttpRequest {
        endpoint: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error("compatibility backend HTTP request to {endpoint} could not be serialized: {source}")]
    SerializeRequestJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("compatibility backend HTTP request to {endpoint} returned status {status}: {body}")]
    HttpStatus {
        endpoint: String,
        status: u16,
        body: String,
    },
    #[error("compatibility backend HTTP response from {endpoint} could not be read: {source}")]
    HttpResponseRead {
        endpoint: String,
        #[source]
        source: std::io::Error,
    },
    #[error("compatibility backend HTTP response from {endpoint} was not valid JSON: {source}")]
    InvalidResponseJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("compatibility backend HTTP response from {endpoint} contained no choices")]
    EmptyChoicesInResponse { endpoint: String },
    #[error(
        "compatibility backend {selected_backend:?} does not support streaming generate in this preview contract"
    )]
    StreamingNotSupported { selected_backend: SelectedBackend },
    #[error(
        "compatibility backend {selected_backend:?} does not support sampling option {option} in this preview contract"
    )]
    UnsupportedSamplingOption {
        selected_backend: SelectedBackend,
        option: &'static str,
    },
}

pub fn run_blocking_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &CompatibilityBackendConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, CompatibilityBackendError> {
    match config {
        CompatibilityBackendConfig::LlamaCpp(config) => {
            if runtime.selected_backend != SelectedBackend::LlamaCpp {
                return Err(CompatibilityBackendError::BackendConfigMismatch {
                    configured_backend: SelectedBackend::LlamaCpp,
                    resolved_backend: runtime.selected_backend,
                });
            }
            run_llama_cpp_generate(request_id, runtime, config, request)
        }
        CompatibilityBackendConfig::Vllm(config) => {
            if runtime.selected_backend != SelectedBackend::Vllm {
                return Err(CompatibilityBackendError::BackendConfigMismatch {
                    configured_backend: SelectedBackend::Vllm,
                    resolved_backend: runtime.selected_backend,
                });
            }
            run_openai_compatible_generate(
                request_id,
                runtime,
                config,
                request,
                SelectedBackend::Vllm,
            )
        }
        CompatibilityBackendConfig::MistralRs(config) => {
            if runtime.selected_backend != SelectedBackend::MistralRs {
                return Err(CompatibilityBackendError::BackendConfigMismatch {
                    configured_backend: SelectedBackend::MistralRs,
                    resolved_backend: runtime.selected_backend,
                });
            }
            run_openai_compatible_generate(
                request_id,
                runtime,
                config,
                request,
                SelectedBackend::MistralRs,
            )
        }
        CompatibilityBackendConfig::Mlx(config) => {
            if runtime.selected_backend != SelectedBackend::Mlx {
                return Err(CompatibilityBackendError::BackendConfigMismatch {
                    configured_backend: SelectedBackend::Mlx,
                    resolved_backend: runtime.selected_backend,
                });
            }
            run_mlx_generate(request_id, runtime, config, request)
        }
    }
}

pub(crate) fn start_streaming_generate(
    runtime: &RuntimeReport,
    config: &CompatibilityBackendConfig,
    request: &GenerateRequest,
) -> Result<CompatibilityStreamHandle, CompatibilityBackendError> {
    match config {
        CompatibilityBackendConfig::LlamaCpp(config) => {
            if runtime.selected_backend != SelectedBackend::LlamaCpp {
                return Err(CompatibilityBackendError::BackendConfigMismatch {
                    configured_backend: SelectedBackend::LlamaCpp,
                    resolved_backend: runtime.selected_backend,
                });
            }
            start_llama_cpp_streaming_generate(config, request)
        }
        CompatibilityBackendConfig::Vllm(config) => {
            if runtime.selected_backend != SelectedBackend::Vllm {
                return Err(CompatibilityBackendError::BackendConfigMismatch {
                    configured_backend: SelectedBackend::Vllm,
                    resolved_backend: runtime.selected_backend,
                });
            }
            start_openai_compatible_streaming_generate(config, request, SelectedBackend::Vllm)
        }
        CompatibilityBackendConfig::MistralRs(config) => {
            if runtime.selected_backend != SelectedBackend::MistralRs {
                return Err(CompatibilityBackendError::BackendConfigMismatch {
                    configured_backend: SelectedBackend::MistralRs,
                    resolved_backend: runtime.selected_backend,
                });
            }
            start_openai_compatible_streaming_generate(config, request, SelectedBackend::MistralRs)
        }
        CompatibilityBackendConfig::Mlx(config) => {
            if runtime.selected_backend != SelectedBackend::Mlx {
                return Err(CompatibilityBackendError::BackendConfigMismatch {
                    configured_backend: SelectedBackend::Mlx,
                    resolved_backend: runtime.selected_backend,
                });
            }
            start_mlx_streaming_generate(config, request)
        }
    }
}

fn run_llama_cpp_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &LlamaCppConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, CompatibilityBackendError> {
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
) -> Result<CompatibilityStreamHandle, CompatibilityBackendError> {
    match config {
        LlamaCppConfig::Cli(_) => Err(CompatibilityBackendError::StreamingNotSupported {
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
) -> Result<GenerateResponse, CompatibilityBackendError> {
    if !request.input_tokens.is_empty() {
        return Err(CompatibilityBackendError::UnsupportedTokenPrompt {
            selected_backend: SelectedBackend::LlamaCpp,
        });
    }

    let prompt_text =
        request
            .input_text
            .clone()
            .ok_or(CompatibilityBackendError::MissingInputText {
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
    command.args(&config.extra_args);

    let output =
        run_command_with_timeout(command, command_display.clone(), COMPATIBILITY_CLI_TIMEOUT)?;

    if !output.status.success() {
        return Err(CompatibilityBackendError::CommandFailed {
            command: command_display,
            status: output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "terminated by signal".to_string()),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    let output_text = String::from_utf8(output.stdout).map_err(|source| {
        CompatibilityBackendError::NonUtf8Output {
            command: config.cli_path.display().to_string(),
            source,
        }
    })?;
    let output_text = trim_single_trailing_newline(output_text);

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
        // CLI output does not provide a structured stop reason; keep it unknown instead of
        // falsely reporting a token-budget stop.
        finish_reason: None,
        step_count: 0,
        ttft_step: None,
        route: GenerateRouteReport {
            execution_plan: Some("compatibility.llama_cpp.blocking_cli".to_string()),
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
) -> Result<GenerateResponse, CompatibilityBackendError> {
    let prompt = build_llama_cpp_prompt(request)?;
    let endpoint = config.completion_url();
    let payload = LlamaCppCompletionRequest {
        prompt,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: request.sampling.top_k,
        repeat_penalty: request.sampling.repetition_penalty,
        seed: request.sampling.seed,
        n_predict: request.max_output_tokens,
        stream: false,
        return_tokens: true,
        return_progress: false,
    };

    let response = send_json_post_request(&endpoint, &payload, None)?;
    let response: LlamaCppCompletionResponse = parse_json_response(&endpoint, response)?;

    let output_tokens = response.tokens;
    let output_token_logprobs = vec![None; output_tokens.len()];
    // Use tokens_evaluated for prompt token count when we don't have token IDs (text input).
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
        output_text: Some(response.content),
        prompt_token_count,
        output_token_count: None,
        status: GenerateStatus::Finished,
        finish_reason: finish_reason_from_stop_type(response.stop, response.stop_type.as_deref()),
        step_count: 0,
        ttft_step: None,
        route: llama_cpp_server_completion_route(
            "compatibility.llama_cpp.server_completion",
            response.tokens_cached,
        ),
        runtime: runtime.clone(),
    })
}

fn start_llama_cpp_server_completion_stream(
    config: &LlamaCppServerCompletionConfig,
    request: &GenerateRequest,
) -> Result<CompatibilityStreamHandle, CompatibilityBackendError> {
    let prompt = build_llama_cpp_prompt(request)?;
    let endpoint = config.completion_url();
    let payload = LlamaCppCompletionRequest {
        prompt,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: request.sampling.top_k,
        repeat_penalty: request.sampling.repetition_penalty,
        seed: request.sampling.seed,
        n_predict: request.max_output_tokens,
        stream: true,
        return_tokens: true,
        return_progress: true,
    };

    let response = send_json_post_request(&endpoint, &payload, Some("text/event-stream"))?;

    let reader: Box<dyn Read + Send> = Box::new(response.into_reader());
    Ok(CompatibilityStreamHandle::new(
        endpoint,
        CompatibilityStreamFormat::LlamaCppCompletion,
        reader,
    ))
}

fn run_mlx_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &MlxConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, CompatibilityBackendError> {
    match config {
        MlxConfig::Cli(config) => run_mlx_cli_generate(request_id, runtime, config, request),
        MlxConfig::ServerCompletions(config) => run_openai_compatible_generate(
            request_id,
            runtime,
            config,
            request,
            SelectedBackend::Mlx,
        ),
    }
}

fn start_mlx_streaming_generate(
    config: &MlxConfig,
    request: &GenerateRequest,
) -> Result<CompatibilityStreamHandle, CompatibilityBackendError> {
    match config {
        MlxConfig::Cli(_) => Err(CompatibilityBackendError::StreamingNotSupported {
            selected_backend: SelectedBackend::Mlx,
        }),
        MlxConfig::ServerCompletions(config) => {
            start_openai_compatible_streaming_generate(config, request, SelectedBackend::Mlx)
        }
    }
}

fn run_mlx_cli_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &MlxCliConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, CompatibilityBackendError> {
    if !request.input_tokens.is_empty() {
        return Err(CompatibilityBackendError::UnsupportedTokenPrompt {
            selected_backend: SelectedBackend::Mlx,
        });
    }

    let prompt_text =
        request
            .input_text
            .clone()
            .ok_or(CompatibilityBackendError::MissingInputText {
                selected_backend: SelectedBackend::Mlx,
            })?;

    if (request.sampling.repetition_penalty - 1.0).abs() > f32::EPSILON {
        return Err(CompatibilityBackendError::UnsupportedSamplingOption {
            selected_backend: SelectedBackend::Mlx,
            option: "repetition_penalty",
        });
    }

    let command_display = config.cli_path.display().to_string();
    let mut command = Command::new(&config.cli_path);
    if config.invoke_as_python_module {
        command.arg("-m").arg("mlx_lm.generate");
    }
    command
        .arg("--model")
        .arg(&config.model_path)
        .arg("--prompt")
        .arg(&prompt_text)
        .arg("--ignore-chat-template")
        .arg("--max-tokens")
        .arg(request.max_output_tokens.to_string())
        .arg("--temp")
        .arg(request.sampling.temperature.to_string())
        .arg("--top-p")
        .arg(request.sampling.top_p.to_string())
        .arg("--top-k")
        .arg(request.sampling.top_k.to_string())
        .arg("--seed")
        .arg(request.sampling.seed.to_string())
        .arg("--verbose")
        .arg("false");
    command.args(&config.extra_args);

    let output =
        run_command_with_timeout(command, command_display.clone(), COMPATIBILITY_CLI_TIMEOUT)?;

    if !output.status.success() {
        return Err(CompatibilityBackendError::CommandFailed {
            command: command_display,
            status: output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "terminated by signal".to_string()),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    let output_text = String::from_utf8(output.stdout).map_err(|source| {
        CompatibilityBackendError::NonUtf8Output {
            command: config.cli_path.display().to_string(),
            source,
        }
    })?;
    let output_text = trim_single_trailing_newline(output_text);

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
        // CLI output does not provide a structured stop reason; keep it unknown instead of
        // falsely reporting a token-budget stop.
        finish_reason: None,
        step_count: 0,
        ttft_step: None,
        route: GenerateRouteReport {
            execution_plan: Some("compatibility.mlx.blocking_cli".to_string()),
            attention_route: None,
            kv_mode: None,
            prefix_cache_path: None,
            barrier_mode: None,
            crossover_decisions: Default::default(),
        },
        runtime: runtime.clone(),
    })
}

fn run_openai_compatible_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &OpenAiCompatibleServerConfig,
    request: &GenerateRequest,
    selected_backend: SelectedBackend,
) -> Result<GenerateResponse, CompatibilityBackendError> {
    let prompt = build_openai_compatible_prompt(request, selected_backend)?;
    let endpoint = config.completions_url();
    let payload = openai_compatible_payload(request, prompt, selected_backend, false);

    let response = send_json_post_request(&endpoint, &payload, None)?;
    let response: OpenAiCompletionResponse = parse_json_response(&endpoint, response)?;
    let usage = response.usage;
    let choice = response.choices.into_iter().next().ok_or_else(|| {
        CompatibilityBackendError::EmptyChoicesInResponse {
            endpoint: endpoint.clone(),
        }
    })?;

    Ok(GenerateResponse {
        request_id,
        model_id: request.model_id.clone(),
        prompt_tokens: request.input_tokens.clone(),
        prompt_text: request.input_text.clone(),
        output_tokens: Vec::new(),
        output_token_logprobs: Vec::new(),
        output_text: Some(choice.text),
        prompt_token_count: usage.as_ref().map(|u| u.prompt_tokens),
        output_token_count: usage.as_ref().map(|u| u.completion_tokens),
        status: GenerateStatus::Finished,
        finish_reason: finish_reason_from_openai(choice.finish_reason.as_deref()),
        step_count: 0,
        ttft_step: None,
        route: openai_compatible_route(selected_backend, false),
        runtime: runtime.clone(),
    })
}

fn start_openai_compatible_streaming_generate(
    config: &OpenAiCompatibleServerConfig,
    request: &GenerateRequest,
    selected_backend: SelectedBackend,
) -> Result<CompatibilityStreamHandle, CompatibilityBackendError> {
    let prompt = build_openai_compatible_prompt(request, selected_backend)?;
    let endpoint = config.completions_url();
    let payload = openai_compatible_payload(request, prompt, selected_backend, true);

    let response = send_json_post_request(&endpoint, &payload, Some("text/event-stream"))?;

    let reader: Box<dyn Read + Send> = Box::new(response.into_reader());
    Ok(CompatibilityStreamHandle::new(
        endpoint,
        CompatibilityStreamFormat::OpenAiCompletion,
        reader,
    ))
}

fn build_llama_cpp_prompt(
    request: &GenerateRequest,
) -> Result<LlamaCppPrompt<'_>, CompatibilityBackendError> {
    match (
        request.input_tokens.is_empty(),
        request.input_text.as_deref(),
    ) {
        (false, Some(_)) => Err(CompatibilityBackendError::AmbiguousPromptInput {
            selected_backend: SelectedBackend::LlamaCpp,
        }),
        (false, None) => Ok(LlamaCppPrompt::Tokens(&request.input_tokens)),
        (true, Some(text)) => Ok(LlamaCppPrompt::Text(text)),
        (true, None) => Err(CompatibilityBackendError::MissingPromptInput {
            selected_backend: SelectedBackend::LlamaCpp,
        }),
    }
}

fn build_openai_compatible_prompt(
    request: &GenerateRequest,
    selected_backend: SelectedBackend,
) -> Result<&str, CompatibilityBackendError> {
    match (
        request.input_tokens.is_empty(),
        request.input_text.as_deref(),
    ) {
        (false, Some(_)) => {
            Err(CompatibilityBackendError::AmbiguousPromptInput { selected_backend })
        }
        (false, None) => {
            Err(CompatibilityBackendError::UnsupportedTokenPrompt { selected_backend })
        }
        (true, Some(text)) => Ok(text),
        (true, None) => Err(CompatibilityBackendError::MissingInputText { selected_backend }),
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
        Some("eos") => Some(GenerateFinishReason::Stop),
        // When stop=true but stop_type is absent or unrecognized, default to MaxOutputTokens
        // to match the native path's behavior for requests that finish without a clear stop reason.
        _ => Some(GenerateFinishReason::MaxOutputTokens),
    }
}

fn compatibility_http_agent() -> &'static ureq::Agent {
    COMPATIBILITY_HTTP_AGENT.get_or_init(|| {
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
    accept: Option<&str>,
) -> Result<ureq::Response, CompatibilityBackendError>
where
    T: Serialize + ?Sized,
{
    let body = serde_json::to_vec(payload).map_err(|source| {
        CompatibilityBackendError::SerializeRequestJson {
            endpoint: endpoint.to_string(),
            source,
        }
    })?;
    let mut request = compatibility_http_agent()
        .post(endpoint)
        .set("Content-Type", "application/json");
    if let Some(accept) = accept {
        request = request.set("Accept", accept);
    }

    match request.send_bytes(&body) {
        Ok(response) => Ok(response),
        Err(ureq::Error::Status(status, response)) => {
            let body = response
                .into_string()
                .unwrap_or_else(|_| "<failed to read response body>".to_string());
            Err(CompatibilityBackendError::HttpStatus {
                endpoint: endpoint.to_string(),
                status,
                body: body.trim().to_string(),
            })
        }
        Err(source) => Err(CompatibilityBackendError::HttpRequest {
            endpoint: endpoint.to_string(),
            source: Box::new(source),
        }),
    }
}

fn parse_json_response<T>(
    endpoint: &str,
    response: ureq::Response,
) -> Result<T, CompatibilityBackendError>
where
    T: DeserializeOwned,
{
    serde_json::from_reader(response.into_reader()).map_err(|source| {
        CompatibilityBackendError::InvalidResponseJson {
            endpoint: endpoint.to_string(),
            source,
        }
    })
}

fn finish_reason_from_openai(finish_reason: Option<&str>) -> Option<GenerateFinishReason> {
    match finish_reason {
        Some("length") => Some(GenerateFinishReason::MaxOutputTokens),
        Some("stop") => Some(GenerateFinishReason::Stop),
        None => None,
        // Unknown finish reasons (e.g. "content_filter"): default to MaxOutputTokens so the
        // response is marked complete rather than silently losing the finish reason.
        Some(_) => Some(GenerateFinishReason::MaxOutputTokens),
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

fn run_command_with_timeout(
    mut command: Command,
    command_display: String,
    timeout: Duration,
) -> Result<Output, CompatibilityBackendError> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| CompatibilityBackendError::CommandLaunch {
            command: command_display.clone(),
            source,
        })?;
    let deadline = Instant::now() + timeout;

    loop {
        match child
            .try_wait()
            .map_err(|source| CompatibilityBackendError::CommandLaunch {
                command: command_display.clone(),
                source,
            })? {
            Some(_) => {
                return child.wait_with_output().map_err(|source| {
                    CompatibilityBackendError::CommandLaunch {
                        command: command_display,
                        source,
                    }
                });
            }
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(CompatibilityBackendError::CommandTimedOut {
                    command: command_display,
                    timeout_seconds: timeout.as_secs(),
                });
            }
            None => thread::sleep(Duration::from_millis(10)),
        }
    }
}

fn normalize_base_url(mut value: String) -> String {
    while value.ends_with('/') {
        value.pop();
    }
    value
}

#[derive(Debug, Serialize)]
struct LlamaCppCompletionRequest<'a> {
    prompt: LlamaCppPrompt<'a>,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    seed: u64,
    n_predict: u32,
    stream: bool,
    return_tokens: bool,
    return_progress: bool,
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
    /// Number of prompt tokens evaluated by the server. Used to report accurate usage.
    #[serde(default)]
    tokens_evaluated: u32,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    seed: u64,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OpenAiStreamOptions>,
}

#[derive(Debug, Serialize)]
struct OpenAiStreamOptions {
    include_usage: bool,
}

fn openai_compatible_payload<'a>(
    request: &'a GenerateRequest,
    prompt: &'a str,
    selected_backend: SelectedBackend,
    stream: bool,
) -> OpenAiCompletionRequest<'a> {
    let include_mlx_fields = matches!(selected_backend, SelectedBackend::Mlx);
    OpenAiCompletionRequest {
        model: &request.model_id,
        prompt,
        max_tokens: request.max_output_tokens,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        seed: request.sampling.seed,
        stream,
        top_k: include_mlx_fields.then_some(request.sampling.top_k),
        repetition_penalty: include_mlx_fields.then_some(request.sampling.repetition_penalty),
        stream_options: (include_mlx_fields && stream).then_some(OpenAiStreamOptions {
            include_usage: true,
        }),
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct OpenAiCompletionChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiCompletionUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct OpenAiCompletionResponse {
    #[serde(default)]
    choices: Vec<OpenAiCompletionChoice>,
    #[serde(default)]
    usage: Option<OpenAiCompletionUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiCompletionChunk {
    #[serde(default)]
    choices: Vec<OpenAiCompletionChoice>,
    #[serde(default)]
    usage: Option<OpenAiCompletionUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

fn parse_openai_completion_stream_chunk(
    payload: &str,
) -> Result<CompatibilityStreamChunk, serde_json::Error> {
    let response: OpenAiCompletionChunk = serde_json::from_str(payload)?;
    let choice = response.choices.into_iter().next().unwrap_or_default();
    let usage = response.usage;
    let prompt_progress = usage.as_ref().map(|usage| CompatibilityPromptProgress {
        total: usage.prompt_tokens,
        cache: usage
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens)
            .unwrap_or_default(),
        processed: usage.prompt_tokens,
    });

    Ok(CompatibilityStreamChunk {
        content: choice.text,
        tokens: Vec::new(),
        stop: choice.finish_reason.is_some(),
        stop_type: choice.finish_reason,
        prompt_progress,
        prompt_token_count: usage.as_ref().map(|usage| usage.prompt_tokens),
        output_token_count: usage.as_ref().map(|usage| usage.completion_tokens),
    })
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

fn openai_compatible_route(
    selected_backend: SelectedBackend,
    streaming: bool,
) -> GenerateRouteReport {
    let execution_plan = match (selected_backend, streaming) {
        (SelectedBackend::Vllm, false) => "compatibility.vllm.server_completions",
        (SelectedBackend::Vllm, true) => "compatibility.vllm.server_completions_stream",
        (SelectedBackend::MistralRs, false) => "compatibility.mistral_rs.server_completions",
        (SelectedBackend::MistralRs, true) => "compatibility.mistral_rs.server_completions_stream",
        (SelectedBackend::Mlx, false) => "compatibility.mlx.server_completions",
        (SelectedBackend::Mlx, true) => "compatibility.mlx.server_completions_stream",
        _ => "compatibility.openai.server_completions",
    };

    GenerateRouteReport {
        execution_plan: Some(execution_plan.to_string()),
        attention_route: None,
        kv_mode: None,
        prefix_cache_path: None,
        barrier_mode: None,
        crossover_decisions: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::Value;

    use super::*;
    use crate::backend::{CapabilityReport, ResolutionPolicy, SupportTier};

    fn sample_runtime(selected_backend: SelectedBackend) -> RuntimeReport {
        RuntimeReport {
            selected_backend,
            support_tier: SupportTier::Compatibility,
            resolution_policy: ResolutionPolicy::AllowCompat,
            capabilities: CapabilityReport::compatibility_baseline(),
            fallback_reason: Some("native preview not ready".to_string()),
            host: Default::default(),
            metal_toolchain: Default::default(),
            native_runtime: None,
            native_model: None,
        }
    }

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            model_id: "qwen3_dense".to_string(),
            input_tokens: Vec::new(),
            input_text: Some("hello compatibility".to_string()),
            max_output_tokens: 2,
            sampling: GenerateSampling {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
                repetition_penalty: 1.0,
                seed: 7,
                deterministic: Some(true),
            },
            metadata: None,
        }
    }

    #[test]
    fn compatibility_http_agent_is_reused_across_requests() {
        assert!(std::ptr::eq(
            compatibility_http_agent(),
            compatibility_http_agent()
        ));
    }

    #[test]
    fn compatibility_cli_command_timeout_kills_hung_process() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-engine-sleep-cli-{unique}.py"));
        write_executable_script(
            &path,
            r#"#!/usr/bin/env python3
import time
time.sleep(10)
"#,
        );
        let error = run_command_with_timeout(
            Command::new(&path),
            path.display().to_string(),
            Duration::from_millis(20),
        )
        .expect_err("hung CLI command should time out");

        match error {
            CompatibilityBackendError::CommandTimedOut {
                command,
                timeout_seconds,
            } => {
                assert_eq!(command, path.display().to_string());
                assert_eq!(timeout_seconds, 0);
            }
            other => panic!("unexpected error: {other}"),
        }

        let _ = fs::remove_file(path);
    }

    fn fake_llama_cli_script() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-engine-fake-llama-{unique}.py"));
        let script = r#"#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

args = sys.argv[1:]
prompt = args[args.index("--prompt") + 1]
model = args[args.index("--model") + 1]
assert pathlib.Path(model).exists(), model
assert "--simple-io" in args
assert "--no-display-prompt" in args
assert "--single-turn" in args
assert "--log-disable" in args
assert args[args.index("--n-predict") + 1] == "2"
assert args[args.index("--seed") + 1] == "7"
assert args[args.index("--temp") + 1] == "0"
assert args[args.index("--top-p") + 1] == "1"
assert args[args.index("--top-k") + 1] == "1"
assert args[args.index("--repeat-penalty") + 1] == "1"
sys.stdout.write(f"compat::{prompt}")
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

    fn write_executable_script(path: &PathBuf, script: &str) {
        fs::write(path, script).expect("fake script should be written");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = fs::metadata(path)
                .expect("script metadata should exist")
                .permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(path, permissions).expect("script should be executable");
        }
    }

    fn fake_mlx_cli_script() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-engine-fake-mlx-{unique}.py"));
        let script = r#"#!/usr/bin/env python3
from __future__ import annotations

import sys

args = sys.argv[1:]
model = args[args.index("--model") + 1]
prompt = args[args.index("--prompt") + 1]
assert args[args.index("--max-tokens") + 1] == "2"
assert args[args.index("--temp") + 1] == "0"
assert args[args.index("--top-p") + 1] == "1"
assert args[args.index("--top-k") + 1] == "1"
assert args[args.index("--seed") + 1] == "7"
assert args[args.index("--verbose") + 1] == "false"
assert "--ignore-chat-template" in args
sys.stdout.write(f"mlx::{model}::{prompt}")
"#;

        write_executable_script(&path, script);
        path
    }

    fn fake_python_mlx_cli_script() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("python3-ax-engine-fake-mlx-{unique}.py"));
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
assert args[args.index("--top-k") + 1] == "1"
assert args[args.index("--seed") + 1] == "7"
assert args[args.index("--verbose") + 1] == "false"
assert "--ignore-chat-template" in args
sys.stdout.write(f"mlx::{model}::{prompt}")
"#;

        write_executable_script(&path, script);
        path
    }

    fn spawn_single_completion_server(
        response_status: u16,
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
            let request_head =
                String::from_utf8(request[..header_end].to_vec()).expect("headers should be utf8");
            assert!(
                request_head.starts_with("POST /completion HTTP/1.1\r\n"),
                "unexpected request head: {request_head}"
            );
            let body =
                String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
            let payload: Value = serde_json::from_str(&body).expect("request body should be json");
            assert_request(payload);

            let status_text = match response_status {
                200 => "OK",
                500 => "Internal Server Error",
                _ => "Response",
            };
            let response = format!(
                "HTTP/1.1 {response_status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream
                .write_all(response.as_bytes())
                .expect("response should write");
        });

        (format!("http://{address}"), handle)
    }

    fn spawn_single_openai_completion_server(
        response_status: u16,
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
            let request_head =
                String::from_utf8(request[..header_end].to_vec()).expect("headers should be utf8");
            assert!(
                request_head.starts_with("POST /v1/completions HTTP/1.1\r\n"),
                "unexpected request head: {request_head}"
            );
            let body =
                String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
            let payload: Value = serde_json::from_str(&body).expect("request body should be json");
            assert_request(payload);

            let status_text = match response_status {
                200 => "OK",
                500 => "Internal Server Error",
                _ => "Response",
            };
            let response = format!(
                "HTTP/1.1 {response_status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
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

    #[test]
    fn llama_cpp_cli_generate_uses_text_prompt_and_returns_text_output() {
        let script_path = fake_llama_cli_script();
        let model_path = std::env::temp_dir().join("ax-engine-fake-model.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        let response = run_blocking_generate(
            11,
            &sample_runtime(SelectedBackend::LlamaCpp),
            &CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::new(script_path, &model_path)),
            &sample_request(),
        )
        .expect("compatibility generate should succeed");

        assert_eq!(response.request_id, 11);
        assert_eq!(response.prompt_tokens, Vec::<u32>::new());
        assert_eq!(response.prompt_text.as_deref(), Some("hello compatibility"));
        assert_eq!(
            response.output_text.as_deref(),
            Some("compat::hello compatibility")
        );
        assert_eq!(response.output_tokens, Vec::<u32>::new());
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("compatibility.llama_cpp.blocking_cli")
        );
        assert_eq!(response.finish_reason, None);
    }

    #[test]
    fn llama_cpp_cli_generate_rejects_token_prompt_input() {
        let script_path = fake_llama_cli_script();
        let model_path = std::env::temp_dir().join("ax-engine-fake-model-token-input.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        let request = GenerateRequest {
            input_tokens: vec![1, 2, 3],
            ..sample_request()
        };
        let error = run_blocking_generate(
            12,
            &sample_runtime(SelectedBackend::LlamaCpp),
            &CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::new(script_path, &model_path)),
            &request,
        )
        .expect_err("cli compatibility generate should reject token prompt input");

        match error {
            CompatibilityBackendError::UnsupportedTokenPrompt { selected_backend } => {
                assert_eq!(selected_backend, SelectedBackend::LlamaCpp);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn llama_cpp_server_completion_supports_token_prompt_requests() {
        let (server_url, server_handle) = spawn_single_completion_server(
            200,
            serde_json::json!({
                "content": "compat server output",
                "tokens": [8, 9],
                "stop": true,
                "stop_type": "limit"
            })
            .to_string(),
            |payload| {
                assert_eq!(payload["prompt"], serde_json::json!([1, 2, 3]));
                assert_eq!(payload["stream"], Value::Bool(false));
                assert_eq!(payload["return_tokens"], Value::Bool(true));
                assert_eq!(payload["return_progress"], Value::Bool(false));
                assert_eq!(payload["n_predict"], Value::Number(2.into()));
                assert_eq!(payload["seed"], Value::Number(7.into()));
            },
        );

        let request = GenerateRequest {
            input_tokens: vec![1, 2, 3],
            input_text: None,
            ..sample_request()
        };
        let response = run_blocking_generate(
            13,
            &sample_runtime(SelectedBackend::LlamaCpp),
            &CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::server_completion(server_url)),
            &request,
        )
        .expect("server completion generate should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(response.request_id, 13);
        assert_eq!(response.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(response.prompt_text, None);
        assert_eq!(response.output_tokens, vec![8, 9]);
        assert_eq!(
            response.output_text.as_deref(),
            Some("compat server output")
        );
        assert_eq!(
            response.finish_reason,
            Some(GenerateFinishReason::MaxOutputTokens)
        );
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("compatibility.llama_cpp.server_completion")
        );
    }

    #[test]
    fn llama_cpp_server_completion_reports_delegated_prompt_cache_hits() {
        let (server_url, server_handle) = spawn_single_completion_server(
            200,
            serde_json::json!({
                "content": "compat server output",
                "tokens": [8, 9],
                "stop": true,
                "stop_type": "limit",
                "tokens_cached": 3
            })
            .to_string(),
            |payload| {
                assert_eq!(payload["prompt"], serde_json::json!([1, 2, 3]));
                assert_eq!(payload["stream"], Value::Bool(false));
                assert_eq!(payload["return_tokens"], Value::Bool(true));
                assert_eq!(payload["return_progress"], Value::Bool(false));
            },
        );

        let request = GenerateRequest {
            input_tokens: vec![1, 2, 3],
            input_text: None,
            ..sample_request()
        };
        let response = run_blocking_generate(
            14,
            &sample_runtime(SelectedBackend::LlamaCpp),
            &CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::server_completion(server_url)),
            &request,
        )
        .expect("server completion generate should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(
            response.route.prefix_cache_path.as_deref(),
            Some("delegated_prompt_cache")
        );
        assert_eq!(
            response
                .route
                .crossover_decisions
                .get("delegated_cached_tokens"),
            Some(&3)
        );
    }

    #[test]
    fn llama_cpp_server_completion_supports_text_prompt_requests() {
        let (server_url, server_handle) = spawn_single_completion_server(
            200,
            serde_json::json!({
                "content": "compat text output",
                "tokens": [4, 5],
                "stop": true,
                "stop_type": "eos"
            })
            .to_string(),
            |payload| {
                assert_eq!(
                    payload["prompt"],
                    Value::String("hello compatibility".to_string())
                );
                assert_eq!(
                    payload["temperature"],
                    Value::Number(serde_json::Number::from_f64(0.0).unwrap())
                );
                assert_eq!(
                    payload["top_p"],
                    Value::Number(serde_json::Number::from_f64(1.0).unwrap())
                );
                assert_eq!(payload["top_k"], Value::Number(1.into()));
            },
        );

        let response = run_blocking_generate(
            14,
            &sample_runtime(SelectedBackend::LlamaCpp),
            &CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::server_completion(server_url)),
            &sample_request(),
        )
        .expect("server completion generate should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(response.request_id, 14);
        assert_eq!(response.prompt_text.as_deref(), Some("hello compatibility"));
        assert_eq!(response.output_tokens, vec![4, 5]);
        assert_eq!(response.output_text.as_deref(), Some("compat text output"));
        assert_eq!(response.finish_reason, Some(GenerateFinishReason::Stop));
    }

    #[test]
    fn llama_cpp_server_completion_rejects_ambiguous_mixed_prompt_requests() {
        let request = GenerateRequest {
            input_tokens: vec![1, 2, 3],
            input_text: Some("hello compatibility".to_string()),
            ..sample_request()
        };
        let error = run_blocking_generate(
            15,
            &sample_runtime(SelectedBackend::LlamaCpp),
            &CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::server_completion(
                "http://127.0.0.1:1",
            )),
            &request,
        )
        .expect_err("mixed prompt request should fail before transport");

        match error {
            CompatibilityBackendError::AmbiguousPromptInput { selected_backend } => {
                assert_eq!(selected_backend, SelectedBackend::LlamaCpp);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn llama_cpp_server_completion_defaults_finish_reason_when_stop_type_absent() {
        // When stop=true but stop_type is not included in the response (older llama.cpp versions
        // or unusual server behavior), finish_reason should default to MaxOutputTokens rather than
        // returning None, consistent with native path behavior for an unknown stop reason.
        let (server_url, server_handle) = spawn_single_completion_server(
            200,
            serde_json::json!({
                "content": "no stop type",
                "tokens": [77],
                "stop": true
                // stop_type deliberately absent
            })
            .to_string(),
            |_| {},
        );

        let request = GenerateRequest {
            input_tokens: vec![1],
            input_text: None,
            ..sample_request()
        };
        let response = run_blocking_generate(
            99,
            &sample_runtime(SelectedBackend::LlamaCpp),
            &CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::server_completion(server_url)),
            &request,
        )
        .expect("server completion generate should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(
            response.finish_reason,
            Some(GenerateFinishReason::MaxOutputTokens),
            "absent stop_type with stop=true should default to MaxOutputTokens"
        );
    }

    #[test]
    fn openai_compatible_generate_errors_on_empty_choices() {
        // When the server returns a 200 response with an empty choices array, the error should be
        // reported rather than silently producing an empty response.
        let (server_url, server_handle) = spawn_single_openai_completion_server(
            200,
            serde_json::json!({ "choices": [] }).to_string(),
            |_| {},
        );

        let error = run_blocking_generate(
            98,
            &sample_runtime(SelectedBackend::Vllm),
            &CompatibilityBackendConfig::Vllm(OpenAiCompatibleServerConfig::new(server_url)),
            &sample_request(),
        )
        .expect_err("empty choices should be an error");
        server_handle.join().expect("server thread should finish");

        assert!(
            matches!(
                error,
                CompatibilityBackendError::EmptyChoicesInResponse { .. }
            ),
            "expected EmptyChoicesInResponse, got: {error}"
        );
    }

    #[test]
    fn vllm_server_completion_uses_openai_compatible_contract() {
        let (server_url, server_handle) = spawn_single_openai_completion_server(
            200,
            serde_json::json!({
                "choices": [{
                    "text": "compat vllm output",
                    "finish_reason": "length"
                }]
            })
            .to_string(),
            |payload| {
                assert_eq!(payload["model"], Value::String("qwen3_dense".to_string()));
                assert_eq!(
                    payload["prompt"],
                    Value::String("hello compatibility".to_string())
                );
                assert_eq!(payload["max_tokens"], Value::Number(2.into()));
                assert_eq!(payload["stream"], Value::Bool(false));
                assert_eq!(payload["seed"], Value::Number(7.into()));
            },
        );

        let response = run_blocking_generate(
            16,
            &sample_runtime(SelectedBackend::Vllm),
            &CompatibilityBackendConfig::Vllm(OpenAiCompatibleServerConfig::new(server_url)),
            &sample_request(),
        )
        .expect("vllm compatibility generate should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(response.request_id, 16);
        assert_eq!(response.output_tokens, Vec::<u32>::new());
        assert_eq!(response.output_text.as_deref(), Some("compat vllm output"));
        assert_eq!(
            response.finish_reason,
            Some(GenerateFinishReason::MaxOutputTokens)
        );
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("compatibility.vllm.server_completions")
        );
    }

    #[test]
    fn mistral_rs_server_completion_uses_openai_compatible_contract() {
        let (server_url, server_handle) = spawn_single_openai_completion_server(
            200,
            serde_json::json!({
                "choices": [{
                    "text": "compat mistral output",
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
            |payload| {
                assert_eq!(
                    payload["prompt"],
                    Value::String("hello compatibility".to_string())
                );
                assert_eq!(payload["stream"], Value::Bool(false));
            },
        );

        let response = run_blocking_generate(
            17,
            &sample_runtime(SelectedBackend::MistralRs),
            &CompatibilityBackendConfig::MistralRs(OpenAiCompatibleServerConfig::new(server_url)),
            &sample_request(),
        )
        .expect("mistral.rs compatibility generate should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(response.request_id, 17);
        assert_eq!(
            response.output_text.as_deref(),
            Some("compat mistral output")
        );
        assert_eq!(response.finish_reason, Some(GenerateFinishReason::Stop));
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("compatibility.mistral_rs.server_completions")
        );
    }

    #[test]
    fn mlx_server_completion_uses_openai_compatible_contract() {
        let (server_url, server_handle) = spawn_single_openai_completion_server(
            200,
            serde_json::json!({
                "choices": [{
                    "text": "compat mlx output",
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
            |payload| {
                assert_eq!(
                    payload["prompt"],
                    Value::String("hello compatibility".to_string())
                );
                assert_eq!(payload["stream"], Value::Bool(false));
                assert_eq!(payload["top_k"], Value::Number(1.into()));
                assert_eq!(
                    payload["repetition_penalty"],
                    Value::Number(serde_json::Number::from_f64(1.0).unwrap())
                );
            },
        );

        let response = run_blocking_generate(
            19,
            &sample_runtime(SelectedBackend::Mlx),
            &CompatibilityBackendConfig::Mlx(MlxConfig::server_completions(server_url)),
            &sample_request(),
        )
        .expect("mlx compatibility generate should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(response.request_id, 19);
        assert_eq!(response.output_text.as_deref(), Some("compat mlx output"));
        assert_eq!(response.finish_reason, Some(GenerateFinishReason::Stop));
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("compatibility.mlx.server_completions")
        );
    }

    #[test]
    fn mlx_cli_generate_uses_text_prompt_and_returns_text_output() {
        let script_path = fake_mlx_cli_script();
        let model_path = std::env::temp_dir().join("ax-engine-fake-mlx-model");

        let response = run_blocking_generate(
            20,
            &sample_runtime(SelectedBackend::Mlx),
            &CompatibilityBackendConfig::Mlx(MlxConfig::cli(script_path, &model_path)),
            &sample_request(),
        )
        .expect("mlx cli compatibility generate should succeed");

        assert_eq!(response.request_id, 20);
        assert_eq!(response.prompt_tokens, Vec::<u32>::new());
        assert_eq!(response.prompt_text.as_deref(), Some("hello compatibility"));
        let expected_output = format!("mlx::{}::hello compatibility", model_path.display());
        assert_eq!(
            response.output_text.as_deref(),
            Some(expected_output.as_str())
        );
        assert_eq!(response.output_tokens, Vec::<u32>::new());
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("compatibility.mlx.blocking_cli")
        );
        assert_eq!(response.finish_reason, None);
    }

    #[test]
    fn mlx_cli_generate_supports_python_module_launchers() {
        let script_path = fake_python_mlx_cli_script();
        let model_path = std::env::temp_dir().join("ax-engine-fake-mlx-python-model");

        let response = run_blocking_generate(
            22,
            &sample_runtime(SelectedBackend::Mlx),
            &CompatibilityBackendConfig::Mlx(MlxConfig::cli(script_path, &model_path)),
            &sample_request(),
        )
        .expect("mlx python module compatibility generate should succeed");

        let expected_output = format!("mlx::{}::hello compatibility", model_path.display());
        assert_eq!(
            response.output_text.as_deref(),
            Some(expected_output.as_str())
        );
        assert_eq!(
            response.route.execution_plan.as_deref(),
            Some("compatibility.mlx.blocking_cli")
        );
    }

    #[test]
    fn mlx_cli_generate_rejects_token_prompt_input() {
        let script_path = fake_mlx_cli_script();
        let model_path = std::env::temp_dir().join("ax-engine-fake-mlx-model-token-input");

        let request = GenerateRequest {
            input_tokens: vec![1, 2, 3],
            ..sample_request()
        };
        let error = run_blocking_generate(
            21,
            &sample_runtime(SelectedBackend::Mlx),
            &CompatibilityBackendConfig::Mlx(MlxConfig::cli(script_path, &model_path)),
            &request,
        )
        .expect_err("mlx cli compatibility generate should reject token prompt input");

        match error {
            CompatibilityBackendError::UnsupportedTokenPrompt { selected_backend } => {
                assert_eq!(selected_backend, SelectedBackend::Mlx);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn mlx_cli_generate_rejects_non_default_repetition_penalty() {
        let script_path = fake_mlx_cli_script();
        let model_path = std::env::temp_dir().join("ax-engine-fake-mlx-model-penalty");

        let mut request = sample_request();
        request.sampling.repetition_penalty = 1.1;
        let error = run_blocking_generate(
            23,
            &sample_runtime(SelectedBackend::Mlx),
            &CompatibilityBackendConfig::Mlx(MlxConfig::cli(script_path, &model_path)),
            &request,
        )
        .expect_err("mlx cli compatibility generate should reject unsupported repetition penalty");

        match error {
            CompatibilityBackendError::UnsupportedSamplingOption {
                selected_backend,
                option,
            } => {
                assert_eq!(selected_backend, SelectedBackend::Mlx);
                assert_eq!(option, "repetition_penalty");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn openai_compatible_server_stream_maps_sse_chunks() {
        let (server_url, server_handle) = spawn_single_openai_completion_server(
            200,
            concat!(
                "data: {\"choices\":[{\"text\":\"he\",\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"text\":\"llo\",\"finish_reason\":\"length\"}]}\n\n",
                "data: [DONE]\n\n"
            )
            .to_string(),
            |payload| {
                assert_eq!(payload["stream"], Value::Bool(true));
                assert_eq!(
                    payload["prompt"],
                    Value::String("hello compatibility".to_string())
                );
            },
        );

        let mut stream = start_streaming_generate(
            &sample_runtime(SelectedBackend::Vllm),
            &CompatibilityBackendConfig::Vllm(OpenAiCompatibleServerConfig::new(server_url)),
            &sample_request(),
        )
        .expect("openai-compatible stream should start");

        let first = stream
            .next_chunk()
            .expect("first chunk should parse")
            .expect("first chunk should exist");
        let second = stream
            .next_chunk()
            .expect("second chunk should parse")
            .expect("second chunk should exist");
        let terminal = stream.next_chunk().expect("done marker should parse");
        server_handle.join().expect("server thread should finish");

        assert_eq!(first.content, "he");
        assert!(!first.stop);
        assert_eq!(second.content, "llo");
        assert!(second.stop);
        assert_eq!(second.stop_type.as_deref(), Some("length"));
        assert!(terminal.is_none());
    }

    #[test]
    fn openai_compatible_server_rejects_token_prompt_input() {
        let request = GenerateRequest {
            input_tokens: vec![1, 2, 3],
            input_text: None,
            ..sample_request()
        };

        let error = run_blocking_generate(
            18,
            &sample_runtime(SelectedBackend::Vllm),
            &CompatibilityBackendConfig::Vllm(OpenAiCompatibleServerConfig::new(
                "http://127.0.0.1:1",
            )),
            &request,
        )
        .expect_err("openai-compatible compatibility generate should reject token prompts");

        match error {
            CompatibilityBackendError::UnsupportedTokenPrompt { selected_backend } => {
                assert_eq!(selected_backend, SelectedBackend::Vllm);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn openai_compatible_generate_defaults_finish_reason_for_unknown_values() {
        // When the backend returns a finish_reason that isn't "stop" or "length" (e.g.
        // "content_filter"), the response should still have a non-null finish_reason rather
        // than silently losing the completion signal.
        let (server_url, server_handle) = spawn_single_openai_completion_server(
            200,
            serde_json::json!({
                "choices": [{
                    "text": "filtered output",
                    "finish_reason": "content_filter"
                }]
            })
            .to_string(),
            |_| {},
        );

        let response = run_blocking_generate(
            97,
            &sample_runtime(SelectedBackend::Vllm),
            &CompatibilityBackendConfig::Vllm(OpenAiCompatibleServerConfig::new(server_url)),
            &sample_request(),
        )
        .expect("generate should succeed despite unknown finish_reason");
        server_handle.join().expect("server thread should finish");

        assert_eq!(
            response.finish_reason,
            Some(GenerateFinishReason::MaxOutputTokens),
            "unknown finish_reason should default to MaxOutputTokens"
        );
    }

    #[test]
    fn openai_compatible_generate_reports_usage_token_counts() {
        // When the upstream OpenAI-compatible server includes a usage object, the token counts
        // should be available in the response so the server can report accurate usage.
        let (server_url, server_handle) = spawn_single_openai_completion_server(
            200,
            serde_json::json!({
                "choices": [{
                    "text": "usage output",
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17
                }
            })
            .to_string(),
            |_| {},
        );

        let response = run_blocking_generate(
            96,
            &sample_runtime(SelectedBackend::Vllm),
            &CompatibilityBackendConfig::Vllm(OpenAiCompatibleServerConfig::new(server_url)),
            &sample_request(),
        )
        .expect("generate should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(
            response.prompt_token_count,
            Some(12),
            "prompt_token_count should be populated from usage object"
        );
        assert_eq!(
            response.output_token_count,
            Some(5),
            "output_token_count should be populated from usage object"
        );
    }

    #[test]
    fn llama_cpp_server_completion_reports_prompt_token_count_from_tokens_evaluated() {
        // When the llama.cpp server reports tokens_evaluated, it should be used for the
        // prompt token count (useful for text-prompt requests that have no input_tokens).
        let (server_url, server_handle) = spawn_single_completion_server(
            200,
            serde_json::json!({
                "content": "response text",
                "tokens": [10, 11, 12],
                "stop": true,
                "stop_type": "eos",
                "tokens_evaluated": 8
            })
            .to_string(),
            |_| {},
        );

        let response = run_blocking_generate(
            95,
            &sample_runtime(SelectedBackend::LlamaCpp),
            &CompatibilityBackendConfig::LlamaCpp(LlamaCppConfig::server_completion(server_url)),
            &sample_request(), // uses input_text, not input_tokens
        )
        .expect("generate should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(
            response.prompt_token_count,
            Some(8),
            "prompt_token_count should be set from tokens_evaluated when input_tokens is empty"
        );
        assert_eq!(
            response.output_token_count, None,
            "output_token_count is None since output_tokens holds the actual token IDs"
        );
    }
}
