use std::fmt;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use url::Url;

use crate::backend::{RuntimeReport, SelectedBackend};
use crate::delegated_http::{
    DelegatedBearerCredential, DelegatedHttpHeaders, DelegatedHttpRequestError,
    DelegatedHttpRequestOptions, DelegatedHttpRetryPolicy, DelegatedHttpTimeouts,
    DelegatedProxyPolicy, DelegatedRedirectPolicy, DelegatedTlsPolicy, parse_json_response,
    send_get_with_options, send_json_post_with_options,
};
use crate::delegated_openai::{
    DEFAULT_MAX_DELEGATED_SSE_FRAME_BYTES, DelegatedChatMessage, DelegatedImageLimits,
    DelegatedOpenAiSseError, DelegatedOpenAiStreamHandle, DelegatedOpenAiValidationError,
    validate_delegated_image_budget,
};
use crate::generate::{
    GenerateFinishReason, GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateSampling,
    GenerateStatus,
};

const VLLM_USER_AGENT: &str = concat!("ax-engine-sdk/", env!("CARGO_PKG_VERSION"));
pub const UNLIMITED_OCR_DEFAULT_CONTEXT_LENGTH: u32 = 32_768;
pub const UNLIMITED_OCR_DEFAULT_MAX_OUTPUT_TOKENS: u32 = 8_192;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VllmModelProfile {
    #[default]
    OpenAiCompatible,
    UnlimitedOcr,
}

#[derive(Clone, Eq, PartialEq)]
pub struct NormalizedDelegatedBaseUrl {
    value: String,
    loopback: bool,
}

impl NormalizedDelegatedBaseUrl {
    pub fn parse(value: impl AsRef<str>, allow_remote: bool) -> Result<Self, VllmConfigError> {
        let raw = value.as_ref().trim();
        if raw.is_empty() {
            return Err(VllmConfigError::EmptyBaseUrl);
        }
        let mut url =
            Url::parse(raw).map_err(|error| VllmConfigError::InvalidBaseUrl(error.to_string()))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(VllmConfigError::UnsupportedUrlScheme {
                scheme: url.scheme().to_string(),
            });
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(VllmConfigError::EmbeddedCredentials);
        }
        if url.query().is_some() || url.fragment().is_some() {
            return Err(VllmConfigError::QueryOrFragment);
        }
        let host = url.host_str().ok_or(VllmConfigError::MissingHost)?;
        let loopback = host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|address| address.is_loopback());
        if !loopback && !allow_remote {
            return Err(VllmConfigError::RemoteEndpointNotAllowed);
        }
        if !loopback && url.scheme() != "https" {
            return Err(VllmConfigError::RemoteEndpointRequiresTls);
        }

        let mut path = url.path().trim_end_matches('/').to_string();
        if !path.ends_with("/v1") {
            path.push_str("/v1");
        }
        url.set_path(&path);
        let value = url.to_string().trim_end_matches('/').to_string();
        Ok(Self { value, loopback })
    }

    pub fn as_str(&self) -> &str {
        &self.value
    }

    pub fn is_loopback(&self) -> bool {
        self.loopback
    }

    pub fn models_url(&self) -> String {
        format!("{}/models", self.value)
    }

    pub fn completions_url(&self) -> String {
        format!("{}/completions", self.value)
    }

    pub fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.value)
    }

    pub fn redacted_authority(&self) -> String {
        if self.loopback {
            return "loopback".to_string();
        }
        Url::parse(&self.value)
            .ok()
            .and_then(|url| url.host_str().map(ToOwned::to_owned))
            .unwrap_or_else(|| "remote".to_string())
    }
}

impl fmt::Debug for NormalizedDelegatedBaseUrl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NormalizedDelegatedBaseUrl")
            .field("endpoint", &self.redacted_authority())
            .field("loopback", &self.loopback)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VllmConfig {
    ServerCompletion(VllmServerCompletionConfig),
}

impl VllmConfig {
    pub fn server_completion(
        base_url: impl AsRef<str>,
        upstream_model_id: impl Into<String>,
    ) -> Result<Self, VllmConfigError> {
        Ok(Self::ServerCompletion(VllmServerCompletionConfig::new(
            base_url,
            upstream_model_id,
        )?))
    }

    pub fn server(&self) -> &VllmServerCompletionConfig {
        match self {
            Self::ServerCompletion(config) => config,
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct VllmServerCompletionConfig {
    pub base_url: NormalizedDelegatedBaseUrl,
    pub upstream_model_id: String,
    pub timeouts: DelegatedHttpTimeouts,
    pub auth: Option<DelegatedBearerCredential>,
    pub tls: DelegatedTlsPolicy,
    pub allow_remote: bool,
    pub max_in_flight: Option<usize>,
    pub model_profile: VllmModelProfile,
    pub runtime_profile: Option<String>,
    pub image_limits: DelegatedImageLimits,
    pub max_sse_frame_bytes: usize,
}

impl VllmServerCompletionConfig {
    pub fn new(
        base_url: impl AsRef<str>,
        upstream_model_id: impl Into<String>,
    ) -> Result<Self, VllmConfigError> {
        Self::new_with_remote_policy(base_url, upstream_model_id, false)
    }

    pub fn new_with_remote_policy(
        base_url: impl AsRef<str>,
        upstream_model_id: impl Into<String>,
        allow_remote: bool,
    ) -> Result<Self, VllmConfigError> {
        let upstream_model_id = upstream_model_id.into();
        let upstream_model_id = upstream_model_id.trim();
        if upstream_model_id.is_empty() {
            return Err(VllmConfigError::EmptyUpstreamModelId);
        }
        Ok(Self {
            base_url: NormalizedDelegatedBaseUrl::parse(base_url, allow_remote)?,
            upstream_model_id: upstream_model_id.to_string(),
            timeouts: DelegatedHttpTimeouts::default(),
            auth: None,
            tls: DelegatedTlsPolicy::default(),
            allow_remote,
            max_in_flight: None,
            model_profile: VllmModelProfile::OpenAiCompatible,
            runtime_profile: None,
            image_limits: DelegatedImageLimits::default(),
            max_sse_frame_bytes: DEFAULT_MAX_DELEGATED_SSE_FRAME_BYTES,
        })
    }

    pub fn with_timeouts(mut self, timeouts: DelegatedHttpTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    pub fn with_auth(mut self, auth: Option<DelegatedBearerCredential>) -> Self {
        self.auth = auth;
        self
    }

    pub fn with_tls(mut self, tls: DelegatedTlsPolicy) -> Self {
        self.tls = tls;
        self
    }

    pub fn with_max_in_flight(
        mut self,
        max_in_flight: Option<usize>,
    ) -> Result<Self, VllmConfigError> {
        if max_in_flight == Some(0) {
            return Err(VllmConfigError::InvalidMaxInFlight);
        }
        self.max_in_flight = max_in_flight;
        Ok(self)
    }

    pub fn with_model_profile(mut self, model_profile: VllmModelProfile) -> Self {
        self.model_profile = model_profile;
        self
    }

    pub fn with_runtime_profile(
        mut self,
        runtime_profile: Option<String>,
    ) -> Result<Self, VllmConfigError> {
        self.runtime_profile = match runtime_profile {
            Some(profile) if profile.trim().is_empty() => {
                return Err(VllmConfigError::EmptyRuntimeProfile);
            }
            Some(profile) => Some(profile.trim().to_string()),
            None => None,
        };
        Ok(self)
    }

    fn request_options(
        &self,
        retry: DelegatedHttpRetryPolicy,
        accept: &str,
        request_id: Option<&str>,
    ) -> DelegatedHttpRequestOptions {
        let mut headers = DelegatedHttpHeaders::default()
            .with_accept(accept)
            .with_bearer(self.auth.clone())
            .with_user_agent(VLLM_USER_AGENT);
        if let Some(request_id) = request_id {
            headers = headers.with_request_id(request_id);
        }
        DelegatedHttpRequestOptions {
            timeouts: self.timeouts,
            retry,
            headers,
            tls: self.tls.clone(),
            proxy: DelegatedProxyPolicy::Disabled,
            redirects: DelegatedRedirectPolicy::Disabled,
            ..DelegatedHttpRequestOptions::default()
        }
    }
}

impl fmt::Debug for VllmServerCompletionConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VllmServerCompletionConfig")
            .field("base_url", &self.base_url)
            .field("upstream_model_id", &self.upstream_model_id)
            .field("timeouts", &self.timeouts)
            .field("auth", &self.auth)
            .field("tls", &self.tls)
            .field("allow_remote", &self.allow_remote)
            .field("max_in_flight", &self.max_in_flight)
            .field("model_profile", &self.model_profile)
            .field("runtime_profile", &self.runtime_profile)
            .field("image_limits", &self.image_limits)
            .field("max_sse_frame_bytes", &self.max_sse_frame_bytes)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum VllmConfigError {
    #[error("vLLM base URL must not be empty")]
    EmptyBaseUrl,
    #[error("vLLM base URL is invalid: {0}")]
    InvalidBaseUrl(String),
    #[error("vLLM base URL scheme {scheme} is not supported")]
    UnsupportedUrlScheme { scheme: String },
    #[error("vLLM base URL must include a host")]
    MissingHost,
    #[error("vLLM base URL must not contain embedded credentials")]
    EmbeddedCredentials,
    #[error("vLLM base URL must not contain a query or fragment")]
    QueryOrFragment,
    #[error("remote vLLM endpoint requires explicit allow_remote")]
    RemoteEndpointNotAllowed,
    #[error("remote vLLM endpoint requires verified HTTPS")]
    RemoteEndpointRequiresTls,
    #[error("vLLM upstream model id must not be empty")]
    EmptyUpstreamModelId,
    #[error("vLLM max_in_flight must be greater than zero")]
    InvalidMaxInFlight,
    #[error("vLLM runtime profile must not be empty")]
    EmptyRuntimeProfile,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VllmReadiness {
    Ready,
    Degraded,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VllmReadinessReport {
    pub readiness: VllmReadiness,
    pub upstream_model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub served_model_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<u32>,
}

pub fn check_readiness(config: &VllmConfig) -> Result<VllmReadinessReport, VllmBackendError> {
    let config = config.server();
    let endpoint = config.base_url.models_url();
    let options = config.request_options(
        DelegatedHttpRetryPolicy::readiness_default(),
        "application/json",
        Some("ax-vllm-readiness"),
    );
    let response = send_get_with_options(&endpoint, &options)
        .map_err(|error| map_http_error(&endpoint, error))?;
    let response: VllmModelsResponse = parse_vllm_json_response(response, &endpoint)?;
    let model = response
        .data
        .into_iter()
        .find(|model| model.id == config.upstream_model_id)
        .ok_or_else(|| VllmBackendError::ModelNotAdvertised {
            endpoint: endpoint.clone(),
            expected: config.upstream_model_id.clone(),
        })?;
    Ok(VllmReadinessReport {
        readiness: VllmReadiness::Ready,
        upstream_model_id: model.id,
        served_model_root: model.root.filter(|value| !value.trim().is_empty()),
        max_model_len: model.max_model_len.filter(|value| *value > 0),
    })
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct VllmRequestExtensions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_special_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<OrderedFloat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vllm_xargs: Option<VllmXargs>,
}

impl VllmRequestExtensions {
    pub fn unlimited_ocr(image_count: usize) -> Self {
        Self {
            skip_special_tokens: Some(false),
            repetition_penalty: None,
            vllm_xargs: Some(VllmXargs {
                ngram_size: Some(35),
                window_size: Some(if image_count > 1 { 1024 } else { 128 }),
            }),
        }
    }

    pub fn validate(&self) -> Result<(), VllmRequestValidationError> {
        if let Some(penalty) = self.repetition_penalty
            && (!penalty.get().is_finite() || penalty.get() <= 0.0)
        {
            return Err(VllmRequestValidationError::InvalidRepetitionPenalty);
        }
        if let Some(xargs) = self.vllm_xargs.as_ref() {
            xargs.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct OrderedFloat(f32);

impl OrderedFloat {
    pub fn new(value: f32) -> Result<Self, VllmRequestValidationError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(VllmRequestValidationError::InvalidRepetitionPenalty);
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl Eq for OrderedFloat {}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VllmXargs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ngram_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_size: Option<u32>,
}

impl VllmXargs {
    pub fn validate(&self) -> Result<(), VllmRequestValidationError> {
        let ngram_size = self
            .ngram_size
            .ok_or(VllmRequestValidationError::IncompleteNgramOptions)?;
        let window_size = self
            .window_size
            .ok_or(VllmRequestValidationError::IncompleteNgramOptions)?;
        if ngram_size == 0 || window_size == 0 || ngram_size > window_size {
            return Err(VllmRequestValidationError::InvalidNgramOptions {
                ngram_size,
                window_size,
            });
        }
        if ngram_size > 1024 || window_size > 65_536 {
            return Err(VllmRequestValidationError::NgramOptionsTooLarge);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct VllmChatGenerateRequest {
    /// AX-facing model identity. The configured upstream id is serialized on the wire.
    pub model_id: String,
    pub messages: Vec<DelegatedChatMessage>,
    pub max_output_tokens: u32,
    pub sampling: GenerateSampling,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
    pub extensions: VllmRequestExtensions,
}

impl VllmChatGenerateRequest {
    pub fn validate(
        &self,
        config: &VllmServerCompletionConfig,
    ) -> Result<(), VllmRequestValidationError> {
        if self.messages.is_empty() {
            return Err(VllmRequestValidationError::EmptyMessages);
        }
        if self.max_output_tokens == 0 {
            return Err(VllmRequestValidationError::InvalidMaxOutputTokens);
        }
        if !self.sampling.temperature.is_finite() || self.sampling.temperature < 0.0 {
            return Err(VllmRequestValidationError::InvalidTemperature);
        }
        if !self.sampling.top_p.is_finite()
            || self.sampling.top_p <= 0.0
            || self.sampling.top_p > 1.0
        {
            return Err(VllmRequestValidationError::InvalidTopP);
        }
        if self
            .sampling
            .min_p
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(VllmRequestValidationError::InvalidMinP);
        }
        self.extensions.validate()?;
        validate_delegated_image_budget(&self.messages, config.image_limits)?;
        if config.model_profile == VllmModelProfile::UnlimitedOcr {
            let image_count = image_count(&self.messages);
            if image_count == 0 {
                return Err(VllmRequestValidationError::UnlimitedOcrRequiresImage);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum VllmRequestValidationError {
    #[error(transparent)]
    DelegatedOpenAi(#[from] DelegatedOpenAiValidationError),
    #[error("vLLM chat messages must not be empty")]
    EmptyMessages,
    #[error("vLLM max output tokens must be greater than zero")]
    InvalidMaxOutputTokens,
    #[error("vLLM temperature must be finite and non-negative")]
    InvalidTemperature,
    #[error("vLLM top_p must be finite, greater than zero, and no greater than one")]
    InvalidTopP,
    #[error("vLLM min_p must be finite and between zero and one")]
    InvalidMinP,
    #[error("Unlimited-OCR vLLM profile requires at least one inline image")]
    UnlimitedOcrRequiresImage,
    #[error("vLLM repetition_penalty must be positive and finite")]
    InvalidRepetitionPenalty,
    #[error("vLLM ngram_size and window_size must be provided together")]
    IncompleteNgramOptions,
    #[error(
        "vLLM ngram_size={ngram_size} must be positive and no larger than window_size={window_size}"
    )]
    InvalidNgramOptions { ngram_size: u32, window_size: u32 },
    #[error("vLLM n-gram options exceed the certified profile limits")]
    NgramOptionsTooLarge,
}

fn image_count(messages: &[DelegatedChatMessage]) -> usize {
    messages
        .iter()
        .filter_map(|message| match &message.content {
            crate::delegated_openai::DelegatedChatContent::Parts(parts) => Some(parts),
            crate::delegated_openai::DelegatedChatContent::Text(_) => None,
        })
        .flatten()
        .filter(|part| {
            matches!(
                part,
                crate::delegated_openai::DelegatedChatContentPart::ImageUrl { .. }
            )
        })
        .count()
}

#[derive(Debug, Error)]
pub enum VllmBackendError {
    #[error("vLLM backend expected Vllm, got {resolved_backend:?}")]
    BackendConfigMismatch { resolved_backend: SelectedBackend },
    #[error(transparent)]
    InvalidRequest(#[from] VllmRequestValidationError),
    #[error("vLLM delegated backend requires input_text; token-array prompts are unsupported")]
    MissingInputText,
    #[error("vLLM delegated backend does not accept input_tokens")]
    UnsupportedTokenPrompt,
    #[error("failed to serialize vLLM request JSON for {endpoint}: {source}")]
    SerializeRequestJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("vLLM HTTP request to {endpoint} failed: {source}")]
    HttpRequest {
        endpoint: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error("vLLM HTTP transport config for {endpoint} failed: {message}")]
    HttpConfig { endpoint: String, message: String },
    #[error("vLLM HTTP response from {endpoint} returned status {status}: {body}")]
    HttpStatus {
        endpoint: String,
        status: u16,
        body: String,
        truncated: bool,
    },
    #[error("vLLM HTTP response from {endpoint} was not valid JSON: {source}")]
    InvalidResponseJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("vLLM at {endpoint} does not advertise configured model {expected}")]
    ModelNotAdvertised { endpoint: String, expected: String },
    #[error("vLLM response from {endpoint} did not include a completion choice")]
    MissingCompletionChoice { endpoint: String },
    #[error("vLLM response from {endpoint} included non-string assistant content")]
    InvalidAssistantContent { endpoint: String },
    #[error(transparent)]
    Sse(#[from] DelegatedOpenAiSseError),
}

#[derive(Debug)]
pub struct VllmStreamChunkResult {
    pub text: String,
    pub finish_reason: Option<String>,
    pub prompt_token_count: Option<u32>,
    pub output_token_count: Option<u32>,
}

#[derive(Debug)]
pub struct VllmStreamHandle {
    inner: DelegatedOpenAiStreamHandle,
}

impl VllmStreamHandle {
    pub fn next_chunk(&mut self) -> Result<Option<VllmStreamChunkResult>, VllmBackendError> {
        self.inner
            .next_chunk()
            .map(|chunk| {
                chunk.map(|chunk| VllmStreamChunkResult {
                    text: chunk.text,
                    finish_reason: chunk.finish_reason,
                    prompt_token_count: chunk.prompt_token_count,
                    output_token_count: chunk.output_token_count,
                })
            })
            .map_err(VllmBackendError::from)
    }
}

pub fn run_blocking_chat_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &VllmConfig,
    request: &VllmChatGenerateRequest,
) -> Result<GenerateResponse, VllmBackendError> {
    ensure_vllm_backend(runtime)?;
    let config = config.server();
    request.validate(config)?;
    let endpoint = config.base_url.chat_completions_url();
    let payload = build_chat_payload(config, request, false);
    let response = send_generation_post(&endpoint, &payload, config, None)?;
    let response: VllmChatCompletionResponse = parse_vllm_json_response(response, &endpoint)?;
    let choice = response.choices.into_iter().next().ok_or_else(|| {
        VllmBackendError::MissingCompletionChoice {
            endpoint: endpoint.clone(),
        }
    })?;
    let content =
        choice
            .message
            .content
            .ok_or_else(|| VllmBackendError::InvalidAssistantContent {
                endpoint: endpoint.clone(),
            })?;
    Ok(build_delegated_response(
        request_id,
        &request.model_id,
        runtime,
        None,
        content,
        response.usage.as_ref().map(|usage| usage.prompt_tokens),
        response.usage.as_ref().map(|usage| usage.completion_tokens),
        finish_reason_from_vllm(choice.finish_reason.as_deref()),
        "vllm.server_chat_completion",
    ))
}

pub fn start_streaming_chat_generate(
    runtime: &RuntimeReport,
    config: &VllmConfig,
    request: &VllmChatGenerateRequest,
) -> Result<VllmStreamHandle, VllmBackendError> {
    ensure_vllm_backend(runtime)?;
    let config = config.server();
    request.validate(config)?;
    let endpoint = config.base_url.chat_completions_url();
    let payload = build_chat_payload(config, request, true);
    let response = send_generation_post(&endpoint, &payload, config, Some("text/event-stream"))?;
    Ok(VllmStreamHandle {
        inner: DelegatedOpenAiStreamHandle::new(
            endpoint,
            Box::new(response.into_reader()),
            config.max_sse_frame_bytes,
        ),
    })
}

pub fn run_blocking_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &VllmConfig,
    request: &GenerateRequest,
) -> Result<GenerateResponse, VllmBackendError> {
    ensure_vllm_backend(runtime)?;
    let prompt = completion_prompt_text(request)?;
    let config = config.server();
    let endpoint = config.base_url.completions_url();
    let payload = VllmCompletionPayload {
        model: &config.upstream_model_id,
        prompt: &prompt,
        max_tokens: request.max_output_tokens,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: (request.sampling.top_k > 0).then_some(request.sampling.top_k),
        min_p: request.sampling.min_p,
        repetition_penalty: (request.sampling.repetition_penalty != 1.0)
            .then_some(request.sampling.repetition_penalty),
        seed: Some(request.sampling.seed),
        stream: false,
        stop: &request.stop_sequences,
        stream_options: None,
    };
    let response = send_generation_post(&endpoint, &payload, config, None)?;
    let response: VllmCompletionResponse = parse_vllm_json_response(response, &endpoint)?;
    let choice = response.choices.into_iter().next().ok_or_else(|| {
        VllmBackendError::MissingCompletionChoice {
            endpoint: endpoint.clone(),
        }
    })?;
    Ok(build_delegated_response(
        request_id,
        &request.model_id,
        runtime,
        Some(prompt),
        choice.text,
        response.usage.as_ref().map(|usage| usage.prompt_tokens),
        response.usage.as_ref().map(|usage| usage.completion_tokens),
        finish_reason_from_vllm(choice.finish_reason.as_deref()),
        "vllm.server_completion",
    ))
}

pub fn start_streaming_generate(
    runtime: &RuntimeReport,
    config: &VllmConfig,
    request: &GenerateRequest,
) -> Result<VllmStreamHandle, VllmBackendError> {
    ensure_vllm_backend(runtime)?;
    let prompt = completion_prompt_text(request)?;
    let config = config.server();
    let endpoint = config.base_url.completions_url();
    let payload = VllmCompletionPayload {
        model: &config.upstream_model_id,
        prompt: &prompt,
        max_tokens: request.max_output_tokens,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: (request.sampling.top_k > 0).then_some(request.sampling.top_k),
        min_p: request.sampling.min_p,
        repetition_penalty: (request.sampling.repetition_penalty != 1.0)
            .then_some(request.sampling.repetition_penalty),
        seed: Some(request.sampling.seed),
        stream: true,
        stop: &request.stop_sequences,
        stream_options: Some(VllmStreamOptions {
            include_usage: true,
        }),
    };
    let response = send_generation_post(&endpoint, &payload, config, Some("text/event-stream"))?;
    Ok(VllmStreamHandle {
        inner: DelegatedOpenAiStreamHandle::new(
            endpoint,
            Box::new(response.into_reader()),
            config.max_sse_frame_bytes,
        ),
    })
}

pub fn finish_reason_from_vllm(value: Option<&str>) -> Option<GenerateFinishReason> {
    match value {
        Some("stop") => Some(GenerateFinishReason::Stop),
        Some("length") => Some(GenerateFinishReason::MaxOutputTokens),
        Some("content_filter") => Some(GenerateFinishReason::ContentFilter),
        Some(other) => {
            tracing::warn!(
                finish_reason = other,
                "vLLM returned an unknown finish reason; reporting an error finish"
            );
            Some(GenerateFinishReason::Error)
        }
        None => None,
    }
}

fn ensure_vllm_backend(runtime: &RuntimeReport) -> Result<(), VllmBackendError> {
    if runtime.selected_backend == SelectedBackend::Vllm {
        Ok(())
    } else {
        Err(VllmBackendError::BackendConfigMismatch {
            resolved_backend: runtime.selected_backend,
        })
    }
}

fn completion_prompt_text(request: &GenerateRequest) -> Result<String, VllmBackendError> {
    if !request.input_tokens.is_empty() {
        return Err(VllmBackendError::UnsupportedTokenPrompt);
    }
    request
        .input_text
        .clone()
        .ok_or(VllmBackendError::MissingInputText)
}

fn send_generation_post<T>(
    endpoint: &str,
    payload: &T,
    config: &VllmServerCompletionConfig,
    accept: Option<&str>,
) -> Result<ureq::Response, VllmBackendError>
where
    T: Serialize + ?Sized,
{
    let options = config.request_options(
        DelegatedHttpRetryPolicy::Never,
        accept.unwrap_or("application/json"),
        None,
    );
    send_json_post_with_options(endpoint, payload, &options)
        .map_err(|error| map_http_error(endpoint, error))
}

fn map_http_error(endpoint: &str, error: DelegatedHttpRequestError) -> VllmBackendError {
    match error {
        DelegatedHttpRequestError::Serialize(source) => VllmBackendError::SerializeRequestJson {
            endpoint: endpoint.to_string(),
            source,
        },
        DelegatedHttpRequestError::Status {
            status,
            body,
            truncated,
        } => VllmBackendError::HttpStatus {
            endpoint: endpoint.to_string(),
            status,
            body,
            truncated,
        },
        DelegatedHttpRequestError::Request(source) => VllmBackendError::HttpRequest {
            endpoint: endpoint.to_string(),
            source,
        },
        DelegatedHttpRequestError::Config(source) => VllmBackendError::HttpConfig {
            endpoint: endpoint.to_string(),
            message: source.to_string(),
        },
    }
}

fn parse_vllm_json_response<T>(
    response: ureq::Response,
    endpoint: &str,
) -> Result<T, VllmBackendError>
where
    T: DeserializeOwned,
{
    parse_json_response(response, |source| VllmBackendError::InvalidResponseJson {
        endpoint: endpoint.to_string(),
        source,
    })
}

fn build_delegated_response(
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

fn build_chat_payload<'a>(
    config: &'a VllmServerCompletionConfig,
    request: &'a VllmChatGenerateRequest,
    stream: bool,
) -> VllmChatCompletionPayload<'a> {
    VllmChatCompletionPayload {
        model: &config.upstream_model_id,
        messages: &request.messages,
        max_tokens: request.max_output_tokens,
        temperature: request.sampling.temperature,
        top_p: request.sampling.top_p,
        top_k: (request.sampling.top_k > 0).then_some(request.sampling.top_k),
        min_p: request.sampling.min_p,
        repetition_penalty: request.extensions.repetition_penalty.map(OrderedFloat::get),
        seed: request.seed,
        stream,
        stop: &request.stop_sequences,
        skip_special_tokens: request.extensions.skip_special_tokens,
        vllm_xargs: request.extensions.vllm_xargs.as_ref(),
        stream_options: stream.then_some(VllmStreamOptions {
            include_usage: true,
        }),
    }
}

#[derive(Debug, Serialize)]
struct VllmChatCompletionPayload<'a> {
    model: &'a str,
    messages: &'a [DelegatedChatMessage],
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    stream: bool,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    stop: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    skip_special_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vllm_xargs: Option<&'a VllmXargs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<VllmStreamOptions>,
}

#[derive(Debug, Serialize)]
struct VllmCompletionPayload<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    stream: bool,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    stop: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<VllmStreamOptions>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct VllmStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
struct VllmModelsResponse {
    #[serde(default)]
    data: Vec<VllmModelEntry>,
}

#[derive(Debug, Deserialize)]
struct VllmModelEntry {
    id: String,
    #[serde(default)]
    root: Option<String>,
    #[serde(default)]
    max_model_len: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct VllmUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct VllmChatCompletionResponse {
    #[serde(default)]
    choices: Vec<VllmChatCompletionChoice>,
    #[serde(default)]
    usage: Option<VllmUsage>,
}

#[derive(Debug, Deserialize)]
struct VllmChatCompletionChoice {
    #[serde(default)]
    message: VllmChatCompletionMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct VllmChatCompletionMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VllmCompletionResponse {
    #[serde(default)]
    choices: Vec<VllmCompletionChoice>,
    #[serde(default)]
    usage: Option<VllmUsage>,
}

#[derive(Debug, Deserialize)]
struct VllmCompletionChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use serde_json::Value;

    use super::*;
    use crate::backend::{BackendPolicy, ResolvedBackend};
    use crate::delegated_openai::{DelegatedChatMessage, DelegatedChatRole};

    #[test]
    fn normalizes_v1_and_enforces_remote_tls_policy() {
        let local = NormalizedDelegatedBaseUrl::parse("http://localhost:8000/", false)
            .unwrap_or_else(|error| panic!("loopback URL should pass: {error}"));
        assert_eq!(local.as_str(), "http://localhost:8000/v1");
        let already_v1 = NormalizedDelegatedBaseUrl::parse("http://127.0.0.1:8000/v1/", false)
            .unwrap_or_else(|error| panic!("v1 URL should pass: {error}"));
        assert_eq!(already_v1.as_str(), "http://127.0.0.1:8000/v1");
        assert_eq!(
            NormalizedDelegatedBaseUrl::parse("http://example.test", true),
            Err(VllmConfigError::RemoteEndpointRequiresTls)
        );
        assert_eq!(
            NormalizedDelegatedBaseUrl::parse("https://example.test", false),
            Err(VllmConfigError::RemoteEndpointNotAllowed)
        );
        assert!(NormalizedDelegatedBaseUrl::parse("https://example.test", true).is_ok());
        assert_eq!(
            NormalizedDelegatedBaseUrl::parse("https://user:secret@example.test", true),
            Err(VllmConfigError::EmbeddedCredentials)
        );
    }

    #[test]
    fn config_debug_redacts_bearer_secret() {
        let credential = DelegatedBearerCredential::new("super-secret")
            .unwrap_or_else(|error| panic!("credential should pass: {error}"));
        let config = VllmServerCompletionConfig::new("http://127.0.0.1:8000", "candidate")
            .unwrap_or_else(|error| panic!("config should pass: {error}"))
            .with_auth(Some(credential));
        let debug = format!("{config:?}");
        assert!(!debug.contains("super-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn readiness_retries_get_and_generation_never_retries_post() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .unwrap_or_else(|error| panic!("listener should bind: {error}"));
        let address = listener
            .local_addr()
            .unwrap_or_else(|error| panic!("listener should have address: {error}"));
        let handle = thread::spawn(move || {
            for attempt in 0..2 {
                let (mut stream, _) = listener
                    .accept()
                    .unwrap_or_else(|error| panic!("GET should connect: {error}"));
                let _request = read_http_request(&mut stream);
                if attempt == 0 {
                    stream
                        .write_all(
                            b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .unwrap_or_else(|error| panic!("503 should write: {error}"));
                } else {
                    let body = r#"{"data":[{"id":"candidate","root":"baidu/Unlimited-OCR","max_model_len":32768}]}"#;
                    write_json_response(&mut stream, body);
                }
            }
        });
        let config = VllmConfig::server_completion(format!("http://{address}"), "candidate")
            .unwrap_or_else(|error| panic!("config should pass: {error}"));
        let readiness = check_readiness(&config)
            .unwrap_or_else(|error| panic!("readiness should retry: {error}"));
        assert_eq!(readiness.readiness, VllmReadiness::Ready);
        assert_eq!(readiness.max_model_len, Some(32_768));
        handle
            .join()
            .unwrap_or_else(|_| panic!("server thread should finish"));

        let listener = TcpListener::bind("127.0.0.1:0")
            .unwrap_or_else(|error| panic!("listener should bind: {error}"));
        let address = listener
            .local_addr()
            .unwrap_or_else(|error| panic!("listener should have address: {error}"));
        let handle = thread::spawn(move || {
            let (stream, _) = listener
                .accept()
                .unwrap_or_else(|error| panic!("POST should connect: {error}"));
            drop(stream);
            listener
                .set_nonblocking(true)
                .unwrap_or_else(|error| panic!("listener should become nonblocking: {error}"));
            thread::sleep(std::time::Duration::from_millis(100));
            assert!(
                listener.accept().is_err(),
                "generation POST must not open a retry connection"
            );
        });
        let config = VllmConfig::server_completion(format!("http://{address}"), "candidate")
            .unwrap_or_else(|error| panic!("config should pass: {error}"));
        let runtime = runtime_report();
        let request = VllmChatGenerateRequest {
            model_id: "public-model".to_string(),
            messages: vec![DelegatedChatMessage::text(DelegatedChatRole::User, "hello")],
            max_output_tokens: 8,
            sampling: GenerateSampling::default(),
            seed: None,
            stop_sequences: Vec::new(),
            extensions: VllmRequestExtensions::default(),
        };
        assert!(run_blocking_chat_generate(1, &runtime, &config, &request).is_err());
        handle
            .join()
            .unwrap_or_else(|_| panic!("server thread should finish"));
    }

    #[test]
    fn blocking_chat_uses_upstream_model_and_preserves_public_identity() {
        let (base_url, handle) = spawn_json_server(
            r#"{"choices":[{"message":{"content":"recognized"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":2}}"#,
            |request, payload| {
                assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
                assert_eq!(payload["model"], "upstream-model");
                assert_eq!(payload["messages"][0]["content"], "hello");
                assert_eq!(payload["stream"], false);
            },
        );
        let config = VllmConfig::server_completion(base_url, "upstream-model")
            .unwrap_or_else(|error| panic!("config should pass: {error}"));
        let request = VllmChatGenerateRequest {
            model_id: "public-model".to_string(),
            messages: vec![DelegatedChatMessage::text(DelegatedChatRole::User, "hello")],
            max_output_tokens: 8,
            sampling: GenerateSampling::default(),
            seed: None,
            stop_sequences: Vec::new(),
            extensions: VllmRequestExtensions::default(),
        };
        let response = run_blocking_chat_generate(7, &runtime_report(), &config, &request)
            .unwrap_or_else(|error| panic!("chat should pass: {error}"));
        assert_eq!(response.model_id, "public-model");
        assert_eq!(response.output_text.as_deref(), Some("recognized"));
        assert_eq!(response.prompt_token_count, Some(9));
        assert_eq!(response.output_token_count, Some(2));
        handle
            .join()
            .unwrap_or_else(|_| panic!("server thread should finish"));
    }

    fn runtime_report() -> RuntimeReport {
        RuntimeReport::from_resolution(
            &BackendPolicy::allow_vllm(),
            &ResolvedBackend::vllm("test vLLM route"),
        )
    }

    fn spawn_json_server(
        body: &'static str,
        assertion: impl FnOnce(&str, Value) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .unwrap_or_else(|error| panic!("listener should bind: {error}"));
        let address = listener
            .local_addr()
            .unwrap_or_else(|error| panic!("listener should have address: {error}"));
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener
                .accept()
                .unwrap_or_else(|error| panic!("request should connect: {error}"));
            let request = read_http_request(&mut stream);
            let request_text = String::from_utf8_lossy(&request);
            let body_start = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| position + 4)
                .unwrap_or_else(|| panic!("headers should terminate"));
            let payload = serde_json::from_slice(&request[body_start..])
                .unwrap_or_else(|error| panic!("body should be JSON: {error}"));
            assertion(&request_text, payload);
            write_json_response(&mut stream, body);
        });
        (format!("http://{address}"), handle)
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        let mut body_end = None;
        loop {
            let read = stream
                .read(&mut buffer)
                .unwrap_or_else(|error| panic!("request should read: {error}"));
            if read == 0 {
                return request;
            }
            request.extend_from_slice(&buffer[..read]);
            if body_end.is_none()
                && let Some(header_end) = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|position| position + 4)
            {
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                body_end = Some(header_end + length);
            }
            if body_end.is_some_and(|body_end| request.len() >= body_end) {
                return request;
            }
        }
    }

    fn write_json_response(stream: &mut std::net::TcpStream, body: &str) {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(response.as_bytes())
            .unwrap_or_else(|error| panic!("response should write: {error}"));
    }
}
