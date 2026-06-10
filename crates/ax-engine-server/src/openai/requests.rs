use ax_engine_sdk::{
    EngineTokenizer, GenerateRequest, GenerateSampling, LlamaCppChatGenerateRequest,
    MlxLmChatGenerateRequest, RequestMultimodalInputs, SelectedBackend,
};
use axum::Json;
use axum::http::StatusCode;
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::app_state::AppState;
use crate::chat;
use crate::errors::{ErrorResponse, error_response};
use crate::openai::chat_requests::{build_llama_cpp_chat_messages, build_mlx_lm_chat_messages};
use crate::openai::schema::{
    OpenAiChatCompletionHttpRequest, OpenAiCompletionHttpRequest, OpenAiPromptInput,
    OpenAiStopInput,
};

pub(crate) const DEFAULT_OPENAI_MAX_TOKENS: u32 = 256;
static OPENAI_SEED_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(crate) use crate::openai::chat_requests::{
    chat_template_kwargs_for_model_id, openai_chat_stop_sequences, render_openai_chat_prompt,
};
use crate::openai::chat_requests::{
    messages_contain_inline_media, render_gemma4_unified_chat_with_media,
};
use crate::tasks::run_blocking_http_task;

pub(crate) struct OpenAiBuiltRequest {
    pub(crate) generate_request: GenerateRequest,
    pub(crate) stream: bool,
    pub(crate) response_options: OpenAiResponseOptions,
}

struct OpenAiBuiltPayload {
    sampling: GenerateSampling,
    multimodal_inputs: RequestMultimodalInputs,
    stop_sequences: Vec<String>,
    stream: bool,
    metadata: Option<String>,
}

pub(crate) struct GenerateRequestParts {
    pub(crate) input_tokens: Vec<u32>,
    pub(crate) input_text: Option<String>,
    pub(crate) multimodal_inputs: RequestMultimodalInputs,
    pub(crate) max_output_tokens: u32,
    pub(crate) sampling: GenerateSampling,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) metadata: Option<String>,
}

pub(crate) struct OpenAiBuiltMlxLmChatRequest {
    pub(crate) chat_request: MlxLmChatGenerateRequest,
    pub(crate) stream: bool,
    pub(crate) response_options: OpenAiResponseOptions,
}

pub(crate) struct OpenAiBuiltLlamaCppChatRequest {
    pub(crate) chat_request: LlamaCppChatGenerateRequest,
    pub(crate) stream: bool,
    pub(crate) response_options: OpenAiResponseOptions,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OpenAiResponseOptions {
    pub(crate) include_logprobs: bool,
    pub(crate) include_reasoning: bool,
    pub(crate) validate_json_object: bool,
    pub(crate) parse_tool_calls: bool,
}

impl OpenAiResponseOptions {
    fn from_completion_request(
        request: &OpenAiCompletionHttpRequest,
    ) -> Result<Self, (StatusCode, Json<ErrorResponse>)> {
        reject_unsupported_top_logprobs(request.top_logprobs)?;
        reject_unsupported_completion_logprobs(request.logprobs)?;
        Ok(Self {
            include_logprobs: request.logprobs.is_some(),
            include_reasoning: false,
            validate_json_object: openai_response_format_is_json_object(
                request.response_format.as_ref(),
            ),
            parse_tool_calls: false,
        })
    }

    fn from_chat_request(
        request: &OpenAiChatCompletionHttpRequest,
    ) -> Result<Self, (StatusCode, Json<ErrorResponse>)> {
        reject_unsupported_top_logprobs(request.top_logprobs)?;
        Ok(Self {
            include_logprobs: request.logprobs,
            include_reasoning: openai_reasoning_is_enabled(request.reasoning.as_ref()),
            validate_json_object: openai_response_format_is_json_object(
                request.response_format.as_ref(),
            ),
            parse_tool_calls: openai_tools_are_enabled(
                request.tools.as_ref(),
                request.tool_choice.as_ref(),
            ),
        })
    }

    /// Streaming chunks do not carry logprob, reasoning, or validated
    /// JSON-object payloads yet; fail closed instead of silently dropping a
    /// contract the caller asked for.
    pub(crate) fn reject_unsupported_streaming_contract(
        self,
        stream: bool,
    ) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
        if !stream {
            return Ok(());
        }
        if self.validate_json_object {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "unsupported_parameter",
                "response_format json_object validation is not supported for streaming requests yet"
                    .to_string(),
            ));
        }
        if self.include_logprobs {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "unsupported_parameter",
                "logprobs are not supported for streaming requests yet".to_string(),
            ));
        }
        if self.include_reasoning {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "unsupported_parameter",
                "reasoning output is not supported for streaming requests yet".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct OpenAiSamplingParams {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    repetition_context_size: Option<u32>,
    seed: Option<u64>,
}

impl OpenAiSamplingParams {
    fn from_completion_request(request: &OpenAiCompletionHttpRequest) -> Self {
        Self {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            min_p: request.min_p,
            repetition_penalty: request.repetition_penalty,
            repetition_context_size: request.repetition_context_size,
            seed: request.seed,
        }
    }

    fn from_chat_request(request: &OpenAiChatCompletionHttpRequest) -> Self {
        Self {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            min_p: request.min_p,
            repetition_penalty: request.repetition_penalty,
            repetition_context_size: request.repetition_context_size,
            seed: request.seed,
        }
    }
}

pub(crate) fn build_openai_completion_request(
    state: &AppState,
    request: OpenAiCompletionHttpRequest,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = openai_max_tokens(request.max_tokens);
    let sampling_params = OpenAiSamplingParams::from_completion_request(&request);
    let response_options = OpenAiResponseOptions::from_completion_request(&request)?;
    response_options.reject_unsupported_streaming_contract(request.stream)?;
    let structured_output = openai_response_format_is_structured(request.response_format.as_ref());
    let metadata = openai_workload_metadata(request.metadata, false, structured_output);
    let multimodal_inputs = request.multimodal_inputs;
    reject_openai_multimodal_inputs_without_native_mlx(
        state,
        "OpenAI completions",
        &multimodal_inputs,
    )?;
    let (input_tokens, input_text) = match request.prompt {
        OpenAiPromptInput::Text(text) => {
            reject_openai_multimodal_inputs_without_tokens(
                "OpenAI completions",
                &multimodal_inputs,
                false,
            )?;
            (Vec::new(), Some(text))
        }
        OpenAiPromptInput::TextBatch(prompts) => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!(
                    "batch completion prompts are not supported by this preview endpoint; send one prompt per request (received {})",
                    prompts.len()
                ),
            ));
        }
        OpenAiPromptInput::Tokens(tokens) => {
            reject_openai_multimodal_inputs_without_tokens(
                "OpenAI completions",
                &multimodal_inputs,
                !tokens.is_empty(),
            )?;
            (tokens, None)
        }
    };

    let payload = OpenAiBuiltPayload {
        sampling: build_openai_sampling(state, sampling_params),
        multimodal_inputs,
        stop_sequences: request
            .stop
            .map(OpenAiStopInput::into_vec)
            .unwrap_or_default(),
        stream: request.stream,
        metadata,
    };

    build_openai_generate_request(
        state,
        input_tokens,
        input_text,
        max_output_tokens,
        payload,
        response_options,
    )
}

/// Build an OpenAI chat request, offloading the build to the blocking pool
/// when the messages carry inline media. Media preprocessing decodes
/// image/audio bytes and can wait on an ffmpeg child process for MP4/WebM
/// video — seconds-scale blocking work that must not stall the async executor
/// threads. Text-only requests build inline; that path is template rendering.
pub(crate) async fn build_openai_chat_request_offloading_media(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    if !messages_contain_inline_media(&request.messages) {
        return build_openai_chat_request(state, request);
    }
    let state = state.clone();
    run_blocking_http_task(move || build_openai_chat_request(&state, request)).await
}

pub(crate) fn build_openai_chat_request(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = openai_max_tokens(request.max_tokens);
    let sampling_params = OpenAiSamplingParams::from_chat_request(&request);
    let response_options = OpenAiResponseOptions::from_chat_request(&request)?;
    response_options.reject_unsupported_streaming_contract(request.stream)?;
    let mut input_tokens = request.input_tokens;
    let mut multimodal_inputs = request.multimodal_inputs;
    reject_openai_multimodal_inputs_without_native_mlx(state, "OpenAI chat", &multimodal_inputs)?;
    reject_openai_multimodal_inputs_without_tokens(
        "OpenAI chat",
        &multimodal_inputs,
        !input_tokens.is_empty(),
    )?;
    let input_text = if !input_tokens.is_empty() {
        // Keep message validation and raw-media rejection even when the AX
        // extension supplies an already-tokenized prompt.
        let _ = build_mlx_lm_chat_messages(&request.messages)?;
        None
    } else {
        validate_native_chat_artifacts(state)?;
        // On native MLX, decode inline base64 image/audio parts into Gemma 4
        // unified soft-token spans + tensors. Falls back to text-only rendering
        // when there is no inline media.
        let media_prompt = if state.runtime_report.selected_backend == SelectedBackend::Mlx
            && multimodal_inputs.is_empty()
        {
            let model_dir = state.session_config.mlx_model_artifacts_dir().ok_or_else(|| {
                error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    "native MLX multimodal chat requires mlx_model_artifacts_dir with tokenizer.json and config".to_string(),
                )
            })?;
            render_gemma4_unified_chat_with_media(
                state.model_id.as_ref(),
                model_dir,
                &request.messages,
            )?
        } else {
            None
        };
        match media_prompt {
            Some(prompt) => {
                input_tokens = prompt.input_tokens;
                multimodal_inputs.gemma4_unified = Some(prompt.runtime_inputs);
                None
            }
            None => Some(render_openai_chat_prompt(
                state.model_id.as_ref(),
                &request.messages,
            )?),
        }
    };
    let tool_call = openai_tools_are_enabled(request.tools.as_ref(), request.tool_choice.as_ref());
    let structured_output = openai_response_format_is_structured(request.response_format.as_ref());
    let metadata = openai_workload_metadata(request.metadata, tool_call, structured_output);

    let payload = OpenAiBuiltPayload {
        sampling: build_openai_sampling(state, sampling_params),
        multimodal_inputs,
        stop_sequences: openai_chat_stop_sequences(state.model_id.as_ref(), request.stop),
        stream: request.stream,
        metadata,
    };

    build_openai_generate_request(
        state,
        input_tokens,
        input_text,
        max_output_tokens,
        payload,
        response_options,
    )
}

fn validate_native_chat_artifacts(
    state: &AppState,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    chat::validate_native_chat_artifact(
        state.model_id.as_ref(),
        state.session_config.mlx_model_artifacts_dir.as_deref(),
    )
    .map_err(|message| error_response(StatusCode::BAD_REQUEST, "invalid_request", message))
}

pub(crate) fn build_openai_mlx_lm_chat_request(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltMlxLmChatRequest, (StatusCode, Json<ErrorResponse>)> {
    reject_delegated_chat_extensions(&request.input_tokens, &request.multimodal_inputs)?;
    let max_output_tokens = openai_max_tokens(request.max_tokens);
    let sampling_params = OpenAiSamplingParams::from_chat_request(&request);
    let response_options = OpenAiResponseOptions::from_chat_request(&request)?;
    response_options.reject_unsupported_streaming_contract(request.stream)?;
    let messages = build_mlx_lm_chat_messages(&request.messages)?;
    let sampling = build_openai_sampling_with_default_repetition_penalty(sampling_params, 1.0);
    let stop_sequences = openai_chat_stop_sequences(state.model_id.as_ref(), request.stop);
    let tool_call = openai_tools_are_enabled(request.tools.as_ref(), request.tool_choice.as_ref());
    let structured_output = openai_response_format_is_structured(request.response_format.as_ref());
    let metadata = openai_workload_metadata(request.metadata, tool_call, structured_output);

    Ok(OpenAiBuiltMlxLmChatRequest {
        chat_request: MlxLmChatGenerateRequest {
            model_id: state.model_id.to_string(),
            messages,
            max_output_tokens,
            sampling,
            stop_sequences,
            metadata,
            chat_template_kwargs: chat_template_kwargs_for_model_id(state.model_id.as_ref()),
        },
        stream: request.stream,
        response_options,
    })
}

pub(crate) fn build_openai_llama_cpp_chat_request(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltLlamaCppChatRequest, (StatusCode, Json<ErrorResponse>)> {
    reject_delegated_chat_extensions(&request.input_tokens, &request.multimodal_inputs)?;
    let max_output_tokens = openai_max_tokens(request.max_tokens);
    let sampling_params = OpenAiSamplingParams::from_chat_request(&request);
    let response_options = OpenAiResponseOptions::from_chat_request(&request)?;
    response_options.reject_unsupported_streaming_contract(request.stream)?;
    let messages = build_llama_cpp_chat_messages(&request.messages)?;
    let sampling = build_openai_sampling_with_default_repetition_penalty(sampling_params, 1.0);
    let stop_sequences = openai_chat_stop_sequences(state.model_id.as_ref(), request.stop);
    let tool_call = openai_tools_are_enabled(request.tools.as_ref(), request.tool_choice.as_ref());
    let structured_output = openai_response_format_is_structured(request.response_format.as_ref());
    let metadata = openai_workload_metadata(request.metadata, tool_call, structured_output);

    Ok(OpenAiBuiltLlamaCppChatRequest {
        chat_request: LlamaCppChatGenerateRequest {
            model_id: state.model_id.to_string(),
            messages,
            max_output_tokens,
            sampling,
            stop_sequences,
            metadata,
        },
        stream: request.stream,
        response_options,
    })
}

fn build_openai_generate_request(
    state: &AppState,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    max_output_tokens: u32,
    payload: OpenAiBuiltPayload,
    response_options: OpenAiResponseOptions,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let (input_tokens, input_text) =
        tokenize_native_mlx_text_input(state, input_tokens, input_text)?;

    Ok(OpenAiBuiltRequest {
        generate_request: build_generate_request_internal(
            state,
            GenerateRequestParts {
                input_tokens,
                input_text,
                multimodal_inputs: payload.multimodal_inputs,
                max_output_tokens,
                sampling: payload.sampling,
                stop_sequences: payload.stop_sequences,
                metadata: payload.metadata,
            },
        ),
        stream: payload.stream,
        response_options,
    })
}

/// `(tokens, optional decoded text)` on success, or an HTTP error response.
type NativeMlxTokenizeResult =
    Result<(Vec<u32>, Option<String>), (StatusCode, Json<ErrorResponse>)>;

fn tokenize_native_mlx_text_input(
    state: &AppState,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
) -> NativeMlxTokenizeResult {
    if state.runtime_report.selected_backend != SelectedBackend::Mlx {
        return Ok((input_tokens, input_text));
    }
    if !input_tokens.is_empty() || input_text.is_none() {
        return Ok((input_tokens, input_text));
    }

    let input_text = input_text.expect("checked above");
    let Some(model_dir) = state.session_config.mlx_model_artifacts_dir() else {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "OpenAI-compatible text endpoints on native MLX require mlx_model_artifacts_dir with tokenizer.json".to_string(),
        ));
    };
    let tokenizer = EngineTokenizer::from_model_dir(model_dir).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("failed to load tokenizer for native MLX OpenAI text endpoint: {error}"),
        )
    })?;
    let input_tokens = tokenizer.encode(&input_text, false).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("failed to tokenize OpenAI text prompt for native MLX backend: {error}"),
        )
    })?;

    Ok((input_tokens, None))
}

fn reject_openai_multimodal_inputs_without_native_mlx(
    state: &AppState,
    route: &str,
    multimodal_inputs: &RequestMultimodalInputs,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if multimodal_inputs.is_empty() || state.runtime_report.selected_backend == SelectedBackend::Mlx
    {
        return Ok(());
    }
    Err(error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!(
            "{route} multimodal_inputs require native MLX backend; delegated text backends cannot consume Gemma4UnifiedRuntimeInputs"
        ),
    ))
}

fn reject_openai_multimodal_inputs_without_tokens(
    route: &str,
    multimodal_inputs: &RequestMultimodalInputs,
    has_input_tokens: bool,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if multimodal_inputs.is_empty() || has_input_tokens {
        return Ok(());
    }
    Err(error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!(
            "{route} multimodal_inputs require pre-tokenized input because Gemma4 unified media spans are absolute positions in the expanded prompt"
        ),
    ))
}

fn reject_delegated_chat_extensions(
    input_tokens: &[u32],
    multimodal_inputs: &RequestMultimodalInputs,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if !input_tokens.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "OpenAI chat input_tokens require native MLX backend; delegated chat backends render text messages upstream".to_string(),
        ));
    }
    if !multimodal_inputs.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "OpenAI chat multimodal_inputs require native MLX backend; delegated text backends cannot consume Gemma4UnifiedRuntimeInputs".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn build_generate_request_internal(
    state: &AppState,
    parts: GenerateRequestParts,
) -> GenerateRequest {
    GenerateRequest {
        model_id: state.model_id.to_string(),
        input_tokens: parts.input_tokens,
        input_text: parts.input_text,
        multimodal_inputs: parts.multimodal_inputs,
        max_output_tokens: parts.max_output_tokens,
        sampling: parts.sampling,
        stop_sequences: parts.stop_sequences,
        metadata: parts.metadata,
    }
}

fn build_openai_sampling(state: &AppState, params: OpenAiSamplingParams) -> GenerateSampling {
    let temperature = params.temperature.unwrap_or(0.0);
    let default_repetition_penalty =
        default_native_mlx_openai_repetition_penalty(state, temperature);
    build_openai_sampling_with_default_repetition_penalty(
        OpenAiSamplingParams {
            temperature: Some(temperature),
            ..params
        },
        default_repetition_penalty,
    )
}

fn build_openai_sampling_with_default_repetition_penalty(
    params: OpenAiSamplingParams,
    default_repetition_penalty: f32,
) -> GenerateSampling {
    let temperature = params.temperature.unwrap_or(0.0);
    GenerateSampling {
        temperature,
        top_p: params.top_p.unwrap_or(1.0),
        top_k: params.top_k.unwrap_or(0),
        min_p: params.min_p,
        repetition_penalty: params
            .repetition_penalty
            .unwrap_or(default_repetition_penalty),
        repetition_context_size: params.repetition_context_size,
        seed: params
            .seed
            .unwrap_or_else(|| default_openai_seed(temperature)),
        deterministic: None,
        ignore_eos: false,
    }
}

fn default_native_mlx_openai_repetition_penalty(state: &AppState, temperature: f32) -> f32 {
    if state.runtime_report.selected_backend != SelectedBackend::Mlx || temperature > 0.0 {
        return 1.0;
    }

    let model_id = state.model_id.to_ascii_lowercase();
    if model_id.contains("glm") {
        return 1.0;
    }
    if model_id.contains("qwen") || model_id.contains("gemma") {
        return 1.1;
    }
    1.0
}

fn default_openai_seed(temperature: f32) -> u64 {
    if temperature <= 0.0 {
        return 0;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let counter = OPENAI_SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let seed = now ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if seed == 0 { counter } else { seed }
}

fn openai_workload_metadata(
    metadata: Option<String>,
    tool_call: bool,
    structured_output: bool,
) -> Option<String> {
    if !tool_call && !structured_output {
        return metadata;
    }

    let mut hints = Map::new();
    if tool_call {
        hints.insert("ax_speculative_tool_call".to_string(), Value::Bool(true));
    }
    if structured_output {
        hints.insert(
            "ax_speculative_structured_output".to_string(),
            Value::Bool(true),
        );
    }

    let Some(metadata) = metadata else {
        return Some(Value::Object(hints).to_string());
    };
    if let Ok(Value::Object(mut object)) = serde_json::from_str::<Value>(&metadata) {
        for (key, value) in hints {
            object.entry(key).or_insert(value);
        }
        return Some(Value::Object(object).to_string());
    }

    let suffix = hints
        .keys()
        .map(|key| format!("{key}=true"))
        .collect::<Vec<_>>()
        .join("; ");
    Some(format!("{metadata}; {suffix}"))
}

fn openai_tools_are_enabled(tools: Option<&Value>, tool_choice: Option<&Value>) -> bool {
    tools.map(openai_value_is_present).unwrap_or(false)
        || tool_choice
            .map(openai_tool_choice_enables_tool_call)
            .unwrap_or(false)
}

fn openai_tool_choice_enables_tool_call(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "none" | "false" | "off")
        }
        Value::Array(values) => !values.is_empty(),
        Value::Object(object) => !object.is_empty(),
        Value::Number(value) => value.as_u64().unwrap_or(1) != 0,
    }
}

fn openai_response_format_is_structured(response_format: Option<&Value>) -> bool {
    let Some(response_format) = response_format else {
        return false;
    };
    match response_format {
        Value::Null => false,
        Value::String(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "text" | "none" | "false" | "off")
        }
        Value::Object(object) => object
            .get("type")
            .and_then(Value::as_str)
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !matches!(value.as_str(), "text" | "none" | "false" | "off")
            })
            .unwrap_or(!object.is_empty()),
        value => openai_value_is_present(value),
    }
}

fn openai_response_format_is_json_object(response_format: Option<&Value>) -> bool {
    let Some(response_format) = response_format else {
        return false;
    };
    match response_format {
        Value::String(value) => value.trim().eq_ignore_ascii_case("json_object"),
        Value::Object(object) => object
            .get("type")
            .and_then(Value::as_str)
            .map(|value| value.trim().eq_ignore_ascii_case("json_object"))
            .unwrap_or(false),
        _ => false,
    }
}

fn openai_value_is_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(object) => !object.is_empty(),
        Value::Number(value) => value.as_u64().unwrap_or(1) != 0,
    }
}

fn reject_unsupported_top_logprobs(
    top_logprobs: Option<u32>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if top_logprobs.unwrap_or(0) == 0 {
        return Ok(());
    }
    Err(error_response(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        "top_logprobs is not supported yet; AX currently exposes sampled-token logprobs only"
            .to_string(),
    ))
}

/// Legacy completions `logprobs` is a top-N alternative count, not a flag;
/// anything above sampled-token-only (`0`) fails closed until the runner
/// emits top-N alternatives.
fn reject_unsupported_completion_logprobs(
    logprobs: Option<u32>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if logprobs.unwrap_or(0) == 0 {
        return Ok(());
    }
    Err(error_response(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        "completions logprobs above 0 request top-N alternatives, which are not supported yet; \
         send logprobs=0 for sampled-token logprobs"
            .to_string(),
    ))
}

fn openai_reasoning_is_enabled(reasoning: Option<&Value>) -> bool {
    let Some(reasoning) = reasoning else {
        return false;
    };
    match reasoning {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "true" | "auto" | "exposed" | "include" | "enabled"
        ),
        Value::Object(object) => object
            .get("enabled")
            .or_else(|| object.get("include"))
            .or_else(|| object.get("mode"))
            .map(openai_value_is_present)
            .unwrap_or(!object.is_empty()),
        value => openai_value_is_present(value),
    }
}

fn openai_max_tokens(max_tokens: Option<u32>) -> u32 {
    max_tokens.unwrap_or(DEFAULT_OPENAI_MAX_TOKENS)
}
