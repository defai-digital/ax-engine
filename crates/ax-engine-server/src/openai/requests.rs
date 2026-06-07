use ax_engine_sdk::{
    EngineTokenizer, GenerateRequest, GenerateSampling, LlamaCppChatGenerateRequest,
    MlxLmChatGenerateRequest, SelectedBackend,
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

pub(crate) struct OpenAiBuiltRequest {
    pub(crate) generate_request: GenerateRequest,
    pub(crate) stream: bool,
}

struct OpenAiBuiltPayload {
    sampling: GenerateSampling,
    stop_sequences: Vec<String>,
    stream: bool,
    metadata: Option<String>,
}

pub(crate) struct OpenAiBuiltMlxLmChatRequest {
    pub(crate) chat_request: MlxLmChatGenerateRequest,
    pub(crate) stream: bool,
}

pub(crate) struct OpenAiBuiltLlamaCppChatRequest {
    pub(crate) chat_request: LlamaCppChatGenerateRequest,
    pub(crate) stream: bool,
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
    let structured_output = openai_response_format_is_structured(request.response_format.as_ref());
    let metadata = openai_workload_metadata(request.metadata, false, structured_output);
    let (input_tokens, input_text) = match request.prompt {
        OpenAiPromptInput::Text(text) => (Vec::new(), Some(text)),
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
        OpenAiPromptInput::Tokens(tokens) => (tokens, None),
    };

    let payload = OpenAiBuiltPayload {
        sampling: build_openai_sampling(state, sampling_params),
        stop_sequences: request
            .stop
            .map(OpenAiStopInput::into_vec)
            .unwrap_or_default(),
        stream: request.stream,
        metadata,
    };

    build_openai_generate_request(state, input_tokens, input_text, max_output_tokens, payload)
}

pub(crate) fn build_openai_chat_request(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = openai_max_tokens(request.max_tokens);
    let sampling_params = OpenAiSamplingParams::from_chat_request(&request);
    validate_native_chat_artifacts(state)?;
    let input_text = render_openai_chat_prompt(state.model_id.as_ref(), &request.messages)?;
    let tool_call = openai_tools_are_enabled(request.tools.as_ref(), request.tool_choice.as_ref());
    let structured_output = openai_response_format_is_structured(request.response_format.as_ref());
    let metadata = openai_workload_metadata(request.metadata, tool_call, structured_output);

    let payload = OpenAiBuiltPayload {
        sampling: build_openai_sampling(state, sampling_params),
        stop_sequences: openai_chat_stop_sequences(state.model_id.as_ref(), request.stop),
        stream: request.stream,
        metadata,
    };

    build_openai_generate_request(
        state,
        Vec::new(),
        Some(input_text),
        max_output_tokens,
        payload,
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
    let max_output_tokens = openai_max_tokens(request.max_tokens);
    let sampling_params = OpenAiSamplingParams::from_chat_request(&request);
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
    })
}

pub(crate) fn build_openai_llama_cpp_chat_request(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltLlamaCppChatRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = openai_max_tokens(request.max_tokens);
    let sampling_params = OpenAiSamplingParams::from_chat_request(&request);
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
    })
}

fn build_openai_generate_request(
    state: &AppState,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    max_output_tokens: u32,
    payload: OpenAiBuiltPayload,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let (input_tokens, input_text) =
        tokenize_native_mlx_text_input(state, input_tokens, input_text)?;

    Ok(OpenAiBuiltRequest {
        generate_request: build_generate_request_internal(
            state,
            input_tokens,
            input_text,
            max_output_tokens,
            payload.sampling,
            payload.stop_sequences,
            payload.metadata,
        ),
        stream: payload.stream,
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

pub(crate) fn build_generate_request_internal(
    state: &AppState,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    max_output_tokens: u32,
    sampling: GenerateSampling,
    stop_sequences: Vec<String>,
    metadata: Option<String>,
) -> GenerateRequest {
    GenerateRequest {
        model_id: state.model_id.to_string(),
        input_tokens,
        input_text,
        max_output_tokens,
        sampling,
        stop_sequences,
        metadata,
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

fn openai_max_tokens(max_tokens: Option<u32>) -> u32 {
    max_tokens.unwrap_or(DEFAULT_OPENAI_MAX_TOKENS)
}
