use ax_engine_sdk::{
    GenerateRequest, GenerateSampling, LlamaCppChatGenerateRequest, LlamaCppChatMessage,
    MlxLmChatGenerateRequest, MlxLmChatMessage,
};
use axum::Json;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::chat;
use crate::errors::{ErrorResponse, error_response};
use crate::openai::schema::{
    OpenAiChatCompletionHttpRequest, OpenAiChatContent, OpenAiChatMessage,
    OpenAiCompletionHttpRequest, OpenAiPromptInput, OpenAiStopInput,
};

pub(crate) const DEFAULT_OPENAI_MAX_TOKENS: u32 = 256;

type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);
type ChatMessagePairs = Vec<(String, String)>;

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

pub(crate) fn build_openai_completion_request(
    state: &AppState,
    request: OpenAiCompletionHttpRequest,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = openai_max_tokens(request.max_tokens);
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
        sampling: build_openai_sampling(
            request.temperature,
            request.top_p,
            request.top_k,
            request.min_p,
            request.repetition_penalty,
            request.repetition_context_size,
            request.seed,
        ),
        stop_sequences: request
            .stop
            .map(OpenAiStopInput::into_vec)
            .unwrap_or_default(),
        stream: request.stream,
        metadata: request.metadata,
    };

    Ok(build_openai_generate_request(
        state,
        input_tokens,
        input_text,
        max_output_tokens,
        payload,
    ))
}

pub(crate) fn build_openai_chat_request(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = openai_max_tokens(request.max_tokens);
    let input_text = render_openai_chat_prompt(state.model_id.as_ref(), &request.messages)?;

    let payload = OpenAiBuiltPayload {
        sampling: build_openai_sampling(
            request.temperature,
            request.top_p,
            request.top_k,
            request.min_p,
            request.repetition_penalty,
            request.repetition_context_size,
            request.seed,
        ),
        stop_sequences: openai_chat_stop_sequences(state.model_id.as_ref(), request.stop),
        stream: request.stream,
        metadata: request.metadata,
    };

    Ok(build_openai_generate_request(
        state,
        Vec::new(),
        Some(input_text),
        max_output_tokens,
        payload,
    ))
}

pub(crate) fn build_openai_mlx_lm_chat_request(
    state: &AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<OpenAiBuiltMlxLmChatRequest, (StatusCode, Json<ErrorResponse>)> {
    let max_output_tokens = openai_max_tokens(request.max_tokens);
    let messages = build_mlx_lm_chat_messages(&request.messages)?;
    let sampling = build_openai_sampling(
        request.temperature,
        request.top_p,
        request.top_k,
        request.min_p,
        request.repetition_penalty,
        request.repetition_context_size,
        request.seed,
    );
    let stop_sequences = openai_chat_stop_sequences(state.model_id.as_ref(), request.stop);

    Ok(OpenAiBuiltMlxLmChatRequest {
        chat_request: MlxLmChatGenerateRequest {
            model_id: state.model_id.to_string(),
            messages,
            max_output_tokens,
            sampling,
            stop_sequences,
            metadata: request.metadata,
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
    let messages = build_llama_cpp_chat_messages(&request.messages)?;
    let sampling = build_openai_sampling(
        request.temperature,
        request.top_p,
        request.top_k,
        request.min_p,
        request.repetition_penalty,
        request.repetition_context_size,
        request.seed,
    );
    let stop_sequences = openai_chat_stop_sequences(state.model_id.as_ref(), request.stop);

    Ok(OpenAiBuiltLlamaCppChatRequest {
        chat_request: LlamaCppChatGenerateRequest {
            model_id: state.model_id.to_string(),
            messages,
            max_output_tokens,
            sampling,
            stop_sequences,
            metadata: request.metadata,
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
) -> OpenAiBuiltRequest {
    OpenAiBuiltRequest {
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
    }
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

fn build_openai_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    repetition_context_size: Option<u32>,
    seed: Option<u64>,
) -> GenerateSampling {
    GenerateSampling {
        temperature: temperature.unwrap_or(0.0),
        top_p: top_p.unwrap_or(1.0),
        top_k: top_k.unwrap_or(0),
        min_p,
        repetition_penalty: repetition_penalty.unwrap_or(1.0),
        repetition_context_size,
        seed: seed.unwrap_or(0),
        deterministic: None,
    }
}

fn openai_max_tokens(max_tokens: Option<u32>) -> u32 {
    max_tokens.unwrap_or(DEFAULT_OPENAI_MAX_TOKENS)
}

pub(crate) fn render_openai_chat_prompt(
    model_id: &str,
    messages: &[OpenAiChatMessage],
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    let rendered_messages = render_openai_chat_message_pairs(messages)?;
    chat::render_prompt(model_id, &rendered_messages).map_err(chat_error_response)
}

fn build_mlx_lm_chat_messages(
    messages: &[OpenAiChatMessage],
) -> Result<Vec<MlxLmChatMessage>, (StatusCode, Json<ErrorResponse>)> {
    if messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "chat.completions requires at least one message".to_string(),
        ));
    }

    messages
        .iter()
        .map(|message| {
            let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
            let content = render_openai_chat_content(&message.content)?;
            Ok(MlxLmChatMessage::new(role, content))
        })
        .collect()
}

fn build_llama_cpp_chat_messages(
    messages: &[OpenAiChatMessage],
) -> Result<Vec<LlamaCppChatMessage>, (StatusCode, Json<ErrorResponse>)> {
    if messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "chat.completions requires at least one message".to_string(),
        ));
    }

    messages
        .iter()
        .map(|message| {
            let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
            let content = render_openai_chat_content(&message.content)?;
            Ok(LlamaCppChatMessage::new(role, content))
        })
        .collect()
}

pub(crate) fn chat_template_kwargs_for_model_id(model_id: &str) -> Option<serde_json::Value> {
    chat::template_kwargs_for_model_id(model_id)
}

pub(crate) fn openai_chat_stop_sequences(
    model_id: &str,
    stop: Option<OpenAiStopInput>,
) -> Vec<String> {
    chat::stop_sequences(
        model_id,
        stop.map(OpenAiStopInput::into_vec).unwrap_or_default(),
    )
}

fn render_openai_chat_message_pairs(
    messages: &[OpenAiChatMessage],
) -> Result<ChatMessagePairs, HttpErrorResponse> {
    if messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "chat.completions requires at least one message".to_string(),
        ));
    }
    messages
        .iter()
        .map(|message| {
            let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
            let content = render_openai_chat_content(&message.content)?;
            Ok((role.to_string(), content))
        })
        .collect()
}

fn chat_error_response(message: String) -> HttpErrorResponse {
    error_response(StatusCode::BAD_REQUEST, "invalid_request", message)
}

fn render_openai_chat_content(
    content: &OpenAiChatContent,
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    match content {
        OpenAiChatContent::Text(text) => Ok(text.clone()),
        OpenAiChatContent::Parts(parts) => {
            let mut rendered = String::new();
            for part in parts {
                if part.part_type != "text" {
                    return Err(error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request",
                        format!(
                            "unsupported chat content part type {}; AX preview currently accepts text-only chat messages",
                            part.part_type
                        ),
                    ));
                }
                let text = part.text.as_deref().ok_or_else(|| {
                    error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request",
                        "text chat content parts require a text field".to_string(),
                    )
                })?;
                rendered.push_str(text);
            }
            Ok(rendered)
        }
    }
}
