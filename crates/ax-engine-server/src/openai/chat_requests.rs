use ax_engine_sdk::{LlamaCppChatMessage, MlxLmChatMessage};
use axum::Json;
use axum::http::StatusCode;

use crate::chat;
use crate::errors::{ErrorResponse, error_response};
use crate::openai::schema::{OpenAiChatContent, OpenAiChatMessage, OpenAiStopInput};

type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);
type ChatMessagePairs = Vec<(String, String)>;

pub(crate) fn render_openai_chat_prompt(
    model_id: &str,
    messages: &[OpenAiChatMessage],
) -> Result<String, HttpErrorResponse> {
    let rendered_messages = render_openai_chat_message_pairs(messages)?;
    chat::render_prompt(model_id, &rendered_messages).map_err(chat_error_response)
}

pub(crate) fn build_mlx_lm_chat_messages(
    messages: &[OpenAiChatMessage],
) -> Result<Vec<MlxLmChatMessage>, HttpErrorResponse> {
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
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

pub(crate) fn build_llama_cpp_chat_messages(
    messages: &[OpenAiChatMessage],
) -> Result<Vec<LlamaCppChatMessage>, HttpErrorResponse> {
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
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
        return Err(empty_chat_messages_error());
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

fn empty_chat_messages_error() -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        "chat.completions requires at least one message".to_string(),
    )
}

fn chat_error_response(message: String) -> HttpErrorResponse {
    error_response(StatusCode::BAD_REQUEST, "invalid_request", message)
}

fn render_openai_chat_content(content: &OpenAiChatContent) -> Result<String, HttpErrorResponse> {
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
