use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::{GenerateFinishReason, GenerateResponse};
use axum::Json;
use axum::response::IntoResponse;

use super::schema::{
    OpenAiChatCompletionChoice, OpenAiChatCompletionResponse, OpenAiChatMessageResponse,
    OpenAiCompletionChoice, OpenAiCompletionResponse, OpenAiStreamKind, OpenAiUsage,
};

impl OpenAiStreamKind {
    pub(crate) fn response_id(self, request_id: u64) -> String {
        match self {
            Self::Completion => format!("cmpl-{request_id}"),
            Self::ChatCompletion => format!("chatcmpl-{request_id}"),
        }
    }

    pub(crate) fn stream_chunk_object(self) -> &'static str {
        match self {
            Self::Completion => "text_completion.chunk",
            Self::ChatCompletion => "chat.completion.chunk",
        }
    }

    pub(crate) fn build_non_stream_response(
        self,
        response: &GenerateResponse,
        request_id: u64,
    ) -> axum::response::Response {
        let id = self.response_id(request_id);
        match self {
            Self::Completion => Json(openai_completion_response(response, id)).into_response(),
            Self::ChatCompletion => {
                Json(openai_chat_completion_response(response, id)).into_response()
            }
        }
    }
}

pub(crate) fn openai_completion_response(
    response: &GenerateResponse,
    id: String,
) -> OpenAiCompletionResponse {
    OpenAiCompletionResponse {
        id,
        object: "text_completion",
        created: unix_timestamp_secs(),
        model: response.model_id.clone(),
        system_fingerprint: None,
        choices: vec![OpenAiCompletionChoice {
            index: 0,
            text: response.output_text.clone().unwrap_or_default(),
            finish_reason: openai_finish_reason(response.finish_reason),
        }],
        usage: openai_usage(response),
    }
}

pub(crate) fn openai_chat_completion_response(
    response: &GenerateResponse,
    id: String,
) -> OpenAiChatCompletionResponse {
    OpenAiChatCompletionResponse {
        id,
        object: "chat.completion",
        created: unix_timestamp_secs(),
        model: response.model_id.clone(),
        system_fingerprint: None,
        choices: vec![OpenAiChatCompletionChoice {
            index: 0,
            message: OpenAiChatMessageResponse {
                role: "assistant",
                content: response.output_text.clone().unwrap_or_default(),
            },
            finish_reason: openai_finish_reason(response.finish_reason),
        }],
        usage: openai_usage(response),
    }
}

pub(crate) fn openai_usage(response: &GenerateResponse) -> Option<OpenAiUsage> {
    let (prompt_tokens, completion_tokens) = response.known_usage()?;
    Some(OpenAiUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    })
}

pub(crate) fn openai_finish_reason(
    finish_reason: Option<GenerateFinishReason>,
) -> Option<&'static str> {
    match finish_reason {
        Some(GenerateFinishReason::Stop) => Some("stop"),
        Some(GenerateFinishReason::MaxOutputTokens) => Some("length"),
        Some(GenerateFinishReason::ContentFilter) => Some("content_filter"),
        Some(GenerateFinishReason::Cancelled | GenerateFinishReason::Error) | None => None,
    }
}

pub(crate) fn finish_reason_from_llama_cpp_chat(
    value: Option<&str>,
) -> Option<GenerateFinishReason> {
    match value {
        Some("stop") => Some(GenerateFinishReason::Stop),
        Some("length") => Some(GenerateFinishReason::MaxOutputTokens),
        Some("content_filter") => Some(GenerateFinishReason::ContentFilter),
        Some(_) | None => None,
    }
}

pub(crate) fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
