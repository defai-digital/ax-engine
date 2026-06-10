use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::{GenerateFinishReason, GenerateResponse};
use axum::Json;
use axum::response::IntoResponse;
use serde_json::Value;

use super::requests::OpenAiResponseOptions;
use super::schema::{
    OpenAiChatCompletionChoice, OpenAiChatCompletionResponse, OpenAiChatLogprobs,
    OpenAiChatMessageResponse, OpenAiChatTokenLogprob, OpenAiCompletionChoice,
    OpenAiCompletionLogprobs, OpenAiCompletionResponse, OpenAiFunctionCall, OpenAiStreamKind,
    OpenAiToolCall, OpenAiUsage,
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
        options: OpenAiResponseOptions,
        native_reasoning: Option<String>,
    ) -> axum::response::Response {
        let id = self.response_id(request_id);
        match self {
            Self::Completion => {
                Json(openai_completion_response(response, id, options)).into_response()
            }
            Self::ChatCompletion => Json(openai_chat_completion_response(
                response,
                id,
                options,
                native_reasoning,
            ))
            .into_response(),
        }
    }
}

pub(crate) fn openai_completion_response(
    response: &GenerateResponse,
    id: String,
    options: OpenAiResponseOptions,
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
            logprobs: openai_completion_logprobs(response, options),
            finish_reason: openai_finish_reason(response.finish_reason),
        }],
        usage: openai_usage(response),
    }
}

pub(crate) fn openai_chat_completion_response(
    response: &GenerateResponse,
    id: String,
    options: OpenAiResponseOptions,
    native_reasoning: Option<String>,
) -> OpenAiChatCompletionResponse {
    let raw_content = response.output_text.clone().unwrap_or_default();
    // The native MLX decode extracts Gemma 4 thinking channels at the token
    // level, so their framing never survives into `output_text`; when the
    // decode supplied reasoning, use it instead of re-scanning text markers.
    let (mut content, reasoning_content) = match native_reasoning {
        Some(reasoning) if options.include_reasoning => (raw_content, Some(reasoning)),
        _ => split_reasoning_content(&raw_content, options.include_reasoning),
    };
    let tool_calls = if options.parse_tool_calls {
        extract_tool_calls(&mut content)
    } else {
        None
    };
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
                content,
                reasoning_content,
                tool_calls,
            },
            logprobs: openai_chat_logprobs(response, options),
            finish_reason: openai_finish_reason(response.finish_reason),
        }],
        usage: openai_usage(response),
    }
}

fn openai_completion_logprobs(
    response: &GenerateResponse,
    options: OpenAiResponseOptions,
) -> Option<OpenAiCompletionLogprobs> {
    if !options.include_logprobs || response.output_tokens.is_empty() {
        return None;
    }
    let token_logprobs = sampled_token_logprobs(response)?;
    Some(OpenAiCompletionLogprobs {
        tokens: response
            .output_tokens
            .iter()
            .map(|token| token.to_string())
            .collect(),
        text_offset: (0..response.output_tokens.len() as u32).collect(),
        top_logprobs: vec![None; response.output_tokens.len()],
        token_logprobs,
    })
}

fn openai_chat_logprobs(
    response: &GenerateResponse,
    options: OpenAiResponseOptions,
) -> Option<OpenAiChatLogprobs> {
    if !options.include_logprobs || response.output_tokens.is_empty() {
        return None;
    }
    let token_logprobs = sampled_token_logprobs(response)?;
    let content = response
        .output_tokens
        .iter()
        .zip(token_logprobs)
        .map(|(token, logprob)| OpenAiChatTokenLogprob {
            token: token.to_string(),
            logprob: logprob.unwrap_or_default(),
            bytes: None,
            top_logprobs: Vec::new(),
        })
        .collect::<Vec<_>>();
    (!content.is_empty()).then_some(OpenAiChatLogprobs { content })
}

/// Sampled logprobs are all-or-nothing: a partially populated vector would
/// force misaligned (or fabricated) entries in the OpenAI logprob arrays, so
/// any gap omits the whole block.
fn sampled_token_logprobs(response: &GenerateResponse) -> Option<Vec<Option<f32>>> {
    if response.output_token_logprobs.len() != response.output_tokens.len() {
        return None;
    }
    (!response.output_token_logprobs.is_empty()
        && response.output_token_logprobs.iter().all(Option::is_some))
    .then(|| response.output_token_logprobs.clone())
}

fn split_reasoning_content(text: &str, include_reasoning: bool) -> (String, Option<String>) {
    if !include_reasoning {
        return (text.to_string(), None);
    }
    if let Some((content, reasoning)) = split_tagged_reasoning(text, "<think>", "</think>") {
        return (content, Some(reasoning));
    }
    if let Some((content, reasoning)) =
        split_tagged_reasoning(text, "<|channel>thought\n", "<channel|>")
    {
        return (content, Some(reasoning));
    }
    (text.to_string(), None)
}

fn split_tagged_reasoning(text: &str, start: &str, end: &str) -> Option<(String, String)> {
    let start_index = text.find(start)?;
    let reasoning_start = start_index + start.len();
    let relative_end = text[reasoning_start..].find(end)?;
    let end_index = reasoning_start + relative_end;
    let reasoning = text[reasoning_start..end_index].trim().to_string();
    let content = format!("{}{}", &text[..start_index], &text[end_index + end.len()..])
        .trim()
        .to_string();
    Some((content, reasoning))
}

// Only explicit `<tool_call>` markers are parsed. Treating any bare JSON
// object in the content as a tool call misfires when tools are offered but
// the model legitimately answers with JSON; bare-JSON parser families can be
// added per model family once their templates are rendered server-side.
fn extract_tool_calls(content: &mut String) -> Option<Vec<OpenAiToolCall>> {
    let (tool_json, remaining) = extract_tool_call_payload(content)?;
    let function = parse_tool_call_function(&tool_json)?;
    *content = remaining.trim().to_string();
    Some(vec![OpenAiToolCall {
        id: "call_0".to_string(),
        tool_type: "function",
        function,
    }])
}

fn extract_tool_call_payload(content: &str) -> Option<(Value, String)> {
    let start = content.find("<tool_call>")?;
    let body_start = start + "<tool_call>".len();
    let relative_end = content[body_start..].find("</tool_call>")?;
    let end = body_start + relative_end;
    let value = serde_json::from_str(content[body_start..end].trim()).ok()?;
    let remaining = format!(
        "{}{}",
        &content[..start],
        &content[end + "</tool_call>".len()..]
    );
    Some((value, remaining))
}

fn parse_tool_call_function(value: &Value) -> Option<OpenAiFunctionCall> {
    let object = value.as_object()?;
    if let Some(function) = object.get("function").and_then(Value::as_object) {
        let name = function.get("name")?.as_str()?.to_string();
        let arguments = serialize_tool_arguments(function.get("arguments"));
        return Some(OpenAiFunctionCall { name, arguments });
    }
    let name = object.get("name")?.as_str()?.to_string();
    let arguments = serialize_tool_arguments(object.get("arguments"));
    Some(OpenAiFunctionCall { name, arguments })
}

fn serialize_tool_arguments(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(value) => serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
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
        Some(GenerateFinishReason::Cancelled) => Some("cancel"),
        Some(GenerateFinishReason::MaxOutputTokens) => Some("length"),
        Some(GenerateFinishReason::ContentFilter) => Some("content_filter"),
        Some(GenerateFinishReason::Error) | None => None,
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
