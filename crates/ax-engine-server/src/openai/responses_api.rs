use axum::Json;
use axum::body::to_bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::app_state::AppState;
use crate::errors::{ErrorResponse, error_response};
use crate::openai::chat::openai_chat_completions;
use crate::openai::schema::OpenAiChatCompletionHttpRequest;

const MAX_CHAT_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OpenAiResponsesRequest {
    #[serde(default)]
    model: Option<String>,
    input: Value,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    max_tool_calls: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_logprobs: Option<u32>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    text: Option<Value>,
    #[serde(default)]
    reasoning: Option<Value>,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    truncation: Option<String>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    #[serde(rename = "safety_identifier")]
    _safety_identifier: Option<String>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    prompt_cache_key: Option<String>,
    #[serde(default)]
    prompt_cache_retention: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    conversation: Option<Value>,
    #[serde(default)]
    background: Option<bool>,
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    include: Option<Vec<String>>,
    #[serde(default)]
    stream_options: Option<Value>,
}

pub(crate) async fn openai_responses(
    State(state): State<AppState>,
    Json(request): Json<OpenAiResponsesRequest>,
) -> Result<axum::response::Response, HttpErrorResponse> {
    validate_stateless_contract(&request)?;
    let chat_request = build_chat_request(&request)?;
    let chat_response = openai_chat_completions(State(state), Json(chat_request)).await?;
    let status = chat_response.status();
    if !status.is_success() {
        return Ok(chat_response);
    }
    let body = to_bytes(chat_response.into_body(), MAX_CHAT_RESPONSE_BYTES)
        .await
        .map_err(|error| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                format!("failed to read chat completion for Responses API: {error}"),
            )
        })?;
    let chat: Value = serde_json::from_slice(&body).map_err(|error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("failed to decode chat completion for Responses API: {error}"),
        )
    })?;
    Ok(Json(build_responses_output(&request, &chat)?).into_response())
}

fn validate_stateless_contract(request: &OpenAiResponsesRequest) -> Result<(), HttpErrorResponse> {
    if request.stream {
        return Err(unsupported(
            "stream is not supported by the stateless /v1/responses subset",
        ));
    }
    if request.store == Some(true) {
        return Err(unsupported(
            "store=true requires persisted Responses state, which AX Engine does not implement",
        ));
    }
    if request
        .previous_response_id
        .as_deref()
        .is_some_and(|id| !id.is_empty())
    {
        return Err(unsupported(
            "previous_response_id requires persisted Responses state, which AX Engine does not implement",
        ));
    }
    if request.conversation.as_ref().is_some_and(value_is_present) {
        return Err(unsupported(
            "conversation requires persisted Responses state, which AX Engine does not implement",
        ));
    }
    if request.background == Some(true) {
        return Err(unsupported(
            "background responses are not supported by AX Engine",
        ));
    }
    if request.max_tool_calls.is_some() {
        return Err(unsupported(
            "max_tool_calls is not supported by the stateless /v1/responses subset",
        ));
    }
    if request
        .service_tier
        .as_deref()
        .is_some_and(|tier| !matches!(tier, "auto" | "default"))
    {
        return Err(unsupported(
            "service_tier supports only auto/default on local AX Engine",
        ));
    }
    if request
        .prompt_cache_key
        .as_deref()
        .is_some_and(|key| !key.is_empty())
        || request
            .prompt_cache_retention
            .as_deref()
            .is_some_and(|retention| !retention.is_empty())
    {
        return Err(unsupported(
            "explicit prompt-cache routing is not supported by AX Engine",
        ));
    }
    if request
        .stream_options
        .as_ref()
        .is_some_and(value_is_present)
    {
        return Err(unsupported(
            "stream_options requires streaming Responses events, which AX Engine does not implement",
        ));
    }
    if request.prompt.as_ref().is_some_and(value_is_present) {
        return Err(unsupported(
            "hosted prompt references are not supported by AX Engine",
        ));
    }
    if request
        .include
        .as_ref()
        .is_some_and(|items| !items.is_empty())
    {
        return Err(unsupported(
            "include expansions are not supported by the stateless /v1/responses subset",
        ));
    }
    if request
        .truncation
        .as_deref()
        .is_some_and(|value| value != "disabled")
    {
        return Err(unsupported(
            "only truncation=disabled is supported by AX Engine",
        ));
    }
    Ok(())
}

fn build_chat_request(
    request: &OpenAiResponsesRequest,
) -> Result<OpenAiChatCompletionHttpRequest, HttpErrorResponse> {
    let messages = responses_messages(request.instructions.as_deref(), &request.input)?;
    let tools = request.tools.as_ref().map(responses_tools).transpose()?;
    let tool_choice = request
        .tool_choice
        .as_ref()
        .map(responses_tool_choice)
        .transpose()?;
    let response_format = request
        .text
        .as_ref()
        .map(responses_text_format)
        .transpose()?;
    let metadata = request.metadata.as_ref().map(Value::to_string).or_else(|| {
        request
            .user
            .as_ref()
            .map(|user| json!({"user": user}).to_string())
    });
    serde_json::from_value(json!({
        "model": request.model.clone(),
        "messages": messages,
        "max_completion_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_logprobs": request.top_logprobs,
        "stream": false,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": response_format,
        "reasoning": request.reasoning.clone(),
        "metadata": metadata,
    }))
    .map_err(|error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("failed to build internal chat request for Responses API: {error}"),
        )
    })
}

fn responses_messages(
    instructions: Option<&str>,
    input: &Value,
) -> Result<Vec<Value>, HttpErrorResponse> {
    let mut messages = Vec::new();
    if let Some(instructions) = instructions.filter(|value| !value.is_empty()) {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    match input {
        Value::String(text) => messages.push(json!({"role": "user", "content": text})),
        Value::Array(items) => {
            if items.is_empty() {
                return Err(invalid("input must not be empty"));
            }
            for (index, item) in items.iter().enumerate() {
                messages.push(response_input_item(item, index)?);
            }
        }
        _ => {
            return Err(invalid(
                "input must be a string or an array of Responses input items",
            ));
        }
    }
    if messages.is_empty() {
        return Err(invalid("input must not be empty"));
    }
    Ok(messages)
}

fn response_input_item(item: &Value, index: usize) -> Result<Value, HttpErrorResponse> {
    let object = item
        .as_object()
        .ok_or_else(|| invalid(format!("input[{index}] must be an object")))?;
    match object.get("type").and_then(Value::as_str) {
        None | Some("message") => response_message_item(object, index),
        Some("function_call") => response_function_call_item(object, index),
        Some("function_call_output") => response_function_output_item(object, index),
        Some(item_type) => Err(unsupported(format!(
            "input[{index}] type '{item_type}' is not supported by the stateless /v1/responses subset"
        ))),
    }
}

fn response_message_item(
    object: &Map<String, Value>,
    index: usize,
) -> Result<Value, HttpErrorResponse> {
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("input[{index}].role must be a string")))?;
    let content = response_item_text(object.get("content"), index)?;
    Ok(json!({"role": role, "content": content}))
}

fn response_item_text(content: Option<&Value>, index: usize) -> Result<String, HttpErrorResponse> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for (part_index, part) in parts.iter().enumerate() {
                if let Some(part_text) = part.as_str() {
                    text.push_str(part_text);
                    continue;
                }
                let part = part.as_object().ok_or_else(|| {
                    invalid(format!(
                        "input[{index}].content[{part_index}] must be a text object"
                    ))
                })?;
                let part_type = part.get("type").and_then(Value::as_str).unwrap_or("text");
                if !matches!(part_type, "input_text" | "output_text" | "text") {
                    return Err(unsupported(format!(
                        "input[{index}].content[{part_index}] type '{part_type}' is not supported by /v1/responses"
                    )));
                }
                let part_text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    invalid(format!(
                        "input[{index}].content[{part_index}].text must be a string"
                    ))
                })?;
                text.push_str(part_text);
            }
            Ok(text)
        }
        Some(Value::Null) | None => Ok(String::new()),
        Some(_) => Err(invalid(format!(
            "input[{index}].content must be a string or text-part array"
        ))),
    }
}

fn response_function_call_item(
    object: &Map<String, Value>,
    index: usize,
) -> Result<Value, HttpErrorResponse> {
    let name = required_string(object, "name", index)?;
    let call_id = object
        .get("call_id")
        .or_else(|| object.get("id"))
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("input[{index}].call_id must be a string")))?;
    let arguments = object
        .get("arguments")
        .map(value_as_text)
        .unwrap_or_else(|| "{}".to_string());
    Ok(json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments}
        }]
    }))
}

fn response_function_output_item(
    object: &Map<String, Value>,
    index: usize,
) -> Result<Value, HttpErrorResponse> {
    let call_id = required_string(object, "call_id", index)?;
    let output = object
        .get("output")
        .map(value_as_text)
        .ok_or_else(|| invalid(format!("input[{index}].output is required")))?;
    Ok(json!({
        "role": "tool",
        "tool_call_id": call_id,
        "content": output
    }))
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    index: usize,
) -> Result<&'a str, HttpErrorResponse> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("input[{index}].{field} must be a string")))
}

fn responses_tools(value: &Value) -> Result<Value, HttpErrorResponse> {
    let tools = value
        .as_array()
        .ok_or_else(|| invalid("tools must be an array"))?;
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let object = tool
                .as_object()
                .ok_or_else(|| invalid(format!("tools[{index}] must be an object")))?;
            if object.get("type").and_then(Value::as_str) != Some("function") {
                let tool_type = object
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                return Err(unsupported(format!(
                    "tools[{index}] type '{tool_type}' is not supported; only function tools are available"
                )));
            }
            if let Some(function) = object.get("function") {
                return Ok(json!({"type": "function", "function": function}));
            }
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid(format!("tools[{index}].name must be a string")))?;
            let mut function = Map::new();
            function.insert("name".to_string(), json!(name));
            for field in ["description", "parameters", "strict"] {
                if let Some(value) = object.get(field) {
                    function.insert(field.to_string(), value.clone());
                }
            }
            Ok(json!({"type": "function", "function": function}))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn responses_tool_choice(value: &Value) -> Result<Value, HttpErrorResponse> {
    let Some(object) = value.as_object() else {
        return Ok(value.clone());
    };
    if object.get("type").and_then(Value::as_str) != Some("function") {
        return Err(unsupported(
            "object tool_choice only supports type=function",
        ));
    }
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("tool_choice.name must be a string"))?;
    Ok(json!({"type": "function", "function": {"name": name}}))
}

fn responses_text_format(value: &Value) -> Result<Option<Value>, HttpErrorResponse> {
    let format = value.get("format").unwrap_or(value);
    let format_type = format.get("type").and_then(Value::as_str).unwrap_or("text");
    match format_type {
        "text" => Ok(None),
        "json_object" => Ok(Some(json!({"type": "json_object"}))),
        "json_schema" => Ok(Some(json!({
            "type": "json_schema",
            "json_schema": format
        }))),
        other => Err(unsupported(format!(
            "text.format.type '{other}' is not supported"
        ))),
    }
}

fn build_responses_output(
    request: &OpenAiResponsesRequest,
    chat: &Value,
) -> Result<Value, HttpErrorResponse> {
    let chat_id = chat
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| internal("chat completion response did not include id"))?;
    let created_at = chat
        .get("created")
        .and_then(Value::as_u64)
        .ok_or_else(|| internal("chat completion response did not include created"))?;
    let model = chat
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| internal("chat completion response did not include model"))?;
    let choice = chat
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| internal("chat completion response did not include a choice"))?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| internal("chat completion choice did not include a message"))?;
    let response_suffix = chat_id.strip_prefix("chatcmpl-").unwrap_or(chat_id);
    let mut output = Vec::new();
    if let Some(reasoning) = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        output.push(json!({
            "id": format!("rs_{response_suffix}"),
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning}]
        }));
    }
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let tool_calls = message.get("tool_calls").and_then(Value::as_array);
    if !content.is_empty() || tool_calls.is_none_or(Vec::is_empty) {
        output.push(json!({
            "id": format!("msg_{response_suffix}"),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content,
                "annotations": [],
                "logprobs": []
            }]
        }));
    }
    if let Some(tool_calls) = tool_calls {
        for (index, call) in tool_calls.iter().enumerate() {
            let call_id = call
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| internal("chat completion tool call did not include an id"))?;
            let function = call
                .get("function")
                .ok_or_else(|| internal("chat completion tool call did not include function"))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    internal("chat completion tool call did not include function.name")
                })?;
            let arguments = function
                .get("arguments")
                .map(value_as_text)
                .unwrap_or_else(|| "{}".to_string());
            output.push(json!({
                "id": format!("fc_{response_suffix}_{index}"),
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": arguments
            }));
        }
    }
    let finish_reason = choice.get("finish_reason").and_then(Value::as_str);
    let incomplete_reason = match finish_reason {
        Some("length") => Some("max_output_tokens"),
        Some("content_filter") => Some("content_filter"),
        Some(_) | None => None,
    };
    let usage = responses_usage(chat.get("usage"));
    Ok(json!({
        "id": format!("resp_{response_suffix}"),
        "object": "response",
        "created_at": created_at,
        "status": if incomplete_reason.is_some() { "incomplete" } else { "completed" },
        "error": null,
        "incomplete_details": incomplete_reason.map(|reason| json!({"reason": reason})),
        "instructions": request.instructions.clone(),
        "max_output_tokens": request.max_output_tokens,
        "model": model,
        "output": output,
        "parallel_tool_calls": request.parallel_tool_calls.unwrap_or(true),
        "previous_response_id": null,
        "reasoning": request.reasoning.clone().unwrap_or_else(|| json!({})),
        "store": false,
        "temperature": request.temperature,
        "text": request.text.clone().unwrap_or_else(|| json!({"format": {"type": "text"}})),
        "tool_choice": request.tool_choice.clone().unwrap_or_else(|| json!("auto")),
        "tools": request.tools.clone().unwrap_or_else(|| json!([])),
        "top_p": request.top_p,
        "truncation": request.truncation.as_deref().unwrap_or("disabled"),
        "usage": usage,
        "metadata": request.metadata.clone().unwrap_or_else(|| json!({})),
        "user": request.user.clone(),
        "service_tier": request.service_tier.as_deref().unwrap_or("default")
    }))
}

fn responses_usage(usage: Option<&Value>) -> Value {
    let Some(usage) = usage.filter(|value| !value.is_null()) else {
        return Value::Null;
    };
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|value| value.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt_tokens.saturating_add(completion_tokens)
    })
}

fn value_as_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn value_is_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
        Value::Number(_) => true,
    }
}

fn invalid(message: impl Into<String>) -> HttpErrorResponse {
    error_response(StatusCode::BAD_REQUEST, "invalid_request", message.into())
}

fn unsupported(message: impl Into<String>) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        message.into(),
    )
}

fn internal(message: impl Into<String>) -> HttpErrorResponse {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        message.into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_history_and_tools_map_to_chat_contract() {
        let messages = responses_messages(
            None,
            &json!([
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Look this up"}]
                },
                {
                    "type": "function_call",
                    "call_id": "call_previous",
                    "name": "lookup",
                    "arguments": "{\"query\":\"old\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_previous",
                    "output": "old result"
                }
            ]),
        )
        .expect("function history should map");
        assert_eq!(
            Value::Array(messages),
            json!([
                {"role": "user", "content": "Look this up"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_previous",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"query\":\"old\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_previous", "content": "old result"}
            ])
        );

        let tools = responses_tools(&json!([{
            "type": "function",
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {"type": "object"}
        }]))
        .expect("function tools should map");
        assert_eq!(
            tools,
            json!([{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {"type": "object"}
                }
            }])
        );
        assert_eq!(
            responses_tool_choice(&json!({"type": "function", "name": "lookup"}))
                .expect("function tool choice should map"),
            json!({"type": "function", "function": {"name": "lookup"}})
        );
    }

    #[test]
    fn chat_tool_output_maps_to_responses_function_call() {
        let request: OpenAiResponsesRequest = serde_json::from_value(json!({
            "input": "Look this up",
            "tools": [{"type": "function", "name": "lookup"}]
        }))
        .expect("Responses request should deserialize");
        let response = build_responses_output(
            &request,
            &json!({
                "id": "chatcmpl-42",
                "created": 123,
                "model": "qwen3",
                "choices": [{
                    "message": {
                        "content": null,
                        "tool_calls": [{
                            "id": "call_next",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{\"query\":\"new\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3}
            }),
        )
        .expect("chat tool output should map");

        assert_eq!(response["output"].as_array().map(Vec::len), Some(1));
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["call_id"], "call_next");
        assert_eq!(response["output"][0]["name"], "lookup");
        assert_eq!(response["output"][0]["arguments"], "{\"query\":\"new\"}");
        assert_eq!(response["usage"]["total_tokens"], 14);
    }

    #[test]
    fn content_filter_finish_maps_to_incomplete_response() {
        let request: OpenAiResponsesRequest =
            serde_json::from_value(json!({"input": "unsafe request"}))
                .expect("Responses request should deserialize");
        let response = build_responses_output(
            &request,
            &json!({
                "id": "chatcmpl-filtered",
                "created": 123,
                "model": "qwen3",
                "choices": [{
                    "message": {"content": ""},
                    "finish_reason": "content_filter"
                }],
                "usage": null
            }),
        )
        .expect("filtered chat response should map");

        assert_eq!(response["status"], "incomplete");
        assert_eq!(response["incomplete_details"]["reason"], "content_filter");
        assert_eq!(response["usage"], Value::Null);
    }
}
