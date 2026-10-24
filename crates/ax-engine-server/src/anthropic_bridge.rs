//! Anthropic Messages features carried over the OpenAI chat path: streaming,
//! tool use, extended thinking and image blocks.
//!
//! A request that needs any of them is rewritten as an OpenAI chat
//! completion request, run through the same handler `/v1/chat/completions`
//! uses (so tool-call parsing, reasoning split, media and admission behave
//! identically), and its response is translated back: a JSON body into one
//! Anthropic message, an SSE stream into Anthropic stream events. Plain text
//! requests keep the direct path in `anthropic.rs`.

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::app_state::AppState;
use crate::errors::{ErrorResponse, error_response};
use crate::openai::chat::openai_chat_completions;
use crate::openai::schema::OpenAiChatCompletionHttpRequest;

type HttpError = (StatusCode, Json<ErrorResponse>);

/// System blocks that start with this marker carry per-request billing
/// metadata with random values; dropping them keeps the rendered prompt, and
/// therefore the prefix cache, stable across turns.
pub(crate) const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

const MAX_CHAT_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

fn invalid(message: impl Into<String>) -> HttpError {
    error_response(StatusCode::BAD_REQUEST, "invalid_request", message.into())
}

/// Whether the request needs the OpenAI bridge rather than the plain text path.
pub(crate) fn needs_bridge(body: &Value) -> bool {
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let tools = body
        .get("tools")
        .is_some_and(|tools| tools.as_array().is_none_or(|tools| !tools.is_empty()));
    let thinking = body
        .get("thinking")
        .is_some_and(|thinking| !thinking.is_null());
    let rich_blocks = body
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|blocks| {
                        blocks
                            .iter()
                            .any(|block| block.get("type").and_then(Value::as_str) != Some("text"))
                    })
            })
        });
    streaming || tools || thinking || rich_blocks
}

/// Text of a `system` field (string or text blocks), billing blocks dropped.
pub(crate) fn system_text(system: &Value) -> Result<String, HttpError> {
    match system {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(if text.starts_with(BILLING_HEADER_PREFIX) {
            String::new()
        } else {
            text.clone()
        }),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                if block.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(invalid(
                        "Anthropic system blocks must be text blocks".to_string(),
                    ));
                }
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("Anthropic system text block requires text"))?;
                if !text.starts_with(BILLING_HEADER_PREFIX) {
                    parts.push(text);
                }
            }
            Ok(parts.join("\n"))
        }
        _ => Err(invalid("Anthropic system must be a string or text blocks")),
    }
}

fn text_of_tool_result(content: Option<&Value>) -> Result<String, HttpError> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(blocks)) => {
            let mut text = String::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        text.push_str(block.get("text").and_then(Value::as_str).unwrap_or(""))
                    }
                    other => {
                        return Err(invalid(format!(
                            "unsupported block type {other:?} inside tool_result; only text is supported"
                        )));
                    }
                }
            }
            Ok(text)
        }
        Some(_) => Err(invalid(
            "tool_result content must be a string or text blocks",
        )),
    }
}

fn image_part(block: &Value) -> Result<Value, HttpError> {
    let source = block
        .get("source")
        .ok_or_else(|| invalid("Anthropic image block requires source"))?;
    let url = match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("base64 image source requires media_type"))?;
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("base64 image source requires data"))?;
            format!("data:{media_type};base64,{data}")
        }
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("url image source requires url"))?
            .to_string(),
        other => {
            return Err(invalid(format!(
                "unsupported Anthropic image source type {other:?}"
            )));
        }
    };
    Ok(json!({"type": "image_url", "image_url": {"url": url}}))
}

/// Translate one Anthropic message into OpenAI chat messages. A user turn
/// with tool results becomes one `tool` message per result, then the user's
/// own content.
fn translate_message(message: &Value, out: &mut Vec<Value>) -> Result<(), HttpError> {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("Anthropic message requires role"))?;
    if role != "user" && role != "assistant" {
        return Err(invalid(format!(
            "unsupported Anthropic message role '{role}'; only user and assistant are supported"
        )));
    }
    let content = message
        .get("content")
        .ok_or_else(|| invalid("Anthropic message requires content"))?;
    let blocks = match content {
        Value::String(text) => {
            out.push(json!({"role": role, "content": text}));
            return Ok(());
        }
        Value::Array(blocks) => blocks,
        _ => {
            return Err(invalid(
                "Anthropic message content must be a string or blocks",
            ));
        }
    };
    let mut parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning = String::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("Anthropic text block requires text"))?;
                parts.push(json!({"type": "text", "text": text}));
            }
            Some("image") if role == "user" => parts.push(image_part(block)?),
            Some("tool_result") if role == "user" => {
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("tool_result requires tool_use_id"))?;
                let mut text = text_of_tool_result(block.get("content"))?;
                if block.get("is_error").and_then(Value::as_bool) == Some(true) {
                    text = format!("Error: {text}");
                }
                out.push(json!({"role": "tool", "tool_call_id": id, "content": text}));
            }
            Some("tool_use") if role == "assistant" => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("tool_use requires id"))?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("tool_use requires name"))?;
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": input.to_string()},
                }));
            }
            Some("thinking") if role == "assistant" => {
                reasoning.push_str(block.get("thinking").and_then(Value::as_str).unwrap_or(""));
            }
            // Redacted thinking has no plaintext to replay.
            Some("redacted_thinking") if role == "assistant" => {}
            other => {
                return Err(invalid(format!(
                    "unsupported Anthropic content block type {other:?} in a {role} message"
                )));
            }
        }
    }
    if parts.is_empty() && tool_calls.is_empty() && reasoning.is_empty() {
        return Ok(());
    }
    let all_text = parts
        .iter()
        .all(|part| part.get("type").and_then(Value::as_str) == Some("text"));
    let content = if all_text {
        Value::String(
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<String>(),
        )
    } else {
        Value::Array(parts)
    };
    let mut message = Map::new();
    message.insert("role".into(), Value::String(role.into()));
    message.insert("content".into(), content);
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !reasoning.is_empty() {
        message.insert("reasoning_content".into(), Value::String(reasoning));
    }
    out.push(Value::Object(message));
    Ok(())
}

fn translate_tool_choice(choice: &Value) -> Result<Value, HttpError> {
    match choice.get("type").and_then(Value::as_str) {
        Some("auto") => Ok(json!("auto")),
        Some("any") => Ok(json!("required")),
        Some("none") => Ok(json!("none")),
        Some("tool") => {
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("tool_choice type tool requires name"))?;
            Ok(json!({"type": "function", "function": {"name": name}}))
        }
        other => Err(invalid(format!(
            "unsupported Anthropic tool_choice {other:?}"
        ))),
    }
}

/// Rewrite an Anthropic Messages body as an OpenAI chat completion body.
pub(crate) fn to_openai_chat_body(body: &Value) -> Result<Value, HttpError> {
    let max_tokens = body
        .get("max_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("Anthropic Messages requires max_tokens for this endpoint"))?;
    let mut messages = Vec::new();
    let system = system_text(body.get("system").unwrap_or(&Value::Null))?;
    if !system.trim().is_empty() {
        messages.push(json!({"role": "system", "content": system}));
    }
    for message in body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Anthropic Messages requires messages"))?
    {
        translate_message(message, &mut messages)?;
    }
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let mut out = Map::new();
    out.insert("messages".into(), Value::Array(messages));
    out.insert("max_tokens".into(), json!(max_tokens));
    out.insert("stream".into(), json!(streaming));
    if streaming {
        out.insert("stream_options".into(), json!({"include_usage": true}));
    }
    for key in ["model", "temperature", "top_p", "top_k"] {
        if let Some(value) = body.get(key).filter(|value| !value.is_null()) {
            out.insert(key.into(), value.clone());
        }
    }
    if let Some(stops) = body
        .get("stop_sequences")
        .and_then(Value::as_array)
        .filter(|stops| !stops.is_empty())
    {
        out.insert("stop".into(), Value::Array(stops.clone()));
    }
    if let Some(tools) = body
        .get("tools")
        .and_then(Value::as_array)
        .filter(|tools| !tools.is_empty())
    {
        let mut converted = Vec::with_capacity(tools.len());
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("Anthropic tool requires name"))?;
            let mut function = Map::new();
            function.insert("name".into(), json!(name));
            if let Some(description) = tool.get("description") {
                function.insert("description".into(), description.clone());
            }
            function.insert(
                "parameters".into(),
                tool.get("input_schema")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object"})),
            );
            converted.push(json!({"type": "function", "function": function}));
        }
        out.insert("tools".into(), Value::Array(converted));
        if let Some(choice) = body.get("tool_choice").filter(|value| !value.is_null()) {
            out.insert("tool_choice".into(), translate_tool_choice(choice)?);
        }
    } else if body
        .get("tools")
        .is_some_and(|tools| !tools.is_null() && !tools.is_array())
    {
        return Err(invalid("Anthropic tools must be an array"));
    }
    match body
        .get("thinking")
        .filter(|value| !value.is_null())
        .map(|thinking| thinking.get("type").and_then(Value::as_str))
    {
        None => {}
        Some(Some("disabled")) => {
            out.insert(
                "chat_template_kwargs".into(),
                json!({"enable_thinking": false}),
            );
        }
        Some(Some("enabled")) => {
            out.insert(
                "chat_template_kwargs".into(),
                json!({"enable_thinking": true}),
            );
            if let Some(budget) = body
                .get("thinking")
                .and_then(|thinking| thinking.get("budget_tokens"))
                .and_then(Value::as_u64)
            {
                out.insert("ax_max_think_tokens".into(), json!(budget));
            }
        }
        Some(other) => {
            return Err(invalid(format!(
                "unsupported Anthropic thinking type {other:?}"
            )));
        }
    }
    Ok(Value::Object(out))
}

fn stop_reason(finish_reason: Option<&str>) -> &'static str {
    match finish_reason {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        Some("content_filter") => "refusal",
        _ => "end_turn",
    }
}

fn parse_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
}

/// One OpenAI chat completion JSON into one Anthropic message.
pub(crate) fn from_openai_chat_response(id: &str, chat: &Value) -> Value {
    let choice = chat.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    let message = choice.get("message").cloned().unwrap_or(Value::Null);
    let mut content = Vec::new();
    if let Some(reasoning) = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        content.push(json!({"type": "thinking", "thinking": reasoning, "signature": ""}));
    }
    if let Some(text) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        content.push(json!({"type": "text", "text": text}));
    }
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        content.push(json!({
            "type": "tool_use",
            "id": call.get("id").cloned().unwrap_or(json!("")),
            "name": call.pointer("/function/name").cloned().unwrap_or(json!("")),
            "input": parse_arguments(
                call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("{}")
            ),
        }));
    }
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": chat.get("model").cloned().unwrap_or(json!("")),
        "stop_reason": stop_reason(choice.get("finish_reason").and_then(Value::as_str)),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": chat.pointer("/usage/prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            "output_tokens": chat.pointer("/usage/completion_tokens").and_then(Value::as_u64).unwrap_or(0),
        },
    })
}

/// Incremental OpenAI chunk -> Anthropic stream event translation.
#[derive(Default)]
pub(crate) struct StreamTranslator {
    started: bool,
    /// Index of the open content block and its kind (`thinking`, `text`, or
    /// `tool_use` with the OpenAI tool-call index it mirrors).
    open: Option<(usize, OpenBlock)>,
    next_index: usize,
    finish_reason: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Clone, Copy, PartialEq)]
enum OpenBlock {
    Thinking,
    Text,
    Tool(u64),
}

fn event(name: &str, data: &Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

impl StreamTranslator {
    pub(crate) fn start(&mut self, id: &str, model: &str) -> String {
        self.started = true;
        event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": id, "type": "message", "role": "assistant", "content": [],
                    "model": model, "stop_reason": Value::Null, "stop_sequence": Value::Null,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }),
        )
    }

    fn close_open(&mut self, out: &mut String) {
        if let Some((index, _)) = self.open.take() {
            out.push_str(&event(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": index}),
            ));
        }
    }

    fn open_block(&mut self, kind: OpenBlock, block: Value, out: &mut String) -> usize {
        self.close_open(out);
        let index = self.next_index;
        self.next_index += 1;
        out.push_str(&event(
            "content_block_start",
            &json!({"type": "content_block_start", "index": index, "content_block": block}),
        ));
        self.open = Some((index, kind));
        index
    }

    fn ensure_open(
        &mut self,
        kind: OpenBlock,
        block: impl FnOnce() -> Value,
        out: &mut String,
    ) -> usize {
        match self.open {
            Some((index, open)) if open == kind => index,
            _ => self.open_block(kind, block(), out),
        }
    }

    /// Translate one OpenAI chunk; returns the Anthropic events it produced.
    pub(crate) fn chunk(&mut self, chunk: &Value) -> String {
        let mut out = String::new();
        if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
            self.input_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            self.output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
        }
        let Some(choice) = chunk.pointer("/choices/0") else {
            return out;
        };
        let delta = choice.get("delta").cloned().unwrap_or(Value::Null);
        if let Some(text) = delta
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            let index = self.ensure_open(
                OpenBlock::Thinking,
                || json!({"type": "thinking", "thinking": ""}),
                &mut out,
            );
            out.push_str(&event(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": index,
                        "delta": {"type": "thinking_delta", "thinking": text}}),
            ));
        }
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            let index = self.ensure_open(
                OpenBlock::Text,
                || json!({"type": "text", "text": ""}),
                &mut out,
            );
            out.push_str(&event(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": text}}),
            ));
        }
        for call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let tool_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let index = match self.open {
                Some((index, OpenBlock::Tool(open))) if open == tool_index => index,
                _ => {
                    let block = json!({
                        "type": "tool_use",
                        "id": call.get("id").cloned().unwrap_or(json!("")),
                        "name": call.pointer("/function/name").cloned().unwrap_or(json!("")),
                        "input": {},
                    });
                    self.open_block(OpenBlock::Tool(tool_index), block, &mut out)
                }
            };
            if let Some(arguments) = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .filter(|arguments| !arguments.is_empty())
            {
                out.push_str(&event(
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": index,
                            "delta": {"type": "input_json_delta", "partial_json": arguments}}),
                ));
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
        out
    }

    /// Closing events after the upstream stream ended.
    pub(crate) fn finish(&mut self) -> String {
        let mut out = String::new();
        self.close_open(&mut out);
        out.push_str(&event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason(self.finish_reason.as_deref()),
                          "stop_sequence": Value::Null},
                "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
            }),
        ));
        out.push_str(&event("message_stop", &json!({"type": "message_stop"})));
        out
    }
}

fn anthropic_stream_error(message: &str) -> String {
    event(
        "error",
        &json!({"type": "error", "error": {"type": "api_error", "message": message}}),
    )
}

/// Run an Anthropic request through the OpenAI chat handler and translate
/// the result.
pub(crate) async fn run_bridged(
    state: AppState,
    body: Value,
    request_id: u64,
) -> Result<Response, HttpError> {
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let chat_body = to_openai_chat_body(&body)?;
    let chat_request: OpenAiChatCompletionHttpRequest = serde_json::from_value(chat_body)
        .map_err(|error| invalid(format!("invalid Anthropic request: {error}")))?;
    let id = format!("msg-{request_id}");
    let response = openai_chat_completions(State(state), Ok(Json(chat_request))).await?;
    if !streaming {
        let bytes = axum::body::to_bytes(response.into_body(), MAX_CHAT_RESPONSE_BYTES)
            .await
            .map_err(|error| {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    format!("failed to read chat response: {error}"),
                )
            })?;
        let chat: Value = serde_json::from_slice(&bytes).map_err(|error| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                format!("chat response is not JSON: {error}"),
            )
        })?;
        return Ok(Json(from_openai_chat_response(&id, &chat)).into_response());
    }

    Ok(bridge_stream(response.into_body(), id, model))
}

fn bridge_stream(body: Body, id: String, model: String) -> Response {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);
    let mut upstream = body.into_data_stream();
    tokio::spawn(async move {
        let mut translator = StreamTranslator::default();
        if tx.send(Ok(translator.start(&id, &model))).await.is_err() {
            return;
        }
        let mut buffer = String::new();
        loop {
            let frame = tokio::select! {
                biased;
                _ = tx.closed() => return,
                frame = upstream.next() => frame,
            };
            let Some(frame) = frame else {
                break;
            };
            let bytes = match frame {
                Ok(bytes) => bytes,
                Err(error) => {
                    let _ = tx
                        .send(Ok(anthropic_stream_error(&error.to_string())))
                        .await;
                    return;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(end) = buffer.find("\n\n") {
                let raw: String = buffer.drain(..end + 2).collect();
                for line in raw.lines() {
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data == "[DONE]" || data.is_empty() {
                        continue;
                    }
                    let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                        continue;
                    };
                    if let Some(error) = chunk.get("error") {
                        let message = error
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("generation failed");
                        let _ = tx.send(Ok(anthropic_stream_error(message))).await;
                        return;
                    }
                    let events = translator.chunk(&chunk);
                    if !events.is_empty() && tx.send(Ok(events)).await.is_err() {
                        return;
                    }
                }
            }
        }
        let _ = tx.send(Ok(translator.finish())).await;
    });
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(ReceiverStream::new(rx)),
    )
        .into_response()
}

/// `/v1/messages/count_tokens`: prompt tokens of the rendered request on the
/// native path (the same rendering generation uses).
pub(crate) async fn anthropic_count_tokens(
    State(state): State<AppState>,
    payload: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<Value>, HttpError> {
    let Json(mut body) = payload
        .map_err(|rejection| invalid(format!("invalid request body: {}", rejection.body_text())))?;
    // count_tokens takes no max_tokens; any positive value renders the same prompt.
    if let Some(object) = body.as_object_mut() {
        object.insert("max_tokens".into(), json!(1));
        object.insert("stream".into(), json!(false));
    }
    let chat_body = to_openai_chat_body(&body)?;
    let chat_request: OpenAiChatCompletionHttpRequest = serde_json::from_value(chat_body)
        .map_err(|error| invalid(format!("invalid Anthropic request: {error}")))?;
    let live =
        crate::openai::validation::select_openai_model(&state, chat_request.model.as_deref())?;
    if crate::backends::mlx_lm::is_selected(&live)
        || crate::backends::llama_cpp::supports_server_chat(&live)
    {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "count_tokens needs the native MLX backend, which renders and tokenizes the prompt"
                .to_string(),
        ));
    }
    let built = crate::openai::requests::build_openai_chat_request_offloading_media(
        &live,
        &state.media,
        chat_request,
    )
    .await?;
    Ok(Json(
        json!({"input_tokens": built.generate_request.input_tokens.len()}),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn disconnect_drops_an_idle_upstream_stream() {
        let (upstream_tx, upstream_rx) = mpsc::channel::<Result<String, std::io::Error>>(1);
        let response = bridge_stream(
            Body::from_stream(ReceiverStream::new(upstream_rx)),
            "msg-cancel".to_string(),
            "m".to_string(),
        );
        let mut downstream = response.into_body().into_data_stream();
        let start = downstream.next().await.unwrap().unwrap();
        assert!(
            std::str::from_utf8(&start)
                .unwrap()
                .contains("message_start")
        );
        drop(downstream);
        tokio::time::timeout(std::time::Duration::from_secs(1), upstream_tx.closed())
            .await
            .expect("disconnect must release upstream even while no model output is arriving");
    }

    #[test]
    fn request_translation_covers_tools_results_thinking_and_billing() {
        let body = json!({
            "model": "m",
            "max_tokens": 64,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=1; cch=abc"},
                {"type": "text", "text": "You are terse."}
            ],
            "tools": [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
            "thinking": {"type": "enabled", "budget_tokens": 256},
            "messages": [
                {"role": "user", "content": "list files"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "use ls", "signature": "s"},
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "a.txt"}]},
                    {"type": "text", "text": "and now?"}
                ]}
            ]
        });
        assert!(needs_bridge(&body));
        let chat = to_openai_chat_body(&body).unwrap();
        assert_eq!(
            chat["messages"][0],
            json!({"role": "system", "content": "You are terse."})
        );
        assert_eq!(
            chat["messages"][2]["tool_calls"][0]["function"]["arguments"],
            json!("{\"cmd\":\"ls\"}")
        );
        assert_eq!(chat["messages"][2]["reasoning_content"], json!("use ls"));
        assert_eq!(
            chat["messages"][3],
            json!({"role": "tool", "tool_call_id": "t1", "content": "a.txt"})
        );
        assert_eq!(
            chat["messages"][4],
            json!({"role": "user", "content": "and now?"})
        );
        assert_eq!(
            chat["tools"][0]["function"]["parameters"],
            json!({"type": "object"})
        );
        assert_eq!(chat["tool_choice"], json!("required"));
        assert_eq!(
            chat["chat_template_kwargs"],
            json!({"enable_thinking": true})
        );
        assert_eq!(chat["ax_max_think_tokens"], json!(256));
        serde_json::from_value::<OpenAiChatCompletionHttpRequest>(chat)
            .expect("translated body is a valid chat request");
        // Plain text stays on the direct path.
        assert!(!needs_bridge(
            &json!({"max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]})
        ));
    }

    #[test]
    fn response_translation_emits_thinking_text_and_tool_use() {
        let chat = json!({
            "model": "m",
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": "running it",
                "reasoning_content": "think",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}]
            }}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4}
        });
        let message = from_openai_chat_response("msg-1", &chat);
        assert_eq!(message["content"][0]["type"], json!("thinking"));
        assert_eq!(
            message["content"][1],
            json!({"type": "text", "text": "running it"})
        );
        assert_eq!(message["content"][2]["input"], json!({"cmd": "ls"}));
        assert_eq!(message["stop_reason"], json!("tool_use"));
        assert_eq!(
            message["usage"],
            json!({"input_tokens": 9, "output_tokens": 4})
        );
    }

    #[test]
    fn stream_translation_orders_blocks_and_closes_them() {
        let mut translator = StreamTranslator::default();
        let mut events = translator.start("msg-2", "m");
        for chunk in [
            json!({"choices": [{"delta": {"reasoning_content": "hm"}}]}),
            json!({"choices": [{"delta": {"content": "Hi"}}]}),
            json!({"choices": [{"delta": {"content": " there"}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                "function": {"name": "bash", "arguments": "{\"cmd\""}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"arguments": ":\"ls\"}"}}]}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}),
        ] {
            events.push_str(&translator.chunk(&chunk));
        }
        events.push_str(&translator.finish());
        let names: Vec<&str> = events
            .lines()
            .filter_map(|line| line.strip_prefix("event: "))
            .collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert!(events.contains("\"stop_reason\":\"tool_use\""));
        assert!(events.contains("\"output_tokens\":7"));
        assert!(events.contains("\"partial_json\":\":\\\"ls\\\"}\""));
    }
}
