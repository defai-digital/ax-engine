use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::{GenerateFinishReason, GenerateResponse};
use axum::Json;
use axum::response::IntoResponse;
use serde_json::Value;

use super::requests::{OpenAiResponseOptions, OpenAiToolContract};
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
        extract_tool_calls(&mut content, options.tool_contract.as_deref())
    } else {
        None
    };
    let finish_reason = if tool_calls.as_ref().is_some_and(|calls| !calls.is_empty()) {
        Some("tool_calls")
    } else {
        openai_finish_reason(response.finish_reason)
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
            finish_reason,
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
fn extract_tool_calls(
    content: &mut String,
    tool_contract: Option<&OpenAiToolContract>,
) -> Option<Vec<OpenAiToolCall>> {
    let mut remaining = content.clone();
    let mut calls = Vec::new();
    while let Some((mut function, next_remaining)) = extract_tool_call_payload(&remaining) {
        if let Some(contract) = tool_contract {
            function.name = contract.canonical_tool_name(&function.name);
            function.arguments = contract.canonical_arguments(&function.name, function.arguments);
        }
        calls.push(OpenAiToolCall {
            id: format!("call_{}", calls.len()),
            tool_type: "function",
            function,
        });
        remaining = next_remaining;
    }
    if calls.is_empty() {
        return None;
    }
    *content = remaining.trim().to_string();
    Some(calls)
}

fn extract_tool_call_payload(content: &str) -> Option<(OpenAiFunctionCall, String)> {
    match (
        content.find("<tool_call>"),
        content.find("<|tool_call>call:"),
    ) {
        (Some(xml), Some(gemma)) if xml <= gemma => extract_xml_tool_call_payload(content),
        (Some(_), None) => extract_xml_tool_call_payload(content),
        (Some(_), Some(_)) | (None, Some(_)) => extract_gemma4_tool_call_payload(content),
        (None, None) => None,
    }
}

fn extract_xml_tool_call_payload(content: &str) -> Option<(OpenAiFunctionCall, String)> {
    let start = content.find("<tool_call>")?;
    let body_start = start + "<tool_call>".len();
    let relative_end = content[body_start..].find("</tool_call>");
    let end = relative_end
        .map(|offset| body_start + offset)
        .unwrap_or(content.len());
    let function = parse_tool_call_body(content[body_start..end].trim())?;
    let suffix_start = relative_end
        .map(|_| end + "</tool_call>".len())
        .unwrap_or(content.len());
    let remaining = format!("{}{}", &content[..start], &content[suffix_start..]);
    Some((function, remaining))
}

fn extract_gemma4_tool_call_payload(content: &str) -> Option<(OpenAiFunctionCall, String)> {
    let start = content.find("<|tool_call>call:")?;
    let name_start = start + "<|tool_call>call:".len();
    let name_end = name_start + content[name_start..].find('{')?;
    let name = content[name_start..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let body_start = name_end + 1;
    let body_end = find_matching_gemma4_object_end(content, body_start)?;
    let marker_start = body_end + 1;
    let marker = "<tool_call|>";
    if !content[marker_start..].starts_with(marker) {
        return None;
    }
    let arguments = parse_gemma4_arguments(&content[body_start..body_end])?;
    let remaining = format!(
        "{}{}",
        &content[..start],
        &content[marker_start + marker.len()..]
    );
    Some((OpenAiFunctionCall { name, arguments }, remaining))
}

fn find_matching_gemma4_object_end(content: &str, body_start: usize) -> Option<usize> {
    let mut depth = 1usize;
    let mut index = body_start;
    while index < content.len() {
        if content[index..].starts_with("<|\"|>") {
            index += "<|\"|>".len();
            let relative_end = content[index..].find("<|\"|>")?;
            index += relative_end + "<|\"|>".len();
            continue;
        }
        let ch = content[index..].chars().next()?;
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        index += ch.len_utf8();
    }
    None
}

fn parse_tool_call_body(body: &str) -> Option<OpenAiFunctionCall> {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        return parse_tool_call_function(&value);
    }
    parse_qwen_function_tool_call(body)
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

fn parse_qwen_function_tool_call(body: &str) -> Option<OpenAiFunctionCall> {
    let function_marker = "<function=";
    let function_start = body.find(function_marker)?;
    let name_start = function_start + function_marker.len();
    let name_end = name_start + body[name_start..].find('>')?;
    let name = unescape_xml_text(body[name_start..name_end].trim());
    if name.is_empty() {
        return None;
    }

    let body_start = name_end + 1;
    let body_end = body[body_start..]
        .find("</function>")
        .map(|offset| body_start + offset)
        .unwrap_or(body.len());
    let inner = body[body_start..body_end].trim();
    let parameters = parse_qwen_tool_parameters(inner);
    let arguments = if parameters.is_empty() && !inner.is_empty() {
        serde_json::from_str::<Value>(inner)
            .ok()
            .map(|value| serialize_tool_arguments(Some(&value)))?
    } else {
        serde_json::to_string(&Value::Object(parameters)).ok()?
    };
    Some(OpenAiFunctionCall { name, arguments })
}

fn parse_gemma4_arguments(body: &str) -> Option<String> {
    let mut parser = Gemma4DslParser::new(body);
    let value = parser.parse_object_body()?;
    serde_json::to_string(&value).ok()
}

struct Gemma4DslParser<'a> {
    input: &'a str,
    index: usize,
}

impl<'a> Gemma4DslParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, index: 0 }
    }

    fn parse_object_body(&mut self) -> Option<Value> {
        let mut object = serde_json::Map::new();
        loop {
            self.skip_ws();
            if self.index >= self.input.len() {
                break;
            }
            let key = self.parse_key()?;
            self.skip_ws();
            self.consume_char(':')?;
            let value = self.parse_value()?;
            object.insert(key, value);
            self.skip_ws();
            if self.peek_char() == Some(',') {
                self.index += 1;
            }
        }
        Some(Value::Object(object))
    }

    fn parse_value(&mut self) -> Option<Value> {
        self.skip_ws();
        if self.input[self.index..].starts_with("<|\"|>") {
            return self.parse_quoted_string().map(Value::String);
        }
        match self.peek_char()? {
            '{' => {
                self.index += 1;
                let value = self.parse_braced_object()?;
                self.consume_char('}')?;
                Some(value)
            }
            '[' => {
                self.index += 1;
                let mut values = Vec::new();
                loop {
                    self.skip_ws();
                    if self.peek_char() == Some(']') {
                        self.index += 1;
                        break;
                    }
                    values.push(self.parse_value()?);
                    self.skip_ws();
                    match self.peek_char()? {
                        ',' => self.index += 1,
                        ']' => {
                            self.index += 1;
                            break;
                        }
                        _ => return None,
                    }
                }
                Some(Value::Array(values))
            }
            _ => self.parse_atom(),
        }
    }

    fn parse_braced_object(&mut self) -> Option<Value> {
        let mut object = serde_json::Map::new();
        loop {
            self.skip_ws();
            if self.peek_char() == Some('}') {
                break;
            }
            let key = self.parse_key()?;
            self.skip_ws();
            self.consume_char(':')?;
            let value = self.parse_value()?;
            object.insert(key, value);
            self.skip_ws();
            match self.peek_char()? {
                ',' => self.index += 1,
                '}' => break,
                _ => return None,
            }
        }
        Some(Value::Object(object))
    }

    fn parse_key(&mut self) -> Option<String> {
        self.skip_ws();
        if self.input[self.index..].starts_with("<|\"|>") {
            return self.parse_quoted_string();
        }
        let start = self.index;
        while let Some(ch) = self.peek_char() {
            if matches!(ch, ':' | ',' | '}' | ']') || ch.is_whitespace() {
                break;
            }
            self.index += ch.len_utf8();
        }
        (self.index > start).then(|| self.input[start..self.index].to_string())
    }

    fn parse_quoted_string(&mut self) -> Option<String> {
        let marker = "<|\"|>";
        if !self.input[self.index..].starts_with(marker) {
            return None;
        }
        self.index += marker.len();
        let relative_end = self.input[self.index..].find(marker)?;
        let value = self.input[self.index..self.index + relative_end].to_string();
        self.index += relative_end + marker.len();
        Some(value)
    }

    fn parse_atom(&mut self) -> Option<Value> {
        let start = self.index;
        while let Some(ch) = self.peek_char() {
            if matches!(ch, ',' | '}' | ']') {
                break;
            }
            self.index += ch.len_utf8();
        }
        let atom = self.input[start..self.index].trim();
        if atom.is_empty() {
            return None;
        }
        match atom {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            "null" => Some(Value::Null),
            value => serde_json::from_str::<Value>(value)
                .ok()
                .or_else(|| Some(Value::String(value.to_string()))),
        }
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek_char() {
            if !ch.is_whitespace() {
                break;
            }
            self.index += ch.len_utf8();
        }
    }

    fn consume_char(&mut self, expected: char) -> Option<()> {
        self.skip_ws();
        if self.peek_char()? != expected {
            return None;
        }
        self.index += expected.len_utf8();
        Some(())
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.index..].chars().next()
    }
}

fn unescape_xml_text(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn parse_qwen_tool_parameters(body: &str) -> serde_json::Map<String, Value> {
    let mut parameters = serde_json::Map::new();
    let parameter_marker = "<parameter=";
    let mut offset = 0;
    while let Some(relative_start) = body[offset..].find(parameter_marker) {
        let name_start = offset + relative_start + parameter_marker.len();
        let Some(relative_name_end) = body[name_start..].find('>') else {
            break;
        };
        let name_end = name_start + relative_name_end;
        let name = unescape_xml_text(body[name_start..name_end].trim());
        if name.is_empty() {
            offset = name_end + 1;
            continue;
        }

        let value_start = name_end + 1;
        let value_end = qwen_parameter_value_end(body, value_start);
        let raw_value = unescape_xml_text(body[value_start..value_end].trim());
        let value =
            serde_json::from_str(&raw_value).unwrap_or_else(|_| Value::String(raw_value.clone()));
        parameters.insert(name, value);
        offset = value_end;
    }
    parameters
}

/// End index (exclusive) of a `<parameter=>` value, matching the reference
/// `qwen3_coder_xml` parser's alternation: an explicit `</parameter>` close is
/// preferred, but a missing close is treated as an implicit terminator at the
/// next `<parameter=`, `</function>`, or end of body. Qwen3-Coder models
/// frequently truncate or omit the closing tag, and the earlier `break`-on-
/// missing dropped the entire parameter (and, because the residual XML body is
/// not bare JSON, the whole tool call) — surfacing it as plain text that the
/// guard then blocks as `unexecutable_tool_text`.
fn qwen_parameter_value_end(body: &str, value_start: usize) -> usize {
    let next_param = body[value_start..]
        .find("<parameter=")
        .map(|relative| value_start + relative);
    let function_end = body[value_start..]
        .find("</function>")
        .map(|relative| value_start + relative);
    let next_implicit = match (next_param, function_end) {
        (Some(param), Some(func)) => param.min(func),
        (Some(only), None) | (None, Some(only)) => only,
        (None, None) => body.len(),
    };
    // Prefer the explicit `</parameter>` close only when it belongs to *this*
    // parameter, i.e. it precedes the next parameter / function close. An
    // explicit close that lands past the next delimiter belongs to a later
    // parameter, so this one was truncated and must end at the implicit
    // delimiter instead of greedily absorbing the following parameter.
    if let Some(relative) = body[value_start..].find("</parameter>") {
        let explicit = value_start + relative;
        if explicit <= next_implicit {
            return explicit;
        }
    }
    next_implicit
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
