use std::collections::BTreeMap;
use std::path::Path;

use ax_engine_sdk::{
    EngineTokenizer, Gemma4UnifiedAudioInput, Gemma4UnifiedAudioRuntimeInput, Gemma4UnifiedError,
    Gemma4UnifiedImageInput, Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedModality,
    Gemma4UnifiedProcessorConfig, Gemma4UnifiedRuntimeInputs, Gemma4UnifiedSoftTokenRange,
    Gemma4UnifiedTokenSpan, Gemma4UnifiedVideoInput, Gemma4UnifiedVideoRuntimeInput,
    LlamaCppChatMessage, MlxLmChatMessage,
};
use axum::Json;
use axum::http::StatusCode;
use serde_json::{Map, Value};

use crate::chat;
use crate::errors::{ErrorResponse, error_response};
use crate::multimodal::{
    self, MediaError, MediaProcessors, PreprocessedAudio, PreprocessedImage, PreprocessedVideo,
    VideoFrame,
};
use crate::openai::schema::{
    OpenAiChatContent, OpenAiChatContentPart, OpenAiChatMessage, OpenAiStopInput,
};

type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);
type ChatMessagePairs = Vec<(String, String)>;

/// A native-MLX chat prompt with Gemma 4 unified media already expanded into
/// soft-token spans and preprocessed tensors.
pub(crate) struct Gemma4UnifiedChatPrompt {
    pub(crate) input_tokens: Vec<u32>,
    pub(crate) runtime_inputs: Gemma4UnifiedRuntimeInputs,
}

pub(crate) fn render_openai_chat_prompt(
    model_id: &str,
    messages: &[OpenAiChatMessage],
) -> Result<String, HttpErrorResponse> {
    render_openai_chat_prompt_with_tools(model_id, messages, None, None)
}

pub(crate) fn render_openai_chat_prompt_with_tools(
    model_id: &str,
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
) -> Result<String, HttpErrorResponse> {
    if matches!(
        chat::ChatPromptTemplate::for_model_id(model_id),
        chat::ChatPromptTemplate::Gemma4
    ) && gemma4_tool_surface_is_present(messages, tools, tool_choice)
    {
        return render_gemma4_openai_chat_prompt(messages, tools);
    }
    let tool_contract_style = qwen_tool_contract_style(model_id);
    let rendered_messages = render_openai_chat_message_pairs(messages, tool_contract_style)?;
    let rendered_messages =
        prepend_qwen_tool_contract(model_id, rendered_messages, tools, tool_choice);
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
            let content = render_openai_chat_content(message.content.as_ref())?;
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
            let content = render_openai_chat_content(message.content.as_ref())?;
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
    tool_contract_style: QwenToolContractStyle,
) -> Result<ChatMessagePairs, HttpErrorResponse> {
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }
    messages
        .iter()
        .map(|message| {
            let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
            let mut content = render_openai_chat_content(message.content.as_ref())?;
            if role == "assistant"
                && let Some(tool_calls) = message.tool_calls.as_ref()
                && let Some(rendered_tool_calls) =
                    render_assistant_tool_calls(tool_calls, tool_contract_style)
            {
                if !content.trim().is_empty() {
                    if tool_contract_style.uses_xml_tool_calls() {
                        content.push_str("\n\n");
                    } else {
                        content.push('\n');
                    }
                }
                content.push_str(&rendered_tool_calls);
            }
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

fn render_openai_chat_content(
    content: Option<&OpenAiChatContent>,
) -> Result<String, HttpErrorResponse> {
    let Some(content) = content else {
        return Ok(String::new());
    };
    match content {
        OpenAiChatContent::Text(text) => Ok(text.clone()),
        OpenAiChatContent::Parts(parts) => {
            let mut rendered = String::new();
            for part in parts {
                match chat_content_part_kind(part) {
                    OpenAiChatContentPartKind::Text => {
                        let text = part.text.as_deref().ok_or_else(|| {
                            error_response(
                                StatusCode::BAD_REQUEST,
                                "invalid_request",
                                format!(
                                    "{} chat content parts require a text field",
                                    part.part_type
                                ),
                            )
                        })?;
                        rendered.push_str(text);
                    }
                    OpenAiChatContentPartKind::Media(kind) => {
                        return Err(openai_media_part_error(kind, part));
                    }
                    OpenAiChatContentPartKind::Unsupported => {
                        return Err(error_response(
                            StatusCode::BAD_REQUEST,
                            "invalid_request",
                            format!(
                                "unsupported chat content part type {}; AX preview currently accepts text-only OpenAI chat content plus processed multimodal_inputs on /v1/generate",
                                part.part_type
                            ),
                        ));
                    }
                }
            }
            Ok(rendered)
        }
    }
}

fn gemma4_tool_surface_is_present(
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
) -> bool {
    tools.map(openai_value_is_present).unwrap_or(false)
        || tool_choice
            .map(tool_choice_forces_tool_call)
            .unwrap_or(false)
        || messages.iter().any(|message| {
            matches!(message.role.as_str(), "tool" | "function")
                || message
                    .tool_calls
                    .as_ref()
                    .is_some_and(openai_value_is_present)
        })
}

fn render_gemma4_openai_chat_prompt(
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
) -> Result<String, HttpErrorResponse> {
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }

    let mut prompt = String::from("<bos>");
    let mut index = 0usize;
    let has_tools = tools.map(openai_value_is_present).unwrap_or(false);
    let first_role = messages
        .first()
        .map(|message| message.role.trim())
        .unwrap_or_default();

    if has_tools || matches!(first_role, "system" | "developer") {
        prompt.push_str("<|turn>system\n");
        if matches!(first_role, "system" | "developer") {
            let content = render_openai_chat_content(messages[0].content.as_ref())?;
            prompt.push_str(content.trim());
            index = 1;
        }
        if let Some(tools) = tools
            && openai_value_is_present(tools)
        {
            for declaration in render_gemma4_tool_declarations(tools) {
                prompt.push_str("<|tool>");
                prompt.push_str(&declaration);
                prompt.push_str("<tool|>");
            }
        }
        prompt.push_str("<turn|>\n");
    }

    let mut suppress_generation_prompt = false;
    while index < messages.len() {
        let message = &messages[index];
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        if matches!(role, "tool" | "function") {
            index += 1;
            continue;
        }

        let turn = if role == "assistant" { "model" } else { role };
        prompt.push_str("<|turn>");
        prompt.push_str(turn);
        prompt.push('\n');

        if role == "assistant" {
            let mut rendered_tool_call = false;
            let mut tool_names_by_id = BTreeMap::new();
            if let Some(tool_calls) = message.tool_calls.as_ref() {
                for call in gemma4_tool_calls(tool_calls) {
                    if let Some(id) = call.id {
                        tool_names_by_id.insert(id, call.name.clone());
                    }
                    prompt.push_str("<|tool_call>call:");
                    prompt.push_str(&call.name);
                    prompt.push('{');
                    prompt.push_str(&format_gemma4_arguments_object(&call.arguments));
                    prompt.push_str("}<tool_call|>");
                    rendered_tool_call = true;
                }
            }

            let (tool_response_count, next_index) = render_following_gemma4_tool_responses(
                messages,
                index + 1,
                &tool_names_by_id,
                &mut prompt,
            )?;
            let content = render_openai_chat_content(message.content.as_ref())?;
            if !content.trim().is_empty() {
                prompt.push_str(strip_gemma4_thinking_from_history(&content).trim());
            }

            if rendered_tool_call && tool_response_count == 0 {
                prompt.push_str("<|tool_response>");
                suppress_generation_prompt = true;
            } else if !(tool_response_count > 0 && content.trim().is_empty()) {
                prompt.push_str("<turn|>\n");
                suppress_generation_prompt = false;
            } else {
                suppress_generation_prompt = true;
            }
            index = next_index;
            continue;
        }

        let content = render_openai_chat_content(message.content.as_ref())?;
        prompt.push_str(content.trim());
        prompt.push_str("<turn|>\n");
        suppress_generation_prompt = false;
        index += 1;
    }

    if !suppress_generation_prompt {
        prompt.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    }
    Ok(prompt)
}

#[derive(Debug)]
struct Gemma4ToolCall {
    id: Option<String>,
    name: String,
    arguments: Value,
}

fn gemma4_tool_calls(value: &Value) -> Vec<Gemma4ToolCall> {
    let calls = match value {
        Value::Array(calls) => calls.as_slice(),
        Value::Object(_) => std::slice::from_ref(value),
        _ => return Vec::new(),
    };
    calls
        .iter()
        .filter_map(|call| {
            let object = call.as_object()?;
            let function = object.get("function")?.as_object()?;
            let name = function.get("name")?.as_str()?.to_string();
            Some(Gemma4ToolCall {
                id: object.get("id").and_then(Value::as_str).map(str::to_string),
                name,
                arguments: normalize_tool_arguments(function.get("arguments")),
            })
        })
        .collect()
}

fn render_following_gemma4_tool_responses(
    messages: &[OpenAiChatMessage],
    mut index: usize,
    tool_names_by_id: &BTreeMap<String, String>,
    prompt: &mut String,
) -> Result<(usize, usize), HttpErrorResponse> {
    let mut count = 0usize;
    while index < messages.len() {
        let message = &messages[index];
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        if !matches!(role, "tool" | "function") {
            break;
        }
        let tool_name = message
            ._tool_call_id
            .as_ref()
            .and_then(|id| tool_names_by_id.get(id))
            .or(message._name.as_ref())
            .map(String::as_str)
            .unwrap_or("unknown");
        let content = render_openai_chat_content(message.content.as_ref())?;
        prompt.push_str("<|tool_response>response:");
        prompt.push_str(tool_name);
        prompt.push_str("{value:");
        prompt.push_str(&format_gemma4_argument(&Value::String(content), false));
        prompt.push_str("}<tool_response|>");
        count += 1;
        index += 1;
    }
    Ok((count, index))
}

fn render_gemma4_tool_declarations(tools: &Value) -> Vec<String> {
    match tools {
        Value::Array(items) => items
            .iter()
            .filter_map(render_gemma4_tool_declaration)
            .collect(),
        value => render_gemma4_tool_declaration(value).into_iter().collect(),
    }
}

fn render_gemma4_tool_declaration(tool: &Value) -> Option<String> {
    let function = tool.get("function").and_then(Value::as_object)?;
    let name = function.get("name")?.as_str()?;
    let description = function
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut rendered = String::new();
    rendered.push_str("declaration:");
    rendered.push_str(name);
    rendered.push_str("{description:");
    rendered.push_str(&format_gemma4_argument(
        &Value::String(description.to_string()),
        true,
    ));
    if let Some(parameters) = function.get("parameters").and_then(Value::as_object) {
        rendered.push_str(",parameters:{");
        if let Some(properties) = parameters.get("properties").and_then(Value::as_object) {
            rendered.push_str("properties:{ ");
            rendered.push_str(&format_gemma4_parameters(properties));
            rendered.push_str(" },");
        }
        if let Some(required) = parameters.get("required") {
            rendered.push_str("required:");
            rendered.push_str(&format_gemma4_argument(required, true));
            rendered.push(',');
        }
        if let Some(param_type) = parameters.get("type") {
            rendered.push_str("type:");
            rendered.push_str(&format_gemma4_upper_type_argument(param_type));
            rendered.push('}');
        } else {
            rendered.push('}');
        }
    }
    rendered.push('}');
    Some(rendered)
}

fn format_gemma4_parameters(properties: &serde_json::Map<String, Value>) -> String {
    let standard_keys = ["description", "type", "properties", "required", "nullable"];
    let mut rendered = Vec::new();
    let mut keys = properties.keys().collect::<Vec<_>>();
    keys.sort();
    for key in keys {
        if standard_keys.contains(&key.as_str()) {
            continue;
        }
        let Some(value) = properties.get(key).and_then(Value::as_object) else {
            continue;
        };
        let mut fields = Vec::new();
        if let Some(description) = value.get("description").and_then(Value::as_str) {
            fields.push(format!(
                "description:{}",
                format_gemma4_argument(&Value::String(description.to_string()), true)
            ));
        }
        if value.get("nullable").and_then(Value::as_bool) == Some(true) {
            fields.push("nullable:true".to_string());
        }
        if let Some(enum_values) = value.get("enum") {
            fields.push(format!(
                "enum:{}",
                format_gemma4_argument(enum_values, true)
            ));
        }
        if let Some(nested) = value.get("properties").and_then(Value::as_object) {
            fields.push(format!(
                "properties:{{{}}}",
                format_gemma4_parameters(nested)
            ));
        }
        if let Some(required) = value.get("required") {
            fields.push(format!(
                "required:{}",
                format_gemma4_argument(required, true)
            ));
        }
        if let Some(items) = value.get("items") {
            fields.push(format!("items:{}", format_gemma4_argument(items, false)));
        }
        if let Some(param_type) = value.get("type") {
            fields.push(format!(
                "type:{}",
                format_gemma4_upper_type_argument(param_type)
            ));
        }
        rendered.push(format!("{key}:{{{}}}", fields.join(",")));
    }
    rendered.join(",")
}

fn format_gemma4_arguments_object(arguments: &Value) -> String {
    match arguments {
        Value::Object(object) => {
            let mut keys = object.keys().collect::<Vec<_>>();
            keys.sort();
            keys.into_iter()
                .filter_map(|key| {
                    object
                        .get(key)
                        .map(|value| format!("{key}:{}", format_gemma4_argument(value, false)))
                })
                .collect::<Vec<_>>()
                .join(",")
        }
        Value::String(value) => value.clone(),
        _ => String::new(),
    }
}

fn format_gemma4_upper_type_argument(value: &Value) -> String {
    match value {
        Value::String(value) => {
            format_gemma4_argument(&Value::String(value.to_ascii_uppercase()), true)
        }
        Value::Array(values) => {
            let upper = Value::Array(
                values
                    .iter()
                    .map(|value| match value {
                        Value::String(value) => Value::String(value.to_ascii_uppercase()),
                        value => value.clone(),
                    })
                    .collect(),
            );
            format_gemma4_argument(&upper, true)
        }
        value => format_gemma4_argument(value, true),
    }
}

fn format_gemma4_argument(value: &Value, escape_keys: bool) -> String {
    match value {
        Value::String(value) => format!("<|\"|>{value}<|\"|>"),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Null => "null".to_string(),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(|value| format_gemma4_argument(value, escape_keys))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(object) => {
            let mut keys = object.keys().collect::<Vec<_>>();
            keys.sort();
            let fields = keys
                .into_iter()
                .filter_map(|key| {
                    object.get(key).map(|value| {
                        let rendered_key = if escape_keys {
                            format!("<|\"|>{key}<|\"|>")
                        } else {
                            key.to_string()
                        };
                        format!(
                            "{rendered_key}:{}",
                            format_gemma4_argument(value, escape_keys)
                        )
                    })
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{fields}}}")
        }
    }
}

fn strip_gemma4_thinking_from_history(content: &str) -> String {
    let mut remaining = content;
    let mut rendered = String::new();
    while let Some(start) = remaining.find("<|channel>") {
        rendered.push_str(&remaining[..start]);
        let body_start = start + "<|channel>".len();
        let Some(relative_end) = remaining[body_start..].find("<channel|>") else {
            remaining = "";
            break;
        };
        remaining = &remaining[body_start + relative_end + "<channel|>".len()..];
    }
    rendered.push_str(remaining);
    rendered.trim().to_string()
}

fn prepend_qwen_tool_contract(
    model_id: &str,
    mut messages: ChatMessagePairs,
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
) -> ChatMessagePairs {
    if !matches!(
        chat::ChatPromptTemplate::for_model_id(model_id),
        chat::ChatPromptTemplate::QwenChatMl
    ) {
        return messages;
    }
    let style = qwen_tool_contract_style(model_id);
    let Some(contract) = render_tool_contract_system_message(tools, tool_choice, style) else {
        return messages;
    };
    if let Some((role, content)) = messages.first_mut()
        && role == "system"
    {
        content.push_str("\n\n");
        content.push_str(&contract);
    } else {
        let content = if style == QwenToolContractStyle::CoderXml {
            format!(
                "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.\n\n{contract}"
            )
        } else {
            contract
        };
        messages.insert(0, ("system".to_string(), content));
    }
    messages
}

fn render_tool_contract_system_message(
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    style: QwenToolContractStyle,
) -> Option<String> {
    let tools = tools?;
    if !openai_value_is_present(tools) {
        return None;
    }

    match style {
        QwenToolContractStyle::JsonTools => {
            render_json_tool_contract_system_message(tools, tool_choice)
        }
        QwenToolContractStyle::FunctionXml => {
            render_qwen_function_tool_contract_system_message(tools, tool_choice)
        }
        QwenToolContractStyle::CoderXml => {
            render_qwen_coder_tool_contract_system_message(tools, tool_choice)
        }
    }
}

fn render_json_tool_contract_system_message(
    tools: &Value,
    tool_choice: Option<&Value>,
) -> Option<String> {
    let mut message = String::from(
        "# Tools\n\n\
         You may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n\
         <tools>\n",
    );
    for line in render_json_tool_lines(tools) {
        message.push_str(&line);
        message.push('\n');
    }
    message.push_str(
        "</tools>\n\n\
         For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
         <tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
         </tool_call>",
    );

    if let Some(choice) = tool_choice
        && tool_choice_forces_tool_call(choice)
    {
        message.push_str(
            "\nThe current tool_choice requires using a tool when a matching function is available.",
        );
    }

    Some(message)
}

fn render_qwen_function_tool_contract_system_message(
    tools: &Value,
    tool_choice: Option<&Value>,
) -> Option<String> {
    let mut message = String::from(
        "# Tools\n\n\
         You have access to the following functions:\n\n\
         <tools>",
    );
    for line in render_json_tool_lines(tools) {
        message.push('\n');
        message.push_str(&line);
    }
    message.push_str("\n</tools>");
    message.push_str(
        "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
         <tool_call>\n\
         <function=example_function_name>\n\
         <parameter=example_parameter_1>\n\
         value_1\n\
         </parameter>\n\
         <parameter=example_parameter_2>\n\
         This is the value for the second parameter\n\
         that can span\n\
         multiple lines\n\
         </parameter>\n\
         </function>\n\
         </tool_call>\n\n\
         <IMPORTANT>\n\
         Reminder:\n\
         - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
         - Required parameters MUST be specified\n\
         - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
         - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
         </IMPORTANT>",
    );

    if let Some(choice) = tool_choice
        && tool_choice_forces_tool_call(choice)
    {
        message.push_str(
            "\nThe current tool_choice requires using a tool when a matching function is available.",
        );
    }

    Some(message)
}

fn render_qwen_coder_tool_contract_system_message(
    tools: &Value,
    tool_choice: Option<&Value>,
) -> Option<String> {
    let mut message = String::from("# Tools\n\nYou have access to the following tools:\n\n<tools>");
    for line in render_qwen_tool_declarations(tools) {
        message.push('\n');
        message.push_str(&line);
    }
    message.push_str("\n</tools>");
    message.push_str(
        "\n\nIf you choose to call a tool ONLY reply in the following format with NO suffix:\n\n\
         <tool_call>\n\
         <function=example_function_name>\n\
         <parameter=example_parameter_1>\n\
         value_1\n\
         </parameter>\n\
         <parameter=example_parameter_2>\n\
         value_2\n\
         </parameter>\n\
         </function>\n\
         </tool_call>\n\n\
         <IMPORTANT>\n\
         Reminder:\n\
         - Function calls MUST follow the specified format: the tool calling block MUST begin with an opening <tool_call> tag and end with a closing </tool_call> tag.\n\
         - Required parameters MUST be specified\n\
         - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
         - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
         </IMPORTANT>",
    );

    if let Some(choice) = tool_choice
        && tool_choice_forces_tool_call(choice)
    {
        message.push_str(
            "\nThe current tool_choice requires using a tool when a matching function is available.",
        );
    }

    Some(message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QwenToolContractStyle {
    JsonTools,
    FunctionXml,
    CoderXml,
}

impl QwenToolContractStyle {
    fn uses_xml_tool_calls(self) -> bool {
        matches!(self, Self::FunctionXml | Self::CoderXml)
    }
}

fn qwen_tool_contract_style(model_id: &str) -> QwenToolContractStyle {
    let normalized = normalize_model_id_token(model_id);
    if chat::is_qwen_non_thinking_only_model(model_id) {
        QwenToolContractStyle::CoderXml
    } else if normalized.contains("qwen3-next")
        || normalized.contains("qwen3-5")
        || normalized.contains("qwen35")
        || normalized.contains("qwen3-6")
        || normalized.contains("qwen36")
    {
        QwenToolContractStyle::FunctionXml
    } else {
        QwenToolContractStyle::JsonTools
    }
}

fn normalize_model_id_token(model_id: &str) -> String {
    model_id
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect()
}

fn render_json_tool_lines(tools: &Value) -> Vec<String> {
    match tools {
        Value::Array(items) => items.iter().map(compact_json).collect(),
        value => vec![compact_json(value)],
    }
}

fn render_qwen_tool_declarations(tools: &Value) -> Vec<String> {
    match tools {
        Value::Array(items) => items
            .iter()
            .filter_map(render_qwen_tool_declaration)
            .collect(),
        value => render_qwen_tool_declaration(value).into_iter().collect(),
    }
}

fn render_qwen_tool_declaration(tool: &Value) -> Option<String> {
    let function = openai_tool_function(tool)?;
    let name = function.get("name")?.as_str()?;
    let mut rendered = String::new();
    rendered.push_str("<function>\n<name>");
    rendered.push_str(&escape_xml_text(name));
    rendered.push_str("</name>");
    if let Some(description) = function.get("description").and_then(Value::as_str) {
        let description = description.trim();
        if !description.is_empty() {
            rendered.push_str("\n<description>");
            rendered.push_str(&escape_xml_text(description));
            rendered.push_str("</description>");
        }
    }

    rendered.push_str("\n<parameters>");
    if let Some(parameters) = function.get("parameters").and_then(Value::as_object) {
        if let Some(properties) = parameters.get("properties").and_then(Value::as_object) {
            for (parameter_name, parameter_fields) in properties {
                rendered.push_str("\n<parameter>\n<name>");
                rendered.push_str(&escape_xml_text(parameter_name));
                rendered.push_str("</name>");
                if let Some(parameter_type) = parameter_fields.get("type") {
                    rendered.push_str("\n<type>");
                    rendered.push_str(&escape_xml_text(&stringify_json_scalar(parameter_type)));
                    rendered.push_str("</type>");
                }
                if let Some(description) =
                    parameter_fields.get("description").and_then(Value::as_str)
                {
                    let description = description.trim();
                    if !description.is_empty() {
                        rendered.push_str("\n<description>");
                        rendered.push_str(&escape_xml_text(description));
                        rendered.push_str("</description>");
                    }
                }
                render_schema_extra(&mut rendered, parameter_fields, &["type", "description"]);
                rendered.push_str("\n</parameter>");
            }
        }
        render_schema_extra(
            &mut rendered,
            &Value::Object(parameters.clone()),
            &["type", "properties"],
        );
    }
    rendered.push_str("\n</parameters>\n</function>");
    Some(rendered)
}

fn render_schema_extra(rendered: &mut String, value: &Value, handled_keys: &[&str]) {
    let Some(object) = value.as_object() else {
        return;
    };
    for (key, value) in object {
        if handled_keys.iter().any(|handled| *handled == key) {
            continue;
        }
        rendered.push('\n');
        rendered.push('<');
        rendered.push_str(&escape_xml_text(key));
        rendered.push('>');
        rendered.push_str(&escape_xml_text(&stringify_json_scalar(value)));
        rendered.push_str("</");
        rendered.push_str(&escape_xml_text(key));
        rendered.push('>');
    }
}

fn openai_tool_function(tool: &Value) -> Option<&Map<String, Value>> {
    let object = tool.as_object()?;
    object
        .get("function")
        .and_then(Value::as_object)
        .or_else(|| object.get("name").is_some().then_some(object))
}

fn render_assistant_tool_calls(value: &Value, style: QwenToolContractStyle) -> Option<String> {
    let calls = match value {
        Value::Array(calls) => calls.as_slice(),
        Value::Object(_) => std::slice::from_ref(value),
        _ => return None,
    };

    let rendered = calls
        .iter()
        .filter_map(|call| render_assistant_tool_call(call, style))
        .collect::<Vec<_>>();
    (!rendered.is_empty()).then(|| rendered.join("\n"))
}

fn render_assistant_tool_call(value: &Value, style: QwenToolContractStyle) -> Option<String> {
    let object = value.as_object()?;
    let function = object.get("function").and_then(Value::as_object)?;
    let name = function.get("name")?.as_str()?;
    let arguments = normalize_tool_arguments(function.get("arguments"));
    match style {
        QwenToolContractStyle::JsonTools => {
            let name_json = serde_json::to_string(name).ok()?;
            Some(format!(
                "<tool_call>\n{{\"name\": {name_json}, \"arguments\": {}}}\n</tool_call>",
                compact_json(&arguments)
            ))
        }
        QwenToolContractStyle::FunctionXml | QwenToolContractStyle::CoderXml => {
            Some(render_qwen_xml_tool_call(name, &arguments))
        }
    }
}

fn normalize_tool_arguments(value: Option<&Value>) -> Value {
    match value {
        Some(Value::String(value)) => {
            serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.clone()))
        }
        Some(value) => value.clone(),
        None => serde_json::json!({}),
    }
}

fn tool_choice_forces_tool_call(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "auto" | "none" | "false" | "off")
        }
        Value::Array(values) => !values.is_empty(),
        Value::Object(object) => !object.is_empty(),
        Value::Number(value) => value.as_u64().unwrap_or(1) != 0,
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

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn render_qwen_xml_tool_call(name: &str, arguments: &Value) -> String {
    let mut rendered = String::new();
    rendered.push_str("<tool_call>\n<function=");
    rendered.push_str(&escape_xml_text(name));
    rendered.push('>');
    if let Some(args) = arguments.as_object() {
        for (key, value) in args {
            rendered.push_str("\n<parameter=");
            rendered.push_str(&escape_xml_text(key));
            rendered.push_str(">\n");
            rendered.push_str(&escape_xml_text(&stringify_json_scalar(value)));
            rendered.push_str("\n</parameter>");
        }
    }
    rendered.push_str("\n</function>\n</tool_call>");
    rendered
}

fn stringify_json_scalar(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        value => compact_json(value),
    }
}

fn escape_xml_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenAiChatContentPartKind {
    Text,
    Media(OpenAiChatMediaKind),
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenAiChatMediaKind {
    Image,
    Audio,
    Video,
}

fn chat_content_part_kind(part: &OpenAiChatContentPart) -> OpenAiChatContentPartKind {
    match part.part_type.as_str() {
        "text" | "input_text" => OpenAiChatContentPartKind::Text,
        "image_url" | "input_image" | "image" => {
            OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Image)
        }
        "input_audio" | "audio_url" | "audio" => {
            OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Audio)
        }
        "video_url" | "input_video" | "video" => {
            OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Video)
        }
        _ => OpenAiChatContentPartKind::Unsupported,
    }
}

fn openai_media_part_error(
    media_kind: OpenAiChatMediaKind,
    part: &OpenAiChatContentPart,
) -> HttpErrorResponse {
    let field_hint = match media_kind {
        OpenAiChatMediaKind::Image => {
            if part.image_url.is_some() {
                "image_url"
            } else {
                "image payload"
            }
        }
        OpenAiChatMediaKind::Audio => {
            if part.input_audio.is_some() {
                "input_audio"
            } else if part.audio_url.is_some() {
                "audio_url"
            } else {
                "audio payload"
            }
        }
        OpenAiChatMediaKind::Video => {
            if part.video_url.is_some() {
                "video_url"
            } else {
                "video payload"
            }
        }
    };
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!(
            "OpenAI chat content part type {} includes {field_hint}, but this request path is text-only. Inline media requires native MLX Gemma4 chat without pre-supplied multimodal_inputs; for manual multimodal control, send preprocessed multimodal_inputs.gemma4_unified to /v1/generate",
            part.part_type
        ),
    )
}

/// True when any message carries an inline media content part (image/audio/video).
pub(crate) fn messages_contain_inline_media(messages: &[OpenAiChatMessage]) -> bool {
    messages.iter().any(|message| match &message.content {
        Some(OpenAiChatContent::Text(_)) | None => false,
        Some(OpenAiChatContent::Parts(parts)) => parts.iter().any(|part| {
            matches!(
                chat_content_part_kind(part),
                OpenAiChatContentPartKind::Media(_)
            )
        }),
    })
}

/// Build a native-MLX chat prompt when the messages carry inline base64 image or
/// audio parts, expanding them into Gemma 4 unified soft-token spans and the
/// encoder-free connector's preprocessed tensors. Returns `Ok(None)` when there
/// is no inline media, so the caller falls back to the text-only path.
pub(crate) fn render_gemma4_unified_chat_with_media(
    model_id: &str,
    model_dir: &Path,
    messages: &[OpenAiChatMessage],
) -> Result<Option<Gemma4UnifiedChatPrompt>, HttpErrorResponse> {
    if !messages_contain_inline_media(messages) {
        return Ok(None);
    }
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }

    let tokenizer = EngineTokenizer::from_model_dir(model_dir).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("failed to load tokenizer for multimodal chat: {error}"),
        )
    })?;
    let MediaProcessors {
        config,
        normalization,
        video_vision,
        video_max_frames,
    } = multimodal::load_processor_config(model_dir).map_err(media_error_response)?;

    let image_placeholder = placeholder_string(&tokenizer, config.tokens.image_token_id, "image")?;
    let audio_placeholder = placeholder_string(&tokenizer, config.tokens.audio_token_id, "audio")?;
    // A video_token_id of 0 means the model declares no video token.
    let video_placeholder = if config.tokens.video_token_id != 0 {
        Some(placeholder_string(
            &tokenizer,
            config.tokens.video_token_id,
            "video",
        )?)
    } else {
        None
    };

    // Render the prompt with one placeholder token per media item, collecting the
    // raw bytes in document order so they line up with the placeholder positions.
    let mut collected = CollectedMedia::default();
    let mut pairs: ChatMessagePairs = Vec::with_capacity(messages.len());
    for message in messages {
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        let content = render_content_collecting_media(
            message.content.as_ref(),
            &image_placeholder,
            &audio_placeholder,
            video_placeholder.as_deref(),
            &mut collected,
        )?;
        pairs.push((role.to_string(), content));
    }

    let prompt = chat::render_prompt(model_id, &pairs).map_err(chat_error_response)?;
    let tokens = tokenizer.encode(&prompt, false).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("failed to tokenize multimodal chat prompt: {error}"),
        )
    })?;

    // Preprocess raw media into the encoder-free connector's tensors.
    let images: Vec<PreprocessedImage> = collected
        .images
        .iter()
        .map(|bytes| multimodal::preprocess_image(bytes, &config.vision, &normalization))
        .collect::<Result<_, _>>()
        .map_err(media_error_response)?;
    let audios: Vec<PreprocessedAudio> = collected
        .audios
        .iter()
        .map(|bytes| match config.audio.as_ref() {
            Some(processor) => multimodal::preprocess_audio(bytes, processor),
            None => Err(MediaError::Unsupported(
                "model has no audio feature extractor; audio input is not supported".to_string(),
            )),
        })
        .collect::<Result<_, _>>()
        .map_err(media_error_response)?;
    // Video frames use a lower per-frame soft-token budget and carry mm:ss
    // timestamp tokens, matching the reference Gemma4UnifiedVideoProcessor.
    let mut videos: Vec<PreprocessedVideo> = Vec::with_capacity(collected.videos.len());
    let mut video_timestamp_tokens: Vec<Vec<Vec<u32>>> = Vec::with_capacity(collected.videos.len());
    for bytes in &collected.videos {
        let frames = multimodal::decode_video_frames(bytes, video_max_frames)
            .map_err(media_error_response)?;
        let timestamps = build_video_timestamp_tokens(&tokenizer, &frames)?;
        let preprocessed =
            multimodal::preprocess_video_frames(&frames, &video_vision, &normalization)
                .map_err(media_error_response)?;
        videos.push(preprocessed);
        video_timestamp_tokens.push(timestamps);
    }

    // Expand all placeholders in a single pass so every span indexes the final
    // token stream (separate passes would shift later positions).
    let image_inputs: Vec<Gemma4UnifiedImageInput> = images
        .iter()
        .map(|image| Gemma4UnifiedImageInput {
            width: image.width,
            height: image.height,
        })
        .collect();
    let audio_inputs: Vec<Gemma4UnifiedAudioInput> = audios
        .iter()
        .map(|audio| Gemma4UnifiedAudioInput {
            sample_count: audio.sample_count,
        })
        .collect();
    let video_inputs: Vec<Gemma4UnifiedVideoInput> = videos
        .iter()
        .zip(&video_timestamp_tokens)
        .map(|(video, timestamps)| Gemma4UnifiedVideoInput {
            frame_count: video.frame_count,
            soft_tokens_per_frame: video.soft_tokens_per_frame,
            timestamp_token_ids_per_frame: timestamps.clone(),
        })
        .collect();
    let expanded = expand_media(
        &config,
        &tokens,
        &image_inputs,
        &audio_inputs,
        &video_inputs,
    )?;

    let runtime_inputs = build_runtime_inputs(images, audios, videos, expanded.spans);
    runtime_inputs
        .validate_for_prompt_len(expanded.tokens.len())
        .map_err(|error| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("multimodal inputs do not match expanded prompt: {error}"),
            )
        })?;

    Ok(Some(Gemma4UnifiedChatPrompt {
        input_tokens: expanded.tokens,
        runtime_inputs,
    }))
}

/// Raw media bytes collected while rendering, in document order per modality.
#[derive(Default)]
struct CollectedMedia {
    images: Vec<Vec<u8>>,
    audios: Vec<Vec<u8>>,
    videos: Vec<Vec<u8>>,
}

fn placeholder_string(
    tokenizer: &EngineTokenizer,
    token_id: u32,
    label: &str,
) -> Result<String, HttpErrorResponse> {
    tokenizer.id_to_token(token_id).ok_or_else(|| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!(
                "tokenizer has no surface form for the Gemma4 {label} placeholder token {token_id}"
            ),
        )
    })
}

fn render_content_collecting_media(
    content: Option<&OpenAiChatContent>,
    image_placeholder: &str,
    audio_placeholder: &str,
    video_placeholder: Option<&str>,
    collected: &mut CollectedMedia,
) -> Result<String, HttpErrorResponse> {
    let Some(content) = content else {
        return Ok(String::new());
    };
    match content {
        OpenAiChatContent::Text(text) => Ok(text.clone()),
        OpenAiChatContent::Parts(parts) => {
            let mut rendered = String::new();
            for part in parts {
                match chat_content_part_kind(part) {
                    OpenAiChatContentPartKind::Text => {
                        let text = part.text.as_deref().ok_or_else(|| {
                            error_response(
                                StatusCode::BAD_REQUEST,
                                "invalid_request",
                                format!(
                                    "{} chat content parts require a text field",
                                    part.part_type
                                ),
                            )
                        })?;
                        rendered.push_str(text);
                    }
                    OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Image) => {
                        collected.images.push(image_part_bytes(part)?);
                        rendered.push_str(image_placeholder);
                    }
                    OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Audio) => {
                        collected.audios.push(audio_part_bytes(part)?);
                        rendered.push_str(audio_placeholder);
                    }
                    OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Video) => {
                        let placeholder = video_placeholder.ok_or_else(|| {
                            error_response(
                                StatusCode::BAD_REQUEST,
                                "invalid_request",
                                "model declares no video token; video chat content is not supported".to_string(),
                            )
                        })?;
                        collected.videos.push(video_part_bytes(part)?);
                        rendered.push_str(placeholder);
                    }
                    OpenAiChatContentPartKind::Unsupported => {
                        return Err(error_response(
                            StatusCode::BAD_REQUEST,
                            "invalid_request",
                            format!("unsupported chat content part type {}", part.part_type),
                        ));
                    }
                }
            }
            Ok(rendered)
        }
    }
}

fn image_part_bytes(part: &OpenAiChatContentPart) -> Result<Vec<u8>, HttpErrorResponse> {
    let value = part
        .image_url
        .as_ref()
        .ok_or_else(|| media_payload_missing("image", "image_url"))?;
    let url =
        data_value_url(value).ok_or_else(|| media_payload_missing("image", "image_url.url"))?;
    let (_mime, bytes) = multimodal::decode_data_uri(url).map_err(media_error_response)?;
    Ok(bytes)
}

fn audio_part_bytes(part: &OpenAiChatContentPart) -> Result<Vec<u8>, HttpErrorResponse> {
    if let Some(input_audio) = part.input_audio.as_ref() {
        let data = input_audio
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| media_payload_missing("audio", "input_audio.data"))?;
        return multimodal::decode_base64(data).map_err(media_error_response);
    }
    if let Some(audio_url) = part.audio_url.as_ref() {
        let url = data_value_url(audio_url)
            .ok_or_else(|| media_payload_missing("audio", "audio_url.url"))?;
        let (_mime, bytes) = multimodal::decode_data_uri(url).map_err(media_error_response)?;
        return Ok(bytes);
    }
    Err(media_payload_missing("audio", "input_audio or audio_url"))
}

fn video_part_bytes(part: &OpenAiChatContentPart) -> Result<Vec<u8>, HttpErrorResponse> {
    let value = part
        .video_url
        .as_ref()
        .ok_or_else(|| media_payload_missing("video", "video_url"))?;
    let url =
        data_value_url(value).ok_or_else(|| media_payload_missing("video", "video_url.url"))?;
    let (_mime, bytes) = multimodal::decode_data_uri(url).map_err(media_error_response)?;
    Ok(bytes)
}

/// Tokenize the `mm:ss` timestamp prefix for each video frame, matching the
/// reference `get_video_repl`: a leading space on every frame except the first,
/// and always a trailing space, e.g. `"00:00 "` then `" 00:02 "`.
fn build_video_timestamp_tokens(
    tokenizer: &EngineTokenizer,
    frames: &[VideoFrame],
) -> Result<Vec<Vec<u32>>, HttpErrorResponse> {
    frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let seconds = frame.timestamp_seconds.max(0.0);
            let minutes = (seconds / 60.0).floor() as u32;
            let secs = (seconds % 60.0).floor() as u32;
            let stamp = format!("{minutes:02}:{secs:02}");
            let prefix = if index == 0 {
                format!("{stamp} ")
            } else {
                format!(" {stamp} ")
            };
            tokenizer.encode(&prefix, false).map_err(|error| {
                error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    format!("failed to tokenize video timestamp: {error}"),
                )
            })
        })
        .collect()
}

/// OpenAI media URL fields are either a bare string or `{ "url": "..." }`.
fn data_value_url(value: &Value) -> Option<&str> {
    match value {
        Value::String(url) => Some(url.as_str()),
        _ => value.get("url").and_then(Value::as_str),
    }
}

/// A video span plus the per-frame soft-token ranges it occupies in the final
/// token stream.
type VideoSpan = (Gemma4UnifiedTokenSpan, Vec<Gemma4UnifiedSoftTokenRange>);

/// The expanded token stream and the spans for each modality, in document order.
#[derive(Debug)]
struct ExpandedMedia {
    tokens: Vec<u32>,
    spans: MediaSpans,
}

#[derive(Debug, Default)]
struct MediaSpans {
    image: Vec<Gemma4UnifiedTokenSpan>,
    audio: Vec<Gemma4UnifiedTokenSpan>,
    video: Vec<VideoSpan>,
}

/// Single-pass expansion of image, audio, and video placeholder tokens into
/// their boundary + soft-token replacements, recording spans against the final
/// stream. A single pass keeps every span (and video frame range) aligned with
/// the post-expansion token positions.
fn expand_media(
    config: &Gemma4UnifiedProcessorConfig,
    tokens: &[u32],
    image_inputs: &[Gemma4UnifiedImageInput],
    audio_inputs: &[Gemma4UnifiedAudioInput],
    video_inputs: &[Gemma4UnifiedVideoInput],
) -> Result<ExpandedMedia, HttpErrorResponse> {
    let image_token = config.tokens.image_token_id;
    let audio_token = config.tokens.audio_token_id;
    let video_token = config.tokens.video_token_id;

    let image_markers = tokens.iter().filter(|&&t| t == image_token).count();
    let audio_markers = tokens.iter().filter(|&&t| t == audio_token).count();
    if image_markers != image_inputs.len() {
        return Err(placeholder_count_error(
            "image",
            image_inputs.len(),
            image_markers,
        ));
    }
    if audio_markers != audio_inputs.len() {
        return Err(placeholder_count_error(
            "audio",
            audio_inputs.len(),
            audio_markers,
        ));
    }
    if video_token != 0 {
        let video_markers = tokens.iter().filter(|&&t| t == video_token).count();
        if video_markers != video_inputs.len() {
            return Err(placeholder_count_error(
                "video",
                video_inputs.len(),
                video_markers,
            ));
        }
    }

    let mut out = Vec::with_capacity(tokens.len());
    let mut spans = MediaSpans::default();
    let mut image_index = 0usize;
    let mut audio_index = 0usize;
    let mut video_index = 0usize;

    for (token_index, &token) in tokens.iter().enumerate() {
        if token == image_token {
            let input = image_inputs[image_index];
            image_index += 1;
            let replacement = config
                .image_replacement_tokens(input)
                .map_err(gemma4_unified_error_response)?;
            let soft = config
                .image_soft_tokens(input)
                .map_err(gemma4_unified_error_response)?;
            spans.image.push(Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Image,
                placeholder_index: token_index,
                replacement_start: out.len(),
                soft_token_count: soft,
                replacement_token_count: replacement.len() as u32,
            });
            out.extend_from_slice(&replacement);
        } else if token == audio_token {
            let input = audio_inputs[audio_index].clone();
            audio_index += 1;
            let replacement = config
                .audio_replacement_tokens(input.clone())
                .map_err(gemma4_unified_error_response)?;
            let soft = config
                .audio_soft_tokens(input)
                .map_err(gemma4_unified_error_response)?;
            spans.audio.push(Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Audio,
                placeholder_index: token_index,
                replacement_start: out.len(),
                soft_token_count: soft,
                replacement_token_count: replacement.len() as u32,
            });
            out.extend_from_slice(&replacement);
        } else if video_token != 0 && token == video_token {
            let input = video_inputs[video_index].clone();
            video_index += 1;
            let (replacement, relative_ranges) = config
                .video_replacement_tokens_with_ranges(input)
                .map_err(gemma4_unified_error_response)?;
            let replacement_start = out.len();
            let soft = relative_ranges
                .iter()
                .map(|range| range.soft_token_count)
                .sum();
            // The ranges are relative to the replacement block; offset them onto
            // the final stream.
            let ranges = relative_ranges
                .into_iter()
                .map(|range| Gemma4UnifiedSoftTokenRange {
                    start: replacement_start + range.start,
                    soft_token_count: range.soft_token_count,
                })
                .collect();
            spans.video.push((
                Gemma4UnifiedTokenSpan {
                    modality: Gemma4UnifiedModality::Video,
                    placeholder_index: token_index,
                    replacement_start,
                    soft_token_count: soft,
                    replacement_token_count: replacement.len() as u32,
                },
                ranges,
            ));
            out.extend_from_slice(&replacement);
        } else {
            out.push(token);
        }
    }

    Ok(ExpandedMedia { tokens: out, spans })
}

fn build_runtime_inputs(
    images: Vec<PreprocessedImage>,
    audios: Vec<PreprocessedAudio>,
    videos: Vec<PreprocessedVideo>,
    spans: MediaSpans,
) -> Gemma4UnifiedRuntimeInputs {
    let images = images
        .into_iter()
        .zip(spans.image)
        .map(|(image, span)| Gemma4UnifiedImageRuntimeInput {
            span,
            pixel_values: image.pixel_values,
            pixel_position_ids: image.pixel_position_ids,
        })
        .collect();
    let audios = audios
        .into_iter()
        .zip(spans.audio)
        .map(|(audio, span)| Gemma4UnifiedAudioRuntimeInput {
            span,
            input_features: audio.input_features,
            frame_count: audio.frame_count,
            feature_count: audio.feature_count,
        })
        .collect();
    let videos = videos
        .into_iter()
        .zip(spans.video)
        .map(
            |(video, (span, soft_token_ranges))| Gemma4UnifiedVideoRuntimeInput {
                span,
                soft_token_ranges,
                pixel_values: video.pixel_values,
                pixel_position_ids: video.pixel_position_ids,
                frame_count: video.frame_count,
            },
        )
        .collect();
    Gemma4UnifiedRuntimeInputs {
        images,
        audios,
        videos,
    }
}

fn media_payload_missing(media: &str, field: &str) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!("{media} chat content part is missing a {field} payload"),
    )
}

fn placeholder_count_error(media: &str, expected: usize, actual: usize) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!(
            "expected {expected} {media} placeholder token(s) after tokenization, found {actual}; the {media} placeholder may not round-trip through this tokenizer"
        ),
    )
}

fn media_error_response(error: MediaError) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        error.to_string(),
    )
}

fn gemma4_unified_error_response(error: Gemma4UnifiedError) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        error.to_string(),
    )
}

#[cfg(test)]
mod media_tests {
    use super::*;
    use serde_json::json;

    fn processor_config() -> Gemma4UnifiedProcessorConfig {
        let model = json!({
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "video_token_id": 258884,
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
            "boa_token_id": 256000,
            "eoa_token_index": 258883,
            "vision_config": {
                "patch_size": 16,
                "model_patch_size": 48,
                "pooling_kernel_size": 3,
                "num_soft_tokens": 280
            }
        });
        let processor = json!({
            "image_processor": {
                "patch_size": 16,
                "model_patch_size": 48,
                "pooling_kernel_size": 3,
                "max_soft_tokens": 280
            },
            "feature_extractor": {"sampling_rate": 16000},
            "audio_seq_length": 1500
        });
        Gemma4UnifiedProcessorConfig::from_model_and_processor_config(&model, &processor)
            .expect("processor config should parse")
    }

    #[test]
    fn single_pass_keeps_audio_span_aligned_after_image_expansion() {
        let config = processor_config();
        // text, <image>, text, <audio>, text
        let tokens = [10u32, 258880, 11, 258881, 12];
        let images = [Gemma4UnifiedImageInput {
            width: 224,
            height: 224,
        }];
        let audios = [Gemma4UnifiedAudioInput { sample_count: 1600 }];

        let expanded = expand_media(&config, &tokens, &images, &audios, &[]).expect("expansion");
        let image_spans = &expanded.spans.image;
        let audio_spans = &expanded.spans.audio;

        assert_eq!(image_spans.len(), 1);
        assert_eq!(audio_spans.len(), 1);

        // Image: boi + 256 soft + eoi = 258 tokens, starting right after token 10.
        let image = &image_spans[0];
        assert_eq!(image.replacement_start, 1);
        assert_eq!(image.soft_token_count, 256);
        assert_eq!(image.replacement_token_count, 258);

        // Audio span must account for the image expansion shift: 1 (token 10)
        // + 258 (image replacement) + 1 (token 11) = 260.
        let audio = &audio_spans[0];
        assert_eq!(audio.replacement_start, 260);
        assert_eq!(audio.soft_token_count, 3); // ceil(1600 / 640)
        assert_eq!(audio.replacement_token_count, 5); // boa + 3 soft + eoa

        // Total length: 3 plain tokens + 258 + 5 replacements.
        assert_eq!(expanded.tokens.len(), 3 + 258 + 5);
        // Boundary tokens landed where the spans say they did.
        assert_eq!(expanded.tokens[image.replacement_start], 255999); // boi
        assert_eq!(expanded.tokens[audio.replacement_start], 256000); // boa
    }

    #[test]
    fn video_expansion_aligns_per_frame_ranges() {
        let config = processor_config();
        // text, <video>, text
        let tokens = [10u32, 258884, 11];
        let videos = [Gemma4UnifiedVideoInput {
            frame_count: 2,
            soft_tokens_per_frame: 4,
            timestamp_token_ids_per_frame: Vec::new(),
        }];

        let expanded = expand_media(&config, &tokens, &[], &[], &videos).expect("expansion");
        assert_eq!(expanded.spans.video.len(), 1);
        let (span, ranges) = &expanded.spans.video[0];

        // Each frame contributes boi + 4 soft + eoi = 6 tokens; two frames = 12.
        assert_eq!(span.replacement_start, 1);
        assert_eq!(span.soft_token_count, 8); // 2 frames * 4
        assert_eq!(span.replacement_token_count, 12);
        assert_eq!(ranges.len(), 2);

        // Frame ranges point at the soft tokens (after each boi) in the final
        // stream: frame 0 at 1+1=2, frame 1 at 1+6+1=8.
        assert_eq!(ranges[0].start, 2);
        assert_eq!(ranges[0].soft_token_count, 4);
        assert_eq!(ranges[1].start, 8);
        assert_eq!(ranges[1].soft_token_count, 4);
        assert_eq!(expanded.tokens[span.replacement_start], 255999); // boi of frame 0
        for range in ranges {
            assert_eq!(expanded.tokens[range.start], 258884); // video soft token
        }
    }

    #[test]
    fn rejects_placeholder_count_mismatch() {
        let config = processor_config();
        let tokens = [10u32, 258880, 11]; // one image marker
        let error = expand_media(
            &config,
            &tokens,
            &[], // but zero image inputs
            &[],
            &[],
        )
        .expect_err("mismatch should fail");
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(error.1.error.message.contains("image placeholder"));
    }
}
