use std::collections::BTreeMap;
use std::path::Path;

use ax_engine_sdk::{
    EngineTokenizer, Gemma4UnifiedAudioInput, Gemma4UnifiedAudioRuntimeInput, Gemma4UnifiedError,
    Gemma4UnifiedImageInput, Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedModality,
    Gemma4UnifiedProcessorConfig, Gemma4UnifiedRuntimeInputs, Gemma4UnifiedSoftTokenRange,
    Gemma4UnifiedTokenSpan, Gemma4UnifiedVideoInput, Gemma4UnifiedVideoRuntimeInput,
    LlamaCppChatMessage, MlxLmChatMessage, Qwen3VlImageRuntimeInput, Qwen3VlRuntimeInputs,
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
use crate::openai::tool_names;

type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);
type ChatMessagePairs = Vec<(String, String)>;

/// A native-MLX chat prompt with Gemma 4 unified media already expanded into
/// soft-token spans and preprocessed tensors.
pub(crate) struct Gemma4UnifiedChatPrompt {
    pub(crate) input_tokens: Vec<u32>,
    pub(crate) runtime_inputs: Gemma4UnifiedRuntimeInputs,
}

/// Native-MLX Qwen3-VL chat prompt: placeholder-expanded tokens + patch tensors
/// for LLaVA-style scatter into the certified qwen3 text graph (WS-V2).
pub(crate) struct Qwen3VlChatPrompt {
    pub(crate) input_tokens: Vec<u32>,
    pub(crate) runtime_inputs: Qwen3VlRuntimeInputs,
}

/// True when the model id / family label is a Qwen3-VL variant.
pub(crate) fn is_qwen3_vl_model_id(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    lower.contains("qwen3-vl")
        || lower.contains("qwen3_vl")
        || lower.contains("qwen3vl")
        || lower.contains("qwen3-vl-moe")
        || lower.contains("qwen3_vl_moe")
}

/// Default Qwen3-VL ViT geometry (common HF defaults; overridable via config later).
const QWEN3_VL_DEFAULT_PATCH_SIZE: u32 = 14;
const QWEN3_VL_DEFAULT_SPATIAL_MERGE: u32 = 2;
const QWEN3_VL_DEFAULT_MAX_SOFT: u32 = 1024;
/// Placeholder text embedded in the rendered prompt for each image; tokenized
/// then expanded to soft-token count in the token stream.
const QWEN3_VL_IMAGE_PLACEHOLDER: &str = "<|image_pad|>";

/// Options for OpenAI chat prompt rendering beyond tools/messages.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ChatPromptRenderOptions {
    /// When true, Gemma 4 injects `<|think|>` and leaves the thought channel
    /// open for generation (Google `enable_thinking=true`). Default false
    /// pre-fills an empty thought channel so short answers skip CoT.
    pub(crate) enable_thinking: bool,
}

/// Tools-free convenience wrapper for tests that don't care about tool
/// rendering.
#[cfg(test)]
pub(crate) fn render_openai_chat_prompt(
    model_id: &str,
    messages: &[OpenAiChatMessage],
) -> Result<String, HttpErrorResponse> {
    render_openai_chat_prompt_with_tools(model_id, messages, None, None)
}

/// Test convenience wrapper for the default prompt-rendering options.
#[cfg(test)]
pub(crate) fn render_openai_chat_prompt_with_tools(
    model_id: &str,
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
) -> Result<String, HttpErrorResponse> {
    render_openai_chat_prompt_with_options(
        model_id,
        messages,
        tools,
        tool_choice,
        ChatPromptRenderOptions::default(),
    )
}

pub(crate) fn render_openai_chat_prompt_with_options(
    model_id: &str,
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    options: ChatPromptRenderOptions,
) -> Result<String, HttpErrorResponse> {
    validate_openai_tool_names(messages, tools, tool_choice)?;
    // Always use the dedicated Gemma 4 renderer so multi-turn history,
    // tool-call turn continuation, and enable_thinking stay aligned with
    // Google's 2026-07-09 canonical chat template (not only when tools are
    // present).
    if matches!(
        chat::ChatPromptTemplate::for_model_id(model_id),
        chat::ChatPromptTemplate::Gemma4
    ) {
        return render_gemma4_openai_chat_prompt(
            model_id,
            messages,
            tools,
            options.enable_thinking,
        );
    }
    if matches!(
        chat::ChatPromptTemplate::for_model_id(model_id),
        chat::ChatPromptTemplate::Glm47
    ) {
        return render_glm_openai_chat_prompt(model_id, messages, tools, tool_choice);
    }
    let template = chat::ChatPromptTemplate::for_model_id(model_id);
    // Always use the dedicated Ministral renderer: structured tool-call history
    // can be present even when this request does not redeclare the tool schemas.
    if matches!(template, chat::ChatPromptTemplate::MinistralInstruct) {
        return render_ministral_openai_chat_prompt(messages, tools);
    }
    // Llama 4 hub: inject function schemas into the first user turn (not a
    // Qwen-style system tools block).
    if matches!(template, chat::ChatPromptTemplate::Llama4)
        && tools.map(openai_value_is_present).unwrap_or(false)
    {
        return render_llama4_openai_chat_prompt(messages, tools);
    }
    // Llama 3 hub: tools go into the first user turn with a short JSON-call
    // instruction (AX still omits the default knowledge-date system preamble).
    if matches!(template, chat::ChatPromptTemplate::Llama3)
        && tools.map(openai_value_is_present).unwrap_or(false)
    {
        return render_llama3_openai_chat_prompt(messages, tools);
    }
    let tool_contract_style = qwen_tool_contract_style(model_id);
    let rendered_messages = render_openai_chat_message_pairs(messages, tool_contract_style)?;
    let rendered_messages =
        prepend_qwen_tool_contract(model_id, rendered_messages, tools, tool_choice);
    chat::render_prompt(model_id, &rendered_messages).map_err(chat_error_response)
}

/// Render an OpenAI chat request for the GLM 4.x family, including native GLM
/// tool calling. GLM declares tool signatures in a leading `<|system|>` block
/// and emits/consumes calls as
/// `<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`
/// (see the model's `chat_template.jinja`). String argument values are emitted
/// raw; non-string values are JSON-encoded. Used for every GLM chat request:
/// with no tools the contract is omitted, so output matches the plain GLM chat
/// rendering.
fn render_glm_openai_chat_prompt(
    model_id: &str,
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
) -> Result<String, HttpErrorResponse> {
    let rendered_messages = render_glm_chat_message_pairs(messages)?;
    let rendered_messages = prepend_glm_tool_contract(rendered_messages, tools, tool_choice);
    chat::render_prompt(model_id, &rendered_messages).map_err(chat_error_response)
}

/// Ministral hub template (exact marker spacing from chat_template.jinja):
/// - `[AVAILABLE_TOOLS][{…}][/AVAILABLE_TOOLS]` only before the **last** user turn
/// - assistant tool calls: `[TOOL_CALLS][{…, "id": "XXXXXXXXX"}]</s>`
/// - tool results: `[TOOL_RESULTS]{"content": …, "call_id": "XXXXXXXXX"}[/TOOL_RESULTS]`
/// - system folds into `[INST]` only when the final message is a user
/// - plain assistant is `content</s>`
fn render_ministral_openai_chat_prompt(
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
) -> Result<String, HttpErrorResponse> {
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }
    let tools = tools.filter(|value| openai_value_is_present(value));
    let mut loop_start = 0usize;
    let mut system: Option<String> = None;
    if let Some(first) = messages.first()
        && chat::normalize_role(&first.role).map_err(chat_error_response)? == "system"
    {
        system = Some(render_openai_chat_content(first.content.as_ref())?);
        loop_start = 1;
    }

    // Match the hub template's validation pass. Tool calls/results do not
    // advance the conversational parity; all other turns must start with a
    // user and alternate with assistants.
    let mut expects_user = true;
    for message in &messages[loop_start..] {
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        let has_tool_calls = message
            .tool_calls
            .as_ref()
            .is_some_and(openai_value_is_present);
        if has_tool_calls && role != "assistant" {
            return Err(chat_error_response(
                "Ministral tool_calls are only valid on assistant messages".to_string(),
            ));
        }
        if matches!(role, "tool" | "function") || has_tool_calls {
            continue;
        }
        if !matches!(role, "user" | "assistant") || (role == "user") != expects_user {
            return Err(chat_error_response(
                "After the optional system message, Ministral conversation roles must alternate \
                 user/assistant/user/assistant/..."
                    .to_string(),
            ));
        }
        expects_user = !expects_user;
    }

    let last_user_idx = messages
        .iter()
        .rposition(|message| matches!(chat::normalize_role(&message.role), Ok("user")));
    let mut prompt = String::from("<s>");
    for (index, message) in messages.iter().enumerate().skip(loop_start) {
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        let has_tool_calls = message
            .tool_calls
            .as_ref()
            .is_some_and(openai_value_is_present);

        if has_tool_calls && role == "assistant" {
            // Hub: `[TOOL_CALLS][` + objects + `]` + eos
            prompt.push_str("[TOOL_CALLS]");
            prompt.push_str(&ministral_tool_calls_array(message.tool_calls.as_ref())?);
            prompt.push_str("</s>");
            continue;
        }

        match role {
            "user" => {
                let content = render_openai_chat_content(message.content.as_ref())?;
                // AVAILABLE_TOOLS only on the last user message (hub condition).
                if Some(index) == last_user_idx
                    && let Some(tools) = tools
                {
                    // Hub: `[AVAILABLE_TOOLS][` … `]`
                    prompt.push_str("[AVAILABLE_TOOLS]");
                    prompt.push_str(&ministral_available_tools_array(tools)?);
                    prompt.push_str("[/AVAILABLE_TOOLS]");
                }
                prompt.push_str("[INST]");
                if index + 1 == messages.len()
                    && let Some(system_text) = system.take()
                {
                    prompt.push_str(&system_text);
                    prompt.push_str("\n\n");
                }
                prompt.push_str(&content);
                prompt.push_str("[/INST]");
            }
            "assistant" => {
                let content = render_openai_chat_content(message.content.as_ref())?;
                // Hub: message["content"] + eos_token (no inserted whitespace).
                prompt.push_str(&content);
                prompt.push_str("</s>");
            }
            "tool" | "function" => {
                // Hub: `[TOOL_RESULTS]{"content": ` + content + `, "call_id": "…"}[/TOOL_RESULTS]`
                let content = render_openai_chat_content(message.content.as_ref())?;
                let call_id = message
                    ._tool_call_id
                    .as_deref()
                    .filter(|id| ministral_tool_call_id_is_valid(id))
                    .ok_or_else(|| {
                        chat_error_response(
                            "Ministral tool results require tool_call_id of length 9 \
                             (hub chat_template.jinja)"
                                .to_string(),
                        )
                    })?;
                prompt.push_str("[TOOL_RESULTS]{\"content\": ");
                prompt.push_str(&content);
                prompt.push_str(", \"call_id\": \"");
                prompt.push_str(call_id);
                prompt.push_str("\"}[/TOOL_RESULTS]");
            }
            _ => {
                return Err(chat_error_response(
                    "Ministral only supports user and assistant roles, plus an optional initial \
                     system message and tool results"
                        .to_string(),
                ));
            }
        }
    }
    Ok(prompt)
}

/// Llama 4 hub template with tools:
/// - first user turn carries function schemas + instruction
/// - assistant tool_calls: `<|python_start|>…content…<|python_end|>{"name", "parameters"}`
/// - tool results: `<|header_start|>ipython` with content as JSON string
fn render_llama4_openai_chat_prompt(
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
) -> Result<String, HttpErrorResponse> {
    render_llama_family_tools_prompt(
        messages,
        tools,
        LlamaToolPromptSyntax {
            bos: "<|begin_of_text|>",
            header_open: "<|header_start|>",
            header_close: "<|header_end|>\n\n",
            eot: "<|eot|>",
            environment_ipython: false,
            python_tool_call_wrappers: true,
        },
    )
}

/// Llama 3 hub tools path (AX omits the default knowledge-date system).
fn render_llama3_openai_chat_prompt(
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
) -> Result<String, HttpErrorResponse> {
    render_llama_family_tools_prompt(
        messages,
        tools,
        LlamaToolPromptSyntax {
            bos: "<|begin_of_text|>",
            header_open: "<|start_header_id|>",
            header_close: "<|end_header_id|>\n\n",
            eot: "<|eot_id|>",
            environment_ipython: true,
            python_tool_call_wrappers: false,
        },
    )
}

struct LlamaToolPromptSyntax<'a> {
    bos: &'a str,
    header_open: &'a str,
    header_close: &'a str,
    eot: &'a str,
    environment_ipython: bool,
    python_tool_call_wrappers: bool,
}

fn render_llama_family_tools_prompt(
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
    syntax: LlamaToolPromptSyntax<'_>,
) -> Result<String, HttpErrorResponse> {
    let LlamaToolPromptSyntax {
        bos,
        header_open,
        header_close,
        eot,
        environment_ipython,
        python_tool_call_wrappers,
    } = syntax;
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }
    let tools = tools.filter(|value| openai_value_is_present(value));
    let mut prompt = String::from(bos);
    let mut index = 0usize;
    let mut system_text = String::new();
    if let Some(first) = messages.first() {
        let role = chat::normalize_role(&first.role).map_err(chat_error_response)?;
        if role == "system" {
            system_text = render_openai_chat_content(first.content.as_ref())?;
            index = 1;
        }
    }

    if !system_text.trim().is_empty() {
        prompt.push_str(header_open);
        prompt.push_str("system");
        prompt.push_str(header_close);
        if tools.is_some() {
            prompt.push_str("Environment: ipython\n");
        }
        prompt.push_str(system_text.trim());
        prompt.push_str(eot);
    } else if tools.is_some() && environment_ipython {
        // Llama 3 always emits a system block with Environment: ipython when tools are set.
        prompt.push_str(header_open);
        prompt.push_str("system");
        prompt.push_str(header_close);
        prompt.push_str("Environment: ipython\n");
        prompt.push_str(eot);
    }

    let mut injected_tools = false;
    while index < messages.len() {
        let message = &messages[index];
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        let has_tool_calls = message
            .tool_calls
            .as_ref()
            .is_some_and(openai_value_is_present);

        if role == "tool" || role == "function" {
            // Hub: <|header_start|>ipython … content|tojson … <|eot|>
            let content = render_openai_chat_content(message.content.as_ref())?;
            prompt.push_str(header_open);
            prompt.push_str("ipython");
            prompt.push_str(header_close);
            // Jinja `content | tojson` for strings yields a quoted JSON string.
            prompt.push_str(&serde_json::to_string(&content).unwrap_or_else(|_| {
                format!("\"{}\"", content.replace('\\', "\\\\").replace('"', "\\\""))
            }));
            prompt.push_str(eot);
            index += 1;
            continue;
        }

        if role == "assistant" && has_tool_calls {
            let content = render_openai_chat_content(message.content.as_ref())?;
            let calls = llama_tool_call_payloads(message.tool_calls.as_ref())?;
            if !python_tool_call_wrappers && calls.len() != 1 {
                return Err(chat_error_response(
                    "Llama 3 supports exactly one tool call per assistant message \
                     (hub chat_template.jinja)"
                        .to_string(),
                ));
            }
            prompt.push_str(header_open);
            prompt.push_str("assistant");
            prompt.push_str(header_close);
            if python_tool_call_wrappers {
                // Hub Llama 4:
                // <|python_start|>{content}<|python_end|>{"name":…,"parameters":…}<|eot|>
                prompt.push_str("<|python_start|>");
                if !content.trim().is_empty() {
                    prompt.push_str(&content);
                }
                prompt.push_str("<|python_end|>");
            }
            for call in calls {
                prompt.push_str(&call);
            }
            prompt.push_str(eot);
            index += 1;
            continue;
        }

        if role == "user" && !injected_tools && tools.is_some() {
            let content = render_openai_chat_content(message.content.as_ref())?;
            prompt.push_str(header_open);
            prompt.push_str("user");
            prompt.push_str(header_close);
            prompt.push_str(
                "Given the following functions, please respond with a JSON for a function call \
                 with its proper arguments that best answers the given prompt.\n\n\
                 Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.\
                 Do not use variables.\n\n",
            );
            if let Some(tools) = tools {
                for tool_json in pretty_tool_json_lines(tools) {
                    prompt.push_str(&tool_json);
                    prompt.push_str("\n\n");
                }
            }
            prompt.push_str(content.trim());
            prompt.push_str(eot);
            injected_tools = true;
            index += 1;
            continue;
        }

        // Regular user/assistant without tool surface.
        let content = render_openai_chat_content(message.content.as_ref())?;
        let header_role = match role {
            "assistant" => "assistant",
            "system" => "system",
            _ => "user",
        };
        prompt.push_str(header_open);
        prompt.push_str(header_role);
        prompt.push_str(header_close);
        if !content.trim().is_empty() {
            prompt.push_str(content.trim());
        }
        prompt.push_str(eot);
        index += 1;
    }

    prompt.push_str(header_open);
    prompt.push_str("assistant");
    prompt.push_str(header_close);
    Ok(prompt)
}

/// Hub Ministral AVAILABLE_TOOLS array, including its JSON whitespace contract.
fn ministral_available_tools_array(tools: &Value) -> Result<String, HttpErrorResponse> {
    let items = match tools {
        Value::Array(items) => items.as_slice(),
        Value::Object(_) => std::slice::from_ref(tools),
        _ => {
            return Err(chat_error_response(
                "Ministral tools must be an object or array".to_string(),
            ));
        }
    };
    let mut rendered = Vec::with_capacity(items.len());
    for (index, tool) in items.iter().enumerate() {
        let function = tool
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                chat_error_response(format!(
                    "Ministral tools[{index}] requires a function object"
                ))
            })?;
        let fields = function
            .iter()
            .filter(|(key, _)| key.as_str() != "return")
            .map(|(key, value)| {
                format!(
                    "{}: {}",
                    jinja_json(&Value::String(key.clone())),
                    jinja_json(value)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        rendered.push(format!(
            "{{\"type\": \"function\", \"function\": {{{fields}}}}}"
        ));
    }
    Ok(format!("[{}]", rendered.join(", ")))
}

/// Hub: `[TOOL_CALLS][{function_json without trailing }, "id": "XXXXXXXXX"}]`
fn ministral_tool_calls_array(tool_calls: Option<&Value>) -> Result<String, HttpErrorResponse> {
    let Some(tool_calls) = tool_calls else {
        return Ok("[]".to_string());
    };
    let calls = match tool_calls {
        Value::Array(calls) => calls.as_slice(),
        Value::Object(_) => std::slice::from_ref(tool_calls),
        _ => {
            return Err(chat_error_response(
                "Ministral tool_calls must be an object or array".to_string(),
            ));
        }
    };
    let mut parts = Vec::new();
    for call in calls {
        let object = call.as_object().ok_or_else(|| {
            chat_error_response("Ministral tool_calls entries must be objects".to_string())
        })?;
        let function = object
            .get("function")
            .filter(|value| value.is_object())
            .ok_or_else(|| {
                chat_error_response(
                    "Ministral tool_calls entries require a function object".to_string(),
                )
            })?;
        let id = object
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| ministral_tool_call_id_is_valid(id))
            .ok_or_else(|| {
                chat_error_response(
                    "Ministral tool call ids must be alphanumeric length 9 \
                     (hub chat_template.jinja)"
                        .to_string(),
                )
            })?;
        // function|tojson without final `}`, then append `, "id": "…"}`
        let function_json = jinja_json(function);
        let without_close = function_json
            .strip_suffix('}')
            .unwrap_or(function_json.as_str());
        parts.push(format!("{without_close}, \"id\": \"{id}\"}}"));
    }
    Ok(format!("[{}]", parts.join(", ")))
}

fn ministral_tool_call_id_is_valid(id: &str) -> bool {
    id.len() == 9 && id.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn pretty_tool_json_lines(tools: &Value) -> Vec<String> {
    match tools {
        Value::Array(items) => items
            .iter()
            .map(|item| serde_json::to_string_pretty(item).unwrap_or_else(|_| compact_json(item)))
            .collect(),
        other => {
            vec![serde_json::to_string_pretty(other).unwrap_or_else(|_| compact_json(other))]
        }
    }
}

/// Hub Llama tool-call payload: `{"name": "…", "parameters": {…}}` (parameters, not arguments).
fn llama_tool_call_payloads(tool_calls: Option<&Value>) -> Result<Vec<String>, HttpErrorResponse> {
    let tool_calls = tool_calls.ok_or_else(|| {
        chat_error_response("Llama assistant tool_calls must be present".to_string())
    })?;
    let calls = match tool_calls {
        Value::Array(calls) => calls.as_slice(),
        Value::Object(_) => std::slice::from_ref(tool_calls),
        _ => {
            return Err(chat_error_response(
                "Llama assistant tool_calls must be an object or array".to_string(),
            ));
        }
    };
    let mut payloads = Vec::with_capacity(calls.len());
    for (index, call) in calls.iter().enumerate() {
        let function = call
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                chat_error_response(format!(
                    "Llama assistant tool_calls[{index}] requires a function object"
                ))
            })?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                chat_error_response(format!(
                    "Llama assistant tool_calls[{index}].function requires a name"
                ))
            })?;
        let arguments = normalize_tool_arguments(function.get("arguments"));
        payloads.push(format!(
            "{{\"name\": {}, \"parameters\": {}}}",
            jinja_json(&Value::String(name.to_string())),
            jinja_json(&arguments)
        ));
    }
    Ok(payloads)
}

fn render_glm_chat_message_pairs(
    messages: &[OpenAiChatMessage],
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
                && let Some(rendered_tool_calls) = render_glm_assistant_tool_calls(tool_calls)
            {
                content.push_str(&rendered_tool_calls);
            }
            Ok((role.to_string(), content))
        })
        .collect()
}

fn prepend_glm_tool_contract(
    mut messages: ChatMessagePairs,
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
) -> ChatMessagePairs {
    let Some(contract) = render_glm_tool_contract_system_message(tools, tool_choice) else {
        return messages;
    };
    // GLM emits a dedicated leading `<|system|># Tools` block, separate from any
    // user-provided system message, so insert it at the front rather than
    // merging into an existing system turn.
    messages.insert(0, ("system".to_string(), contract));
    messages
}

fn render_glm_tool_contract_system_message(
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
) -> Option<String> {
    let tools = tools?;
    if !openai_value_is_present(tools) {
        return None;
    }
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
         For each function call, output the function name and arguments within the following XML format:\n\
         <tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>",
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

fn render_glm_assistant_tool_calls(value: &Value) -> Option<String> {
    let calls = match value {
        Value::Array(calls) => calls.as_slice(),
        Value::Object(_) => std::slice::from_ref(value),
        _ => return None,
    };
    let rendered = calls
        .iter()
        .filter_map(render_glm_assistant_tool_call)
        .collect::<Vec<_>>();
    (!rendered.is_empty()).then(|| rendered.join(""))
}

fn render_glm_assistant_tool_call(value: &Value) -> Option<String> {
    let object = value.as_object()?;
    let function = object.get("function").and_then(Value::as_object)?;
    let name = function.get("name")?.as_str()?;
    let arguments = normalize_tool_arguments(function.get("arguments"));
    Some(render_glm_xml_tool_call(name, &arguments))
}

/// Render a single GLM `<tool_call>` block. Mirrors the model's
/// `chat_template.jinja`: string values are emitted verbatim, non-string values
/// are JSON-encoded.
fn render_glm_xml_tool_call(name: &str, arguments: &Value) -> String {
    let mut rendered = String::from("<tool_call>");
    rendered.push_str(name);
    if let Some(args) = arguments.as_object() {
        for (key, value) in args {
            rendered.push_str("<arg_key>");
            rendered.push_str(key);
            rendered.push_str("</arg_key><arg_value>");
            rendered.push_str(&glm_tool_arg_value(value));
            rendered.push_str("</arg_value>");
        }
    }
    rendered.push_str("</tool_call>");
    rendered
}

fn glm_tool_arg_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        other => compact_json(other),
    }
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

fn validate_openai_tool_names(
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
) -> Result<(), HttpErrorResponse> {
    if let Some(tools) = tools {
        for (index, tool) in openai_tool_items(tools).iter().enumerate() {
            if let Some(function) = openai_tool_function(tool)
                && let Some(name) = function.get("name").and_then(Value::as_str)
            {
                validate_openai_tool_name(name, format!("tools[{index}].function.name"))?;
            }
        }
    }

    if let Some(name) = forced_tool_choice_name(tool_choice) {
        validate_openai_tool_name(name, "tool_choice.function.name")?;
    }

    for (message_index, message) in messages.iter().enumerate() {
        if matches!(message.role.as_str(), "tool" | "function")
            && let Some(name) = message._name.as_deref()
        {
            validate_openai_tool_name(name, format!("messages[{message_index}].name"))?;
        }
        if let Some(tool_calls) = message.tool_calls.as_ref() {
            for (call_index, call) in openai_tool_items(tool_calls).iter().enumerate() {
                let Some(function) = call.get("function").and_then(Value::as_object) else {
                    continue;
                };
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    validate_openai_tool_name(
                        name,
                        format!("messages[{message_index}].tool_calls[{call_index}].function.name"),
                    )?;
                }
            }
        }
    }

    Ok(())
}

fn openai_tool_items(value: &Value) -> Vec<&Value> {
    match value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(_) => vec![value],
        _ => Vec::new(),
    }
}

fn forced_tool_choice_name(tool_choice: Option<&Value>) -> Option<&str> {
    tool_choice?
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
}

fn validate_openai_tool_name(name: &str, field: impl AsRef<str>) -> Result<(), HttpErrorResponse> {
    if tool_names::is_valid(name) {
        return Ok(());
    }
    Err(error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!(
            "{} must be a non-empty tool identifier containing only ASCII letters, digits, '_', '-', or '.'",
            field.as_ref()
        ),
    ))
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
                    OpenAiChatContentPartKind::Video => {
                        return Err(unsupported_video_error(&part.part_type));
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

/// Tracked message kind for Gemma 4 generation-prompt decisions.
///
/// Mirrors the official template's `prev_message_type` used to decide whether
/// to open a new model turn, leave a bare `<|tool_response>`, or open a
/// thinking channel after a tool result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Gemma4PrevMessageType {
    ToolCall,
    ToolResponse,
}

fn render_gemma4_openai_chat_prompt(
    model_id: &str,
    messages: &[OpenAiChatMessage],
    tools: Option<&Value>,
    enable_thinking: bool,
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

    // Official template: system/tools/thinking share the first system turn.
    if enable_thinking || has_tools || matches!(first_role, "system" | "developer") {
        prompt.push_str("<|turn>system\n");
        if enable_thinking {
            prompt.push_str("<|think|>\n");
        }
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

    // prev_message_type drives generation prompt; prev_non_tool_role detects
    // consecutive assistant messages that must stay in one model turn
    // (tool-call → tool-response → answer).
    let mut prev_message_type: Option<Gemma4PrevMessageType> = None;
    let mut prev_non_tool_role: Option<&'static str> = None;

    while index < messages.len() {
        let message = &messages[index];
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        if matches!(role, "tool" | "function") {
            // Tool messages are consumed as followers of the preceding
            // assistant tool_calls; standalone tool rows are skipped.
            index += 1;
            continue;
        }

        // Official template resets prev_message_type at each non-tool message.
        prev_message_type = None;
        let turn = if role == "assistant" { "model" } else { role };
        let continue_same_model_turn =
            role == "assistant" && prev_non_tool_role == Some("assistant");
        if !continue_same_model_turn {
            prompt.push_str("<|turn>");
            prompt.push_str(turn);
            prompt.push('\n');
        }

        let mut rendered_tool_call = false;
        let mut tool_response_count = 0usize;
        let mut next_index = index + 1;

        if role == "assistant" {
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
                    prev_message_type = Some(Gemma4PrevMessageType::ToolCall);
                }
            }

            let (response_count, after_tools) = render_following_gemma4_tool_responses(
                messages,
                index + 1,
                &tool_names_by_id,
                &mut prompt,
            )?;
            tool_response_count = response_count;
            next_index = after_tools;
            if tool_response_count > 0 {
                prev_message_type = Some(Gemma4PrevMessageType::ToolResponse);
            }
        }

        let content = render_openai_chat_content(message.content.as_ref())?;
        let content_for_prompt = if role == "assistant" {
            chat::strip_gemma4_thinking_from_history(&content)
        } else {
            content.trim().to_string()
        };
        let has_content = !content_for_prompt.is_empty();
        if has_content {
            prompt.push_str(&content_for_prompt);
        }

        let next_non_tool_role = next_non_tool_role(messages, next_index);
        // Keep the model turn open when another assistant message follows
        // (agent loop: call → response → final answer in one turn).
        let continues_into_next = role == "assistant"
            && next_non_tool_role == Some("assistant")
            && (!rendered_tool_call || tool_response_count > 0);

        if rendered_tool_call && tool_response_count == 0 {
            // Pending tool call: open the response slot for the client/model.
            prompt.push_str("<|tool_response>");
            prev_message_type = Some(Gemma4PrevMessageType::ToolCall);
        } else if continues_into_next {
            // Do not close the model turn yet.
        } else if !(tool_response_count > 0 && !has_content && next_non_tool_role.is_none()) {
            // Close the turn unless this is a terminal tool_response with no
            // answer content (generation continues in-turn after the result).
            prompt.push_str("<turn|>\n");
        }

        prev_non_tool_role = Some(role);
        index = next_index;
    }

    // Generation prompt — matches official add_generation_prompt block.
    match prev_message_type {
        Some(Gemma4PrevMessageType::ToolCall) => {
            // Already opened <|tool_response>; model continues there.
        }
        Some(Gemma4PrevMessageType::ToolResponse) => {
            if enable_thinking {
                prompt.push_str("<|channel>thought\n");
            }
            // Thinking off: continue generation immediately after the response
            // (no new model turn, no empty thought prefill).
        }
        None => {
            prompt.push_str("<|turn>model\n");
            // Gemma 4 IT: empty thought prefill when thinking is off.
            // DiffusionGemma hub template opens the model turn only.
            if !enable_thinking && !chat::is_diffusion_gemma(model_id) {
                prompt.push_str("<|channel>thought\n<channel|>");
            }
        }
    }
    Ok(prompt)
}

fn next_non_tool_role(messages: &[OpenAiChatMessage], start: usize) -> Option<&'static str> {
    for message in messages.iter().skip(start) {
        let Ok(role) = chat::normalize_role(&message.role) else {
            continue;
        };
        if !matches!(role, "tool" | "function") {
            return Some(role);
        }
    }
    None
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
            // Match official template spacing: properties:{...},
            rendered.push_str("properties:{");
            rendered.push_str(&format_gemma4_parameters(properties));
            rendered.push_str("},");
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
    // Official template iterates every property name (dictsort); parameter
    // names that collide with schema keywords are still valid tool params.
    let mut rendered = Vec::new();
    let mut keys = properties.keys().collect::<Vec<_>>();
    keys.sort();
    for key in keys {
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
        let type_upper = value
            .get("type")
            .and_then(Value::as_str)
            .map(|t| t.to_ascii_uppercase());
        if type_upper.as_deref() == Some("STRING")
            && let Some(enum_values) = value.get("enum")
        {
            fields.push(format!(
                "enum:{}",
                format_gemma4_argument(enum_values, true)
            ));
        }
        if type_upper.as_deref() == Some("ARRAY")
            && let Some(items) = value.get("items")
        {
            // Official template renders nested items objects with typed
            // fields; fall back to argument formatting for simple items.
            if let Some(items_obj) = items.as_object() {
                fields.push(format!(
                    "items:{{{}}}",
                    format_gemma4_items_object(items_obj)
                ));
            } else {
                fields.push(format!("items:{}", format_gemma4_argument(items, false)));
            }
        }
        if value.get("nullable").and_then(Value::as_bool) == Some(true) {
            fields.push("nullable:true".to_string());
        }
        if type_upper.as_deref() == Some("OBJECT") {
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

fn format_gemma4_items_object(items: &serde_json::Map<String, Value>) -> String {
    let mut fields = Vec::new();
    let mut keys = items.keys().collect::<Vec<_>>();
    keys.sort();
    for key in keys {
        let Some(value) = items.get(key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        match key.as_str() {
            "properties" => {
                if let Some(nested) = value.as_object() {
                    fields.push(format!(
                        "properties:{{{}}}",
                        format_gemma4_parameters(nested)
                    ));
                }
            }
            "required" => {
                fields.push(format!("required:{}", format_gemma4_argument(value, true)));
            }
            "type" => {
                fields.push(format!("type:{}", format_gemma4_upper_type_argument(value)));
            }
            _ => {
                fields.push(format!("{key}:{}", format_gemma4_argument(value, true)));
            }
        }
    }
    fields.join(",")
}

// strip_gemma4_thinking_from_history lives in chat.rs (canonical channel
// markers) and is used for both the dedicated Gemma renderer and the plain
// chat.rs Gemma4 history path.

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
    let normalized = chat::normalize_model_id_token(model_id);
    if chat::uses_qwen_coder_xml_tool_contract(model_id) {
        // Coder-Next: XML tool *declarations* + function= call format.
        QwenToolContractStyle::CoderXml
    } else if normalized.contains("qwen3-next")
        || normalized.contains("qwen3-5")
        || normalized.contains("qwen35")
        // Hub Qwen3.6 templates match Qwen3.5: JSON tool schemas + function= calls
        // (not Coder-Next XML declarations).
        || normalized.contains("qwen3-6")
        || normalized.contains("qwen36")
    {
        QwenToolContractStyle::FunctionXml
    } else {
        QwenToolContractStyle::JsonTools
    }
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
        Value::Number(value) => json_number_is_nonzero(value),
    }
}

fn openai_value_is_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(object) => !object.is_empty(),
        Value::Number(value) => json_number_is_nonzero(value),
    }
}

fn json_number_is_nonzero(value: &serde_json::Number) -> bool {
    value
        .as_i64()
        .map(|value| value != 0)
        .or_else(|| value.as_u64().map(|value| value != 0))
        .or_else(|| value.as_f64().map(|value| value != 0.0))
        .unwrap_or(true)
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

/// Match Python `json.dumps` spacing used by hub Jinja `tojson` filters
/// (space after `:` and `,`, object key insertion order preserved).
fn jinja_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string()),
        Value::Array(values) => {
            let body = values.iter().map(jinja_json).collect::<Vec<_>>().join(", ");
            format!("[{body}]")
        }
        Value::Object(object) => {
            let body = object
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string()),
                        jinja_json(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{body}}}")
        }
    }
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
    /// Video content part; accepted only when model capability gate allows.
    Video,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenAiChatMediaKind {
    Image,
    Audio,
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
        // Product serving scope is text + image + audio. Keep lower-level
        // video preprocessing code isolated, but reject video on the public
        // OpenAI surface instead of advertising a partially-supported path.
        "video_url" | "input_video" | "video" => OpenAiChatContentPartKind::Video,
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

fn unsupported_video_error(part_type: &str) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "unsupported_modality",
        format!(
            "OpenAI chat content part type {part_type} is not supported for this model; \
requires gemma4_unified vision capability (data URI only, no remote URLs)"
        ),
    )
}

pub(crate) fn reject_video_chat_content(
    messages: &[OpenAiChatMessage],
    video_supported: bool,
) -> Result<(), HttpErrorResponse> {
    for part in messages
        .iter()
        .filter_map(|message| match &message.content {
            Some(OpenAiChatContent::Parts(parts)) => Some(parts.as_slice()),
            Some(OpenAiChatContent::Text(_)) | None => None,
        })
        .flatten()
    {
        if matches!(
            chat_content_part_kind(part),
            OpenAiChatContentPartKind::Video
        ) && !video_supported
        {
            return Err(unsupported_video_error(&part.part_type));
        }
    }
    Ok(())
}

/// True when any message carries a supported inline media content part
/// (image/audio/video).
pub(crate) fn messages_contain_inline_media(messages: &[OpenAiChatMessage]) -> bool {
    messages.iter().any(|message| match &message.content {
        Some(OpenAiChatContent::Text(_)) | None => false,
        Some(OpenAiChatContent::Parts(parts)) => parts.iter().any(|part| {
            matches!(
                chat_content_part_kind(part),
                OpenAiChatContentPartKind::Media(_) | OpenAiChatContentPartKind::Video
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

    let tokenizer = EngineTokenizer::from_model_dir_cached(model_dir).map_err(|error| {
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

/// Build a native-MLX Qwen3-VL chat prompt from inline `image_url` parts.
///
/// Images are decoded (data URI only), patchified for the portable ViT, and
/// attached as [`Qwen3VlRuntimeInputs`]. Audio/video are rejected (out of
/// Qwen3-VL product scope for this path). Returns `Ok(None)` with no media.
pub(crate) fn render_qwen3_vl_chat_with_media(
    model_id: &str,
    model_dir: &Path,
    messages: &[OpenAiChatMessage],
) -> Result<Option<Qwen3VlChatPrompt>, HttpErrorResponse> {
    if !messages_contain_inline_media(messages) {
        return Ok(None);
    }
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }

    // Reject audio/video on the Qwen3-VL chat surface (image-only scope).
    for part in messages
        .iter()
        .filter_map(|m| match &m.content {
            Some(OpenAiChatContent::Parts(parts)) => Some(parts.as_slice()),
            _ => None,
        })
        .flatten()
    {
        match chat_content_part_kind(part) {
            OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Audio) => {
                return Err(error_response(
                    StatusCode::BAD_REQUEST,
                    "unsupported_modality",
                    "qwen3_vl chat accepts image_url media only; audio is not supported"
                        .to_string(),
                ));
            }
            OpenAiChatContentPartKind::Video => {
                return Err(error_response(
                    StatusCode::BAD_REQUEST,
                    "unsupported_modality",
                    "qwen3_vl chat accepts image_url media only; video is not supported"
                        .to_string(),
                ));
            }
            _ => {}
        }
    }

    let tokenizer = EngineTokenizer::from_model_dir_cached(model_dir).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("failed to load tokenizer for qwen3_vl multimodal chat: {error}"),
        )
    })?;

    let mut collected = CollectedMedia::default();
    let mut pairs: ChatMessagePairs = Vec::with_capacity(messages.len());
    for message in messages {
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        let content = render_content_collecting_media(
            message.content.as_ref(),
            QWEN3_VL_IMAGE_PLACEHOLDER,
            "", // audio unused (rejected above)
            &mut collected,
        )?;
        pairs.push((role.to_string(), content));
    }
    if collected.images.is_empty() {
        return Ok(None);
    }

    let prompt = chat::render_prompt(model_id, &pairs).map_err(chat_error_response)?;
    let base_tokens = tokenizer.encode(&prompt, false).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("failed to tokenize qwen3_vl chat prompt: {error}"),
        )
    })?;

    // Encode placeholder alone to locate it in the prompt token stream.
    let ph_tokens = tokenizer
        .encode(QWEN3_VL_IMAGE_PLACEHOLDER, false)
        .map_err(|error| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("failed to tokenize image placeholder: {error}"),
            )
        })?;
    if ph_tokens.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "tokenizer produced empty image placeholder tokens".to_string(),
        ));
    }

    let mut placeholder_starts = find_subsequence_starts(&base_tokens, &ph_tokens);
    if placeholder_starts.len() != collected.images.len() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!(
                "qwen3_vl image placeholders {} != collected images {}",
                placeholder_starts.len(),
                collected.images.len()
            ),
        ));
    }

    // Expand each single placeholder span into soft_token_count tokens (reuse
    // the first pad token id as filler so the sequence length matches ViT
    // soft tokens for scatter positions).
    let pad_id = ph_tokens[0];
    let mut images_rt = Vec::with_capacity(collected.images.len());
    let mut expanded = Vec::with_capacity(base_tokens.len().saturating_mul(2));
    let mut cursor = 0usize;
    // Process left-to-right; expansion shifts later indices so rebuild by walk.
    placeholder_starts.sort_unstable();
    for (image_i, start) in placeholder_starts.iter().copied().enumerate() {
        if start < cursor {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "overlapping qwen3_vl image placeholders".to_string(),
            ));
        }
        expanded.extend_from_slice(&base_tokens[cursor..start]);
        let (patches, geom) = patchify_qwen3_vl_image(&collected.images[image_i])?;
        let soft = geom.soft_token_count;
        let replacement_start = expanded.len();
        for _ in 0..soft {
            expanded.push(pad_id);
        }
        // Consume the original placeholder token span.
        cursor = start + ph_tokens.len();
        images_rt.push(Qwen3VlImageRuntimeInput {
            placeholder_index: replacement_start,
            soft_token_count: soft,
            patches: patches.patches,
            num_patches: patches.num_patches,
            patch_dim: patches.patch_dim,
            height: geom.height,
            width: geom.width,
            patch_size: geom.patch_size,
            spatial_merge_size: geom.spatial_merge_size,
        });
    }
    expanded.extend_from_slice(&base_tokens[cursor..]);

    let runtime_inputs = Qwen3VlRuntimeInputs { images: images_rt };
    runtime_inputs
        .validate_for_prompt_len(expanded.len())
        .map_err(|error| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("qwen3_vl multimodal inputs invalid: {error}"),
            )
        })?;

    Ok(Some(Qwen3VlChatPrompt {
        input_tokens: expanded,
        runtime_inputs,
    }))
}

#[derive(Clone, Copy)]
struct Qwen3VlGeom {
    height: u32,
    width: u32,
    patch_size: u32,
    spatial_merge_size: u32,
    soft_token_count: u32,
}

struct Qwen3VlPatches {
    patches: Vec<f32>,
    num_patches: u32,
    patch_dim: u32,
}

/// Decode image bytes → fixed-grid RGB patches for the portable Qwen3-VL ViT.
fn patchify_qwen3_vl_image(
    bytes: &[u8],
) -> Result<(Qwen3VlPatches, Qwen3VlGeom), HttpErrorResponse> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("failed to probe image: {error}"),
            )
        })?;
    reader.limits({
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits
    });
    let rgb = reader
        .decode()
        .map_err(|error| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("failed to decode image: {error}"),
            )
        })?
        .to_rgb8();

    let patch = QWEN3_VL_DEFAULT_PATCH_SIZE;
    let merge = QWEN3_VL_DEFAULT_SPATIAL_MERGE;
    let unit = patch.saturating_mul(merge).max(1);
    // Snap to a modest grid (≤ max soft tokens) while keeping aspect.
    let (ow, oh) = (rgb.width().max(1), rgb.height().max(1));
    let max_side_units = ((QWEN3_VL_DEFAULT_MAX_SOFT as f32).sqrt().floor() as u32).max(1);
    let scale = (max_side_units as f32 * unit as f32 / ow.max(oh) as f32).min(1.0);
    let tw = (((ow as f32 * scale) / unit as f32).ceil() as u32).max(1) * unit;
    let th = (((oh as f32 * scale) / unit as f32).ceil() as u32).max(1) * unit;
    let resized = if tw == ow && th == oh {
        rgb
    } else {
        image::imageops::resize(&rgb, tw, th, image::imageops::FilterType::Triangle)
    };

    let gh = th / unit;
    let gw = tw / unit;
    let soft = gh.saturating_mul(gw).min(QWEN3_VL_DEFAULT_MAX_SOFT);
    if soft == 0 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "qwen3_vl image collapsed to zero soft tokens".to_string(),
        ));
    }
    // Portable ViT: one patch token per soft token; patch_dim = unit² * 3 RGB.
    let patch_dim = unit.saturating_mul(unit).saturating_mul(3);
    let mut patches = Vec::with_capacity((soft as usize).saturating_mul(patch_dim as usize));
    for gy in 0..gh {
        for gx in 0..gw {
            if patches.len() / patch_dim as usize >= soft as usize {
                break;
            }
            let x0 = gx * unit;
            let y0 = gy * unit;
            for py in 0..unit {
                for px in 0..unit {
                    let p = resized.get_pixel(x0 + px, y0 + py).0;
                    patches.push(p[0] as f32 / 255.0);
                    patches.push(p[1] as f32 / 255.0);
                    patches.push(p[2] as f32 / 255.0);
                }
            }
        }
    }
    // Truncate to soft * patch_dim if we overshot.
    patches.truncate(soft as usize * patch_dim as usize);

    Ok((
        Qwen3VlPatches {
            patches,
            num_patches: soft,
            patch_dim,
        },
        Qwen3VlGeom {
            height: th,
            width: tw,
            patch_size: patch,
            spatial_merge_size: merge,
            soft_token_count: soft,
        },
    ))
}

fn find_subsequence_starts(haystack: &[u32], needle: &[u32]) -> Vec<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return Vec::new();
    }
    let mut starts = Vec::new();
    let last = haystack.len() - needle.len();
    for i in 0..=last {
        if haystack[i..i + needle.len()] == *needle {
            starts.push(i);
        }
    }
    starts
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
                    OpenAiChatContentPartKind::Video => {
                        collected.videos.push(video_part_bytes(part)?);
                        // Video uses the same soft-token placeholder surface as image
                        // at the chat template layer; expansion happens later.
                        rendered.push_str(image_placeholder);
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

/// Decode video payload from a chat content part. Data URIs only (no remote
/// `http(s)` and no bare filesystem paths) — preserves the SSRF-free posture.
fn video_part_bytes(part: &OpenAiChatContentPart) -> Result<Vec<u8>, HttpErrorResponse> {
    let value = part
        .video_url
        .as_ref()
        .or(part.image_url.as_ref())
        .ok_or_else(|| media_payload_missing("video", "video_url"))?;
    let url =
        data_value_url(value).ok_or_else(|| media_payload_missing("video", "video_url.url"))?;
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

    #[test]
    fn openai_tool_presence_treats_numeric_zero_as_false() {
        assert!(!openai_value_is_present(&json!(0)));
        assert!(!openai_value_is_present(&json!(0.0)));
        assert!(!tool_choice_forces_tool_call(&json!(0)));
        assert!(!tool_choice_forces_tool_call(&json!(0.0)));
        assert!(openai_value_is_present(&json!(-1)));
        assert!(tool_choice_forces_tool_call(&json!(0.5)));
    }

    #[test]
    fn is_qwen3_vl_model_id_detects_vl_families() {
        assert!(is_qwen3_vl_model_id("Qwen/Qwen3-VL-8B-Instruct"));
        assert!(is_qwen3_vl_model_id("ax-qwen3_vl-4bit"));
        assert!(is_qwen3_vl_model_id("qwen3-vl-moe"));
        assert!(!is_qwen3_vl_model_id("qwen3.6-27b"));
        assert!(!is_qwen3_vl_model_id("gemma4-unified"));
    }

    #[test]
    fn find_subsequence_starts_locates_placeholders() {
        let hay = [1u32, 2, 9, 9, 3, 9, 9, 4];
        let needle = [9u32, 9];
        assert_eq!(find_subsequence_starts(&hay, &needle), vec![2, 5]);
        assert!(find_subsequence_starts(&hay, &[7]).is_empty());
    }

    #[test]
    fn patchify_qwen3_vl_image_produces_valid_runtime_input() {
        // 1×1 red PNG
        let png = {
            use image::{ImageBuffer, ImageEncoder, Rgb};
            let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_pixel(28, 28, Rgb([255, 0, 0]));
            let mut buf = Vec::new();
            image::codecs::png::PngEncoder::new(&mut buf)
                .write_image(img.as_raw(), 28, 28, image::ExtendedColorType::Rgb8)
                .expect("encode png");
            buf
        };
        let (patches, geom) = patchify_qwen3_vl_image(&png).expect("patchify");
        assert!(geom.soft_token_count > 0);
        assert_eq!(patches.num_patches, geom.soft_token_count);
        assert_eq!(
            patches.patches.len(),
            patches.num_patches as usize * patches.patch_dim as usize
        );
        let rt = Qwen3VlImageRuntimeInput {
            placeholder_index: 0,
            soft_token_count: geom.soft_token_count,
            patches: patches.patches,
            num_patches: patches.num_patches,
            patch_dim: patches.patch_dim,
            height: geom.height,
            width: geom.width,
            patch_size: geom.patch_size,
            spatial_merge_size: geom.spatial_merge_size,
        };
        rt.validate(8).expect("valid against prompt len");
        let inputs = Qwen3VlRuntimeInputs { images: vec![rt] };
        assert!(!inputs.media_prefix_key("fp").is_empty());
    }
}
