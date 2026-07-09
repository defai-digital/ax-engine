use ax_engine_sdk::{GenerateFinishReason, GenerateResponse};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::app_state::AppState;
use crate::backends::{llama_cpp, mlx_lm};
use crate::errors::{ErrorResponse, error_response, map_session_error};
use crate::generation::native::run_stateless_generate_request;
use crate::openai::generation::populate_native_mlx_output_text;
use crate::openai::requests::{
    OpenAiBuiltLlamaCppChatRequest, OpenAiBuiltMlxLmChatRequest, OpenAiBuiltRequest,
    build_openai_chat_request_offloading_media, build_openai_llama_cpp_chat_request,
    build_openai_mlx_lm_chat_request,
};
use crate::openai::schema::{
    OpenAiChatCompletionHttpRequest, OpenAiChatContent, OpenAiChatMessage, OpenAiStopInput,
    OpenAiStreamKind,
};
use crate::openai::validation::validate_openai_request;
use crate::tasks::run_blocking_session_task;

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicMessagesRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    system: Option<AnthropicContent>,
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    thinking: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct AnthropicMessageResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: &'static str,
    role: &'static str,
    content: Vec<AnthropicResponseContentBlock>,
    model: String,
    stop_reason: &'static str,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
struct AnthropicResponseContentBlock {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
}

#[derive(Debug, Serialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

pub(crate) async fn anthropic_messages(
    State(state): State<AppState>,
    Json(request): Json<AnthropicMessagesRequest>,
) -> Result<Json<AnthropicMessageResponse>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    validate_openai_request(&live, request.model.as_deref())?;
    let openai_request = request.into_openai_chat_request()?;
    let (request_id, response) =
        run_anthropic_messages_generation(state, live, openai_request).await?;
    Ok(Json(anthropic_message_response(request_id, &response)))
}

impl AnthropicMessagesRequest {
    fn into_openai_chat_request(
        self,
    ) -> Result<OpenAiChatCompletionHttpRequest, (StatusCode, Json<ErrorResponse>)> {
        if self.stream {
            return Err(invalid_request(
                "Anthropic Messages streaming is not supported by this preview endpoint; send stream=false",
            ));
        }
        // `tool_choice` "none" opts out explicitly; "auto" (the SDK default
        // some adapters always serialize) requests nothing once `tools` is
        // empty, which the tools_in_use check already guarantees here.
        if tools_in_use(self.tools.as_ref())
            || feature_requested(self.tool_choice.as_ref(), &["none", "auto"])
        {
            return Err(invalid_request(
                "Anthropic Messages tool use is not supported by this inference-only endpoint",
            ));
        }
        // `thinking: {"type": "disabled"}` is the documented way to turn
        // extended thinking off; it must not trip the rejection.
        if feature_requested(self.thinking.as_ref(), &["disabled"]) {
            return Err(invalid_request(
                "Anthropic extended thinking is not supported by this inference-only endpoint",
            ));
        }
        let Some(max_tokens) = self.max_tokens else {
            return Err(invalid_request(
                "Anthropic Messages requires max_tokens for this endpoint",
            ));
        };

        let mut messages = Vec::new();
        if let Some(system) = self.system {
            let content = system.into_text("system")?;
            if !content.trim().is_empty() {
                messages.push(OpenAiChatMessage {
                    role: "system".to_string(),
                    content: Some(OpenAiChatContent::Text(content)),
                    tool_calls: None,
                    _tool_call_id: None,
                    _name: None,
                });
            }
        }

        for message in self.messages {
            let role = match message.role.as_str() {
                "user" | "assistant" => message.role,
                role => {
                    return Err(invalid_request(format!(
                        "unsupported Anthropic message role '{role}'; only user and assistant are supported"
                    )));
                }
            };
            messages.push(OpenAiChatMessage {
                role,
                content: Some(OpenAiChatContent::Text(
                    message.content.into_text("messages[].content")?,
                )),
                tool_calls: None,
                _tool_call_id: None,
                _name: None,
            });
        }

        Ok(OpenAiChatCompletionHttpRequest {
            model: self.model,
            messages,
            input_tokens: Vec::new(),
            max_tokens: Some(max_tokens),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: None,
            repetition_penalty: None,
            repetition_context_size: None,
            stop: self
                .stop_sequences
                .filter(|values| !values.is_empty())
                .map(OpenAiStopInput::Multiple),
            seed: None,
            stream: false,
            n: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: false,
            top_logprobs: None,
            reasoning: None,
            metadata: None,
            multimodal_inputs: Default::default(),
            response_format: None,
            tools: None,
            tool_choice: None,
        })
    }
}

impl AnthropicContent {
    fn into_text(
        self,
        field_name: &'static str,
    ) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
        match self {
            Self::Text(text) => Ok(text),
            Self::Blocks(blocks) => {
                let mut text = String::new();
                for block in blocks {
                    if block.block_type != "text" {
                        return Err(invalid_request(format!(
                            "unsupported Anthropic content block type '{}' in {field_name}; only text blocks are supported",
                            block.block_type
                        )));
                    }
                    let Some(block_text) = block.text else {
                        return Err(invalid_request(format!(
                            "Anthropic text content block in {field_name} requires text"
                        )));
                    };
                    text.push_str(&block_text);
                }
                Ok(text)
            }
        }
    }
}

async fn run_anthropic_messages_generation(
    state: AppState,
    live: crate::app_state::LiveState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<(u64, GenerateResponse), (StatusCode, Json<ErrorResponse>)> {
    if mlx_lm::is_selected(&live) {
        let OpenAiBuiltMlxLmChatRequest {
            chat_request,
            stream,
            response_options: _,
        } = build_openai_mlx_lm_chat_request(&live, request)?;
        reject_unexpected_stream(stream)?;
        let request_id = state.allocate_request_id();
        let runtime = live.runtime_report.clone();
        let mlx_lm_backend = mlx_lm::config(&live).map_err(map_session_error)?;
        let response = run_blocking_session_task(move || {
            mlx_lm::run_chat_generate(request_id, &runtime, &mlx_lm_backend, &chat_request)
        })
        .await?;
        return Ok((request_id, response));
    }

    if llama_cpp::supports_server_chat(&live) {
        let OpenAiBuiltLlamaCppChatRequest {
            chat_request,
            stream,
            response_options: _,
        } = build_openai_llama_cpp_chat_request(&live, request)?;
        reject_unexpected_stream(stream)?;
        let request_id = state.allocate_request_id();
        let runtime = live.runtime_report.clone();
        let llama_backend = llama_cpp::config(&live).map_err(map_session_error)?;
        let response = run_blocking_session_task(move || {
            llama_cpp::run_chat_generate(request_id, &runtime, &llama_backend, &chat_request)
        })
        .await?;
        return Ok((request_id, response));
    }

    let OpenAiBuiltRequest {
        generate_request,
        stream,
        response_options: _,
    } = build_openai_chat_request_offloading_media(&live, request).await?;
    reject_unexpected_stream(stream)?;
    let (request_id, mut response) =
        run_stateless_generate_request(&state, &live, generate_request).await?;
    populate_native_mlx_output_text(
        &live,
        &mut response,
        OpenAiStreamKind::ChatCompletion,
        false,
    )?;
    Ok((request_id, response))
}

fn anthropic_message_response(
    request_id: u64,
    response: &GenerateResponse,
) -> AnthropicMessageResponse {
    let (input_tokens, output_tokens) = response
        .known_usage()
        .unwrap_or((0, response.output_tokens.len() as u32));
    AnthropicMessageResponse {
        id: format!("msg-{request_id}"),
        response_type: "message",
        role: "assistant",
        content: vec![AnthropicResponseContentBlock {
            block_type: "text",
            text: response.output_text.clone().unwrap_or_default(),
        }],
        model: response.model_id.clone(),
        stop_reason: anthropic_stop_reason(response.finish_reason),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens,
            output_tokens,
        },
    }
}

fn anthropic_stop_reason(finish_reason: Option<GenerateFinishReason>) -> &'static str {
    match finish_reason {
        Some(GenerateFinishReason::MaxOutputTokens) => "max_tokens",
        Some(GenerateFinishReason::ContentFilter) => "refusal",
        Some(GenerateFinishReason::Stop)
        | Some(GenerateFinishReason::Cancelled)
        | Some(GenerateFinishReason::Error)
        | None => "end_turn",
    }
}

fn reject_unexpected_stream(stream: bool) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if stream {
        return Err(invalid_request(
            "Anthropic Messages streaming is not supported by this preview endpoint",
        ));
    }
    Ok(())
}

/// The Anthropic API accepts `tools: []` as "no tools" (some SDK adapters
/// always serialize the key), so only a non-empty array counts as tool use.
/// Malformed non-array values still count so the unsupported-feature error
/// surfaces them.
fn tools_in_use(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Array(tools)) => !tools.is_empty(),
        Some(_) => true,
    }
}

/// A feature payload is "requested" unless it is absent, null, or carries one
/// of the documented opt-out `type` values. Malformed values still count as
/// requested so the unsupported-feature error surfaces them.
fn feature_requested(value: Option<&Value>, opt_out_types: &[&str]) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(value) => !value
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| opt_out_types.contains(&kind)),
    }
}

fn invalid_request(message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    error_response(StatusCode::BAD_REQUEST, "invalid_request", message.into())
}

#[cfg(test)]
mod feature_gate_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn feature_requested_truth_table() {
        let tool_choice_opt_outs = ["none", "auto"];
        // Absent, null, and documented opt-out types are not requests.
        assert!(!feature_requested(None, &tool_choice_opt_outs));
        assert!(!feature_requested(
            Some(&Value::Null),
            &tool_choice_opt_outs
        ));
        assert!(!feature_requested(
            Some(&json!({"type": "none"})),
            &tool_choice_opt_outs
        ));
        assert!(!feature_requested(
            Some(&json!({"type": "auto"})),
            &tool_choice_opt_outs
        ));
        assert!(!feature_requested(
            Some(&json!({"type": "disabled"})),
            &["disabled"]
        ));
        // Real requests and malformed values must trip the rejection.
        assert!(feature_requested(
            Some(&json!({"type": "any"})),
            &tool_choice_opt_outs
        ));
        assert!(feature_requested(
            Some(&json!({"type": "tool", "name": "bash"})),
            &tool_choice_opt_outs
        ));
        assert!(feature_requested(Some(&json!({})), &tool_choice_opt_outs));
        assert!(feature_requested(
            Some(&json!("none")),
            &tool_choice_opt_outs
        ));
        assert!(feature_requested(
            Some(&json!({"type": 5})),
            &tool_choice_opt_outs
        ));
        assert!(feature_requested(
            Some(&json!({"type": "enabled", "budget_tokens": 1024})),
            &["disabled"]
        ));
    }

    #[test]
    fn tools_in_use_only_for_non_empty_arrays() {
        assert!(!tools_in_use(None));
        assert!(!tools_in_use(Some(&Value::Null)));
        assert!(!tools_in_use(Some(&json!([]))));
        assert!(tools_in_use(Some(&json!([{"name": "bash"}]))));
        assert!(tools_in_use(Some(&json!(0))));
    }

    #[test]
    fn anthropic_stop_reason_never_returns_null() {
        // The Anthropic Messages API requires stop_reason to be a non-null
        // string ("end_turn", "max_tokens", "stop_sequence", or "tool_use").
        use ax_engine_sdk::GenerateFinishReason;
        assert_eq!(
            anthropic_stop_reason(Some(GenerateFinishReason::Stop)),
            "end_turn"
        );
        assert_eq!(
            anthropic_stop_reason(Some(GenerateFinishReason::MaxOutputTokens)),
            "max_tokens"
        );
        assert_eq!(
            anthropic_stop_reason(Some(GenerateFinishReason::ContentFilter)),
            "refusal"
        );
        // Cancelled, Error, and None must NOT produce null.
        assert_eq!(
            anthropic_stop_reason(Some(GenerateFinishReason::Cancelled)),
            "end_turn"
        );
        assert_eq!(
            anthropic_stop_reason(Some(GenerateFinishReason::Error)),
            "end_turn"
        );
        assert_eq!(anthropic_stop_reason(None), "end_turn");
    }
}
