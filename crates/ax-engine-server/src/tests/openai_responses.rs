use crate::openai::chunks::{chat_tool_calls_delta_chunk, chat_tool_calls_final_chunk};
use crate::openai::generation::validate_openai_json_object_response;
use crate::openai::requests::{OpenAiResponseOptions, OpenAiToolContract};
use crate::openai::responses::{
    openai_chat_completion_response, openai_completion_response, openai_finish_reason,
};
use ax_engine_sdk::{
    CapabilityReport, GenerateFinishReason, GenerateResponse, GenerateRouteReport, GenerateStatus,
    ResolutionPolicy, RuntimeReport, SelectedBackend, SupportTier,
};
use serde_json::json;
use std::sync::Arc;

#[test]
fn finish_reason_maps_terminal_labels_without_hiding_cancellations() {
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::Stop)),
        Some("stop")
    );
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::MaxOutputTokens)),
        Some("length")
    );
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::ContentFilter)),
        Some("content_filter")
    );
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::Cancelled)),
        Some("cancel")
    );
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::Error)),
        None
    );
    assert_eq!(openai_finish_reason(None), None);
}

#[test]
fn completion_response_includes_sampled_logprobs_when_available() {
    let response = sample_generate_response("hello", vec![10, 11], vec![Some(-0.25), Some(-0.5)]);

    let openai = openai_completion_response(
        &response,
        "cmpl-test".to_string(),
        OpenAiResponseOptions {
            include_logprobs: true,
            ..Default::default()
        },
    );

    let logprobs = openai.choices[0]
        .logprobs
        .as_ref()
        .expect("sampled logprobs should be present");
    assert_eq!(logprobs.tokens, vec!["10", "11"]);
    assert_eq!(logprobs.token_logprobs, vec![Some(-0.25), Some(-0.5)]);
    assert_eq!(logprobs.top_logprobs, vec![None, None]);
}

#[test]
fn completion_response_omits_unavailable_logprobs() {
    let response = sample_generate_response("hello", vec![10, 11], Vec::new());

    let openai = openai_completion_response(
        &response,
        "cmpl-test".to_string(),
        OpenAiResponseOptions {
            include_logprobs: true,
            ..Default::default()
        },
    );

    assert!(openai.choices[0].logprobs.is_none());
}

#[test]
fn logprobs_are_omitted_when_partially_observed_to_keep_arrays_aligned() {
    let response = sample_generate_response("hello", vec![10, 11], vec![Some(-0.25), None]);

    let options = OpenAiResponseOptions {
        include_logprobs: true,
        ..Default::default()
    };
    let completion =
        openai_completion_response(&response, "cmpl-test".to_string(), options.clone());
    assert!(completion.choices[0].logprobs.is_none());

    let chat =
        openai_chat_completion_response(&response, "chatcmpl-test".to_string(), options, None);
    assert!(chat.choices[0].logprobs.is_none());
}

#[test]
fn chat_response_exposes_reasoning_only_when_requested() {
    let response = sample_generate_response(
        "<think>check constraints</think>\nfinal answer",
        vec![1],
        vec![Some(-0.1)],
    );

    let default_openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        Default::default(),
        None,
    );
    assert_eq!(
        default_openai.choices[0].message.content,
        "<think>check constraints</think>\nfinal answer"
    );
    assert!(
        default_openai.choices[0]
            .message
            .reasoning_content
            .is_none()
    );

    let explicit_openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            include_reasoning: true,
            ..Default::default()
        },
        None,
    );
    assert_eq!(explicit_openai.choices[0].message.content, "final answer");
    assert_eq!(
        explicit_openai.choices[0]
            .message
            .reasoning_content
            .as_deref(),
        Some("check constraints")
    );
}

#[test]
fn chat_response_prefers_native_decode_reasoning_over_text_markers() {
    // Native MLX Gemma 4 decode strips channel framing at the token level and
    // hands the reasoning text alongside the cleaned content; the response
    // builder must surface it without re-scanning for markers.
    let response = sample_generate_response("final answer", vec![1], Vec::new());

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            include_reasoning: true,
            ..Default::default()
        },
        Some("check constraints".to_string()),
    );
    assert_eq!(openai.choices[0].message.content, "final answer");
    assert_eq!(
        openai.choices[0].message.reasoning_content.as_deref(),
        Some("check constraints")
    );

    // Without the explicit opt-in the native reasoning stays internal.
    let default_openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        Default::default(),
        Some("check constraints".to_string()),
    );
    assert!(
        default_openai.choices[0]
            .message
            .reasoning_content
            .is_none()
    );
}

#[test]
fn chat_response_extracts_tool_call_when_tool_contract_requested() {
    let response = sample_generate_response(
        r#"Before <tool_call>{"name":"lookup","arguments":{"query":"ax"}}</tool_call> after"#,
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    assert_eq!(message.content, "");
    assert_eq!(openai.choices[0].finish_reason, Some("tool_calls"));
    let tool_call = &message
        .tool_calls
        .as_ref()
        .expect("tool call should be parsed")[0];
    assert_eq!(tool_call.function.name, "lookup");
    assert_eq!(tool_call.function.arguments, r#"{"query":"ax"}"#);
}

#[test]
fn chat_response_extracts_glm_tool_call() {
    // GLM 4.x emits arg_key/arg_value pairs inside the shared <tool_call> tag,
    // with string values raw and non-string values JSON-encoded.
    let response = sample_generate_response(
        "<tool_call>get_weather\
         <arg_key>city</arg_key><arg_value>Tokyo</arg_value>\
         <arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    assert_eq!(message.content, "");
    assert_eq!(openai.choices[0].finish_reason, Some("tool_calls"));
    let tool_call = &message
        .tool_calls
        .as_ref()
        .expect("glm tool call should be parsed")[0];
    assert_eq!(tool_call.function.name, "get_weather");
    // String stays a string; bare number decodes to JSON number.
    assert_eq!(tool_call.function.arguments, r#"{"city":"Tokyo","days":3}"#);
}

#[test]
fn chat_response_extracts_gemma4_tool_call_from_ollama_dsl() {
    let response = sample_generate_response(
        r#"Before <|tool_call>call:lookup{limit:2,query:<|"|>AX<|"|>,exact:true}<tool_call|> after"#,
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    assert_eq!(message.content, "");
    let tool_call = &message
        .tool_calls
        .as_ref()
        .expect("Gemma4 tool call should be parsed")[0];
    assert_eq!(tool_call.function.name, "lookup");
    let arguments: serde_json::Value =
        serde_json::from_str(&tool_call.function.arguments).expect("arguments are JSON");
    assert_eq!(arguments["query"], "AX");
    assert_eq!(arguments["limit"], 2);
    assert_eq!(arguments["exact"], true);
    assert_eq!(openai.choices[0].finish_reason, Some("tool_calls"));
}

#[test]
fn chat_response_extracts_bare_gemma4_tool_call_from_live_output() {
    let response =
        sample_generate_response(r#"call:read_file{path:README.md}"#, Vec::new(), Vec::new());

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    assert_eq!(message.content, "");
    let tool_call = &message
        .tool_calls
        .as_ref()
        .expect("bare Gemma4 tool call should be parsed")[0];
    assert_eq!(tool_call.function.name, "read_file");
    let arguments: serde_json::Value =
        serde_json::from_str(&tool_call.function.arguments).expect("arguments are JSON");
    assert_eq!(arguments["path"], "README.md");
    assert_eq!(openai.choices[0].finish_reason, Some("tool_calls"));
}

#[test]
fn chat_response_extracts_multiple_tool_calls() {
    let response = sample_generate_response(
        r#"<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>
<tool_call>{"function":{"name":"run_command","arguments":"{\"cmd\":\"cargo test\"}"}}</tool_call>"#,
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    let tool_calls = message
        .tool_calls
        .as_ref()
        .expect("tool calls should be parsed");
    assert_eq!(message.content, "");
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0].id, "call_0");
    assert_eq!(tool_calls[0].function.name, "read_file");
    assert_eq!(tool_calls[0].function.arguments, r#"{"path":"README.md"}"#);
    assert_eq!(tool_calls[1].id, "call_1");
    assert_eq!(tool_calls[1].function.name, "run_command");
    assert_eq!(tool_calls[1].function.arguments, r#"{"cmd":"cargo test"}"#);
    assert_eq!(openai.choices[0].finish_reason, Some("tool_calls"));
}

#[test]
fn chat_response_extracts_qwen_function_parameter_tool_call() {
    let response = sample_generate_response(
        r#"<tool_call><function=todo_write>
<parameter=todos>
[{"content":"create index.html","status":"pending"}]
</parameter>
</function></tool_call>"#,
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    let tool_call = &message
        .tool_calls
        .as_ref()
        .expect("qwen function tool call should be parsed")[0];
    assert_eq!(message.content, "");
    assert_eq!(openai.choices[0].finish_reason, Some("tool_calls"));
    assert_eq!(tool_call.function.name, "todo_write");
    assert_eq!(
        tool_call.function.arguments,
        r#"{"todos":[{"content":"create index.html","status":"pending"}]}"#
    );
}

#[test]
fn chat_response_recovers_qwen_function_tool_call_without_closing_tags() {
    let response = sample_generate_response(
        r#"I'll create it now.

<tool_call>
<function=todo_write>
{"explanation":"Creating a responsive coffee shop website in Traditional Chinese","tasks":[{"file_path":"index.html","status":"in_progress"}]}"#,
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    let tool_call = &message
        .tool_calls
        .as_ref()
        .expect("partial qwen function tool call should be parsed")[0];
    assert_eq!(message.content, "");
    assert_eq!(openai.choices[0].finish_reason, Some("tool_calls"));
    assert_eq!(tool_call.function.name, "todo_write");
    assert_eq!(
        tool_call.function.arguments,
        r#"{"explanation":"Creating a responsive coffee shop website in Traditional Chinese","tasks":[{"file_path":"index.html","status":"in_progress"}]}"#
    );
}

#[test]
fn chat_response_recovers_qwen_tool_call_when_parameter_close_is_truncated() {
    // Qwen3-Coder models frequently emit the value but truncate the closing
    // </parameter> tag (max_tokens or premature stop). The reference
    // qwen3_coder_xml parser terminates the value at </function>; AX must do
    // the same instead of dropping the whole tool call onto the plain-text
    // path (which the guard then blocks as `unexecutable_tool_text`).
    let response = sample_generate_response(
        r#"<tool_call><function=todo_write>
<parameter=todos>
[{"content":"create index.html","status":"pending"}]
</function></tool_call>"#,
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    let tool_call = &message
        .tool_calls
        .as_ref()
        .expect("tool call with a truncated </parameter> should still parse")[0];
    assert_eq!(message.content, "");
    assert_eq!(openai.choices[0].finish_reason, Some("tool_calls"));
    assert_eq!(tool_call.function.name, "todo_write");
    assert_eq!(
        tool_call.function.arguments,
        r#"{"todos":[{"content":"create index.html","status":"pending"}]}"#
    );
}

#[test]
fn chat_response_recovers_qwen_tool_call_when_parameter_close_missing_and_function_truncated() {
    // Both </parameter> and </function> omitted: the value runs to end of body.
    // Mirrors the second-round `did you finish?` output the model emitted in
    // the field report, which previously surfaced as plain text.
    let response = sample_generate_response(
        r#"I haven't done it yet.
<tool_call>
<function=write_file>
<parameter=path>index.html
<parameter=content><!DOCTYPE html><html></html>"#,
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    let tool_calls = message
        .tool_calls
        .as_ref()
        .expect("tool call with truncated tags should still parse");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "write_file");
    let args: serde_json::Value =
        serde_json::from_str(&tool_calls[0].function.arguments).expect("arguments are JSON");
    // path terminates at the next <parameter=; content runs to end of body.
    assert_eq!(args["path"], "index.html");
    assert_eq!(args["content"], "<!DOCTYPE html><html></html>");
    // Tool-call turns suppress prose preambles so coding clients receive a
    // tool-call-only assistant message.
    assert_eq!(message.content, "");
}

#[test]
fn chat_response_recovers_qwen_tool_call_when_inner_parameter_close_is_truncated() {
    // A parameter whose own </parameter> close is missing must not greedily
    // absorb a *later* parameter that does carry a close tag: the unbounded
    // find of </parameter> previously swallowed `content` into `path` and
    // dropped `content` entirely.
    let response = sample_generate_response(
        r#"<tool_call><function=edit>
<parameter=path>
/tmp/a.txt
<parameter=content>
hello
</parameter>
</function></tool_call>"#,
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    let tool_calls = message
        .tool_calls
        .as_ref()
        .expect("tool call with an inner truncated </parameter> should still parse");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "edit");
    let args: serde_json::Value =
        serde_json::from_str(&tool_calls[0].function.arguments).expect("arguments are JSON");
    assert_eq!(args["path"], "/tmp/a.txt");
    assert_eq!(args["content"], "hello");
    assert_eq!(message.content, "");
}

#[test]
fn chat_response_canonicalizes_qwen_tool_calls_against_openai_tools_contract() {
    let response = sample_generate_response(
        r#"<tool_call><function=write_file>
<parameter=path>index.html</parameter>
<parameter=content><!DOCTYPE html><html></html></parameter>
</function></tool_call>"#,
        Vec::new(),
        Vec::new(),
    );

    let tools = json!([
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["filePath", "content"]
                }
            }
        }
    ]);
    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            tool_contract: OpenAiToolContract::from_tools(Some(&tools)).map(Arc::new),
            ..Default::default()
        },
        None,
    );

    let tool_call = &openai.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool call should be parsed")[0];
    assert_eq!(tool_call.function.name, "write");
    assert_eq!(
        tool_call.function.arguments,
        r#"{"content":"<!DOCTYPE html><html></html>","filePath":"index.html"}"#
    );
}

#[test]
fn chat_tool_call_stream_chunks_use_openai_delta_shape() {
    let response = sample_generate_response(
        r#"<tool_call>{"name":"lookup","arguments":{"query":"ax"}}</tool_call>"#,
        Vec::new(),
        Vec::new(),
    );
    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );
    let tool_calls = openai.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool call should parse");

    let delta = serde_json::to_value(chat_tool_calls_delta_chunk(
        7,
        "qwen3".to_string(),
        Some("assistant"),
        tool_calls,
    ))
    .expect("chunk should serialize");
    let first_call = &delta["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(delta["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(first_call["index"], 0);
    assert_eq!(first_call["id"], "call_0");
    assert_eq!(first_call["type"], "function");
    assert_eq!(first_call["function"]["name"], "lookup");
    assert_eq!(first_call["function"]["arguments"], r#"{"query":"ax"}"#);

    let final_chunk = serde_json::to_value(chat_tool_calls_final_chunk(7, "qwen3".to_string()))
        .expect("final chunk should serialize");
    assert_eq!(final_chunk["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn chat_response_leaves_bare_json_content_alone_even_with_tools_enabled() {
    // A model can legitimately answer with JSON while tools are offered; only
    // explicit <tool_call> markers may be converted into tool calls.
    let response = sample_generate_response(
        r#"{"name":"Alice","arguments":"ignored"}"#,
        Vec::new(),
        Vec::new(),
    );

    let openai = openai_chat_completion_response(
        &response,
        "chatcmpl-test".to_string(),
        OpenAiResponseOptions {
            parse_tool_calls: true,
            ..Default::default()
        },
        None,
    );

    let message = &openai.choices[0].message;
    assert!(message.tool_calls.is_none());
    assert_eq!(message.content, r#"{"name":"Alice","arguments":"ignored"}"#);
}

#[test]
fn json_object_validation_rejects_invalid_model_output() {
    let response = sample_generate_response("not json", Vec::new(), Vec::new());

    let error = validate_openai_json_object_response(
        &response,
        &OpenAiResponseOptions {
            validate_json_object: true,
            ..Default::default()
        },
    )
    .expect_err("invalid JSON should fail closed");

    assert_eq!(error.0, axum::http::StatusCode::BAD_GATEWAY);
    assert_eq!(error.1.0.error.code.as_deref(), Some("invalid_output"));
}

#[test]
fn json_object_validation_accepts_json_objects_only() {
    let object_response = sample_generate_response(r#"{"ok":true}"#, Vec::new(), Vec::new());
    let options = OpenAiResponseOptions {
        validate_json_object: true,
        ..Default::default()
    };
    validate_openai_json_object_response(&object_response, &options)
        .expect("JSON object should be accepted");

    let array_response = sample_generate_response(r#"[1,2]"#, Vec::new(), Vec::new());
    let _ = validate_openai_json_object_response(&array_response, &options)
        .expect_err("non-object JSON should fail closed");
}

fn sample_generate_response(
    output_text: &str,
    output_tokens: Vec<u32>,
    output_token_logprobs: Vec<Option<f32>>,
) -> GenerateResponse {
    GenerateResponse {
        request_id: 7,
        model_id: "qwen3".to_string(),
        prompt_tokens: vec![1, 2],
        prompt_text: None,
        output_tokens,
        output_token_logprobs,
        output_text: Some(output_text.to_string()),
        prompt_token_count: None,
        output_token_count: None,
        status: GenerateStatus::Finished,
        finish_reason: Some(GenerateFinishReason::Stop),
        step_count: 1,
        ttft_step: Some(1),
        route: GenerateRouteReport::default(),
        runtime: RuntimeReport {
            selected_backend: SelectedBackend::Mlx,
            support_tier: SupportTier::MlxPreview,
            resolution_policy: ResolutionPolicy::MlxOnly,
            capabilities: CapabilityReport::mlx_preview(),
            fallback_reason: None,
            host: Default::default(),
            metal_toolchain: Default::default(),
            mlx_runtime: None,
            mlx_model: None,
        },
    }
}
