use crate::openai::generation::validate_openai_json_object_response;
use crate::openai::requests::OpenAiResponseOptions;
use crate::openai::responses::{
    openai_chat_completion_response, openai_completion_response, openai_finish_reason,
};
use ax_engine_sdk::{
    CapabilityReport, GenerateFinishReason, GenerateResponse, GenerateRouteReport, GenerateStatus,
    ResolutionPolicy, RuntimeReport, SelectedBackend, SupportTier,
};

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
    let completion = openai_completion_response(&response, "cmpl-test".to_string(), options);
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
    assert_eq!(message.content, "Before  after");
    let tool_call = &message
        .tool_calls
        .as_ref()
        .expect("tool call should be parsed")[0];
    assert_eq!(tool_call.function.name, "lookup");
    assert_eq!(tool_call.function.arguments, r#"{"query":"ax"}"#);
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
        OpenAiResponseOptions {
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
    validate_openai_json_object_response(&object_response, options)
        .expect("JSON object should be accepted");

    let array_response = sample_generate_response(r#"[1,2]"#, Vec::new(), Vec::new());
    let _ = validate_openai_json_object_response(&array_response, options)
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
