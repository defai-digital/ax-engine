use crate::chat;
use crate::openai::requests::{
    build_openai_chat_request, chat_template_kwargs_for_model_id, openai_chat_stop_sequences,
    render_openai_chat_prompt,
};
use crate::openai::schema::{OpenAiChatCompletionHttpRequest, OpenAiChatMessage, OpenAiStopInput};
use axum::http::StatusCode;
use serde_json::json;

use super::test_app_state;

#[test]
fn openai_chat_prompt_renderer_uses_model_family_templates() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"}
    ]))
    .expect("sample messages should deserialize");

    assert_eq!(
        render_openai_chat_prompt("qwen3", &messages).expect("qwen prompt"),
        "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(
        render_openai_chat_prompt("Meta-Llama-3.1-8B-Instruct", &messages).expect("llama prompt"),
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    assert_eq!(
        render_openai_chat_prompt("gemma4-e2b", &messages).expect("gemma4 prompt"),
        "<bos><|turn>system\nBe concise.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n"
    );
    assert_eq!(
        render_openai_chat_prompt("glm4_moe_lite", &messages).expect("glm prompt"),
        "[gMASK]<sop><|system|>Be concise.<|user|>Hello<|assistant|></think>"
    );
    assert_eq!(
        render_openai_chat_prompt("unknown-local-model", &messages).expect("plain prompt"),
        "system: Be concise.\nuser: Hello\nassistant:"
    );
}

#[test]
fn openai_chat_stop_sequences_merge_family_defaults_with_user_stop() {
    assert_eq!(
        openai_chat_stop_sequences("qwen3", None),
        vec!["<|im_end|>".to_string()]
    );
    assert_eq!(
        openai_chat_stop_sequences("Meta-Llama-3.1-8B-Instruct", None),
        vec!["<|eot_id|>".to_string()]
    );
    assert_eq!(
        openai_chat_stop_sequences("gemma4-e2b", None),
        vec!["<turn|>".to_string()]
    );
    assert_eq!(
        openai_chat_stop_sequences("mlx-community/GLM-4.7-Flash-4bit", None),
        vec![
            "<|endoftext|>".to_string(),
            "<|user|>".to_string(),
            "<|observation|>".to_string()
        ]
    );
    assert_eq!(
        openai_chat_stop_sequences(
            "gemma4-e2b",
            Some(OpenAiStopInput::Multiple(vec!["custom".to_string()]))
        ),
        vec!["custom".to_string(), "<turn|>".to_string()]
    );
    assert_eq!(
        openai_chat_stop_sequences(
            "gemma4-e2b",
            Some(OpenAiStopInput::Multiple(vec![
                "custom".to_string(),
                "<turn|>".to_string()
            ]))
        ),
        vec!["custom".to_string(), "<turn|>".to_string()]
    );
}

#[test]
fn openai_chat_prompt_renderer_rejects_known_families_without_verified_fallback() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Hello"}
    ]))
    .expect("sample messages should deserialize");

    for model_id in [
        "google/gemma-3-4b-it",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "mistral3-small",
        "mixtral-8x7b-instruct",
        "deepseek-ai/DeepSeek-V3",
    ] {
        let error = render_openai_chat_prompt(model_id, &messages)
            .expect_err("known unsupported chat fallback should fail closed");
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(
            error.1.error.message.contains("not supported yet"),
            "unexpected error for {model_id}: {}",
            error.1.error.message
        );
    }
}

#[test]
fn openai_chat_template_kwargs_disable_thinking_for_qwen_and_glm() {
    assert_eq!(
        chat_template_kwargs_for_model_id("qwen3"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("mlx-community/GLM-4.7-Flash-4bit"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(chat_template_kwargs_for_model_id("gemma4-e2b"), None);
    assert_eq!(
        chat_template_kwargs_for_model_id("deepseek-ai/DeepSeek-V3"),
        None
    );
}

#[test]
fn native_chat_renderer_keeps_thinking_open_for_qwen_reasoning_models() {
    // Companion to the kwargs test below: the native MLX path doesn't go
    // through mlx_lm's chat template; it builds the prompt locally. Qwen3.6
    // / Qwen3-Next / Qwen3-Coder-Next must end with `<think>\n` (thinking
    // allowed), NOT `<think>\n\n</think>\n\n` (thinking skipped), otherwise
    // the same #13 failure mode (premature stop / repetition loop on
    // reasoning prompts) reproduces on the native path even though the
    // delegated path is fixed.
    let messages = vec![("user".to_string(), "hi".to_string())];
    let thinking =
        chat::render_prompt_with_template(chat::ChatPromptTemplate::QwenChatMl, &messages, true)
            .expect("render");
    assert!(
        thinking.ends_with("<|im_start|>assistant\n<think>\n"),
        "thinking-enabled suffix should end with `<think>\\n` only: {thinking}"
    );
    assert!(
        !thinking.contains("</think>"),
        "thinking-enabled suffix must not pre-close the think block: {thinking}"
    );

    let no_thinking =
        chat::render_prompt_with_template(chat::ChatPromptTemplate::QwenChatMl, &messages, false)
            .expect("render");
    assert!(
        no_thinking.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "thinking-disabled suffix should pre-close the think block: {no_thinking}"
    );
}

#[test]
fn openai_chat_template_kwargs_omitted_for_qwen_reasoning_models() {
    // Issue #13: Qwen3.6 / Qwen3-Next / Qwen3-Coder-Next must NOT receive
    // enable_thinking=false — their chat templates inject an empty
    // <think></think> block that breaks reasoning-required prompts (math,
    // multi-step) by causing premature <|im_end|> emission.
    assert_eq!(
        chat_template_kwargs_for_model_id("Qwen3.6-35B-A3B-5bit"),
        None
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("qwen3_6-35b-a3b-ud-mlx-4bit"),
        None
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("mlx-community/Qwen3.6-35B-A3B-4bit"),
        None
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("Qwen3-Coder-Next-4bit"),
        None
    );
}

#[test]
fn openai_glm_prompt_renderer_preserves_tool_observation_shape() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "call tool"},
        {"role": "assistant", "content": "<tool_call>x</tool_call>"},
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "continue"}
    ]))
    .expect("sample messages should deserialize");

    assert_eq!(
        render_openai_chat_prompt("mlx-community/GLM-4.7-Flash-4bit", &messages)
            .expect("glm prompt"),
        "[gMASK]<sop><|user|>call tool<|assistant|></think><tool_call>x</tool_call><|observation|><tool_response>tool result</tool_response><|user|>continue<|assistant|></think>"
    );
}

#[tokio::test]
async fn openai_chat_request_applies_gemma4_default_stop_to_native_generate() {
    let state = test_app_state(|args| {
        args.model_id = "gemma4-e2b".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&state, request).expect("chat request should build");

    assert_eq!(
        built.generate_request.stop_sequences,
        vec!["<turn|>".to_string()]
    );
}

#[tokio::test]
async fn openai_chat_request_rejects_unsupported_ax_rendered_family() {
    let state = test_app_state(|args| {
        args.model_id = "deepseek-ai/DeepSeek-V3".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&state, request) {
        Ok(_) => panic!("AX-rendered fallback should fail closed"),
        Err(error) => error,
    };
    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(error.1.error.message.contains("deepseek"));
}
