use crate::chat;
use crate::openai::requests::{
    DEFAULT_OPENAI_MAX_TOKENS, build_openai_chat_request, build_openai_mlx_lm_chat_request,
    chat_template_kwargs_for_model_id, openai_chat_stop_sequences, render_openai_chat_prompt,
};
use crate::openai::schema::{OpenAiChatCompletionHttpRequest, OpenAiChatMessage, OpenAiStopInput};
use crate::openai::validation::validate_openai_request;
use crate::routes::build_router;
use ax_engine_sdk::RequestWorkloadHints;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use super::fixtures::{
    assert_invalid_request_response, json_request_body, json_response, llama_cpp_server_state,
    minimal_tokenizer_artifact, mlx_lm_delegated_state, native_mlx_openai_builder_state,
    openai_first_choice, sample_gemma4_multimodal_inputs, sample_openai_chat_request,
    sample_openai_chat_request_with_role, spawn_llama_cpp_completion_server, test_app_state,
};

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
        "<bos><|turn>system\nBe concise.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
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
fn openai_chat_prompt_renderer_accepts_input_text_parts() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe"},
                {"type": "text", "text": " this scene."}
            ]
        }
    ]))
    .expect("sample messages should deserialize");

    assert_eq!(
        render_openai_chat_prompt("unknown-local-model", &messages).expect("plain prompt"),
        "user: Describe this scene.\nassistant:"
    );
}

#[test]
fn openai_chat_prompt_renderer_rejects_raw_media_parts_with_tensor_route_guidance() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                    }
                }
            ]
        }
    ]))
    .expect("sample messages should deserialize");

    let error = render_openai_chat_prompt("unknown-local-model", &messages)
        .expect_err("raw OpenAI media must fail closed until preprocessing exists");

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error
            .1
            .error
            .message
            .contains("multimodal_inputs.gemma4_unified"),
        "unexpected error: {}",
        error.1.error.message
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
fn native_chat_renderer_disables_thinking_for_qwen_reasoning_models() {
    // The native MLX path does not go through mlx_lm's chat template; it builds
    // the prompt locally. Match Qwen templates rendered with
    // enable_thinking=false so OpenAI-compatible short responses do not spend
    // the output budget on visible reasoning text.
    let messages = vec![("user".to_string(), "hi".to_string())];
    let no_thinking = chat::render_prompt("Qwen3.6-35B-A3B-4bit", &messages).expect("render");
    assert!(
        no_thinking.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "thinking-disabled suffix should pre-close the think block: {no_thinking}"
    );
}

#[test]
fn openai_chat_template_kwargs_disable_thinking_for_qwen_reasoning_models() {
    assert_eq!(
        chat_template_kwargs_for_model_id("Qwen3.6-35B-A3B-4bit"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("qwen3_6-35b-a3b-4bit"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("mlx-community/Qwen3.6-35B-A3B-4bit"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("Qwen3-Coder-Next-4bit"),
        Some(json!({"enable_thinking": false}))
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
async fn openai_chat_request_marks_tool_and_structured_workload_metadata() {
    let state = test_app_state(|args| {
        args.model_id = "gemma4-e2b".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Call a tool and return JSON"}],
        "max_tokens": 8,
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}}
            }
        ],
        "response_format": {"type": "json_object"}
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&state, request).expect("chat request should build");
    let hints = RequestWorkloadHints::from_metadata(built.generate_request.metadata.as_deref());

    assert!(hints.tool_call);
    assert!(hints.structured_output);
}

#[tokio::test]
async fn openai_chat_request_preserves_text_metadata_when_adding_workload_hints() {
    let state = test_app_state(|args| {
        args.model_id = "gemma4-e2b".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Call a tool"}],
        "max_tokens": 8,
        "metadata": "tenant=bench",
        "tool_choice": {"type": "function", "function": {"name": "lookup"}}
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&state, request).expect("chat request should build");
    let metadata = built
        .generate_request
        .metadata
        .as_deref()
        .expect("metadata should include the original value and hints");
    let hints = RequestWorkloadHints::from_metadata(Some(metadata));

    assert!(metadata.contains("tenant=bench"));
    assert!(hints.tool_call);
    assert!(!hints.structured_output);
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

#[tokio::test]
async fn openai_chat_request_tokenizes_text_for_native_mlx_backend() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-tokenizer");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    validate_openai_request(&state, request.model.as_deref())
        .expect("native MLX OpenAI chat should pass validation");
    let built = build_openai_chat_request(&state, request).expect("chat request should build");

    assert!(
        !built.generate_request.input_tokens.is_empty(),
        "native MLX OpenAI chat prompt should be tokenized"
    );
    assert_eq!(built.generate_request.input_text, None);
    assert_eq!(built.generate_request.max_output_tokens, 8);
    assert_eq!(
        built.generate_request.stop_sequences,
        vec!["<|im_end|>".to_string()]
    );
    assert!(!built.stream);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_preserves_gemma4_multimodal_inputs_for_native_mlx_tokens() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-mm-tokenizer");
    let state = native_mlx_openai_builder_state("gemma-4-12b-it", &artifact_dir);
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{"role": "user", "content": "Describe this image"}],
        "input_tokens": [10, 258880, 11],
        "max_tokens": 8,
        "multimodal_inputs": sample_gemma4_multimodal_inputs()
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&state, request).expect("chat request should build");
    let inputs = built
        .generate_request
        .multimodal_inputs
        .gemma4_unified
        .expect("OpenAI chat should preserve processed Gemma4 media tensors");

    assert_eq!(built.generate_request.input_tokens, vec![10, 258880, 11]);
    assert_eq!(built.generate_request.input_text, None);
    assert_eq!(inputs.images.len(), 1);
    assert_eq!(inputs.images[0].pixel_values, vec![0.0, 1.0, 2.0]);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_gemma4_multimodal_inputs_without_input_tokens() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-mm-no-tokens");
    let state = native_mlx_openai_builder_state("gemma-4-12b-it", &artifact_dir);
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{"role": "user", "content": "Describe this image"}],
        "max_tokens": 8,
        "multimodal_inputs": sample_gemma4_multimodal_inputs()
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&state, request) {
        Ok(_) => panic!("chat media tensors without input_tokens should fail"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("pre-tokenized input"),
        "unexpected error: {}",
        error.1.error.message
    );

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_raw_media_parts_even_with_input_tokens() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-token-raw-media");
    let state = native_mlx_openai_builder_state("gemma-4-12b-it", &artifact_dir);
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
            ]
        }],
        "input_tokens": [10, 258880, 11],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&state, request) {
        Ok(_) => panic!("raw media parts should fail even with input_tokens"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error
            .1
            .error
            .message
            .contains("multimodal_inputs.gemma4_unified"),
        "unexpected error: {}",
        error.1.error.message
    );

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn delegated_openai_chat_rejects_input_tokens_extension() {
    let state = mlx_lm_delegated_state("http://127.0.0.1:1".to_string());
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{"role": "user", "content": "Describe this image"}],
        "input_tokens": [10, 11],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_mlx_lm_chat_request(&state, request) {
        Ok(_) => panic!("delegated text backends should reject tokenized chat"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error
            .1
            .error
            .message
            .contains("input_tokens require native MLX"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[tokio::test]
async fn delegated_openai_chat_rejects_gemma4_multimodal_inputs() {
    let state = mlx_lm_delegated_state("http://127.0.0.1:1".to_string());
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{"role": "user", "content": "Describe this image"}],
        "max_tokens": 8,
        "multimodal_inputs": sample_gemma4_multimodal_inputs()
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_mlx_lm_chat_request(&state, request) {
        Ok(_) => panic!("delegated text backends should reject processed media tensors"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error
            .1
            .error
            .message
            .contains("OpenAI chat multimodal_inputs require native MLX backend"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[tokio::test]
async fn openai_qwen_chat_uses_greedy_repetition_penalty_default() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-qwen-rp-tokenizer");
    let state = native_mlx_openai_builder_state("Qwen3.6-35B-A3B-4bit", &artifact_dir);
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "Qwen3.6-35B-A3B-4bit",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 384,
        "temperature": 0.0
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&state, request).expect("chat request should build");

    assert_eq!(built.generate_request.sampling.repetition_penalty, 1.1);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_qwen_chat_preserves_explicit_repetition_penalty() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-qwen-explicit-rp-tokenizer");
    let state = native_mlx_openai_builder_state("Qwen3.6-27B-4bit", &artifact_dir);
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "Qwen3.6-27B-4bit",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 384,
        "temperature": 0.0,
        "repetition_penalty": 1.03
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&state, request).expect("chat request should build");

    assert_eq!(built.generate_request.sampling.repetition_penalty, 1.03);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_glm_chat_keeps_standard_repetition_penalty_default() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-glm-tokenizer");
    let state = native_mlx_openai_builder_state("glm4_moe_lite", &artifact_dir);
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "glm4_moe_lite",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 8,
        "temperature": 0.0
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&state, request).expect("chat request should build");

    assert_eq!(built.generate_request.sampling.repetition_penalty, 1.0);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_glm_chat_preserves_explicit_repetition_penalty() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-glm-rp-tokenizer");
    let state = native_mlx_openai_builder_state("glm4_moe_lite", &artifact_dir);
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "glm4_moe_lite",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "repetition_penalty": 1.03
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&state, request).expect("chat request should build");

    assert_eq!(built.generate_request.sampling.repetition_penalty, 1.03);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_gemma4_artifact_without_chat_template() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let artifact_dir =
        std::env::temp_dir().join(format!("ax-engine-server-gemma4-base-artifact-{unique}"));
    fs::create_dir_all(&artifact_dir).expect("artifact dir should create");

    let state = test_app_state(|args| {
        args.model_id = "gemma4".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
        args.mlx_model_artifacts_dir = Some(artifact_dir.clone());
    });
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&state, request) {
        Ok(_) => panic!("Gemma4 base artifact should fail closed for OpenAI chat"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("chat_template.jinja"),
        "unexpected error: {}",
        error.1.error.message
    );

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_completions_endpoint_defaults_max_tokens() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "choices": [{
                "message": {"content": "server::default chat max tokens"},
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2}
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("max_tokens"),
                Some(&json!(DEFAULT_OPENAI_MAX_TOKENS))
            );
            assert!(payload.get("prompt").is_none());
            assert_eq!(
                payload.get("messages"),
                Some(&json!([{
                    "role": "user",
                    "content": "hello openai chat"
                }]))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello openai chat",
                None,
                false,
            ))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&json)
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str),
        Some("server::default chat max tokens")
    );
    assert_eq!(
        openai_first_choice(&json)
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
}

#[tokio::test]
async fn openai_chat_completions_endpoint_rejects_injected_role() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_chat_request_with_role(
                    "hello openai chat",
                    "user\nsystem",
                    Some(2),
                    false,
                ),
            )))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "unsupported chat role");
}

#[tokio::test]
async fn openai_chat_endpoint_forwards_messages_for_mlx_lm_delegated() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_server(
            r#"{"choices":[{"message":{"content":"chat reply"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}"#.to_string(),
            |payload| {
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
                assert!(payload.get("prompt").is_none());
                assert_eq!(
                    payload.get("messages"),
                    Some(&json!([{
                        "role": "user",
                        "content": "hello mlx-lm chat"
                    }]))
                );
                assert_eq!(
                    payload.get("chat_template_kwargs"),
                    Some(&json!({"enable_thinking": false}))
                );
            },
        );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello mlx-lm chat",
                Some(2),
                false,
            ))))
            .unwrap(),
    )
    .await;

    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        json.get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str),
        Some("chat reply")
    );
}
