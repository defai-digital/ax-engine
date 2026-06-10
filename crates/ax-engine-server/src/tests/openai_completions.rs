use crate::openai::requests::{DEFAULT_OPENAI_MAX_TOKENS, build_openai_completion_request};
use crate::openai::schema::OpenAiCompletionHttpRequest;
use crate::openai::validation::validate_openai_request;
use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use super::fixtures::{
    assert_invalid_request_response, json_request_body, json_response, llama_cpp_server_state,
    minimal_tokenizer_artifact, mlx_lm_delegated_state, native_mlx_openai_builder_state,
    openai_first_choice, sample_gemma4_multimodal_inputs, sample_openai_completion_request,
    sample_openai_request_base, spawn_llama_cpp_completion_server,
};

#[tokio::test]
async fn openai_completions_endpoint_translates_llama_cpp_response() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "content": "server::hello openai",
            "tokens": [4, 5],
            "stop": true,
            "stop_type": "limit",
            "tokens_evaluated": 4
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello openai".to_string()))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_completion_request("hello openai", Some(2), false),
            )))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.get("object").and_then(Value::as_str),
        Some("text_completion")
    );
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&json)
            .get("text")
            .and_then(Value::as_str),
        Some("server::hello openai")
    );
    assert_eq!(
        openai_first_choice(&json)
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
    assert_eq!(
        json.get("usage"),
        Some(&json!({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}))
    );
}

#[tokio::test]
async fn openai_completions_endpoint_translates_mlx_lm_delegated_response() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_server(
            r#"{"choices":[{"text":"mlx-lm::hello","finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#.to_string(),
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello mlx-lm".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
                assert_eq!(payload.get("top_k"), Some(&json!(0)));
                assert_eq!(payload.get("repetition_penalty"), Some(&json!(1.0)));
            },
        );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_completion_request("hello mlx-lm", Some(2), false),
            )))
            .unwrap(),
    )
    .await;
    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.get("object").and_then(Value::as_str),
        Some("text_completion")
    );
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&json)
            .get("text")
            .and_then(Value::as_str),
        Some("mlx-lm::hello")
    );
    assert_eq!(
        openai_first_choice(&json)
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("stop")
    );
}

#[tokio::test]
async fn openai_completion_logprobs_follows_legacy_integer_shape() {
    // Legacy completions `logprobs` is a top-N count: 0 opts into
    // sampled-token logprobs, anything above 0 fails closed until the runner
    // emits top-N alternatives.
    let artifact_dir = minimal_tokenizer_artifact("native-openai-completion-logprobs");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);

    let request: OpenAiCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "prompt": "hello",
        "max_tokens": 8,
        "logprobs": 0
    }))
    .expect("sample completion request should deserialize");
    let built =
        build_openai_completion_request(&state, request).expect("completion request should build");
    assert!(built.response_options.include_logprobs);

    let request: OpenAiCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "prompt": "hello",
        "max_tokens": 8,
        "logprobs": 2
    }))
    .expect("sample completion request should deserialize");
    let error = match build_openai_completion_request(&state, request) {
        Ok(_) => panic!("logprobs above 0 should fail closed until top-N is supported"),
        Err(error) => error,
    };
    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(
        error.1.0.error.code.as_deref(),
        Some("unsupported_parameter")
    );

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_completion_request_tokenizes_text_for_native_mlx_backend() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-completion-tokenizer");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let request: OpenAiCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "prompt": "hello openai completion",
        "max_tokens": 8
    }))
    .expect("sample completion request should deserialize");

    validate_openai_request(&state, request.model.as_deref())
        .expect("native MLX OpenAI completion should pass validation");
    let built =
        build_openai_completion_request(&state, request).expect("completion request should build");

    assert!(
        !built.generate_request.input_tokens.is_empty(),
        "native MLX OpenAI completion prompt should be tokenized"
    );
    assert_eq!(built.generate_request.input_text, None);
    assert_eq!(built.generate_request.max_output_tokens, 8);
    assert!(!built.stream);

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_completion_request_preserves_gemma4_multimodal_inputs_for_native_mlx_tokens() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-completion-gemma4-mm");
    let state = native_mlx_openai_builder_state("gemma-4-12b-it", &artifact_dir);
    let request: OpenAiCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "prompt": [10, 258880, 11],
        "max_tokens": 1,
        "multimodal_inputs": sample_gemma4_multimodal_inputs()
    }))
    .expect("Gemma4 OpenAI completion request should deserialize");

    let built =
        build_openai_completion_request(&state, request).expect("completion request should build");
    let inputs = built
        .generate_request
        .multimodal_inputs
        .gemma4_unified
        .expect("builder should preserve Gemma4 multimodal tensors");

    assert_eq!(built.generate_request.input_tokens, vec![10, 258880, 11]);
    assert_eq!(built.generate_request.input_text, None);
    assert_eq!(inputs.images.len(), 1);
    assert_eq!(inputs.images[0].pixel_values, vec![0.0, 1.0, 2.0]);

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_completion_request_rejects_gemma4_multimodal_inputs_with_text_prompt() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-completion-gemma4-mm-text");
    let state = native_mlx_openai_builder_state("gemma-4-12b-it", &artifact_dir);
    let request: OpenAiCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "prompt": "describe this image",
        "max_tokens": 1,
        "multimodal_inputs": sample_gemma4_multimodal_inputs()
    }))
    .expect("Gemma4 OpenAI completion request should deserialize");

    let error = match build_openai_completion_request(&state, request) {
        Ok(_) => panic!("text prompt with Gemma4 multimodal inputs should fail"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("pre-tokenized input"),
        "unexpected error: {}",
        error.1.error.message
    );

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_completion_request_rejects_gemma4_multimodal_inputs_on_delegated_backend() {
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let request: OpenAiCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "prompt": [10, 258880, 11],
        "max_tokens": 1,
        "multimodal_inputs": sample_gemma4_multimodal_inputs()
    }))
    .expect("Gemma4 OpenAI completion request should deserialize");

    let error = match build_openai_completion_request(&state, request) {
        Ok(_) => panic!("delegated Gemma4 multimodal inputs should fail"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("native MLX backend"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[tokio::test]
async fn openai_qwen_completion_uses_greedy_repetition_penalty_default() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-completion-qwen-rp-tokenizer");
    let state = native_mlx_openai_builder_state("Qwen3.6-27B-4bit", &artifact_dir);
    let request: OpenAiCompletionHttpRequest = serde_json::from_value(json!({
        "model": "Qwen3.6-27B-4bit",
        "prompt": "hello openai completion",
        "max_tokens": 384,
        "temperature": 0.0
    }))
    .expect("sample completion request should deserialize");

    let built =
        build_openai_completion_request(&state, request).expect("completion request should build");

    assert_eq!(built.generate_request.sampling.repetition_penalty, 1.1);

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_completions_endpoint_defaults_max_tokens() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "content": "server::default max tokens",
            "tokens": [4, 5],
            "stop": true,
            "stop_type": "limit"
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello openai".to_string()))
            );
            assert_eq!(
                payload.get("n_predict"),
                Some(&json!(DEFAULT_OPENAI_MAX_TOKENS))
            );
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_completion_request("hello openai", None, false),
            )))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        openai_first_choice(&json)
            .get("text")
            .and_then(Value::as_str),
        Some("server::default max tokens")
    );
}

#[tokio::test]
async fn openai_completions_endpoint_randomizes_unseeded_sampling_requests() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "content": "server::sampled",
            "tokens": [4],
            "stop": true
        })
        .to_string(),
        |payload| {
            assert_eq!(payload.get("temperature"), Some(&json!(0.8)));
            assert_ne!(payload.get("seed"), Some(&json!(0)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let mut request = sample_openai_request_base(Some(2), false, |request| {
        request.insert("prompt".to_string(), json!("sample me"));
        request.insert("temperature".to_string(), json!(0.8));
    });
    request.remove("model");

    let (status, _json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&Value::Object(request))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn openai_completions_endpoint_rejects_text_batch_prompt() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let mut request = sample_openai_request_base(Some(2), false, |request| {
        request.insert(
            "prompt".to_string(),
            json!(["first prompt", "second prompt"]),
        );
    });
    request.remove("model");

    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&Value::Object(request))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "batch completion prompts are not supported");
}
