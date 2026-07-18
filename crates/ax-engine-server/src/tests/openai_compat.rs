use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};

use crate::routes::build_router;

use super::fixtures::{
    json_request_body, json_response, minimal_tokenizer_artifact, native_mlx_openai_builder_state,
};

#[tokio::test]
async fn apply_template_endpoint_renders_openai_messages() {
    let artifact_dir = minimal_tokenizer_artifact("compat-apply-template");
    let app = build_router(native_mlx_openai_builder_state("qwen3", &artifact_dir));
    let request = Request::post("/apply-template")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hello"}]
        }))))
        .expect("request should build");

    let (status, body) = json_response(&app, request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body.get("prompt").and_then(Value::as_str),
        Some("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n")
    );

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn apply_template_endpoint_renders_tools_matching_real_generation() {
    // /apply-template exists to preview the exact prompt /v1/chat/completions
    // will send; before this fix it hardcoded tools=None, so a request with
    // tools rendered a materially different (tool-contract-free) prompt than
    // what generation would actually see.
    let artifact_dir = minimal_tokenizer_artifact("compat-apply-template-tools");
    let app = build_router(native_mlx_openai_builder_state("qwen3", &artifact_dir));
    let request_body = json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "Call a tool"}],
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}}
        }]
    });

    let with_tools = Request::post("/apply-template")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&request_body)))
        .expect("request should build");
    let (status, body) = json_response(&app, with_tools).await;
    assert_eq!(status, StatusCode::OK);
    let prompt = body
        .get("prompt")
        .and_then(Value::as_str)
        .expect("prompt should be a string");
    assert!(
        prompt.contains("lookup"),
        "prompt must render the tool contract, got: {prompt}"
    );

    let mut tools_free_body = request_body.clone();
    tools_free_body
        .as_object_mut()
        .expect("request body is an object")
        .remove("tools");
    let without_tools = Request::post("/apply-template")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&tools_free_body)))
        .expect("request should build");
    let (status, body) = json_response(&app, without_tools).await;
    assert_eq!(status, StatusCode::OK);
    let tools_free_prompt = body
        .get("prompt")
        .and_then(Value::as_str)
        .expect("prompt should be a string");
    assert_ne!(
        prompt, tools_free_prompt,
        "a tools-containing request must render a different prompt than a tools-free one"
    );

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn apply_template_endpoint_preserves_qwen36_preclosed_think_prompt() {
    let artifact_dir = minimal_tokenizer_artifact("compat-apply-template-qwen36");
    let app = build_router(native_mlx_openai_builder_state(
        "Qwen3.6-35B-A3B-4bit",
        &artifact_dir,
    ));
    let request = Request::post("/apply-template")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&json!({
            "model": "Qwen3.6-35B-A3B-4bit",
            "messages": [{"role": "user", "content": "hello"}]
        }))))
        .expect("request should build");

    let (status, body) = json_response(&app, request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body.get("prompt").and_then(Value::as_str),
        Some("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    );

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn v1_apply_template_alias_renders_openai_messages() {
    let artifact_dir = minimal_tokenizer_artifact("compat-v1-apply-template");
    let app = build_router(native_mlx_openai_builder_state("qwen3", &artifact_dir));
    let request = Request::post("/v1/apply-template")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hello"}]
        }))))
        .expect("request should build");

    let (status, body) = json_response(&app, request).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.get("prompt").and_then(Value::as_str).is_some());

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn tokenize_endpoint_returns_token_ids() {
    let artifact_dir = minimal_tokenizer_artifact("compat-tokenize");
    let app = build_router(native_mlx_openai_builder_state("qwen3", &artifact_dir));
    let request = Request::post("/tokenize")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&json!({
            "model": "qwen3",
            "content": "hello openai chat"
        }))))
        .expect("request should build");

    let (status, body) = json_response(&app, request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.get("tokens"), Some(&json!([1, 5, 6])));

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn tokenize_endpoint_can_return_pieces() {
    let artifact_dir = minimal_tokenizer_artifact("compat-tokenize-pieces");
    let app = build_router(native_mlx_openai_builder_state("qwen3", &artifact_dir));
    let request = Request::post("/v1/tokenize")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&json!({
            "model": "qwen3",
            "content": "hello",
            "with_pieces": true
        }))))
        .expect("request should build");

    let (status, body) = json_response(&app, request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body.get("tokens"),
        Some(&json!([{"id": 1, "piece": "hello"}]))
    );

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn tokenize_endpoint_rejects_model_mismatch() {
    let artifact_dir = minimal_tokenizer_artifact("compat-tokenize-model-mismatch");
    let app = build_router(native_mlx_openai_builder_state("qwen3", &artifact_dir));
    let request = Request::post("/tokenize")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&json!({
            "model": "other",
            "content": "hello"
        }))))
        .expect("request should build");

    let (status, body) = json_response(&app, request).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body.get("error")
            .and_then(|error| error.get("code"))
            .and_then(Value::as_str),
        Some("model_not_found")
    );
    assert!(
        body.get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("is not loaded")
    );

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn detokenize_endpoint_returns_content() {
    let artifact_dir = minimal_tokenizer_artifact("compat-detokenize");
    let app = build_router(native_mlx_openai_builder_state("qwen3", &artifact_dir));
    let request = Request::post("/detokenize")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&json!({
            "model": "qwen3",
            "tokens": [1, 5, 6]
        }))))
        .expect("request should build");

    let (status, body) = json_response(&app, request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.get("content"), Some(&json!("hello openai chat")));

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn detokenize_endpoint_rejects_model_mismatch() {
    let artifact_dir = minimal_tokenizer_artifact("compat-detokenize-model-mismatch");
    let app = build_router(native_mlx_openai_builder_state("qwen3", &artifact_dir));
    let request = Request::post("/v1/detokenize")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_request_body(&json!({
            "model": "other",
            "tokens": [1]
        }))))
        .expect("request should build");

    let (status, body) = json_response(&app, request).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body.get("error")
            .and_then(|error| error.get("code"))
            .and_then(Value::as_str),
        Some("model_not_found")
    );
    assert!(
        body.get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("is not loaded")
    );

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}
