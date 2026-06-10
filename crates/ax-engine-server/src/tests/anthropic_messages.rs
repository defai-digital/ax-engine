use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use super::fixtures::{
    assert_invalid_request_response, json_request_body, json_response, llama_cpp_server_state,
    spawn_llama_cpp_completion_server,
};

#[tokio::test]
async fn anthropic_messages_endpoint_translates_text_request_and_response() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "choices": [{
                "message": {"content": "server::anthropic reply"},
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}
        })
        .to_string(),
        |payload| {
            assert_eq!(payload.get("prompt"), None);
            assert_eq!(payload.get("max_tokens"), Some(&json!(17)));
            assert_eq!(payload.get("temperature"), Some(&json!(0.2)));
            assert_eq!(payload.get("top_p"), Some(&json!(0.9)));
            assert_eq!(payload.get("top_k"), Some(&json!(40)));
            assert_eq!(payload.get("stop"), Some(&json!(["DONE", "<|im_end|>"])));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            assert_eq!(
                payload.get("messages"),
                Some(&json!([
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hello anthropic messages"}
                ]))
            );
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "system": [{"type": "text", "text": "Be concise."}],
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "hello anthropic messages"}]
                }],
                "max_tokens": 17,
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "stop_sequences": ["DONE"]
            }))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(
        response
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .starts_with("msg-")
    );
    assert_eq!(
        response.get("type").and_then(Value::as_str),
        Some("message")
    );
    assert_eq!(
        response.get("role").and_then(Value::as_str),
        Some("assistant")
    );
    assert_eq!(response.get("model").and_then(Value::as_str), Some("qwen3"));
    assert_eq!(
        response
            .get("content")
            .and_then(Value::as_array)
            .and_then(|content| content.first())
            .and_then(|block| block.get("text"))
            .and_then(Value::as_str),
        Some("server::anthropic reply")
    );
    assert_eq!(
        response.get("stop_reason").and_then(Value::as_str),
        Some("max_tokens")
    );
    assert_eq!(
        response.get("usage"),
        Some(&json!({"input_tokens": 7, "output_tokens": 3}))
    );
}

#[tokio::test]
async fn anthropic_messages_endpoint_rejects_tool_use() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "tools": [{"name": "bash", "input_schema": {"type": "object"}}]
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&response, "tool use is not supported");
}

#[tokio::test]
async fn anthropic_messages_endpoint_accepts_empty_tools_and_disabled_thinking() {
    // `tools: []`, `tool_choice: {"type": "none"}`, and
    // `thinking: {"type": "disabled"}` are valid Anthropic payloads that use
    // no unsupported feature; they must not be rejected.
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "choices": [{
                "message": {"content": "no tools used"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2}
        })
        .to_string(),
        |_payload| {},
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "tools": [],
                "tool_choice": {"type": "none"},
                "thinking": {"type": "disabled"}
            }))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        response
            .get("content")
            .and_then(Value::as_array)
            .and_then(|content| content.first())
            .and_then(|block| block.get("text"))
            .and_then(Value::as_str),
        Some("no tools used")
    );
}

#[tokio::test]
async fn anthropic_messages_endpoint_rejects_enabled_thinking_and_malformed_tools() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "thinking": {"type": "enabled", "budget_tokens": 1024}
            }))))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&response, "extended thinking is not supported");

    // Malformed non-array `tools` values still surface the rejection.
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "tools": 0
            }))))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&response, "tool use is not supported");
}

#[tokio::test]
async fn anthropic_messages_endpoint_rejects_non_text_content_blocks() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "messages": [{
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]
                }],
                "max_tokens": 4
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&response, "only text blocks are supported");
}
