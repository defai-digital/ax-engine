use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use super::fixtures::{
    json_request_body, llama_cpp_server_state, mlx_lm_delegated_state, openai_first_choice,
    parse_openai_sse_payloads, sample_openai_chat_request, sample_openai_completion_request,
    spawn_llama_cpp_completion_stream_server, text_response,
};

#[tokio::test]
async fn openai_completions_stream_endpoint_emits_openai_sse_chunks() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            serde_json::json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            }),
            serde_json::json!({
                "content": " world",
                "tokens": [5],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        |payload| {
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello stream".to_string()))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_completion_request("hello stream", Some(2), true),
            )))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));
    let payloads = parse_openai_sse_payloads(&body);
    assert_eq!(payloads.len(), 3);
    assert_eq!(payloads[0].get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&payloads[0])
            .get("text")
            .and_then(Value::as_str),
        Some("hello")
    );
    assert_eq!(
        openai_first_choice(&payloads[1])
            .get("text")
            .and_then(Value::as_str),
        Some(" world")
    );
    assert_eq!(
        openai_first_choice(&payloads[2])
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn openai_chat_completions_stream_endpoint_emits_openai_sse_chunks() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": "chat"}, "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": " stream"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            }),
        ],
        |payload| {
            assert!(payload.get("prompt").is_none());
            assert_eq!(
                payload.get("messages"),
                Some(&json!([{
                    "role": "user",
                    "content": "hello chat stream"
                }]))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello chat stream",
                Some(2),
                true,
            ))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));
    let payloads = parse_openai_sse_payloads(&body);
    assert_eq!(payloads.len(), 3);
    assert_eq!(payloads[0].get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&payloads[0])
            .get("delta")
            .and_then(|delta| delta.get("role"))
            .and_then(Value::as_str),
        Some("assistant")
    );
    assert_eq!(
        openai_first_choice(&payloads[0])
            .get("delta")
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str),
        Some("chat")
    );
    assert_eq!(
        openai_first_choice(&payloads[1])
            .get("delta")
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str),
        Some(" stream")
    );
    assert_eq!(
        openai_first_choice(&payloads[2])
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
}

#[tokio::test]
async fn openai_chat_stream_endpoint_emits_sse_for_mlx_lm_delegated() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": " hello"}, "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
            }),
        ],
        |payload| {
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert!(payload.get("prompt").is_none());
            assert_eq!(
                payload.get("messages"),
                Some(&json!([{
                    "role": "user",
                    "content": "hello mlx-lm chat stream"
                }]))
            );
            assert_eq!(
                payload.get("chat_template_kwargs"),
                Some(&json!({"enable_thinking": false}))
            );
        },
    );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello mlx-lm chat stream",
                Some(2),
                true,
            ))))
            .unwrap(),
    )
    .await;

    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));

    let payloads = parse_openai_sse_payloads(&body);
    let text_chunks: Vec<&str> = payloads
        .iter()
        .filter_map(|p| {
            openai_first_choice(p)
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        })
        .collect();
    assert_eq!(text_chunks, vec![" hello", " world"]);
    assert_eq!(payloads[0].get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(payloads.last().expect("final chunk should exist"))
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("stop")
    );
}
