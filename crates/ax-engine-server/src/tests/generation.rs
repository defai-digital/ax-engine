use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use super::fixtures::{
    json_request_body, json_response, llama_cpp_server_state, llama_cpp_state,
    mlx_lm_delegated_state, normalize_measurement_fields, parse_sse_events, sample_http_request,
    sample_sdk_request, sample_text_http_request, sdk_session_for_state, sdk_stream_payload,
    spawn_llama_cpp_completion_server, spawn_llama_cpp_completion_stream_server, text_response,
};

#[tokio::test]
async fn llama_cpp_generate_endpoint_runs_text_request_through_sdk() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "content": "server::hello from server",
            "tokens": [4, 5],
            "stop": true,
            "stop_type": "limit"
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello from server".to_string()))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
        },
    );
    let state = llama_cpp_server_state(llama_server_url);
    let app = build_router(state.clone());
    let request_body = sample_text_http_request("hello from server", 2);
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&request_body)))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.get("prompt_text").and_then(|value| value.as_str()),
        Some("hello from server")
    );
    assert_eq!(
        json.get("output_text").and_then(|value| value.as_str()),
        Some("server::hello from server")
    );
    assert_eq!(
        json.get("runtime")
            .and_then(|runtime| runtime.get("selected_backend"))
            .and_then(|value| value.as_str()),
        Some("llama_cpp")
    );
}

#[tokio::test]
async fn mlx_lm_delegated_generate_uses_text_and_usage_count_contract() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_server(
            r#"{"choices":[{"text":"mlx-lm::hello","finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#.to_string(),
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello mlx-lm".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            },
        );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_text_http_request(
                "hello mlx-lm",
                2,
            ))))
            .unwrap(),
    )
    .await;
    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.get("output_text").and_then(Value::as_str),
        Some("mlx-lm::hello")
    );
    assert!(
        json.get("output_tokens")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
    );
    assert_eq!(json.get("output_token_count"), Some(&json!(3)));
    assert_eq!(
        json.get("finish_reason").and_then(Value::as_str),
        Some("stop")
    );
}

#[tokio::test]
async fn llama_cpp_stream_endpoint_runs_server_backed_stream_through_sdk() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        2,
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
            assert_eq!(payload.get("prompt"), Some(&json!([1, 2, 3])));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
        },
    );
    let state = llama_cpp_server_state(llama_server_url);
    let app = build_router(state.clone());
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate/stream")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_http_request(
                &[1, 2, 3],
                2,
            ))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));

    let mut actual = parse_sse_events(&body);
    let mut sdk_session = sdk_session_for_state(&state);
    let mut expected = sdk_session
        .stream_generate_with_request_id(1, sample_sdk_request(&[1, 2, 3], 2))
        .expect("sdk llama.cpp stream should start")
        .map(|event| event.map(sdk_stream_payload))
        .collect::<Result<Vec<_>, _>>()
        .expect("sdk llama.cpp stream should complete");
    for (_, payload) in &mut actual {
        normalize_measurement_fields(payload);
    }
    for (_, payload) in &mut expected {
        normalize_measurement_fields(payload);
    }

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn mlx_lm_delegated_generate_stream_endpoint_emits_sse_chunks() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            serde_json::json!({
                "choices": [{"index": 0, "text": " hello", "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "text": " world", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
            }),
        ],
        |payload| {
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello delegated stream".to_string()))
            );
        },
    );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate/stream")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_text_http_request(
                "hello delegated stream",
                2,
            ))))
            .unwrap(),
    )
    .await;

    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));

    let events = parse_sse_events(&body);
    let response_event = events
        .iter()
        .find(|(name, _)| name == "response")
        .expect("SSE stream should contain a response event");
    assert_eq!(
        response_event
            .1
            .get("response")
            .and_then(|r| r.get("output_text"))
            .and_then(Value::as_str),
        Some(" hello world")
    );
}

#[tokio::test]
async fn llama_cpp_cli_stream_endpoint_fails_closed() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate/stream")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_text_http_request(
                "hello from stream",
                2,
            ))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        json.get("error")
            .and_then(|value| value.get("code"))
            .and_then(|value| value.as_str()),
        Some("invalid_request")
    );
}
