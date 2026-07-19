use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use crate::routes::build_router;

use super::fixtures::{
    json_request_body, json_response, llama_cpp_server_state, spawn_llama_cpp_completion_server,
};

#[tokio::test]
async fn responses_endpoint_maps_stateless_text_input_and_output() {
    let (server_url, server_handle) = spawn_llama_cpp_completion_server(
        json!({
            "choices": [{
                "message": {"content": "response answer"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        })
        .to_string(),
        |payload| {
            assert_eq!(payload.get("max_tokens"), Some(&json!(7)));
            assert_eq!(
                payload.get("messages"),
                Some(&json!([
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello Responses API"}
                ]))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
        },
    );
    let app = build_router(llama_cpp_server_state(server_url));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": super::fixtures::TEST_MODEL_ID,
                "instructions": "Be concise.",
                "input": "Hello Responses API",
                "max_output_tokens": 7,
                "store": false
            }))))
            .expect("request should build"),
    )
    .await;
    server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["object"], "response");
    assert_eq!(response["status"], "completed");
    assert_eq!(response["store"], false);
    assert_eq!(response["output"][0]["type"], "message");
    assert_eq!(
        response["output"][0]["content"][0]["text"],
        "response answer"
    );
    assert_eq!(response["usage"]["input_tokens"], 5);
    assert_eq!(response["usage"]["output_tokens"], 2);
    assert_eq!(response["usage"]["total_tokens"], 7);
}

#[tokio::test]
async fn responses_endpoint_rejects_persisted_state_contracts() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    for (field, value, expected) in [
        (
            "store",
            json!(true),
            "store=true requires persisted Responses state",
        ),
        (
            "previous_response_id",
            json!("resp_prior"),
            "previous_response_id requires persisted Responses state",
        ),
        (
            "background",
            json!(true),
            "background responses are not supported",
        ),
    ] {
        let mut body = json!({"input": "hello"});
        body[field] = value;
        let (status, response) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&body)))
                .expect("request should build"),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response["error"]["code"], "unsupported_parameter");
        assert!(
            response["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains(expected))
        );
    }
}

#[tokio::test]
async fn responses_endpoint_rejects_mcp_tools_explicitly() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "input": "hello",
                "tools": [{"type": "mcp", "server_label": "workspace"}]
            }))))
            .expect("request should build"),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(response["error"]["code"], "unsupported_parameter");
    assert!(
        response["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("only function tools are available"))
    );
}

#[tokio::test]
async fn responses_endpoint_rejects_empty_input_array() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "instructions": "Do not treat instructions alone as input.",
                "input": []
            }))))
            .expect("request should build"),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(response["error"]["code"], "invalid_request");
    assert!(
        response["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("input must not be empty"))
    );
}
