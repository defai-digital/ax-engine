use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;

use super::fixtures::{
    json_request_body, json_response, llama_cpp_server_state, llama_cpp_state,
    spawn_llama_cpp_completion_stream_server,
};

#[tokio::test]
async fn model_load_rejects_empty_model_id_before_touching_path() {
    let app = build_router(llama_cpp_state());
    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "  ",
                "model_path": "/definitely/not/a/model"
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], json!("invalid_request"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("model_id must not be empty")
    );
}

#[tokio::test]
async fn model_load_rejects_missing_model_path() {
    let app = build_router(llama_cpp_state());
    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "ax-engine/qwen3-coder-next",
                "model_path": "/definitely/not/a/model"
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], json!("invalid_request"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("model_path does not exist")
    );
}

#[tokio::test]
async fn model_load_rejects_delegated_backend_before_relabeling_model() {
    let temp_model_dir = std::env::temp_dir().join(format!(
        "ax-engine-server-model-load-delegated-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_model_dir).expect("temp model dir should create");

    let app = build_router(llama_cpp_state());
    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "ax-engine/qwen3-coder-next",
                "model_path": temp_model_dir
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], json!("unsupported_backend"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("only supported on the MLX-native backend")
    );

    std::fs::remove_dir_all(temp_model_dir).expect("temp model dir should clean up");
}

#[tokio::test]
async fn model_load_rejects_swap_while_requests_are_in_flight() {
    // Request state lives entirely inside the EngineSession instance with no
    // cross-session registry: swapping the session out from under a
    // non-terminal /v1/requests entry would silently orphan it (the client's
    // next /v1/requests/:id or /v1/step call would 404 instead of surfacing
    // a real terminal state). The load must fail closed instead.
    //
    // The stepwise submit path opens the streaming HTTP connection
    // synchronously, so a real (if minimal) mock server is required even
    // though this test never calls /v1/step. One non-terminal chunk keeps
    // the request Active without needing to drive it to completion.
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![serde_json::json!({
            "content": "partial",
            "tokens": [9],
            "stop": false
        })],
        |_payload| {},
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));

    // A real (if otherwise-irrelevant) directory: the handler checks
    // model_path existence before reaching the in-flight-requests check
    // this test exercises.
    let temp_model_dir = std::env::temp_dir().join(format!(
        "ax-engine-server-model-load-in-flight-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_model_dir).expect("temp model dir should create");

    let (submit_status, submit_body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "input_tokens": [1, 2, 3],
                "max_output_tokens": 4
            }))))
            .unwrap(),
    )
    .await;
    assert_eq!(submit_status, StatusCode::CREATED);
    let request_id = submit_body["request_id"]
        .as_u64()
        .expect("submitted request should report an id");

    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "ax-engine/qwen3-coder-next",
                "model_path": temp_model_dir
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], json!("requests_in_flight"));

    // The in-flight request must still be reachable afterward — the load was
    // rejected before touching the session, not silently orphaned.
    let (snapshot_status, snapshot_body) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri(format!("/v1/requests/{request_id}"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(snapshot_status, StatusCode::OK);
    assert_eq!(snapshot_body["request_id"], json!(request_id));

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
    std::fs::remove_dir_all(temp_model_dir).expect("temp model dir should clean up");
}
