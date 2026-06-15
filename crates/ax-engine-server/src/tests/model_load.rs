use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;

use super::fixtures::{json_request_body, json_response, llama_cpp_state};

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
