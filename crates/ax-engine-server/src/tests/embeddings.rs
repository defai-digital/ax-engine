use crate::app_state::{EmbeddingBatchKey, EmbeddingBatchRequestOptions};
use crate::embeddings::microbatch::{collect_embedding_batch_groups, pooling_code};
use crate::routes::build_router;
use ax_engine_sdk::EmbeddingPooling;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use super::{
    assert_invalid_request_response, json_request_body, json_response, llama_cpp_server_state,
};

fn sample_openai_embedding_request(input: &[u32], pooling: Option<&str>) -> Value {
    let mut request = serde_json::Map::new();
    request.insert("model".to_string(), json!(super::TEST_MODEL_ID));
    request.insert("input".to_string(), json!(input));
    if let Some(pooling) = pooling {
        request.insert("pooling".to_string(), json!(pooling));
    }
    Value::Object(request)
}

#[test]
fn microbatch_groups_requests_by_options() {
    let groups = collect_embedding_batch_groups(&[
        EmbeddingBatchRequestOptions {
            pooling: EmbeddingPooling::Last,
            normalize: true,
        },
        EmbeddingBatchRequestOptions {
            pooling: EmbeddingPooling::Last,
            normalize: true,
        },
        EmbeddingBatchRequestOptions {
            pooling: EmbeddingPooling::Mean,
            normalize: true,
        },
        EmbeddingBatchRequestOptions {
            pooling: EmbeddingPooling::Last,
            normalize: false,
        },
    ]);

    assert_eq!(groups.len(), 3);
    assert_eq!(
        groups[0],
        (
            EmbeddingBatchKey {
                pooling_code: pooling_code(EmbeddingPooling::Mean),
                normalize: true
            },
            vec![2]
        )
    );
    assert_eq!(
        groups[1],
        (
            EmbeddingBatchKey {
                pooling_code: pooling_code(EmbeddingPooling::Last),
                normalize: false
            },
            vec![3]
        )
    );
    assert_eq!(
        groups[2],
        (
            EmbeddingBatchKey {
                pooling_code: pooling_code(EmbeddingPooling::Last),
                normalize: true
            },
            vec![0, 1]
        )
    );
}

#[tokio::test]
async fn openai_embeddings_endpoint_rejects_empty_input() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_embedding_request(&[], None),
            )))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "input must not be empty");
}

#[tokio::test]
async fn openai_embeddings_endpoint_accepts_batch_shape() {
    // Verify the request body for `input: [[1,2,3],[4,5,6]]` round-trips
    // through serde untagged enum into the Batch variant. We only check
    // deserialization + early validation (model-not-found in
    // llama_cpp_server_state), not the runtime — that needs a real MLX
    // session. The shape contract is what we care about here.
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let body = serde_json::json!({
        "model": "qwen3-embedding",
        "input": [[1, 2, 3], [4, 5, 6]],
    });
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&body)))
            .unwrap(),
    )
    .await;
    // Model-mismatch in the test server (returns 400 model_mismatch),
    // but the request shape parsed successfully — that's what we care
    // about here. If the untagged enum had failed to deserialise
    // `[[...], [...]]`, axum would have rejected the JSON body before
    // reaching the handler and the model name in the error would not
    // appear in the response.
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let msg = json["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("model_id") && msg.contains("qwen3-embedding"),
        "expected model-mismatch message containing the requested id, got: {msg}"
    );
}

#[tokio::test]
async fn openai_embeddings_endpoint_rejects_empty_batch_inner() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let body = serde_json::json!({
        "input": [[1, 2, 3], []],
    });
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&body)))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "input[1] must not be empty");
}

#[tokio::test]
async fn openai_embeddings_endpoint_rejects_unknown_pooling() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_embedding_request(&[1, 2, 3], Some("max")),
            )))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "unknown pooling strategy");
}
