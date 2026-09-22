use crate::app_state::{EmbeddingBatchKey, EmbeddingBatchRequestOptions};
use crate::embeddings::microbatch::{collect_embedding_batch_groups, pooling_code};
use crate::openai::embeddings::{DEFAULT_EMBED_MAX_ITEMS, embedding_input_tokens};
use crate::openai::schema::{EmbeddingInput, OpenAiEmbeddingRequest};
use crate::routes::build_router;
use ax_engine_sdk::EmbeddingPooling;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use super::fixtures::{
    assert_invalid_request_response, json_request_body, json_response, llama_cpp_server_state,
    minimal_tokenizer_artifact, native_mlx_openai_builder_state,
};

fn sample_openai_embedding_request(input: &[u32], pooling: Option<&str>) -> Value {
    let mut request = serde_json::Map::new();
    request.insert("model".to_string(), json!(super::fixtures::TEST_MODEL_ID));
    request.insert("input".to_string(), json!(input));
    if let Some(pooling) = pooling {
        request.insert("pooling".to_string(), json!(pooling));
    }
    Value::Object(request)
}

#[test]
fn openai_embedding_request_accepts_text_and_text_batch_shapes() {
    let single: OpenAiEmbeddingRequest = serde_json::from_value(json!({"input": "hello"}))
        .expect("single text embedding input should deserialize");
    assert!(matches!(single.input, EmbeddingInput::Text(ref text) if text == "hello"));

    let batch: OpenAiEmbeddingRequest =
        serde_json::from_value(json!({"input": ["hello", "openai"]}))
            .expect("text batch embedding input should deserialize");
    assert!(matches!(
        batch.input,
        EmbeddingInput::TextBatch(ref texts) if texts == &["hello", "openai"]
    ));
}

#[tokio::test]
async fn openai_text_embedding_input_uses_model_tokenizer_and_eos() {
    let artifact_dir = minimal_tokenizer_artifact("openai-text-embedding-input");
    let state = native_mlx_openai_builder_state("qwen3-embedding", &artifact_dir);
    let live = state.snapshot();

    let single = embedding_input_tokens(&live, EmbeddingInput::Text("hello".to_string()))
        .expect("single text should tokenize");
    assert_eq!(single, vec![vec![1, 2]]);

    let batch = embedding_input_tokens(
        &live,
        EmbeddingInput::TextBatch(vec!["hello openai".to_string(), "chat".to_string()]),
    )
    .expect("text batch should tokenize");
    assert_eq!(batch, vec![vec![1, 5, 2], vec![6, 2]]);

    std::fs::remove_dir_all(artifact_dir).expect("tokenizer artifact should clean up");
}

#[tokio::test]
async fn openai_text_embedding_input_preserves_empty_input_validation() {
    let artifact_dir = minimal_tokenizer_artifact("openai-empty-text-embedding-input");
    let state = native_mlx_openai_builder_state("qwen3-embedding", &artifact_dir);
    let live = state.snapshot();

    let single = embedding_input_tokens(&live, EmbeddingInput::Text(String::new()))
        .expect("empty single text should reach shared validation");
    assert_eq!(single, vec![Vec::<u32>::new()]);

    let batch = embedding_input_tokens(
        &live,
        EmbeddingInput::TextBatch(vec!["hello".to_string(), String::new()]),
    )
    .expect("empty batch text should reach shared validation");
    assert_eq!(batch, vec![vec![1, 2], Vec::<u32>::new()]);

    std::fs::remove_dir_all(artifact_dir).expect("tokenizer artifact should clean up");
}

#[tokio::test]
async fn openai_embeddings_endpoint_rejects_unsupported_encoding_and_dimensions() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    for body in [
        json!({"input": [1, 2, 3], "encoding_format": "base64"}),
        json!({"input": [1, 2, 3], "dimensions": 128}),
    ] {
        let (status, response) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&body)))
                .expect("request should build"),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response["error"]["code"], "unsupported_parameter");
    }
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

#[tokio::test]
async fn openai_embeddings_rejects_batches_above_the_item_cap() {
    // The documented max_items cap (default 2048) bounds the batch size
    // regardless of per-item token counts; a 2049-item batch of one-token
    // inputs must be rejected with the invalid_request envelope.
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let batch: Vec<Vec<u32>> = (0..=DEFAULT_EMBED_MAX_ITEMS).map(|_| vec![1]).collect();
    let body = json!({
        "model": super::fixtures::TEST_MODEL_ID,
        "input": batch,
    });
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&body)))
            .expect("request should build"),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(
        &json,
        &format!("exceeding the maximum of {DEFAULT_EMBED_MAX_ITEMS}"),
    );
}

#[tokio::test]
async fn openai_embeddings_rejects_batches_above_the_total_token_cap() {
    // The batch-total token cap (default 8192 * 64) bounds the sum of every
    // item's tokens: 65 items of 8192 tokens each is 532480 > 524288 while
    // every item stays under the 8192 per-item cap.
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let batch: Vec<Vec<u32>> = (0..65).map(|_| vec![1_u32; 8192]).collect();
    let body = json!({
        "model": super::fixtures::TEST_MODEL_ID,
        "input": batch,
    });
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&body)))
            .expect("request should build"),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "batch token count");
}

#[tokio::test]
async fn openai_embeddings_token_cap_is_enforced_per_item() {
    // The 8192 DEFAULT_EMBED_MAX_TOKENS cap bounds each item alone: two
    // 5000-token inputs (batch total 10000) must pass validation, while a
    // single 9000-token input is rejected. Model-matched state so the
    // request reaches the cap check; the passing case then fails later on
    // the unreachable delegated backend (never with the cap message).
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let batch: Vec<Vec<u32>> = (0..2).map(|_| (0..5000).collect()).collect();
    let body = json!({
        "model": super::fixtures::TEST_MODEL_ID,
        "input": batch,
    });
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&body)))
            .expect("request should build"),
    )
    .await;
    let message = json["error"]["message"].as_str().unwrap_or_default();
    assert!(
        status != StatusCode::BAD_REQUEST || !message.contains("exceeds maximum"),
        "two per-item-valid inputs must not hit the token cap, got {status}: {message}"
    );

    let single: Vec<u32> = (0..9000).collect();
    let body = json!({
        "model": super::fixtures::TEST_MODEL_ID,
        "input": single,
    });
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&body)))
            .expect("request should build"),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "exceeds maximum");
}

#[tokio::test]
async fn openai_embeddings_mixed_input_shapes_return_the_json_envelope() {
    // A mixed input array fails the untagged EmbeddingInput enum; it must
    // come back as the documented 400 invalid_request JSON envelope, not
    // axum's plain-text 422.
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": super::fixtures::TEST_MODEL_ID,
                "input": ["hello", [1, 2, 3]]
            }))))
            .expect("request should build"),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "invalid request body");
}

#[tokio::test]
async fn embedding_records_endpoint_requires_tokenizer_artifacts() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let body = serde_json::json!({
        "model": super::fixtures::TEST_MODEL_ID,
        "records": [{
            "id": "doc-1",
            "fields": {
                "title": "Release notes",
                "body": "AX Engine embedding ingestion"
            },
            "metadata": {"source": "unit-test"}
        }],
        "render_template": "title: {title}\nbody: {body}",
        "chunking": {"max_tokens": 128, "overlap_tokens": 16}
    });
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embedding_records")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&body)))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(
        &json,
        "embedding record ingestion requires mlx_model_artifacts_dir",
    );
}
