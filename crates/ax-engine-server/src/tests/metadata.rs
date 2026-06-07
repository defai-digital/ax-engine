use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;

use super::fixtures::{
    json_response, llama_cpp_state, minimal_tokenizer_artifact, native_mlx_openai_builder_state,
};

#[tokio::test]
async fn models_reports_ax_code_safe_capabilities() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let model = &json["data"][0];
    assert!(
        !model.is_null(),
        "models response should include one model card"
    );

    assert_eq!(model["capabilities"]["toolcall"], json!(false));
    assert_eq!(model["owned_by"], json!("ax-engine"));
    assert_eq!(model["capabilities"]["input"]["text"], json!(true));
    assert_eq!(
        model["ax_engine"]["openai_tool_calling_supported"],
        json!(false)
    );
    assert_eq!(
        model["ax_engine"]["openai_chat_completions_supported"],
        json!(true)
    );
    assert_eq!(model["context_length"], json!(16 * 1024u32));
    assert_eq!(model["limit"]["output"], json!(2048u32));
}

#[tokio::test]
async fn models_advertises_openai_text_support_for_native_mlx() {
    // Native MLX serves the OpenAI text endpoints (see `validate_openai_text_backend`),
    // so `/v1/models` must advertise them rather than reporting them as unsupported.
    let artifact_dir = minimal_tokenizer_artifact("native-mlx-metadata-tokenizer");
    let app = build_router(native_mlx_openai_builder_state("qwen3", &artifact_dir));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let model = &json["data"][0];
    assert!(
        !model.is_null(),
        "models response should include one model card"
    );
    assert_eq!(model["capabilities"]["input"]["text"], json!(true));
    assert_eq!(model["capabilities"]["output"]["text"], json!(true));
    assert_eq!(
        model["ax_engine"]["openai_chat_completions_supported"],
        json!(true)
    );
    assert_eq!(
        model["ax_engine"]["openai_completions_supported"],
        json!(true)
    );
    assert_eq!(
        model["ax_engine"]["openai_text_input_supported"],
        json!(true)
    );
}
