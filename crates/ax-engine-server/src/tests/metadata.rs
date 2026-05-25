use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;

use super::fixtures::{json_response, llama_cpp_state};

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
