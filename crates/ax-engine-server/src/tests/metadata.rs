use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use std::fs;
use std::path::Path;

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
    assert_eq!(model["capabilities"]["input"]["audio"], json!(false));
    assert_eq!(model["capabilities"]["input"]["image"], json!(false));
    assert_eq!(model["capabilities"]["input"]["video"], json!(false));
    assert_eq!(model["capabilities"]["output"]["text"], json!(true));
    assert_eq!(model["capabilities"]["attachment"], json!(false));
    assert_eq!(model["capabilities"]["toolcall"], json!(true));
    assert_eq!(model["capabilities"]["interleaved"], json!(false));
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
    assert_eq!(
        model["ax_engine"]["openai_tool_calling_supported"],
        json!(true)
    );
    assert_eq!(
        model["ax_engine"]["native_multimodal_input_supported"],
        json!(false)
    );
    assert_eq!(
        model["ax_engine"]["gemma4_unified_multimodal_input_supported"],
        json!(false)
    );
    assert_eq!(
        model["ax_engine"]["openai_tokenized_multimodal_input_supported"],
        json!(false)
    );
}

#[tokio::test]
async fn models_advertises_tool_calls_for_ax_code_qwen_coder_next_id() {
    let artifact_dir = minimal_tokenizer_artifact("qwen3-coder-next-metadata-tokenizer");
    let app = build_router(native_mlx_openai_builder_state(
        "ax-engine/qwen3_coder_next",
        &artifact_dir,
    ));
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
    assert_eq!(model["id"], json!("ax-engine/qwen3_coder_next"));
    assert_eq!(model["capabilities"]["toolcall"], json!(true));
    assert_eq!(
        model["ax_engine"]["openai_tool_calling_supported"],
        json!(true)
    );
    assert_eq!(model["ax_engine"]["primary_use"], json!("coding"));
    assert_eq!(model["ax_engine"]["chat_default"], json!(false));
    assert_eq!(model["ax_engine"]["coding_supported"], json!(true));
    assert_eq!(model["ax_engine"]["coding_only"], json!(true));
}

#[tokio::test]
async fn models_advertises_qwen36_as_general_chat_with_coding_support() {
    let artifact_dir = minimal_tokenizer_artifact("qwen36-metadata-tokenizer");
    let app = build_router(native_mlx_openai_builder_state(
        "Qwen3.6-35B-A3B-4bit",
        &artifact_dir,
    ));
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
    assert_eq!(model["id"], json!("Qwen3.6-35B-A3B-4bit"));
    assert_eq!(model["capabilities"]["toolcall"], json!(true));
    assert_eq!(
        model["ax_engine"]["openai_tool_calling_supported"],
        json!(true)
    );
    assert_eq!(model["ax_engine"]["primary_use"], json!("general"));
    assert_eq!(model["ax_engine"]["chat_default"], json!(true));
    assert_eq!(model["ax_engine"]["coding_supported"], json!(true));
    assert_eq!(model["ax_engine"]["coding_only"], json!(false));
}

#[tokio::test]
async fn models_advertises_processed_gemma4_unified_modalities_for_native_mlx() {
    let artifact_dir = minimal_tokenizer_artifact("gemma4-unified-metadata-tokenizer");
    write_gemma4_unified_manifest(&artifact_dir);
    let app = build_router(native_mlx_openai_builder_state(
        "gemma-4-12b-it",
        &artifact_dir,
    ));
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
    assert_eq!(model["capabilities"]["input"]["audio"], json!(true));
    assert_eq!(model["capabilities"]["input"]["image"], json!(true));
    assert_eq!(model["capabilities"]["input"]["video"], json!(true));
    assert_eq!(model["capabilities"]["output"]["text"], json!(true));
    assert_eq!(model["capabilities"]["output"]["audio"], json!(false));
    assert_eq!(model["capabilities"]["output"]["image"], json!(false));
    assert_eq!(model["capabilities"]["output"]["video"], json!(false));
    assert_eq!(model["capabilities"]["attachment"], json!(true));
    assert_eq!(model["capabilities"]["toolcall"], json!(true));
    assert_eq!(model["capabilities"]["interleaved"], json!(true));
    assert_eq!(
        model["ax_engine"]["openai_tool_calling_supported"],
        json!(true)
    );
    assert_eq!(
        model["ax_engine"]["native_multimodal_input_supported"],
        json!(true)
    );
    assert_eq!(
        model["ax_engine"]["gemma4_unified_multimodal_input_supported"],
        json!(true)
    );
    assert_eq!(
        model["ax_engine"]["openai_tokenized_multimodal_input_supported"],
        json!(true)
    );
}

fn write_gemma4_unified_manifest(artifact_dir: &Path) {
    let manifest = json!({
        "tensors": [
            {"role": "gemma4_unified_vision_patch_dense"},
            {"role": "gemma4_unified_vision_patch_dense_bias"},
            {"role": "gemma4_unified_vision_patch_norm1"},
            {"role": "gemma4_unified_vision_patch_norm1_bias"},
            {"role": "gemma4_unified_vision_patch_norm2"},
            {"role": "gemma4_unified_vision_patch_norm2_bias"},
            {"role": "gemma4_unified_vision_position_embedding"},
            {"role": "gemma4_unified_vision_position_norm"},
            {"role": "gemma4_unified_vision_position_norm_bias"},
            {"role": "gemma4_unified_vision_projection"},
            {"role": "gemma4_unified_audio_projection"}
        ]
    });
    fs::write(
        artifact_dir.join("model-manifest.json"),
        manifest.to_string(),
    )
    .expect("Gemma4 unified manifest should write");
}
