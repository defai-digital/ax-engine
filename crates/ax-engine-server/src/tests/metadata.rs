use crate::app_state::build_live_state;
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
async fn models_advertises_reasoning_for_openclaw_qwen_thinking_variants() {
    let artifact_dir = minimal_tokenizer_artifact("openclaw-qwen-reasoning-metadata");
    for (model_id, expected) in [
        ("qwen3.5-9b-mtp", true),
        ("qwen3.6-27b-mtp", true),
        ("qwen3-vl-8b-thinking", true),
        ("qwen3-vl-8b-instruct", false),
    ] {
        let app = build_router(native_mlx_openai_builder_state(model_id, &artifact_dir));
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert_eq!(status, StatusCode::OK, "{model_id}");
        assert_eq!(
            json["data"][0]["capabilities"]["reasoning"],
            json!(expected),
            "{model_id}"
        );
    }
    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn models_lists_every_loaded_model() {
    let state = llama_cpp_state();
    let config = state.snapshot().session_config.as_ref().clone();
    let second = build_live_state("gemma-4-12b-it".to_string(), config)
        .expect("second delegated state should build");
    assert!(state.publish_live(second, true).is_none());
    let app = build_router(state.clone());

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
    let ids = json["data"]
        .as_array()
        .expect("models data should be an array")
        .iter()
        .map(|model| model["id"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["gemma-4-12b-it", "qwen3"]);

    let removed = state
        .remove_live("gemma-4-12b-it")
        .expect("second model should remove");
    removed.retire().await.expect("second worker should retire");
}

#[tokio::test]
async fn health_and_discovery_list_every_loaded_model() {
    let state = llama_cpp_state();
    let config = state.snapshot().session_config.as_ref().clone();
    let second = build_live_state("gemma-4-12b-it".to_string(), config)
        .expect("second delegated state should build");
    assert!(state.publish_live(second, false).is_none());
    let app = build_router(state.clone());

    for uri in ["/health", "/v1/discovery"] {
        let (status, json) = json_response(
            &app,
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::OK, "{uri}");
        assert_eq!(
            json["models"],
            json!(["gemma-4-12b-it", "qwen3"]),
            "{uri} should list every loaded model"
        );
        assert_eq!(
            json["model_id"],
            json!("qwen3"),
            "{uri} default model should be unchanged by a non-default add"
        );
    }

    let removed = state
        .remove_live("gemma-4-12b-it")
        .expect("second model should remove");
    removed.retire().await.expect("second worker should retire");
}

#[tokio::test]
async fn readiness_fails_when_any_loaded_model_worker_is_unavailable() {
    let state = llama_cpp_state();
    let config = state.snapshot().session_config.as_ref().clone();
    let second = build_live_state("gemma-4-12b-it".to_string(), config)
        .expect("second delegated state should build");
    let second_service = second.generation_service.clone();
    state.publish_live(second, false);
    second_service
        .shutdown()
        .await
        .expect("second worker should stop");
    let app = build_router(state.clone());

    for uri in ["/health", "/v1/discovery"] {
        let (status, body) = json_response(
            &app,
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri}");
        assert_eq!(
            body["error"]["code"],
            json!("generation_worker_unavailable"),
            "{uri}"
        );
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("gemma-4-12b-it"),
            "{uri}"
        );
    }

    let removed = state
        .remove_live("gemma-4-12b-it")
        .expect("second model should remove");
    removed.retire().await.expect("second worker should retire");
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
    assert_eq!(model["capabilities"]["input"]["video"], json!(false));
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

#[tokio::test]
async fn models_advertises_gemma4_vl_encoder_tower_image_and_video() {
    // Encoder-VL packages (family gemma4_vl) ship vision_tower + embed_vision
    // like standard gemma4; capability discovery must not require family==gemma4.
    let artifact_dir = minimal_tokenizer_artifact("gemma4-vl-encoder-metadata");
    fs::write(
        artifact_dir.join("model-manifest.json"),
        json!({
            "model_family": "gemma4_vl",
            "tensors": [
                {"name": "vision_tower.embeddings.patch_embedding.weight", "role": "other"},
                {"name": "embed_vision.embedding_projection.weight", "role": "other"}
            ]
        })
        .to_string(),
    )
    .expect("gemma4_vl manifest should write");
    let app = build_router(native_mlx_openai_builder_state(
        "gemma-4-e2b-it",
        &artifact_dir,
    ));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let model = &json["data"][0];
    assert_eq!(
        model["capabilities"]["input"]["image"],
        json!(true),
        "gemma4_vl with vision_tower+embed_vision must advertise image input"
    );
    assert_eq!(
        model["capabilities"]["input"]["video"],
        json!(true),
        "gemma4_vl encoder-VL per-frame ViT path must advertise video when env allows"
    );
    assert_eq!(
        model["ax_engine"]["native_multimodal_input_supported"],
        json!(true)
    );
}

#[tokio::test]
async fn models_advertises_named_qwen_and_minicpm_media_towers() {
    let qwen_dir = minimal_tokenizer_artifact("qwen-vl-named-tower-metadata");
    fs::write(
        qwen_dir.join("model-manifest.json"),
        json!({
            "model_family": "qwen3_vl",
            "tensors": [
                {"name": "vision_tower.patch_embed.proj.weight", "role": "other"},
                {"name": "vision_tower.merger.linear_fc1.weight", "role": "other"}
            ]
        })
        .to_string(),
    )
    .expect("Qwen3-VL manifest should write");
    let qwen_app = build_router(native_mlx_openai_builder_state("qwen3-vl-4b", &qwen_dir));
    let (status, json) = json_response(
        &qwen_app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["data"][0]["capabilities"]["input"]["image"], true);
    assert_eq!(json["data"][0]["capabilities"]["input"]["video"], true);

    let minicpm_dir = minimal_tokenizer_artifact("minicpm-v46-named-tower-metadata");
    fs::write(
        minicpm_dir.join("model-manifest.json"),
        json!({
            "model_family": "minicpmv4_6",
            "tensors": [
                {"name": "vision_tower.embeddings.patch_embedding.weight", "role": "other"},
                {"name": "vit_merger.layers.0.mlp.fc1.weight", "role": "other"}
            ]
        })
        .to_string(),
    )
    .expect("MiniCPM-V manifest should write");
    let minicpm_app = build_router(native_mlx_openai_builder_state(
        "minicpm-v-4.6",
        &minicpm_dir,
    ));
    let (status, json) = json_response(
        &minicpm_app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["data"][0]["capabilities"]["input"]["image"], true);
    assert_eq!(json["data"][0]["capabilities"]["input"]["video"], false);
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
