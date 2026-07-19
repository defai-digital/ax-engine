use crate::grpc::AxEngineGrpcService;
use crate::grpc::proto;
use crate::grpc::proto::ax_engine_server::AxEngine;
use crate::openai::generation::populate_native_mlx_output_text;
use crate::openai::schema::OpenAiStreamKind;
use ax_engine_sdk::{
    EngineTokenizer, GenerateFinishReason, GenerateResponse, GenerateStatus, SelectedBackend,
};
use std::sync::mpsc;
use std::time::Duration;

use super::fixtures::{
    llama_cpp_server_state, minimal_tokenizer_artifact, native_mlx_openai_builder_state,
};

#[tokio::test]
async fn grpc_health_remains_available_while_session_is_busy() {
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let service = AxEngineGrpcService::new(state.clone());
    let live = state.snapshot();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    live.generation_service
        .submit(move |_| {
            entered_tx.send(()).expect("test should receive entry");
            release_rx.recv().expect("test should release worker");
        })
        .expect("test job should submit");
    entered_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("test job should start");

    let response = service
        .health(tonic::Request::new(proto::HealthRequest {}))
        .await
        .expect("busy live worker should report healthy")
        .into_inner();
    assert_eq!(response.status, "ok");
    assert_eq!(response.service, "ax-engine-server");

    release_tx.send(()).expect("test should release worker");
    tokio::time::timeout(Duration::from_secs(1), async {
        while live.generation_service.is_busy() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker should become idle");
    let response = service
        .health(tonic::Request::new(proto::HealthRequest {}))
        .await
        .expect("idle live worker should report healthy")
        .into_inner();
    assert_eq!(response.status, "ok");
    assert_eq!(response.service, "ax-engine-server");
}

#[tokio::test]
async fn grpc_models_reports_stable_ax_engine_owner() {
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let service = AxEngineGrpcService::new(state);

    let response = service
        .models(tonic::Request::new(proto::ModelsRequest {}))
        .await
        .expect("models should respond")
        .into_inner();

    assert_eq!(response.object, "list");
    let model = response.data.first().expect("model card");
    assert_eq!(model.owned_by, "ax-engine");
}

/// Regression for gRPC unary chat/completion/generate: native MLX leaves
/// `output_text` unset; the unary handlers must call
/// `populate_native_mlx_output_text` so clients receive decoded content.
#[tokio::test]
async fn populate_native_mlx_output_text_sets_decoded_content_for_unary_contract() {
    let artifact_dir = minimal_tokenizer_artifact("grpc-unary-populate-decode");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    assert_eq!(live.runtime_report.selected_backend, SelectedBackend::Mlx);

    let tokenizer =
        EngineTokenizer::from_model_dir(&artifact_dir).expect("fixture tokenizer loads");
    let output_tokens = tokenizer
        .encode("hello", false)
        .expect("encode hello with fixture vocab");
    assert!(
        !output_tokens.is_empty(),
        "fixture tokenizer must produce tokens for hello"
    );

    let mut response = GenerateResponse {
        request_id: 7,
        model_id: live.model_id.to_string(),
        prompt_tokens: vec![1],
        prompt_text: None,
        output_tokens: output_tokens.clone(),
        output_token_logprobs: Vec::new(),
        output_text: None,
        prompt_token_count: None,
        output_token_count: None,
        status: GenerateStatus::Finished,
        finish_reason: Some(GenerateFinishReason::Stop),
        step_count: 1,
        ttft_step: Some(1),
        route: Default::default(),
        runtime: live.runtime_report.clone(),
        performance: Default::default(),
    };

    populate_native_mlx_output_text(
        &live,
        &mut response,
        OpenAiStreamKind::ChatCompletion,
        false,
    )
    .expect("native MLX decode must succeed for unary contract");

    let decoded = response
        .output_text
        .as_deref()
        .expect("populate must set output_text");
    assert!(
        !decoded.is_empty(),
        "decoded chat content must be non-empty"
    );
    // Round-trip through the same shipped tokenizer path the populate helper uses.
    let expected = tokenizer
        .decode(&output_tokens, true)
        .expect("decode tokens");
    assert_eq!(decoded, expected);

    // Already-populated responses must not be overwritten.
    response.output_text = Some("keep-me".to_string());
    populate_native_mlx_output_text(&live, &mut response, OpenAiStreamKind::Completion, false)
        .expect("second populate is a no-op when text is set");
    assert_eq!(response.output_text.as_deref(), Some("keep-me"));

    std::fs::remove_dir_all(artifact_dir).expect("cleanup artifact dir");
}
