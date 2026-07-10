use crate::app_state::ServerLimits;
use crate::grpc::AxEngineGrpcService;
use crate::grpc::proto;
use crate::grpc::proto::ax_engine_server::AxEngine;
use crate::rate_limit::RateLimitConfig;
use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};

use super::fixtures::{
    json_request_body, json_response, llama_cpp_state, sample_text_http_request, text_response,
};

#[tokio::test]
async fn concurrency_limit_is_shared_by_http_and_grpc_engine_jobs() {
    let state = llama_cpp_state().with_limits(ServerLimits {
        max_concurrent_requests: Some(1),
        ..Default::default()
    });
    let held_permit = state
        .admission
        .try_admit()
        .expect("first engine job should be admitted");
    let app = build_router(state.clone());

    let (health_status, _, _) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(health_status, StatusCode::OK);

    let (http_status, http_body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_text_http_request(
                "overloaded",
                1,
            ))))
            .unwrap(),
    )
    .await;
    assert_eq!(http_status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        http_body
            .get("error")
            .and_then(|error| error.get("code"))
            .and_then(|code| code.as_str()),
        Some("concurrency_limit")
    );

    let grpc_error = AxEngineGrpcService::new(state.clone())
        .generate(tonic::Request::new(proto::GenerateRequest {
            model: "qwen3".to_string(),
            input_tokens: Vec::new(),
            input_text: "overloaded".to_string(),
            max_output_tokens: 1,
            sampling: None,
            metadata: String::new(),
        }))
        .await
        .expect_err("gRPC should share the occupied engine-job capacity");
    assert_eq!(grpc_error.code(), tonic::Code::ResourceExhausted);
    assert_eq!(state.admission.active_jobs(), 1);

    drop(held_permit);
    assert_eq!(state.admission.active_jobs(), 0);
}

#[tokio::test]
async fn model_drain_rejects_new_http_and_grpc_engine_jobs() {
    let state = llama_cpp_state();
    let drain = state.admission.begin_drain();
    let app = build_router(state.clone());

    let (http_status, http_body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_text_http_request(
                "draining", 1,
            ))))
            .unwrap(),
    )
    .await;
    assert_eq!(http_status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(http_body["error"]["code"], "model_loading");

    let grpc_error = AxEngineGrpcService::new(state)
        .generate(tonic::Request::new(proto::GenerateRequest {
            model: "qwen3".to_string(),
            input_tokens: Vec::new(),
            input_text: "draining".to_string(),
            max_output_tokens: 1,
            sampling: None,
            metadata: String::new(),
        }))
        .await
        .expect_err("gRPC should reject jobs during model drain");
    assert_eq!(grpc_error.code(), tonic::Code::Unavailable);

    drop(drain);
}

#[tokio::test]
async fn rate_limit_sheds_load_after_burst_is_exhausted() {
    let app = build_router(llama_cpp_state().with_limits(ServerLimits {
        rate_limit: Some(RateLimitConfig {
            rps: 0.0001, // effectively no refill within the test's duration
            burst: 1.0,
        }),
        ..Default::default()
    }));

    let first = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let (first_status, _, _) = text_response(&app, first).await;
    assert_eq!(
        first_status,
        StatusCode::OK,
        "first request consumes the sole burst token"
    );

    let second = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let (second_status, _, body) = text_response(&app, second).await;
    assert_eq!(second_status, StatusCode::TOO_MANY_REQUESTS);
    assert!(body.contains("rate limit exceeded"));
}

#[tokio::test]
async fn default_limits_preserve_unlimited_behavior() {
    let app = build_router(llama_cpp_state());

    for _ in 0..5 {
        let (status, _, _) = text_response(
            &app,
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
}
