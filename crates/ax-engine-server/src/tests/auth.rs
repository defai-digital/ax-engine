use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};

use super::fixtures::{json_response, llama_cpp_state, text_response};

#[tokio::test]
async fn api_key_auth_rejects_missing_or_invalid_bearer_token() {
    let app = build_router(llama_cpp_state().with_api_key(Some("secret".to_string())));

    let (status, _, body) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(body.contains("missing or invalid bearer token"));

    let (status, _, _) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header(header::AUTHORIZATION, "Bearer wrong")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn api_key_auth_accepts_valid_bearer_token_and_leaves_health_public() {
    let app = build_router(llama_cpp_state().with_api_key(Some("secret".to_string())));

    let (health_status, health_json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(health_status, StatusCode::OK);
    assert_eq!(health_json["status"], "ok");

    let (models_status, models_json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header(header::AUTHORIZATION, "Bearer secret")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(models_status, StatusCode::OK);
    assert_eq!(models_json["object"], "list");
}

#[tokio::test]
async fn metrics_route_requires_auth_and_emits_prometheus_text() {
    let app = build_router(llama_cpp_state().with_api_key(Some("secret".to_string())));

    let (unauthorized_status, _, _) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(unauthorized_status, StatusCode::UNAUTHORIZED);

    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/metrics")
            .header(header::AUTHORIZATION, "Bearer secret")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.contains("text/plain"));
    assert!(body.contains("# TYPE ax_engine_http_requests_total counter"));
    assert!(body.contains("ax_engine_http_requests_in_flight"));
}
