use crate::app_state::ServerLimits;
use crate::rate_limit::RateLimitConfig;
use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};

use super::fixtures::{llama_cpp_state, text_response};

#[tokio::test]
async fn concurrency_limit_sheds_load_via_state_limits() {
    let app = build_router(llama_cpp_state().with_limits(ServerLimits {
        max_concurrent_requests: Some(0), // any in-flight request exceeds a 0 cap
        ..Default::default()
    }));

    // The semaphore is only constructed when the cap is > 0 in practice, but
    // a 0-permit `Semaphore` always fails `try_acquire_owned`, so this
    // exercises the router wiring end-to-end without needing a second
    // concurrent request in flight.
    let (status, _, body) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert!(body.contains("maximum concurrent request limit"));
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
