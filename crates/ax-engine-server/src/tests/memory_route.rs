use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use super::fixtures::{json_response, llama_cpp_state};
use crate::routes::build_router;

#[tokio::test]
async fn get_v1_memory_reports_footprint_and_sampled_at() {
    let state = llama_cpp_state();
    let app = build_router(state);
    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/memory")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let sampled_at = body["sampled_at"].as_str().expect("sampled_at");
    assert!(
        sampled_at.contains('T') && sampled_at.ends_with('Z'),
        "{sampled_at}"
    );
    let footprint = body.get("phys_footprint_bytes");
    let peak = body.get("lifetime_max_phys_footprint_bytes");
    let paired = match (footprint, peak) {
        (Some(serde_json::Value::Null), Some(serde_json::Value::Null)) => true,
        (Some(serde_json::Value::Number(footprint)), Some(serde_json::Value::Number(peak))) => {
            matches!(
                (footprint.as_u64(), peak.as_u64()),
                (Some(footprint), Some(peak)) if peak >= footprint
            )
        }
        _ => false,
    };
    assert!(
        paired,
        "footprint and lifetime peak must both be null or both be u64 with peak >= footprint: {footprint:?} {peak:?}"
    );
    #[cfg(feature = "mlx-native-server")]
    {
        assert!(body["mlx"].get("active_bytes").is_some());
        assert!(body["mlx"].get("cache_bytes").is_some());
        assert!(body["mlx"].get("peak_bytes").is_some());
    }
}

#[tokio::test]
async fn get_v1_memory_is_absent_when_api_polish_is_off() {
    let mut state = llama_cpp_state();
    let mut env = *state.env;
    env.api_polish = false;
    state.env = Arc::new(env);
    let app = build_router(state);
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/memory")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("request should succeed");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
