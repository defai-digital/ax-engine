use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use super::fixtures::{json_response, llama_cpp_server_state, llama_cpp_state};
use crate::routes::build_router;
use ax_engine_sdk::SelectedBackend;

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
    assert_eq!(body["models"], serde_json::json!([]));
    let sampled_at = body["sampled_at"].as_str().expect("sampled_at");
    assert!(
        sampled_at.contains('T') && sampled_at.ends_with('Z'),
        "{sampled_at}"
    );
    let footprint = body.get("phys_footprint_bytes");
    let peak = body.get("lifetime_max_phys_footprint_bytes");
    match (footprint, peak) {
        (Some(serde_json::Value::Null), Some(serde_json::Value::Null)) => {}
        (Some(serde_json::Value::Number(footprint)), Some(serde_json::Value::Number(peak))) => {
            let live = footprint.as_u64().expect("phys_footprint_bytes is a u64");
            let peak = peak
                .as_u64()
                .expect("lifetime_max_phys_footprint_bytes is a u64");
            assert!(peak >= live, "live footprint {live}, lifetime peak {peak}");
        }
        _ => panic!(
            "footprint and lifetime peak must both be null or both be u64 with peak >= live: {footprint:?} {peak:?}"
        ),
    }
    #[cfg(feature = "mlx-native-server")]
    {
        assert!(body["mlx"].get("active_bytes").is_some());
        assert!(body["mlx"].get("cache_bytes").is_some());
        assert!(body["mlx"].get("peak_bytes").is_some());
    }
}

#[tokio::test]
async fn get_v1_memory_reports_ready_native_models_by_generation() {
    let state = llama_cpp_state();
    let sibling = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let mut native = sibling.snapshot();
    native.model_id = Arc::new("flash-next".to_string());
    native.runtime_report.selected_backend = SelectedBackend::Mlx;
    state.publish_live(native, false);
    let app = build_router(state.clone());
    let sample = || {
        Request::builder()
            .uri("/v1/memory")
            .body(Body::empty())
            .unwrap()
    };
    let (status, body) = json_response(&app, sample()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["models"],
        serde_json::json!([{"id": "flash-next", "expert_streaming_active": false}])
    );

    state.remove_live("flash-next").expect("remove sibling");
    let (status, body) = json_response(&app, sample()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["models"], serde_json::json!([]));
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
