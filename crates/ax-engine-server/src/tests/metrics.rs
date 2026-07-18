use ax_engine_sdk::EngineStepReport;
use axum::body::Body;
use axum::http::{Request, StatusCode};

use super::fixtures::{llama_cpp_state, text_response};
use crate::routes::build_router;

/// `/metrics` must stay a passive read: engine-step gauges only reflect
/// reports recorded by the generation worker, and scraping
/// before any step keeps them hidden entirely.
#[tokio::test]
async fn metrics_step_gauges_appear_only_after_recorded_steps() {
    let state = llama_cpp_state();
    let metrics = state.metrics.clone();
    let app = build_router(state);

    let (status, _, body) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        !body.contains("ax_engine_step_scheduled_requests"),
        "engine-step gauges must stay hidden until a step is observed"
    );
    assert!(body.contains("ax_engine_jobs_in_flight 0\n"));
    assert!(body.contains("ax_engine_generation_jobs_pending 0\n"));
    assert!(body.contains("ax_engine_generation_commands_queued 0\n"));
    assert!(body.contains("ax_engine_generation_command_queue_capacity 256\n"));
    assert!(body.contains("ax_engine_generation_active_streams 0\n"));
    assert!(body.contains("ax_engine_generation_buffered_stream_events 0\n"));
    assert!(body.contains("ax_engine_generation_saturated_commands_total 0\n"));
    assert!(body.contains("ax_engine_generation_stream_backlog_overflows_total 0\n"));
    assert!(body.contains("ax_engine_generation_worker_ready 1\n"));

    metrics.record_step_report(
        "qwen3",
        &EngineStepReport {
            scheduled_requests: 3,
            scheduled_tokens: 17,
            kv_usage_blocks: 9,
            prefix_hits: 2,
            ..Default::default()
        },
    );
    metrics.record_step_report(
        "qwen3",
        &EngineStepReport {
            scheduled_requests: 1,
            scheduled_tokens: 5,
            kv_usage_blocks: 4,
            prefix_hits: 1,
            ..Default::default()
        },
    );

    let (status, _, body) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // Gauges hold the latest step; prefix hits accumulate across steps.
    assert!(body.contains("ax_engine_steps_total 2\n"));
    assert!(body.contains("ax_engine_step_scheduled_requests 1\n"));
    assert!(body.contains("ax_engine_step_scheduled_tokens 5\n"));
    assert!(body.contains("ax_engine_step_kv_usage_blocks 4\n"));
    assert!(body.contains("ax_engine_step_prefix_hits_total 3\n"));
    // Per-model labeled series accompany the unlabeled aggregates.
    assert!(body.contains("ax_engine_steps_total{model=\"qwen3\"} 2\n"));
    assert!(body.contains("ax_engine_step_prefix_hits_total{model=\"qwen3\"} 3\n"));
}
