use std::collections::BTreeMap;

use ax_engine_sdk::{EngineStepReport, GenerateRouteReport};
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
    assert!(body.contains("ax_engine_model_jobs_in_flight{model=\"qwen3\"} 0\n"));
    assert!(body.contains("ax_engine_generation_jobs_pending 0\n"));
    assert!(body.contains("ax_engine_generation_commands_queued 0\n"));
    assert!(body.contains("ax_engine_generation_command_queue_capacity 256\n"));
    assert!(body.contains("ax_engine_generation_active_streams 0\n"));
    assert!(body.contains("ax_engine_generation_buffered_stream_events 0\n"));
    assert!(body.contains("ax_engine_generation_saturated_commands_total 0\n"));
    assert!(body.contains("ax_engine_generation_stream_backlog_overflows_total 0\n"));
    assert!(body.contains("ax_engine_generation_worker_ready 1\n"));
    assert!(body.contains("ax_engine_model_memory_weight_artifact_available{model=\"qwen3\"} 0\n"));
    assert!(body.contains("ax_engine_model_memory_kv_report_available{model=\"qwen3\"} 0\n"));

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
            route: Some(GenerateRouteReport {
                crossover_decisions: BTreeMap::from([
                    ("ax_mlx_kv_request_snapshots".to_string(), 1),
                    ("ax_mlx_kv_logical_kib".to_string(), 64),
                    ("ax_mlx_kv_capacity_kib".to_string(), 96),
                    ("ax_mlx_kv_linear_state_kib".to_string(), 4),
                    ("ax_mlx_kv_full_attention_layers".to_string(), 8),
                    ("ax_mlx_kv_linear_state_layers".to_string(), 2),
                ]),
                ..Default::default()
            }),
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
    assert!(body.contains("ax_engine_model_memory_kv_report_available{model=\"qwen3\"} 1\n"));
    assert!(body.contains("ax_engine_model_memory_kv_capacity_bytes{model=\"qwen3\"} 98304\n"));
    assert!(body.contains("ax_engine_model_memory_kv_physical_bytes{model=\"qwen3\"} 102400\n"));
    assert!(body.contains(
        "ax_engine_model_kv_topology_info{model=\"qwen3\",attention_storage=\"contiguous\",sliding_storage=\"none\",recurrent_state=\"present\",rollback_strategy=\"restore_replay\"} 1\n"
    ));
}
