use std::collections::BTreeMap;

use ax_engine_sdk::{EngineStepReport, GenerateRouteReport};
use axum::body::Body;
use axum::http::{Request, StatusCode};

use super::fixtures::{llama_cpp_state, text_response};
use crate::generation::service::TerminalRequestStats;
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
            kv_allocated_blocks_total: 23,
            kv_released_blocks_total: 19,
            kv_cache_evictions_total: 7,
            kv_free_blocks: 1000,
            kv_block_tables: 1,
            kv_prompt_entries: 1,
            kv_block_ref_entries: 24,
            kv_live_prefix_index_keys: 1,
            kv_live_prefix_request_refs: 1,
            kv_cached_blocks: 23,
            kv_cached_child_index_keys: 4,
            kv_cached_child_edges: 5,
            request_active_records: 1,
            request_terminal_snapshots: 12,
            request_terminal_snapshot_order: 12,
            request_terminal_snapshot_bytes: 4096,
            route: Some(GenerateRouteReport {
                crossover_decisions: BTreeMap::from([
                    ("ax_mlx_kv_request_snapshots".to_string(), 1),
                    ("ax_mlx_kv_logical_kib".to_string(), 64),
                    ("ax_mlx_kv_capacity_kib".to_string(), 96),
                    ("ax_mlx_kv_linear_state_kib".to_string(), 4),
                    ("ax_mlx_kv_full_attention_layers".to_string(), 8),
                    ("ax_mlx_kv_linear_state_layers".to_string(), 2),
                    ("ax_mtp_draft_tokens".to_string(), 7),
                    ("ax_mtp_accepted_tokens".to_string(), 5),
                    ("ax_mtp_direct_fallback_steps".to_string(), 1),
                    ("ax_mtp_mtp_only_accept_rate_ewma_x1000".to_string(), 714),
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
    assert!(body.contains("ax_engine_scheduled_requests_total 4\n"));
    assert!(body.contains("ax_engine_scheduled_tokens_total 22\n"));
    assert!(body.contains("ax_engine_step_scheduled_requests 1\n"));
    assert!(body.contains("ax_engine_step_scheduled_tokens 5\n"));
    assert!(body.contains("ax_engine_step_kv_usage_blocks 4\n"));
    assert!(body.contains("ax_engine_step_prefix_hits_total 3\n"));
    // MTP/speculation series: counters accumulate, the EWMA gauge holds the
    // latest reported value and is not zeroed by steps without MTP telemetry.
    assert!(body.contains("ax_engine_mtp_draft_tokens_total 7\n"));
    assert!(body.contains("ax_engine_mtp_accepted_tokens_total 5\n"));
    assert!(body.contains("ax_engine_mtp_direct_fallback_steps_total 1\n"));
    assert!(body.contains("ax_engine_mtp_accept_rate_ewma_x1000 714\n"));
    assert!(body.contains("ax_engine_mtp_accepted_tokens_total{model=\"qwen3\"} 5\n"));
    assert!(body.contains("ax_engine_kv_allocated_blocks_total 23\n"));
    assert!(body.contains("ax_engine_kv_released_blocks_total 19\n"));
    assert!(body.contains("ax_engine_kv_cache_evictions_total 7\n"));
    assert!(body.contains("ax_engine_kv_cached_child_edges 5\n"));
    assert!(body.contains("ax_engine_request_terminal_snapshots 12\n"));
    assert!(body.contains("ax_engine_request_terminal_snapshot_bytes 4096\n"));
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

/// Node-saturation series follow the AX Serving fleet-dispatch contract:
/// config-derived series are always present once a model is loaded, while
/// measurement-derived series stay hidden until real traffic produces them.
#[tokio::test]
async fn metrics_saturation_series_feed_fleet_dispatch_contract() {
    let state = llama_cpp_state();
    let metrics = state.metrics.clone();
    let kv_blocks_total = u64::from(state.snapshot().session_config.kv_config.total_blocks);
    let app = build_router(state);

    let scrape = |app: &axum::Router| {
        let app = app.clone();
        async move {
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
            body
        }
    };

    let body = scrape(&app).await;
    assert!(body.contains(&format!("ax_runtime_kv_pages_total {kv_blocks_total}\n")));
    assert!(body.contains("ax_runtime_queue_depth 0\n"));
    // The batched-decode cohort cap governs only native MLX models; a
    // delegated-only (llama.cpp) node must not advertise batch headroom.
    assert!(!body.contains("ax_runtime_max_batch_size"));
    // Measurement-derived series stay hidden before any step or request.
    assert!(!body.contains("ax_runtime_kv_utilization"));
    assert!(!body.contains("ax_engine_step_waiting_requests"));
    assert!(!body.contains("ax_runtime_ttft_p95_ms"));
    assert!(!body.contains("ax_runtime_decode_tok_per_sec"));
    // The scrape itself is a counted HTTP request, so the error ratio is
    // already exported — at exactly 0 while nothing has failed.
    assert!(body.contains("ax_runtime_error_rate 0\n"));

    metrics.record_step_report(
        "qwen3",
        &EngineStepReport {
            kv_usage_blocks: 4,
            waiting_requests: 2,
            ..Default::default()
        },
    );
    for ttft_us in [100_000, 200_000, 300_000] {
        metrics.record_terminal_request(TerminalRequestStats {
            ttft_us: Some(ttft_us),
            decode_tok_per_sec: None,
        });
    }
    metrics.record_terminal_request(TerminalRequestStats {
        ttft_us: Some(400_000),
        decode_tok_per_sec: Some(84.0),
    });
    metrics.begin_http_request();
    metrics.finish_http_request(StatusCode::INTERNAL_SERVER_ERROR);

    let body = scrape(&app).await;
    assert!(body.contains("ax_engine_step_waiting_requests 2\n"));
    assert!(body.contains("ax_engine_step_waiting_requests{model=\"qwen3\"} 2\n"));
    // Queue depth combines worker command queue (0) and scheduler waiting (2).
    assert!(body.contains("ax_runtime_queue_depth 2\n"));
    #[allow(clippy::cast_precision_loss)]
    let expected_kv_utilization = 4.0_f64 / kv_blocks_total as f64;
    assert!(
        body.contains(&format!(
            "ax_runtime_kv_utilization {expected_kv_utilization}\n"
        )),
        "body must contain kv utilization {expected_kv_utilization}"
    );
    // Nearest-rank p95 over [100, 200, 300, 400] ms is 400.
    assert!(body.contains("ax_runtime_ttft_p95_ms 400\n"));
    // A single decode sample seeds the EWMA directly.
    assert!(body.contains("ax_runtime_decode_tok_per_sec 84\n"));
    assert!(body.contains("ax_engine_generation_completed_requests_total 4\n"));
    // One failed request out of all counted requests (including the scrapes
    // themselves): strictly positive ratio below 1.
    assert!(body.contains("ax_runtime_error_rate 0."));
}
