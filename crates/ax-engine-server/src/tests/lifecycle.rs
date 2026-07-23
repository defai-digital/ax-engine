use crate::app_state::build_live_state;
use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::sync::atomic::Ordering;
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use super::fixtures::{
    json_request_body, json_response, llama_cpp_server_state, normalize_measurement_fields,
    sample_http_request, sample_sdk_request, sdk_session_for_state,
    spawn_llama_cpp_completion_stream_server,
};

#[tokio::test]
async fn submit_request_is_not_globally_blocked_by_a_model_load() {
    // The lifecycle-mutation flag serializes registry changes; it must not be
    // used as a process-wide request gate. Target generations use their own
    // admission controller when they actually need to drain.
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![json!({
            "content": "done",
            "tokens": [4],
            "stop": true,
            "stop_type": "limit"
        })],
        |_| {},
    );
    let state = llama_cpp_server_state(llama_server_url);
    state.loading.store(true, Ordering::Release);
    let app = build_router(state);

    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_http_request(
                &[1, 2, 3],
                2,
            ))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED, "{body}");

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}

#[tokio::test]
async fn step_routes_to_an_explicit_loaded_model() {
    let state = super::fixtures::llama_cpp_state();
    let config = state.snapshot().session_config.as_ref().clone();
    let second = build_live_state("gemma-4-12b-it".to_string(), config)
        .expect("second delegated state should build");
    state.publish_live(second, false);
    let app = build_router(state.clone());

    let (status, _) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/step?model=gemma-4-12b-it")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (missing_status, missing) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/step?model=not-loaded")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(missing_status, StatusCode::BAD_REQUEST);
    assert_eq!(missing["error"]["code"], json!("model_not_found"));

    let removed = state
        .remove_live("gemma-4-12b-it")
        .expect("second model should remove");
    removed.retire().await.expect("second worker should retire");
}

#[tokio::test]
async fn request_lifecycle_routes_directly_past_an_unrelated_saturated_worker() {
    let (llama_server_url, llama_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![json!({
            "content": "hello",
            "tokens": [4],
            "stop": false
        })],
        |_| {},
    );
    let state = super::fixtures::llama_cpp_server_state(llama_server_url);
    let config = state.snapshot().session_config.as_ref().clone();
    let second = build_live_state("gemma-4-12b-it".to_string(), config)
        .expect("second delegated state should build");
    state.publish_live(second, false);
    let app = build_router(state.clone());

    let mut request = sample_http_request(&[1, 2, 3], 2);
    request["model"] = json!("gemma-4-12b-it");
    let (submit_status, submit) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&request)))
            .unwrap(),
    )
    .await;
    assert_eq!(submit_status, StatusCode::CREATED, "{submit}");
    let request_id = submit["request_id"]
        .as_u64()
        .expect("request id should be returned");

    let unrelated_service = state.snapshot().generation_service;
    let (entered_tx, entered_rx) = std_mpsc::channel();
    let (release_tx, release_rx) = std_mpsc::channel();
    unrelated_service
        .submit(move |_| {
            let _ = entered_tx.send(());
            let _ = release_rx.recv();
        })
        .expect("blocking command should enqueue");
    entered_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("unrelated worker should block");
    for _ in 0..unrelated_service.command_queue_capacity() {
        unrelated_service
            .submit(|_| {})
            .expect("unrelated queue should fill to capacity");
    }

    let (snapshot_status, snapshot) = json_response(
        &app,
        Request::builder()
            .uri(format!("/v1/requests/{request_id}"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(snapshot_status, StatusCode::OK);
    assert_eq!(snapshot["request_id"], json!(request_id));

    let (cancel_status, cancelled) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri(format!("/v1/requests/{request_id}/cancel"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(cancel_status, StatusCode::OK);
    assert_eq!(cancelled["request_id"], json!(request_id));
    assert!(state.request_owner_is_terminal(request_id));

    release_tx
        .send(())
        .expect("unrelated worker should release");
    let deadline = Instant::now() + Duration::from_secs(1);
    while unrelated_service.is_busy() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!unrelated_service.is_busy());
    let removed = state
        .remove_live("gemma-4-12b-it")
        .expect("second model should remove");
    removed.retire().await.expect("second worker should retire");
    llama_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}

#[tokio::test]
async fn llama_cpp_stepwise_request_endpoints_share_sdk_lifecycle() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        2,
        vec![
            serde_json::json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            }),
            serde_json::json!({
                "content": " world",
                "tokens": [5],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        |payload| {
            assert_eq!(payload.get("prompt"), Some(&json!([1, 2, 3])));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
        },
    );
    let state = llama_cpp_server_state(llama_server_url);
    let app = build_router(state.clone());
    let request_body = sample_http_request(&[1, 2, 3], 2);
    let (status, submit_json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&request_body)))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    let request_id = 1u64;
    assert_eq!(state.admission.active_jobs(), 1);

    let mut sdk_session = sdk_session_for_state(&state);
    sdk_session
        .submit_generate_with_request_id(request_id, sample_sdk_request(&[1, 2, 3], 2))
        .expect("sdk llama.cpp submit should succeed");
    let expected_submit = serde_json::to_value(
        sdk_session
            .request_report(request_id)
            .expect("sdk llama.cpp report should exist"),
    )
    .expect("sdk llama.cpp submit report should serialize");
    assert_eq!(submit_json, expected_submit);

    let mut observed_steps = 0_u64;
    for _ in 0..4 {
        let expected_snapshot = serde_json::to_value(
            sdk_session
                .request_report(request_id)
                .expect("sdk llama.cpp request report should still exist"),
        )
        .expect("sdk llama.cpp snapshot should serialize");
        let (snapshot_status, snapshot_json) = json_response(
            &app,
            Request::builder()
                .uri(format!("/v1/requests/{request_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(snapshot_status, StatusCode::OK);
        assert_eq!(snapshot_json, expected_snapshot);

        if expected_snapshot
            .get("state")
            .and_then(|value| value.as_str())
            == Some("finished")
        {
            break;
        }

        let expected_step = serde_json::to_value(
            sdk_session
                .step_report()
                .expect("sdk llama.cpp step should succeed"),
        )
        .expect("sdk llama.cpp step should serialize");
        let (step_status, step_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/step")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(step_status, StatusCode::OK);
        observed_steps += 1;
        let mut step_json = step_json;
        let mut expected_step = expected_step;
        normalize_measurement_fields(&mut step_json);
        normalize_measurement_fields(&mut expected_step);
        assert_eq!(step_json, expected_step);
    }

    let expected_terminal = serde_json::to_value(
        sdk_session
            .request_report(request_id)
            .expect("sdk llama.cpp terminal request should exist"),
    )
    .expect("sdk llama.cpp terminal snapshot should serialize");
    let (terminal_status, terminal_json) = json_response(
        &app,
        Request::builder()
            .uri(format!("/v1/requests/{request_id}"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(terminal_status, StatusCode::OK);
    assert_eq!(terminal_json, expected_terminal);
    assert_eq!(state.admission.active_jobs(), 0);
    assert_eq!(state.snapshot().generation_service.pending_jobs(), 0);
    let step_models = state.metrics.engine_step_gauges_per_model();
    assert!(
        !step_models.is_empty(),
        "step metrics should be observed per model"
    );
    let steps_total: u64 = step_models.iter().map(|(_, step)| step.steps_total).sum();
    assert_eq!(steps_total, observed_steps);

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}

#[tokio::test]
async fn llama_cpp_stepwise_request_endpoints_aggregate_multiple_active_requests() {
    let expected_prompts = vec![json!([1, 2, 3]), json!([7, 8, 9])];
    let expected_prompts_for_request = expected_prompts.clone();
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        4,
        vec![
            serde_json::json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            }),
            serde_json::json!({
                "content": " world",
                "tokens": [5],
                "stop": true,
                "stop_type": "limit"
            }),
        ],
        move |payload| {
            let prompt = payload.get("prompt").expect("prompt should be present");
            assert!(
                expected_prompts_for_request
                    .iter()
                    .any(|candidate| prompt == candidate)
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
        },
    );
    let state = llama_cpp_server_state(llama_server_url);
    let app = build_router(state.clone());

    let (first_submit_status, first_submit_json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_http_request(
                &[1, 2, 3],
                2,
            ))))
            .unwrap(),
    )
    .await;
    let (second_submit_status, second_submit_json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_http_request(
                &[7, 8, 9],
                2,
            ))))
            .unwrap(),
    )
    .await;
    assert_eq!(first_submit_status, StatusCode::CREATED);
    assert_eq!(second_submit_status, StatusCode::CREATED);
    assert_eq!(state.admission.active_jobs(), 2);

    let first_request_id = first_submit_json
        .get("request_id")
        .and_then(|value| value.as_u64())
        .expect("first request_id should exist");
    let second_request_id = second_submit_json
        .get("request_id")
        .and_then(|value| value.as_u64())
        .expect("second request_id should exist");

    let mut sdk_session = sdk_session_for_state(&state);
    sdk_session
        .submit_generate_with_request_id(first_request_id, sample_sdk_request(&[1, 2, 3], 2))
        .expect("first sdk llama.cpp submit should succeed");
    sdk_session
        .submit_generate_with_request_id(second_request_id, sample_sdk_request(&[7, 8, 9], 2))
        .expect("second sdk llama.cpp submit should succeed");

    for _ in 0..2 {
        let expected_step = serde_json::to_value(
            sdk_session
                .step_report()
                .expect("sdk llama.cpp aggregated step should succeed"),
        )
        .expect("sdk llama.cpp step should serialize");
        let (step_status, step_json) = json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/step")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(step_status, StatusCode::OK);
        let mut step_json = step_json;
        let mut expected_step = expected_step;
        normalize_measurement_fields(&mut step_json);
        normalize_measurement_fields(&mut expected_step);
        assert_eq!(step_json, expected_step);

        for request_id in [first_request_id, second_request_id] {
            let expected_snapshot = serde_json::to_value(
                sdk_session
                    .request_report(request_id)
                    .expect("sdk llama.cpp snapshot should exist"),
            )
            .expect("sdk llama.cpp snapshot should serialize");
            let (snapshot_status, snapshot_json) = json_response(
                &app,
                Request::builder()
                    .uri(format!("/v1/requests/{request_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(snapshot_status, StatusCode::OK);
            assert_eq!(snapshot_json, expected_snapshot);
        }
    }
    assert_eq!(state.admission.active_jobs(), 0);
    assert_eq!(state.snapshot().generation_service.pending_jobs(), 0);

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}

#[tokio::test]
async fn llama_cpp_cancel_endpoint_surfaces_cancelled_snapshot() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![serde_json::json!({
            "content": "hello",
            "tokens": [4],
            "stop": false
        })],
        |payload| {
            assert_eq!(payload.get("prompt"), Some(&json!([7, 8, 9])));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
        },
    );
    let state = llama_cpp_server_state(llama_server_url);
    let app = build_router(state.clone());
    let (_, submit_json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_http_request(
                &[7, 8, 9],
                2,
            ))))
            .unwrap(),
    )
    .await;
    let request_id = submit_json
        .get("request_id")
        .and_then(|value| value.as_u64())
        .expect("request_id should exist");

    let (status, cancel_json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri(format!("/v1/requests/{request_id}/cancel"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        cancel_json.get("state").and_then(|value| value.as_str()),
        Some("cancelled")
    );
    assert_eq!(
        cancel_json
            .get("cancel_requested")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(state.admission.active_jobs(), 0);
    assert_eq!(state.snapshot().generation_service.pending_jobs(), 0);

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}

#[tokio::test]
async fn step_error_cancels_request_and_releases_admission() {
    let (llama_server_url, llama_cpp_server_handle) =
        spawn_llama_cpp_completion_stream_server(1, vec![], |_| {});
    let state = llama_cpp_server_state(llama_server_url);
    let app = build_router(state.clone());
    let (submit_status, submit_json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_http_request(
                &[7, 8, 9],
                2,
            ))))
            .unwrap(),
    )
    .await;
    assert_eq!(submit_status, StatusCode::CREATED);
    let request_id = submit_json
        .get("request_id")
        .and_then(Value::as_u64)
        .expect("request_id should exist");

    let (step_status, _) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/step")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(step_status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(state.admission.active_jobs(), 0);
    assert_eq!(state.snapshot().generation_service.pending_jobs(), 0);

    let (cancel_status, cancel_json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri(format!("/v1/requests/{request_id}/cancel"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(cancel_status, StatusCode::OK);
    assert_eq!(
        cancel_json.get("state").and_then(Value::as_str),
        Some("cancelled")
    );
    assert_eq!(state.admission.active_jobs(), 0);

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}
