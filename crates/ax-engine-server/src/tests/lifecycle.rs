use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::sync::atomic::Ordering;

use super::fixtures::{
    json_request_body, json_response, llama_cpp_server_state, normalize_measurement_fields,
    sample_http_request, sample_sdk_request, sdk_session_for_state,
    spawn_llama_cpp_completion_stream_server,
};

#[tokio::test]
async fn submit_request_rejects_while_a_model_load_is_in_progress() {
    // Regression test for the race this check closes: `load_model` only
    // checks for in-flight requests once, before spawning a load that can
    // take tens of seconds — it never re-checks or blocks new submissions
    // during that window. Simulate being inside that window directly via
    // `state.loading` rather than a real slow load.
    let (llama_server_url, llama_cpp_server_handle) =
        spawn_llama_cpp_completion_stream_server(0, vec![], |_payload| {});
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

    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(
        body.get("error")
            .and_then(|error| error.get("code"))
            .and_then(|code| code.as_str()),
        Some("model_loading")
    );

    llama_cpp_server_handle
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
    let app = build_router(llama_cpp_server_state(llama_server_url));
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

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
}
