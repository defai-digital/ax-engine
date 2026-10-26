use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;

use super::fixtures::{
    json_request_body, json_response, llama_cpp_server_state, llama_cpp_state,
    spawn_llama_cpp_completion_stream_server,
};

#[tokio::test]
async fn model_load_rejects_empty_model_id_before_touching_path() {
    let app = build_router(llama_cpp_state());
    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "  ",
                "model_path": "/definitely/not/a/model"
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], json!("invalid_request"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("model_id must not be empty")
    );
}

#[tokio::test]
async fn model_load_rejects_missing_model_path() {
    let app = build_router(llama_cpp_state());
    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "ax-engine/qwen3-coder-next",
                "model_path": "/definitely/not/a/model"
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], json!("invalid_request"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("model_path does not exist")
    );
}

#[tokio::test]
async fn model_add_rejects_memory_constrained_policy() {
    let temp_model_dir = std::env::temp_dir().join(format!(
        "ax-engine-server-model-add-policy-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_model_dir).expect("temp model dir should create");
    let app = build_router(llama_cpp_state());
    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "qwen3.6-27b",
                "model_path": temp_model_dir,
                "load_policy": "memory_constrained",
                "load_mode": "add"
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], json!("invalid_request"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("incompatible")
    );
    std::fs::remove_dir_all(temp_model_dir).expect("temp model dir should clean up");
}

#[tokio::test]
async fn model_load_rejects_replace_with_make_default_false() {
    let temp_model_dir = std::env::temp_dir().join(format!(
        "ax-engine-server-model-make-default-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_model_dir).expect("temp model dir should create");
    let app = build_router(llama_cpp_state());
    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "qwen3.6-27b",
                "model_path": temp_model_dir,
                "make_default": false
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], json!("invalid_request"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("make_default")
    );
    std::fs::remove_dir_all(temp_model_dir).expect("temp model dir should clean up");
}

#[tokio::test]
async fn model_unload_rejects_last_or_unknown_model() {
    let state = llama_cpp_state();
    let _active_request = state
        .admission
        .try_admit()
        .expect("test request should hold admission open");
    let app = build_router(state);

    let (last_status, last_body) = tokio::time::timeout(
        std::time::Duration::from_millis(250),
        json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/model/unload")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(&json!({"model_id": "qwen3"}))))
                .unwrap(),
        ),
    )
    .await
    .expect("invalid unload must not wait for active traffic");
    assert_eq!(last_status, StatusCode::CONFLICT);
    assert_eq!(last_body["error"]["code"], json!("last_model"));

    let (missing_status, missing_body) = tokio::time::timeout(
        std::time::Duration::from_millis(250),
        json_response(
            &app,
            Request::builder()
                .method("POST")
                .uri("/v1/model/unload")
                .header("content-type", "application/json")
                .body(Body::from(json_request_body(
                    &json!({"model_id": "not-loaded"}),
                )))
                .unwrap(),
        ),
    )
    .await
    .expect("unknown unload must not wait for active traffic");
    assert_eq!(missing_status, StatusCode::BAD_REQUEST);
    assert_eq!(missing_body["error"]["code"], json!("model_not_found"));
}

#[tokio::test]
async fn model_load_rejects_delegated_backend_before_relabeling_model() {
    let temp_model_dir = std::env::temp_dir().join(format!(
        "ax-engine-server-model-load-delegated-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_model_dir).expect("temp model dir should create");

    let app = build_router(llama_cpp_state());
    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "ax-engine/qwen3-coder-next",
                "model_path": temp_model_dir
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], json!("unsupported_backend"));
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("only supported on the MLX-native backend")
    );

    std::fs::remove_dir_all(temp_model_dir).expect("temp model dir should clean up");
}

#[tokio::test]
async fn model_load_rejects_swap_while_requests_are_in_flight() {
    // Request state lives entirely inside the EngineSession instance with no
    // cross-session registry: swapping the session out from under a
    // non-terminal /v1/requests entry would silently orphan it (the client's
    // next /v1/requests/:id or /v1/step call would 404 instead of surfacing
    // a real terminal state). The load must fail closed instead.
    //
    // The stepwise submit path opens the streaming HTTP connection
    // synchronously, so a real (if minimal) mock server is required even
    // though this test never calls /v1/step. One non-terminal chunk keeps
    // the request Active without needing to drive it to completion.
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![serde_json::json!({
            "content": "partial",
            "tokens": [9],
            "stop": false
        })],
        |_payload| {},
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));

    // A real (if otherwise-irrelevant) directory: the handler checks
    // model_path existence before reaching the in-flight-requests check
    // this test exercises.
    let temp_model_dir = std::env::temp_dir().join(format!(
        "ax-engine-server-model-load-in-flight-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_model_dir).expect("temp model dir should create");

    let (submit_status, submit_body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "input_tokens": [1, 2, 3],
                "max_output_tokens": 4
            }))))
            .unwrap(),
    )
    .await;
    assert_eq!(submit_status, StatusCode::CREATED);
    let request_id = submit_body["request_id"]
        .as_u64()
        .expect("submitted request should report an id");

    let (status, body) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/model/load")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model_id": "ax-engine/qwen3-coder-next",
                "model_path": temp_model_dir
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], json!("requests_in_flight"));

    // The in-flight request must still be reachable afterward — the load was
    // rejected before touching the session, not silently orphaned.
    let (snapshot_status, snapshot_body) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri(format!("/v1/requests/{request_id}"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(snapshot_status, StatusCode::OK);
    assert_eq!(snapshot_body["request_id"], json!(request_id));

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
    std::fs::remove_dir_all(temp_model_dir).expect("temp model dir should clean up");
}

#[tokio::test]
async fn unloading_repeated_models_releases_workers_and_contexts() {
    use crate::model_load::{UnloadWaitPolicy, perform_unload};
    let state = llama_cpp_state();
    let sibling = state.snapshot().generation_service;
    for index in 0..12 {
        let model_id = format!("temporary-{index}");
        let live = crate::app_state::build_live_state(
            model_id.clone(),
            state.snapshot().session_config.as_ref().clone(),
            &crate::args::ServerEnvConfig::default(),
        )
        .expect("fixture model builds");
        let service = live.generation_service.clone();
        let context = std::sync::Arc::downgrade(&live.stateless_generate_context);
        state.publish_live(live, false);
        let policy = if index % 2 == 0 {
            UnloadWaitPolicy::WaitForIdle
        } else {
            UnloadWaitPolicy::AbortIfBusy
        };
        perform_unload(&state, model_id.clone(), policy)
            .await
            .expect("unload succeeds");
        assert!(
            !service.is_ready(),
            "unload must join the worker, not just hide the model"
        );
        assert!(
            context.upgrade().is_none(),
            "unload must release the stateless weight/prefix context"
        );
        assert!(state.snapshot_for_model(Some(&model_id)).is_none());
        assert!(sibling.is_ready(), "sibling worker must remain available");
        assert_eq!(state.model_ids().len(), 1);
    }
    sibling.shutdown().await.expect("fixture shutdown");
}

#[tokio::test]
async fn unload_finishes_retirement_after_caller_disconnects() {
    use crate::model_load::{UnloadWaitPolicy, perform_unload};
    let state = llama_cpp_state();
    let live = crate::app_state::build_live_state(
        "temporary".to_string(),
        state.snapshot().session_config.as_ref().clone(),
        &crate::args::ServerEnvConfig::default(),
    )
    .expect("fixture model builds");
    let service = live.generation_service.clone();
    state.publish_live(live, false);
    // Publication creates a generation-specific admission controller. Hold
    // the published controller so this is a real in-flight registry request.
    let admission = state
        .snapshot_for_model(Some("temporary"))
        .expect("published model")
        .admission;
    let permit = admission.try_admit().expect("hold one in-flight request");
    let task_state = state.clone();
    let caller = tokio::spawn(async move {
        perform_unload(
            &task_state,
            "temporary".to_string(),
            UnloadWaitPolicy::WaitForIdle,
        )
        .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !matches!(
            admission.try_admit(),
            Err(crate::admission::AdmissionError::Draining)
        ) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("unload starts draining");
    caller.abort();
    let _ = caller.await;
    drop(permit);
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while state.loading.load(std::sync::atomic::Ordering::Acquire) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("detached retirement completes");
    assert!(!service.is_ready());
    assert!(state.snapshot_for_model(Some("temporary")).is_none());
    let sibling = state.snapshot().generation_service;
    assert!(sibling.is_ready());
    sibling.shutdown().await.expect("fixture shutdown");
}
