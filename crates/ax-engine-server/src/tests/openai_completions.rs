use crate::openai::requests::DEFAULT_OPENAI_MAX_TOKENS;
use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use super::fixtures::{
    assert_invalid_request_response, json_request_body, json_response, llama_cpp_server_state,
    mlx_lm_delegated_state, openai_first_choice, sample_openai_completion_request,
    sample_openai_request_base, spawn_llama_cpp_completion_server,
};

#[tokio::test]
async fn openai_completions_endpoint_translates_llama_cpp_response() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "content": "server::hello openai",
            "tokens": [4, 5],
            "stop": true,
            "stop_type": "limit",
            "tokens_evaluated": 4
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello openai".to_string()))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_completion_request("hello openai", Some(2), false),
            )))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.get("object").and_then(Value::as_str),
        Some("text_completion")
    );
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&json)
            .get("text")
            .and_then(Value::as_str),
        Some("server::hello openai")
    );
    assert_eq!(
        openai_first_choice(&json)
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
    assert_eq!(
        json.get("usage"),
        Some(&json!({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}))
    );
}

#[tokio::test]
async fn openai_completions_endpoint_translates_mlx_lm_delegated_response() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_server(
            r#"{"choices":[{"text":"mlx-lm::hello","finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#.to_string(),
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello mlx-lm".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
                assert_eq!(payload.get("top_k"), Some(&json!(0)));
                assert_eq!(payload.get("repetition_penalty"), Some(&json!(1.1)));
            },
        );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_completion_request("hello mlx-lm", Some(2), false),
            )))
            .unwrap(),
    )
    .await;
    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.get("object").and_then(Value::as_str),
        Some("text_completion")
    );
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&json)
            .get("text")
            .and_then(Value::as_str),
        Some("mlx-lm::hello")
    );
    assert_eq!(
        openai_first_choice(&json)
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("stop")
    );
}

#[tokio::test]
async fn openai_completions_endpoint_defaults_max_tokens() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "content": "server::default max tokens",
            "tokens": [4, 5],
            "stop": true,
            "stop_type": "limit"
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello openai".to_string()))
            );
            assert_eq!(
                payload.get("n_predict"),
                Some(&json!(DEFAULT_OPENAI_MAX_TOKENS))
            );
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_completion_request("hello openai", None, false),
            )))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        openai_first_choice(&json)
            .get("text")
            .and_then(Value::as_str),
        Some("server::default max tokens")
    );
}

#[tokio::test]
async fn openai_completions_endpoint_randomizes_unseeded_sampling_requests() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "content": "server::sampled",
            "tokens": [4],
            "stop": true
        })
        .to_string(),
        |payload| {
            assert_eq!(payload.get("temperature"), Some(&json!(0.8)));
            assert_ne!(payload.get("seed"), Some(&json!(0)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let mut request = sample_openai_request_base(Some(2), false, |request| {
        request.insert("prompt".to_string(), json!("sample me"));
        request.insert("temperature".to_string(), json!(0.8));
    });
    request.remove("model");

    let (status, _json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&Value::Object(request))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn openai_completions_endpoint_rejects_text_batch_prompt() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let mut request = sample_openai_request_base(Some(2), false, |request| {
        request.insert(
            "prompt".to_string(),
            json!(["first prompt", "second prompt"]),
        );
    });
    request.remove("model");

    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&Value::Object(request))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "batch completion prompts are not supported");
}
