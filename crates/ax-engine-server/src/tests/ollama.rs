use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use super::fixtures::{
    json_request_body, json_response, llama_cpp_server_state, llama_cpp_state,
    spawn_llama_cpp_completion_server, text_response,
};

fn assert_unsupported_parameter_response(json: &Value, message_fragment: &str) {
    assert_eq!(
        json.get("error")
            .and_then(|error| error.get("code"))
            .and_then(Value::as_str),
        Some("unsupported_parameter")
    );
    assert!(
        json.get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains(message_fragment),
        "error message should include expected fragment"
    );
}

#[tokio::test]
async fn ollama_tags_lists_loaded_ax_engine_model() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/api/tags")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let model = &json["models"][0];
    assert_eq!(model["name"], json!("qwen3"));
    assert_eq!(model["model"], json!("qwen3"));
    assert_eq!(model["details"]["format"], json!("ax-engine"));
    assert_eq!(model["details"]["family"], json!("qwen"));
    assert!(
        model["modified_at"]
            .as_str()
            .unwrap_or_default()
            .ends_with('Z')
    );
}

#[tokio::test]
async fn ollama_show_returns_loaded_model_metadata() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/show")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({"model": "qwen3"}))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["details"]["format"], json!("ax-engine"));
    assert_eq!(json["details"]["family"], json!("qwen"));
    assert_eq!(json["model_info"]["general.name"], json!("qwen3"));
    assert_eq!(
        json["model_info"]["ax_engine.context_length"],
        json!(16 * 1024)
    );
    assert!(
        json["modified_at"]
            .as_str()
            .unwrap_or_default()
            .ends_with('Z')
    );
    assert!(
        json["capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("completion"))
    );
    assert!(
        json["modelfile"]
            .as_str()
            .unwrap_or_default()
            .contains("FROM qwen3")
    );
}

#[tokio::test]
async fn ollama_show_accepts_verbose_false_probe() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/show")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "verbose": false
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["details"]["family"], json!("qwen"));
}

#[tokio::test]
async fn ollama_show_rejects_verbose_true_until_supported() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/show")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "verbose": true
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_unsupported_parameter_response(&json, "`verbose`");
}

#[tokio::test]
async fn ollama_show_rejects_unknown_fields() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/show")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "license": "unsupported"
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_unsupported_parameter_response(&json, "`license`");
}

#[tokio::test]
async fn ollama_ps_lists_current_model_as_loaded() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/api/ps")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let model = &json["models"][0];
    assert_eq!(model["name"], json!("qwen3"));
    assert_eq!(model["details"]["format"], json!("ax-engine"));
    assert_eq!(model["size_vram"], json!(0));
    assert_eq!(model["context_length"], json!(16 * 1024));
    assert!(
        model["expires_at"]
            .as_str()
            .unwrap_or_default()
            .ends_with('Z')
    );
}

#[tokio::test]
async fn ollama_version_returns_ax_engine_version_string() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/api/version")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["version"], json!(env!("CARGO_PKG_VERSION")));
}

#[tokio::test]
async fn ollama_chat_non_stream_maps_to_openai_chat_and_returns_ollama_shape() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        json!({
            "choices": [{
                "message": {"content": "chat reply"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        })
        .to_string(),
        |payload| {
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            assert!(payload.get("prompt").is_none());
            assert_eq!(
                payload.get("messages"),
                Some(&json!([{"role": "user", "content": "hello ollama"}]))
            );
            assert_eq!(payload.get("max_tokens"), Some(&json!(7)));
            assert_eq!(payload.get("temperature"), Some(&json!(0.0)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "keep_alive": "5m",
                "stream": false,
                "messages": [{"role": "user", "content": "hello ollama"}],
                "options": {"num_predict": 7, "temperature": 0.0}
            }))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"], json!("qwen3"));
    assert_eq!(json["message"]["role"], json!("assistant"));
    assert_eq!(json["message"]["content"], json!("chat reply"));
    assert_eq!(json["done"], json!(true));
    assert_eq!(json["done_reason"], json!("stop"));
    assert_eq!(json["prompt_eval_count"], json!(4));
    assert_eq!(json["eval_count"], json!(2));
}

#[tokio::test]
async fn ollama_chat_defaults_to_ndjson_stream_shape() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        json!({
            "choices": [{
                "message": {"content": "stream reply"},
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
        })
        .to_string(),
        |payload| {
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            assert_eq!(
                payload.get("messages"),
                Some(&json!([{"role": "user", "content": "hello stream"}]))
            );
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hello stream"}]
            }))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("application/x-ndjson"));
    let lines = body
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("line should be json"))
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["message"]["content"], json!("stream reply"));
    assert_eq!(lines[0]["done"], json!(false));
    assert_eq!(lines[1]["done"], json!(true));
    assert_eq!(lines[1]["done_reason"], json!("length"));
    assert_eq!(lines[1]["eval_count"], json!(5));
}

#[tokio::test]
async fn ollama_generate_non_stream_returns_ollama_shape() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        json!({
            "content": "generated text",
            "tokens": [10, 11],
            "stop": true,
            "stop_type": "eos"
        })
        .to_string(),
        |payload| {
            assert_eq!(payload.get("prompt"), Some(&json!("hello generate")));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            assert_eq!(payload.get("n_predict"), Some(&json!(9)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "stream": false,
                "prompt": "hello generate",
                "keep_alive": 0,
                "raw": true,
                "options": {"num_predict": 9}
            }))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"], json!("qwen3"));
    assert_eq!(json["response"], json!("generated text"));
    assert_eq!(json["done"], json!(true));
    assert_eq!(json["done_reason"], json!("stop"));
    assert_eq!(json["eval_count"], json!(2));
}

#[tokio::test]
async fn ollama_generate_raw_true_does_not_prepend_system_prompt() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        json!({
            "content": "raw text",
            "tokens": [10],
            "stop": true,
            "stop_type": "eos"
        })
        .to_string(),
        |payload| {
            assert_eq!(payload.get("prompt"), Some(&json!("raw prompt")));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "stream": false,
                "system": "system prompt",
                "prompt": "raw prompt",
                "raw": true
            }))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["response"], json!("raw text"));
}

#[tokio::test]
async fn ollama_generate_defaults_to_ndjson_stream_shape() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        json!({
            "content": "streamed generated text",
            "tokens": [12, 13, 14],
            "stop": true,
            "stop_type": "limit"
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("prompt"),
                Some(&json!("hello streamed generate"))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "prompt": "hello streamed generate"
            }))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("application/x-ndjson"));
    let lines = body
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("line should be json"))
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["response"], json!("streamed generated text"));
    assert_eq!(lines[0]["done"], json!(false));
    assert_eq!(lines[1]["done"], json!(true));
    assert_eq!(lines[1]["done_reason"], json!("length"));
    assert_eq!(lines[1]["eval_count"], json!(3));
}

#[tokio::test]
async fn ollama_generate_empty_prompt_returns_load_or_unload_noop() {
    let app = build_router(llama_cpp_state());
    let (load_status, load_json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({"model": "qwen3"}))))
            .unwrap(),
    )
    .await;

    assert_eq!(load_status, StatusCode::OK);
    assert_eq!(load_json["model"], json!("qwen3"));
    assert_eq!(load_json["response"], json!(""));
    assert_eq!(load_json["done"], json!(true));
    assert!(load_json.get("done_reason").is_none());

    let (unload_status, unload_json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "keep_alive": 0
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(unload_status, StatusCode::OK);
    assert_eq!(unload_json["response"], json!(""));
    assert_eq!(unload_json["done"], json!(true));
    assert_eq!(unload_json["done_reason"], json!("unload"));
}

#[tokio::test]
async fn ollama_generate_rejects_context_replay_until_supported() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "prompt": "hello",
                "context": [1, 2, 3]
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_unsupported_parameter_response(&json, "`context`");
}

#[tokio::test]
async fn ollama_generate_rejects_unknown_top_level_fields() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "prompt": "hello",
                "suffix": "unsupported suffix"
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_unsupported_parameter_response(&json, "`suffix`");
}

#[tokio::test]
async fn ollama_generate_rejects_unknown_options_fields() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "prompt": "hello",
                "options": {"num_ctx": 4096}
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_unsupported_parameter_response(&json, "`options.num_ctx`");
}

#[tokio::test]
async fn ollama_chat_rejects_unknown_top_level_fields() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "think": true,
                "messages": [{"role": "user", "content": "hello"}]
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_unsupported_parameter_response(&json, "`think`");
}

#[tokio::test]
async fn ollama_chat_rejects_unknown_message_fields() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "messages": [{
                    "role": "user",
                    "content": "hello",
                    "thinking": "unsupported"
                }]
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_unsupported_parameter_response(&json, "`messages[].thinking`");
}

#[tokio::test]
async fn ollama_chat_rejects_unknown_tool_call_fields() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "model": "qwen3",
                "messages": [{
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "lookup",
                            "arguments": {}
                        }
                    }]
                }]
            }))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_unsupported_parameter_response(&json, "`messages[].tool_calls[].id`");
}
