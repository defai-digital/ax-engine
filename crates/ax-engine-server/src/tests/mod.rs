use crate::app_state::{AppState, build_app_state};
use crate::args::{self, ServerArgs};
use crate::openai::requests::DEFAULT_OPENAI_MAX_TOKENS;
use crate::routes::build_router;
use ax_engine_sdk::{EngineSession, GenerateRequest, GenerateSampling, GenerateStreamEvent};
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;

mod embeddings;
mod openai_chat;
mod openai_responses;

const TEST_MODEL_ID: &str = "qwen3";

fn sample_http_request_base(input_name: &str, input: Value, max_output_tokens: u32) -> Value {
    let mut request = serde_json::Map::new();
    request.insert("model".to_string(), json!(TEST_MODEL_ID));
    request.insert(input_name.to_string(), input);
    request.insert("max_output_tokens".to_string(), json!(max_output_tokens));
    Value::Object(request)
}

fn sample_http_request(input_tokens: &[u32], max_output_tokens: u32) -> Value {
    sample_http_request_base("input_tokens", json!(input_tokens), max_output_tokens)
}

fn sample_text_http_request(input_text: &str, max_output_tokens: u32) -> Value {
    sample_http_request_base("input_text", json!(input_text), max_output_tokens)
}

fn sample_sdk_request(input_tokens: &[u32], max_output_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        model_id: TEST_MODEL_ID.to_string(),
        input_tokens: input_tokens.to_vec(),
        input_text: None,
        max_output_tokens,
        sampling: GenerateSampling::default(),
        stop_sequences: Vec::new(),
        metadata: None,
    }
}

fn sample_openai_request_base(
    max_tokens: Option<u32>,
    stream: bool,
    with_fields: impl FnOnce(&mut serde_json::Map<String, Value>),
) -> serde_json::Map<String, Value> {
    sample_openai_request_base_fields(|request| {
        if let Some(max_tokens) = max_tokens {
            request.insert("max_tokens".to_string(), json!(max_tokens));
        }
        request.insert("stream".to_string(), json!(stream));
        with_fields(request);
    })
}

fn sample_openai_request_base_fields(
    with_fields: impl FnOnce(&mut serde_json::Map<String, Value>),
) -> serde_json::Map<String, Value> {
    let mut request = serde_json::Map::new();
    request.insert("model".to_string(), json!(TEST_MODEL_ID));
    with_fields(&mut request);
    request
}

fn sample_openai_completion_request(prompt: &str, max_tokens: Option<u32>, stream: bool) -> Value {
    let request = sample_openai_request_base(max_tokens, stream, |request| {
        request.insert("prompt".to_string(), json!(prompt));
    });
    Value::Object(request)
}

fn sample_openai_chat_request(message: &str, max_tokens: Option<u32>, stream: bool) -> Value {
    sample_openai_chat_request_with_role(message, "user", max_tokens, stream)
}

fn sample_openai_chat_request_with_role(
    message: &str,
    role: &str,
    max_tokens: Option<u32>,
    stream: bool,
) -> Value {
    let request = sample_openai_request_base(max_tokens, stream, |request| {
        request.insert(
            "messages".to_string(),
            json!([{
                "role": role,
                "content": message,
            }]),
        );
    });
    Value::Object(request)
}

fn sample_openai_embedding_request(input: &[u32], pooling: Option<&str>) -> Value {
    let request = sample_openai_request_base_fields(|request| {
        request.insert("input".to_string(), json!(input));
        if let Some(pooling) = pooling {
            request.insert("pooling".to_string(), json!(pooling));
        }
    });
    Value::Object(request)
}

fn sdk_session_for_state(state: &AppState) -> EngineSession {
    EngineSession::new(state.session_config.as_ref().clone()).expect("sdk session should build")
}

fn sdk_stream_payload(event: GenerateStreamEvent) -> (String, Value) {
    match event {
        GenerateStreamEvent::Request(payload) => (
            "request".to_string(),
            serde_json::to_value(payload).expect("request payload should serialize"),
        ),
        GenerateStreamEvent::Step(payload) => (
            "step".to_string(),
            serde_json::to_value(payload).expect("step payload should serialize"),
        ),
        GenerateStreamEvent::Response(payload) => (
            "response".to_string(),
            serde_json::to_value(payload).expect("response payload should serialize"),
        ),
    }
}

fn parse_sse_events(body: &str) -> Vec<(String, Value)> {
    parse_sse_frames(body)
        .into_iter()
        .filter_map(|(name, data)| {
            name.map(|event_name| {
                let payload = serde_json::from_str(&data).expect("sse payload should be json");
                (event_name, payload)
            })
        })
        .collect()
}

fn parse_openai_sse_payloads(body: &str) -> Vec<Value> {
    let mut events = Vec::new();
    for (_, data) in parse_sse_frames(body) {
        if data == "[DONE]" {
            break;
        }
        events.push(serde_json::from_str(&data).expect("openai sse payload should parse"));
    }
    events
}

fn openai_first_choice(payload: &Value) -> &Value {
    payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .expect("openai payload should include a first choice")
}

fn parse_sse_frames(body: &str) -> Vec<(Option<String>, String)> {
    let mut frames = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_data = String::new();

    for line in body.lines() {
        if line.is_empty() {
            if !current_data.is_empty() {
                frames.push((current_name.take(), current_data));
                current_data = String::new();
            } else {
                current_name = None;
            }
            continue;
        }
        if let Some(value) = line.strip_prefix("event: ") {
            current_name = Some(value.to_string());
            continue;
        }
        if let Some(value) = line.strip_prefix("data: ") {
            if !current_data.is_empty() {
                current_data.push('\n');
            }
            current_data.push_str(value);
        }
    }

    if !current_data.is_empty() {
        frames.push((current_name, current_data));
    }

    frames
}

fn assert_invalid_request_response(json: &Value, message_fragment: &str) {
    assert_eq!(
        json.get("error")
            .and_then(|error| error.get("code"))
            .and_then(Value::as_str),
        Some("invalid_request")
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

fn json_request_body<T: serde::Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).expect("request body should serialize")
}

fn normalize_measurement_fields(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("cpu_time_us");
            map.remove("runner_time_us");
            for value in map.values_mut() {
                normalize_measurement_fields(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                normalize_measurement_fields(value);
            }
        }
        _ => {}
    }
}

fn base_server_args() -> ServerArgs {
    ServerArgs {
        host: "127.0.0.1".to_string(),
        port: 8080,
        model_id: TEST_MODEL_ID.to_string(),
        preset: None,
        list_presets: false,
        deterministic: true,
        max_batch_tokens: 2048,
        cache_group_id: 0,
        block_size_tokens: 16,
        total_blocks: 1024,
        mlx: false,
        support_tier: args::PreviewSupportTier::LlamaCpp,
        llama_cli_path: "llama-cli".to_string(),
        llama_model_path: None,
        llama_server_url: None,
        mlx_lm_server_url: None,
        delegated_http_connect_timeout_secs:
            ax_engine_sdk::DelegatedHttpTimeouts::default_connect_secs(),
        delegated_http_read_timeout_secs: ax_engine_sdk::DelegatedHttpTimeouts::default_io_secs(),
        delegated_http_write_timeout_secs: ax_engine_sdk::DelegatedHttpTimeouts::default_io_secs(),
        mlx_model_artifacts_dir: None,
        resolve_model_artifacts: args::ModelArtifactResolution::ExplicitOnly,
        hf_cache_root: None,
        disable_ngram_acceleration: false,
        prefill_chunk: None,
        experimental_mlx_kv_compression: args::PreviewMlxKvCompression::Disabled,
        experimental_mlx_kv_compression_hot_window_tokens:
            ax_engine_sdk::KvCompressionConfig::DEFAULT_HOT_WINDOW_TOKENS,
        experimental_mlx_kv_compression_min_context_tokens:
            ax_engine_sdk::KvCompressionConfig::DEFAULT_MIN_CONTEXT_TOKENS,
        grpc_bind_address: None,
    }
}

fn fake_llama_cpp_script() -> std::path::PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("ax-engine-server-llama-cpp-{unique}.py"));
    let script = r#"#!/usr/bin/env python3
from __future__ import annotations

import sys

args = sys.argv[1:]
prompt = args[args.index("--prompt") + 1]
sys.stdout.write(f"server::{prompt}")
"#;

    fs::write(&path, script).expect("fake script should be written");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(&path)
            .expect("script metadata should exist")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).expect("script should be executable");
    }
    path
}

fn test_app_state<F>(customize: F) -> AppState
where
    F: FnOnce(&mut ServerArgs),
{
    let mut args = base_server_args();
    customize(&mut args);

    let session_config = args.session_config().expect("session config should build");
    let session = EngineSession::new(session_config.clone()).expect("session should build");
    build_app_state(args.model_id.clone(), session).expect("app state should build")
}

fn llama_cpp_state() -> AppState {
    let script_path = fake_llama_cpp_script();
    let model_path = std::env::temp_dir().join("ax-engine-server-llama-cpp-model.gguf");
    fs::write(&model_path, "fake gguf").expect("fake model should be written");
    test_app_state(|args| {
        args.llama_cli_path = script_path.display().to_string();
        args.llama_model_path = Some(model_path);
    })
}

fn llama_cpp_server_state(server_url: String) -> AppState {
    test_app_state(|args| {
        args.llama_server_url = Some(server_url);
    })
}

fn mlx_lm_delegated_state(server_url: String) -> AppState {
    test_app_state(|args| {
        args.support_tier = args::PreviewSupportTier::MlxLmDelegated;
        args.mlx_lm_server_url = Some(server_url);
    })
}

fn spawn_llama_cpp_completion_server(
    response_body: String,
    assert_request: impl FnMut(Value) + Send + 'static,
) -> (String, thread::JoinHandle<()>) {
    spawn_llama_cpp_completion_mock_server(
        1,
        assert_request,
        "application/json",
        response_body,
        None,
    )
}

fn spawn_llama_cpp_completion_stream_server(
    expected_requests: usize,
    chunks: Vec<Value>,
    assert_request: impl FnMut(Value) + Send + 'static,
) -> (String, thread::JoinHandle<()>) {
    let response_body = build_sse_event_stream_body(&chunks);
    spawn_llama_cpp_completion_mock_server(
        expected_requests,
        assert_request,
        "text/event-stream",
        response_body,
        Some("Cache-Control: no-cache"),
    )
}

fn spawn_llama_cpp_completion_mock_server(
    expected_requests: usize,
    mut assert_request: impl FnMut(Value) + Send + 'static,
    content_type: &'static str,
    response_body: String,
    extra_header: Option<&'static str>,
) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
    let address = listener.local_addr().expect("listener should have address");
    let handle = thread::spawn(move || {
        for _ in 0..expected_requests {
            let (mut stream, _) = listener.accept().expect("request should arrive");
            let payload = read_http_request_payload(&mut stream);
            assert_request(payload);
            write_http_response(&mut stream, content_type, &response_body, extra_header);
        }
    });

    (format!("http://{address}"), handle)
}

fn build_sse_event_stream_body(chunks: &[Value]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str("data: ");
        body.push_str(&serde_json::to_string(chunk).expect("chunk payload should serialize"));
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    let mut header_end = None;
    let mut content_length = None;

    loop {
        let bytes_read = stream.read(&mut buffer).expect("request should read");
        assert!(
            bytes_read > 0,
            "client closed connection before request completed"
        );
        request.extend_from_slice(&buffer[..bytes_read]);

        if header_end.is_none() {
            header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4);
            if let Some(end) = header_end {
                let headers =
                    String::from_utf8(request[..end].to_vec()).expect("headers should be utf8");
                content_length = Some(
                    headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            if name.eq_ignore_ascii_case("content-length") {
                                Some(
                                    value
                                        .trim()
                                        .parse::<usize>()
                                        .expect("content-length should parse"),
                                )
                            } else {
                                None
                            }
                        })
                        .expect("content-length header should exist"),
                );
            }
        }

        if let (Some(end), Some(length)) = (header_end, content_length) {
            if request.len() >= end + length {
                request.truncate(end + length);
                return request;
            }
        }
    }
}

fn read_http_request_payload(stream: &mut std::net::TcpStream) -> Value {
    let request = read_http_request(stream);
    let header_end = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
        .expect("request should include header terminator");
    let body = String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
    serde_json::from_str(&body).expect("request body should be json")
}

fn write_http_response(
    stream: &mut std::net::TcpStream,
    content_type: &str,
    body: &str,
    extra_header: Option<&str>,
) {
    let mut headers = format!(
        "Content-Type: {}\r\nContent-Length: {}\r\nConnection: close",
        content_type,
        body.len()
    );
    if let Some(extra_header) = extra_header {
        headers.push_str(&format!("\r\n{extra_header}"));
    }

    let response = format!("HTTP/1.1 200 OK\r\n{headers}\r\n\r\n{body}");
    stream
        .write_all(response.as_bytes())
        .expect("response should write");
}

async fn json_response(app: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(request)
        .await
        .expect("request should succeed");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should read");
    let json = serde_json::from_slice(&bytes).expect("response should be json");
    (status, json)
}

async fn text_response(app: &Router, request: Request<Body>) -> (StatusCode, String, String) {
    let response = app
        .clone()
        .oneshot(request)
        .await
        .expect("request should succeed");
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should read");
    let body = String::from_utf8(bytes.to_vec()).expect("body should be utf8");
    (status, content_type, body)
}

#[tokio::test]
async fn llama_cpp_generate_endpoint_runs_text_request_through_sdk() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "content": "server::hello from server",
            "tokens": [4, 5],
            "stop": true,
            "stop_type": "limit"
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello from server".to_string()))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
        },
    );
    let state = llama_cpp_server_state(llama_server_url);
    let app = build_router(state.clone());
    let request_body = sample_text_http_request("hello from server", 2);
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&request_body)))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.get("prompt_text").and_then(|value| value.as_str()),
        Some("hello from server")
    );
    assert_eq!(
        json.get("output_text").and_then(|value| value.as_str()),
        Some("server::hello from server")
    );
    assert_eq!(
        json.get("runtime")
            .and_then(|runtime| runtime.get("selected_backend"))
            .and_then(|value| value.as_str()),
        Some("llama_cpp")
    );
}

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
                assert_eq!(payload.get("repetition_penalty"), Some(&json!(1.0)));
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
async fn mlx_lm_delegated_generate_uses_text_and_usage_count_contract() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_server(
            r#"{"choices":[{"text":"mlx-lm::hello","finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#.to_string(),
            |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&Value::String("hello mlx-lm".to_string()))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
            },
        );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_text_http_request(
                "hello mlx-lm",
                2,
            ))))
            .unwrap(),
    )
    .await;
    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json.get("output_text").and_then(Value::as_str),
        Some("mlx-lm::hello")
    );
    assert!(
        json.get("output_tokens")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
    );
    assert_eq!(json.get("output_token_count"), Some(&json!(3)));
    assert_eq!(
        json.get("finish_reason").and_then(Value::as_str),
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

#[tokio::test]
async fn openai_embeddings_endpoint_rejects_empty_input() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_embedding_request(&[], None),
            )))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "input must not be empty");
}

#[tokio::test]
async fn openai_embeddings_endpoint_accepts_batch_shape() {
    // Verify the request body for `input: [[1,2,3],[4,5,6]]` round-trips
    // through serde untagged enum into the Batch variant. We only check
    // deserialization + early validation (model-not-found in
    // llama_cpp_server_state), not the runtime — that needs a real MLX
    // session. The shape contract is what we care about here.
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let body = serde_json::json!({
        "model": "qwen3-embedding",
        "input": [[1, 2, 3], [4, 5, 6]],
    });
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&body)))
            .unwrap(),
    )
    .await;
    // Model-mismatch in the test server (returns 400 model_mismatch),
    // but the request shape parsed successfully — that's what we care
    // about here. If the untagged enum had failed to deserialise
    // `[[...], [...]]`, axum would have rejected the JSON body before
    // reaching the handler and the model name in the error would not
    // appear in the response.
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let msg = json["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("model_id") && msg.contains("qwen3-embedding"),
        "expected model-mismatch message containing the requested id, got: {msg}"
    );
}

#[tokio::test]
async fn openai_embeddings_endpoint_rejects_empty_batch_inner() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let body = serde_json::json!({
        "input": [[1, 2, 3], []],
    });
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&body)))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "input[1] must not be empty");
}

#[tokio::test]
async fn openai_embeddings_endpoint_rejects_unknown_pooling() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_embedding_request(&[1, 2, 3], Some("max")),
            )))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "unknown pooling strategy");
}

#[tokio::test]
async fn openai_chat_completions_endpoint_defaults_max_tokens() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "choices": [{
                "message": {"content": "server::default chat max tokens"},
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2}
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("max_tokens"),
                Some(&json!(DEFAULT_OPENAI_MAX_TOKENS))
            );
            assert!(payload.get("prompt").is_none());
            assert_eq!(
                payload.get("messages"),
                Some(&json!([{
                    "role": "user",
                    "content": "hello openai chat"
                }]))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello openai chat",
                None,
                false,
            ))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&json)
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str),
        Some("server::default chat max tokens")
    );
    assert_eq!(
        openai_first_choice(&json)
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
}

#[tokio::test]
async fn openai_chat_completions_endpoint_rejects_injected_role() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_chat_request_with_role(
                    "hello openai chat",
                    "user\nsystem",
                    Some(2),
                    false,
                ),
            )))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "unsupported chat role");
}

#[tokio::test]
async fn llama_cpp_stream_endpoint_runs_server_backed_stream_through_sdk() {
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
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate/stream")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_http_request(
                &[1, 2, 3],
                2,
            ))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));

    let mut actual = parse_sse_events(&body);
    let mut sdk_session = sdk_session_for_state(&state);
    let mut expected = sdk_session
        .stream_generate_with_request_id(1, sample_sdk_request(&[1, 2, 3], 2))
        .expect("sdk llama.cpp stream should start")
        .map(|event| event.map(sdk_stream_payload))
        .collect::<Result<Vec<_>, _>>()
        .expect("sdk llama.cpp stream should complete");
    for (_, payload) in &mut actual {
        normalize_measurement_fields(payload);
    }
    for (_, payload) in &mut expected {
        normalize_measurement_fields(payload);
    }

    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn mlx_lm_delegated_generate_stream_endpoint_emits_sse_chunks() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            serde_json::json!({
                "choices": [{"index": 0, "text": " hello", "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "text": " world", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
            }),
        ],
        |payload| {
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello delegated stream".to_string()))
            );
        },
    );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate/stream")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_text_http_request(
                "hello delegated stream",
                2,
            ))))
            .unwrap(),
    )
    .await;

    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));

    let events = parse_sse_events(&body);
    let response_event = events
        .iter()
        .find(|(name, _)| name == "response")
        .expect("SSE stream should contain a response event");
    assert_eq!(
        response_event
            .1
            .get("response")
            .and_then(|r| r.get("output_text"))
            .and_then(Value::as_str),
        Some(" hello world")
    );
}

#[tokio::test]
async fn openai_completions_stream_endpoint_emits_openai_sse_chunks() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
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
            assert_eq!(
                payload.get("prompt"),
                Some(&Value::String("hello stream".to_string()))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_completion_request("hello stream", Some(2), true),
            )))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));
    let payloads = parse_openai_sse_payloads(&body);
    assert_eq!(payloads.len(), 3);
    assert_eq!(payloads[0].get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&payloads[0])
            .get("text")
            .and_then(Value::as_str),
        Some("hello")
    );
    assert_eq!(
        openai_first_choice(&payloads[1])
            .get("text")
            .and_then(Value::as_str),
        Some(" world")
    );
    assert_eq!(
        openai_first_choice(&payloads[2])
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn openai_chat_completions_stream_endpoint_emits_openai_sse_chunks() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": "chat"}, "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": " stream"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            }),
        ],
        |payload| {
            assert!(payload.get("prompt").is_none());
            assert_eq!(
                payload.get("messages"),
                Some(&json!([{
                    "role": "user",
                    "content": "hello chat stream"
                }]))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello chat stream",
                Some(2),
                true,
            ))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));
    let payloads = parse_openai_sse_payloads(&body);
    assert_eq!(payloads.len(), 3);
    assert_eq!(payloads[0].get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&payloads[0])
            .get("delta")
            .and_then(|delta| delta.get("role"))
            .and_then(Value::as_str),
        Some("assistant")
    );
    assert_eq!(
        openai_first_choice(&payloads[0])
            .get("delta")
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str),
        Some("chat")
    );
    assert_eq!(
        openai_first_choice(&payloads[1])
            .get("delta")
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str),
        Some(" stream")
    );
    assert_eq!(
        openai_first_choice(&payloads[2])
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
}

#[tokio::test]
async fn openai_chat_stream_endpoint_emits_sse_for_mlx_lm_delegated() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_stream_server(
        1,
        vec![
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": " hello"}, "finish_reason": null}],
                "usage": null
            }),
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
            }),
        ],
        |payload| {
            assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            assert!(payload.get("prompt").is_none());
            assert_eq!(
                payload.get("messages"),
                Some(&json!([{
                    "role": "user",
                    "content": "hello mlx-lm chat stream"
                }]))
            );
            assert_eq!(
                payload.get("chat_template_kwargs"),
                Some(&json!({"enable_thinking": false}))
            );
        },
    );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello mlx-lm chat stream",
                Some(2),
                true,
            ))))
            .unwrap(),
    )
    .await;

    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.starts_with("text/event-stream"));

    let payloads = parse_openai_sse_payloads(&body);
    let text_chunks: Vec<&str> = payloads
        .iter()
        .filter_map(|p| {
            openai_first_choice(p)
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        })
        .collect();
    assert_eq!(text_chunks, vec![" hello", " world"]);
    assert_eq!(payloads[0].get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(payloads.last().expect("final chunk should exist"))
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("stop")
    );
}

#[tokio::test]
async fn openai_chat_endpoint_forwards_messages_for_mlx_lm_delegated() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_server(
            r#"{"choices":[{"message":{"content":"chat reply"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}"#.to_string(),
            |payload| {
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
                assert!(payload.get("prompt").is_none());
                assert_eq!(
                    payload.get("messages"),
                    Some(&json!([{
                        "role": "user",
                        "content": "hello mlx-lm chat"
                    }]))
                );
                assert_eq!(
                    payload.get("chat_template_kwargs"),
                    Some(&json!({"enable_thinking": false}))
                );
            },
        );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello mlx-lm chat",
                Some(2),
                false,
            ))))
            .unwrap(),
    )
    .await;

    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        json.get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str),
        Some("chat reply")
    );
}

#[tokio::test]
async fn llama_cpp_cli_stream_endpoint_fails_closed() {
    let app = build_router(llama_cpp_state());
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/generate/stream")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_text_http_request(
                "hello from stream",
                2,
            ))))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        json.get("error")
            .and_then(|value| value.get("code"))
            .and_then(|value| value.as_str()),
        Some("invalid_request")
    );
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
