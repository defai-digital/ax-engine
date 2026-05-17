use crate::app_state::{AppState, build_app_state};
use crate::args::{self, ServerArgs};
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
mod lifecycle;
mod openai_chat;
mod openai_completions;
mod openai_responses;
mod openai_streaming;

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
