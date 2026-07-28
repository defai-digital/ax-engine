use crate::app_state::{AppState, LiveState, build_app_state};
use crate::args::{self, ServerArgs};
use ax_engine_sdk::{
    EngineSession, GenerateRequest, GenerateSampling, GenerateStreamEvent, SelectedBackend,
};
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;

pub(super) const TEST_MODEL_ID: &str = "qwen3";

pub(super) fn sample_http_request_base(
    input_name: &str,
    input: Value,
    max_output_tokens: u32,
) -> Value {
    let mut request = serde_json::Map::new();
    request.insert("model".to_string(), json!(TEST_MODEL_ID));
    request.insert(input_name.to_string(), input);
    request.insert("max_output_tokens".to_string(), json!(max_output_tokens));
    Value::Object(request)
}

pub(super) fn sample_http_request(input_tokens: &[u32], max_output_tokens: u32) -> Value {
    sample_http_request_base("input_tokens", json!(input_tokens), max_output_tokens)
}

pub(super) fn sample_text_http_request(input_text: &str, max_output_tokens: u32) -> Value {
    sample_http_request_base("input_text", json!(input_text), max_output_tokens)
}

pub(super) fn sample_sdk_request(input_tokens: &[u32], max_output_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        model_id: TEST_MODEL_ID.to_string(),
        input_tokens: input_tokens.to_vec(),
        input_text: None,
        multimodal_inputs: Default::default(),
        max_output_tokens,
        sampling: GenerateSampling::default(),
        stop_sequences: Vec::new(),
        metadata: None,
    }
}

pub(super) fn sample_gemma4_multimodal_inputs() -> Value {
    json!({
        "gemma4_unified": {
            "images": [{
                "span": {
                    "modality": "image",
                    "placeholder_index": 1,
                    "replacement_start": 1,
                    "soft_token_count": 1,
                    "replacement_token_count": 3
                },
                "pixel_values": [0.0, 1.0, 2.0],
                "pixel_position_ids": [[0, 0]]
            }],
            "audios": [],
            "videos": []
        }
    })
}

pub(super) fn sample_openai_request_base(
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

pub(super) fn sample_openai_request_base_fields(
    with_fields: impl FnOnce(&mut serde_json::Map<String, Value>),
) -> serde_json::Map<String, Value> {
    let mut request = serde_json::Map::new();
    request.insert("model".to_string(), json!(TEST_MODEL_ID));
    with_fields(&mut request);
    request
}

pub(super) fn sample_openai_completion_request(
    prompt: &str,
    max_tokens: Option<u32>,
    stream: bool,
) -> Value {
    let request = sample_openai_request_base(max_tokens, stream, |request| {
        request.insert("prompt".to_string(), json!(prompt));
    });
    Value::Object(request)
}

pub(super) fn sample_openai_chat_request(
    message: &str,
    max_tokens: Option<u32>,
    stream: bool,
) -> Value {
    sample_openai_chat_request_with_role(message, "user", max_tokens, stream)
}

pub(super) fn sample_openai_chat_request_with_role(
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

pub(super) fn sdk_session_for_state(state: &AppState) -> EngineSession {
    let live = state.snapshot();
    EngineSession::new(live.session_config.as_ref().clone()).expect("sdk session should build")
}

pub(super) fn sdk_stream_payload(event: GenerateStreamEvent) -> (String, Value) {
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

pub(super) fn parse_sse_events(body: &str) -> Vec<(String, Value)> {
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

pub(super) fn parse_openai_sse_payloads(body: &str) -> Vec<Value> {
    let mut events = Vec::new();
    for (_, data) in parse_sse_frames(body) {
        if data == "[DONE]" {
            break;
        }
        events.push(serde_json::from_str(&data).expect("openai sse payload should parse"));
    }
    events
}

pub(super) fn openai_first_choice(payload: &Value) -> &Value {
    payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .expect("openai payload should include a first choice")
}

pub(super) fn parse_sse_frames(body: &str) -> Vec<(Option<String>, String)> {
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

pub(super) fn assert_invalid_request_response(json: &Value, message_fragment: &str) {
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

pub(super) fn json_request_body<T: serde::Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).expect("request body should serialize")
}

pub(super) fn normalize_measurement_fields(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // Drop wall-clock and path-dependent performance counters so HTTP
            // SSE payloads can be compared to an independent SDK stream of the
            // same request without flaking on microsecond timing or token
            // accounting that differs when the same mock server is hit twice.
            for key in [
                "cpu_time_us",
                "runner_time_us",
                "generation_time_us",
                "time_to_first_token_us",
                "total_time_us",
                "generation_token_count",
            ] {
                map.remove(key);
            }
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

pub(super) fn base_server_args() -> ServerArgs {
    ServerArgs {
        host: "127.0.0.1".to_string(),
        port: 8080,
        model_id: TEST_MODEL_ID.to_string(),
        api_key: None,
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
        mlx_mtp_enable_ngram_stacking: false,
        mlx_mtp_disable_ngram_stacking: false,
        speculation_profile: None,
        prefill_chunk: None,
        multi_prefill_fair: false,
        max_prefill_tokens_per_request_per_step: 0,
        max_inflight_prefill_requests: 0,
        grpc_bind_address: None,
        max_concurrent_requests: None,
        max_concurrent_requests_per_model: None,
        max_request_body_bytes: None,
        request_timeout_secs: None,
        grpc_request_timeout_secs: None,
        rate_limit_rps: None,
        rate_limit_burst: None,
        stream_idle_timeout_secs: None,
        model_idle_timeout_secs: None,
        stream_max_duration_secs: None,
        advertise_lan: false,
        lan_cluster: None,
        lan_instance_name: None,
        lan_advertise_host: None,
        allow_open_lan: false,
    }
}

pub(super) fn fake_llama_cpp_script() -> std::path::PathBuf {
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

pub(super) fn test_app_state<F>(customize: F) -> AppState
where
    F: FnOnce(&mut ServerArgs),
{
    let mut args = base_server_args();
    customize(&mut args);

    let session_config = args.session_config().expect("session config should build");
    build_app_state(args.model_id.clone(), session_config).expect("app state should build")
}

pub(super) fn llama_cpp_state() -> AppState {
    let script_path = fake_llama_cpp_script();
    let model_path = std::env::temp_dir().join("ax-engine-server-llama-cpp-model.gguf");
    fs::write(&model_path, "fake gguf").expect("fake model should be written");
    test_app_state(|args| {
        args.llama_cli_path = script_path.display().to_string();
        args.llama_model_path = Some(model_path);
    })
}

pub(super) fn llama_cpp_server_state(server_url: String) -> AppState {
    test_app_state(|args| {
        args.llama_server_url = Some(server_url);
    })
}

pub(super) fn mlx_lm_delegated_state(server_url: String) -> AppState {
    test_app_state(|args| {
        args.support_tier = args::PreviewSupportTier::MlxLmDelegated;
        args.mlx_lm_server_url = Some(server_url);
    })
}

pub(super) fn native_mlx_openai_builder_state(model_id: &str, artifacts_dir: &Path) -> AppState {
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let mut live: LiveState = state.snapshot();
    live.model_id = Arc::new(model_id.to_string());
    live.session_config = Arc::new(
        live.session_config
            .as_ref()
            .clone()
            .with_mlx_model_artifacts_dir(artifacts_dir),
    );
    live.runtime_report.selected_backend = SelectedBackend::Mlx;
    state.swap_live(live);
    state
}

pub(super) fn minimal_tokenizer_artifact(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ax-engine-server-{label}-{unique}"));
    fs::create_dir_all(&dir).expect("tokenizer artifact dir should create");
    fs::write(dir.join("config.json"), r#"{"eos_token_id":2}"#).expect("config should write");
    fs::write(
        dir.join("tokenizer.json"),
        r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "<|im_start|>": 2,
      "user": 3,
      "assistant": 4,
      "openai": 5,
      "chat": 6,
      "completion": 7
    },
    "unk_token": "[UNK]"
  }
}"#,
    )
    .expect("tokenizer should write");
    dir
}

/// A native-MLX artifact dir for a Gemma 4 unified-style multimodal model: a
/// `config.json` + `preprocessor_config.json` describing the encoder-free
/// connector, and a `tokenizer.json` whose added tokens make the image/audio/
/// video placeholders round-trip to single ids. Uses a tiny synthetic vision
/// config (model_patch_size 8, max_soft_tokens 4) so a 16x16 image yields
/// exactly 4 soft tokens.
pub(super) fn gemma4_unified_artifact(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ax-engine-server-{label}-{unique}"));
    fs::create_dir_all(&dir).expect("artifact dir should create");
    fs::write(
        dir.join("config.json"),
        r#"{
  "eos_token_id": 2,
  "image_token_id": 100,
  "audio_token_id": 101,
  "video_token_id": 104,
  "boi_token_id": 102,
  "eoi_token_id": 103,
  "boa_token_id": 105,
  "eoa_token_index": 106,
  "vision_config": {
    "patch_size": 4,
    "model_patch_size": 8,
    "pooling_kernel_size": 2,
    "num_soft_tokens": 4
  }
}"#,
    )
    .expect("config should write");
    fs::write(
        dir.join("preprocessor_config.json"),
        r#"{
  "image_processor": {
    "patch_size": 4,
    "model_patch_size": 8,
    "pooling_kernel_size": 2,
    "max_soft_tokens": 4
  },
  "feature_extractor": {"sampling_rate": 16000},
  "audio_seq_length": 1500
}"#,
    )
    .expect("preprocessor config should write");
    fs::write(
        dir.join("tokenizer.json"),
        r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 100, "content": "<img>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 101, "content": "<aud>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 104, "content": "<vid>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "describe": 1,
      "<img>": 100,
      "<aud>": 101,
      "<vid>": 104
    },
    "unk_token": "[UNK]"
  }
}"#,
    )
    .expect("tokenizer should write");
    dir
}

/// Minimal native-MLX artifact for Qwen3-VL chat image path tests.
pub(super) fn qwen3_vl_artifact(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ax-engine-server-{label}-{unique}"));
    fs::create_dir_all(&dir).expect("artifact dir should create");
    fs::write(
        dir.join("config.json"),
        r#"{
  "model_type": "qwen3_vl",
  "eos_token_id": 2
}"#,
    )
    .expect("config should write");
    fs::write(
        dir.join("model-manifest.json"),
        r#"{
  "schema_version": "ax.native_model.v1",
  "model_family": "qwen3_vl",
  "tensor_format": "safetensors",
  "layer_count": 1,
  "hidden_size": 8,
  "intermediate_size": 16,
  "attention_head_count": 2,
  "attention_head_dim": 4,
  "kv_head_count": 2,
  "vocab_size": 32,
  "tie_word_embeddings": false,
  "tensors": [
    {"name": "visual.patch_embed.proj.weight", "role": "qwen3_vl_vision_patch_embed", "dtype": "f32", "shape": [8, 6], "file": "w.safetensors", "offset_bytes": 0, "length_bytes": 192},
    {"name": "visual.merger.weight", "role": "qwen3_vl_vision_merger", "dtype": "f32", "shape": [8, 8], "file": "w.safetensors", "offset_bytes": 192, "length_bytes": 256}
  ]
}"#,
    )
    .expect("manifest should write");
    fs::write(
        dir.join("tokenizer.json"),
        r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 10, "content": "<|image_pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "describe": 1,
      "user": 2,
      "assistant": 3,
      "<|im_start|>": 4,
      "<|im_end|>": 5,
      "<|image_pad|>": 10
    },
    "unk_token": "[UNK]"
  }
}"#,
    )
    .expect("tokenizer should write");
    dir
}

pub(super) fn spawn_llama_cpp_completion_server(
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

pub(super) fn spawn_llama_cpp_completion_stream_server(
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

pub(super) fn spawn_llama_cpp_completion_mock_server(
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

pub(super) fn build_sse_event_stream_body(chunks: &[Value]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str("data: ");
        body.push_str(&serde_json::to_string(chunk).expect("chunk payload should serialize"));
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

pub(super) fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
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

pub(super) fn read_http_request_payload(stream: &mut std::net::TcpStream) -> Value {
    let request = read_http_request(stream);
    let header_end = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
        .expect("request should include header terminator");
    let body = String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
    serde_json::from_str(&body).expect("request body should be json")
}

pub(super) fn write_http_response(
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

pub(super) async fn json_response(app: &Router, request: Request<Body>) -> (StatusCode, Value) {
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

pub(super) async fn text_response(
    app: &Router,
    request: Request<Body>,
) -> (StatusCode, String, String) {
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
