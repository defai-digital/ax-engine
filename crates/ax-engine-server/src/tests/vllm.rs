use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};

use crate::args::VllmModelProfileArg;
use crate::routes::build_router;

use super::fixtures::{
    build_sse_event_stream_body, json_request_body, json_response, openai_first_choice,
    parse_openai_sse_payloads, spawn_vllm_chat_server, text_response, vllm_delegated_state,
};

const UPSTREAM_MODEL_ID: &str = "baidu/Unlimited-OCR";
const ONE_PIXEL_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

#[tokio::test]
async fn unlimited_ocr_vllm_route_matches_direct_provider_golden_payload() {
    let response_body = json!({
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "<ocr>ok</ocr>"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 17,
            "completion_tokens": 5,
            "total_tokens": 22
        }
    })
    .to_string();
    let (server_url, server_handle) = spawn_vllm_chat_server(
        UPSTREAM_MODEL_ID,
        "application/json",
        response_body,
        |request| {
            assert_eq!(
                request.headers.get("content-type").map(String::as_str),
                Some("application/json")
            );
            assert_eq!(
                request.headers.get("accept").map(String::as_str),
                Some("application/json")
            );
            assert_eq!(
                request.payload,
                Some(json!({
                    "model": UPSTREAM_MODEL_ID,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "<image>document parsing."},
                            {
                                "type": "image_url",
                                "image_url": {"url": ONE_PIXEL_PNG}
                            }
                        ]
                    }],
                    "max_tokens": 8192,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": false,
                    "skip_special_tokens": false,
                    "vllm_xargs": {
                        "ngram_size": 35,
                        "window_size": 128
                    }
                }))
            );
        },
    );
    let app = build_router(vllm_delegated_state(
        server_url,
        UPSTREAM_MODEL_ID,
        VllmModelProfileArg::UnlimitedOcr,
    ));
    let request = json!({
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>document parsing."},
                {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG}}
            ]
        }],
        "temperature": 0.0,
        "top_p": 1.0
    });
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&request)))
            .expect("request should build"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "unexpected response: {response}");
    server_handle
        .join()
        .expect("vLLM mock server should finish");

    assert_eq!(response.get("model"), Some(&Value::String("qwen3".into())));
    assert_eq!(
        openai_first_choice(&response)
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str),
        Some("<ocr>ok</ocr>")
    );
    assert_eq!(
        response
            .get("usage")
            .and_then(|usage| usage.get("prompt_tokens")),
        Some(&json!(17))
    );
}

#[tokio::test]
async fn unlimited_ocr_multi_image_uses_the_certified_1024_window() {
    let (server_url, server_handle) = spawn_vllm_chat_server(
        UPSTREAM_MODEL_ID,
        "application/json",
        json!({
            "choices": [{
                "message": {"role": "assistant", "content": "two pages"},
                "finish_reason": "stop"
            }]
        })
        .to_string(),
        |request| {
            let payload = request.payload.expect("generation payload should exist");
            assert_eq!(
                payload.get("vllm_xargs"),
                Some(&json!({"ngram_size": 35, "window_size": 1024}))
            );
            let content = payload["messages"][0]["content"]
                .as_array()
                .expect("ordered content parts should be an array");
            assert_eq!(content.len(), 3);
            assert_eq!(content[0]["type"], "text");
            assert_eq!(content[1]["type"], "image_url");
            assert_eq!(content[2]["type"], "image_url");
        },
    );
    let app = build_router(vllm_delegated_state(
        server_url,
        UPSTREAM_MODEL_ID,
        VllmModelProfileArg::UnlimitedOcr,
    ));
    let request = json!({
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image><image>document parsing."},
                {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG}},
                {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG}}
            ]
        }]
    });
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&request)))
            .expect("request should build"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "unexpected response: {response}");
    server_handle
        .join()
        .expect("vLLM mock server should finish");
}

#[tokio::test]
async fn vllm_stream_handles_role_only_usage_only_and_done_frames() {
    let upstream_body = build_sse_event_stream_body(&[
        json!({
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null
            }],
            "usage": null
        }),
        json!({
            "choices": [{
                "index": 0,
                "delta": {"content": "page"},
                "finish_reason": null
            }],
            "usage": null
        }),
        json!({
            "choices": [{
                "index": 0,
                "delta": {"content": " text"},
                "finish_reason": "length"
            }],
            "usage": null
        }),
        json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9
            }
        }),
    ]);
    let (server_url, server_handle) = spawn_vllm_chat_server(
        UPSTREAM_MODEL_ID,
        "text/event-stream",
        upstream_body,
        |request| {
            let payload = request.payload.expect("generation payload should exist");
            assert_eq!(payload.get("model"), Some(&json!(UPSTREAM_MODEL_ID)));
            assert_eq!(payload.get("stream"), Some(&json!(true)));
            assert_eq!(
                payload.get("stream_options"),
                Some(&json!({"include_usage": true}))
            );
        },
    );
    let app = build_router(vllm_delegated_state(
        server_url,
        UPSTREAM_MODEL_ID,
        VllmModelProfileArg::OpenAiCompatible,
    ));
    let request = json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "read this"}],
        "max_tokens": 2,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&request)))
            .expect("request should build"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "unexpected stream body: {body}");
    server_handle
        .join()
        .expect("vLLM mock server should finish");

    assert!(content_type.starts_with("text/event-stream"));
    assert!(body.contains("data: [DONE]"));
    let payloads = parse_openai_sse_payloads(&body);
    let text = payloads
        .iter()
        .filter_map(|payload| {
            payload
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("delta"))
                .and_then(|delta| delta.get("content"))
                .and_then(Value::as_str)
        })
        .collect::<String>();
    assert_eq!(text, "page text");
    assert_eq!(
        openai_first_choice(&payloads[2])
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
    assert_eq!(
        payloads
            .last()
            .and_then(|payload| payload.get("usage"))
            .and_then(|usage| usage.get("total_tokens")),
        Some(&json!(9))
    );
}
