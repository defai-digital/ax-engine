use crate::routes::build_router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};

use super::fixtures::{json_response, llama_cpp_state, text_response};

#[tokio::test]
async fn api_key_auth_rejects_missing_or_invalid_bearer_token() {
    let app = build_router(llama_cpp_state().with_api_key(Some("secret".to_string())));

    let (status, _, body) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(body.contains("missing or invalid bearer token"));

    let (status, _, _) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header(header::AUTHORIZATION, "Bearer wrong")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn api_key_auth_accepts_valid_bearer_token_and_leaves_health_public() {
    let app = build_router(llama_cpp_state().with_api_key(Some("secret".to_string())));

    let (health_status, health_json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(health_status, StatusCode::OK);
    assert_eq!(health_json["status"], "ok");

    let (models_status, models_json) = json_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header(header::AUTHORIZATION, "Bearer secret")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(models_status, StatusCode::OK);
    assert_eq!(models_json["object"], "list");
}

#[tokio::test]
async fn metrics_route_requires_auth_and_emits_prometheus_text() {
    let app = build_router(llama_cpp_state().with_api_key(Some("secret".to_string())));

    let (unauthorized_status, _, _) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(unauthorized_status, StatusCode::UNAUTHORIZED);

    let (status, content_type, body) = text_response(
        &app,
        Request::builder()
            .method("GET")
            .uri("/metrics")
            .header(header::AUTHORIZATION, "Bearer secret")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(content_type.contains("text/plain"));
    assert!(body.contains("# TYPE ax_engine_http_requests_total counter"));
    assert!(body.contains("ax_engine_http_requests_in_flight"));
}

#[test]
fn keyless_host_allowlist_admits_local_names_and_ip_literals_only() {
    use crate::routes::HostAllowlist;
    let default = HostAllowlist::from_env(None);
    for host in [
        "127.0.0.1:31418",
        "localhost:31418",
        "LOCALHOST",
        "[::1]:31418",
        "192.168.1.20:31418",
        "mac-studio.local:31418",
        "app.localhost",
    ] {
        assert!(default.host_is_allowed(host), "{host}");
    }
    for host in [
        "evil.example:31418",
        "rebind.attacker.com",
        "localhost.evil.com",
    ] {
        assert!(!default.host_is_allowed(host), "{host}");
    }
    let custom = HostAllowlist::from_env(Some("studio.lan, Inference.Example."));
    assert!(custom.host_is_allowed("studio.lan:31418"));
    assert!(custom.host_is_allowed("inference.example"));
    assert!(HostAllowlist::from_env(Some("*")).host_is_allowed("anything.example"));
}

#[tokio::test]
async fn keyless_server_rejects_dns_rebinding_hosts_and_foreign_origins() {
    let app = build_router(llama_cpp_state());
    let request = |host: &str, origin: Option<&str>| {
        let mut builder = Request::builder()
            .uri("/v1/models")
            .header(header::HOST, host);
        if let Some(origin) = origin {
            builder = builder.header(header::ORIGIN, origin);
        }
        builder.body(Body::empty()).unwrap()
    };
    let (status, _, _) = text_response(&app, request("127.0.0.1:31418", None)).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, _) = text_response(&app, request("rebind.attacker.com:31418", None)).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _, _) = text_response(
        &app,
        request("127.0.0.1:31418", Some("https://attacker.example")),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "cross-site browser origin");
    let (status, _, _) = text_response(&app, request("127.0.0.1:31418", Some("null"))).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "opaque origin");
    let (status, _, _) = text_response(
        &app,
        request("127.0.0.1:31418", Some("http://localhost:3000")),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "a local web UI stays allowed");

    // With an API key the key is the boundary; any Host is accepted.
    let keyed = build_router(llama_cpp_state().with_api_key(Some("secret".to_string())));
    let response = Request::builder()
        .uri("/v1/models")
        .header(header::HOST, "studio.example")
        .header(header::AUTHORIZATION, "Bearer secret")
        .body(Body::empty())
        .unwrap();
    let (status, _, _) = text_response(&keyed, response).await;
    assert_eq!(status, StatusCode::OK);
}
