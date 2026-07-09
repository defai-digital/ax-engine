use std::sync::Arc;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Request};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use tokio::sync::Semaphore;

use super::DEFAULT_MAX_REQUEST_BODY_BYTES;
use super::anthropic::anthropic_messages;
use super::app_state::{AppState, ServerMetrics};
use super::embeddings::records::embedding_records;
use super::generation::lifecycle::{
    cancel_request, request_snapshot, step_request, submit_request,
};
use super::generation::native::generate;
use super::generation::streaming::generate_stream;
use super::metadata::{health, models, runtime_info};
use super::metrics::prometheus_metrics;
use super::model_load::load_model;
use super::ollama::{
    ollama_chat, ollama_generate, ollama_ps, ollama_show, ollama_tags, ollama_version,
};
use super::openai::chat::openai_chat_completions;
use super::openai::compat::{apply_template, detokenize, props, slots, tokenize};
use super::openai::completions::openai_completions;
use super::openai::embeddings::openai_embeddings;

/// Opt-in cap on the number of in-flight HTTP requests. Unset (or non-positive)
/// means no limit, preserving the default behavior. Operators running the server
/// multi-tenant set this to bound concurrent GPU work and shed load with HTTP
/// 429 instead of exhausting memory under a request flood.
const MAX_CONCURRENT_REQUESTS_ENV: &str = "AX_ENGINE_MAX_CONCURRENT_REQUESTS";
const MAX_REQUEST_BODY_BYTES_ENV: &str = "AX_ENGINE_MAX_REQUEST_BODY_BYTES";

/// Opt-in per-request timeout in seconds, applied to both the HTTP router and
/// the gRPC server. Unset (or non-positive) means no timeout, preserving the
/// default behavior. The timeout bounds the time to produce a response; for
/// streaming endpoints that is the time to the first byte of the stream, so
/// long generations that are actively streaming are not cut off — only hung
/// handlers and stuck non-streaming generations are.
const REQUEST_TIMEOUT_SECS_ENV: &str = "AX_ENGINE_REQUEST_TIMEOUT_SECS";

pub(crate) fn build_router(state: AppState) -> Router {
    let router = Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/metrics", get(prometheus_metrics))
        .route("/v1/runtime", get(runtime_info))
        .route("/v1/models", get(models))
        .route("/api/tags", get(ollama_tags))
        .route("/api/show", post(ollama_show))
        .route("/api/ps", get(ollama_ps))
        .route("/api/version", get(ollama_version))
        .route("/api/chat", post(ollama_chat))
        .route("/api/generate", post(ollama_generate))
        .route("/props", get(props))
        .route("/slots", get(slots))
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/apply-template", post(apply_template))
        .route("/v1/apply-template", post(apply_template))
        .route("/v1/model/load", post(load_model))
        .route("/v1/embeddings", post(openai_embeddings))
        .route("/v1/embedding_records", post(embedding_records))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/step", post(step_request))
        .route("/v1/requests", post(submit_request))
        .route("/v1/requests/:request_id", get(request_snapshot))
        .route("/v1/requests/:request_id/cancel", post(cancel_request))
        .route("/v1/generate/stream", post(generate_stream))
        .route("/v1/generate", post(generate));

    // Apply the concurrency limiter (when enabled) closest to the handlers, so a
    // rejected request never starts engine work. Requests beyond the limit get
    // an immediate 429 rather than queueing unboundedly.
    let router = match max_concurrent_requests_from_env() {
        Some(limit) => {
            let semaphore = Arc::new(Semaphore::new(limit));
            router.layer(middleware::from_fn(move |request: Request, next: Next| {
                let semaphore = Arc::clone(&semaphore);
                async move {
                    match semaphore.try_acquire_owned() {
                        Ok(permit) => {
                            let response = next.run(request).await;
                            drop(permit);
                            response
                        }
                        Err(_) => (
                            StatusCode::TOO_MANY_REQUESTS,
                            "server is at its maximum concurrent request limit; retry shortly",
                        )
                            .into_response(),
                    }
                }
            }))
        }
        None => router,
    };

    let router = match state.api_key.clone() {
        Some(api_key) => router.layer(middleware::from_fn(move |request: Request, next: Next| {
            let api_key = api_key.clone();
            async move {
                if is_health_probe(request.uri().path())
                    || request_has_valid_bearer_token(&request, api_key.as_ref())
                {
                    return next.run(request).await;
                }
                (
                    StatusCode::UNAUTHORIZED,
                    [(header::WWW_AUTHENTICATE, "Bearer")],
                    "missing or invalid bearer token",
                )
                    .into_response()
            }
        })),
        None => router,
    };

    // Inside the metrics layer so timed-out requests are recorded as 408s.
    let router = match request_timeout_from_env() {
        Some(timeout) => router.layer(middleware::from_fn(
            move |request: Request, next: Next| async move {
                match tokio::time::timeout(timeout, next.run(request)).await {
                    Ok(response) => response,
                    Err(_) => (
                        StatusCode::REQUEST_TIMEOUT,
                        "request exceeded the configured server timeout",
                    )
                        .into_response(),
                }
            },
        )),
        None => router,
    };

    let metrics = state.metrics.clone();
    let router = router.layer(middleware::from_fn(move |request: Request, next: Next| {
        let metrics = metrics.clone();
        async move {
            let guard = InFlightRequestGuard::new(metrics);
            let response = next.run(request).await;
            guard.finish(response.status());
            response
        }
    }));

    router
        .layer(DefaultBodyLimit::max(max_request_body_bytes_from_env()))
        .with_state(state)
}

/// RAII pairing for [`ServerMetrics::begin_http_request`]. A plain
/// begin/finish call pair leaks the `http_requests_in_flight` gauge forever
/// whenever the request future is dropped before completion — e.g. a client
/// disconnect or reverse-proxy timeout during a slow LLM generation, which
/// cancels the in-flight `next.run(request).await` mid-poll and skips any
/// code after it. `Drop` guarantees the gauge is always decremented exactly
/// once, matching the cancel-safe `OwnedSemaphorePermit` pattern the
/// concurrency-limiter layer above already relies on for the same reason.
struct InFlightRequestGuard {
    metrics: Arc<ServerMetrics>,
    finished: bool,
}

impl InFlightRequestGuard {
    fn new(metrics: Arc<ServerMetrics>) -> Self {
        metrics.begin_http_request();
        Self {
            metrics,
            finished: false,
        }
    }

    fn finish(mut self, status: StatusCode) {
        self.finished = true;
        self.metrics.finish_http_request(status);
    }
}

impl Drop for InFlightRequestGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.metrics.abandon_http_request();
        }
    }
}

fn is_health_probe(path: &str) -> bool {
    matches!(path, "/health" | "/healthz")
}

fn request_has_valid_bearer_token<B>(request: &axum::http::Request<B>, expected: &str) -> bool {
    request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| bearer_value_matches(value, expected))
}

/// Validate a raw `authorization` header value against the configured key.
/// Shared by the HTTP middleware and the gRPC auth layer.
pub(crate) fn bearer_value_matches(value: &str, expected: &str) -> bool {
    let Some((scheme, token)) = value.split_once(' ') else {
        return false;
    };
    scheme.eq_ignore_ascii_case("Bearer") && constant_time_str_eq(token.trim(), expected)
}

/// Compare without short-circuiting on the first mismatched byte, so response
/// latency does not leak how long a matching token prefix is. Token length
/// remains observable, which is standard for bearer-token checks.
fn constant_time_str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn max_concurrent_requests_from_env() -> Option<usize> {
    parse_max_concurrent_requests(std::env::var(MAX_CONCURRENT_REQUESTS_ENV).ok())
}

fn max_request_body_bytes_from_env() -> usize {
    parse_max_request_body_bytes(std::env::var(MAX_REQUEST_BODY_BYTES_ENV).ok())
        .unwrap_or(DEFAULT_MAX_REQUEST_BODY_BYTES)
}

pub(crate) fn request_timeout_from_env() -> Option<std::time::Duration> {
    parse_request_timeout_secs(std::env::var(REQUEST_TIMEOUT_SECS_ENV).ok())
        .map(std::time::Duration::from_secs)
}

/// Parse the concurrency cap: a positive integer enables the limit; anything else
/// (unset, empty, zero, negative, non-numeric) disables it.
fn parse_max_concurrent_requests(value: Option<String>) -> Option<usize> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&limit| limit > 0)
}

/// Parse the request timeout: a positive integer (seconds) enables the timeout;
/// anything else (unset, empty, zero, negative, non-numeric) disables it.
fn parse_request_timeout_secs(value: Option<String>) -> Option<u64> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
}

/// Parse the request body byte cap: a positive integer overrides the default;
/// unset, empty, zero, negative, and non-numeric values keep the safe default.
fn parse_max_request_body_bytes(value: Option<String>) -> Option<usize> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&limit| limit > 0)
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};

    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use super::{
        InFlightRequestGuard, ServerMetrics, constant_time_str_eq, parse_max_concurrent_requests,
        parse_max_request_body_bytes, parse_request_timeout_secs, request_has_valid_bearer_token,
    };

    #[test]
    fn constant_time_eq_matches_str_eq_semantics() {
        assert!(constant_time_str_eq("secret", "secret"));
        assert!(!constant_time_str_eq("secret", "secres"));
        assert!(!constant_time_str_eq("secret", "Secret"));
        assert!(!constant_time_str_eq("secret", "secret1"));
        assert!(!constant_time_str_eq("", "secret"));
        assert!(constant_time_str_eq("", ""));
    }

    #[test]
    fn parses_positive_limit() {
        assert_eq!(
            parse_max_concurrent_requests(Some("8".to_string())),
            Some(8)
        );
        assert_eq!(
            parse_max_concurrent_requests(Some("  16 ".to_string())),
            Some(16)
        );
    }

    #[test]
    fn rejects_disabled_or_invalid_values() {
        assert_eq!(parse_max_concurrent_requests(None), None);
        assert_eq!(parse_max_concurrent_requests(Some(String::new())), None);
        assert_eq!(parse_max_concurrent_requests(Some("0".to_string())), None);
        assert_eq!(parse_max_concurrent_requests(Some("-4".to_string())), None);
        assert_eq!(
            parse_max_concurrent_requests(Some("lots".to_string())),
            None
        );
    }

    #[test]
    fn parses_positive_request_body_limit() {
        assert_eq!(
            parse_max_request_body_bytes(Some("134217728".to_string())),
            Some(134217728)
        );
        assert_eq!(
            parse_max_request_body_bytes(Some(" 268435456 ".to_string())),
            Some(268435456)
        );
    }

    #[test]
    fn rejects_disabled_or_invalid_request_body_limit() {
        assert_eq!(parse_max_request_body_bytes(None), None);
        assert_eq!(parse_max_request_body_bytes(Some(String::new())), None);
        assert_eq!(parse_max_request_body_bytes(Some("0".to_string())), None);
        assert_eq!(parse_max_request_body_bytes(Some("-1".to_string())), None);
        assert_eq!(
            parse_max_request_body_bytes(Some("256MiB".to_string())),
            None
        );
    }

    #[test]
    fn parses_positive_request_timeout() {
        assert_eq!(parse_request_timeout_secs(Some("30".to_string())), Some(30));
        assert_eq!(
            parse_request_timeout_secs(Some(" 600 ".to_string())),
            Some(600)
        );
    }

    #[test]
    fn rejects_disabled_or_invalid_request_timeout() {
        assert_eq!(parse_request_timeout_secs(None), None);
        assert_eq!(parse_request_timeout_secs(Some(String::new())), None);
        assert_eq!(parse_request_timeout_secs(Some("0".to_string())), None);
        assert_eq!(parse_request_timeout_secs(Some("-30".to_string())), None);
        assert_eq!(parse_request_timeout_secs(Some("30s".to_string())), None);
    }

    #[test]
    fn validates_bearer_authorization_header() {
        let request = Request::builder()
            .header(header::AUTHORIZATION, "Bearer secret")
            .body(Body::empty())
            .unwrap();
        assert!(request_has_valid_bearer_token(&request, "secret"));

        let lower_scheme = Request::builder()
            .header(header::AUTHORIZATION, "bearer secret")
            .body(Body::empty())
            .unwrap();
        assert!(request_has_valid_bearer_token(&lower_scheme, "secret"));

        let wrong_token = Request::builder()
            .header(header::AUTHORIZATION, "Bearer other")
            .body(Body::empty())
            .unwrap();
        assert!(!request_has_valid_bearer_token(&wrong_token, "secret"));

        let missing = Request::builder().body(Body::empty()).unwrap();
        assert!(!request_has_valid_bearer_token(&missing, "secret"));
    }

    #[test]
    fn in_flight_guard_decrements_on_normal_finish() {
        let metrics = Arc::new(ServerMetrics::default());
        let guard = InFlightRequestGuard::new(metrics.clone());
        assert_eq!(metrics.http_requests_in_flight.load(Ordering::Relaxed), 1);

        guard.finish(StatusCode::OK);
        assert_eq!(metrics.http_requests_in_flight.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.http_status_2xx_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn in_flight_guard_decrements_on_drop_without_finish() {
        // Reproduces a client disconnect / reverse-proxy timeout: the
        // request future is dropped mid-poll, so `finish()` is never
        // called. Before the fix, `http_requests_in_flight` leaked
        // permanently in this case.
        let metrics = Arc::new(ServerMetrics::default());
        {
            let _guard = InFlightRequestGuard::new(metrics.clone());
            assert_eq!(metrics.http_requests_in_flight.load(Ordering::Relaxed), 1);
        }
        assert_eq!(
            metrics.http_requests_in_flight.load(Ordering::Relaxed),
            0,
            "dropping the guard without finish() must still release the in-flight gauge"
        );
        // No status bucket should be attributed to an abandoned request.
        assert_eq!(metrics.http_status_2xx_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.http_status_4xx_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.http_status_5xx_total.load(Ordering::Relaxed), 0);
    }
}
