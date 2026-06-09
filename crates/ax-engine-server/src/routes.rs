use std::sync::Arc;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Request};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use tokio::sync::Semaphore;

use super::DEFAULT_MAX_REQUEST_BODY_BYTES;
use super::app_state::AppState;
use super::generation::lifecycle::{
    cancel_request, request_snapshot, step_request, submit_request,
};
use super::generation::native::generate;
use super::generation::streaming::generate_stream;
use super::metadata::{health, models, runtime_info};
use super::openai::chat::openai_chat_completions;
use super::openai::completions::openai_completions;
use super::openai::embeddings::openai_embeddings;

/// Opt-in cap on the number of in-flight HTTP requests. Unset (or non-positive)
/// means no limit, preserving the default behavior. Operators running the server
/// multi-tenant set this to bound concurrent GPU work and shed load with HTTP
/// 429 instead of exhausting memory under a request flood.
const MAX_CONCURRENT_REQUESTS_ENV: &str = "AX_ENGINE_MAX_CONCURRENT_REQUESTS";
const MAX_REQUEST_BODY_BYTES_ENV: &str = "AX_ENGINE_MAX_REQUEST_BODY_BYTES";

pub(crate) fn build_router(state: AppState) -> Router {
    let router = Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/v1/runtime", get(runtime_info))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(openai_embeddings))
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

    router
        .layer(DefaultBodyLimit::max(max_request_body_bytes_from_env()))
        .with_state(state)
}

fn max_concurrent_requests_from_env() -> Option<usize> {
    parse_max_concurrent_requests(std::env::var(MAX_CONCURRENT_REQUESTS_ENV).ok())
}

fn max_request_body_bytes_from_env() -> usize {
    parse_max_request_body_bytes(std::env::var(MAX_REQUEST_BODY_BYTES_ENV).ok())
        .unwrap_or(DEFAULT_MAX_REQUEST_BODY_BYTES)
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
    use super::{parse_max_concurrent_requests, parse_max_request_body_bytes};

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
}
