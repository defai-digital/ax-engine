//! Bearer-token authentication for the gRPC surface.
//!
//! Mirrors the HTTP auth middleware in `routes.rs`: when `--api-key` is
//! configured, every RPC except the health probe must carry
//! `authorization: Bearer <key>`. Implemented as a tower layer (not a tonic
//! interceptor) because interceptors cannot observe the request path, and the
//! health exemption is path-based to match the HTTP `/health` behavior.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::http;
use tonic::Status;
use tonic::body::BoxBody;
use tower::{Layer, Service};

use super::routes::bearer_value_matches;

/// gRPC path of the health RPC, exempt from bearer auth so probes keep
/// working without credentials, matching the HTTP `/health` exemption.
const GRPC_HEALTH_PATH: &str = "/ax_engine.v1.AxEngine/Health";

#[derive(Clone)]
pub(crate) struct GrpcAuthLayer {
    api_key: Option<Arc<String>>,
}

impl GrpcAuthLayer {
    pub(crate) fn new(api_key: Option<Arc<String>>) -> Self {
        Self { api_key }
    }
}

impl<S> Layer<S> for GrpcAuthLayer {
    type Service = GrpcAuth<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GrpcAuth {
            inner,
            api_key: self.api_key.clone(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct GrpcAuth<S> {
    inner: S,
    api_key: Option<Arc<String>>,
}

impl<S, ReqBody> Service<http::Request<ReqBody>> for GrpcAuth<S>
where
    S: Service<http::Request<ReqBody>, Response = http::Response<BoxBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
    ReqBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: http::Request<ReqBody>) -> Self::Future {
        let authorized = match self.api_key.as_deref() {
            None => true,
            Some(expected) => {
                request.uri().path() == GRPC_HEALTH_PATH
                    || request
                        .headers()
                        .get(http::header::AUTHORIZATION)
                        .and_then(|value| value.to_str().ok())
                        .is_some_and(|value| bearer_value_matches(value, expected))
            }
        };
        if !authorized {
            return Box::pin(std::future::ready(Ok(Status::unauthenticated(
                "missing or invalid bearer token",
            )
            .into_http())));
        }
        // Swap `inner` for a fresh clone so the moved service is the one that
        // was polled ready (the standard tower clone-and-replace pattern).
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        Box::pin(async move { inner.call(request).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct AlwaysOk;

    impl Service<http::Request<BoxBody>> for AlwaysOk {
        type Response = http::Response<BoxBody>;
        type Error = std::convert::Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _request: http::Request<BoxBody>) -> Self::Future {
            std::future::ready(Ok(http::Response::new(tonic::body::empty_body())))
        }
    }

    fn request(path: &str, authorization: Option<&str>) -> http::Request<BoxBody> {
        let mut builder = http::Request::builder().uri(path);
        if let Some(value) = authorization {
            builder = builder.header(http::header::AUTHORIZATION, value);
        }
        builder
            .body(tonic::body::empty_body())
            .expect("test request must build")
    }

    fn grpc_status(response: &http::Response<BoxBody>) -> Option<&str> {
        response
            .headers()
            .get("grpc-status")
            .and_then(|value| value.to_str().ok())
    }

    async fn call(
        api_key: Option<&str>,
        path: &str,
        authorization: Option<&str>,
    ) -> http::Response<BoxBody> {
        let layer = GrpcAuthLayer::new(api_key.map(|key| Arc::new(key.to_string())));
        let mut service = layer.layer(AlwaysOk);
        service
            .call(request(path, authorization))
            .await
            .expect("infallible")
    }

    #[tokio::test]
    async fn no_api_key_passes_through() {
        let response = call(None, "/ax_engine.v1.AxEngine/Generate", None).await;
        assert_eq!(grpc_status(&response), None);
    }

    #[tokio::test]
    async fn missing_token_is_unauthenticated() {
        let response = call(Some("secret"), "/ax_engine.v1.AxEngine/Generate", None).await;
        // 16 = UNAUTHENTICATED
        assert_eq!(grpc_status(&response), Some("16"));
    }

    #[tokio::test]
    async fn wrong_token_is_unauthenticated() {
        let response = call(
            Some("secret"),
            "/ax_engine.v1.AxEngine/Generate",
            Some("Bearer wrong"),
        )
        .await;
        assert_eq!(grpc_status(&response), Some("16"));
    }

    #[tokio::test]
    async fn valid_token_passes_through() {
        let response = call(
            Some("secret"),
            "/ax_engine.v1.AxEngine/Generate",
            Some("Bearer secret"),
        )
        .await;
        assert_eq!(grpc_status(&response), None);
    }

    #[tokio::test]
    async fn scheme_is_case_insensitive() {
        let response = call(
            Some("secret"),
            "/ax_engine.v1.AxEngine/Generate",
            Some("bearer secret"),
        )
        .await;
        assert_eq!(grpc_status(&response), None);
    }

    #[tokio::test]
    async fn health_probe_is_exempt() {
        let response = call(Some("secret"), GRPC_HEALTH_PATH, None).await;
        assert_eq!(grpc_status(&response), None);
    }
}
