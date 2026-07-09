//! Request metrics for the gRPC surface.
//!
//! Mirrors the HTTP `InFlightRequestGuard`/metrics wiring in `routes.rs` and
//! the tower `Layer`/`Service` structure of `grpc_auth.rs`, since gRPC
//! traffic was previously entirely uncounted by `/metrics`.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::http;
use tonic::body::BoxBody;
use tower::{Layer, Service};

use crate::app_state::ServerMetrics;

#[derive(Clone)]
pub(crate) struct GrpcMetricsLayer {
    metrics: Arc<ServerMetrics>,
}

impl GrpcMetricsLayer {
    pub(crate) fn new(metrics: Arc<ServerMetrics>) -> Self {
        Self { metrics }
    }
}

impl<S> Layer<S> for GrpcMetricsLayer {
    type Service = GrpcMetrics<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GrpcMetrics {
            inner,
            metrics: Arc::clone(&self.metrics),
        }
    }
}

#[derive(Clone)]
pub(crate) struct GrpcMetrics<S> {
    inner: S,
    metrics: Arc<ServerMetrics>,
}

impl<S, ReqBody> Service<http::Request<ReqBody>> for GrpcMetrics<S>
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
        let metrics = Arc::clone(&self.metrics);
        // Swap `inner` for a fresh clone so the moved service is the one
        // that was polled ready (the standard tower clone-and-replace
        // pattern, matching `GrpcAuth::call`).
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        Box::pin(async move {
            let guard = InFlightGrpcRequestGuard::new(metrics);
            let response = inner.call(request).await?;
            // `grpc-status` lives in HTTP/2 trailers for successful unary
            // and all streaming responses, which this tower layer cannot
            // see — those are absent here and counted as "ok" by
            // `ServerMetrics::finish_grpc_request`'s documented default.
            let status = response
                .headers()
                .get("grpc-status")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            guard.finish(status.as_deref());
            Ok(response)
        })
    }
}

/// RAII pairing for [`ServerMetrics::begin_grpc_request`], mirroring
/// `routes.rs`'s `InFlightRequestGuard` so a cancelled/dropped call future
/// still decrements the in-flight gauge exactly once.
struct InFlightGrpcRequestGuard {
    metrics: Arc<ServerMetrics>,
    finished: bool,
}

impl InFlightGrpcRequestGuard {
    fn new(metrics: Arc<ServerMetrics>) -> Self {
        metrics.begin_grpc_request();
        Self {
            metrics,
            finished: false,
        }
    }

    fn finish(mut self, status: Option<&str>) {
        self.finished = true;
        self.metrics.finish_grpc_request(status);
    }
}

impl Drop for InFlightGrpcRequestGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.metrics.abandon_grpc_request();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    #[derive(Clone)]
    struct FixedStatus {
        status: Option<&'static str>,
    }

    impl Service<http::Request<BoxBody>> for FixedStatus {
        type Response = http::Response<BoxBody>;
        type Error = std::convert::Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _request: http::Request<BoxBody>) -> Self::Future {
            let mut response = http::Response::new(tonic::body::empty_body());
            if let Some(status) = self.status {
                response
                    .headers_mut()
                    .insert("grpc-status", status.parse().expect("valid header value"));
            }
            std::future::ready(Ok(response))
        }
    }

    fn request() -> http::Request<BoxBody> {
        http::Request::builder()
            .uri("/ax_engine.v1.AxEngine/Generate")
            .body(tonic::body::empty_body())
            .expect("test request must build")
    }

    #[tokio::test]
    async fn counts_ok_status_header() {
        let metrics = Arc::new(ServerMetrics::default());
        let layer = GrpcMetricsLayer::new(Arc::clone(&metrics));
        let mut service = layer.layer(FixedStatus { status: Some("0") });
        service.call(request()).await.expect("infallible");

        assert_eq!(metrics.grpc_requests_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.grpc_requests_in_flight.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.grpc_status_ok_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.grpc_status_error_total.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn counts_error_status_header() {
        let metrics = Arc::new(ServerMetrics::default());
        let layer = GrpcMetricsLayer::new(Arc::clone(&metrics));
        let mut service = layer.layer(FixedStatus {
            status: Some("16"), // UNAUTHENTICATED
        });
        service.call(request()).await.expect("infallible");

        assert_eq!(metrics.grpc_status_ok_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.grpc_status_error_total.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn missing_status_header_counts_as_ok() {
        // Matches successful unary responses and all streaming RPCs, whose
        // real status lives in trailers this layer cannot see.
        let metrics = Arc::new(ServerMetrics::default());
        let layer = GrpcMetricsLayer::new(Arc::clone(&metrics));
        let mut service = layer.layer(FixedStatus { status: None });
        service.call(request()).await.expect("infallible");

        assert_eq!(metrics.grpc_status_ok_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.grpc_status_error_total.load(Ordering::Relaxed), 0);
    }
}
