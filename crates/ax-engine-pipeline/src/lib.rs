//! Authenticated HTTP data plane for one static AX Engine pipeline rank.

pub mod artifacts;
pub mod client;
pub mod gateway;

use std::sync::{Arc, Mutex};

use ax_engine_core::{ActivationFrame, PipelineTopology};
use ax_engine_mlx::pipeline::{PipelineRankError, PipelineRankExecutor, PipelineRankOutput};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use mlx_sys::{argmax, eval};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const CLUSTER_WORKER_TOKEN_HEADER: &str = "x-ax-cluster-worker-token";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenStepRequest {
    pub request_id: u64,
    pub request_sequence: u64,
    pub token_offset: u64,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenStepResponse {
    pub request_id: u64,
    pub request_sequence: u64,
    pub token_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankHealth {
    pub ready: bool,
    pub rank: u16,
    pub generation: u64,
    pub cluster_id: String,
    pub manifest_digest: String,
    pub model_artifact_digest: String,
}

pub enum RankStepOutput {
    Activation(Vec<u8>),
    Token(TokenStepResponse),
}

pub trait RankProcessor: Send + Sync + 'static {
    fn health(&self) -> RankHealth;
    fn process_tokens(&self, request: TokenStepRequest)
    -> Result<RankStepOutput, RankServiceError>;
    fn process_activation(&self, bytes: &[u8]) -> Result<RankStepOutput, RankServiceError>;
    fn close_request(&self, request_id: u64) -> Result<(), RankServiceError>;
}

pub struct MlxRankProcessor {
    topology: PipelineTopology,
    rank: u16,
    executor: Mutex<PipelineRankExecutor>,
    maximum_activation_bytes: u64,
}

impl MlxRankProcessor {
    pub fn new(
        topology: PipelineTopology,
        rank: u16,
        executor: PipelineRankExecutor,
        maximum_activation_bytes: u64,
    ) -> Self {
        Self {
            topology,
            rank,
            executor: Mutex::new(executor),
            maximum_activation_bytes,
        }
    }

    fn encode_output(
        &self,
        request_id: u64,
        request_sequence: u64,
        output: PipelineRankOutput,
    ) -> Result<RankStepOutput, RankServiceError> {
        match output {
            PipelineRankOutput::Activation(frame) => {
                Ok(RankStepOutput::Activation(frame.encode(&self.topology)?))
            }
            PipelineRankOutput::Logits(logits) => {
                let token = argmax(&logits, None);
                eval(&[&token]);
                Ok(RankStepOutput::Token(TokenStepResponse {
                    request_id,
                    request_sequence,
                    token_id: token.first_u32_unchecked(),
                }))
            }
        }
    }

    fn lock_executor(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, PipelineRankExecutor>, RankServiceError> {
        self.executor
            .lock()
            .map_err(|_| RankServiceError::ExecutorPoisoned)
    }
}

impl RankProcessor for MlxRankProcessor {
    fn health(&self) -> RankHealth {
        RankHealth {
            ready: self.executor.lock().is_ok(),
            rank: self.rank,
            generation: self.topology.generation,
            cluster_id: self.topology.cluster_id.clone(),
            manifest_digest: self.topology.manifest_digest.clone(),
            model_artifact_digest: self.topology.model_artifact_digest.clone(),
        }
    }

    fn process_tokens(
        &self,
        request: TokenStepRequest,
    ) -> Result<RankStepOutput, RankServiceError> {
        let mut executor = self.lock_executor()?;
        let output = executor.execute_tokens(
            request.request_id,
            request.request_sequence,
            request.token_offset,
            &request.token_ids,
        )?;
        drop(executor);
        self.encode_output(request.request_id, request.request_sequence, output)
    }

    fn process_activation(&self, bytes: &[u8]) -> Result<RankStepOutput, RankServiceError> {
        let frame = ActivationFrame::decode(bytes, &self.topology, self.maximum_activation_bytes)?;
        let request_id = frame.header.request_id;
        let request_sequence = frame.header.request_sequence;
        let mut executor = self.lock_executor()?;
        let output = executor.execute_activation(&frame)?;
        drop(executor);
        self.encode_output(request_id, request_sequence, output)
    }

    fn close_request(&self, request_id: u64) -> Result<(), RankServiceError> {
        self.lock_executor()?.close_request(request_id);
        Ok(())
    }
}

#[derive(Clone)]
struct RankApiState {
    processor: Arc<dyn RankProcessor>,
    worker_token: Arc<str>,
}

pub fn router(
    processor: Arc<dyn RankProcessor>,
    worker_token: String,
    maximum_body_bytes: usize,
) -> Result<Router, RankServiceError> {
    if worker_token.len() < 16 {
        return Err(RankServiceError::WeakWorkerToken);
    }
    let state = RankApiState {
        processor,
        worker_token: Arc::from(worker_token),
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/internal/pipeline/tokens", post(tokens))
        .route("/internal/pipeline/activation", post(activation))
        .route(
            "/internal/pipeline/requests/:request_id/close",
            post(close_request),
        )
        .layer(DefaultBodyLimit::max(maximum_body_bytes))
        .with_state(state))
}

async fn health(State(state): State<RankApiState>) -> Json<RankHealth> {
    Json(state.processor.health())
}

async fn tokens(
    State(state): State<RankApiState>,
    headers: HeaderMap,
    Json(request): Json<TokenStepRequest>,
) -> Response {
    if !authorized(&headers, &state.worker_token) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let processor = Arc::clone(&state.processor);
    match tokio::task::spawn_blocking(move || processor.process_tokens(request)).await {
        Ok(Ok(output)) => output.into_response(),
        Ok(Err(error)) => error.into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("pipeline execution task failed: {error}"),
        )
            .into_response(),
    }
}

async fn activation(
    State(state): State<RankApiState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !authorized(&headers, &state.worker_token) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let processor = Arc::clone(&state.processor);
    match tokio::task::spawn_blocking(move || processor.process_activation(&body)).await {
        Ok(Ok(output)) => output.into_response(),
        Ok(Err(error)) => error.into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("pipeline execution task failed: {error}"),
        )
            .into_response(),
    }
}

async fn close_request(
    State(state): State<RankApiState>,
    headers: HeaderMap,
    axum::extract::Path(request_id): axum::extract::Path<u64>,
) -> Response {
    if !authorized(&headers, &state.worker_token) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let processor = Arc::clone(&state.processor);
    match tokio::task::spawn_blocking(move || processor.close_request(request_id)).await {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        Ok(Err(error)) => error.into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("pipeline close task failed: {error}"),
        )
            .into_response(),
    }
}

fn authorized(headers: &HeaderMap, expected: &str) -> bool {
    let provided = headers
        .get(CLUSTER_WORKER_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    constant_time_eq(provided, expected)
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.as_bytes()
        .iter()
        .zip(right.as_bytes())
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

impl IntoResponse for RankStepOutput {
    fn into_response(self) -> Response {
        match self {
            Self::Activation(bytes) => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/x-ax-pipeline-frame")],
                bytes,
            )
                .into_response(),
            Self::Token(token) => (StatusCode::OK, Json(token)).into_response(),
        }
    }
}

#[derive(Debug, Error)]
pub enum RankServiceError {
    #[error(transparent)]
    Contract(#[from] ax_engine_core::PipelineContractError),
    #[error(transparent)]
    Executor(#[from] PipelineRankError),
    #[error("pipeline executor mutex is poisoned")]
    ExecutorPoisoned,
    #[error("cluster worker token must contain at least 16 bytes")]
    WeakWorkerToken,
}

impl IntoResponse for RankServiceError {
    fn into_response(self) -> Response {
        (StatusCode::UNPROCESSABLE_ENTITY, self.to_string()).into_response()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt as _;

    use super::*;

    struct TestProcessor {
        calls: AtomicUsize,
    }

    impl RankProcessor for TestProcessor {
        fn health(&self) -> RankHealth {
            RankHealth {
                ready: true,
                rank: 1,
                generation: 7,
                cluster_id: "cluster".into(),
                manifest_digest: "manifest".into(),
                model_artifact_digest: "model".into(),
            }
        }

        fn process_tokens(
            &self,
            request: TokenStepRequest,
        ) -> Result<RankStepOutput, RankServiceError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(RankStepOutput::Token(TokenStepResponse {
                request_id: request.request_id,
                request_sequence: request.request_sequence,
                token_id: 42,
            }))
        }

        fn process_activation(&self, bytes: &[u8]) -> Result<RankStepOutput, RankServiceError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(RankStepOutput::Activation(bytes.to_vec()))
        }

        fn close_request(&self, _request_id: u64) -> Result<(), RankServiceError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    fn app(processor: Arc<TestProcessor>) -> Router {
        router(processor, "0123456789abcdef".into(), 1024).expect("router should build")
    }

    #[tokio::test]
    async fn health_is_public_but_execution_requires_worker_token() {
        let processor = Arc::new(TestProcessor {
            calls: AtomicUsize::new(0),
        });
        let health = app(Arc::clone(&processor))
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .expect("health request"),
            )
            .await
            .expect("health response");
        assert_eq!(health.status(), StatusCode::OK);

        let unauthorized = app(Arc::clone(&processor))
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/pipeline/tokens")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"request_id":1,"request_sequence":1,"token_offset":0,"token_ids":[1]}"#,
                    ))
                    .expect("token request"),
            )
            .await
            .expect("unauthorized response");
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(processor.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn authorized_token_step_returns_greedy_token() {
        let processor = Arc::new(TestProcessor {
            calls: AtomicUsize::new(0),
        });
        let response = app(Arc::clone(&processor))
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/pipeline/tokens")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(CLUSTER_WORKER_TOKEN_HEADER, "0123456789abcdef")
                    .body(Body::from(
                        r#"{"request_id":9,"request_sequence":2,"token_offset":4,"token_ids":[3]}"#,
                    ))
                    .expect("token request"),
            )
            .await
            .expect("token response");
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1024)
            .await
            .expect("token body");
        let token =
            serde_json::from_slice::<TokenStepResponse>(&bytes).expect("token response JSON");
        assert_eq!(
            token,
            TokenStepResponse {
                request_id: 9,
                request_sequence: 2,
                token_id: 42,
            }
        );
        assert_eq!(processor.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn weak_worker_token_is_rejected_at_startup() {
        let processor = Arc::new(TestProcessor {
            calls: AtomicUsize::new(0),
        });
        assert!(matches!(
            router(processor, "short".into(), 1024),
            Err(RankServiceError::WeakWorkerToken)
        ));
    }
}
