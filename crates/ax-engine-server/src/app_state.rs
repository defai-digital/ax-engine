use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use ax_engine_sdk::{
    EmbeddingPooling, EngineSession, EngineSessionConfig, EngineSessionError, EngineStepReport,
    RuntimeReport, StatelessGenerateContext,
};
use tokio::sync::{mpsc, oneshot};

use crate::admission::{AdmissionController, AdmissionPermit};
use crate::generation::service::NativeGenerationService;
use crate::generation::streaming::StreamDeadlines;
use crate::rate_limit::RateLimitConfig;

/// All state that changes atomically when a new model is loaded.
/// Wrapped behind `Arc<RwLock<>>` in `AppState` so every in-flight request
/// that has already cloned its Arcs continues with the old model, while new
/// requests immediately see the replacement.
#[derive(Clone)]
pub(crate) struct LiveState {
    pub(crate) model_id: Arc<String>,
    pub(crate) session_config: Arc<EngineSessionConfig>,
    pub(crate) stateless_generate_context: Arc<StatelessGenerateContext>,
    pub(crate) runtime_report: RuntimeReport,
    pub(crate) generation_service: Arc<NativeGenerationService>,
    pub(crate) embedding_batcher: Arc<EmbeddingMicroBatcher>,
}

#[derive(Clone)]
pub(crate) struct AppState {
    /// Swappable model state — take a snapshot at the start of each handler.
    pub(crate) live: Arc<parking_lot::RwLock<LiveState>>,
    pub(crate) api_key: Option<Arc<String>>,
    pub(crate) metrics: Arc<ServerMetrics>,
    /// Set to true while a model load is in progress; prevents concurrent loads.
    pub(crate) loading: Arc<AtomicBool>,
    pub(crate) limits: Arc<ServerLimits>,
    pub(crate) admission: Arc<AdmissionController>,
    pub(crate) stepwise_admission: Arc<parking_lot::Mutex<HashMap<u64, AdmissionPermit>>>,
    next_request_id: Arc<AtomicU64>,
}

impl AppState {
    pub(crate) fn new(live: LiveState) -> Self {
        Self {
            live: Arc::new(parking_lot::RwLock::new(live)),
            api_key: None,
            metrics: Arc::new(ServerMetrics::default()),
            loading: Arc::new(AtomicBool::new(false)),
            limits: Arc::new(ServerLimits::default()),
            admission: Arc::new(AdmissionController::new(None)),
            stepwise_admission: Arc::new(parking_lot::Mutex::new(HashMap::new())),
            next_request_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Clone all live-model fields atomically. The read lock is held only for
    /// the duration of the Arc clones — never across an await point.
    pub(crate) fn snapshot(&self) -> LiveState {
        self.live.read().clone()
    }

    /// Replace the live model state. Called by the load endpoint after
    /// successfully building a new session outside the lock.
    pub(crate) fn swap_live(&self, new: LiveState) {
        *self.live.write() = new;
    }

    pub(crate) fn allocate_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::AcqRel)
    }

    pub(crate) fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key.map(Arc::new);
        self
    }

    pub(crate) fn with_limits(mut self, limits: ServerLimits) -> Self {
        self.admission = Arc::new(AdmissionController::new(limits.max_concurrent_requests));
        self.limits = Arc::new(limits);
        self
    }
}

/// Resource limits resolved from CLI flags / env vars at startup (see
/// `ServerArgs::resolved_*` in `args.rs`). All fields default to "disabled"
/// except `max_request_body_bytes`, which always enforces the built-in
/// safe default — this preserves today's behavior exactly when no operator
/// configuration is supplied.
pub(crate) struct ServerLimits {
    pub(crate) max_concurrent_requests: Option<usize>,
    pub(crate) max_request_body_bytes: usize,
    pub(crate) request_timeout: Option<Duration>,
    pub(crate) grpc_request_timeout: Option<Duration>,
    pub(crate) rate_limit: Option<RateLimitConfig>,
    pub(crate) stream_deadlines: StreamDeadlines,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            max_concurrent_requests: None,
            max_request_body_bytes: crate::DEFAULT_MAX_REQUEST_BODY_BYTES,
            request_timeout: None,
            grpc_request_timeout: None,
            rate_limit: None,
            stream_deadlines: StreamDeadlines::default(),
        }
    }
}

#[derive(Default)]
pub(crate) struct ServerMetrics {
    pub(crate) http_requests_total: AtomicU64,
    pub(crate) http_requests_in_flight: AtomicU64,
    pub(crate) http_status_2xx_total: AtomicU64,
    pub(crate) http_status_4xx_total: AtomicU64,
    pub(crate) http_status_5xx_total: AtomicU64,
    pub(crate) grpc_requests_total: AtomicU64,
    pub(crate) grpc_requests_in_flight: AtomicU64,
    pub(crate) grpc_status_ok_total: AtomicU64,
    pub(crate) grpc_status_error_total: AtomicU64,
    engine_step_observed: AtomicBool,
    engine_step_scheduled_requests: AtomicU64,
    engine_step_scheduled_tokens: AtomicU64,
    engine_step_kv_usage_blocks: AtomicU64,
    engine_step_prefix_hits_total: AtomicU64,
}

/// Point-in-time copy of the engine-step gauges cached by
/// [`ServerMetrics::record_step_report`]. `/metrics` reads this snapshot;
/// it must never call `EngineSession::step_report` itself, because that call
/// advances the engine (native) or consumes request stream chunks (llama.cpp).
#[derive(Clone, Copy, Debug)]
pub(crate) struct EngineStepGauges {
    pub(crate) scheduled_requests: u64,
    pub(crate) scheduled_tokens: u64,
    pub(crate) kv_usage_blocks: u64,
    pub(crate) prefix_hits_total: u64,
}

impl ServerMetrics {
    pub(crate) fn begin_http_request(&self) {
        self.http_requests_total.fetch_add(1, Ordering::Relaxed);
        self.http_requests_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn finish_http_request(&self, status: axum::http::StatusCode) {
        self.http_requests_in_flight.fetch_sub(1, Ordering::Relaxed);
        if status.is_success() {
            self.http_status_2xx_total.fetch_add(1, Ordering::Relaxed);
        } else if status.is_client_error() {
            self.http_status_4xx_total.fetch_add(1, Ordering::Relaxed);
        } else if status.is_server_error() {
            self.http_status_5xx_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Decrement the in-flight gauge without recording a status bucket.
    /// Used when a request's future is dropped before completion (client
    /// disconnect, upstream cancellation) instead of running to a real
    /// response — there is no status code to attribute in that case.
    pub(crate) fn abandon_http_request(&self) {
        self.http_requests_in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    pub(crate) fn begin_grpc_request(&self) {
        self.grpc_requests_total.fetch_add(1, Ordering::Relaxed);
        self.grpc_requests_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// `status` is the `grpc-status` response header value when present
    /// ("0" is `tonic::Code::Ok`). It is absent for successful unary
    /// responses and for all streaming RPCs, since their real status lives
    /// in HTTP/2 trailers rather than headers at the tower layer that calls
    /// this — those are counted as "ok" by default (see `grpc_metrics.rs`).
    pub(crate) fn finish_grpc_request(&self, status: Option<&str>) {
        self.grpc_requests_in_flight.fetch_sub(1, Ordering::Relaxed);
        if matches!(status, None | Some("0")) {
            self.grpc_status_ok_total.fetch_add(1, Ordering::Relaxed);
        } else {
            self.grpc_status_error_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Mirrors [`Self::abandon_http_request`] for a cancelled/dropped gRPC
    /// call future — decrement in-flight without attributing a status.
    pub(crate) fn abandon_grpc_request(&self) {
        self.grpc_requests_in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    pub(crate) fn record_step_report(&self, report: &EngineStepReport) {
        self.engine_step_scheduled_requests
            .store(report.scheduled_requests as u64, Ordering::Relaxed);
        self.engine_step_scheduled_tokens
            .store(report.scheduled_tokens as u64, Ordering::Relaxed);
        self.engine_step_kv_usage_blocks
            .store(report.kv_usage_blocks as u64, Ordering::Relaxed);
        self.engine_step_prefix_hits_total
            .fetch_add(report.prefix_hits as u64, Ordering::Relaxed);
        self.engine_step_observed.store(true, Ordering::Release);
    }

    pub(crate) fn engine_step_gauges(&self) -> Option<EngineStepGauges> {
        if !self.engine_step_observed.load(Ordering::Acquire) {
            return None;
        }
        Some(EngineStepGauges {
            scheduled_requests: self.engine_step_scheduled_requests.load(Ordering::Relaxed),
            scheduled_tokens: self.engine_step_scheduled_tokens.load(Ordering::Relaxed),
            kv_usage_blocks: self.engine_step_kv_usage_blocks.load(Ordering::Relaxed),
            prefix_hits_total: self.engine_step_prefix_hits_total.load(Ordering::Relaxed),
        })
    }
}

/// Build a `LiveState` from a freshly created session. Used both at startup
/// and by the load endpoint when swapping to a new model.
pub(crate) fn build_live_state(
    model_id: String,
    session: EngineSession,
) -> Result<LiveState, EngineSessionError> {
    let session_config = session.config().clone();
    let stateless_generate_context =
        StatelessGenerateContext::new(session_config.clone()).map(Arc::new)?;
    let runtime_report = session.runtime_report();
    let generation_service = NativeGenerationService::spawn(session);
    let embedding_batcher = EmbeddingMicroBatcher::spawn(generation_service.clone());
    Ok(LiveState {
        model_id: Arc::new(model_id),
        session_config: Arc::new(session_config),
        stateless_generate_context,
        runtime_report,
        generation_service,
        embedding_batcher,
    })
}

pub(crate) fn build_app_state(
    model_id: String,
    session: EngineSession,
) -> Result<AppState, EngineSessionError> {
    let live = build_live_state(model_id, session)?;
    Ok(AppState::new(live))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ax_engine_sdk::{PreviewBackendRequest, PreviewSessionConfigRequest, SupportTier};

    use super::*;

    fn test_state(model_id: &str) -> AppState {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::LlamaCpp,
                llama_model_path: Some(PathBuf::from("fake-model.gguf")),
                ..PreviewBackendRequest::default()
            },
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview session config should build");
        let session = EngineSession::new(config).expect("session should build");
        build_app_state(model_id.to_string(), session).expect("app state should build")
    }

    #[tokio::test]
    async fn snapshot_reads_live_state() {
        let state = test_state("first");
        let live = state.snapshot();
        assert_eq!(live.model_id.as_ref().as_str(), "first");
    }

    #[tokio::test]
    async fn swap_live_replaces_state() {
        let state = test_state("first");
        let replacement = test_state("second").snapshot();
        state.swap_live(replacement);
        let live = state.snapshot();
        assert_eq!(live.model_id.as_ref().as_str(), "second");
    }
}

#[derive(Clone)]
pub(crate) struct EmbeddingMicroBatcher {
    pub(crate) sender: mpsc::Sender<EmbeddingBatchItem>,
}

pub(crate) struct EmbeddingBatchItem {
    pub(crate) input: Vec<u32>,
    pub(crate) pooling: EmbeddingPooling,
    pub(crate) normalize: bool,
    pub(crate) admission_permit: AdmissionPermit,
    pub(crate) response_tx: oneshot::Sender<Result<Vec<f32>, EngineSessionError>>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct EmbeddingBatchKey {
    pub(crate) pooling_code: u8,
    pub(crate) normalize: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbeddingBatchRequestOptions {
    pub(crate) pooling: EmbeddingPooling,
    pub(crate) normalize: bool,
}

#[derive(Clone)]
pub(crate) struct EmbeddingBatchRunItem {
    pub(crate) input: Vec<u32>,
    pub(crate) pooling: EmbeddingPooling,
    pub(crate) normalize: bool,
}
