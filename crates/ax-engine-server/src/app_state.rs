use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ax_engine_sdk::{
    EmbeddingPooling, EngineSession, EngineSessionConfig, EngineSessionError, EngineStepReport,
    RuntimeReport, StatelessGenerateContext,
};
use tokio::sync::{Mutex, mpsc, oneshot};

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) model_id: Arc<String>,
    pub(crate) api_key: Option<Arc<String>>,
    pub(crate) metrics: Arc<ServerMetrics>,
    pub(crate) session_config: Arc<EngineSessionConfig>,
    pub(crate) stateless_generate_context: Arc<StatelessGenerateContext>,
    pub(crate) runtime_report: RuntimeReport,
    pub(crate) request_session: Arc<Mutex<EngineSession>>,
    pub(crate) embedding_batcher: Arc<EmbeddingMicroBatcher>,
    next_request_id: Arc<AtomicU64>,
}

impl AppState {
    pub(crate) fn new(
        model_id: String,
        session_config: EngineSessionConfig,
        stateless_generate_context: Arc<StatelessGenerateContext>,
        runtime_report: RuntimeReport,
        request_session: Arc<Mutex<EngineSession>>,
        embedding_batcher: Arc<EmbeddingMicroBatcher>,
    ) -> Self {
        Self {
            model_id: Arc::new(model_id),
            api_key: None,
            metrics: Arc::new(ServerMetrics::default()),
            session_config: Arc::new(session_config),
            stateless_generate_context,
            runtime_report,
            request_session,
            embedding_batcher,
            next_request_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub(crate) fn allocate_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::AcqRel)
    }

    pub(crate) fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key.map(Arc::new);
        self
    }
}

#[derive(Default)]
pub(crate) struct ServerMetrics {
    pub(crate) http_requests_total: AtomicU64,
    pub(crate) http_requests_in_flight: AtomicU64,
    pub(crate) http_status_2xx_total: AtomicU64,
    pub(crate) http_status_4xx_total: AtomicU64,
    pub(crate) http_status_5xx_total: AtomicU64,
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

pub(crate) fn build_app_state(
    model_id: String,
    session: EngineSession,
) -> Result<AppState, EngineSessionError> {
    let session_config = session.config().clone();
    let stateless_generate_context =
        StatelessGenerateContext::new(session_config.clone()).map(Arc::new)?;
    let runtime_report = session.runtime_report();
    let request_session = Arc::new(Mutex::new(session));
    let embedding_batcher = EmbeddingMicroBatcher::spawn(request_session.clone());

    Ok(AppState::new(
        model_id,
        session_config,
        stateless_generate_context,
        runtime_report,
        request_session,
        embedding_batcher,
    ))
}

#[derive(Clone)]
pub(crate) struct EmbeddingMicroBatcher {
    pub(crate) sender: mpsc::UnboundedSender<EmbeddingBatchItem>,
}

pub(crate) struct EmbeddingBatchItem {
    pub(crate) input: Vec<u32>,
    pub(crate) pooling: EmbeddingPooling,
    pub(crate) normalize: bool,
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
