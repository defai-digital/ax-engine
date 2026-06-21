use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ax_engine_sdk::{
    EmbeddingPooling, EngineSession, EngineSessionConfig, EngineSessionError, EngineStepReport,
    RuntimeReport, StatelessGenerateContext,
};
use tokio::sync::{Mutex, mpsc, oneshot};

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
    pub(crate) request_session: Arc<Mutex<EngineSession>>,
    pub(crate) embedding_batcher: Arc<EmbeddingMicroBatcher>,
}

#[derive(Clone)]
pub(crate) struct AppState {
    /// Swappable model state — take a snapshot at the start of each handler.
    pub(crate) live: Arc<std::sync::RwLock<LiveState>>,
    pub(crate) api_key: Option<Arc<String>>,
    pub(crate) metrics: Arc<ServerMetrics>,
    /// Set to true while a model load is in progress; prevents concurrent loads.
    pub(crate) loading: Arc<AtomicBool>,
    next_request_id: Arc<AtomicU64>,
}

impl AppState {
    pub(crate) fn new(live: LiveState) -> Self {
        Self {
            live: Arc::new(std::sync::RwLock::new(live)),
            api_key: None,
            metrics: Arc::new(ServerMetrics::default()),
            loading: Arc::new(AtomicBool::new(false)),
            next_request_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Clone all live-model fields atomically. The read lock is held only for
    /// the duration of the Arc clones — never across an await point.
    pub(crate) fn snapshot(&self) -> LiveState {
        match self.live.read() {
            Ok(live) => live.clone(),
            Err(poisoned) => {
                tracing::error!("live model state lock was poisoned; recovering snapshot");
                poisoned.into_inner().clone()
            }
        }
    }

    /// Replace the live model state. Called by the load endpoint after
    /// successfully building a new session outside the lock.
    pub(crate) fn swap_live(&self, new: LiveState) {
        match self.live.write() {
            Ok(mut live) => *live = new,
            Err(poisoned) => {
                tracing::error!("live model state lock was poisoned; recovering model swap");
                *poisoned.into_inner() = new;
            }
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
    let request_session = Arc::new(Mutex::new(session));
    let embedding_batcher = EmbeddingMicroBatcher::spawn(request_session.clone());
    Ok(LiveState {
        model_id: Arc::new(model_id),
        session_config: Arc::new(session_config),
        stateless_generate_context,
        runtime_report,
        request_session,
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
    use std::panic::{self, AssertUnwindSafe};
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
    async fn snapshot_recovers_poisoned_live_lock() {
        let state = test_state("first");
        let result = panic::catch_unwind(AssertUnwindSafe({
            let state = state.clone();
            move || {
                let _live = state.live.write().expect("live lock should be available");
                panic!("poison live lock");
            }
        }));
        assert!(result.is_err());

        let live = state.snapshot();
        assert_eq!(live.model_id.as_ref().as_str(), "first");
    }

    #[tokio::test]
    async fn swap_live_recovers_poisoned_live_lock() {
        let state = test_state("first");
        let replacement = test_state("second").snapshot();
        let result = panic::catch_unwind(AssertUnwindSafe({
            let state = state.clone();
            move || {
                let _live = state.live.write().expect("live lock should be available");
                panic!("poison live lock");
            }
        }));
        assert!(result.is_err());

        state.swap_live(replacement);
        let live = state.snapshot();
        assert_eq!(live.model_id.as_ref().as_str(), "second");
    }
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
