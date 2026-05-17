use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ax_engine_sdk::{
    EmbeddingPooling, EngineSession, EngineSessionConfig, EngineSessionError, RuntimeReport,
    StatelessGenerateContext,
};
use tokio::sync::{Mutex, mpsc, oneshot};

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) model_id: Arc<String>,
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
