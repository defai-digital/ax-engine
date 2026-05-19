use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ax_engine_sdk::{
    EmbeddingPooling, EngineSession, EngineSessionConfig, EngineSessionError, MlxPrefixCacheStore,
    RuntimeReport, StatelessGenerateContext,
};
use tokio::sync::{Mutex, mpsc, oneshot};

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) model_id: Arc<String>,
    pub(crate) session_config: Arc<EngineSessionConfig>,
    pub(crate) stateless_generate_context: Arc<StatelessGenerateContext>,
    pub(crate) runtime_report: RuntimeReport,
    pub(crate) request_session: Arc<Mutex<EngineSession>>,
    pub(crate) mlx_prefix_cache: Option<MlxPrefixCacheStore>,
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
        mlx_prefix_cache: Option<MlxPrefixCacheStore>,
        embedding_batcher: Arc<EmbeddingMicroBatcher>,
    ) -> Self {
        Self {
            model_id: Arc::new(model_id),
            session_config: Arc::new(session_config),
            stateless_generate_context,
            runtime_report,
            request_session,
            mlx_prefix_cache,
            embedding_batcher,
            next_request_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub(crate) fn allocate_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::AcqRel)
    }

    pub(crate) fn uses_native_mlx_backend(&self) -> bool {
        self.session_config
            .resolved_backend
            .selected_backend
            .is_mlx()
    }

    pub(crate) fn request_session_parts(
        &self,
    ) -> (EngineSessionConfig, Option<MlxPrefixCacheStore>) {
        (
            self.session_config.as_ref().clone(),
            self.mlx_prefix_cache.clone(),
        )
    }

    pub(crate) fn build_request_session_from_parts(
        session_config: EngineSessionConfig,
        mlx_prefix_cache: Option<MlxPrefixCacheStore>,
    ) -> Result<EngineSession, EngineSessionError> {
        match mlx_prefix_cache {
            Some(prefix_cache) => {
                EngineSession::new_with_shared_mlx_prefix_cache(session_config, prefix_cache)
            }
            None => EngineSession::new(session_config),
        }
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
    let mlx_prefix_cache = session_config
        .resolved_backend
        .selected_backend
        .is_mlx()
        .then(MlxPrefixCacheStore::from_env);
    let request_session = Arc::new(Mutex::new(session));
    let embedding_batcher = EmbeddingMicroBatcher::spawn(request_session.clone());

    Ok(AppState::new(
        model_id,
        session_config,
        stateless_generate_context,
        runtime_report,
        request_session,
        mlx_prefix_cache,
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
