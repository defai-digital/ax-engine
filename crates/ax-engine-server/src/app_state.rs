use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use ax_engine_sdk::{
    EmbeddingPooling, EngineSessionConfig, EngineSessionError, EngineStepReport, RuntimeReport,
    StatelessGenerateContext,
};
use tokio::sync::{mpsc, oneshot};

use crate::admission::{AdmissionController, AdmissionPermit};
use crate::generation::service::{
    GenerationPressureEvent, GenerationServiceStartError, NativeGenerationService,
};
use crate::generation::streaming::StreamDeadlines;
use crate::rate_limit::RateLimitConfig;

/// All state owned by one loaded model.
#[derive(Clone)]
pub(crate) struct LiveState {
    pub(crate) generation: u64,
    pub(crate) model_id: Arc<String>,
    pub(crate) session_config: Arc<EngineSessionConfig>,
    pub(crate) stateless_generate_context: Arc<StatelessGenerateContext>,
    pub(crate) runtime_report: RuntimeReport,
    pub(crate) generation_service: Arc<NativeGenerationService>,
    pub(crate) embedding_batcher: Arc<EmbeddingMicroBatcher>,
}

/// Process-local model registry. One model remains the default for requests
/// that omit `model`; callers that provide a model id are routed to the
/// matching independently-owned generation service.
pub(crate) struct LiveModelRegistry {
    default_model_id: String,
    models: BTreeMap<String, LiveState>,
}

/// Metadata published on `GET /v1/discovery` and mDNS TXT (no secrets).
#[derive(Clone, Debug, Default)]
pub(crate) struct DiscoveryMeta {
    pub(crate) instance_id: String,
    pub(crate) cluster: Option<String>,
    pub(crate) version: String,
}

#[derive(Clone)]
pub(crate) struct AppState {
    /// Loaded model states — take a snapshot at the start of each handler.
    pub(crate) live: Arc<parking_lot::RwLock<LiveModelRegistry>>,
    pub(crate) api_key: Option<Arc<String>>,
    pub(crate) metrics: Arc<ServerMetrics>,
    /// Set to true while a model load is in progress; prevents concurrent loads.
    pub(crate) loading: Arc<AtomicBool>,
    pub(crate) limits: Arc<ServerLimits>,
    pub(crate) admission: Arc<AdmissionController>,
    pub(crate) discovery: Arc<DiscoveryMeta>,
    next_live_generation: Arc<AtomicU64>,
    next_request_id: Arc<AtomicU64>,
}

impl AppState {
    pub(crate) fn new(mut live: LiveState) -> Self {
        live.generation = 1;
        let metrics = Arc::new(ServerMetrics::default());
        attach_generation_metrics(&live, &metrics);
        let default_model_id = live.model_id.as_ref().clone();
        let mut models = BTreeMap::new();
        models.insert(default_model_id.clone(), live);
        Self {
            live: Arc::new(parking_lot::RwLock::new(LiveModelRegistry {
                default_model_id,
                models,
            })),
            api_key: None,
            metrics,
            loading: Arc::new(AtomicBool::new(false)),
            limits: Arc::new(ServerLimits::default()),
            admission: Arc::new(AdmissionController::new(None)),
            discovery: Arc::new(DiscoveryMeta::default()),
            next_live_generation: Arc::new(AtomicU64::new(2)),
            next_request_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Clone all live-model fields atomically. The read lock is held only for
    /// the duration of the Arc clones — never across an await point.
    pub(crate) fn snapshot(&self) -> LiveState {
        let registry = self.live.read();
        registry.models[&registry.default_model_id].clone()
    }

    /// Resolve an explicitly requested model, or the default model when the
    /// request omits `model`.
    pub(crate) fn snapshot_for_model(&self, model_id: Option<&str>) -> Option<LiveState> {
        let registry = self.live.read();
        let model_id = model_id.unwrap_or(&registry.default_model_id);
        registry.models.get(model_id).cloned()
    }

    /// Snapshot every loaded model in stable id order.
    pub(crate) fn snapshots(&self) -> Vec<LiveState> {
        self.live.read().models.values().cloned().collect()
    }

    pub(crate) fn model_ids(&self) -> Vec<String> {
        self.live.read().models.keys().cloned().collect()
    }

    /// Remove a loaded model. The last model cannot be removed because every
    /// request that omits `model` must continue to resolve deterministically.
    pub(crate) fn remove_live(&self, model_id: &str) -> Result<LiveState, &'static str> {
        let mut registry = self.live.write();
        if !registry.models.contains_key(model_id) {
            return Err("model_not_found");
        }
        if registry.models.len() == 1 {
            return Err("last_model");
        }
        let next_default = if registry.default_model_id == model_id {
            registry
                .models
                .keys()
                .find(|key| key.as_str() != model_id)
                .cloned()
                .ok_or("last_model")?
        } else {
            registry.default_model_id.clone()
        };
        let removed = registry.models.remove(model_id).ok_or("model_not_found")?;
        registry.default_model_id = next_default;
        Ok(removed)
    }

    /// Add or replace a named model while retaining every other loaded model.
    /// The published model becomes the default when `make_default` is true.
    pub(crate) fn publish_live(&self, mut new: LiveState, make_default: bool) -> Option<LiveState> {
        new.generation = self.next_live_generation.fetch_add(1, Ordering::AcqRel);
        attach_generation_metrics(&new, &self.metrics);
        let model_id = new.model_id.as_ref().clone();
        let mut registry = self.live.write();
        let previous = registry.models.insert(model_id.clone(), new);
        if make_default {
            registry.default_model_id = model_id;
        }
        previous
    }

    /// Replace the live model state. Called by the load endpoint after
    /// successfully building a new session outside the lock.
    pub(crate) fn swap_live(&self, mut new: LiveState) -> LiveState {
        new.generation = self.next_live_generation.fetch_add(1, Ordering::AcqRel);
        attach_generation_metrics(&new, &self.metrics);
        let new_model_id = new.model_id.as_ref().clone();
        let mut registry = self.live.write();
        let old_model_id = registry.default_model_id.clone();
        let previous = registry.models[&old_model_id].clone();
        registry.models.remove(&old_model_id);
        registry.models.insert(new_model_id.clone(), new);
        registry.default_model_id = new_model_id;
        previous
    }

    pub(crate) fn try_admit(
        &self,
        live: &LiveState,
    ) -> Result<AdmissionPermit, crate::admission::AdmissionError> {
        let permit = self.admission.try_admit()?;
        let current_generation = self
            .live
            .read()
            .models
            .get(live.model_id.as_ref())
            .map(|current| current.generation);
        if current_generation != Some(live.generation) {
            drop(permit);
            return Err(crate::admission::AdmissionError::StaleGeneration);
        }
        Ok(permit)
    }

    pub(crate) fn allocate_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::AcqRel)
    }

    pub(crate) fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key.map(Arc::new);
        self
    }

    pub(crate) fn with_discovery(mut self, discovery: DiscoveryMeta) -> Self {
        self.discovery = Arc::new(discovery);
        self
    }

    pub(crate) fn with_limits(mut self, limits: ServerLimits) -> Self {
        self.admission = Arc::new(AdmissionController::new(limits.max_concurrent_requests));
        self.limits = Arc::new(limits);
        self
    }
}

impl LiveState {
    pub(crate) async fn retire(
        self,
    ) -> Result<(), crate::generation::service::GenerationServiceError> {
        let generation_service = Arc::clone(&self.generation_service);
        drop(self);
        generation_service.shutdown().await
    }
}

fn attach_generation_metrics(live: &LiveState, metrics: &Arc<ServerMetrics>) {
    let step_metrics = Arc::downgrade(metrics);
    live.generation_service.set_step_observer(move |report| {
        if let Some(metrics) = step_metrics.upgrade() {
            metrics.record_step_report(report);
        }
    });
    let pressure_metrics = Arc::downgrade(metrics);
    live.generation_service.set_pressure_observer(move |event| {
        if let Some(metrics) = pressure_metrics.upgrade() {
            metrics.record_generation_pressure(event);
        }
    });
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
    pub(crate) generation_saturated_commands_total: AtomicU64,
    pub(crate) generation_stream_backlog_overflows_total: AtomicU64,
    engine_step_observed: AtomicBool,
    engine_steps_total: AtomicU64,
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
    pub(crate) steps_total: u64,
    pub(crate) scheduled_requests: u64,
    pub(crate) scheduled_tokens: u64,
    pub(crate) kv_usage_blocks: u64,
    pub(crate) prefix_hits_total: u64,
}

impl ServerMetrics {
    pub(crate) fn record_generation_pressure(&self, event: GenerationPressureEvent) {
        match event {
            GenerationPressureEvent::CommandSaturated => {
                self.generation_saturated_commands_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            GenerationPressureEvent::StreamBacklogOverflow => {
                self.generation_stream_backlog_overflows_total
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

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
        self.engine_steps_total.fetch_add(1, Ordering::Relaxed);
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
            steps_total: self.engine_steps_total.load(Ordering::Relaxed),
            scheduled_requests: self.engine_step_scheduled_requests.load(Ordering::Relaxed),
            scheduled_tokens: self.engine_step_scheduled_tokens.load(Ordering::Relaxed),
            kv_usage_blocks: self.engine_step_kv_usage_blocks.load(Ordering::Relaxed),
            prefix_hits_total: self.engine_step_prefix_hits_total.load(Ordering::Relaxed),
        })
    }
}

/// Build an initial `LiveState`. The generation service constructs the session
/// on its owning worker thread before this function returns.
pub(crate) fn build_live_state(
    model_id: String,
    session_config: EngineSessionConfig,
) -> Result<LiveState, GenerationServiceStartError> {
    build_live_state_inner(model_id, session_config, false)
}

/// Build a replacement `LiveState` after the current generation has drained.
/// Process-global compiled closures and the MLX allocator cache are cleared on
/// the replacement worker before it constructs the new session.
pub(crate) fn build_replacement_live_state(
    model_id: String,
    session_config: EngineSessionConfig,
) -> Result<LiveState, GenerationServiceStartError> {
    build_live_state_inner(model_id, session_config, true)
}

fn build_live_state_inner(
    model_id: String,
    session_config: EngineSessionConfig,
    replacement: bool,
) -> Result<LiveState, GenerationServiceStartError> {
    let stateless_generate_context =
        StatelessGenerateContext::new(session_config.clone()).map(Arc::new)?;
    let (generation_service, runtime_report) = if replacement {
        NativeGenerationService::spawn_replacement(session_config.clone())?
    } else {
        NativeGenerationService::spawn(session_config.clone())?
    };
    let embedding_batcher = EmbeddingMicroBatcher::spawn(generation_service.clone());
    Ok(LiveState {
        generation: 0,
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
    session_config: EngineSessionConfig,
) -> Result<AppState, GenerationServiceStartError> {
    let live = build_live_state(model_id, session_config)?;
    Ok(AppState::new(live))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::mpsc as std_mpsc;
    use std::time::{Duration, Instant};

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
        build_app_state(model_id.to_string(), config).expect("app state should build")
    }

    fn trigger_command_saturation(service: &Arc<NativeGenerationService>) {
        let (entered_tx, entered_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        service
            .submit(move |_| {
                let _ = entered_tx.send(());
                let _ = release_rx.recv();
            })
            .expect("blocking command should enqueue");
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker should enter blocking command");
        for _ in 0..service.command_queue_capacity() {
            service
                .submit(|_| {})
                .expect("queue should accept commands up to capacity");
        }
        assert!(matches!(
            service.submit(|_| {}),
            Err(crate::generation::service::GenerationServiceError::Saturated)
        ));
        release_tx.send(()).expect("worker should be released");
        let deadline = Instant::now() + Duration::from_secs(1);
        while service.is_busy() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!service.is_busy());
    }

    #[tokio::test]
    async fn snapshot_reads_live_state() {
        let state = test_state("first");
        let live = state.snapshot();
        assert_eq!(live.model_id.as_ref().as_str(), "first");
        assert_eq!(live.generation, 1);
    }

    #[tokio::test]
    async fn swap_live_replaces_state() {
        let state = test_state("first");
        let previous_service = state.snapshot().generation_service;
        let replacement = test_state("second").snapshot();
        let previous = state.swap_live(replacement);
        previous
            .retire()
            .await
            .expect("previous generation should retire");
        let live = state.snapshot();
        assert_eq!(live.model_id.as_ref().as_str(), "second");
        assert_eq!(live.generation, 2);
        assert!(!previous_service.is_ready());
    }

    #[tokio::test]
    async fn registry_routes_explicit_models_and_unloads_safely() {
        let state = test_state("first");
        let first = state.snapshot();
        let second = build_live_state("second".to_string(), first.session_config.as_ref().clone())
            .expect("second model state should build");

        assert!(state.publish_live(second, true).is_none());
        assert_eq!(state.snapshot().model_id.as_ref(), "second");
        assert_eq!(
            state
                .snapshot_for_model(Some("first"))
                .expect("first model remains loaded")
                .model_id
                .as_ref(),
            "first"
        );
        assert_eq!(state.model_ids(), vec!["first", "second"]);

        let removed = state.remove_live("second").expect("second model unloads");
        assert_eq!(state.snapshot().model_id.as_ref(), "first");
        assert!(matches!(state.remove_live("first"), Err("last_model")));
        removed
            .retire()
            .await
            .expect("removed generation worker should retire");
    }

    #[tokio::test]
    async fn stale_live_snapshot_cannot_admit_after_swap() {
        let state = test_state("first");
        let stale = state.snapshot();
        let replacement = test_state("second").snapshot();
        state.swap_live(replacement);

        assert!(matches!(
            state.try_admit(&stale),
            Err(crate::admission::AdmissionError::StaleGeneration)
        ));
        assert_eq!(state.admission.active_jobs(), 0);

        let current = state.snapshot();
        let permit = state
            .try_admit(&current)
            .expect("current generation should be admitted");
        assert_eq!(state.admission.active_jobs(), 1);
        drop(permit);
    }

    #[tokio::test]
    async fn swapped_generation_records_steps_in_shared_metrics() {
        let state = test_state("first");
        let replacement = test_state("second").snapshot();
        state.swap_live(replacement);

        state
            .snapshot()
            .generation_service
            .advance()
            .await
            .expect("delegated idle step should succeed");

        assert!(state.metrics.engine_step_gauges().is_some());
    }

    #[tokio::test]
    async fn pressure_counters_remain_monotonic_across_model_swap() {
        let state = test_state("first");
        trigger_command_saturation(&state.snapshot().generation_service);
        assert_eq!(
            state
                .metrics
                .generation_saturated_commands_total
                .load(Ordering::Relaxed),
            1
        );

        let replacement = test_state("second").snapshot();
        state.swap_live(replacement);
        trigger_command_saturation(&state.snapshot().generation_service);

        assert_eq!(
            state
                .metrics
                .generation_saturated_commands_total
                .load(Ordering::Relaxed),
            2
        );
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
