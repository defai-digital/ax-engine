use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ax_engine_sdk::{
    EmbeddingPooling, EngineSessionConfig, EngineSessionError, EngineStepReport, RuntimeReport,
    StatelessGenerateContext,
};
use tokio::sync::{mpsc, oneshot};

use crate::admission::{AdmissionController, AdmissionPermit};
use crate::generation::service::{
    ExecutionWorkClass, GenerationPressureEvent, GenerationServiceStartError,
    ModelExecutionArbiter, ModelExecutionStats, NativeGenerationService,
};
use crate::generation::streaming::StreamDeadlines;
use crate::lan_advertise::ModelAdvertisement;
use crate::rate_limit::RateLimitConfig;

/// All state owned by one loaded model.
#[derive(Clone)]
pub(crate) struct LiveState {
    pub(crate) generation: u64,
    pub(crate) model_id: Arc<String>,
    /// Admission and drain scope owned by this model generation.
    ///
    /// The process-wide controller remains the hard capacity bound. This
    /// model-local controller lets lifecycle operations stop and drain only
    /// the generation they retire while sibling models continue serving.
    pub(crate) admission: Arc<AdmissionController>,
    pub(crate) session_config: Arc<EngineSessionConfig>,
    pub(crate) stateless_generate_context: Arc<StatelessGenerateContext>,
    pub(crate) runtime_report: RuntimeReport,
    pub(crate) generation_service: Arc<NativeGenerationService>,
    pub(crate) embedding_batcher: Arc<EmbeddingMicroBatcher>,
    /// Unix seconds of the last admitted request for this model; drives the
    /// optional idle-eviction of non-default resident models.
    pub(crate) last_used: Arc<AtomicU64>,
}

/// Process-local model registry. One model remains the default for requests
/// that omit `model`; callers that provide a model id are routed to the
/// matching independently-owned generation service.
pub(crate) struct LiveModelRegistry {
    default_model_id: String,
    models: BTreeMap<String, LiveState>,
}

#[derive(Clone)]
struct RequestOwner {
    model_id: Arc<String>,
    generation: u64,
    terminal: bool,
}

#[derive(Default)]
struct RequestOwners {
    by_id: BTreeMap<u64, RequestOwner>,
    terminal_order: BTreeMap<(String, u64), VecDeque<u64>>,
}

const MAX_TERMINAL_REQUEST_OWNERS_PER_GENERATION: usize = 4096;

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
    execution_arbiter: Arc<ModelExecutionArbiter>,
    model_advertisement: Arc<parking_lot::RwLock<Option<Arc<dyn ModelAdvertisement>>>>,
    request_owners: Arc<parking_lot::RwLock<RequestOwners>>,
    next_live_generation: Arc<AtomicU64>,
    next_request_id: Arc<AtomicU64>,
}

impl AppState {
    pub(crate) fn new(mut live: LiveState) -> Self {
        live.generation = 1;
        let metrics = Arc::new(ServerMetrics::default());
        let execution_arbiter = Arc::new(ModelExecutionArbiter::default());
        let request_owners = Arc::new(parking_lot::RwLock::new(RequestOwners::default()));
        attach_live_state(&live, &metrics, &execution_arbiter, &request_owners);
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
            execution_arbiter,
            model_advertisement: Arc::new(parking_lot::RwLock::new(None)),
            request_owners,
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

    /// Mirror a runtime fair-prefill policy change into the published
    /// generation metadata. The generation check prevents a delayed control
    /// command from rewriting a replacement model's configuration.
    pub(crate) fn record_multi_prefill_policy(
        &self,
        model_id: &str,
        generation: u64,
        max_tokens_per_request_per_step: u32,
        max_inflight_prefill_requests: u32,
    ) -> bool {
        let mut registry = self.live.write();
        let Some(live) = registry.models.get_mut(model_id) else {
            return false;
        };
        if live.generation != generation {
            return false;
        }
        let config = live
            .session_config
            .as_ref()
            .clone()
            .with_multi_prefill_fair(
                max_tokens_per_request_per_step,
                max_inflight_prefill_requests,
            );
        live.session_config = Arc::new(config);
        true
    }

    pub(crate) fn unavailable_model_ids(&self) -> Vec<String> {
        self.live
            .read()
            .models
            .iter()
            .filter(|(_, live)| !live.generation_service.is_ready())
            .map(|(model_id, _)| model_id.clone())
            .collect()
    }

    pub(crate) fn execution_arbiter_stats(
        &self,
    ) -> Vec<(String, ExecutionWorkClass, ModelExecutionStats)> {
        self.execution_arbiter.stats()
    }

    /// Remove a loaded model. The last model cannot be removed because every
    /// request that omits `model` must continue to resolve deterministically.
    pub(crate) fn remove_live(&self, model_id: &str) -> Result<LiveState, &'static str> {
        let (removed, next_default) = {
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
            registry.default_model_id = next_default.clone();
            (removed, next_default)
        };
        self.remove_model_tracking(&removed);
        self.update_advertised_model(&next_default);
        Ok(removed)
    }

    /// Add or replace a named model while retaining every other loaded model.
    /// The published model becomes the default when `make_default` is true.
    pub(crate) fn publish_live(&self, mut new: LiveState, make_default: bool) -> Option<LiveState> {
        new.generation = self.next_live_generation.fetch_add(1, Ordering::AcqRel);
        new.admission = Arc::new(AdmissionController::new(
            self.limits.max_concurrent_requests_per_model,
        ));
        attach_live_state(
            &new,
            &self.metrics,
            &self.execution_arbiter,
            &self.request_owners,
        );
        let model_id = new.model_id.as_ref().clone();
        let mut registry = self.live.write();
        let previous = registry.models.insert(model_id.clone(), new);
        if make_default {
            registry.default_model_id = model_id.clone();
        }
        drop(registry);
        if let Some(previous) = previous.as_ref() {
            self.remove_model_tracking(previous);
        }
        if make_default {
            self.update_advertised_model(&model_id);
        }
        previous
    }

    /// Replace the live model state. Called by the load endpoint after
    /// successfully building a new session outside the lock.
    pub(crate) fn swap_live(&self, mut new: LiveState) -> LiveState {
        new.generation = self.next_live_generation.fetch_add(1, Ordering::AcqRel);
        new.admission = Arc::new(AdmissionController::new(
            self.limits.max_concurrent_requests_per_model,
        ));
        attach_live_state(
            &new,
            &self.metrics,
            &self.execution_arbiter,
            &self.request_owners,
        );
        let new_model_id = new.model_id.as_ref().clone();
        let mut registry = self.live.write();
        let old_model_id = registry.default_model_id.clone();
        let previous = registry.models[&old_model_id].clone();
        registry.models.remove(&old_model_id);
        registry.models.insert(new_model_id.clone(), new);
        registry.default_model_id = new_model_id.clone();
        drop(registry);
        self.remove_model_tracking(&previous);
        self.update_advertised_model(&new_model_id);
        previous
    }

    pub(crate) fn try_admit(
        &self,
        live: &LiveState,
    ) -> Result<AdmissionPermit, crate::admission::AdmissionError> {
        // Acquire the narrow scope first so a draining/saturated model does
        // not transiently consume process capacity. Both leases are merged
        // into one permit and therefore follow the actual engine-job lifetime.
        let model_permit = live.admission.try_admit()?;
        let global_permit = self.admission.try_admit()?;
        let current_generation = self
            .live
            .read()
            .models
            .get(live.model_id.as_ref())
            .map(|current| current.generation);
        if current_generation != Some(live.generation) {
            drop((model_permit, global_permit));
            return Err(crate::admission::AdmissionError::StaleGeneration);
        }
        live.last_used.store(unix_now_secs(), Ordering::Relaxed);
        Ok(model_permit.merge(global_permit))
    }

    pub(crate) fn allocate_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::AcqRel)
    }

    pub(crate) fn register_request_owner(&self, request_id: u64, live: &LiveState) {
        let previous = self.request_owners.write().by_id.insert(
            request_id,
            RequestOwner {
                model_id: Arc::clone(&live.model_id),
                generation: live.generation,
                terminal: false,
            },
        );
        debug_assert!(previous.is_none(), "request IDs are process-unique");
    }

    pub(crate) fn remove_request_owner(&self, request_id: u64) {
        self.request_owners.write().by_id.remove(&request_id);
    }

    pub(crate) fn snapshot_for_request(&self, request_id: u64) -> Option<LiveState> {
        let owner = self.request_owners.read().by_id.get(&request_id).cloned()?;
        self.live
            .read()
            .models
            .get(owner.model_id.as_ref())
            .filter(|live| live.generation == owner.generation)
            .cloned()
    }

    #[cfg(test)]
    pub(crate) fn request_owner_is_terminal(&self, request_id: u64) -> bool {
        self.request_owners
            .read()
            .by_id
            .get(&request_id)
            .is_some_and(|owner| owner.terminal)
    }

    pub(crate) fn set_model_advertisement(&self, advertisement: Arc<dyn ModelAdvertisement>) {
        *self.model_advertisement.write() = Some(advertisement);
        self.update_advertised_model(self.snapshot().model_id.as_ref());
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
        for live in self.live.write().models.values_mut() {
            live.admission = Arc::new(AdmissionController::new(
                limits.max_concurrent_requests_per_model,
            ));
        }
        self.limits = Arc::new(limits);
        self
    }

    fn remove_model_tracking(&self, live: &LiveState) {
        let mut request_owners = self.request_owners.write();
        request_owners.by_id.retain(|_, owner| {
            owner.model_id.as_ref() != live.model_id.as_ref() || owner.generation != live.generation
        });
        request_owners
            .terminal_order
            .remove(&(live.model_id.as_ref().clone(), live.generation));
        drop(request_owners);
        self.metrics.remove_model_step_stats(live.model_id.as_ref());
        self.execution_arbiter.remove_model(live.model_id.as_ref());
    }

    fn update_advertised_model(&self, model_id: &str) {
        let advertisement = self.model_advertisement.read().clone();
        if let Some(advertisement) = advertisement
            && let Err(error) = advertisement.update_model(model_id)
        {
            tracing::warn!(model_id, %error, "failed to update LAN model advertisement");
        }
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

fn attach_live_state(
    live: &LiveState,
    metrics: &Arc<ServerMetrics>,
    execution_arbiter: &Arc<ModelExecutionArbiter>,
    request_owners: &Arc<parking_lot::RwLock<RequestOwners>>,
) {
    live.generation_service
        .set_execution_arbiter(Arc::clone(&live.model_id), Arc::clone(execution_arbiter));
    let request_owners = Arc::downgrade(request_owners);
    live.generation_service
        .set_stepwise_terminal_observer(move |request_id| {
            if let Some(request_owners) = request_owners.upgrade() {
                record_terminal_request_owner(&request_owners, request_id);
            }
        });
    let step_metrics = Arc::downgrade(metrics);
    let step_model_id = live.model_id.clone();
    live.generation_service.set_step_observer(move |report| {
        if let Some(metrics) = step_metrics.upgrade() {
            metrics.record_step_report(step_model_id.as_ref(), report);
        }
    });
    let pressure_metrics = Arc::downgrade(metrics);
    live.generation_service.set_pressure_observer(move |event| {
        if let Some(metrics) = pressure_metrics.upgrade() {
            metrics.record_generation_pressure(event);
        }
    });
}

fn record_terminal_request_owner(
    request_owners: &parking_lot::RwLock<RequestOwners>,
    request_id: u64,
) {
    let mut request_owners = request_owners.write();
    let Some(owner) = request_owners.by_id.get_mut(&request_id) else {
        return;
    };
    if owner.terminal {
        return;
    }
    owner.terminal = true;
    let key = (owner.model_id.as_ref().clone(), owner.generation);
    let order = request_owners
        .terminal_order
        .entry(key.clone())
        .or_default();
    order.push_back(request_id);
    let mut evicted = Vec::new();
    while order.len() > MAX_TERMINAL_REQUEST_OWNERS_PER_GENERATION {
        if let Some(evicted_request_id) = order.pop_front() {
            evicted.push(evicted_request_id);
        }
    }
    for evicted_request_id in evicted {
        let belongs_to_generation =
            request_owners
                .by_id
                .get(&evicted_request_id)
                .is_some_and(|owner| {
                    owner.terminal && owner.model_id.as_ref() == &key.0 && owner.generation == key.1
                });
        if belongs_to_generation {
            request_owners.by_id.remove(&evicted_request_id);
        }
    }
}

/// Resource limits resolved from CLI flags / env vars at startup (see
/// `ServerArgs::resolved_*` in `args.rs`). All fields default to "disabled"
/// except `max_request_body_bytes`, which always enforces the built-in
/// safe default — this preserves today's behavior exactly when no operator
/// configuration is supplied.
pub(crate) struct ServerLimits {
    pub(crate) max_concurrent_requests: Option<usize>,
    pub(crate) max_concurrent_requests_per_model: Option<usize>,
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
            max_concurrent_requests_per_model: None,
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
    /// Engine-step stats keyed by model id. Multi-model serving runs one
    /// engine worker per model; a single set of gauges would interleave
    /// last-writer-wins values from unrelated models.
    engine_step_stats: Mutex<BTreeMap<String, EngineStepStats>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct EngineStepStats {
    steps_total: u64,
    scheduled_requests: u64,
    scheduled_tokens: u64,
    kv_usage_blocks: u64,
    prefix_hits_total: u64,
    memory: Option<ModelMemoryGauges>,
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

/// Latest model-attributed memory geometry reported by the native runner.
///
/// These values describe owned tensor storage, not process RSS. The metrics
/// endpoint keeps that distinction explicit by publishing MLX allocator and
/// host probes separately.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ModelMemoryGauges {
    pub(crate) kv_logical_bytes: u64,
    pub(crate) kv_capacity_bytes: u64,
    pub(crate) kv_linear_state_bytes: u64,
    pub(crate) kv_paged_pool_slab_bytes: u64,
    pub(crate) prefix_cache_payload_bytes: u64,
    pub(crate) full_attention_layers: u64,
    pub(crate) sliding_window_layers: u64,
    pub(crate) rotated_ring_layers: u64,
    pub(crate) linear_state_layers: u64,
    pub(crate) paged_pool_slabs: u64,
}

impl ModelMemoryGauges {
    /// Physical KV bytes attributed to this model without double-counting
    /// logical paged views. A paged pool reports real slab reservations; the
    /// contiguous path reports per-request capacity arrays.
    pub(crate) fn physical_kv_bytes(self) -> u64 {
        let attention_bytes = if self.kv_paged_pool_slab_bytes > 0 {
            self.kv_paged_pool_slab_bytes
        } else {
            self.kv_capacity_bytes
        };
        attention_bytes.saturating_add(self.kv_linear_state_bytes)
    }

    pub(crate) const fn rollback_strategy(self) -> &'static str {
        if self.linear_state_layers > 0 {
            "restore_replay"
        } else if self.rotated_ring_layers > 0 {
            "bounded_cursor_restore"
        } else {
            "o1_trim"
        }
    }

    pub(crate) const fn attention_storage(self) -> &'static str {
        if self.paged_pool_slabs > 0 {
            "paged_pool"
        } else {
            "contiguous"
        }
    }

    pub(crate) const fn sliding_storage(self) -> &'static str {
        if self.rotated_ring_layers > 0 {
            "rotating_ring"
        } else if self.sliding_window_layers > 0 {
            "ordered"
        } else {
            "none"
        }
    }
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

    pub(crate) fn record_step_report(&self, model_id: &str, report: &EngineStepReport) {
        let mut stats = self
            .engine_step_stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !stats.contains_key(model_id) {
            stats.insert(model_id.to_string(), EngineStepStats::default());
        }
        let Some(entry) = stats.get_mut(model_id) else {
            return;
        };
        entry.steps_total = entry.steps_total.saturating_add(1);
        entry.scheduled_requests = report.scheduled_requests as u64;
        entry.scheduled_tokens = report.scheduled_tokens as u64;
        entry.kv_usage_blocks = report.kv_usage_blocks as u64;
        entry.prefix_hits_total = entry
            .prefix_hits_total
            .saturating_add(report.prefix_hits as u64);
        if let Some(route) = report.route.as_ref()
            && route
                .decision("ax_mlx_kv_request_snapshots")
                .is_some_and(|snapshots| snapshots > 0)
        {
            entry.memory = Some(ModelMemoryGauges {
                kv_logical_bytes: route_kib_as_bytes(route, "ax_mlx_kv_logical_kib"),
                kv_capacity_bytes: route_kib_as_bytes(route, "ax_mlx_kv_capacity_kib"),
                kv_linear_state_bytes: route_kib_as_bytes(route, "ax_mlx_kv_linear_state_kib"),
                kv_paged_pool_slab_bytes: route_kib_as_bytes(
                    route,
                    "ax_mlx_kv_paged_pool_slab_kib",
                ),
                prefix_cache_payload_bytes: route_kib_as_bytes(
                    route,
                    "ax_mlx_prefix_cache_bytes_kib",
                ),
                full_attention_layers: u64::from(
                    route
                        .decision("ax_mlx_kv_full_attention_layers")
                        .unwrap_or(0),
                ),
                sliding_window_layers: u64::from(
                    route
                        .decision("ax_mlx_kv_sliding_window_layers")
                        .unwrap_or(0),
                ),
                rotated_ring_layers: u64::from(
                    route.decision("ax_mlx_kv_rotated_ring_layers").unwrap_or(0),
                ),
                linear_state_layers: u64::from(
                    route.decision("ax_mlx_kv_linear_state_layers").unwrap_or(0),
                ),
                paged_pool_slabs: u64::from(
                    route.decision("ax_mlx_kv_paged_pool_slabs").unwrap_or(0),
                ),
            });
        }
    }

    pub(crate) fn remove_model_step_stats(&self, model_id: &str) {
        self.engine_step_stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(model_id);
    }

    /// Per-model engine-step snapshots for `/metrics`, sorted by model id.
    /// Empty until the first step is observed.
    pub(crate) fn engine_step_gauges_per_model(&self) -> Vec<(String, EngineStepGauges)> {
        let stats = self
            .engine_step_stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        stats
            .iter()
            .map(|(model_id, entry)| {
                (
                    model_id.clone(),
                    EngineStepGauges {
                        steps_total: entry.steps_total,
                        scheduled_requests: entry.scheduled_requests,
                        scheduled_tokens: entry.scheduled_tokens,
                        kv_usage_blocks: entry.kv_usage_blocks,
                        prefix_hits_total: entry.prefix_hits_total,
                    },
                )
            })
            .collect()
    }

    pub(crate) fn model_memory_gauges_per_model(&self) -> Vec<(String, ModelMemoryGauges)> {
        self.engine_step_stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter_map(|(model_id, entry)| entry.memory.map(|memory| (model_id.clone(), memory)))
            .collect()
    }
}

fn route_kib_as_bytes(route: &ax_engine_sdk::GenerateRouteReport, key: &str) -> u64 {
    u64::from(route.decision(key).unwrap_or(0)).saturating_mul(1024)
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
    mut session_config: EngineSessionConfig,
    replacement: bool,
) -> Result<LiveState, GenerationServiceStartError> {
    session_config.probe_vllm_readiness()?;
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
        admission: Arc::new(AdmissionController::new(None)),
        session_config: Arc::new(session_config),
        stateless_generate_context,
        runtime_report,
        generation_service,
        embedding_batcher,
        last_used: Arc::new(AtomicU64::new(unix_now_secs())),
    })
}

pub(crate) fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) fn build_app_state(
    model_id: String,
    session_config: EngineSessionConfig,
) -> Result<AppState, GenerationServiceStartError> {
    let live = build_live_state(model_id, session_config)?;
    Ok(AppState::new(live))
}

#[cfg(test)]
mod step_metrics_tests {
    use super::*;

    fn step_report(scheduled_tokens: u32, prefix_hits: u32) -> EngineStepReport {
        EngineStepReport {
            step_id: None,
            scheduled_requests: 1,
            scheduled_tokens,
            ttft_events: 0,
            prefix_hits,
            kv_usage_blocks: 8,
            evictions: 0,
            preempted_requests: 0,
            preempted_tokens: 0,
            cpu_time_us: 0,
            runner_time_us: 0,
            route: None,
            metal_dispatch: None,
        }
    }

    #[test]
    fn step_metrics_are_tracked_per_model_not_last_writer_wins() {
        let metrics = ServerMetrics::default();
        metrics.record_step_report("qwen3.6-27b", &step_report(64, 2));
        metrics.record_step_report("gemma-4-12b", &step_report(128, 0));
        metrics.record_step_report("qwen3.6-27b", &step_report(32, 1));

        let per_model = metrics.engine_step_gauges_per_model();
        assert_eq!(per_model.len(), 2);
        let gemma = &per_model
            .iter()
            .find(|(model, _)| model == "gemma-4-12b")
            .expect("gemma entry")
            .1;
        assert_eq!(gemma.steps_total, 1);
        assert_eq!(gemma.scheduled_tokens, 128);
        assert_eq!(gemma.prefix_hits_total, 0);
        let qwen = &per_model
            .iter()
            .find(|(model, _)| model == "qwen3.6-27b")
            .expect("qwen entry")
            .1;
        assert_eq!(qwen.steps_total, 2);
        // Gauges hold the latest step; counters accumulate.
        assert_eq!(qwen.scheduled_tokens, 32);
        assert_eq!(qwen.prefix_hits_total, 3);
    }

    #[test]
    fn step_metrics_are_empty_before_first_step() {
        let metrics = ServerMetrics::default();
        assert!(metrics.engine_step_gauges_per_model().is_empty());
    }

    #[test]
    fn native_memory_report_separates_paged_pool_from_logical_views() {
        let metrics = ServerMetrics::default();
        let mut report = step_report(2, 0);
        report.route = Some(ax_engine_sdk::GenerateRouteReport {
            crossover_decisions: BTreeMap::from([
                ("ax_mlx_kv_request_snapshots".to_string(), 2),
                ("ax_mlx_kv_logical_kib".to_string(), 600),
                ("ax_mlx_kv_capacity_kib".to_string(), 800),
                ("ax_mlx_kv_linear_state_kib".to_string(), 25),
                ("ax_mlx_kv_paged_pool_slab_kib".to_string(), 500),
                ("ax_mlx_prefix_cache_bytes_kib".to_string(), 40),
                ("ax_mlx_kv_full_attention_layers".to_string(), 48),
                ("ax_mlx_kv_sliding_window_layers".to_string(), 40),
                ("ax_mlx_kv_rotated_ring_layers".to_string(), 40),
                ("ax_mlx_kv_linear_state_layers".to_string(), 24),
                ("ax_mlx_kv_paged_pool_slabs".to_string(), 4),
            ]),
            ..Default::default()
        });

        metrics.record_step_report("hybrid", &report);
        let memory = metrics.model_memory_gauges_per_model()[0].1;

        assert_eq!(memory.kv_logical_bytes, 600 * 1024);
        assert_eq!(memory.kv_capacity_bytes, 800 * 1024);
        assert_eq!(
            memory.physical_kv_bytes(),
            (500 + 25) * 1024,
            "paged slab bytes replace overlapping per-view capacity",
        );
        assert_eq!(memory.rollback_strategy(), "restore_replay");
        assert_eq!(memory.attention_storage(), "paged_pool");
        assert_eq!(memory.sliding_storage(), "rotating_ring");
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::mpsc as std_mpsc;
    use std::time::{Duration, Instant};

    use ax_engine_sdk::{PreviewBackendRequest, PreviewSessionConfigRequest, SupportTier};

    use super::*;

    #[derive(Default)]
    struct RecordedAdvertisement(parking_lot::Mutex<Vec<String>>);

    impl ModelAdvertisement for RecordedAdvertisement {
        fn update_model(&self, model_id: &str) -> Result<(), String> {
            self.0.lock().push(model_id.to_string());
            Ok(())
        }
    }

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
    async fn request_owners_route_directly_and_are_pruned_on_unload() {
        let state = test_state("first");
        let config = state.snapshot().session_config.as_ref().clone();
        let second = build_live_state("second".to_string(), config)
            .expect("second model state should build");
        state.publish_live(second, false);
        let second = state
            .snapshot_for_model(Some("second"))
            .expect("second model should resolve");
        state.register_request_owner(41, &second);

        assert_eq!(
            state
                .snapshot_for_request(41)
                .expect("request owner should resolve")
                .model_id
                .as_ref(),
            "second"
        );
        let removed = state.remove_live("second").expect("second model unloads");
        assert!(state.snapshot_for_request(41).is_none());
        removed
            .retire()
            .await
            .expect("removed generation worker should retire");
    }

    #[tokio::test]
    async fn terminal_request_owners_are_bounded_without_evicting_active_requests() {
        let state = test_state("first");
        let live = state.snapshot();
        state.register_request_owner(1, &live);
        for request_id in 2..=(MAX_TERMINAL_REQUEST_OWNERS_PER_GENERATION as u64 + 2) {
            state.register_request_owner(request_id, &live);
            record_terminal_request_owner(&state.request_owners, request_id);
        }

        assert!(state.snapshot_for_request(1).is_some());
        assert!(state.snapshot_for_request(2).is_none());
        assert!(
            state
                .snapshot_for_request(MAX_TERMINAL_REQUEST_OWNERS_PER_GENERATION as u64 + 2)
                .is_some()
        );
        assert_eq!(
            state.request_owners.read().by_id.len(),
            MAX_TERMINAL_REQUEST_OWNERS_PER_GENERATION + 1
        );
    }

    #[tokio::test]
    async fn unload_prunes_step_metrics_and_refreshes_default_advertisement() {
        let state = test_state("first");
        let advertisement = Arc::new(RecordedAdvertisement::default());
        state.set_model_advertisement(advertisement.clone());
        let config = state.snapshot().session_config.as_ref().clone();
        let second = build_live_state("second".to_string(), config)
            .expect("second model state should build");
        state.publish_live(second, true);
        state.metrics.record_step_report(
            "second",
            &EngineStepReport {
                scheduled_requests: 1,
                scheduled_tokens: 8,
                ..Default::default()
            },
        );
        assert!(
            state
                .metrics
                .engine_step_gauges_per_model()
                .iter()
                .any(|(model_id, _)| model_id == "second")
        );

        let removed = state.remove_live("second").expect("second model unloads");
        assert!(
            state
                .metrics
                .engine_step_gauges_per_model()
                .iter()
                .all(|(model_id, _)| model_id != "second")
        );
        assert_eq!(
            advertisement.0.lock().as_slice(),
            &[
                "first".to_string(),
                "second".to_string(),
                "first".to_string()
            ]
        );
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

        assert!(!state.metrics.engine_step_gauges_per_model().is_empty());
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
