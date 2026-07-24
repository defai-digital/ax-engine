use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ax_engine_sdk::{
    EngineSession, EngineSessionConfig, EngineSessionError, EngineStepReport, GenerateRequest,
    GenerateResponse, GenerateStreamEvent, GenerateStreamState, GenerateStreamStepEvent,
    RuntimeReport, SessionRequestReport, SessionRequestState,
};
use tokio::sync::{mpsc, oneshot};

use crate::admission::AdmissionPermit;

type SessionJob = Box<dyn FnOnce(&mut EngineSession) + Send + 'static>;
type SessionFactory =
    Box<dyn FnOnce() -> Result<EngineSession, EngineSessionError> + Send + 'static>;
type NativeEvent = Result<GenerateStreamEvent, EngineSessionError>;
type SessionResult<T> = Result<T, EngineSessionError>;
type StepObserver = Arc<dyn Fn(&EngineStepReport) + Send + Sync + 'static>;
type PressureObserver = Arc<dyn Fn(GenerationPressureEvent) + Send + Sync + 'static>;
type StepwiseTerminalObserver = Arc<dyn Fn(u64) + Send + Sync + 'static>;

/// Process-wide Metal/MLX turn arbiter.
///
/// Default is exclusive (max concurrent = 1): one model holds the device turn
/// at a time. That protects stream-gap under multi-model load but serializes
/// GPU work and caps S1 thr near pure-sum (≈0.99× mlxcel multi-process).
///
/// Set `AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT=2` to let distinct model workers
/// hold turns together. Each worker already owns a dedicated GPU stream created
/// on its thread (`MlxStream::new_gpu`), so concurrent holds map to concurrent
/// Metal submissions — the same device-layer overlap mlxcel gets from one
/// process per model. Same-model re-entry stays exclusive.
pub(crate) struct ModelExecutionArbiter {
    state: parking_lot::Mutex<ModelExecutionState>,
    ready: parking_lot::Condvar,
    max_concurrent: usize,
}

impl Default for ModelExecutionArbiter {
    fn default() -> Self {
        Self::with_max_concurrent(exec_arbiter_max_concurrent_from_env())
    }
}

impl ModelExecutionArbiter {
    pub(crate) fn with_max_concurrent(max_concurrent: usize) -> Self {
        Self {
            state: parking_lot::Mutex::new(ModelExecutionState::default()),
            ready: parking_lot::Condvar::new(),
            max_concurrent: max_concurrent.max(1),
        }
    }

    #[allow(dead_code)] // retained for concurrent-mode diagnostics / future policy
    pub(crate) fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Optionally force exclusive dual-hold after a multi-token prefill quantum.
    ///
    /// Default **on** when `max_concurrent > 1`: M5 dual-hold A/B without this
    /// window measured S1 gap p95 160–220 ms and thr regression under Metal
    /// contention. Adaptive quantum alone does not keep absolute gap ≤50 ms
    /// while both models submit. Set `AX_SERVER_LONG_PREFILL_EXCLUSIVE=0` to
    /// re-open dual-hold for experimental thr A/Bs.
    pub(crate) fn mark_long_prefill_quantum(&self) {
        if !long_prefill_exclusive_enabled() {
            return;
        }
        // Cover late S1 quanta + grace so interactive decode re-enters exclusive
        // mode before the next prefill chunk is submitted under dual-hold.
        const LONG_PREFILL_EXCLUSIVE_GRACE: Duration = Duration::from_millis(80);
        let mut state = self.state.lock();
        state.long_prefill_exclusive_until = Some(Instant::now() + LONG_PREFILL_EXCLUSIVE_GRACE);
    }

    fn effective_max_concurrent(&self, state: &ModelExecutionState) -> usize {
        if self.max_concurrent <= 1 {
            return 1;
        }
        let long_prefill_active = state
            .long_prefill_exclusive_until
            .is_some_and(|until| Instant::now() < until);
        if long_prefill_active {
            1
        } else {
            self.max_concurrent
        }
    }
}

struct ModelExecutionState {
    /// Models currently holding an execution turn. Size ≤ max_concurrent.
    held_models: BTreeSet<String>,
    last_served: Option<String>,
    last_activity: BTreeMap<String, Instant>,
    waiters: BTreeMap<String, usize>,
    stats: BTreeMap<(String, ExecutionWorkClass), ModelExecutionStats>,
    /// When set, force exclusive (max=1) until this instant so long sibling
    /// prefill cannot dual-hold with interactive decode (S1 gap). Short
    /// multi-stream decode (S3) keeps concurrent slots after this expires.
    long_prefill_exclusive_until: Option<Instant>,
}

impl Default for ModelExecutionState {
    fn default() -> Self {
        Self {
            held_models: BTreeSet::new(),
            last_served: None,
            last_activity: BTreeMap::new(),
            waiters: BTreeMap::new(),
            stats: BTreeMap::new(),
            long_prefill_exclusive_until: None,
        }
    }
}

/// Resolve `AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT` (default 1 = exclusive).
///
/// Values < 1 fall back to 1. Cap at 8 so a mis-set env cannot unbounded-open
/// device contention on large multi-model hosts.
pub(crate) fn exec_arbiter_max_concurrent_from_env() -> usize {
    std::env::var("AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .map(|n| n.min(8))
        .unwrap_or(1)
}

/// When true, multi-token prefill quanta force exclusive arbiter turns for a
/// short grace window (see [`ModelExecutionArbiter::mark_long_prefill_quantum`]).
///
/// Default **true** (kill-switch `AX_SERVER_LONG_PREFILL_EXCLUSIVE=0`). Dual-hold
/// without this window failed S1 gap (160–220 ms p95) and thr on M5; keep
/// exclusive isolation for long sibling prefills and rely on pure GPU cuts
/// for thr ≥1.15×.
fn long_prefill_exclusive_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("AX_SERVER_LONG_PREFILL_EXCLUSIVE") {
            Ok(raw) => {
                let v = raw.trim();
                // Explicit off only.
                !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
            }
            Err(_) => true,
        }
    })
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ExecutionWorkClass {
    EngineStep,
    BulkCommand,
}

impl ExecutionWorkClass {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::EngineStep => "engine_step",
            Self::BulkCommand => "bulk_command",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ModelExecutionStats {
    pub(crate) turns_total: u64,
    pub(crate) wait_us_total: u64,
    pub(crate) wait_us_max: u64,
    pub(crate) hold_us_total: u64,
    pub(crate) hold_us_max: u64,
}

struct ModelExecutionTurn<'a> {
    arbiter: &'a ModelExecutionArbiter,
    model_id: String,
    work_class: ExecutionWorkClass,
    started_at: Instant,
}

impl ModelExecutionArbiter {
    fn acquire(&self, model_id: &str, work_class: ExecutionWorkClass) -> ModelExecutionTurn<'_> {
        let wait_started_at = Instant::now();
        let mut state = self.state.lock();
        *state.waiters.entry(model_id.to_string()).or_default() += 1;
        // Exclusive (max=1): classic round-robin on the single slot.
        // Concurrent (max>1): distinct models may hold together; same model
        // never re-enters while already held. Long multi-token prefill forces
        // exclusive until the isolation window expires (S1 gap vs S3 thr).
        while !can_acquire_turn(&state, model_id, self.effective_max_concurrent(&state)) {
            self.ready.wait(&mut state);
        }
        let remove_waiter = if let Some(waiters) = state.waiters.get_mut(model_id) {
            *waiters -= 1;
            *waiters == 0
        } else {
            false
        };
        if remove_waiter {
            state.waiters.remove(model_id);
        }
        state.held_models.insert(model_id.to_string());
        state.last_served = Some(model_id.to_string());
        state
            .last_activity
            .insert(model_id.to_string(), Instant::now());
        let wait_us = duration_us(wait_started_at.elapsed());
        let stats = state
            .stats
            .entry((model_id.to_string(), work_class))
            .or_default();
        stats.turns_total = stats.turns_total.saturating_add(1);
        stats.wait_us_total = stats.wait_us_total.saturating_add(wait_us);
        stats.wait_us_max = stats.wait_us_max.max(wait_us);
        ModelExecutionTurn {
            arbiter: self,
            model_id: model_id.to_string(),
            work_class,
            started_at: Instant::now(),
        }
    }

    pub(crate) fn stats(&self) -> Vec<(String, ExecutionWorkClass, ModelExecutionStats)> {
        self.state
            .lock()
            .stats
            .iter()
            .map(|((model_id, work_class), stats)| (model_id.clone(), *work_class, *stats))
            .collect()
    }

    pub(crate) fn remove_model(&self, model_id: &str) {
        let mut state = self.state.lock();
        state
            .stats
            .retain(|(stats_model_id, _), _| stats_model_id != model_id);
        state.last_activity.remove(model_id);
        state.waiters.remove(model_id);
        state.held_models.remove(model_id);
        if state.last_served.as_deref() == Some(model_id) {
            state.last_served = None;
        }
    }

    fn has_recent_sibling_activity(&self, model_id: &str, recent_for: Duration) -> bool {
        let now = Instant::now();
        let state = self.state.lock();
        state
            .held_models
            .iter()
            .any(|held_model| held_model.as_str() != model_id)
            || state
                .waiters
                .iter()
                .any(|(waiting_model, waiters)| waiting_model != model_id && *waiters > 0)
            || state.last_activity.iter().any(|(active_model, at)| {
                active_model != model_id && now.saturating_duration_since(*at) <= recent_for
            })
    }
}

impl Drop for ModelExecutionTurn<'_> {
    fn drop(&mut self) {
        let mut state = self.arbiter.state.lock();
        let hold_us = duration_us(self.started_at.elapsed());
        let stats = state
            .stats
            .entry((self.model_id.clone(), self.work_class))
            .or_default();
        stats.hold_us_total = stats.hold_us_total.saturating_add(hold_us);
        stats.hold_us_max = stats.hold_us_max.max(hold_us);
        state.held_models.remove(&self.model_id);
        state
            .last_activity
            .insert(self.model_id.clone(), Instant::now());
        drop(state);
        // Wake all waiters: exclusive mode relies on next_waiting_model();
        // concurrent mode needs broadcast so every eligible waiter re-checks
        // free capacity. Model count is typically small (1-3).
        self.arbiter.ready.notify_all();
    }
}

/// Whether `model_id` may enter a turn under the current hold set.
fn can_acquire_turn(state: &ModelExecutionState, model_id: &str, max_concurrent: usize) -> bool {
    if state.held_models.contains(model_id) {
        return false;
    }
    if state.held_models.len() >= max_concurrent {
        return false;
    }
    if max_concurrent <= 1 {
        // Strict exclusive fairness: only the round-robin head may enter.
        return next_waiting_model(state).as_deref() == Some(model_id);
    }
    // Concurrent: any non-held model may fill a free slot. When more distinct
    // waiters exist than free slots, prefer the round-robin head so one model
    // cannot monopolize re-entry after a release.
    let free_slots = max_concurrent.saturating_sub(state.held_models.len());
    if free_slots == 0 {
        return false;
    }
    let distinct_waiters = state.waiters.len();
    if distinct_waiters <= free_slots {
        return true;
    }
    next_waiting_model(state).as_deref() == Some(model_id)
}

fn duration_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn next_waiting_model(state: &ModelExecutionState) -> Option<String> {
    let first = state.waiters.keys().next()?.clone();
    let Some(last_served) = state.last_served.as_deref() else {
        return Some(first);
    };
    state
        .waiters
        .keys()
        .find(|model_id| model_id.as_str() > last_served)
        .cloned()
        .or(Some(first))
}

#[derive(Clone)]
struct ModelExecutionTarget {
    model_id: Arc<String>,
    arbiter: Arc<ModelExecutionArbiter>,
}

/// Prefill tokens admitted per engine turn when a sibling model is active.
///
/// Historically this was **1**, which serialized a long sibling prefill into
/// one-token turns and destroyed aggregate throughput under multi-model
/// isolation (flip S1: ~0.33× vs mlxcel while stream gap stayed ~35 ms).
///
/// mlxcel deep-review (2026-07-24) + flip notes: size the quantum so one turn's
/// wall time stays under the interactive stream-gap SLO (50 ms). On M5 Max
/// Gemma-12B-class 4-bit prefill (~0.66–0.8 ms/tok) that is roughly 64–96
/// tokens. Prefer 64 as the conservative default; override via
/// `AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS` when calibrating.
/// Sibling-active prefill quantum (tokens per engine turn) — starting point.
///
/// M5 dual-model S1 calibration:
/// - 1 token: historical isolation envelope (gap ~35–150 ms depending on path)
/// - 16/64 fixed tokens: interactive gap p95 ballooned to 0.6–0.8 s because
///   long prefill attention cost grows with position, so fixed-token quanta
///   are not wall-time-safe mid-prompt.
///
/// Default **64** is the *initial* sibling-active quantum (≈ 40–50 ms on M5
/// Max Gemma-12B 4-bit at ~0.66–0.8 ms/tok). When adaptive isolation is on,
/// [`advance_shared_engine`] re-sizes from measured µs/token so one sibling
/// prefill turn stays under the stream-gap SLO budget
/// ([`ADAPTIVE_PREFILL_GAP_SLO_US`]). Override via
/// `AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS`.
pub(crate) const ADAPTIVE_PREFILL_LATENCY_TOKENS_PER_STEP_DEFAULT: u32 = 64;
pub(crate) const ADAPTIVE_PREFILL_THROUGHPUT_TOKENS_PER_STEP: u32 = 256;
/// Interactive p95 stream-gap SLO for one sibling prefill turn.
/// Exclusive S1 gap p95 is ~9 ms vs ~37 ms mlxcel (ratio budget ~33 ms).
/// Use 40 ms / max 128 so early cheap positions take large quanta (fewer
/// arbiter turns). Adaptive shrinks from measured µs/tok as attention cost
/// grows. Cold outliers previously hurt at fixed 96; adaptive start stays 64.
const ADAPTIVE_PREFILL_GAP_SLO_US: u64 = 40_000;
const ADAPTIVE_PREFILL_MAX_TOKENS: u32 = 128;
/// Floor under sibling-active adaptive quanta. Concurrent dual-hold needs the
/// floor at 1 so late long-context quanta can shrink under the 50 ms gap SLO
/// (8-token floor still measured ~160 ms p95 on M5). Exclusive thr is pure-sum
/// limited; tiny late quanta cost thr less than a failed gap gate.
const ADAPTIVE_PREFILL_MIN_TOKENS: u32 = 1;
const ADAPTIVE_PREFILL_SIBLING_ACTIVITY_GRACE: Duration = Duration::from_millis(250);

/// Resolve the sibling-active prefill quantum from an optional env value.
///
/// Empty / invalid / zero values fall back to
/// [`ADAPTIVE_PREFILL_LATENCY_TOKENS_PER_STEP_DEFAULT`] so operators cannot
/// accidentally re-enable the historical 1-token pathological endpoint by
/// setting a garbage env var. Explicit `1` is allowed for gap-tight A/Bs.
pub(crate) fn resolve_adaptive_prefill_latency_tokens(raw: Option<&str>) -> u32 {
    raw.and_then(|value| value.trim().parse().ok())
        .filter(|n: &u32| *n > 0)
        .unwrap_or(ADAPTIVE_PREFILL_LATENCY_TOKENS_PER_STEP_DEFAULT)
}

fn adaptive_prefill_latency_tokens_per_step() -> u32 {
    static CACHED: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        resolve_adaptive_prefill_latency_tokens(
            std::env::var("AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS")
                .ok()
                .as_deref(),
        )
    })
}

/// Size the next sibling prefill quantum from measured µs/token so one turn's
/// wall time targets [`ADAPTIVE_PREFILL_GAP_SLO_US`].
///
/// Using rate (us/tok) instead of binary grow/shrink keeps mid-prefill quanta
/// large enough for S1 thr while still respecting the 50 ms gap SLO as
/// attention cost grows with position.
#[allow(dead_code)]
pub(crate) fn adjust_adaptive_prefill_tokens(current_tokens: u32, last_runner_time_us: u64) -> u32 {
    adjust_adaptive_prefill_tokens_with_work(current_tokens, last_runner_time_us, current_tokens)
}

/// Like [`adjust_adaptive_prefill_tokens`] but uses the actual tokens scheduled
/// on the previous turn when available (more accurate µs/tok).
pub(crate) fn adjust_adaptive_prefill_tokens_with_work(
    current_tokens: u32,
    last_runner_time_us: u64,
    last_scheduled_tokens: u32,
) -> u32 {
    let current = current_tokens.clamp(ADAPTIVE_PREFILL_MIN_TOKENS, ADAPTIVE_PREFILL_MAX_TOKENS);
    let work = last_scheduled_tokens.max(1);
    if last_runner_time_us == 0 {
        return current;
    }
    let us_per_tok = last_runner_time_us.div_ceil(u64::from(work)).max(1);
    let target = (ADAPTIVE_PREFILL_GAP_SLO_US / us_per_tok).max(1);
    let target = u32::try_from(target)
        .unwrap_or(ADAPTIVE_PREFILL_MAX_TOKENS)
        .clamp(ADAPTIVE_PREFILL_MIN_TOKENS, ADAPTIVE_PREFILL_MAX_TOKENS);
    // Over budget: snap to target immediately (protect gap SLO).
    if last_runner_time_us > ADAPTIVE_PREFILL_GAP_SLO_US {
        return target;
    }
    // Under budget: blend toward target to avoid oscillation.
    let blended = if target > current {
        current.saturating_add(((target - current) / 2).max(1))
    } else if target < current {
        current.saturating_sub(((current - target) / 2).max(1))
    } else {
        current
    };
    blended.clamp(ADAPTIVE_PREFILL_MIN_TOKENS, ADAPTIVE_PREFILL_MAX_TOKENS)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GenerationPressureEvent {
    CommandSaturated,
    StreamBacklogOverflow,
}

const COMMAND_QUEUE_CAPACITY: usize = 256;
const COMMANDS_PER_TICK: usize = 64;
const BULK_COMMANDS_PER_ACTIVE_TICK: usize = 1;
// Large enough that a full interactive completion (flip S0: 192 tokens) can
// finish on the generation worker without waiting for the HTTP/SSE consumer.
// Local microbenches on M5 Max showed native generate at ~106 tok/s vs SSE
// stream at ~81 tok/s when the worker was paced by a tiny backlog; decoupling
// decode from detokenize/SSE write closes that gap.
const STREAM_EVENT_CHANNEL_CAPACITY: usize = 512;
// Match the flip S0 generation length so the worker can complete decode while
// the OpenAI SSE adapter drains independently (see STREAM_EVENT_CHANNEL_CAPACITY).
const STREAM_WORKER_BACKLOG_CAPACITY: usize = 256;

enum ServiceCommand {
    Execute(SessionJob),
    StartStream {
        request_id: u64,
        request: GenerateRequest,
        events: mpsc::Sender<NativeEvent>,
        terminal_events: Arc<parking_lot::Mutex<VecDeque<NativeEvent>>>,
        started: oneshot::Sender<Result<(), EngineSessionError>>,
        permit: AdmissionPermit,
    },
    SubmitStepwise {
        request_id: u64,
        request: GenerateRequest,
        permit: AdmissionPermit,
        response: oneshot::Sender<SessionResult<SessionRequestReport>>,
    },
    RequestSnapshot {
        request_id: u64,
        response: oneshot::Sender<SessionResult<SessionRequestReport>>,
    },
    CancelStepwise {
        request_id: u64,
        response: oneshot::Sender<SessionResult<SessionRequestReport>>,
    },
    Advance {
        response: oneshot::Sender<SessionResult<EngineStepReport>>,
    },
    HasActiveStepwise {
        response: oneshot::Sender<bool>,
    },
}

impl ServiceCommand {
    const fn is_bulk(&self) -> bool {
        matches!(self, Self::Execute(_))
    }
}

struct ServiceState {
    alive: AtomicBool,
    pending_jobs: AtomicUsize,
    queued_commands: AtomicUsize,
    active_streams: AtomicUsize,
    buffered_stream_events: AtomicUsize,
    step_observer: parking_lot::RwLock<Option<StepObserver>>,
    pressure_observer: parking_lot::RwLock<Option<PressureObserver>>,
    stepwise_terminal_observer: parking_lot::RwLock<Option<StepwiseTerminalObserver>>,
    execution_target: parking_lot::RwLock<Option<ModelExecutionTarget>>,
    adaptive_prefill_isolation: AtomicBool,
    /// Last engine-step runner wall time (µs) used to feedback-control the
    /// sibling prefill quantum under adaptive isolation.
    last_step_runner_time_us: AtomicU64,
    /// Tokens scheduled on the last engine step (for µs/tok estimation).
    last_step_scheduled_tokens: AtomicU32,
    /// Current sibling-active prefill quantum (tokens/turn).
    adaptive_prefill_tokens: AtomicU32,
}

pub(crate) struct NativeGenerationService {
    sender: parking_lot::Mutex<Option<std::sync::mpsc::Sender<ServiceCommand>>>,
    state: Arc<ServiceState>,
    worker: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
}

pub(crate) struct NativeEventReceiver {
    receiver: mpsc::Receiver<NativeEvent>,
    terminal_events: Arc<parking_lot::Mutex<VecDeque<NativeEvent>>>,
}

impl NativeEventReceiver {
    pub(crate) async fn recv(&mut self) -> Option<NativeEvent> {
        match self.receiver.recv().await {
            Some(event) => Some(event),
            None => self.terminal_events.lock().pop_front(),
        }
    }

    pub(crate) fn blocking_recv(&mut self) -> Option<NativeEvent> {
        match self.receiver.blocking_recv() {
            Some(event) => Some(event),
            None => self.terminal_events.lock().pop_front(),
        }
    }
}

impl NativeGenerationService {
    pub(crate) fn spawn(
        config: EngineSessionConfig,
    ) -> Result<(Arc<Self>, RuntimeReport), GenerationServiceStartError> {
        Self::spawn_with_factory(move || EngineSession::new(config))
    }

    pub(crate) fn spawn_replacement(
        config: EngineSessionConfig,
    ) -> Result<(Arc<Self>, RuntimeReport), GenerationServiceStartError> {
        Self::spawn_with_factory(move || {
            EngineSession::clear_native_model_compile_caches();
            EngineSession::new(config)
        })
    }

    fn spawn_with_factory<F>(
        factory: F,
    ) -> Result<(Arc<Self>, RuntimeReport), GenerationServiceStartError>
    where
        F: FnOnce() -> Result<EngineSession, EngineSessionError> + Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel::<ServiceCommand>();
        let (startup_sender, startup_receiver) = std::sync::mpsc::sync_channel(1);
        let state = Arc::new(ServiceState {
            alive: AtomicBool::new(false),
            pending_jobs: AtomicUsize::new(0),
            queued_commands: AtomicUsize::new(0),
            active_streams: AtomicUsize::new(0),
            buffered_stream_events: AtomicUsize::new(0),
            step_observer: parking_lot::RwLock::new(None),
            pressure_observer: parking_lot::RwLock::new(None),
            stepwise_terminal_observer: parking_lot::RwLock::new(None),
            execution_target: parking_lot::RwLock::new(None),
            adaptive_prefill_isolation: AtomicBool::new(false),
            last_step_runner_time_us: AtomicU64::new(0),
            last_step_scheduled_tokens: AtomicU32::new(0),
            adaptive_prefill_tokens: AtomicU32::new(
                ADAPTIVE_PREFILL_LATENCY_TOKENS_PER_STEP_DEFAULT,
            ),
        });
        let worker_state = Arc::clone(&state);
        let worker = std::thread::Builder::new()
            .name("ax-native-generation".to_string())
            .spawn(move || run_worker(Box::new(factory), receiver, startup_sender, &worker_state))
            .map_err(GenerationServiceStartError::ThreadStart)?;
        match startup_receiver.recv() {
            Ok(Ok(runtime_report)) => Ok((
                Arc::new(Self {
                    sender: parking_lot::Mutex::new(Some(sender)),
                    state,
                    worker: parking_lot::Mutex::new(Some(worker)),
                }),
                runtime_report,
            )),
            Ok(Err(error)) => {
                drop(sender);
                let _ = worker.join();
                Err(GenerationServiceStartError::Engine(error))
            }
            Err(_) => {
                drop(sender);
                let worker_panicked = worker.join().is_err();
                Err(if worker_panicked {
                    GenerationServiceStartError::WorkerPanicked
                } else {
                    GenerationServiceStartError::ReadinessChannelClosed
                })
            }
        }
    }

    pub(crate) async fn execute<T, F>(&self, operation: F) -> Result<T, GenerationServiceError>
    where
        T: Send + 'static,
        F: FnOnce(&mut EngineSession) -> Result<T, EngineSessionError> + Send + 'static,
    {
        let (response_tx, response_rx) = oneshot::channel();
        self.submit(move |session| {
            let _ = response_tx.send(operation(session));
        })?;
        response_rx
            .await
            .map_err(|_| GenerationServiceError::Unavailable)?
            .map_err(GenerationServiceError::Engine)
    }

    pub(crate) fn submit<F>(&self, operation: F) -> Result<(), GenerationServiceError>
    where
        F: FnOnce(&mut EngineSession) + Send + 'static,
    {
        self.enqueue(ServiceCommand::Execute(Box::new(operation)))
    }

    pub(crate) async fn generate(
        &self,
        request_id: u64,
        request: GenerateRequest,
        permit: AdmissionPermit,
    ) -> Result<GenerateResponse, GenerationServiceError> {
        let mut events = self.start_stream(request_id, request, permit).await?;
        let mut observed_event_count = 0_u64;
        while let Some(event) = events.recv().await {
            observed_event_count = observed_event_count.saturating_add(1);
            if let GenerateStreamEvent::Response(response) =
                event.map_err(GenerationServiceError::Engine)?
            {
                return Ok(response.response);
            }
        }
        Err(GenerationServiceError::Engine(
            EngineSessionError::StreamEndedWithoutResponse {
                request_id,
                observed_event_count,
            },
        ))
    }

    pub(crate) async fn start_stream(
        &self,
        request_id: u64,
        request: GenerateRequest,
        permit: AdmissionPermit,
    ) -> Result<NativeEventReceiver, GenerationServiceError> {
        let (events_tx, events_rx) = mpsc::channel(STREAM_EVENT_CHANNEL_CAPACITY);
        let terminal_events = Arc::new(parking_lot::Mutex::new(VecDeque::new()));
        let (started_tx, started_rx) = oneshot::channel();
        self.enqueue(ServiceCommand::StartStream {
            request_id,
            request,
            events: events_tx,
            terminal_events: Arc::clone(&terminal_events),
            started: started_tx,
            permit,
        })?;
        started_rx
            .await
            .map_err(|_| GenerationServiceError::Unavailable)?
            .map_err(GenerationServiceError::Engine)?;
        Ok(NativeEventReceiver {
            receiver: events_rx,
            terminal_events,
        })
    }

    pub(crate) async fn submit_stepwise(
        &self,
        request_id: u64,
        request: GenerateRequest,
        permit: AdmissionPermit,
    ) -> Result<SessionRequestReport, GenerationServiceError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.enqueue(ServiceCommand::SubmitStepwise {
            request_id,
            request,
            permit,
            response: response_tx,
        })?;
        receive_engine_response(response_rx).await
    }

    pub(crate) async fn request_snapshot(
        &self,
        request_id: u64,
    ) -> Result<SessionRequestReport, GenerationServiceError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.enqueue(ServiceCommand::RequestSnapshot {
            request_id,
            response: response_tx,
        })?;
        receive_engine_response(response_rx).await
    }

    pub(crate) async fn cancel_stepwise(
        &self,
        request_id: u64,
    ) -> Result<SessionRequestReport, GenerationServiceError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.enqueue(ServiceCommand::CancelStepwise {
            request_id,
            response: response_tx,
        })?;
        receive_engine_response(response_rx).await
    }

    pub(crate) async fn advance(&self) -> Result<EngineStepReport, GenerationServiceError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.enqueue(ServiceCommand::Advance {
            response: response_tx,
        })?;
        receive_engine_response(response_rx).await
    }

    pub(crate) async fn has_active_stepwise(&self) -> Result<bool, GenerationServiceError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.enqueue(ServiceCommand::HasActiveStepwise {
            response: response_tx,
        })?;
        response_rx
            .await
            .map_err(|_| GenerationServiceError::Unavailable)
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.state.alive.load(Ordering::Acquire)
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.state.pending_jobs.load(Ordering::Acquire) != 0
    }

    pub(crate) fn pending_jobs(&self) -> usize {
        self.state.pending_jobs.load(Ordering::Acquire)
    }

    pub(crate) fn queued_commands(&self) -> usize {
        self.state.queued_commands.load(Ordering::Acquire)
    }

    pub(crate) fn active_streams(&self) -> usize {
        self.state.active_streams.load(Ordering::Acquire)
    }

    pub(crate) fn buffered_stream_events(&self) -> usize {
        self.state.buffered_stream_events.load(Ordering::Acquire)
    }

    pub(crate) const fn command_queue_capacity(&self) -> usize {
        COMMAND_QUEUE_CAPACITY
    }

    pub(crate) fn set_step_observer<F>(&self, observer: F)
    where
        F: Fn(&EngineStepReport) + Send + Sync + 'static,
    {
        *self.state.step_observer.write() = Some(Arc::new(observer));
    }

    pub(crate) fn set_pressure_observer<F>(&self, observer: F)
    where
        F: Fn(GenerationPressureEvent) + Send + Sync + 'static,
    {
        *self.state.pressure_observer.write() = Some(Arc::new(observer));
    }

    pub(crate) fn set_stepwise_terminal_observer<F>(&self, observer: F)
    where
        F: Fn(u64) + Send + Sync + 'static,
    {
        *self.state.stepwise_terminal_observer.write() = Some(Arc::new(observer));
    }

    pub(crate) fn set_execution_arbiter(
        &self,
        model_id: Arc<String>,
        arbiter: Arc<ModelExecutionArbiter>,
    ) {
        *self.state.execution_target.write() = Some(ModelExecutionTarget { model_id, arbiter });
    }

    pub(crate) fn set_adaptive_prefill_isolation(&self, enabled: bool) {
        self.state
            .adaptive_prefill_isolation
            .store(enabled, Ordering::Release);
        if enabled {
            let start = adaptive_prefill_latency_tokens_per_step();
            self.state
                .adaptive_prefill_tokens
                .store(start, Ordering::Release);
            self.state
                .last_step_runner_time_us
                .store(0, Ordering::Release);
        }
    }

    pub(crate) async fn shutdown(&self) -> Result<(), GenerationServiceError> {
        self.sender.lock().take();
        let Some(worker) = self.worker.lock().take() else {
            return Ok(());
        };
        let result = tokio::task::spawn_blocking(move || worker.join())
            .await
            .map_err(|_| GenerationServiceError::Unavailable)?;
        result.map_err(|_| GenerationServiceError::Unavailable)
    }

    fn enqueue(&self, command: ServiceCommand) -> Result<(), GenerationServiceError> {
        let sender = self.sender.lock();
        let Some(sender) = sender.as_ref() else {
            return Err(GenerationServiceError::Unavailable);
        };
        if self
            .state
            .queued_commands
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                (queued < COMMAND_QUEUE_CAPACITY).then_some(queued + 1)
            })
            .is_err()
        {
            record_pressure_event(&self.state, GenerationPressureEvent::CommandSaturated);
            return Err(GenerationServiceError::Saturated);
        }
        self.state.pending_jobs.fetch_add(1, Ordering::AcqRel);
        match sender.send(command) {
            Ok(()) => Ok(()),
            Err(_) => {
                rollback_failed_enqueue(&self.state);
                Err(GenerationServiceError::Unavailable)
            }
        }
    }
}

impl Drop for NativeGenerationService {
    fn drop(&mut self) {
        self.sender.get_mut().take();
    }
}

async fn receive_engine_response<T>(
    receiver: oneshot::Receiver<SessionResult<T>>,
) -> Result<T, GenerationServiceError> {
    receiver
        .await
        .map_err(|_| GenerationServiceError::Unavailable)?
        .map_err(GenerationServiceError::Engine)
}

#[derive(Debug)]
pub(crate) enum GenerationServiceError {
    Engine(EngineSessionError),
    Saturated,
    Unavailable,
}

impl fmt::Display for GenerationServiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engine(error) => error.fmt(formatter),
            Self::Saturated => formatter.write_str("native generation command queue is saturated"),
            Self::Unavailable => formatter.write_str("native generation worker is unavailable"),
        }
    }
}

impl std::error::Error for GenerationServiceError {}

#[derive(Debug)]
pub(crate) enum GenerationServiceStartError {
    Engine(EngineSessionError),
    ReadinessChannelClosed,
    ThreadStart(std::io::Error),
    WorkerPanicked,
}

impl fmt::Display for GenerationServiceStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engine(error) => error.fmt(formatter),
            Self::ReadinessChannelClosed => {
                formatter.write_str("native generation worker exited before reporting readiness")
            }
            Self::ThreadStart(error) => {
                write!(
                    formatter,
                    "failed to start native generation worker: {error}"
                )
            }
            Self::WorkerPanicked => {
                formatter.write_str("native generation worker panicked during startup")
            }
        }
    }
}

impl std::error::Error for GenerationServiceStartError {}

impl From<EngineSessionError> for GenerationServiceStartError {
    fn from(error: EngineSessionError) -> Self {
        Self::Engine(error)
    }
}

fn run_worker(
    factory: SessionFactory,
    receiver: std::sync::mpsc::Receiver<ServiceCommand>,
    startup_sender: std::sync::mpsc::SyncSender<Result<RuntimeReport, EngineSessionError>>,
    state: &ServiceState,
) {
    let _exit_guard = WorkerExitGuard(state);
    let session = match factory() {
        Ok(session) => session,
        Err(error) => {
            let _ = startup_sender.send(Err(error));
            return;
        }
    };
    let runtime_report = session.runtime_report();
    state.alive.store(true, Ordering::Release);
    if startup_sender.send(Ok(runtime_report)).is_err() {
        return;
    }
    // Contain engine/MLX panics to this worker (H1): the MLX FFI turns
    // runtime failures into Rust panics, and in unwind-capable builds
    // (dev/test, `release-pyext`, `release-server`) an uncontained panic
    // would otherwise take the thread down through the runtime's default
    // hook with no structured cleanup or operator guidance. The session,
    // streams, and permits are all owned by the closure, so an unwind
    // drops them: response/stream channels disconnect (mapped to the
    // existing unavailable/503 contracts), admission permits release, and
    // a held ModelExecutionArbiter turn is returned via RAII. The session
    // is never touched after a catch — post-panic engine state is
    // untrusted, so the worker retires and `POST /v1/model/load` is the
    // recovery path (same contract as a stopped worker). Under
    // `panic = "abort"` builds this is a no-op by construction.
    let loop_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_worker_loop(session, receiver, state);
    }));
    if let Err(payload) = loop_outcome {
        tracing::error!(
            panic = %panic_payload_message(payload.as_ref()),
            "native generation worker panicked; this model's worker is retiring \
             (in-flight requests fail, the process and sibling models continue). \
             Recover with POST /v1/model/load"
        );
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "<non-string panic payload>"
    }
}

fn run_worker_loop(
    mut session: EngineSession,
    receiver: std::sync::mpsc::Receiver<ServiceCommand>,
    state: &ServiceState,
) {
    let mut active_streams: BTreeMap<u64, ActiveStream> = BTreeMap::new();
    let mut stepwise_permits: BTreeMap<u64, AdmissionPermit> = BTreeMap::new();
    let mut latency_commands = VecDeque::new();
    let mut bulk_commands = VecDeque::new();
    let mut disconnected = false;
    loop {
        let mut engine_advanced = false;
        if active_streams.is_empty()
            && latency_commands.is_empty()
            && bulk_commands.is_empty()
            && !disconnected
        {
            match receiver.recv() {
                Ok(command) => {
                    queue_worker_command(command, &mut latency_commands, &mut bulk_commands)
                }
                Err(_) => disconnected = true,
            }
        }
        loop {
            match receiver.try_recv() {
                Ok(command) => {
                    queue_worker_command(command, &mut latency_commands, &mut bulk_commands)
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected {
            detach_all_streams_for_shutdown(&mut session, &mut active_streams, state);
            cancel_all_stepwise(&mut session, &mut stepwise_permits, state);
            update_stream_gauges(state, &active_streams);
            return;
        }
        for _ in 0..COMMANDS_PER_TICK {
            let Some(command) = latency_commands.pop_front() else {
                break;
            };
            begin_command(state);
            engine_advanced = handle_command(
                command,
                &mut session,
                &mut active_streams,
                &mut stepwise_permits,
                state,
            );
            if engine_advanced {
                break;
            }
        }
        if !engine_advanced {
            let bulk_budget = if active_streams.is_empty() {
                COMMANDS_PER_TICK
            } else {
                BULK_COMMANDS_PER_ACTIVE_TICK
            };
            for _ in 0..bulk_budget {
                let Some(command) = bulk_commands.pop_front() else {
                    break;
                };
                begin_command(state);
                let _ = handle_command(
                    command,
                    &mut session,
                    &mut active_streams,
                    &mut stepwise_permits,
                    state,
                );
            }
        }
        if engine_advanced {
            // Engine already advanced via command handling.
        } else if !active_streams.is_empty() {
            let _ = advance_shared_engine(
                &mut session,
                &mut active_streams,
                &mut stepwise_permits,
                state,
            );
        } else {
            maintain_streams(&mut session, &mut active_streams, state);
        }
        update_stream_gauges(state, &active_streams);
    }
}

fn queue_worker_command(
    command: ServiceCommand,
    latency_commands: &mut VecDeque<ServiceCommand>,
    bulk_commands: &mut VecDeque<ServiceCommand>,
) {
    if command.is_bulk() {
        bulk_commands.push_back(command);
    } else {
        latency_commands.push_back(command);
    }
}

struct ActiveStream {
    state: GenerateStreamState,
    events: mpsc::Sender<NativeEvent>,
    terminal_events: Arc<parking_lot::Mutex<VecDeque<NativeEvent>>>,
    pending_events: VecDeque<NativeEvent>,
    request_event_pending: bool,
    permit: Option<AdmissionPermit>,
    /// Coalesced token deltas waiting for an SSE emit. First token flushes
    /// immediately (TTFT); later tokens batch up to
    /// [`STREAM_TOKEN_EMIT_BATCH`] to cut per-token channel/SSE overhead.
    pending_delta_tokens: Vec<u32>,
    pending_delta_logprobs: Vec<Option<f32>>,
    pending_step: Option<EngineStepReport>,
    pending_request: Option<SessionRequestReport>,
    first_output_emitted: bool,
}

/// After the first stream token, coalesce this many tokens per SSE frame.
/// Keep at 1 so interactive p95 stream-gap tracks per-token decode (~10 ms)
/// and stays competitive with mlxcel (~11 ms) under the flip gap-ratio gate
/// (≤0.90× and absolute ≤50 ms). Engine burst below keeps GPU fed.
const STREAM_TOKEN_EMIT_BATCH: usize = 1;

/// Single-stream engine steps per worker tick after the first token.
/// Larger than the SSE emit batch so the Metal double-buffer stays fed even
/// when the async SSE consumer is slightly behind.
const STREAM_ENGINE_STEP_BURST: usize = 64;

/// Sibling-active exclusive-arbiter engine step burst.
///
/// Under exclusive multi-model load, a full [`STREAM_ENGINE_STEP_BURST`] HOL
/// blows interactive gap. Burst=1 was the historical safe default, but S1
/// locked gap p95 is ~9 ms vs a ~33 ms ratio budget (and 50 ms absolute), so
/// a small multi-step hold amortizes arbiter reacquire without consuming the
/// gap headroom. Override via `AX_SERVER_SIBLING_ENGINE_STEP_BURST`.
/// Default **4**: exclusive multi-model thr envelope (not 1 = thr tax, not full
/// HOL burst). Concurrent dual-hold A/Bs may set 1 via env.
const SIBLING_ENGINE_STEP_BURST_DEFAULT: usize = 4;

fn sibling_engine_step_burst() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_SERVER_SIBLING_ENGINE_STEP_BURST")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n: &usize| *n > 0)
            .unwrap_or(SIBLING_ENGINE_STEP_BURST_DEFAULT)
            .min(STREAM_ENGINE_STEP_BURST)
    })
}

fn handle_command(
    command: ServiceCommand,
    session: &mut EngineSession,
    active_streams: &mut BTreeMap<u64, ActiveStream>,
    stepwise_permits: &mut BTreeMap<u64, AdmissionPermit>,
    state: &ServiceState,
) -> bool {
    match command {
        ServiceCommand::Execute(job) => {
            let execution_target = state.execution_target.read().clone();
            let _turn = execution_target.as_ref().map(|target| {
                target
                    .arbiter
                    .acquire(target.model_id.as_ref(), ExecutionWorkClass::BulkCommand)
            });
            job(session);
            complete_job(state);
        }
        ServiceCommand::StartStream {
            request_id,
            request,
            events,
            terminal_events,
            started,
            permit,
        } => match session.stream_generate_state_with_request_id(request_id, request) {
            Ok(stream_state) => {
                let previous = active_streams.insert(
                    request_id,
                    ActiveStream {
                        state: stream_state,
                        events,
                        terminal_events,
                        pending_events: VecDeque::new(),
                        request_event_pending: true,
                        permit: Some(permit),
                        pending_delta_tokens: Vec::new(),
                        pending_delta_logprobs: Vec::new(),
                        pending_step: None,
                        pending_request: None,
                        first_output_emitted: false,
                    },
                );
                debug_assert!(previous.is_none(), "request IDs are process-unique");
                if started.send(Ok(())).is_err() {
                    let _ = session.cancel_request(request_id);
                    active_streams.remove(&request_id);
                    complete_job(state);
                }
            }
            Err(error) => {
                let _ = started.send(Err(error));
                complete_job(state);
            }
        },
        ServiceCommand::SubmitStepwise {
            request_id,
            request,
            permit,
            response,
        } => {
            let result = submit_stepwise_request(session, request_id, request);
            match result {
                Ok(report) => {
                    let previous = stepwise_permits.insert(request_id, permit);
                    debug_assert!(previous.is_none(), "request IDs are process-unique");
                    if response.send(Ok(report)).is_err() {
                        let _ = session.cancel_request(request_id);
                        if stepwise_permits.remove(&request_id).is_some() {
                            complete_job(state);
                            record_stepwise_terminal(state, request_id);
                        }
                    }
                }
                Err(error) => {
                    let _ = response.send(Err(error));
                    complete_job(state);
                }
            }
        }
        ServiceCommand::RequestSnapshot {
            request_id,
            response,
        } => {
            let result = request_report(session, request_id);
            if let Ok(report) = result.as_ref() {
                release_terminal_stepwise_permit(report, stepwise_permits, state);
            }
            let _ = response.send(result);
            complete_job(state);
        }
        ServiceCommand::CancelStepwise {
            request_id,
            response,
        } => {
            let result = cancel_stepwise_request(session, request_id);
            if let Ok(report) = result.as_ref() {
                release_terminal_stepwise_permit(report, stepwise_permits, state);
            }
            let _ = response.send(result);
            complete_job(state);
        }
        ServiceCommand::Advance { response } => {
            let result = advance_shared_engine(session, active_streams, stepwise_permits, state);
            let _ = response.send(result);
            complete_job(state);
            return true;
        }
        ServiceCommand::HasActiveStepwise { response } => {
            let _ = response.send(!stepwise_permits.is_empty());
            complete_job(state);
        }
    }
    false
}

fn submit_stepwise_request(
    session: &mut EngineSession,
    request_id: u64,
    request: GenerateRequest,
) -> SessionResult<SessionRequestReport> {
    let request_id = session.submit_generate_with_request_id(request_id, request)?;
    session
        .request_report(request_id)
        .ok_or(EngineSessionError::RequestReportInvariantViolation {
            request_id,
            message: "request missing immediately after submission",
        })
}

fn request_report(session: &EngineSession, request_id: u64) -> SessionResult<SessionRequestReport> {
    session
        .request_report(request_id)
        .ok_or(EngineSessionError::RequestReportInvariantViolation {
            request_id,
            message: "request missing from preview session state",
        })
}

fn cancel_stepwise_request(
    session: &mut EngineSession,
    request_id: u64,
) -> SessionResult<SessionRequestReport> {
    request_report(session, request_id)?;
    session.cancel_request(request_id)?;
    session
        .request_report(request_id)
        .ok_or(EngineSessionError::RequestReportInvariantViolation {
            request_id,
            message: "request missing after cancellation",
        })
}

fn maintain_streams(
    session: &mut EngineSession,
    active_streams: &mut BTreeMap<u64, ActiveStream>,
    service_state: &ServiceState,
) -> bool {
    let mut progressed = false;
    let mut terminal = Vec::new();
    for (request_id, stream) in active_streams.iter_mut() {
        if stream.events.is_closed() {
            let _ = session.cancel_request(*request_id);
            stream.permit.take();
            discard_pending_events(stream, service_state);
            terminal.push(*request_id);
            progressed = true;
            continue;
        }
        match flush_pending_events(stream, service_state) {
            StreamDelivery::Queued => progressed = true,
            StreamDelivery::Closed => {
                let _ = session.cancel_request(*request_id);
                stream.permit.take();
                discard_pending_events(stream, service_state);
                terminal.push(*request_id);
                progressed = true;
                continue;
            }
            StreamDelivery::Backpressured => {}
        }
        if stream.request_event_pending {
            match session.next_stream_event(&mut stream.state) {
                Ok(Some(event)) => {
                    stream.request_event_pending = false;
                    match enqueue_stream_event(stream, Ok(event), service_state) {
                        EnqueueResult::Queued => progressed = true,
                        EnqueueResult::Closed => {
                            let _ = session.cancel_request(*request_id);
                            stream.permit.take();
                            terminal.push(*request_id);
                        }
                        EnqueueResult::Overflow => {
                            record_pressure_event(
                                service_state,
                                GenerationPressureEvent::StreamBacklogOverflow,
                            );
                            detach_stream_with_error(
                                session,
                                *request_id,
                                stream,
                                EngineSessionError::RequestReportInvariantViolation {
                                    request_id: *request_id,
                                    message: "stream consumer exceeded the bounded worker backlog",
                                },
                                service_state,
                            );
                            terminal.push(*request_id);
                        }
                    }
                }
                Ok(None) => terminal.push(*request_id),
                Err(error) => {
                    detach_stream_with_error(session, *request_id, stream, error, service_state);
                    terminal.push(*request_id);
                }
            }
        }
    }
    for request_id in terminal {
        active_streams.remove(&request_id);
        complete_job(service_state);
    }
    progressed
}

fn advance_shared_engine(
    session: &mut EngineSession,
    active_streams: &mut BTreeMap<u64, ActiveStream>,
    stepwise_permits: &mut BTreeMap<u64, AdmissionPermit>,
    service_state: &ServiceState,
) -> SessionResult<EngineStepReport> {
    maintain_streams(session, active_streams, service_state);
    let execution_target = service_state.execution_target.read().clone();
    // When a sibling is active, cap engine-step burst so one model cannot
    // flood Metal with a full STREAM_ENGINE_STEP_BURST of quanta.
    //
    // Exclusive (max=1): large multi-step holds HOL the single device turn.
    // Concurrent (max>1): dual holds still share one GPU — an uncapped
    // prefill worker that loops 64 steps × 64-token quanta dumps ~4k tokens
    // of Gemma work into the Metal queue and was measured to blow S1 gap
    // p95 to ~350–500 ms (concurrent-rotating 2026-07-24). Fair prefill
    // quanta alone are not enough; the worker-level burst must also stay
    // under the stream-gap SLO so Qwen decode kernels keep getting airtime.
    let mut sibling_active_for_burst = false;
    if let Some(target) = execution_target.as_ref().filter(|_| {
        service_state
            .adaptive_prefill_isolation
            .load(Ordering::Acquire)
    }) {
        let sibling_active = target.arbiter.has_recent_sibling_activity(
            target.model_id.as_ref(),
            ADAPTIVE_PREFILL_SIBLING_ACTIVITY_GRACE,
        );
        // Cap burst whenever a sibling is active (exclusive or concurrent).
        sibling_active_for_burst = sibling_active;
        let (enabled, current_tokens, inflight) = session.multi_prefill_policy();
        if sibling_active {
            // Feedback-control quantum from last turn's runner wall time so
            // one prefill chunk stays under the stream-gap SLO mid-prompt
            // (fixed large quanta blow gap late in long Gemma prefills).
            let last_us = service_state
                .last_step_runner_time_us
                .load(Ordering::Acquire);
            let last_toks = service_state
                .last_step_scheduled_tokens
                .load(Ordering::Acquire);
            let current = service_state
                .adaptive_prefill_tokens
                .load(Ordering::Acquire)
                .max(ADAPTIVE_PREFILL_MIN_TOKENS);
            // Only re-size from multi-token steps (prefill quanta). Single-token
            // decode steps have a different cost model and would otherwise push
            // the quantum to MAX after a cheap 1-token decode.
            let adjusted = if last_toks >= 2 {
                adjust_adaptive_prefill_tokens_with_work(current, last_us, last_toks)
            } else {
                current
            };
            service_state
                .adaptive_prefill_tokens
                .store(adjusted, Ordering::Release);
            // Optional exclusive window (AX_SERVER_LONG_PREFILL_EXCLUSIVE=1):
            // force single-hold after multi-token quanta for gap-first isolation.
            // Default off so dual-hold can hide interactive decode under long
            // Gemma prefill (S1 thr). Adaptive quantum sizes the turn for gap.
            target.arbiter.mark_long_prefill_quantum();
            if !enabled || current_tokens != adjusted {
                session.set_multi_prefill_fair(true, adjusted, inflight);
            }
        } else {
            // Sibling idle: restore single-model prefill throughput.
            let start = adaptive_prefill_latency_tokens_per_step();
            service_state
                .adaptive_prefill_tokens
                .store(start, Ordering::Release);
            if enabled {
                session.set_multi_prefill_fair(false, 0, inflight);
            }
        }
    }
    let _turn = execution_target.as_ref().map(|target| {
        target
            .arbiter
            .acquire(target.model_id.as_ref(), ExecutionWorkClass::EngineStep)
    });
    match session.step_report_with_request_ids() {
        Ok((report, request_ids)) => {
            record_step_report(service_state, &report);
            apply_step_to_streams(
                session,
                active_streams,
                &request_ids,
                &report,
                service_state,
            );
            // Single interactive stream: burst multiple engine steps per
            // worker tick so the direct pipeline is not paced by the SSE
            // consumer. Emit batching remains STREAM_TOKEN_EMIT_BATCH for the
            // 50 ms gap cap; engine burst is larger to keep GPU fed.
            // Under multi-model load (sibling active), use a small sibling
            // burst so the Metal queue is not flooded with prefill quanta
            // and the interactive stream keeps gap headroom (see
            // sibling_engine_step_burst).
            let engine_burst = if sibling_active_for_burst {
                sibling_engine_step_burst()
            } else {
                STREAM_ENGINE_STEP_BURST
            };
            if active_streams.len() == 1 && request_ids.len() == 1 && engine_burst > 1 {
                let request_id = request_ids[0];
                for _ in 1..engine_burst {
                    if !active_streams.contains_key(&request_id) {
                        break;
                    }
                    match session.step_report_with_request_ids() {
                        Ok((report, ids)) => {
                            record_step_report(service_state, &report);
                            apply_step_to_streams(
                                session,
                                active_streams,
                                &ids,
                                &report,
                                service_state,
                            );
                        }
                        Err(_) => break,
                    }
                }
            }
            release_terminal_stepwise_permits(session, stepwise_permits, service_state);
            Ok(report)
        }
        Err(error) => {
            tracing::error!(%error, "shared native generation step failed");
            for (request_id, stream) in active_streams.iter_mut() {
                detach_stream_with_error(
                    session,
                    *request_id,
                    stream,
                    EngineSessionError::RequestReportInvariantViolation {
                        request_id: *request_id,
                        message: "shared native generation step failed",
                    },
                    service_state,
                );
            }
            let detached_streams = active_streams.len();
            active_streams.clear();
            for _ in 0..detached_streams {
                complete_job(service_state);
            }
            cancel_all_stepwise(session, stepwise_permits, service_state);
            Err(error)
        }
    }
}

fn apply_step_to_streams(
    session: &mut EngineSession,
    active_streams: &mut BTreeMap<u64, ActiveStream>,
    request_ids: &[u64],
    report: &EngineStepReport,
    service_state: &ServiceState,
) {
    let mut completed = Vec::new();
    for request_id in request_ids {
        let Some(stream) = active_streams.get_mut(request_id) else {
            continue;
        };
        if stream.request_event_pending {
            continue;
        }
        let event =
            match session.next_native_stream_event_after_step(&mut stream.state, report.clone()) {
                Ok(event) => event,
                Err(error) => {
                    detach_stream_with_error(session, *request_id, stream, error, service_state);
                    completed.push(*request_id);
                    continue;
                }
            };
        // Coalesce token steps: first token flushes for TTFT; later tokens wait
        // for STREAM_TOKEN_EMIT_BATCH (or a terminal state) before enqueue so
        // the SSE path is not paced by one frame per token.
        let (event, terminal) = match event {
            GenerateStreamEvent::Step(step) => {
                let step_terminal = request_state_is_terminal(step.request.state);
                if step.delta_tokens.is_empty() && !step_terminal {
                    // Progress-only step with no new tokens; still forward.
                    (GenerateStreamEvent::Step(step), step_terminal)
                } else {
                    stream.pending_delta_tokens.extend(step.delta_tokens);
                    stream
                        .pending_delta_logprobs
                        .extend(step.delta_token_logprobs);
                    stream.pending_step = Some(step.step);
                    stream.pending_request = Some(step.request);
                    let should_flush = step_terminal
                        || !stream.first_output_emitted
                        || stream.pending_delta_tokens.len() >= STREAM_TOKEN_EMIT_BATCH;
                    if !should_flush {
                        continue;
                    }
                    stream.first_output_emitted = true;
                    let delta_tokens = std::mem::take(&mut stream.pending_delta_tokens);
                    let delta_token_logprobs = std::mem::take(&mut stream.pending_delta_logprobs);
                    let step_report = stream
                        .pending_step
                        .take()
                        .expect("pending step set when flushing stream batch");
                    let request = stream
                        .pending_request
                        .take()
                        .expect("pending request set when flushing stream batch");
                    (
                        GenerateStreamEvent::Step(GenerateStreamStepEvent {
                            request,
                            step: step_report,
                            delta_tokens,
                            delta_token_logprobs,
                            delta_text: None,
                        }),
                        step_terminal,
                    )
                }
            }
            other => {
                let is_response = matches!(other, GenerateStreamEvent::Response(_));
                (other, is_response)
            }
        };
        match enqueue_stream_event(stream, Ok(event), service_state) {
            EnqueueResult::Queued => {}
            EnqueueResult::Closed => {
                let _ = session.cancel_request(*request_id);
                stream.permit.take();
                completed.push(*request_id);
                continue;
            }
            EnqueueResult::Overflow => {
                record_pressure_event(
                    service_state,
                    GenerationPressureEvent::StreamBacklogOverflow,
                );
                detach_stream_with_error(
                    session,
                    *request_id,
                    stream,
                    EngineSessionError::RequestReportInvariantViolation {
                        request_id: *request_id,
                        message: "stream consumer exceeded the bounded worker backlog",
                    },
                    service_state,
                );
                completed.push(*request_id);
                continue;
            }
        }
        if terminal {
            let response_event = match session.next_stream_event(&mut stream.state) {
                Ok(Some(event @ GenerateStreamEvent::Response(_))) => Ok(event),
                Ok(_) => Err(EngineSessionError::RequestReportInvariantViolation {
                    request_id: *request_id,
                    message: "terminal native stream did not produce a response event",
                }),
                Err(error) => Err(error),
            };
            detach_terminal_events(stream, response_event, true, service_state);
            completed.push(*request_id);
        }
    }
    for request_id in completed {
        active_streams.remove(&request_id);
        complete_job(service_state);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamDelivery {
    Queued,
    Backpressured,
    Closed,
}

enum EnqueueResult {
    Queued,
    Closed,
    Overflow,
}

fn flush_pending_events(stream: &mut ActiveStream, state: &ServiceState) -> StreamDelivery {
    let mut delivered = false;
    while let Some(event) = stream.pending_events.pop_front() {
        decrement_buffered_stream_events(state, 1);
        match stream.events.try_send(event) {
            Ok(()) => delivered = true,
            Err(mpsc::error::TrySendError::Full(event)) => {
                stream.pending_events.push_front(event);
                increment_buffered_stream_events(state);
                return if delivered {
                    StreamDelivery::Queued
                } else {
                    StreamDelivery::Backpressured
                };
            }
            Err(mpsc::error::TrySendError::Closed(_)) => return StreamDelivery::Closed,
        }
    }
    if delivered {
        StreamDelivery::Queued
    } else {
        StreamDelivery::Backpressured
    }
}

fn enqueue_stream_event(
    stream: &mut ActiveStream,
    event: NativeEvent,
    state: &ServiceState,
) -> EnqueueResult {
    if stream.pending_events.is_empty() {
        match stream.events.try_send(event) {
            Ok(()) => return EnqueueResult::Queued,
            Err(mpsc::error::TrySendError::Closed(_)) => return EnqueueResult::Closed,
            Err(mpsc::error::TrySendError::Full(event)) => {
                if stream.pending_events.len() >= STREAM_WORKER_BACKLOG_CAPACITY {
                    return EnqueueResult::Overflow;
                }
                stream.pending_events.push_back(event);
                increment_buffered_stream_events(state);
                return EnqueueResult::Queued;
            }
        }
    }
    if stream.pending_events.len() >= STREAM_WORKER_BACKLOG_CAPACITY {
        return EnqueueResult::Overflow;
    }
    stream.pending_events.push_back(event);
    increment_buffered_stream_events(state);
    EnqueueResult::Queued
}

fn detach_terminal_events(
    stream: &mut ActiveStream,
    terminal_event: NativeEvent,
    preserve_pending: bool,
    state: &ServiceState,
) {
    let pending_count = stream.pending_events.len();
    let mut terminal_events = stream.terminal_events.lock();
    if preserve_pending {
        terminal_events.extend(stream.pending_events.drain(..));
    } else {
        stream.pending_events.clear();
    }
    decrement_buffered_stream_events(state, pending_count);
    terminal_events.push_back(terminal_event);
    stream.permit.take();
}

fn discard_pending_events(stream: &mut ActiveStream, state: &ServiceState) {
    let pending_count = stream.pending_events.len();
    stream.pending_events.clear();
    decrement_buffered_stream_events(state, pending_count);
}

fn detach_stream_with_error(
    session: &mut EngineSession,
    request_id: u64,
    stream: &mut ActiveStream,
    error: EngineSessionError,
    state: &ServiceState,
) {
    let _ = session.cancel_request(request_id);
    detach_terminal_events(stream, Err(error), false, state);
}

fn detach_all_streams_for_shutdown(
    session: &mut EngineSession,
    active_streams: &mut BTreeMap<u64, ActiveStream>,
    service_state: &ServiceState,
) {
    for (request_id, mut stream) in std::mem::take(active_streams) {
        detach_stream_with_error(
            session,
            request_id,
            &mut stream,
            EngineSessionError::RequestReportInvariantViolation {
                request_id,
                message: "native generation worker shut down",
            },
            service_state,
        );
        complete_job(service_state);
    }
}

fn release_terminal_stepwise_permits(
    session: &EngineSession,
    stepwise_permits: &mut BTreeMap<u64, AdmissionPermit>,
    service_state: &ServiceState,
) {
    let terminal_request_ids = stepwise_permits
        .keys()
        .copied()
        .filter(|request_id| {
            session
                .request_report(*request_id)
                .is_some_and(|report| request_state_is_terminal(report.state))
        })
        .collect::<Vec<_>>();
    for request_id in terminal_request_ids {
        if stepwise_permits.remove(&request_id).is_some() {
            complete_job(service_state);
            record_stepwise_terminal(service_state, request_id);
        }
    }
}

fn release_terminal_stepwise_permit(
    report: &SessionRequestReport,
    stepwise_permits: &mut BTreeMap<u64, AdmissionPermit>,
    service_state: &ServiceState,
) {
    if request_state_is_terminal(report.state)
        && stepwise_permits.remove(&report.request_id).is_some()
    {
        complete_job(service_state);
        record_stepwise_terminal(service_state, report.request_id);
    }
}

fn cancel_all_stepwise(
    session: &mut EngineSession,
    stepwise_permits: &mut BTreeMap<u64, AdmissionPermit>,
    service_state: &ServiceState,
) {
    let request_ids = stepwise_permits.keys().copied().collect::<Vec<_>>();
    for request_id in request_ids {
        let _ = session.cancel_request(request_id);
        if stepwise_permits.remove(&request_id).is_some() {
            complete_job(service_state);
            record_stepwise_terminal(service_state, request_id);
        }
    }
}

fn request_state_is_terminal(state: SessionRequestState) -> bool {
    matches!(
        state,
        SessionRequestState::Finished
            | SessionRequestState::Cancelled
            | SessionRequestState::Failed
    )
}

fn complete_job(state: &ServiceState) {
    if state
        .pending_jobs
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
            pending.checked_sub(1)
        })
        .is_err()
    {
        tracing::error!("native generation pending-job counter underflow");
    }
}

fn begin_command(state: &ServiceState) {
    if state
        .queued_commands
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
            queued.checked_sub(1)
        })
        .is_err()
    {
        tracing::error!("native generation queued-command counter underflow");
    }
}

fn rollback_failed_enqueue(state: &ServiceState) {
    // WorkerExitGuard may reset both counters before send observes the closed receiver.
    let _ = state
        .queued_commands
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
            queued.checked_sub(1)
        });
    let _ = state
        .pending_jobs
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
            pending.checked_sub(1)
        });
}

fn record_step_report(state: &ServiceState, report: &EngineStepReport) {
    // Prefer runner_time_us (GPU/host model work); fall back to cpu_time_us.
    let wall_us = if report.runner_time_us > 0 {
        report.runner_time_us
    } else {
        report.cpu_time_us
    };
    if wall_us > 0 {
        state
            .last_step_runner_time_us
            .store(wall_us, Ordering::Release);
        state
            .last_step_scheduled_tokens
            .store(report.scheduled_tokens.max(1), Ordering::Release);
    }
    let observer = state.step_observer.read().clone();
    if let Some(observer) = observer {
        observer(report);
    }
}

fn record_pressure_event(state: &ServiceState, event: GenerationPressureEvent) {
    let observer = state.pressure_observer.read().clone();
    if let Some(observer) = observer {
        observer(event);
    }
}

fn record_stepwise_terminal(state: &ServiceState, request_id: u64) {
    let observer = state.stepwise_terminal_observer.read().clone();
    if let Some(observer) = observer {
        observer(request_id);
    }
}

fn update_stream_gauges(state: &ServiceState, active_streams: &BTreeMap<u64, ActiveStream>) {
    state
        .active_streams
        .store(active_streams.len(), Ordering::Release);
}

fn increment_buffered_stream_events(state: &ServiceState) {
    state.buffered_stream_events.fetch_add(1, Ordering::AcqRel);
}

fn decrement_buffered_stream_events(state: &ServiceState, count: usize) {
    if count == 0 {
        return;
    }
    if state
        .buffered_stream_events
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |buffered| {
            buffered.checked_sub(count)
        })
        .is_err()
    {
        tracing::error!(count, "native generation buffered-event counter underflow");
    }
}

struct WorkerExitGuard<'a>(&'a ServiceState);

impl Drop for WorkerExitGuard<'_> {
    fn drop(&mut self) {
        let pending = self.0.pending_jobs.load(Ordering::Acquire);
        let queued = self.0.queued_commands.load(Ordering::Acquire);
        let streams = self.0.active_streams.load(Ordering::Acquire);
        let buffered = self.0.buffered_stream_events.load(Ordering::Acquire);
        if pending != 0 || queued != 0 || streams != 0 || buffered != 0 {
            tracing::warn!(
                pending_jobs = pending,
                queued_commands = queued,
                active_streams = streams,
                buffered_stream_events = buffered,
                "native generation worker exiting with non-zero counters"
            );
        }
        self.0.pending_jobs.store(0, Ordering::Release);
        self.0.queued_commands.store(0, Ordering::Release);
        self.0.active_streams.store(0, Ordering::Release);
        self.0.buffered_stream_events.store(0, Ordering::Release);
        self.0.alive.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::mpsc as std_mpsc;
    use std::time::{Duration, Instant};

    use ax_engine_sdk::{
        EngineSessionConfig, PreviewBackendRequest, PreviewSessionConfigRequest, SupportTier,
    };

    use super::*;

    #[test]
    fn sibling_engine_step_burst_defaults_between_one_and_full_burst() {
        // Exclusive multi-model: not 1 (arbiter thr tax) and not full HOL burst.
        assert!(SIBLING_ENGINE_STEP_BURST_DEFAULT >= 2);
        assert!(SIBLING_ENGINE_STEP_BURST_DEFAULT < STREAM_ENGINE_STEP_BURST);
        assert_eq!(SIBLING_ENGINE_STEP_BURST_DEFAULT, 4);
    }

    #[test]
    fn adaptive_prefill_latency_quantum_defaults_to_wall_time_slo_proxy() {
        // Must not regress to the historical 1-token sibling quantum that
        // serialized long prefills into thousands of arbiter turns (flip S1).
        assert_eq!(ADAPTIVE_PREFILL_LATENCY_TOKENS_PER_STEP_DEFAULT, 64);
        assert_eq!(resolve_adaptive_prefill_latency_tokens(None), 64);
        assert_eq!(resolve_adaptive_prefill_latency_tokens(Some("")), 64);
        assert_eq!(resolve_adaptive_prefill_latency_tokens(Some("0")), 64);
        assert_eq!(resolve_adaptive_prefill_latency_tokens(Some("nope")), 64);
        assert_eq!(resolve_adaptive_prefill_latency_tokens(Some("96")), 96);
        assert_eq!(resolve_adaptive_prefill_latency_tokens(Some("1")), 1);
        const {
            assert!(
                ADAPTIVE_PREFILL_LATENCY_TOKENS_PER_STEP_DEFAULT
                    < ADAPTIVE_PREFILL_THROUGHPUT_TOKENS_PER_STEP
            );
            assert!(
                ADAPTIVE_PREFILL_LATENCY_TOKENS_PER_STEP_DEFAULT > 1,
                "must not regress to the 1-token pathological sibling quantum"
            );
        }
    }

    #[test]
    fn adjust_adaptive_prefill_tokens_targets_gap_slo_from_us_per_tok() {
        // No measurement yet → hold.
        assert_eq!(adjust_adaptive_prefill_tokens(16, 0), 16);
        // 16 tokens took 16 ms → 1 ms/tok → target ≈ SLO/1ms; blend up from 16.
        let up = adjust_adaptive_prefill_tokens_with_work(16, 16_000, 16);
        let slo_tok = (ADAPTIVE_PREFILL_GAP_SLO_US / 1_000) as u32;
        assert!(up > 16 && up <= slo_tok, "up={up} slo_tok={slo_tok}");
        // Over budget: snap to target (16 tokens / 80 ms → 5 ms/tok → target 8 @ 40ms SLO).
        assert_eq!(
            adjust_adaptive_prefill_tokens_with_work(16, 80_000, 16),
            (ADAPTIVE_PREFILL_GAP_SLO_US / 5_000) as u32
        );
        // Very expensive → snap to 1.
        assert_eq!(adjust_adaptive_prefill_tokens_with_work(4, 200_000, 4), 1);
        // Cap at max.
        let capped = adjust_adaptive_prefill_tokens_with_work(100, 1_000, 100);
        assert!(capped <= ADAPTIVE_PREFILL_MAX_TOKENS);
    }

    #[test]
    fn long_prefill_exclusive_defaults_on_for_gap_isolation() {
        // Dual-hold without exclusive failed S1 gap; default stays isolation-on.
        assert!(long_prefill_exclusive_enabled());
    }

    fn delegated_config() -> EngineSessionConfig {
        EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::LlamaCpp,
                llama_model_path: Some(PathBuf::from("fake-model.gguf")),
                ..PreviewBackendRequest::default()
            },
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview session config should build")
    }

    fn delegated_service() -> Arc<NativeGenerationService> {
        NativeGenerationService::spawn(delegated_config())
            .expect("service should start")
            .0
    }

    #[tokio::test]
    async fn worker_panic_is_contained_and_marks_unavailable() {
        let service = delegated_service();
        assert!(service.is_ready());

        let result = service
            .execute::<(), _>(|_session| panic!("injected engine panic"))
            .await;
        assert!(
            matches!(result, Err(GenerationServiceError::Unavailable)),
            "a panicking job must fail unavailable, got {result:?}"
        );

        // The process survives; the worker retires and reports not-ready.
        // The response channel disconnects during the unwind, before the
        // worker's exit guard flips `alive`, so bound-poll the transition.
        for _ in 0..200 {
            if !service.is_ready() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(!service.is_ready());
        let followup = service
            .execute(|session| Ok(session.runtime_report().selected_backend))
            .await;
        assert!(
            matches!(followup, Err(GenerationServiceError::Unavailable)),
            "post-panic submissions must fail unavailable, got {followup:?}"
        );

        // The worker thread exits normally (the panic was caught), so
        // shutdown joins cleanly instead of surfacing a panicked join.
        service
            .shutdown()
            .await
            .expect("panicked worker should shut down cleanly");
    }

    #[tokio::test]
    async fn worker_retains_one_session_across_commands() {
        let service = delegated_service();
        let first = service
            .execute(|session| Ok(session.runtime_report().selected_backend))
            .await
            .expect("first command should run");
        let second = service
            .execute(|session| Ok(session.runtime_report().selected_backend))
            .await
            .expect("second command should run");

        assert_eq!(first, second);
        assert!(service.is_ready());
        assert!(!service.is_busy());
    }

    #[tokio::test]
    async fn worker_constructs_and_executes_session_on_same_thread() {
        let construction_thread = Arc::new(parking_lot::Mutex::new(None));
        let recorded_thread = Arc::clone(&construction_thread);
        let (service, _) = NativeGenerationService::spawn_with_factory(move || {
            *recorded_thread.lock() = Some(std::thread::current().id());
            EngineSession::new(delegated_config())
        })
        .expect("service should start");

        let execution_thread = service
            .execute(|_| Ok(std::thread::current().id()))
            .await
            .expect("worker command should run");

        assert_eq!(
            construction_thread
                .lock()
                .as_ref()
                .expect("construction thread should be recorded"),
            &execution_thread
        );
        service.shutdown().await.expect("worker should shut down");
    }

    #[test]
    fn worker_startup_propagates_session_error() {
        let result = NativeGenerationService::spawn_with_factory(|| {
            Err(EngineSessionError::InvalidMaxBatchTokens)
        });

        assert!(matches!(
            result,
            Err(GenerationServiceStartError::Engine(
                EngineSessionError::InvalidMaxBatchTokens
            ))
        ));
    }

    #[tokio::test]
    async fn worker_shutdown_closes_submission_and_joins() {
        let service = delegated_service();

        service.shutdown().await.expect("worker should shut down");

        assert!(!service.is_ready());
        assert!(matches!(
            service.execute(|_| Ok(())).await,
            Err(GenerationServiceError::Unavailable)
        ));
    }

    #[tokio::test]
    async fn detached_terminal_events_follow_channel_events() {
        let (sender, receiver) = mpsc::channel(1);
        let terminal_events = Arc::new(parking_lot::Mutex::new(VecDeque::new()));
        let error = |message| {
            Err(EngineSessionError::RequestReportInvariantViolation {
                request_id: 7,
                message,
            })
        };
        sender
            .try_send(error("queued event"))
            .expect("channel should accept the queued event");
        terminal_events.lock().push_back(error("terminal event"));
        drop(sender);
        let mut events = NativeEventReceiver {
            receiver,
            terminal_events,
        };

        let queued = events
            .recv()
            .await
            .expect("queued event should remain")
            .expect_err("queued event should be the test error");
        let terminal = events
            .recv()
            .await
            .expect("terminal event should remain")
            .expect_err("terminal event should be the test error");

        assert!(queued.to_string().contains("queued event"));
        assert!(terminal.to_string().contains("terminal event"));
        assert!(events.recv().await.is_none());
    }

    #[tokio::test]
    async fn successful_engine_steps_notify_the_observer_once() {
        let service = delegated_service();
        let observed = Arc::new(AtomicUsize::new(0));
        let observer_count = Arc::clone(&observed);
        service.set_step_observer(move |_| {
            observer_count.fetch_add(1, Ordering::Relaxed);
        });

        service
            .advance()
            .await
            .expect("delegated idle step should succeed");

        assert_eq!(observed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn worker_command_queue_rejects_unbounded_growth() {
        let service = delegated_service();
        let saturated = Arc::new(AtomicUsize::new(0));
        let saturated_count = Arc::clone(&saturated);
        service.set_pressure_observer(move |event| {
            if event == GenerationPressureEvent::CommandSaturated {
                saturated_count.fetch_add(1, Ordering::Relaxed);
            }
        });
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
            .expect("worker should enter the blocking command");

        for _ in 0..COMMAND_QUEUE_CAPACITY {
            service
                .submit(|_| {})
                .expect("bounded queue should accept commands up to capacity");
        }
        assert_eq!(service.queued_commands(), COMMAND_QUEUE_CAPACITY);
        assert_eq!(service.pending_jobs(), COMMAND_QUEUE_CAPACITY + 1);
        assert!(matches!(
            service.submit(|_| {}),
            Err(GenerationServiceError::Saturated)
        ));
        assert_eq!(service.queued_commands(), COMMAND_QUEUE_CAPACITY);
        assert_eq!(saturated.load(Ordering::Relaxed), 1);
        assert_eq!(service.active_streams(), 0);
        assert_eq!(service.buffered_stream_events(), 0);

        release_tx.send(()).expect("worker should be released");
        let deadline = Instant::now() + Duration::from_secs(1);
        while service.is_busy() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(service.queued_commands(), 0);
        assert_eq!(service.pending_jobs(), 0);
    }

    #[test]
    fn execution_arbiter_concurrent_allows_distinct_models() {
        let arbiter = Arc::new(ModelExecutionArbiter::with_max_concurrent(2));
        let alpha = arbiter.acquire("alpha", ExecutionWorkClass::EngineStep);
        // Distinct model must enter without waiting for alpha to drop.
        let beta = arbiter.acquire("beta", ExecutionWorkClass::EngineStep);
        assert_eq!(arbiter.max_concurrent(), 2);
        assert!(arbiter.has_recent_sibling_activity("alpha", Duration::from_secs(1)));
        assert!(arbiter.has_recent_sibling_activity("beta", Duration::from_secs(1)));
        drop(alpha);
        drop(beta);
    }

    #[test]
    fn execution_arbiter_rotates_between_waiting_models() {
        // Exclusive default: round-robin fairness on a single device slot.
        let arbiter = Arc::new(ModelExecutionArbiter::with_max_concurrent(1));
        let first_turn = arbiter.acquire("alpha", ExecutionWorkClass::EngineStep);
        let (acquired_tx, acquired_rx) = std_mpsc::channel();
        let mut workers = Vec::new();
        for model_id in ["alpha", "beta"] {
            let worker_arbiter = Arc::clone(&arbiter);
            let acquired_tx = acquired_tx.clone();
            workers.push(std::thread::spawn(move || {
                let _turn = worker_arbiter.acquire(model_id, ExecutionWorkClass::EngineStep);
                acquired_tx
                    .send(model_id)
                    .expect("acquisition should be observed");
            }));
        }
        drop(acquired_tx);

        let deadline = Instant::now() + Duration::from_secs(1);
        while arbiter.state.lock().waiters.len() < 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(arbiter.state.lock().waiters.len(), 2);
        drop(first_turn);

        assert_eq!(
            acquired_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("the next model should acquire"),
            "beta"
        );
        assert_eq!(
            acquired_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("the original model should reacquire"),
            "alpha"
        );
        for worker in workers {
            worker.join().expect("arbiter worker should finish");
        }
        let stats = arbiter.stats();
        let alpha = stats
            .iter()
            .find(|(model_id, work_class, _)| {
                model_id == "alpha" && *work_class == ExecutionWorkClass::EngineStep
            })
            .map(|(_, _, stats)| stats)
            .expect("alpha stats should exist");
        assert_eq!(alpha.turns_total, 2);
        assert!(alpha.hold_us_total >= alpha.hold_us_max);
    }

    #[test]
    fn execution_arbiter_reports_and_forgets_recent_sibling_activity() {
        let arbiter = ModelExecutionArbiter::default();
        assert!(!arbiter.has_recent_sibling_activity("alpha", Duration::from_secs(1)));

        let beta_turn = arbiter.acquire("beta", ExecutionWorkClass::EngineStep);
        assert!(arbiter.has_recent_sibling_activity("alpha", Duration::from_secs(1)));
        assert!(!arbiter.has_recent_sibling_activity("beta", Duration::from_secs(1)));
        drop(beta_turn);

        assert!(arbiter.has_recent_sibling_activity("alpha", Duration::from_secs(1)));
        arbiter.remove_model("beta");
        assert!(!arbiter.has_recent_sibling_activity("alpha", Duration::from_secs(1)));
    }

    #[test]
    fn failed_enqueue_rollback_tolerates_worker_exit_reset() {
        let state = ServiceState {
            alive: AtomicBool::new(false),
            pending_jobs: AtomicUsize::new(1),
            queued_commands: AtomicUsize::new(1),
            active_streams: AtomicUsize::new(0),
            buffered_stream_events: AtomicUsize::new(0),
            step_observer: parking_lot::RwLock::new(None),
            pressure_observer: parking_lot::RwLock::new(None),
            stepwise_terminal_observer: parking_lot::RwLock::new(None),
            execution_target: parking_lot::RwLock::new(None),
            adaptive_prefill_isolation: AtomicBool::new(false),
            last_step_runner_time_us: AtomicU64::new(0),
            last_step_scheduled_tokens: AtomicU32::new(0),
            adaptive_prefill_tokens: AtomicU32::new(
                ADAPTIVE_PREFILL_LATENCY_TOKENS_PER_STEP_DEFAULT,
            ),
        };

        rollback_failed_enqueue(&state);
        assert_eq!(state.queued_commands.load(Ordering::Acquire), 0);
        assert_eq!(state.pending_jobs.load(Ordering::Acquire), 0);

        rollback_failed_enqueue(&state);
        assert_eq!(state.queued_commands.load(Ordering::Acquire), 0);
        assert_eq!(state.pending_jobs.load(Ordering::Acquire), 0);
    }

    #[test]
    fn lifecycle_command_overtakes_queued_bulk_work() {
        let service = delegated_service();
        let (first_entered_tx, first_entered_rx) = std_mpsc::channel();
        let (release_first_tx, release_first_rx) = std_mpsc::channel();
        service
            .submit(move |_| {
                let _ = first_entered_tx.send(());
                let _ = release_first_rx.recv();
            })
            .expect("first bulk command should enqueue");
        first_entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker should enter the first bulk command");

        let (second_entered_tx, second_entered_rx) = std_mpsc::channel();
        service
            .submit(move |_| {
                let _ = second_entered_tx.send(());
            })
            .expect("second bulk command should enqueue");
        let (lifecycle_tx, lifecycle_rx) = oneshot::channel();
        service
            .enqueue(ServiceCommand::HasActiveStepwise {
                response: lifecycle_tx,
            })
            .expect("lifecycle command should enqueue");

        release_first_tx
            .send(())
            .expect("first bulk command should be released");
        assert!(
            !lifecycle_rx
                .blocking_recv()
                .expect("lifecycle response should arrive")
        );
        second_entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("second bulk command should eventually run");
    }
}
