use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use ax_engine_sdk::{
    EngineSession, EngineSessionConfig, EngineSessionError, EngineStepReport, GenerateRequest,
    GenerateResponse, GenerateStreamEvent, GenerateStreamState, RuntimeReport,
    SessionRequestReport, SessionRequestState,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GenerationPressureEvent {
    CommandSaturated,
    StreamBacklogOverflow,
}

const COMMAND_QUEUE_CAPACITY: usize = 256;
const COMMANDS_PER_TICK: usize = 64;
const BULK_COMMANDS_PER_ACTIVE_TICK: usize = 1;
const STREAM_EVENT_CHANNEL_CAPACITY: usize = 128;
const STREAM_WORKER_BACKLOG_CAPACITY: usize = 8;

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
    let mut session = match factory() {
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
        let progressed = if engine_advanced {
            true
        } else if !active_streams.is_empty() {
            let _ = advance_shared_engine(
                &mut session,
                &mut active_streams,
                &mut stepwise_permits,
                state,
            );
            true
        } else {
            maintain_streams(&mut session, &mut active_streams, state)
        };
        update_stream_gauges(state, &active_streams);
        if !progressed && !active_streams.is_empty() {
            std::thread::sleep(Duration::from_millis(1));
        }
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
        let terminal = matches!(
            &event,
            GenerateStreamEvent::Step(step)
                if request_state_is_terminal(step.request.state)
        );
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
    fn failed_enqueue_rollback_tolerates_worker_exit_reset() {
        let state = ServiceState {
            alive: AtomicBool::new(false),
            pending_jobs: AtomicUsize::new(1),
            queued_commands: AtomicUsize::new(1),
            active_streams: AtomicUsize::new(0),
            buffered_stream_events: AtomicUsize::new(0),
            step_observer: parking_lot::RwLock::new(None),
            pressure_observer: parking_lot::RwLock::new(None),
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
