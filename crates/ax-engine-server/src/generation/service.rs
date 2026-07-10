use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ax_engine_sdk::{
    EngineSession, EngineSessionError, GenerateRequest, GenerateResponse, GenerateStreamEvent,
    GenerateStreamState,
};
use tokio::sync::{mpsc, oneshot};

use crate::admission::AdmissionPermit;

type SessionJob = Box<dyn FnOnce(&mut EngineSession) + Send + 'static>;
type NativeEvent = Result<GenerateStreamEvent, EngineSessionError>;

enum ServiceCommand {
    Execute(SessionJob),
    StartStream {
        request_id: u64,
        request: GenerateRequest,
        events: mpsc::Sender<NativeEvent>,
        started: oneshot::Sender<Result<(), EngineSessionError>>,
        permit: AdmissionPermit,
    },
}

struct ServiceState {
    alive: AtomicBool,
    pending_jobs: AtomicUsize,
}

pub(crate) struct NativeGenerationService {
    sender: std::sync::mpsc::Sender<ServiceCommand>,
    state: Arc<ServiceState>,
}

impl NativeGenerationService {
    pub(crate) fn spawn(session: EngineSession) -> Arc<Self> {
        let (sender, receiver) = std::sync::mpsc::channel::<ServiceCommand>();
        let state = Arc::new(ServiceState {
            alive: AtomicBool::new(true),
            pending_jobs: AtomicUsize::new(0),
        });
        let worker_state = Arc::clone(&state);
        std::thread::spawn(move || run_worker(session, receiver, &worker_state));
        Arc::new(Self { sender, state })
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
        self.state.pending_jobs.fetch_add(1, Ordering::AcqRel);
        if self
            .sender
            .send(ServiceCommand::Execute(Box::new(operation)))
            .is_err()
        {
            self.state.pending_jobs.fetch_sub(1, Ordering::AcqRel);
            return Err(GenerationServiceError::Unavailable);
        }
        Ok(())
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
    ) -> Result<mpsc::Receiver<NativeEvent>, GenerationServiceError> {
        let (events_tx, events_rx) = mpsc::channel(128);
        let (started_tx, started_rx) = oneshot::channel();
        self.state.pending_jobs.fetch_add(1, Ordering::AcqRel);
        if self
            .sender
            .send(ServiceCommand::StartStream {
                request_id,
                request,
                events: events_tx,
                started: started_tx,
                permit,
            })
            .is_err()
        {
            self.state.pending_jobs.fetch_sub(1, Ordering::AcqRel);
            return Err(GenerationServiceError::Unavailable);
        }
        started_rx
            .await
            .map_err(|_| GenerationServiceError::Unavailable)?
            .map_err(GenerationServiceError::Engine)?;
        Ok(events_rx)
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
}

#[derive(Debug)]
pub(crate) enum GenerationServiceError {
    Engine(EngineSessionError),
    Unavailable,
}

impl fmt::Display for GenerationServiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engine(error) => error.fmt(formatter),
            Self::Unavailable => formatter.write_str("native generation worker is unavailable"),
        }
    }
}

impl std::error::Error for GenerationServiceError {}

fn run_worker(
    mut session: EngineSession,
    receiver: std::sync::mpsc::Receiver<ServiceCommand>,
    state: &ServiceState,
) {
    let _exit_guard = WorkerExitGuard(state);
    let mut active_streams = BTreeMap::new();
    let mut disconnected = false;
    loop {
        if active_streams.is_empty() && !disconnected {
            match receiver.recv() {
                Ok(command) => handle_command(command, &mut session, &mut active_streams, state),
                Err(_) => disconnected = true,
            }
        }
        loop {
            match receiver.try_recv() {
                Ok(command) => handle_command(command, &mut session, &mut active_streams, state),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        let progressed = advance_streams(&mut session, &mut active_streams, state);
        if disconnected && active_streams.is_empty() {
            return;
        }
        if !progressed && !active_streams.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

struct ActiveStream {
    state: GenerateStreamState,
    events: mpsc::Sender<NativeEvent>,
    pending_event: Option<NativeEvent>,
    terminate_after_pending: bool,
    _permit: AdmissionPermit,
}

fn handle_command(
    command: ServiceCommand,
    session: &mut EngineSession,
    active_streams: &mut BTreeMap<u64, ActiveStream>,
    state: &ServiceState,
) {
    match command {
        ServiceCommand::Execute(job) => {
            job(session);
            state.pending_jobs.fetch_sub(1, Ordering::AcqRel);
        }
        ServiceCommand::StartStream {
            request_id,
            request,
            events,
            started,
            permit,
        } => match session.stream_generate_state_with_request_id(request_id, request) {
            Ok(stream_state) => {
                let previous = active_streams.insert(
                    request_id,
                    ActiveStream {
                        state: stream_state,
                        events,
                        pending_event: None,
                        terminate_after_pending: false,
                        _permit: permit,
                    },
                );
                debug_assert!(previous.is_none(), "request IDs are process-unique");
                let _ = started.send(Ok(()));
            }
            Err(error) => {
                let _ = started.send(Err(error));
                state.pending_jobs.fetch_sub(1, Ordering::AcqRel);
            }
        },
    }
}

fn advance_streams(
    session: &mut EngineSession,
    active_streams: &mut BTreeMap<u64, ActiveStream>,
    service_state: &ServiceState,
) -> bool {
    let mut progressed = false;
    let mut terminal = Vec::new();
    for (request_id, stream) in active_streams.iter_mut() {
        if stream.events.is_closed() {
            let _ = session.cancel_request(*request_id);
            terminal.push(*request_id);
            progressed = true;
            continue;
        }
        if let Some(event) = stream.pending_event.take() {
            match stream.events.try_send(event) {
                Ok(()) => {
                    progressed = true;
                    if stream.terminate_after_pending {
                        terminal.push(*request_id);
                    }
                }
                Err(mpsc::error::TrySendError::Full(event)) => {
                    stream.pending_event = Some(event);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    let _ = session.cancel_request(*request_id);
                    terminal.push(*request_id);
                    progressed = true;
                }
            }
            continue;
        }
        progressed = true;
        let event = match session.next_stream_event(&mut stream.state) {
            Ok(Some(event)) => Ok(event),
            Ok(None) => {
                terminal.push(*request_id);
                continue;
            }
            Err(error) => {
                match stream.events.try_send(Err(error)) {
                    Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => {
                        terminal.push(*request_id);
                    }
                    Err(mpsc::error::TrySendError::Full(event)) => {
                        stream.pending_event = Some(event);
                        stream.terminate_after_pending = true;
                    }
                }
                continue;
            }
        };
        match stream.events.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) => stream.pending_event = Some(event),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                let _ = session.cancel_request(*request_id);
                terminal.push(*request_id);
            }
        }
    }
    for request_id in terminal {
        active_streams.remove(&request_id);
        service_state.pending_jobs.fetch_sub(1, Ordering::AcqRel);
    }
    progressed
}

struct WorkerExitGuard<'a>(&'a ServiceState);

impl Drop for WorkerExitGuard<'_> {
    fn drop(&mut self) {
        self.0.alive.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ax_engine_sdk::{
        EngineSessionConfig, PreviewBackendRequest, PreviewSessionConfigRequest, SupportTier,
    };

    use super::*;

    fn delegated_session() -> EngineSession {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::LlamaCpp,
                llama_model_path: Some(PathBuf::from("fake-model.gguf")),
                ..PreviewBackendRequest::default()
            },
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview session config should build");
        EngineSession::new(config).expect("session should build")
    }

    #[tokio::test]
    async fn worker_retains_one_session_across_commands() {
        let service = NativeGenerationService::spawn(delegated_session());
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
}
