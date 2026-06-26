use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ax_engine_sdk::{
    EngineSession, EngineSessionError, GenerateRequest, GenerateStreamEvent, GenerateStreamState,
};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::app_state::{AppState, LiveState};
use crate::errors::{ErrorResponse, map_session_error};
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::openai::validation::validate_model;
use crate::tasks::run_blocking_session_task;

const STREAM_CHANNEL_CAPACITY: usize = 128;

pub(crate) type StreamEvent = Result<Event, Infallible>;
pub(crate) type StreamEventSender = mpsc::Sender<StreamEvent>;
type StreamEventReceiver = mpsc::Receiver<StreamEvent>;

/// Shared flag set when the SSE client disconnects (all receivers dropped).
pub(crate) type StreamCancelFlag = Arc<AtomicBool>;

pub(crate) async fn generate_stream(
    State(state): State<AppState>,
    Json(request): Json<GenerateHttpRequest>,
) -> Result<Sse<ReceiverStream<StreamEvent>>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    validate_model(&live, request.model.as_deref())?;

    let request = build_generate_request(&live, request);
    let (stream_state, stream_context) = build_stream_state(&state, &live, request).await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_stream_task(
        tx,
        stream_state,
        move |stream_state, tx, cancel| match stream_context {
            StreamStateSource::Stateless(context) => {
                drive_generate_stream_state(stream_state, tx, cancel, |state| {
                    context.next_stream_event(state)
                });
            }
            StreamStateSource::Stateful(mut session) => {
                drive_generate_stream_state(stream_state, tx, cancel, |state| {
                    session.next_stream_event(state)
                });
            }
        },
    );

    Ok(build_keep_alive_stream(rx))
}

pub(crate) enum StreamStateSource {
    Stateless(Arc<ax_engine_sdk::StatelessGenerateContext>),
    Stateful(Box<EngineSession>),
}

/// Builds stream state against the caller's `LiveState` snapshot, so the
/// model that validated/tokenized the request is the one that streams it
/// even if a hot-swap lands mid-request.
pub(crate) async fn build_stream_state(
    state: &AppState,
    live: &LiveState,
    request: GenerateRequest,
) -> Result<(GenerateStreamState, StreamStateSource), (StatusCode, Json<ErrorResponse>)> {
    let request_id = state.allocate_request_id();
    if live
        .stateless_generate_context
        .supports_stateless_streaming()
    {
        let stateless_generate_context = Arc::clone(&live.stateless_generate_context);
        let stream_context = Arc::clone(&stateless_generate_context);
        let stream_state = run_blocking_session_task(move || {
            stream_context.stream_state_with_request_id(request_id, request)
        })
        .await?;

        return Ok((
            stream_state,
            StreamStateSource::Stateless(stateless_generate_context),
        ));
    }

    let stateful_context = Arc::clone(&live.stateless_generate_context);
    let (session, stream_state) = run_blocking_session_task(move || {
        let mut session = stateful_context.build_stateful_session()?;
        let stream_state = session.stream_generate_state_with_request_id(request_id, request)?;
        Ok((session, stream_state))
    })
    .await?;

    Ok((stream_state, StreamStateSource::Stateful(Box::new(session))))
}

pub(crate) fn spawn_stream_task<F>(
    tx: StreamEventSender,
    stream_state: GenerateStreamState,
    driver: F,
) where
    F: FnOnce(&mut GenerateStreamState, StreamEventSender, StreamCancelFlag) + Send + 'static,
{
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_monitor = Arc::clone(&cancel);
    let monitor_tx = tx.clone();
    let error_monitor_tx = monitor_tx.clone();
    // Detect client disconnect: when all receivers are dropped the
    // channel closes and we flip the cancel flag for the blocking task.
    let cancel_monitor_handle = tokio::spawn(async move {
        monitor_tx.closed().await;
        cancel_monitor.store(true, Ordering::Relaxed);
    });
    let handle = tokio::task::spawn_blocking(move || {
        let mut stream_state = stream_state;
        driver(&mut stream_state, tx, cancel);
    });
    tokio::spawn(async move {
        let result = handle.await;
        cancel_monitor_handle.abort();
        if let Err(error) = result {
            tracing::error!(%error, "generate stream task failed");
            send_stream_error_async(
                &error_monitor_tx,
                ErrorResponse::server_error(format!("generate stream task failed: {error}")),
            )
            .await;
        }
    });
}

pub(crate) fn spawn_sse_blocking_stream_task<F>(
    tx: StreamEventSender,
    task_name: &'static str,
    driver: F,
) where
    F: FnOnce(StreamEventSender, StreamCancelFlag) + Send + 'static,
{
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_monitor = Arc::clone(&cancel);
    let monitor_tx = tx.clone();
    let error_monitor_tx = monitor_tx.clone();
    let cancel_monitor_handle = tokio::spawn(async move {
        monitor_tx.closed().await;
        cancel_monitor.store(true, Ordering::Relaxed);
    });
    let handle = tokio::task::spawn_blocking(move || driver(tx, cancel));
    tokio::spawn(async move {
        let result = handle.await;
        cancel_monitor_handle.abort();
        if let Err(error) = result {
            tracing::error!(%error, task = task_name, "SSE stream task failed");
            send_stream_error_async(
                &error_monitor_tx,
                ErrorResponse::server_error(format!("{task_name} task failed: {error}")),
            )
            .await;
        }
    });
}

fn drive_generate_stream_state<N>(
    state: &mut GenerateStreamState,
    tx: StreamEventSender,
    cancel: StreamCancelFlag,
    next_event: N,
) where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    drive_stream_events(
        state,
        &tx,
        &cancel,
        next_event,
        |event| send_sdk_stream_event(&tx, event),
        || {},
    );
}

pub(crate) fn drive_stream_events<N, E, D>(
    state: &mut GenerateStreamState,
    tx: &StreamEventSender,
    cancel: &AtomicBool,
    mut next_event: N,
    mut emit_event: E,
    mut on_done: D,
) where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
    E: FnMut(GenerateStreamEvent) -> bool,
    D: FnMut(),
{
    loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::debug!("stream cancelled: client disconnected");
            return;
        }
        match next_event(state) {
            Ok(Some(event)) => {
                if !emit_event(event) {
                    return;
                }
            }
            Ok(None) => {
                on_done();
                return;
            }
            Err(error) => {
                let (_, Json(error)) = map_session_error(error);
                send_stream_error(tx, error);
                return;
            }
        }
    }
}

fn send_sdk_stream_event(tx: &StreamEventSender, event: GenerateStreamEvent) -> bool {
    let event_name = event.event_name();

    match event {
        GenerateStreamEvent::Request(payload) => send_stream_event(tx, event_name, &payload),
        GenerateStreamEvent::Step(payload) => send_stream_event(tx, event_name, &payload),
        GenerateStreamEvent::Response(payload) => send_stream_event(tx, event_name, &payload),
    }
}

fn send_stream_event<T: Serialize>(tx: &StreamEventSender, event_name: &str, payload: &T) -> bool {
    match serde_json::to_string(payload) {
        Ok(data) => tx
            .blocking_send(Ok(Event::default().event(event_name).data(data)))
            .is_ok(),
        Err(error) => {
            send_stream_error(
                tx,
                ErrorResponse::server_error(format!(
                    "failed to serialize {event_name} event: {error}"
                )),
            );
            false
        }
    }
}

pub(crate) fn build_keep_alive_stream(rx: StreamEventReceiver) -> Sse<ReceiverStream<StreamEvent>> {
    Sse::new(ReceiverStream::new(rx)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keep-alive"),
    )
}

pub(crate) fn send_stream_error(tx: &StreamEventSender, error: ErrorResponse) {
    let payload = serde_json::to_string(&error).unwrap_or_else(|_| {
        "{\"error\":{\"type\":\"server_error\",\"code\":\"engine_error\",\"param\":null,\"message\":\"failed to serialize stream error\"}}".to_string()
    });
    let _ = tx.blocking_send(Ok(Event::default().event("error").data(payload)));
    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
}

async fn send_stream_error_async(tx: &StreamEventSender, error: ErrorResponse) {
    let payload = serde_json::to_string(&error).unwrap_or_else(|_| {
        "{\"error\":{\"type\":\"server_error\",\"code\":\"engine_error\",\"param\":null,\"message\":\"failed to serialize stream error\"}}".to_string()
    });
    let _ = tx
        .send(Ok(Event::default().event("error").data(payload)))
        .await;
    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
}
