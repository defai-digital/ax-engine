use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

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

    Ok(build_keep_alive_stream(rx, state.limits.stream_deadlines))
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

/// Idle-between-events and hard max-duration deadlines for a streaming
/// response. Both default to disabled (`None`), which preserves the
/// "only ends on client disconnect or producer completion" behavior exactly.
#[derive(Clone, Copy, Default)]
pub(crate) struct StreamDeadlines {
    pub(crate) idle_timeout: Option<Duration>,
    pub(crate) max_duration: Option<Duration>,
}

pub(crate) fn build_keep_alive_stream(
    rx: StreamEventReceiver,
    deadlines: StreamDeadlines,
) -> Sse<ReceiverStream<StreamEvent>> {
    let rx = match (deadlines.idle_timeout, deadlines.max_duration) {
        (None, None) => rx,
        _ => spawn_deadline_relay(rx, deadlines),
    };
    Sse::new(ReceiverStream::new(rx)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keep-alive"),
    )
}

/// Bound on a single send into the relay channel. This is independent of
/// `StreamDeadlines` — it targets a *stalled consumer* (a client that has
/// stopped reading without closing the connection: a network stall, a
/// backgrounded mobile client, or a deliberately slow reader) rather than
/// producer idleness. Without this bound, `relay_tx.send(event).await` on a
/// full, undrained channel blocks indefinitely and the loop never returns
/// to check `deadlines` at all — silently defeating both
/// `--stream-idle-timeout-secs` and `--stream-max-duration-secs`.
const RELAY_SEND_TIMEOUT: Duration = Duration::from_secs(30);

/// Relays events from `rx` to a fresh channel, enforcing `deadlines`. On
/// expiry, sends one synthetic error event + `[DONE]` (matching
/// `send_stream_error`'s shape) and returns, dropping the original `rx` —
/// the producer's existing `tx.closed()` disconnect monitor (see
/// `spawn_stream_task`) already watches that, so cancellation is reused for
/// free with no changes needed on the producer side.
fn spawn_deadline_relay(
    mut rx: StreamEventReceiver,
    deadlines: StreamDeadlines,
) -> StreamEventReceiver {
    let (relay_tx, relay_rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    tokio::spawn(async move {
        let start = Instant::now();
        loop {
            let max_duration_remaining = deadlines
                .max_duration
                .map(|max| max.saturating_sub(start.elapsed()));
            if max_duration_remaining == Some(Duration::ZERO) {
                // Best-effort only: if the consumer is stalled this may also
                // time out, in which case we just give up below regardless.
                let _ = tokio::time::timeout(
                    RELAY_SEND_TIMEOUT,
                    send_stream_error_async(
                        &relay_tx,
                        ErrorResponse::server_error(
                            "stream exceeded the configured maximum duration".to_string(),
                        ),
                    ),
                )
                .await;
                return;
            }
            // At least one of idle_timeout/max_duration_remaining is Some
            // here: the caller only spawns this relay when that holds.
            let wait = match (deadlines.idle_timeout, max_duration_remaining) {
                (Some(idle), Some(remaining)) => idle.min(remaining),
                (Some(idle), None) => idle,
                (None, Some(remaining)) => remaining,
                (None, None) => return,
            };
            match tokio::time::timeout(wait, rx.recv()).await {
                Ok(Some(event)) => {
                    match tokio::time::timeout(RELAY_SEND_TIMEOUT, relay_tx.send(event)).await {
                        Ok(Ok(())) => {}
                        Ok(Err(_)) => return, // downstream SSE response dropped
                        Err(_elapsed) => return, // consumer stalled; stop waiting on it
                    }
                }
                Ok(None) => return, // producer finished normally
                Err(_elapsed) => {
                    let hit_max_duration = deadlines
                        .max_duration
                        .is_some_and(|max| start.elapsed() >= max);
                    let message = if hit_max_duration {
                        "stream exceeded the configured maximum duration"
                    } else {
                        "stream exceeded the configured idle timeout"
                    };
                    let _ = tokio::time::timeout(
                        RELAY_SEND_TIMEOUT,
                        send_stream_error_async(
                            &relay_tx,
                            ErrorResponse::server_error(message.to_string()),
                        ),
                    )
                    .await;
                    return;
                }
            }
        }
    });
    relay_rx
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A stalled consumer (never drains `relay_rx`) must not block the
    /// relay loop forever — it must give up and drop its sender once a
    /// single send exceeds `RELAY_SEND_TIMEOUT`, regardless of
    /// `StreamDeadlines`. Without this, a client that stops reading
    /// without closing the connection silently defeats both
    /// `--stream-idle-timeout-secs` and `--stream-max-duration-secs`,
    /// since the loop never returns to check them while blocked on send.
    #[tokio::test(start_paused = true)]
    async fn deadline_relay_gives_up_when_consumer_stalls_past_send_timeout() {
        let (tx, rx) = mpsc::channel::<StreamEvent>(STREAM_CHANNEL_CAPACITY);
        let deadlines = StreamDeadlines {
            idle_timeout: None,
            // Long enough that only RELAY_SEND_TIMEOUT is in play here.
            max_duration: Some(Duration::from_secs(3600)),
        };
        let mut relay_rx = spawn_deadline_relay(rx, deadlines);

        // Fill relay_rx's buffer to capacity with ordinary events (these
        // all forward without blocking), then send one more "canary" event
        // whose forward attempt blocks on the full, undrained relay_rx —
        // the stalled-consumer condition this test exercises.
        for _ in 0..STREAM_CHANNEL_CAPACITY {
            tx.send(Ok(Event::default().data("filler")))
                .await
                .expect("filler send should succeed");
        }
        tx.send(Ok(Event::default().data("canary")))
            .await
            .expect("canary send should succeed");

        // Wait past RELAY_SEND_TIMEOUT WITHOUT touching relay_rx at all —
        // draining it here would itself free capacity and let the "stalled"
        // send succeed, defeating the test. Paused-time auto-advance fires
        // this sleep (and, along the way, the relay's own send-timeout
        // timer) without a real-time wait.
        tokio::time::sleep(RELAY_SEND_TIMEOUT + Duration::from_secs(1)).await;

        // Now drain everything that arrived. The canary must NOT be among
        // it: if it is, the relay never actually gave up on the stalled
        // send, it just eventually delivered it late.
        let mut received = Vec::new();
        while let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_secs(1), relay_rx.recv()).await
        {
            received.push(event);
        }
        assert_eq!(
            received.len(),
            STREAM_CHANNEL_CAPACITY,
            "only the pre-stall filler events should have been forwarded \
             before the relay gave up"
        );
        assert!(
            received
                .iter()
                .all(|event| !format!("{event:?}").contains("canary")),
            "the canary send must have been abandoned once it timed out, \
             not eventually delivered"
        );
    }

    #[tokio::test]
    async fn deadline_relay_forwards_events_normally_when_consumer_drains() {
        let (tx, rx) = mpsc::channel::<StreamEvent>(STREAM_CHANNEL_CAPACITY);
        let deadlines = StreamDeadlines {
            idle_timeout: Some(Duration::from_secs(60)),
            max_duration: Some(Duration::from_secs(60)),
        };
        let mut relay_rx = spawn_deadline_relay(rx, deadlines);

        tx.send(Ok(Event::default().data("hello")))
            .await
            .expect("send should succeed");
        drop(tx);

        let forwarded = relay_rx.recv().await.expect("event should be forwarded");
        let Ok(event) = forwarded;
        assert!(format!("{event:?}").contains("hello"));

        assert!(
            relay_rx.recv().await.is_none(),
            "relay should close its side once the producer finishes"
        );
    }
}
