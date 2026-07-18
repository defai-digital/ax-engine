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

use crate::admission::AdmissionPermit;
use crate::app_state::{AppState, LiveState};
use crate::errors::map_generation_service_error;
use crate::errors::{ErrorResponse, admission_error_response, map_session_error};
use crate::generation::requests::{GenerateHttpRequest, build_generate_request};
use crate::generation::service::{GenerationServiceError, NativeEventReceiver};
use crate::openai::validation::select_model;
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
    request.reject_video_inputs()?;
    let live = select_model(&state, request.model.as_deref())?;

    let request = build_generate_request(&live, request);
    let stream_context = build_stream_state(&state, &live, request).await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_stream_task(tx, stream_context, move |next_event, tx, cancel| {
        drive_generate_stream_state(tx, cancel, next_event);
    })
    .map_err(map_generation_service_error)?;

    Ok(build_keep_alive_stream(rx, state.limits.stream_deadlines))
}

pub(crate) enum StreamStateSource {
    Stateless {
        state: GenerateStreamState,
        context: Arc<ax_engine_sdk::StatelessGenerateContext>,
        permit: AdmissionPermit,
    },
    Stateful {
        state: GenerateStreamState,
        session: Box<EngineSession>,
        permit: AdmissionPermit,
    },
    Service(NativeEventReceiver),
}

/// Build stream state against the caller's model generation.
/// Admission keeps an admitted generation alive through completion and rejects
/// a stale snapshot when a hot-swap finished while the request was prepared.
pub(crate) async fn build_stream_state(
    state: &AppState,
    live: &LiveState,
    request: GenerateRequest,
) -> Result<StreamStateSource, (StatusCode, Json<ErrorResponse>)> {
    let permit = state.try_admit(live).map_err(admission_error_response)?;
    let request_id = state.allocate_request_id();
    if live.runtime_report.selected_backend.is_mlx() {
        let generation_service = live.generation_service.clone();
        let events = generation_service
            .start_stream(request_id, request, permit)
            .await
            .map_err(map_generation_service_error)?;
        return Ok(StreamStateSource::Service(events));
    }

    if live
        .stateless_generate_context
        .supports_stateless_streaming()
    {
        let stateless_generate_context = Arc::clone(&live.stateless_generate_context);
        let stream_context = Arc::clone(&stateless_generate_context);
        let (stream_state, permit) = run_blocking_session_task(move || {
            let stream_state = stream_context.stream_state_with_request_id(request_id, request)?;
            Ok((stream_state, permit))
        })
        .await?;

        return Ok(StreamStateSource::Stateless {
            state: stream_state,
            context: stateless_generate_context,
            permit,
        });
    }

    let stateful_context = Arc::clone(&live.stateless_generate_context);
    let (session, stream_state, permit) = run_blocking_session_task(move || {
        let mut session = stateful_context.build_stateful_session()?;
        let stream_state = session.stream_generate_state_with_request_id(request_id, request)?;
        Ok((session, stream_state, permit))
    })
    .await?;

    Ok(StreamStateSource::Stateful {
        state: stream_state,
        session: Box::new(session),
        permit,
    })
}

pub(crate) fn spawn_stream_task<F>(
    tx: StreamEventSender,
    stream_context: StreamStateSource,
    driver: F,
) -> Result<(), GenerationServiceError>
where
    F: FnOnce(
            &mut dyn FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
            StreamEventSender,
            StreamCancelFlag,
        ) + Send
        + 'static,
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
    match stream_context {
        StreamStateSource::Service(mut events) => {
            let handle = tokio::task::spawn_blocking(move || {
                let mut next_event = || events.blocking_recv().transpose();
                driver(&mut next_event, tx, cancel);
            });
            monitor_stream_task(handle, cancel_monitor_handle, error_monitor_tx);
        }
        StreamStateSource::Stateless {
            mut state,
            context,
            permit,
        } => {
            let handle = tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let mut next_event = || context.next_stream_event(&mut state);
                driver(&mut next_event, tx, cancel);
            });
            monitor_stream_task(handle, cancel_monitor_handle, error_monitor_tx);
        }
        StreamStateSource::Stateful {
            mut state,
            mut session,
            permit,
        } => {
            let handle = tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let mut next_event = || session.next_stream_event(&mut state);
                driver(&mut next_event, tx, cancel);
            });
            monitor_stream_task(handle, cancel_monitor_handle, error_monitor_tx);
        }
    }
    Ok(())
}

fn monitor_stream_task(
    handle: tokio::task::JoinHandle<()>,
    cancel_monitor_handle: tokio::task::JoinHandle<()>,
    error_monitor_tx: StreamEventSender,
) {
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
    permit: AdmissionPermit,
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
    let handle = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        driver(tx, cancel);
    });
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

fn drive_generate_stream_state<N>(tx: StreamEventSender, cancel: StreamCancelFlag, next_event: N)
where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    drive_stream_events(
        &tx,
        &cancel,
        next_event,
        |event| send_sdk_stream_event(&tx, event),
        || {},
    );
}

pub(crate) fn drive_stream_events<N, E, D>(
    tx: &StreamEventSender,
    cancel: &AtomicBool,
    mut next_event: N,
    mut emit_event: E,
    mut on_done: D,
) where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
    E: FnMut(GenerateStreamEvent) -> bool,
    D: FnMut(),
{
    loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::debug!("stream cancelled: client disconnected");
            return;
        }
        match next_event() {
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
    use std::sync::mpsc as std_mpsc;

    use crate::admission::{AdmissionController, AdmissionError};

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

    #[tokio::test]
    async fn disconnected_sse_response_keeps_admission_until_producer_exits() {
        let controller = Arc::new(AdmissionController::new(Some(1)));
        let permit = controller
            .try_admit()
            .expect("stream producer should be admitted");
        let (entered_tx, entered_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);

        spawn_sse_blocking_stream_task(tx, "admission lifetime test", permit, move |_, _| {
            entered_tx.send(()).expect("test should receive entry");
            release_rx.recv().expect("test should release producer");
        });
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("producer should start");
        drop(rx);
        tokio::task::yield_now().await;

        assert_eq!(
            controller.try_admit().err(),
            Some(AdmissionError::Saturated)
        );
        assert_eq!(controller.active_jobs(), 1);

        release_tx.send(()).expect("producer should release");
        tokio::time::timeout(Duration::from_secs(1), async {
            while controller.active_jobs() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("producer should release its permit");
    }
}
