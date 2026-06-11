use std::sync::Arc;

use ax_engine_sdk::{
    EngineSessionError, GenerateRequest, GenerateStreamEvent, GenerateStreamState,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;

use super::conversions::{sdk_stream_event_to_proto, unix_now};
use super::proto;
use crate::app_state::{AppState, LiveState};
use crate::generation::streaming::StreamStateSource;

pub(super) type GenerateEventStream = ReceiverStream<Result<proto::GenerateStreamEvent, Status>>;
pub(super) type ChatChunkStream = ReceiverStream<Result<proto::ChatCompletionChunk, Status>>;
pub(super) type CompletionChunkStream = ReceiverStream<Result<proto::CompletionChunk, Status>>;

pub(super) async fn run_blocking<F, T, E>(f: F) -> Result<T, Status>
where
    F: FnOnce() -> Result<T, E> + Send + 'static,
    T: Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(|e| Status::internal(e.to_string()))
}

fn spawn_grpc_blocking_stream_task<T, F>(
    tx: mpsc::Sender<Result<T, Status>>,
    task_name: &'static str,
    driver: F,
) where
    T: Send + 'static,
    F: FnOnce(mpsc::Sender<Result<T, Status>>) + Send + 'static,
{
    let monitor_tx = tx.clone();
    let handle = tokio::task::spawn_blocking(move || driver(tx));
    tokio::spawn(async move {
        if let Err(error) = handle.await {
            tracing::error!(%error, task = task_name, "gRPC stream task failed");
            let _ = monitor_tx
                .send(Err(Status::internal(format!(
                    "{task_name} task failed: {error}"
                ))))
                .await;
        }
    });
}

pub(super) async fn build_grpc_stream_state(
    state: &AppState,
    request: GenerateRequest,
) -> Result<(GenerateStreamState, StreamStateSource), Status> {
    let live = state.snapshot();
    build_grpc_stream_state_from_live(state, live, request).await
}

async fn build_grpc_stream_state_from_live(
    state: &AppState,
    live: LiveState,
    request: GenerateRequest,
) -> Result<(GenerateStreamState, StreamStateSource), Status> {
    let request_id = state.allocate_request_id();

    if live
        .stateless_generate_context
        .supports_stateless_streaming()
    {
        let ctx = Arc::clone(&live.stateless_generate_context);
        let stream_ctx = Arc::clone(&ctx);
        let ss = run_blocking(move || stream_ctx.stream_state_with_request_id(request_id, request))
            .await?;
        return Ok((ss, StreamStateSource::Stateless(ctx)));
    }

    let stateful_context = Arc::clone(&live.stateless_generate_context);
    let (session, ss) = run_blocking(move || {
        let mut session = stateful_context
            .build_stateful_session()
            .map_err(|e| e.to_string())?;
        let ss = session
            .stream_generate_state_with_request_id(request_id, request)
            .map_err(|e| e.to_string())?;
        Ok::<_, String>((session, ss))
    })
    .await?;

    Ok((ss, StreamStateSource::Stateful(Box::new(session))))
}

pub(super) fn spawn_grpc_generate_stream(
    stream_state: GenerateStreamState,
    tx: mpsc::Sender<Result<proto::GenerateStreamEvent, Status>>,
    stream_context: StreamStateSource,
) {
    spawn_grpc_blocking_stream_task(tx, "grpc generate stream", move |tx| {
        let mut ss = stream_state;
        let result = match stream_context {
            StreamStateSource::Stateless(ctx) => {
                drive_grpc_generate_events(&mut ss, &tx, |s| ctx.next_stream_event(s))
            }
            StreamStateSource::Stateful(mut session) => {
                drive_grpc_generate_events(&mut ss, &tx, |s| session.next_stream_event(s))
            }
        };
        if let Err(status) = result {
            let _ = tx.blocking_send(Err(status));
        }
    });
}

fn drive_grpc_generate_events<N>(
    state: &mut GenerateStreamState,
    tx: &mpsc::Sender<Result<proto::GenerateStreamEvent, Status>>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next(state) {
            Ok(Some(event)) => {
                if tx
                    .blocking_send(Ok(sdk_stream_event_to_proto(event)))
                    .is_err()
                {
                    return Ok(());
                }
            }
            Ok(None) => return Ok(()),
            Err(e) => return Err(Status::internal(e.to_string())),
        }
    }
}

pub(super) fn spawn_grpc_chat_stream(
    stream_state: GenerateStreamState,
    model_id: String,
    tx: mpsc::Sender<Result<proto::ChatCompletionChunk, Status>>,
    stream_context: StreamStateSource,
) {
    spawn_grpc_blocking_stream_task(tx, "grpc chat stream", move |tx| {
        let mut ss = stream_state;
        let mut chat_role_emitted = false;
        let result = match stream_context {
            StreamStateSource::Stateless(ctx) => {
                drive_grpc_chat_events(&mut ss, &model_id, &mut chat_role_emitted, &tx, |s| {
                    ctx.next_stream_event(s)
                })
            }
            StreamStateSource::Stateful(mut session) => {
                drive_grpc_chat_events(&mut ss, &model_id, &mut chat_role_emitted, &tx, |s| {
                    session.next_stream_event(s)
                })
            }
        };
        if let Err(status) = result {
            let _ = tx.blocking_send(Err(status));
        }
    });
}

fn drive_grpc_chat_events<N>(
    state: &mut GenerateStreamState,
    model_id: &str,
    chat_role_emitted: &mut bool,
    tx: &mpsc::Sender<Result<proto::ChatCompletionChunk, Status>>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next(state) {
            Ok(None) => return Ok(()),
            Err(e) => return Err(Status::internal(e.to_string())),
            Ok(Some(GenerateStreamEvent::Request(_) | GenerateStreamEvent::Response(_))) => {}
            Ok(Some(GenerateStreamEvent::Step(payload))) => {
                let Some(delta_text) = payload.delta_text else {
                    continue;
                };
                if delta_text.is_empty() {
                    continue;
                }
                let role = if *chat_role_emitted {
                    String::new()
                } else {
                    *chat_role_emitted = true;
                    "assistant".to_string()
                };
                let chunk = proto::ChatCompletionChunk {
                    id: format!("chatcmpl-{}", payload.request.request_id),
                    object: "chat.completion.chunk".to_string(),
                    created: unix_now(),
                    model: model_id.to_string(),
                    choices: vec![proto::ChatCompletionChunkChoice {
                        index: 0,
                        delta: Some(proto::ChatDelta {
                            role,
                            content: delta_text,
                        }),
                        finish_reason: String::new(),
                    }],
                };
                if tx.blocking_send(Ok(chunk)).is_err() {
                    return Ok(());
                }
            }
        }
    }
}

pub(super) fn spawn_grpc_completion_stream(
    stream_state: GenerateStreamState,
    model_id: String,
    tx: mpsc::Sender<Result<proto::CompletionChunk, Status>>,
    stream_context: StreamStateSource,
) {
    spawn_grpc_blocking_stream_task(tx, "grpc completion stream", move |tx| {
        let mut ss = stream_state;
        let result = match stream_context {
            StreamStateSource::Stateless(ctx) => {
                drive_grpc_completion_events(&mut ss, &model_id, &tx, |s| ctx.next_stream_event(s))
            }
            StreamStateSource::Stateful(mut session) => {
                drive_grpc_completion_events(&mut ss, &model_id, &tx, |s| {
                    session.next_stream_event(s)
                })
            }
        };
        if let Err(status) = result {
            let _ = tx.blocking_send(Err(status));
        }
    });
}

fn drive_grpc_completion_events<N>(
    state: &mut GenerateStreamState,
    model_id: &str,
    tx: &mpsc::Sender<Result<proto::CompletionChunk, Status>>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next(state) {
            Ok(None) => return Ok(()),
            Err(e) => return Err(Status::internal(e.to_string())),
            Ok(Some(GenerateStreamEvent::Request(_) | GenerateStreamEvent::Response(_))) => {}
            Ok(Some(GenerateStreamEvent::Step(payload))) => {
                let Some(delta_text) = payload.delta_text else {
                    continue;
                };
                if delta_text.is_empty() {
                    continue;
                }
                let chunk = proto::CompletionChunk {
                    id: format!("cmpl-{}", payload.request.request_id),
                    object: "text_completion.chunk".to_string(),
                    created: unix_now(),
                    model: model_id.to_string(),
                    choices: vec![proto::CompletionChunkChoice {
                        index: 0,
                        text: delta_text,
                        finish_reason: String::new(),
                    }],
                };
                if tx.blocking_send(Ok(chunk)).is_err() {
                    return Ok(());
                }
            }
        }
    }
}
