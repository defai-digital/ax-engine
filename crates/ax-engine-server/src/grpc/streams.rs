use std::sync::Arc;

use ax_engine_sdk::{
    EngineSessionError, EngineTokenizer, GenerateRequest, GenerateStreamEvent, GenerateStreamState,
    SelectedBackend,
};
use axum::http::StatusCode;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;

use super::conversions::{sdk_stream_event_to_proto, unix_now};
use super::proto;
use crate::app_state::{AppState, LiveState};
use crate::chat::{Gemma4ChannelIds, strip_gemma4_channel_name_header};
use crate::generation::streaming::StreamStateSource;
use crate::openai::streaming::{Gemma4ChannelStreamFilter, IncrementalDecoder};

pub(super) type GenerateEventStream = ReceiverStream<Result<proto::GenerateStreamEvent, Status>>;
pub(super) type ChatChunkStream = ReceiverStream<Result<proto::ChatCompletionChunk, Status>>;
pub(super) type CompletionChunkStream = ReceiverStream<Result<proto::CompletionChunk, Status>>;

pub(super) async fn run_blocking<F, T>(f: F) -> Result<T, Status>
where
    F: FnOnce() -> Result<T, EngineSessionError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(session_error_status)
}

/// Map an `EngineSessionError` onto a tonic `Status` with the same
/// classification the HTTP surface uses, so client validation errors surface
/// as INVALID_ARGUMENT (retryable-by-fixing) instead of INTERNAL.
pub(super) fn session_error_status(error: EngineSessionError) -> Status {
    let (status_code, body) = crate::errors::map_session_error(error);
    let message = body.0.error.message;
    match status_code {
        StatusCode::BAD_REQUEST => Status::invalid_argument(message),
        StatusCode::NOT_FOUND => Status::not_found(message),
        StatusCode::SERVICE_UNAVAILABLE | StatusCode::BAD_GATEWAY => Status::unavailable(message),
        _ => Status::internal(message),
    }
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

/// Tokenizer for decoding native-MLX token streams into gRPC chunk text.
/// Native step events carry `delta_text: None` and raw `delta_tokens`; without
/// this decoder the chat/completion streams emit zero content on the native
/// backend (and, because nothing is ever sent, a disconnected client is never
/// detected). Delegated backends stream text directly and return `None`.
pub(super) fn native_grpc_stream_tokenizer(
    live: &LiveState,
) -> Result<Option<EngineTokenizer>, Status> {
    if live.runtime_report.selected_backend != SelectedBackend::Mlx {
        return Ok(None);
    }
    let Some(model_dir) = live.session_config.mlx_model_artifacts_dir() else {
        return Err(Status::internal(
            "native MLX gRPC streaming requires mlx_model_artifacts_dir with tokenizer.json",
        ));
    };
    EngineTokenizer::from_model_dir_cached(model_dir)
        .map(Some)
        .map_err(|error| {
            Status::internal(format!(
                "failed to load tokenizer for native MLX gRPC stream decode: {error}"
            ))
        })
}

fn grpc_stream_delta_text(
    delta_text: &Option<String>,
    delta_tokens: &[u32],
    decoder: Option<&mut IncrementalDecoder>,
) -> Result<Option<String>, Status> {
    if let Some(delta_text) = delta_text {
        return Ok(Some(delta_text.clone()));
    }
    if delta_tokens.is_empty() {
        return Ok(None);
    }
    let Some(decoder) = decoder else {
        return Ok(None);
    };
    decoder.push(delta_tokens).map(Some).map_err(|error| {
        Status::internal(format!(
            "failed to decode native MLX gRPC stream tokens: {error}"
        ))
    })
}

fn next_grpc_chat_role(chat_role_emitted: &mut bool) -> String {
    if *chat_role_emitted {
        String::new()
    } else {
        *chat_role_emitted = true;
        "assistant".to_string()
    }
}

/// Builds gRPC stream state against the caller's `LiveState` snapshot, so the
/// model that built the request is the one that streams it even if a hot-swap
/// lands mid-request.
pub(super) async fn build_grpc_stream_state(
    state: &AppState,
    live: &LiveState,
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
        let mut session = stateful_context.build_stateful_session()?;
        let ss = session.stream_generate_state_with_request_id(request_id, request)?;
        Ok((session, ss))
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
            Err(e) => return Err(session_error_status(e)),
        }
    }
}

pub(super) fn spawn_grpc_chat_stream(
    stream_state: GenerateStreamState,
    model_id: String,
    tx: mpsc::Sender<Result<proto::ChatCompletionChunk, Status>>,
    stream_context: StreamStateSource,
    tokenizer: Option<EngineTokenizer>,
) {
    spawn_grpc_blocking_stream_task(tx, "grpc chat stream", move |tx| {
        let mut ss = stream_state;
        let mut chat_role_emitted = false;
        // Mirror the HTTP SSE chat path: strip Gemma 4 thinking-channel
        // framing from the native token stream before incremental decode.
        let mut channel_filter = tokenizer.as_ref().and_then(|tokenizer| {
            let ids = Gemma4ChannelIds::from_tokenizer(tokenizer)?;
            Some(Gemma4ChannelStreamFilter::new(
                ids,
                tokenizer.token_to_id("thought"),
            ))
        });
        let mut decoder = tokenizer.map(IncrementalDecoder::new);
        let result = match stream_context {
            StreamStateSource::Stateless(ctx) => drive_grpc_chat_events(
                &mut ss,
                &model_id,
                &mut chat_role_emitted,
                &tx,
                decoder.as_mut(),
                channel_filter.as_mut(),
                |s| ctx.next_stream_event(s),
            ),
            StreamStateSource::Stateful(mut session) => drive_grpc_chat_events(
                &mut ss,
                &model_id,
                &mut chat_role_emitted,
                &tx,
                decoder.as_mut(),
                channel_filter.as_mut(),
                |s| session.next_stream_event(s),
            ),
        };
        if let Err(status) = result {
            let _ = tx.blocking_send(Err(status));
        }
    });
}

fn chat_delta_chunk_proto(
    request_id: u64,
    model_id: &str,
    role: String,
    content: String,
) -> proto::ChatCompletionChunk {
    proto::ChatCompletionChunk {
        id: format!("chatcmpl-{request_id}"),
        object: "chat.completion.chunk".to_string(),
        created: unix_now(),
        model: model_id.to_string(),
        choices: vec![proto::ChatCompletionChunkChoice {
            index: 0,
            delta: Some(proto::ChatDelta { role, content }),
            finish_reason: String::new(),
        }],
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_grpc_chat_events<N>(
    state: &mut GenerateStreamState,
    model_id: &str,
    chat_role_emitted: &mut bool,
    tx: &mpsc::Sender<Result<proto::ChatCompletionChunk, Status>>,
    mut decoder: Option<&mut IncrementalDecoder>,
    mut channel_filter: Option<&mut Gemma4ChannelStreamFilter>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next(state) {
            Ok(None) => return Ok(()),
            Err(e) => return Err(session_error_status(e)),
            Ok(Some(GenerateStreamEvent::Request(_))) => {}
            Ok(Some(GenerateStreamEvent::Response(payload))) => {
                // The model can leave its entire answer inside an unclosed
                // thinking channel; serve that body (minus the channel-name
                // header) before the stream closes (mirrors the SSE path).
                if let Some(filter) = channel_filter.as_mut()
                    && let Some(body_tokens) = filter.take_fallback_tokens()
                    && let Some(decoder) = decoder.as_mut()
                    && let Ok(body_text) = decoder.push(&body_tokens)
                {
                    let body_text = strip_gemma4_channel_name_header(&body_text);
                    if !body_text.is_empty() {
                        let chunk = chat_delta_chunk_proto(
                            payload.response.request_id,
                            model_id,
                            next_grpc_chat_role(chat_role_emitted),
                            body_text.to_string(),
                        );
                        if tx.blocking_send(Ok(chunk)).is_err() {
                            return Ok(());
                        }
                    }
                }
            }
            Ok(Some(GenerateStreamEvent::Step(payload))) => {
                // Native token path: drop Gemma 4 channel-framing tokens
                // before decode so thinking channels never surface as content.
                let filtered;
                let delta_tokens = if payload.delta_text.is_none()
                    && let Some(filter) = channel_filter.as_mut()
                {
                    filtered = filter.filter(&payload.delta_tokens);
                    if filtered.is_empty() {
                        continue;
                    }
                    filtered.as_slice()
                } else {
                    payload.delta_tokens.as_slice()
                };
                let Some(delta_text) = grpc_stream_delta_text(
                    &payload.delta_text,
                    delta_tokens,
                    decoder.as_deref_mut(),
                )?
                else {
                    continue;
                };
                if delta_text.is_empty() {
                    continue;
                }
                if let Some(filter) = channel_filter.as_mut() {
                    filter.kept_output = true;
                }
                let chunk = chat_delta_chunk_proto(
                    payload.request.request_id,
                    model_id,
                    next_grpc_chat_role(chat_role_emitted),
                    delta_text,
                );
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
    tokenizer: Option<EngineTokenizer>,
) {
    spawn_grpc_blocking_stream_task(tx, "grpc completion stream", move |tx| {
        let mut ss = stream_state;
        let mut decoder = tokenizer.map(IncrementalDecoder::new);
        let result = match stream_context {
            StreamStateSource::Stateless(ctx) => {
                drive_grpc_completion_events(&mut ss, &model_id, &tx, decoder.as_mut(), |s| {
                    ctx.next_stream_event(s)
                })
            }
            StreamStateSource::Stateful(mut session) => {
                drive_grpc_completion_events(&mut ss, &model_id, &tx, decoder.as_mut(), |s| {
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
    mut decoder: Option<&mut IncrementalDecoder>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next(state) {
            Ok(None) => return Ok(()),
            Err(e) => return Err(session_error_status(e)),
            Ok(Some(GenerateStreamEvent::Request(_) | GenerateStreamEvent::Response(_))) => {}
            Ok(Some(GenerateStreamEvent::Step(payload))) => {
                let Some(delta_text) = grpc_stream_delta_text(
                    &payload.delta_text,
                    &payload.delta_tokens,
                    decoder.as_deref_mut(),
                )?
                else {
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
