use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ax_engine_sdk::{
    EngineSessionError, EngineTokenizer, GenerateRequest, GenerateStreamEvent, SelectedBackend,
};
use axum::http::StatusCode;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;

use super::conversions::{finish_reason_str, sdk_stream_event_to_proto, unix_now};
use super::proto;
use crate::admission::AdmissionPermit;
use crate::app_state::{AppState, LiveState};
use crate::generation::service::GenerationServiceError;
use crate::generation::streaming::StreamStateSource;
use crate::openai::reasoning_stream::DeepSeekThinkCutter;
use crate::openai::stop::StopSequenceScanner;
use crate::openai::streaming::{ChatChannelStreamFilter, IncrementalDecoder};

/// Client stop strings. The native session ignores `stop_sequences`, so the
/// server enforces them on the decoded text, as the HTTP surfaces do.
struct GrpcStops(Option<StopSequenceScanner>);

impl GrpcStops {
    fn new(sequences: Vec<String>) -> Self {
        Self(StopSequenceScanner::new(sequences))
    }

    /// Text that is safe to emit now, and whether a stop matched.
    fn push(&mut self, text: String) -> (String, bool) {
        match self.0.as_mut() {
            None => (text, false),
            Some(scanner) => {
                let step = scanner.push(&text);
                (step.emit, step.matched)
            }
        }
    }

    /// Withheld text at the end of an unmatched stream.
    fn finish(&mut self) -> String {
        self.0
            .as_mut()
            .map(StopSequenceScanner::finish)
            .unwrap_or_default()
    }
}

/// A failed or cancelled generation ends the stream with an error status,
/// not a normal final chunk (the HTTP surfaces return 502 or an error frame).
fn grpc_terminal_status(response: &ax_engine_sdk::GenerateResponse) -> Result<(), Status> {
    crate::generation::native::reject_failed_generate_status(
        response.status,
        response.finish_reason,
    )
    .map_err(super::requests::openai_error_to_status)
}

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
    permit: AdmissionPermit,
    cancel_monitor_handle: tokio::task::JoinHandle<()>,
    driver: F,
) where
    T: Send + 'static,
    F: FnOnce(mpsc::Sender<Result<T, Status>>) + Send + 'static,
{
    let monitor_tx = tx.clone();
    let handle = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        driver(tx);
    });
    monitor_grpc_stream_task(handle, cancel_monitor_handle, task_name, monitor_tx);
}

fn spawn_grpc_stream_task<T, F>(
    tx: mpsc::Sender<Result<T, Status>>,
    task_name: &'static str,
    stream_context: StreamStateSource,
    driver: F,
) -> Result<(), GenerationServiceError>
where
    T: Send + 'static,
    F: FnOnce(
            &mpsc::Sender<Result<T, Status>>,
            &mut dyn FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
        ) -> Result<(), Status>
        + Send
        + 'static,
{
    // Detect client disconnect the same way the SSE path does: a producer
    // only learns of a closed channel when it next sends, which for a native
    // worker can be after an entire prefill. Flip a cancel flag (and the
    // native worker's disconnect flag) as soon as the receiver is dropped so
    // the drive loop stops before pulling another event and the admission
    // permit is released at the next scheduler boundary.
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_monitor = Arc::clone(&cancel);
    let service_disconnect = match &stream_context {
        StreamStateSource::Service(events) => Some(events.disconnect_flag()),
        StreamStateSource::Stateless { .. } | StreamStateSource::Stateful { .. } => None,
    };
    let monitor_tx = tx.clone();
    // The monitor owns a sender clone, so it must end with the producer:
    // left running, it keeps the channel open after the last event and the
    // client's stream never terminates (the SSE path aborts it the same way).
    let cancel_monitor_handle = tokio::spawn(async move {
        monitor_tx.closed().await;
        cancel_monitor.store(true, Ordering::Relaxed);
        if let Some(disconnected) = service_disconnect {
            disconnected.store(true, Ordering::Release);
        }
    });
    match stream_context {
        StreamStateSource::Service(mut events) => {
            let error_tx = tx.clone();
            let handle = tokio::task::spawn_blocking(move || {
                let mut next_event = || {
                    if cancel.load(Ordering::Relaxed) {
                        return Ok(None);
                    }
                    events.blocking_recv().transpose()
                };
                finish_grpc_stream(&tx, driver(&tx, &mut next_event));
            });
            monitor_grpc_stream_task(handle, cancel_monitor_handle, task_name, error_tx);
        }
        StreamStateSource::Stateless {
            mut state,
            context,
            permit,
        } => {
            spawn_grpc_blocking_stream_task(
                tx,
                task_name,
                permit,
                cancel_monitor_handle,
                move |tx| {
                    let mut next_event = || {
                        if cancel.load(Ordering::Relaxed) {
                            return Ok(None);
                        }
                        context.next_stream_event(&mut state)
                    };
                    finish_grpc_stream(&tx, driver(&tx, &mut next_event));
                },
            );
        }
        StreamStateSource::Stateful {
            mut state,
            mut session,
            permit,
        } => {
            spawn_grpc_blocking_stream_task(
                tx,
                task_name,
                permit,
                cancel_monitor_handle,
                move |tx| {
                    let mut next_event = || {
                        if cancel.load(Ordering::Relaxed) {
                            return Ok(None);
                        }
                        session.next_stream_event(&mut state)
                    };
                    finish_grpc_stream(&tx, driver(&tx, &mut next_event));
                },
            );
        }
    }
    Ok(())
}

/// Ends the client's stream with `status` when the drive loop failed; a
/// closed receiver (client gone) makes the send a no-op.
fn finish_grpc_stream<T>(tx: &mpsc::Sender<Result<T, Status>>, result: Result<(), Status>) {
    if let Err(status) = result {
        let _ = tx.blocking_send(Err(status));
    }
}

fn monitor_grpc_stream_task<T: Send + 'static>(
    handle: tokio::task::JoinHandle<()>,
    cancel_monitor_handle: tokio::task::JoinHandle<()>,
    task_name: &'static str,
    monitor_tx: mpsc::Sender<Result<T, Status>>,
) {
    tokio::spawn(async move {
        let result = handle.await;
        cancel_monitor_handle.abort();
        if let Err(error) = result {
            tracing::error!(%error, task = task_name, "gRPC stream task failed");
            // A panicked producer must end the client's stream with an error,
            // not leave it waiting for events that will never come.
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

/// Build gRPC stream state against the caller's model generation.
/// Admission keeps an admitted generation alive through completion and rejects
/// a stale snapshot when a hot-swap finished while the request was prepared.
pub(super) async fn build_grpc_stream_state(
    state: &AppState,
    live: &LiveState,
    request: GenerateRequest,
) -> Result<StreamStateSource, Status> {
    let permit = state.try_admit(live).map_err(super::admission_status)?;
    let request_id = state.allocate_request_id();

    if live.runtime_report.selected_backend.is_mlx() {
        let generation_service = live.generation_service.clone();
        let events = generation_service
            .start_stream(request_id, request, permit)
            .await
            .map_err(super::generation_service_status)?;
        return Ok(StreamStateSource::Service(events));
    }

    if live
        .stateless_generate_context
        .supports_stateless_streaming()
    {
        let ctx = Arc::clone(&live.stateless_generate_context);
        let stream_ctx = Arc::clone(&ctx);
        let (ss, permit) = run_blocking(move || {
            let ss = stream_ctx.stream_state_with_request_id(request_id, request)?;
            Ok((ss, permit))
        })
        .await?;
        return Ok(StreamStateSource::Stateless {
            state: ss,
            context: ctx,
            permit,
        });
    }

    let stateful_context = Arc::clone(&live.stateless_generate_context);
    let (session, ss, permit) = run_blocking(move || {
        let mut session = stateful_context.build_stateful_session()?;
        let ss = session.stream_generate_state_with_request_id(request_id, request)?;
        Ok((session, ss, permit))
    })
    .await?;

    Ok(StreamStateSource::Stateful {
        state: ss,
        session: Box::new(session),
        permit,
    })
}

pub(super) fn spawn_grpc_generate_stream(
    tx: mpsc::Sender<Result<proto::GenerateStreamEvent, Status>>,
    stream_context: StreamStateSource,
) -> Result<(), GenerationServiceError> {
    spawn_grpc_stream_task(
        tx,
        "grpc generate stream",
        stream_context,
        move |tx, next_event| drive_grpc_generate_events(tx, next_event),
    )
}

fn drive_grpc_generate_events<N>(
    tx: &mpsc::Sender<Result<proto::GenerateStreamEvent, Status>>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next() {
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
    model_id: String,
    tx: mpsc::Sender<Result<proto::ChatCompletionChunk, Status>>,
    stream_context: StreamStateSource,
    tokenizer: Option<EngineTokenizer>,
    stop_sequences: Vec<String>,
    deepseek_think_cut: bool,
) -> Result<(), GenerationServiceError> {
    spawn_grpc_stream_task(
        tx,
        "grpc chat stream",
        stream_context,
        move |tx, next_event| {
            let mut chat_role_emitted = false;
            // Mirror the HTTP SSE chat path: strip model channel framing
            // (Gemma 4 / GPT-OSS Harmony) from the native token stream before
            // incremental decode.
            let mut channel_filter = tokenizer
                .as_ref()
                .and_then(ChatChannelStreamFilter::from_tokenizer);
            let mut decoder = tokenizer.map(IncrementalDecoder::new);
            drive_grpc_chat_events(
                &model_id,
                &mut chat_role_emitted,
                tx,
                decoder.as_mut(),
                channel_filter.as_mut(),
                GrpcStops::new(stop_sequences),
                deepseek_think_cut.then(DeepSeekThinkCutter::new),
                next_event,
            )
        },
    )
}

fn cut_think(cutter: &mut Option<DeepSeekThinkCutter>, text: String) -> String {
    match cutter.as_mut() {
        Some(cutter) => cutter.push(&text),
        None => text,
    }
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
    model_id: &str,
    chat_role_emitted: &mut bool,
    tx: &mpsc::Sender<Result<proto::ChatCompletionChunk, Status>>,
    mut decoder: Option<&mut IncrementalDecoder>,
    mut channel_filter: Option<&mut ChatChannelStreamFilter>,
    mut stops: GrpcStops,
    // DeepSeek with thinking off (gRPC always renders it off): the prompt
    // pre-fills `</think>`, so a re-emitted marker and everything after it is
    // a leak, cut here as on the HTTP stream and the unary sanitize.
    mut think_cutter: Option<DeepSeekThinkCutter>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    let final_chunk = |request_id: u64, finish_reason: String| proto::ChatCompletionChunk {
        id: format!("chatcmpl-{request_id}"),
        object: "chat.completion.chunk".to_string(),
        created: unix_now(),
        model: model_id.to_string(),
        choices: vec![proto::ChatCompletionChunkChoice {
            index: 0,
            delta: Some(proto::ChatDelta {
                role: String::new(),
                content: String::new(),
            }),
            finish_reason,
        }],
    };
    // Emit `text` through the stop filter. Returns false when the stream must
    // end: the client went away, or a stop matched and was finalized.
    let emit = |stops: &mut GrpcStops,
                request_id: u64,
                text: String,
                chat_role_emitted: &mut bool|
     -> bool {
        let (text, matched) = stops.push(text);
        if !text.is_empty() {
            let chunk = chat_delta_chunk_proto(
                request_id,
                model_id,
                next_grpc_chat_role(chat_role_emitted),
                text,
            );
            if tx.blocking_send(Ok(chunk)).is_err() {
                return false;
            }
        }
        if matched {
            let _ = tx.blocking_send(Ok(final_chunk(request_id, "stop".to_string())));
            return false;
        }
        true
    };
    loop {
        match next() {
            Ok(None) => return Ok(()),
            Err(e) => return Err(session_error_status(e)),
            Ok(Some(GenerateStreamEvent::Request(_))) => {}
            Ok(Some(GenerateStreamEvent::Response(payload))) => {
                grpc_terminal_status(&payload.response)?;
                let request_id = payload.response.request_id;
                // The model can leave its entire answer inside an unclosed
                // thinking/final channel; serve that body before the stream
                // closes (mirrors the SSE path).
                if let Some(filter) = channel_filter.as_mut()
                    && let Some(decoder) = decoder.as_mut()
                    && let Some(body_text) = filter.take_fallback_text(decoder)
                    && !emit(
                        &mut stops,
                        request_id,
                        cut_think(&mut think_cutter, body_text),
                        chat_role_emitted,
                    )
                {
                    return Ok(());
                }
                // Text the incremental decoder was still holding for a
                // trailing codepoint is content too (same flush as the SSE
                // path); without it the last character of a byte-level
                // tokenizer stream is silently dropped.
                if let Some(decoder) = decoder.as_mut() {
                    let tail = decoder.finish().map_err(|error| {
                        Status::internal(format!(
                            "failed to decode native MLX gRPC stream tokens: {error}"
                        ))
                    })?;
                    if !tail.is_empty() {
                        if !tail.trim().is_empty()
                            && let Some(filter) = channel_filter.as_mut()
                        {
                            filter.mark_kept_output();
                        }
                        if !emit(
                            &mut stops,
                            request_id,
                            cut_think(&mut think_cutter, tail),
                            chat_role_emitted,
                        ) {
                            return Ok(());
                        }
                    }
                }
                // A held partial `</think>` that never completed is content.
                if let Some(held) = think_cutter.as_mut().map(DeepSeekThinkCutter::finish)
                    && !held.is_empty()
                    && !emit(&mut stops, request_id, held, chat_role_emitted)
                {
                    return Ok(());
                }
                let withheld = stops.finish();
                if !withheld.is_empty() {
                    let chunk = chat_delta_chunk_proto(
                        request_id,
                        model_id,
                        next_grpc_chat_role(chat_role_emitted),
                        withheld,
                    );
                    if tx.blocking_send(Ok(chunk)).is_err() {
                        return Ok(());
                    }
                }
                // Final chunk carries finish_reason (OpenAI SSE parity).
                let finish_reason = payload
                    .response
                    .finish_reason
                    .map(finish_reason_str)
                    .unwrap_or_default();
                if tx
                    .blocking_send(Ok(final_chunk(request_id, finish_reason)))
                    .is_err()
                {
                    return Ok(());
                }
            }
            Ok(Some(GenerateStreamEvent::Step(payload))) => {
                // Native token path: drop channel-framing tokens before decode
                // so analysis/thinking channels never surface as content.
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
                    filter.mark_kept_output();
                }
                let delta_text = cut_think(&mut think_cutter, delta_text);
                if delta_text.is_empty() {
                    continue;
                }
                if !emit(
                    &mut stops,
                    payload.request.request_id,
                    delta_text,
                    chat_role_emitted,
                ) {
                    return Ok(());
                }
            }
        }
    }
}

pub(super) fn spawn_grpc_completion_stream(
    model_id: String,
    tx: mpsc::Sender<Result<proto::CompletionChunk, Status>>,
    stream_context: StreamStateSource,
    tokenizer: Option<EngineTokenizer>,
    stop_sequences: Vec<String>,
) -> Result<(), GenerationServiceError> {
    spawn_grpc_stream_task(
        tx,
        "grpc completion stream",
        stream_context,
        move |tx, next_event| {
            let mut decoder = tokenizer.map(IncrementalDecoder::new);
            drive_grpc_completion_events(
                &model_id,
                tx,
                decoder.as_mut(),
                GrpcStops::new(stop_sequences),
                next_event,
            )
        },
    )
}

fn drive_grpc_completion_events<N>(
    model_id: &str,
    tx: &mpsc::Sender<Result<proto::CompletionChunk, Status>>,
    mut decoder: Option<&mut IncrementalDecoder>,
    mut stops: GrpcStops,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    let chunk = |request_id: u64, text: String, finish_reason: String| proto::CompletionChunk {
        id: format!("cmpl-{request_id}"),
        object: "text_completion.chunk".to_string(),
        created: unix_now(),
        model: model_id.to_string(),
        choices: vec![proto::CompletionChunkChoice {
            index: 0,
            text,
            finish_reason,
        }],
    };
    // Emit `text` through the stop filter. Returns false when the stream must
    // end: the client went away, or a stop matched and was finalized.
    let emit = |stops: &mut GrpcStops, request_id: u64, text: String| -> bool {
        let (text, matched) = stops.push(text);
        if !text.is_empty()
            && tx
                .blocking_send(Ok(chunk(request_id, text, String::new())))
                .is_err()
        {
            return false;
        }
        if matched {
            let _ = tx.blocking_send(Ok(chunk(request_id, String::new(), "stop".to_string())));
            return false;
        }
        true
    };
    loop {
        match next() {
            Ok(None) => return Ok(()),
            Err(e) => return Err(session_error_status(e)),
            Ok(Some(GenerateStreamEvent::Request(_))) => {}
            Ok(Some(GenerateStreamEvent::Response(payload))) => {
                grpc_terminal_status(&payload.response)?;
                let request_id = payload.response.request_id;
                let finish_reason = payload
                    .response
                    .finish_reason
                    .map(finish_reason_str)
                    .unwrap_or_default();
                // Text the incremental decoder was still holding for a
                // trailing codepoint is content too (same flush as the SSE
                // path); without it the last character of a byte-level
                // tokenizer stream is silently dropped.
                if let Some(decoder) = decoder.as_mut() {
                    let tail = decoder.finish().map_err(|error| {
                        Status::internal(format!(
                            "failed to decode native MLX gRPC stream tokens: {error}"
                        ))
                    })?;
                    if !tail.is_empty() && !emit(&mut stops, request_id, tail) {
                        return Ok(());
                    }
                }
                let withheld = stops.finish();
                if !withheld.is_empty()
                    && tx
                        .blocking_send(Ok(chunk(request_id, withheld, String::new())))
                        .is_err()
                {
                    return Ok(());
                }
                if tx
                    .blocking_send(Ok(chunk(request_id, String::new(), finish_reason)))
                    .is_err()
                {
                    return Ok(());
                }
            }
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
                if !emit(&mut stops, payload.request.request_id, delta_text) {
                    return Ok(());
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::mpsc as std_mpsc;
    use std::time::Duration;

    use ax_engine_sdk::{
        CapabilityReport, GenerateFinishReason, GenerateResponse, GenerateStatus,
        GenerateStreamEvent, GenerateStreamResponseEvent, GenerateStreamStepEvent,
        ResolutionPolicy, RuntimeReport, SelectedBackend, SessionRequestReport,
        SessionRequestState, SupportTier,
    };

    use crate::admission::{AdmissionController, AdmissionError};

    use super::*;

    /// The client's stream ends only when every sender is gone. The cancel
    /// monitor holds one, so it must be aborted once the producer finishes.
    #[tokio::test]
    async fn blocking_stream_task_closes_the_channel_after_the_producer_finishes() {
        let controller = Arc::new(AdmissionController::new(None));
        let permit = controller.try_admit().expect("permit");
        let (tx, mut rx) = mpsc::channel::<Result<u32, Status>>(4);
        let monitor_tx = tx.clone();
        let cancel_monitor = tokio::spawn(async move {
            monitor_tx.closed().await;
        });

        spawn_grpc_blocking_stream_task(tx, "test stream", permit, cancel_monitor, |tx| {
            tx.blocking_send(Ok(7)).expect("send");
        });

        let first = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("first event within bound");
        assert!(matches!(first, Some(Ok(7))));
        let end = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("channel must close once the producer is done");
        assert!(end.is_none());
    }

    fn sample_response(finish: Option<GenerateFinishReason>) -> GenerateResponse {
        GenerateResponse {
            request_id: 42,
            model_id: "test-model".to_string(),
            prompt_tokens: vec![1],
            prompt_text: None,
            output_tokens: vec![7, 8],
            output_token_logprobs: Vec::new(),
            output_text: None,
            prompt_token_count: None,
            output_token_count: None,
            status: GenerateStatus::Finished,
            finish_reason: finish,
            step_count: 1,
            ttft_step: Some(1),
            route: Default::default(),
            runtime: RuntimeReport {
                selected_backend: SelectedBackend::Mlx,
                support_tier: SupportTier::MlxPreview,
                resolution_policy: ResolutionPolicy::MlxOnly,
                capabilities: CapabilityReport::mlx_preview(),
                fallback_reason: None,
                host: Default::default(),
                metal_toolchain: Default::default(),
                mlx_runtime: None,
                mlx_model: None,
                delegated_runtime: None,
            },
            performance: Default::default(),
        }
    }

    fn step_event(tokens: Vec<u32>, text: Option<&str>) -> GenerateStreamEvent {
        GenerateStreamEvent::Step(GenerateStreamStepEvent {
            request: SessionRequestReport {
                request_id: 42,
                model_id: "test-model".to_string(),
                state: SessionRequestState::Running,
                prompt_tokens: vec![1],
                processed_prompt_tokens: 1,
                output_tokens: Vec::new(),
                output_token_logprobs: Vec::new(),
                prompt_len: 1,
                output_len: 0,
                max_output_tokens: 16,
                cancel_requested: false,
                execution_plan_ref: None,
                route: Default::default(),
                finish_reason: None,
                terminal_stop_reason: None,
                last_error: None,
            },
            step: Default::default(),
            delta_tokens: tokens,
            delta_token_logprobs: Vec::new(),
            delta_text: text.map(str::to_string),
        })
    }

    fn step_with_delta_text(text: &str) -> GenerateStreamEvent {
        step_event(Vec::new(), Some(text))
    }

    fn step_with_delta_tokens(tokens: Vec<u32>) -> GenerateStreamEvent {
        step_event(tokens, None)
    }

    #[test]
    fn drive_grpc_chat_events_cuts_a_leaked_deepseek_think_close() {
        // Thinking-off DeepSeek pre-fills `</think>`; a re-emitted marker and
        // everything after it must not reach content (unary sanitize parity).
        let (tx, mut rx) = mpsc::channel(16);
        let mut role = false;
        let events = [
            Ok(Some(step_with_delta_text("ans"))),
            Ok(Some(step_with_delta_text("wer</thi"))),
            Ok(Some(step_with_delta_text("nk>next turn"))),
            Ok(Some(GenerateStreamEvent::Response(
                GenerateStreamResponseEvent {
                    response: sample_response(Some(GenerateFinishReason::Stop)),
                },
            ))),
            Ok(None),
        ];
        let mut iter = events.into_iter();
        drive_grpc_chat_events(
            "test-model",
            &mut role,
            &tx,
            None,
            None,
            GrpcStops::new(Vec::new()),
            Some(DeepSeekThinkCutter::new()),
            || iter.next().unwrap_or(Ok(None)),
        )
        .expect("drive should succeed");
        drop(tx);
        let mut content = String::new();
        while let Ok(chunk) = rx.try_recv() {
            let chunk = chunk.expect("chunk ok");
            if let Some(delta) = chunk.choices[0].delta.as_ref() {
                content.push_str(&delta.content);
            }
        }
        assert_eq!(content, "answer");
    }

    #[test]
    fn drive_grpc_chat_events_emits_finish_reason_on_response() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut role = false;
        let events = [
            Ok(Some(step_with_delta_text("hi"))),
            Ok(Some(GenerateStreamEvent::Response(
                GenerateStreamResponseEvent {
                    response: sample_response(Some(GenerateFinishReason::Stop)),
                },
            ))),
            Ok(None),
        ];
        let mut iter = events.into_iter();
        drive_grpc_chat_events(
            "test-model",
            &mut role,
            &tx,
            None,
            None,
            GrpcStops::new(Vec::new()),
            None,
            || iter.next().unwrap_or(Ok(None)),
        )
        .expect("drive should succeed");

        let mut chunks = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            chunks.push(chunk.expect("chunk ok"));
        }
        assert!(
            chunks.len() >= 2,
            "expected content delta + final finish chunk, got {chunks:?}"
        );
        let final_chunk = chunks.last().expect("final");
        assert_eq!(final_chunk.choices[0].finish_reason, "stop");
        assert_eq!(
            final_chunk.choices[0]
                .delta
                .as_ref()
                .map(|d| d.content.as_str())
                .unwrap_or(""),
            ""
        );
        // Content delta must not carry finish_reason.
        assert!(
            chunks[..chunks.len() - 1]
                .iter()
                .all(|c| c.choices[0].finish_reason.is_empty()),
            "only the final Response chunk should set finish_reason"
        );
    }

    #[test]
    fn drive_grpc_chat_events_omits_finish_chunk_without_response_event() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut role = false;
        let events = [
            Ok(Some(step_with_delta_text("only-step"))),
            Ok(None), // stream ends without Response
        ];
        let mut iter = events.into_iter();
        drive_grpc_chat_events(
            "test-model",
            &mut role,
            &tx,
            None,
            None,
            GrpcStops::new(Vec::new()),
            None,
            || iter.next().unwrap_or(Ok(None)),
        )
        .expect("drive should succeed");

        let mut chunks = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            chunks.push(chunk.expect("chunk ok"));
        }
        assert_eq!(chunks.len(), 1, "only the step delta should be emitted");
        assert!(
            chunks[0].choices[0].finish_reason.is_empty(),
            "no finish_reason without Response event"
        );
        assert_eq!(
            chunks[0].choices[0]
                .delta
                .as_ref()
                .map(|d| d.content.as_str()),
            Some("only-step")
        );
    }

    #[test]
    fn drive_grpc_completion_events_emits_finish_reason_on_response() {
        let (tx, mut rx) = mpsc::channel(8);
        let events = [
            Ok(Some(step_with_delta_text("abc"))),
            Ok(Some(GenerateStreamEvent::Response(
                GenerateStreamResponseEvent {
                    response: sample_response(Some(GenerateFinishReason::Stop)),
                },
            ))),
            Ok(None),
        ];
        let mut iter = events.into_iter();
        drive_grpc_completion_events("test-model", &tx, None, GrpcStops::new(Vec::new()), || {
            iter.next().unwrap_or(Ok(None))
        })
        .expect("drive should succeed");

        let mut chunks = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            chunks.push(chunk.expect("chunk ok"));
        }
        assert!(chunks.len() >= 2);
        let final_chunk = chunks.last().expect("final");
        assert_eq!(final_chunk.choices[0].finish_reason, "stop");
        assert!(final_chunk.choices[0].text.is_empty());
    }

    #[test]
    fn grpc_streams_enforce_client_stops_across_chunk_boundaries() {
        // The native session ignores stop_sequences, so the gRPC stream must cut
        // the text itself; a stop split across deltas is still caught.
        let events = || {
            vec![
                Ok(Some(step_with_delta_text("Hello\n"))),
                Ok(Some(step_with_delta_text("\nmore text"))),
                Ok(Some(GenerateStreamEvent::Response(
                    GenerateStreamResponseEvent {
                        response: sample_response(Some(GenerateFinishReason::MaxOutputTokens)),
                    },
                ))),
                Ok(None),
            ]
        };
        let stops = || GrpcStops::new(vec!["\n\n".to_string()]);

        let (tx, mut rx) = mpsc::channel(8);
        let mut iter = events().into_iter();
        drive_grpc_completion_events("test-model", &tx, None, stops(), || {
            iter.next().unwrap_or(Ok(None))
        })
        .expect("drive should succeed");
        let mut text = String::new();
        let mut finishes = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            let chunk = chunk.expect("chunk ok");
            text.push_str(&chunk.choices[0].text);
            if !chunk.choices[0].finish_reason.is_empty() {
                finishes.push(chunk.choices[0].finish_reason.clone());
            }
        }
        assert_eq!(text, "Hello");
        assert_eq!(finishes, vec!["stop".to_string()]);

        let (tx, mut rx) = mpsc::channel(8);
        let mut role = false;
        let mut iter = events().into_iter();
        drive_grpc_chat_events(
            "test-model",
            &mut role,
            &tx,
            None,
            None,
            stops(),
            None,
            || iter.next().unwrap_or(Ok(None)),
        )
        .expect("drive should succeed");
        let mut content = String::new();
        let mut finishes = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            let chunk = chunk.expect("chunk ok");
            if let Some(delta) = chunk.choices[0].delta.as_ref() {
                content.push_str(&delta.content);
            }
            if !chunk.choices[0].finish_reason.is_empty() {
                finishes.push(chunk.choices[0].finish_reason.clone());
            }
        }
        assert_eq!(content, "Hello");
        assert_eq!(finishes, vec!["stop".to_string()]);
    }

    #[test]
    fn grpc_streams_end_failed_generations_with_an_error_status() {
        let mut failed = sample_response(Some(GenerateFinishReason::Error));
        failed.status = GenerateStatus::Failed;
        let events = || {
            vec![
                Ok(Some(step_with_delta_text("partial"))),
                Ok(Some(GenerateStreamEvent::Response(
                    GenerateStreamResponseEvent {
                        response: failed.clone(),
                    },
                ))),
                Ok(None),
            ]
        };
        let (tx, mut rx) = mpsc::channel(8);
        let mut iter = events().into_iter();
        assert!(
            drive_grpc_completion_events(
                "test-model",
                &tx,
                None,
                GrpcStops::new(Vec::new()),
                || { iter.next().unwrap_or(Ok(None)) }
            )
            .is_err()
        );
        while let Ok(chunk) = rx.try_recv() {
            assert!(chunk.expect("chunk ok").choices[0].finish_reason.is_empty());
        }
        let (tx, _rx) = mpsc::channel(8);
        let mut role = false;
        let mut iter = events().into_iter();
        assert!(
            drive_grpc_chat_events(
                "test-model",
                &mut role,
                &tx,
                None,
                None,
                GrpcStops::new(Vec::new()),
                None,
                || iter.next().unwrap_or(Ok(None)),
            )
            .is_err()
        );
    }

    #[test]
    fn drive_grpc_completion_events_omits_finish_chunk_without_response_event() {
        let (tx, mut rx) = mpsc::channel(8);
        let events = [Ok(Some(step_with_delta_text("xyz"))), Ok(None)];
        let mut iter = events.into_iter();
        drive_grpc_completion_events("test-model", &tx, None, GrpcStops::new(Vec::new()), || {
            iter.next().unwrap_or(Ok(None))
        })
        .expect("drive should succeed");

        let mut chunks = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            chunks.push(chunk.expect("chunk ok"));
        }
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].choices[0].finish_reason.is_empty());
        assert_eq!(chunks[0].choices[0].text, "xyz");
    }

    #[test]
    fn drive_grpc_chat_events_flushes_the_decoder_tail_before_finishing() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut role = false;
        let mut decoder = IncrementalDecoder::new(
            crate::openai::streaming::byte_level_test_tokenizer("grpc-chat-tail"),
        );
        let events = [
            Ok(Some(step_with_delta_tokens(vec![4]))), // "ok"
            Ok(Some(step_with_delta_tokens(vec![1]))), // held lead byte
            Ok(Some(GenerateStreamEvent::Response(
                GenerateStreamResponseEvent {
                    response: sample_response(Some(GenerateFinishReason::Stop)),
                },
            ))),
            Ok(None),
        ];
        let mut iter = events.into_iter();
        drive_grpc_chat_events(
            "test-model",
            &mut role,
            &tx,
            Some(&mut decoder),
            None,
            GrpcStops::new(Vec::new()),
            None,
            || iter.next().unwrap_or(Ok(None)),
        )
        .expect("drive should succeed");

        let mut chunks = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            chunks.push(chunk.expect("chunk ok"));
        }
        let contents: Vec<String> = chunks
            .iter()
            .filter_map(|chunk| {
                chunk.choices[0]
                    .delta
                    .as_ref()
                    .map(|delta| delta.content.clone())
            })
            .filter(|content| !content.is_empty())
            .collect();
        assert_eq!(
            contents,
            vec!["ok".to_string(), "\u{FFFD}".to_string()],
            "the held trailing codepoint must flush before the final chunk"
        );
    }

    #[test]
    fn drive_grpc_completion_events_flushes_the_decoder_tail_before_finishing() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut decoder = IncrementalDecoder::new(
            crate::openai::streaming::byte_level_test_tokenizer("grpc-completion-tail"),
        );
        let events = [
            Ok(Some(step_with_delta_tokens(vec![4]))), // "ok"
            Ok(Some(step_with_delta_tokens(vec![1]))), // held lead byte
            Ok(Some(GenerateStreamEvent::Response(
                GenerateStreamResponseEvent {
                    response: sample_response(Some(GenerateFinishReason::Stop)),
                },
            ))),
            Ok(None),
        ];
        let mut iter = events.into_iter();
        drive_grpc_completion_events(
            "test-model",
            &tx,
            Some(&mut decoder),
            GrpcStops::new(Vec::new()),
            || iter.next().unwrap_or(Ok(None)),
        )
        .expect("drive should succeed");

        let mut chunks = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            chunks.push(chunk.expect("chunk ok"));
        }
        let texts: Vec<String> = chunks
            .iter()
            .map(|chunk| chunk.choices[0].text.clone())
            .filter(|text| !text.is_empty())
            .collect();
        assert_eq!(texts, vec!["ok".to_string(), "\u{FFFD}".to_string()]);
    }

    #[tokio::test]
    async fn disconnected_grpc_response_keeps_admission_until_producer_exits() {
        let controller = Arc::new(AdmissionController::new(Some(1)));
        let permit = controller
            .try_admit()
            .expect("stream producer should be admitted");
        let (entered_tx, entered_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let (tx, rx) = mpsc::channel::<Result<proto::GenerateStreamEvent, Status>>(1);

        spawn_grpc_blocking_stream_task(
            tx,
            "admission lifetime test",
            permit,
            tokio::spawn(async {}),
            move |_| {
                entered_tx.send(()).expect("test should receive entry");
                release_rx.recv().expect("test should release producer");
            },
        );
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
