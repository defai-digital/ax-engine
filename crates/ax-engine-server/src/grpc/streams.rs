use std::sync::Arc;

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
use crate::openai::streaming::{ChatChannelStreamFilter, IncrementalDecoder};

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

fn spawn_grpc_stream_task<T, F>(
    tx: mpsc::Sender<Result<T, Status>>,
    task_name: &'static str,
    stream_context: StreamStateSource,
    driver: F,
) -> Result<(), GenerationServiceError>
where
    T: Send + 'static,
    F: FnOnce(
            mpsc::Sender<Result<T, Status>>,
            &mut dyn FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
        ) + Send
        + 'static,
{
    match stream_context {
        StreamStateSource::Service(mut events) => {
            let handle = tokio::task::spawn_blocking(move || {
                let mut next_event = || events.blocking_recv().transpose();
                driver(tx, &mut next_event);
            });
            monitor_grpc_stream_task(handle, task_name);
        }
        StreamStateSource::Stateless {
            mut state,
            context,
            permit,
        } => {
            spawn_grpc_blocking_stream_task(tx, task_name, permit, move |tx| {
                let mut next_event = || context.next_stream_event(&mut state);
                driver(tx, &mut next_event);
            });
        }
        StreamStateSource::Stateful {
            mut state,
            mut session,
            permit,
        } => {
            spawn_grpc_blocking_stream_task(tx, task_name, permit, move |tx| {
                let mut next_event = || session.next_stream_event(&mut state);
                driver(tx, &mut next_event);
            });
        }
    }
    Ok(())
}

fn monitor_grpc_stream_task(handle: tokio::task::JoinHandle<()>, task_name: &'static str) {
    tokio::spawn(async move {
        if let Err(error) = handle.await {
            tracing::error!(%error, task = task_name, "gRPC stream task failed");
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
        move |tx, next_event| {
            let result = drive_grpc_generate_events(&tx, next_event);
            if let Err(status) = result {
                let _ = tx.blocking_send(Err(status));
            }
        },
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
            let result = drive_grpc_chat_events(
                &model_id,
                &mut chat_role_emitted,
                &tx,
                decoder.as_mut(),
                channel_filter.as_mut(),
                next_event,
            );
            if let Err(status) = result {
                let _ = tx.blocking_send(Err(status));
            }
        },
    )
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
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next() {
            Ok(None) => return Ok(()),
            Err(e) => return Err(session_error_status(e)),
            Ok(Some(GenerateStreamEvent::Request(_))) => {}
            Ok(Some(GenerateStreamEvent::Response(payload))) => {
                // The model can leave its entire answer inside an unclosed
                // thinking/final channel; serve that body before the stream
                // closes (mirrors the SSE path).
                if let Some(filter) = channel_filter.as_mut()
                    && let Some(decoder) = decoder.as_mut()
                    && let Some(body_text) = filter.take_fallback_text(decoder)
                {
                    let chunk = chat_delta_chunk_proto(
                        payload.response.request_id,
                        model_id,
                        next_grpc_chat_role(chat_role_emitted),
                        body_text,
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
                let final_chunk = proto::ChatCompletionChunk {
                    id: format!("chatcmpl-{}", payload.response.request_id),
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
                if tx.blocking_send(Ok(final_chunk)).is_err() {
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
    model_id: String,
    tx: mpsc::Sender<Result<proto::CompletionChunk, Status>>,
    stream_context: StreamStateSource,
    tokenizer: Option<EngineTokenizer>,
) -> Result<(), GenerationServiceError> {
    spawn_grpc_stream_task(
        tx,
        "grpc completion stream",
        stream_context,
        move |tx, next_event| {
            let mut decoder = tokenizer.map(IncrementalDecoder::new);
            let result = drive_grpc_completion_events(&model_id, &tx, decoder.as_mut(), next_event);
            if let Err(status) = result {
                let _ = tx.blocking_send(Err(status));
            }
        },
    )
}

fn drive_grpc_completion_events<N>(
    model_id: &str,
    tx: &mpsc::Sender<Result<proto::CompletionChunk, Status>>,
    mut decoder: Option<&mut IncrementalDecoder>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next() {
            Ok(None) => return Ok(()),
            Err(e) => return Err(session_error_status(e)),
            Ok(Some(GenerateStreamEvent::Request(_))) => {}
            Ok(Some(GenerateStreamEvent::Response(payload))) => {
                let finish_reason = payload
                    .response
                    .finish_reason
                    .map(finish_reason_str)
                    .unwrap_or_default();
                let final_chunk = proto::CompletionChunk {
                    id: format!("cmpl-{}", payload.response.request_id),
                    object: "text_completion.chunk".to_string(),
                    created: unix_now(),
                    model: model_id.to_string(),
                    choices: vec![proto::CompletionChunkChoice {
                        index: 0,
                        text: String::new(),
                        finish_reason,
                    }],
                };
                if tx.blocking_send(Ok(final_chunk)).is_err() {
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

#[cfg(test)]
mod tests {
    use std::sync::mpsc as std_mpsc;
    use std::time::Duration;

    use ax_engine_sdk::{
        GenerateFinishReason, GenerateResponse, GenerateStatus, GenerateStreamEvent,
        GenerateStreamResponseEvent, GenerateStreamStepEvent, RuntimeReport, SelectedBackend,
        SessionRequestReport, SessionRequestState, SupportTier, ResolutionPolicy,
        CapabilityReport,
    };

    use crate::admission::{AdmissionController, AdmissionError};

    use super::*;

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
            },
        }
    }

    fn step_with_delta_text(text: &str) -> GenerateStreamEvent {
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
            delta_tokens: Vec::new(),
            delta_token_logprobs: Vec::new(),
            delta_text: Some(text.to_string()),
        })
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
            chunks[0]
                .choices[0]
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
        drive_grpc_completion_events("test-model", &tx, None, || {
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
    fn drive_grpc_completion_events_omits_finish_chunk_without_response_event() {
        let (tx, mut rx) = mpsc::channel(8);
        let events = [Ok(Some(step_with_delta_text("xyz"))), Ok(None)];
        let mut iter = events.into_iter();
        drive_grpc_completion_events("test-model", &tx, None, || {
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

    #[tokio::test]
    async fn disconnected_grpc_response_keeps_admission_until_producer_exits() {
        let controller = Arc::new(AdmissionController::new(Some(1)));
        let permit = controller
            .try_admit()
            .expect("stream producer should be admitted");
        let (entered_tx, entered_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let (tx, rx) = mpsc::channel::<Result<proto::GenerateStreamEvent, Status>>(1);

        spawn_grpc_blocking_stream_task(tx, "admission lifetime test", permit, move |_| {
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
