use ax_engine_sdk::{
    EngineSessionError, EngineTokenizer, GenerateRequest, GenerateStreamEvent, GenerateStreamState,
    LlamaCppChatGenerateRequest, LlamaCppStreamHandle, MlxLmChatGenerateRequest, MlxLmStreamHandle,
    SelectedBackend, finish_reason_from_mlx_lm,
};
use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::Event;
use tokio::sync::mpsc;

use crate::app_state::AppState;
use crate::backends::{llama_cpp, mlx_lm};
use crate::errors::{ErrorResponse, error_response, map_session_error};
use crate::generation::streaming::{
    StreamEventSender, StreamStateSource, build_keep_alive_stream, build_stream_state,
    drive_stream_events, send_stream_error, spawn_sse_blocking_stream_task, spawn_stream_task,
};
use crate::openai::chunks::{
    chat_delta_chunk, chat_final_chunk, completion_delta_chunk, completion_final_chunk,
    next_chat_delta_role,
};
use crate::openai::responses::finish_reason_from_llama_cpp_chat;
use crate::openai::schema::OpenAiStreamKind;
use crate::openai::sse::send_openai_stream_chunk;
use crate::tasks::run_blocking_session_task;

const STREAM_CHANNEL_CAPACITY: usize = 128;

pub(crate) async fn stream_openai_request(
    state: AppState,
    request: GenerateRequest,
    stream_kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let (stream_state, stream_context) = build_stream_state(&state, request).await?;
    let tokenizer = native_mlx_openai_stream_tokenizer(&state)?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_stream_task(
        tx,
        stream_state,
        move |stream_state, tx| match stream_context {
            StreamStateSource::Stateless(context) => {
                drive_openai_stream_state(
                    stream_state,
                    tx,
                    stream_kind,
                    |state| context.next_stream_event(state),
                    tokenizer,
                );
            }
            StreamStateSource::Stateful(mut session) => {
                drive_openai_stream_state(
                    stream_state,
                    tx,
                    stream_kind,
                    |state| session.next_stream_event(state),
                    tokenizer,
                );
            }
        },
    );

    Ok(build_keep_alive_stream(rx).into_response())
}

pub(crate) async fn stream_openai_mlx_lm_chat_request(
    state: AppState,
    request: MlxLmChatGenerateRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let request_id = state.allocate_request_id();
    let model_id = request.model_id.clone();
    let runtime = state.runtime_report.clone();
    let mlx_lm_backend = mlx_lm::config(&state).map_err(map_session_error)?;
    let stream = run_blocking_session_task(move || {
        mlx_lm::start_chat_stream(&runtime, &mlx_lm_backend, &request)
    })
    .await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_sse_blocking_stream_task(tx, "openai mlx_lm chat stream", move |tx| {
        drive_openai_mlx_lm_chat_stream(tx, request_id, model_id, stream);
    });

    Ok(build_keep_alive_stream(rx).into_response())
}

pub(crate) async fn stream_openai_llama_cpp_chat_request(
    state: AppState,
    request: LlamaCppChatGenerateRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let request_id = state.allocate_request_id();
    let model_id = request.model_id.clone();
    let runtime = state.runtime_report.clone();
    let llama_backend = llama_cpp::config(&state).map_err(map_session_error)?;
    let stream = run_blocking_session_task(move || {
        llama_cpp::start_streaming_chat_generate(&runtime, &llama_backend, &request)
    })
    .await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_sse_blocking_stream_task(tx, "openai llama.cpp chat stream", move |tx| {
        drive_openai_llama_cpp_chat_stream(tx, request_id, model_id, stream);
    });

    Ok(build_keep_alive_stream(rx).into_response())
}

fn drive_openai_stream_state<N>(
    state: &mut GenerateStreamState,
    tx: StreamEventSender,
    stream_kind: OpenAiStreamKind,
    next_event: N,
    tokenizer: Option<EngineTokenizer>,
) where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    let mut chat_role_emitted = false;

    drive_stream_events(
        state,
        &tx,
        next_event,
        |event| {
            send_openai_stream_event(
                &tx,
                event,
                stream_kind,
                &mut chat_role_emitted,
                tokenizer.as_ref(),
            )
        },
        || {
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        },
    );
}

fn send_openai_stream_event(
    tx: &StreamEventSender,
    event: GenerateStreamEvent,
    stream_kind: OpenAiStreamKind,
    chat_role_emitted: &mut bool,
    tokenizer: Option<&EngineTokenizer>,
) -> bool {
    match event {
        GenerateStreamEvent::Request(_) => true,
        GenerateStreamEvent::Step(payload) => match stream_kind {
            OpenAiStreamKind::Completion => {
                let Some(delta_text) =
                    stream_delta_text(&payload.delta_text, &payload.delta_tokens, tokenizer, tx)
                else {
                    return true;
                };
                if delta_text.is_empty() {
                    return true;
                }
                let chunk = completion_delta_chunk(
                    payload.request.request_id,
                    payload.request.model_id,
                    delta_text,
                );
                send_openai_stream_chunk(tx, &chunk)
            }
            OpenAiStreamKind::ChatCompletion => {
                let Some(delta_text) =
                    stream_delta_text(&payload.delta_text, &payload.delta_tokens, tokenizer, tx)
                else {
                    return true;
                };
                if delta_text.is_empty() {
                    return true;
                }
                let role = next_chat_delta_role(chat_role_emitted);
                let chunk = chat_delta_chunk(
                    payload.request.request_id,
                    payload.request.model_id,
                    role,
                    delta_text,
                );
                send_openai_stream_chunk(tx, &chunk)
            }
        },
        GenerateStreamEvent::Response(payload) => match stream_kind {
            OpenAiStreamKind::Completion => {
                let chunk = completion_final_chunk(
                    payload.response.request_id,
                    payload.response.model_id,
                    payload.response.finish_reason,
                );
                send_openai_stream_chunk(tx, &chunk)
            }
            OpenAiStreamKind::ChatCompletion => {
                let chunk = chat_final_chunk(
                    payload.response.request_id,
                    payload.response.model_id,
                    payload.response.finish_reason,
                );
                send_openai_stream_chunk(tx, &chunk)
            }
        },
    }
}

fn native_mlx_openai_stream_tokenizer(
    state: &AppState,
) -> Result<Option<EngineTokenizer>, (StatusCode, Json<ErrorResponse>)> {
    if state.runtime_report.selected_backend != SelectedBackend::Mlx {
        return Ok(None);
    }
    let Some(model_dir) = state.session_config.mlx_model_artifacts_dir() else {
        return Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "native MLX OpenAI streaming requires mlx_model_artifacts_dir with tokenizer.json"
                .to_string(),
        ));
    };
    EngineTokenizer::from_model_dir(model_dir)
        .map(Some)
        .map_err(|error| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                format!("failed to load tokenizer for native MLX OpenAI stream decode: {error}"),
            )
        })
}

fn stream_delta_text(
    delta_text: &Option<String>,
    delta_tokens: &[u32],
    tokenizer: Option<&EngineTokenizer>,
    tx: &StreamEventSender,
) -> Option<String> {
    if let Some(delta_text) = delta_text {
        return Some(delta_text.clone());
    }
    if delta_tokens.is_empty() {
        return None;
    }
    let tokenizer = tokenizer?;
    match tokenizer.decode(delta_tokens, true) {
        Ok(text) => Some(text),
        Err(error) => {
            send_stream_error(
                tx,
                ErrorResponse::server_error(format!(
                    "failed to decode native MLX OpenAI stream tokens: {error}"
                )),
            );
            None
        }
    }
}

fn drive_openai_mlx_lm_chat_stream(
    tx: StreamEventSender,
    request_id: u64,
    model_id: String,
    mut stream: MlxLmStreamHandle,
) {
    let mut chat_role_emitted = false;
    loop {
        match stream.next_chunk() {
            Ok(Some(chunk)) => {
                if !chunk.text.is_empty() {
                    let role = next_chat_delta_role(&mut chat_role_emitted);
                    let delta = chat_delta_chunk(request_id, model_id.clone(), role, chunk.text);
                    if !send_openai_stream_chunk(&tx, &delta) {
                        return;
                    }
                }

                if let Some(finish_reason) = chunk.finish_reason {
                    send_openai_mlx_lm_chat_final_chunk(
                        &tx,
                        request_id,
                        &model_id,
                        Some(finish_reason.as_str()),
                    );
                    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                    return;
                }
            }
            Ok(None) => {
                send_openai_mlx_lm_chat_final_chunk(&tx, request_id, &model_id, None);
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                return;
            }
            Err(error) => {
                let (_, Json(error)) = map_session_error(EngineSessionError::from(error));
                send_stream_error(&tx, error);
                return;
            }
        }
    }
}

fn drive_openai_llama_cpp_chat_stream(
    tx: StreamEventSender,
    request_id: u64,
    model_id: String,
    mut stream: LlamaCppStreamHandle,
) {
    let mut chat_role_emitted = false;
    loop {
        match stream.next_chunk() {
            Ok(Some(chunk)) => {
                if !chunk.content.is_empty() {
                    let role = next_chat_delta_role(&mut chat_role_emitted);
                    let delta = chat_delta_chunk(request_id, model_id.clone(), role, chunk.content);
                    if !send_openai_stream_chunk(&tx, &delta) {
                        return;
                    }
                }

                if chunk.stop {
                    send_openai_llama_cpp_chat_final_chunk(
                        &tx,
                        request_id,
                        &model_id,
                        chunk.stop_type.as_deref(),
                    );
                    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                    return;
                }
            }
            Ok(None) => {
                send_openai_llama_cpp_chat_final_chunk(&tx, request_id, &model_id, None);
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                return;
            }
            Err(error) => {
                let (_, Json(error)) = map_session_error(EngineSessionError::from(error));
                send_stream_error(&tx, error);
                return;
            }
        }
    }
}

fn send_openai_mlx_lm_chat_final_chunk(
    tx: &StreamEventSender,
    request_id: u64,
    model_id: &str,
    finish_reason: Option<&str>,
) -> bool {
    let chunk = chat_final_chunk(
        request_id,
        model_id.to_string(),
        finish_reason_from_mlx_lm(finish_reason),
    );
    send_openai_stream_chunk(tx, &chunk)
}

fn send_openai_llama_cpp_chat_final_chunk(
    tx: &StreamEventSender,
    request_id: u64,
    model_id: &str,
    finish_reason: Option<&str>,
) -> bool {
    let chunk = chat_final_chunk(
        request_id,
        model_id.to_string(),
        finish_reason_from_llama_cpp_chat(finish_reason),
    );
    send_openai_stream_chunk(tx, &chunk)
}
