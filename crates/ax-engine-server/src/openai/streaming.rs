use ax_engine_sdk::{
    EngineSessionError, GenerateRequest, GenerateStreamEvent, GenerateStreamState,
    LlamaCppChatGenerateRequest, LlamaCppStreamHandle, MlxLmChatGenerateRequest, MlxLmStreamHandle,
    finish_reason_from_mlx_lm, start_streaming_chat_generate,
    start_streaming_llama_cpp_chat_generate,
};
use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::Event;
use serde::Serialize;
use tokio::sync::mpsc;

use crate::app_state::AppState;
use crate::backends::{llama_cpp, mlx_lm};
use crate::errors::{ErrorResponse, map_session_error};
use crate::generation::streaming::{
    StreamEventSender, StreamStateSource, build_keep_alive_stream, build_stream_state,
    drive_stream_events, send_stream_error, spawn_stream_task,
};
use crate::openai::responses::{
    finish_reason_from_llama_cpp_chat, openai_finish_reason, unix_timestamp_secs,
};
use crate::openai::schema::{
    OpenAiChatCompletionChunk, OpenAiChatCompletionChunkChoice, OpenAiChatDelta,
    OpenAiCompletionChunk, OpenAiCompletionChunkChoice, OpenAiStreamKind,
};
use crate::tasks::run_blocking_session_task;

const STREAM_CHANNEL_CAPACITY: usize = 128;

pub(crate) async fn stream_openai_request(
    state: AppState,
    request: GenerateRequest,
    stream_kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let (stream_state, stream_context) = build_stream_state(&state, request).await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_stream_task(
        tx,
        stream_state,
        move |stream_state, tx| match stream_context {
            StreamStateSource::Stateless(context) => {
                drive_openai_stream_state(stream_state, tx, stream_kind, |state| {
                    context.next_stream_event(state)
                });
            }
            StreamStateSource::Stateful(mut session) => {
                drive_openai_stream_state(stream_state, tx, stream_kind, |state| {
                    session.next_stream_event(state)
                });
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
        start_streaming_chat_generate(&runtime, &mlx_lm_backend, &request)
            .map_err(EngineSessionError::from)
    })
    .await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    tokio::task::spawn_blocking(move || {
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
        start_streaming_llama_cpp_chat_generate(&runtime, &llama_backend, &request)
            .map_err(EngineSessionError::from)
    })
    .await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    tokio::task::spawn_blocking(move || {
        drive_openai_llama_cpp_chat_stream(tx, request_id, model_id, stream);
    });

    Ok(build_keep_alive_stream(rx).into_response())
}

fn drive_openai_stream_state<N>(
    state: &mut GenerateStreamState,
    tx: StreamEventSender,
    stream_kind: OpenAiStreamKind,
    next_event: N,
) where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    let mut chat_role_emitted = false;

    drive_stream_events(
        state,
        &tx,
        next_event,
        |event| send_openai_stream_event(&tx, event, stream_kind, &mut chat_role_emitted),
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
) -> bool {
    match event {
        GenerateStreamEvent::Request(_) => true,
        GenerateStreamEvent::Step(payload) => match stream_kind {
            OpenAiStreamKind::Completion => {
                let Some(delta_text) = payload.delta_text else {
                    return true;
                };
                if delta_text.is_empty() {
                    return true;
                }
                let chunk = OpenAiCompletionChunk {
                    id: stream_kind.response_id(payload.request.request_id),
                    object: stream_kind.stream_chunk_object(),
                    created: unix_timestamp_secs(),
                    model: payload.request.model_id,
                    system_fingerprint: None,
                    choices: vec![OpenAiCompletionChunkChoice {
                        index: 0,
                        text: delta_text,
                        finish_reason: None,
                    }],
                };
                send_openai_stream_chunk(tx, &chunk)
            }
            OpenAiStreamKind::ChatCompletion => {
                let Some(delta_text) = payload.delta_text else {
                    return true;
                };
                if delta_text.is_empty() {
                    return true;
                }
                let chunk = OpenAiChatCompletionChunk {
                    id: stream_kind.response_id(payload.request.request_id),
                    object: stream_kind.stream_chunk_object(),
                    created: unix_timestamp_secs(),
                    model: payload.request.model_id,
                    system_fingerprint: None,
                    choices: vec![OpenAiChatCompletionChunkChoice {
                        index: 0,
                        delta: OpenAiChatDelta {
                            role: if *chat_role_emitted {
                                None
                            } else {
                                *chat_role_emitted = true;
                                Some("assistant")
                            },
                            content: Some(delta_text),
                        },
                        finish_reason: None,
                    }],
                };
                send_openai_stream_chunk(tx, &chunk)
            }
        },
        GenerateStreamEvent::Response(payload) => match stream_kind {
            OpenAiStreamKind::Completion => {
                let chunk = OpenAiCompletionChunk {
                    id: stream_kind.response_id(payload.response.request_id),
                    object: stream_kind.stream_chunk_object(),
                    created: unix_timestamp_secs(),
                    model: payload.response.model_id,
                    system_fingerprint: None,
                    choices: vec![OpenAiCompletionChunkChoice {
                        index: 0,
                        text: String::new(),
                        finish_reason: openai_finish_reason(payload.response.finish_reason),
                    }],
                };
                send_openai_stream_chunk(tx, &chunk)
            }
            OpenAiStreamKind::ChatCompletion => {
                let chunk = OpenAiChatCompletionChunk {
                    id: stream_kind.response_id(payload.response.request_id),
                    object: stream_kind.stream_chunk_object(),
                    created: unix_timestamp_secs(),
                    model: payload.response.model_id,
                    system_fingerprint: None,
                    choices: vec![OpenAiChatCompletionChunkChoice {
                        index: 0,
                        delta: OpenAiChatDelta::default(),
                        finish_reason: openai_finish_reason(payload.response.finish_reason),
                    }],
                };
                send_openai_stream_chunk(tx, &chunk)
            }
        },
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
                    let delta = OpenAiChatCompletionChunk {
                        id: OpenAiStreamKind::ChatCompletion.response_id(request_id),
                        object: OpenAiStreamKind::ChatCompletion.stream_chunk_object(),
                        created: unix_timestamp_secs(),
                        model: model_id.clone(),
                        system_fingerprint: None,
                        choices: vec![OpenAiChatCompletionChunkChoice {
                            index: 0,
                            delta: OpenAiChatDelta {
                                role: if chat_role_emitted {
                                    None
                                } else {
                                    chat_role_emitted = true;
                                    Some("assistant")
                                },
                                content: Some(chunk.text),
                            },
                            finish_reason: None,
                        }],
                    };
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
                    let delta = OpenAiChatCompletionChunk {
                        id: OpenAiStreamKind::ChatCompletion.response_id(request_id),
                        object: OpenAiStreamKind::ChatCompletion.stream_chunk_object(),
                        created: unix_timestamp_secs(),
                        model: model_id.clone(),
                        system_fingerprint: None,
                        choices: vec![OpenAiChatCompletionChunkChoice {
                            index: 0,
                            delta: OpenAiChatDelta {
                                role: if chat_role_emitted {
                                    None
                                } else {
                                    chat_role_emitted = true;
                                    Some("assistant")
                                },
                                content: Some(chunk.content),
                            },
                            finish_reason: None,
                        }],
                    };
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
    let chunk = OpenAiChatCompletionChunk {
        id: OpenAiStreamKind::ChatCompletion.response_id(request_id),
        object: OpenAiStreamKind::ChatCompletion.stream_chunk_object(),
        created: unix_timestamp_secs(),
        model: model_id.to_string(),
        system_fingerprint: None,
        choices: vec![OpenAiChatCompletionChunkChoice {
            index: 0,
            delta: OpenAiChatDelta::default(),
            finish_reason: openai_finish_reason(finish_reason_from_mlx_lm(finish_reason)),
        }],
    };
    send_openai_stream_chunk(tx, &chunk)
}

fn send_openai_llama_cpp_chat_final_chunk(
    tx: &StreamEventSender,
    request_id: u64,
    model_id: &str,
    finish_reason: Option<&str>,
) -> bool {
    let chunk = OpenAiChatCompletionChunk {
        id: OpenAiStreamKind::ChatCompletion.response_id(request_id),
        object: OpenAiStreamKind::ChatCompletion.stream_chunk_object(),
        created: unix_timestamp_secs(),
        model: model_id.to_string(),
        system_fingerprint: None,
        choices: vec![OpenAiChatCompletionChunkChoice {
            index: 0,
            delta: OpenAiChatDelta::default(),
            finish_reason: openai_finish_reason(finish_reason_from_llama_cpp_chat(finish_reason)),
        }],
    };
    send_openai_stream_chunk(tx, &chunk)
}

fn send_openai_stream_chunk<T: Serialize>(tx: &StreamEventSender, payload: &T) -> bool {
    match serde_json::to_string(payload) {
        Ok(data) => tx.blocking_send(Ok(Event::default().data(data))).is_ok(),
        Err(error) => {
            send_stream_error(
                tx,
                ErrorResponse::server_error(format!(
                    "failed to serialize OpenAI stream chunk: {error}"
                )),
            );
            false
        }
    }
}
