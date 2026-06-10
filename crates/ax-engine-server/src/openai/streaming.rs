use ax_engine_sdk::{
    EngineSessionError, EngineTokenizer, EngineTokenizerError, GenerateRequest,
    GenerateFinishReason, GenerateStreamEvent, GenerateStreamState, LlamaCppChatGenerateRequest,
    LlamaCppStreamHandle, MlxLmChatGenerateRequest, MlxLmStreamHandle, SelectedBackend,
    finish_reason_from_mlx_lm,
};
use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::Event;
use tokio::sync::mpsc;

use crate::app_state::AppState;
use crate::backends::{llama_cpp, mlx_lm};
use crate::errors::{ErrorResponse, error_response, map_session_error};
use crate::openai::requests::OpenAiOutputPostprocessing;
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
    output_postprocessing: OpenAiOutputPostprocessing,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let (stream_state, stream_context) = build_stream_state(&state, request).await?;
    let tokenizer = native_mlx_openai_stream_tokenizer(&state)?;
    let mut postprocessing = OpenAiOutputPostprocessingState::new(output_postprocessing);

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
                    &mut postprocessing,
                );
            }
            StreamStateSource::Stateful(mut session) => {
                drive_openai_stream_state(
                    stream_state,
                    tx,
                    stream_kind,
                    |state| session.next_stream_event(state),
                    tokenizer,
                    &mut postprocessing,
                );
            }
        },
    );

    Ok(build_keep_alive_stream(rx).into_response())
}

pub(crate) async fn stream_openai_mlx_lm_chat_request(
    state: AppState,
    request: MlxLmChatGenerateRequest,
    output_postprocessing: OpenAiOutputPostprocessing,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let request_id = state.allocate_request_id();
    let model_id = request.model_id.clone();
    let runtime = state.runtime_report.clone();
    let mlx_lm_backend = mlx_lm::config(&state).map_err(map_session_error)?;
    let stream = run_blocking_session_task(move || {
        mlx_lm::start_chat_stream(&runtime, &mlx_lm_backend, &request)
    })
    .await?;
    let mut postprocessing = OpenAiOutputPostprocessingState::new(output_postprocessing);

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_sse_blocking_stream_task(tx, "openai mlx_lm chat stream", move |tx| {
        drive_openai_mlx_lm_chat_stream(
            tx,
            request_id,
            model_id,
            stream,
            &mut postprocessing,
        );
    });

    Ok(build_keep_alive_stream(rx).into_response())
}

pub(crate) async fn stream_openai_llama_cpp_chat_request(
    state: AppState,
    request: LlamaCppChatGenerateRequest,
    output_postprocessing: OpenAiOutputPostprocessing,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let request_id = state.allocate_request_id();
    let model_id = request.model_id.clone();
    let runtime = state.runtime_report.clone();
    let llama_backend = llama_cpp::config(&state).map_err(map_session_error)?;
    let stream = run_blocking_session_task(move || {
        llama_cpp::start_streaming_chat_generate(&runtime, &llama_backend, &request)
    })
    .await?;
    let mut postprocessing = OpenAiOutputPostprocessingState::new(output_postprocessing);

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_sse_blocking_stream_task(tx, "openai llama.cpp chat stream", move |tx| {
        drive_openai_llama_cpp_chat_stream(
            tx,
            request_id,
            model_id,
            stream,
            &mut postprocessing,
        );
    });

    Ok(build_keep_alive_stream(rx).into_response())
}

fn drive_openai_stream_state<N>(
    state: &mut GenerateStreamState,
    tx: StreamEventSender,
    stream_kind: OpenAiStreamKind,
    next_event: N,
    tokenizer: Option<EngineTokenizer>,
    output_postprocessing: &mut OpenAiOutputPostprocessingState,
) where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    let mut chat_role_emitted = false;
    let mut decoder = tokenizer.map(IncrementalDecoder::new);

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
                output_postprocessing,
                decoder.as_mut(),
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
    output_postprocessing: &mut OpenAiOutputPostprocessingState,
    decoder: Option<&mut IncrementalDecoder>,
) -> bool {
    match event {
        GenerateStreamEvent::Request(_) => true,
        GenerateStreamEvent::Step(payload) => match stream_kind {
            OpenAiStreamKind::Completion => {
                let Some(delta_text) =
                    stream_delta_text(&payload.delta_text, &payload.delta_tokens, decoder, tx)
                else {
                    return true;
                };
                let Some(processed_delta_text) = output_postprocessing.apply_delta_text(&delta_text)
                else {
                    return true;
                };
                if processed_delta_text.is_empty() {
                    return true;
                }
                let chunk = completion_delta_chunk(
                    payload.request.request_id,
                    payload.request.model_id,
                    processed_delta_text,
                );
                send_openai_stream_chunk(tx, &chunk)
            }
            OpenAiStreamKind::ChatCompletion => {
                let Some(delta_text) =
                    stream_delta_text(&payload.delta_text, &payload.delta_tokens, decoder, tx)
                else {
                    return true;
                };
                let Some(processed_delta_text) = output_postprocessing.apply_delta_text(&delta_text)
                else {
                    return true;
                };
                if processed_delta_text.is_empty() {
                    return true;
                }
                let role = next_chat_delta_role(chat_role_emitted);
                let chunk = chat_delta_chunk(
                    payload.request.request_id,
                    payload.request.model_id,
                    role,
                    processed_delta_text,
                );
                send_openai_stream_chunk(tx, &chunk)
            }
        },
        GenerateStreamEvent::Response(payload) => match stream_kind {
            OpenAiStreamKind::Completion => {
                let finish_reason =
                    output_postprocessing.apply_finish_reason(payload.response.finish_reason);
                let chunk = completion_final_chunk(
                    payload.response.request_id,
                    payload.response.model_id,
                    finish_reason,
                );
                send_openai_stream_chunk(tx, &chunk)
            }
            OpenAiStreamKind::ChatCompletion => {
                let finish_reason =
                    output_postprocessing.apply_finish_reason(payload.response.finish_reason);
                let chunk = chat_final_chunk(
                    payload.response.request_id,
                    payload.response.model_id,
                    finish_reason,
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
    decoder: Option<&mut IncrementalDecoder>,
    tx: &StreamEventSender,
) -> Option<String> {
    if let Some(delta_text) = delta_text {
        return Some(delta_text.clone());
    }
    if delta_tokens.is_empty() {
        return None;
    }
    let decoder = decoder?;
    match decoder.push(delta_tokens) {
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

/// Incremental detokenizer for native MLX token streams.
///
/// Byte-level BPE tokenizers (Qwen, GLM, Gemma) split a single non-ASCII
/// codepoint across several tokens, so decoding each step's `delta_tokens` in
/// isolation renders the incomplete byte sequence as U+FFFD (`�`) and corrupts
/// CJK/emoji output. This decodes a small trailing token window each step
/// (O(1) amortized — the same prefix-offset/read-offset scheme production
/// inference servers use) and emits only the newly completed text, holding back
/// a partial trailing codepoint until later tokens finish it.
struct IncrementalDecoder {
    tokenizer: EngineTokenizer,
    tokens: Vec<u32>,
    /// Start of the decode window; everything before it is already emitted.
    prefix_offset: usize,
    /// Boundary inside the window that ends on a complete codepoint — the text
    /// of `tokens[prefix_offset..read_offset]` has already been emitted.
    read_offset: usize,
}

impl IncrementalDecoder {
    fn new(tokenizer: EngineTokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
        }
    }

    fn push(&mut self, delta_tokens: &[u32]) -> Result<String, EngineTokenizerError> {
        self.tokens.extend_from_slice(delta_tokens);
        // `prefix` is the already-emitted, codepoint-complete head of the window;
        // `whole` extends it with the newly appended tokens.
        let prefix = self
            .tokenizer
            .decode(&self.tokens[self.prefix_offset..self.read_offset], true)?;
        let whole = self
            .tokenizer
            .decode(&self.tokens[self.prefix_offset..], true)?;
        match incremental_delta(&prefix, &whole) {
            Some(delta) => {
                self.prefix_offset = self.read_offset;
                self.read_offset = self.tokens.len();
                Ok(delta)
            }
            // Trailing codepoint still incomplete: keep the window and wait for
            // the tokens that complete it (offsets unchanged).
            None => Ok(String::new()),
        }
    }
}

/// Diff the decoded window prefix against the full window decode.
///
/// `prefix` is always a complete (non-`�`-terminated) decode of the
/// already-emitted tokens, so `whole` extends it byte-for-byte. Returns the
/// newly completed suffix to emit, or `None` when the trailing codepoint is
/// still incomplete (decoded as U+FFFD) and must be held back.
fn incremental_delta(prefix: &str, whole: &str) -> Option<String> {
    if whole.len() <= prefix.len() || whole.ends_with('\u{FFFD}') {
        return None;
    }
    // `prefix` is complete, so `whole` starts with it and `prefix.len()` lands on
    // a char boundary. The boundary check keeps the slice panic-free even if a
    // tokenizer ever violated that assumption.
    if !whole.is_char_boundary(prefix.len()) {
        return None;
    }
    Some(whole[prefix.len()..].to_string())
}

fn drive_openai_mlx_lm_chat_stream(
    tx: StreamEventSender,
    request_id: u64,
    model_id: String,
    mut stream: MlxLmStreamHandle,
    output_postprocessing: &mut OpenAiOutputPostprocessingState,
) {
    let mut chat_role_emitted = false;
    loop {
        match stream.next_chunk() {
            Ok(Some(chunk)) => {
                if !chunk.text.is_empty() {
                    if let Some(processed_delta_text) =
                        output_postprocessing.apply_delta_text(&chunk.text)
                    {
                        if !processed_delta_text.is_empty() {
                            let role = next_chat_delta_role(&mut chat_role_emitted);
                            let delta = chat_delta_chunk(
                                request_id,
                                model_id.clone(),
                                role,
                                processed_delta_text,
                            );
                            if !send_openai_stream_chunk(&tx, &delta) {
                                return;
                            }
                        }
                    }
                }

                if let Some(finish_reason) = chunk.finish_reason {
                    let finish_reason = output_postprocessing
                        .apply_finish_reason(finish_reason_from_mlx_lm(Some(finish_reason.as_str())));
                    send_openai_mlx_lm_chat_final_chunk(
                        &tx,
                        request_id,
                        &model_id,
                        finish_reason,
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
    output_postprocessing: &mut OpenAiOutputPostprocessingState,
) {
    let mut chat_role_emitted = false;
    loop {
        match stream.next_chunk() {
            Ok(Some(chunk)) => {
                if !chunk.content.is_empty() {
                    if let Some(processed_delta_text) = output_postprocessing.apply_delta_text(&chunk.content)
                    {
                        if !processed_delta_text.is_empty() {
                            let role = next_chat_delta_role(&mut chat_role_emitted);
                            let delta = chat_delta_chunk(
                                request_id,
                                model_id.clone(),
                                role,
                                processed_delta_text,
                            );
                            if !send_openai_stream_chunk(&tx, &delta) {
                                return;
                            }
                        }
                    }
                }

                if chunk.stop {
                    let finish_reason =
                        output_postprocessing.apply_finish_reason(finish_reason_from_llama_cpp_chat(
                            chunk.stop_type.as_deref(),
                        ));
                    send_openai_llama_cpp_chat_final_chunk(
                        &tx,
                        request_id,
                        &model_id,
                        finish_reason,
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
    finish_reason: Option<GenerateFinishReason>,
) -> bool {
    let chunk = chat_final_chunk(request_id, model_id.to_string(), finish_reason);
    send_openai_stream_chunk(tx, &chunk)
}

fn send_openai_llama_cpp_chat_final_chunk(
    tx: &StreamEventSender,
    request_id: u64,
    model_id: &str,
    finish_reason: Option<GenerateFinishReason>,
) -> bool {
    let chunk = chat_final_chunk(request_id, model_id.to_string(), finish_reason);
    send_openai_stream_chunk(tx, &chunk)
}

struct OpenAiOutputPostprocessingState {
    postprocessing: OpenAiOutputPostprocessing,
    raw_output_text: String,
    emitted_output_text: String,
}

impl OpenAiOutputPostprocessingState {
    fn new(postprocessing: OpenAiOutputPostprocessing) -> Self {
        Self {
            postprocessing,
            raw_output_text: String::new(),
            emitted_output_text: String::new(),
        }
    }

    fn apply_delta_text(&mut self, delta_text: &str) -> Option<String> {
        if self.postprocessing.is_noop() {
            return Some(delta_text.to_string());
        }

        self.raw_output_text.push_str(delta_text);
        let normalized = self.postprocessing.normalize_output_text(&self.raw_output_text);
        let normalized_len = normalized.len();
        let emitted_len = self.emitted_output_text.len();
        if normalized_len < emitted_len {
            self.emitted_output_text = normalized;
            return None;
        }

        let delta_text = normalized[emitted_len..].to_string();
        self.emitted_output_text = normalized;
        if delta_text.is_empty() {
            None
        } else {
            Some(delta_text)
        }
    }

    fn apply_finish_reason(
        &self,
        finish_reason: Option<GenerateFinishReason>,
    ) -> Option<GenerateFinishReason> {
        self.postprocessing.apply_finish_reason(finish_reason)
    }
}

#[cfg(test)]
mod incremental_decode_tests {
    use super::incremental_delta;

    #[test]
    fn emits_full_text_for_ascii() {
        assert_eq!(incremental_delta("", "hello"), Some("hello".to_string()));
    }

    #[test]
    fn emits_only_new_suffix() {
        assert_eq!(incremental_delta("ab", "abc"), Some("c".to_string()));
        assert_eq!(
            incremental_delta("hello", "hello world"),
            Some(" world".to_string())
        );
    }

    #[test]
    fn waits_when_no_new_visible_text() {
        // The new tokens added nothing decodable yet (e.g. a skipped special
        // token): there is no progress to emit.
        assert_eq!(incremental_delta("ab", "ab"), None);
    }

    #[test]
    fn holds_back_incomplete_trailing_codepoint() {
        // A multi-byte codepoint split across step boundaries decodes with a
        // trailing replacement char; it must be held back, not emitted.
        assert_eq!(incremental_delta("", "ab\u{FFFD}"), None);
        assert_eq!(incremental_delta("ab", "ab\u{FFFD}"), None);
    }

    #[test]
    fn emits_multibyte_codepoint_once_complete() {
        // '你' (U+4F60) arrives complete after the held-back step.
        assert_eq!(incremental_delta("ab", "ab你"), Some("你".to_string()));
        assert_eq!(incremental_delta("", "你好"), Some("你好".to_string()));
        // Emoji (4-byte UTF-8) likewise.
        assert_eq!(incremental_delta("", "🚀"), Some("🚀".to_string()));
    }

    #[test]
    fn full_window_stays_on_char_boundary() {
        // Mixed content with a complete leading codepoint and an incomplete tail
        // is held back entirely until the tail completes.
        assert_eq!(incremental_delta("你", "你好\u{FFFD}"), None);
        assert_eq!(incremental_delta("你", "你好世"), Some("好世".to_string()));
    }
}
