use std::sync::atomic::{AtomicBool, Ordering};

use ax_engine_sdk::{
    EngineSessionError, EngineTokenizer, EngineTokenizerError, GenerateFinishReason,
    GenerateRequest, GenerateStreamEvent, LlamaCppChatGenerateRequest, LlamaCppStreamHandle,
    MlxLmChatGenerateRequest, MlxLmStreamHandle, SelectedBackend, finish_reason_from_mlx_lm,
};
use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::Event;
use tokio::sync::mpsc;

use crate::app_state::{AppState, LiveState};
use crate::backends::{llama_cpp, mlx_lm};
use crate::chat::{Gemma4ChannelIds, GptOssHarmonyIds, strip_gemma4_channel_name_header};
use crate::errors::{ErrorResponse, admission_error_response, error_response, map_session_error};
use crate::generation::streaming::{
    StreamCancelFlag, StreamEventSender, build_keep_alive_stream, build_stream_state,
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
    live: LiveState,
    request: GenerateRequest,
    stream_kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let stream_context = build_stream_state(&state, &live, request).await?;
    let tokenizer = native_mlx_openai_stream_tokenizer(&live)?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_stream_task(tx, stream_context, move |next_event, tx, cancel| {
        drive_openai_stream_state(tx, cancel, stream_kind, next_event, tokenizer);
    })
    .map_err(crate::errors::map_generation_service_error)?;

    Ok(build_keep_alive_stream(rx, state.limits.stream_deadlines).into_response())
}

pub(crate) async fn stream_openai_mlx_lm_chat_request(
    state: AppState,
    live: LiveState,
    request: MlxLmChatGenerateRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let permit = state.try_admit(&live).map_err(admission_error_response)?;
    let request_id = state.allocate_request_id();
    let model_id = request.model_id.clone();
    let runtime = live.runtime_report.clone();
    let mlx_lm_backend = mlx_lm::config(&live).map_err(map_session_error)?;
    let (stream, permit) = run_blocking_session_task(move || {
        let stream = mlx_lm::start_chat_stream(&runtime, &mlx_lm_backend, &request)?;
        Ok((stream, permit))
    })
    .await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_sse_blocking_stream_task(
        tx,
        "openai mlx_lm chat stream",
        permit,
        move |tx, cancel| {
            drive_openai_mlx_lm_chat_stream(tx, &cancel, request_id, model_id, stream);
        },
    );

    Ok(build_keep_alive_stream(rx, state.limits.stream_deadlines).into_response())
}

pub(crate) async fn stream_openai_llama_cpp_chat_request(
    state: AppState,
    live: LiveState,
    request: LlamaCppChatGenerateRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let permit = state.try_admit(&live).map_err(admission_error_response)?;
    let request_id = state.allocate_request_id();
    let model_id = request.model_id.clone();
    let runtime = live.runtime_report.clone();
    let llama_backend = llama_cpp::config(&live).map_err(map_session_error)?;
    let (stream, permit) = run_blocking_session_task(move || {
        let stream = llama_cpp::start_streaming_chat_generate(&runtime, &llama_backend, &request)?;
        Ok((stream, permit))
    })
    .await?;

    let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
    spawn_sse_blocking_stream_task(
        tx,
        "openai llama.cpp chat stream",
        permit,
        move |tx, cancel| {
            drive_openai_llama_cpp_chat_stream(tx, &cancel, request_id, model_id, stream);
        },
    );

    Ok(build_keep_alive_stream(rx, state.limits.stream_deadlines).into_response())
}

fn drive_openai_stream_state<N>(
    tx: StreamEventSender,
    cancel: StreamCancelFlag,
    stream_kind: OpenAiStreamKind,
    next_event: N,
    tokenizer: Option<EngineTokenizer>,
) where
    N: FnMut() -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    let mut chat_role_emitted = false;
    // Chat streams over the native token path strip model-specific channel
    // framing (Gemma 4 thinking channels, GPT-OSS Harmony analysis/commentary);
    // raw completion streams keep the verbatim decode.
    let mut channel_filter = match stream_kind {
        OpenAiStreamKind::ChatCompletion => tokenizer
            .as_ref()
            .and_then(ChatChannelStreamFilter::from_tokenizer),
        OpenAiStreamKind::Completion => None,
    };
    let mut decoder = tokenizer.map(IncrementalDecoder::new);

    drive_stream_events(
        &tx,
        &cancel,
        next_event,
        |event| {
            send_openai_stream_event(
                &tx,
                event,
                stream_kind,
                &mut chat_role_emitted,
                decoder.as_mut(),
                channel_filter.as_mut(),
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
    mut decoder: Option<&mut IncrementalDecoder>,
    mut channel_filter: Option<&mut ChatChannelStreamFilter>,
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
                // Native token path: drop channel-framing tokens before decode
                // so analysis/thinking channels never surface as content.
                let filtered;
                let delta_tokens = if payload.delta_text.is_none()
                    && let Some(filter) = channel_filter.as_mut()
                {
                    filtered = filter.filter(&payload.delta_tokens);
                    if filtered.is_empty() {
                        return true;
                    }
                    filtered.as_slice()
                } else {
                    payload.delta_tokens.as_slice()
                };
                let Some(delta_text) =
                    stream_delta_text(&payload.delta_text, delta_tokens, decoder, tx)
                else {
                    return true;
                };
                if delta_text.is_empty() {
                    return true;
                }
                if let Some(filter) = channel_filter.as_mut() {
                    filter.mark_kept_output();
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
                // The model can leave its entire answer inside an unclosed
                // thinking/final channel; serve that body before the final
                // chunk rather than an empty message.
                if let Some(filter) = channel_filter.as_mut()
                    && let Some(decoder) = decoder.as_mut()
                    && let Some(body_text) = filter.take_fallback_text(decoder)
                {
                    let role = next_chat_delta_role(chat_role_emitted);
                    let chunk = chat_delta_chunk(
                        payload.response.request_id,
                        payload.response.model_id.clone(),
                        role,
                        body_text,
                    );
                    if !send_openai_stream_chunk(tx, &chunk) {
                        return false;
                    }
                }
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

/// Unified chat stream filter for model-specific channel framing.
///
/// Prefer GPT-OSS Harmony markers when the tokenizer defines them; otherwise
/// fall back to Gemma 4 channel markers. Models without either keep verbatim
/// decode (filter is `None`).
pub(crate) enum ChatChannelStreamFilter {
    Gemma4(Gemma4ChannelStreamFilter),
    GptOss(GptOssHarmonyStreamFilter),
}

impl ChatChannelStreamFilter {
    pub(crate) fn from_tokenizer(tokenizer: &EngineTokenizer) -> Option<Self> {
        // GPT-OSS uses `<|channel|>` (with trailing `|>`); Gemma uses
        // `<|channel>` / `<channel|>`. The two id sets do not overlap.
        if let Some(ids) = GptOssHarmonyIds::from_tokenizer(tokenizer) {
            return Some(Self::GptOss(GptOssHarmonyStreamFilter::from_tokenizer(
                ids, tokenizer,
            )));
        }
        let ids = Gemma4ChannelIds::from_tokenizer(tokenizer)?;
        Some(Self::Gemma4(Gemma4ChannelStreamFilter::new(
            ids,
            tokenizer.token_to_id("thought"),
        )))
    }

    pub(crate) fn filter(&mut self, delta_tokens: &[u32]) -> Vec<u32> {
        match self {
            Self::Gemma4(filter) => filter.filter(delta_tokens),
            Self::GptOss(filter) => filter.filter(delta_tokens),
        }
    }

    pub(crate) fn mark_kept_output(&mut self) {
        match self {
            Self::Gemma4(filter) => filter.kept_output = true,
            Self::GptOss(filter) => filter.kept_output = true,
        }
    }

    pub(crate) fn take_fallback_text(
        &mut self,
        decoder: &mut IncrementalDecoder,
    ) -> Option<String> {
        match self {
            Self::Gemma4(filter) => {
                let body_tokens = filter.take_fallback_tokens()?;
                let body_text = decoder.push(&body_tokens).ok()?;
                let body_text = strip_gemma4_channel_name_header(&body_text);
                if body_text.is_empty() {
                    None
                } else {
                    Some(body_text.to_string())
                }
            }
            Self::GptOss(filter) => {
                let body_tokens = filter.take_fallback_tokens()?;
                let body_text = decoder.push(&body_tokens).ok()?;
                if body_text.is_empty() {
                    None
                } else {
                    Some(body_text)
                }
            }
        }
    }
}

/// Streaming counterpart of `decode_gemma4_chat_output`: drops Gemma 4
/// channel-framing tokens from a chat token stream, buffering the most recent
/// channel body so an answer the model left inside a thinking channel can
/// still be served at end of stream when nothing else was emitted.
///
/// The generation prompt pre-fills `<|channel>thought\n<channel|>`, and the
/// model often continues that channel anyway: it re-emits the channel name
/// (`thought`) as plain text, optionally reasons, then closes with a stray
/// `<channel|>` before the answer. Streamed tokens cannot be retracted, so the
/// filter suppresses from the start only when the very first token is the
/// channel-name word — answers that start with anything else stream with zero
/// added latency.
pub(crate) struct Gemma4ChannelStreamFilter {
    ids: Gemma4ChannelIds,
    /// Token id of the bare channel-name word (`thought`), when the tokenizer
    /// has it as a single piece.
    thought_lead: Option<u32>,
    state: Gemma4ChannelStreamState,
    in_channel: bool,
    last_channel_body: Vec<u32>,
    /// True once a non-empty outside-channel chunk has been sent.
    pub(crate) kept_output: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Gemma4ChannelStreamState {
    /// Before the first content token: decide between suppressing a channel
    /// continuation and normal pass-through.
    LeadPending,
    /// Buffering a suspected channel continuation until its stray close.
    Suppressing,
    /// Normal pass-through.
    Passing,
}

impl Gemma4ChannelStreamFilter {
    pub(crate) fn new(ids: Gemma4ChannelIds, thought_lead: Option<u32>) -> Self {
        Self {
            ids,
            thought_lead,
            state: Gemma4ChannelStreamState::LeadPending,
            in_channel: false,
            last_channel_body: Vec::new(),
            kept_output: false,
        }
    }

    /// Partition a delta: returns the tokens to stream; tokens inside channel
    /// spans accumulate as the candidate fallback body.
    pub(crate) fn filter(&mut self, delta_tokens: &[u32]) -> Vec<u32> {
        let mut kept = Vec::with_capacity(delta_tokens.len());
        for &token in delta_tokens {
            if self.in_channel {
                if token == self.ids.close {
                    self.in_channel = false;
                } else {
                    self.last_channel_body.push(token);
                }
                continue;
            }
            if token == self.ids.open {
                self.in_channel = true;
                self.last_channel_body.clear();
                if self.state == Gemma4ChannelStreamState::LeadPending {
                    self.state = Gemma4ChannelStreamState::Passing;
                }
                continue;
            }
            match self.state {
                Gemma4ChannelStreamState::LeadPending => {
                    if self.thought_lead == Some(token) {
                        self.state = Gemma4ChannelStreamState::Suppressing;
                        self.last_channel_body.push(token);
                    } else {
                        self.state = Gemma4ChannelStreamState::Passing;
                        if token != self.ids.close {
                            kept.push(token);
                        }
                    }
                }
                Gemma4ChannelStreamState::Suppressing => {
                    if token == self.ids.close {
                        self.state = Gemma4ChannelStreamState::Passing;
                    } else {
                        self.last_channel_body.push(token);
                    }
                }
                Gemma4ChannelStreamState::Passing => {
                    // A stray close after content has streamed: swallow the
                    // marker itself (nothing can be retracted).
                    if token != self.ids.close {
                        kept.push(token);
                    }
                }
            }
        }
        kept
    }

    /// At end of stream: the channel body to serve when no outside-channel
    /// content was emitted.
    pub(crate) fn take_fallback_tokens(&mut self) -> Option<Vec<u32>> {
        if self.kept_output || self.last_channel_body.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.last_channel_body))
    }
}

/// Streaming counterpart of `strip_gpt_oss_harmony_output` / `decode_gpt_oss_chat_output`.
///
/// GPT-OSS chat often emits multi-channel Harmony traffic:
/// `…<|channel|>analysis<|message|>…thinking…<|end|>`
/// `<|start|>assistant<|channel|>final<|message|>ANSWER<|return|>`
///
/// Non-stream chat already strips to the last final-channel body. This filter
/// does the same for token streams: drop control tokens, suppress analysis /
/// commentary bodies, and pass only final-channel content. The generation
/// prompt pre-fills the final channel, so short answers that never re-open a
/// channel stream with zero added latency.
pub(crate) struct GptOssHarmonyStreamFilter {
    ids: GptOssHarmonyIds,
    /// Single-token channel names when the tokenizer exposes them as one piece.
    final_name: Option<u32>,
    analysis_name: Option<u32>,
    commentary_name: Option<u32>,
    /// Used when channel names span multiple tokens.
    tokenizer: Option<EngineTokenizer>,
    state: GptOssHarmonyStreamState,
    pub(crate) kept_output: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum GptOssHarmonyStreamState {
    /// Emit content tokens (final channel prefilled or active).
    Emitting,
    /// After `<|channel|>`, collecting the channel name until `<|message|>`.
    Header { name: Vec<u32> },
    /// Dropping analysis / commentary body until a channel closer.
    Suppressing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GptOssChannelKind {
    Final,
    Suppress,
}

impl GptOssHarmonyStreamFilter {
    pub(crate) fn from_tokenizer(ids: GptOssHarmonyIds, tokenizer: &EngineTokenizer) -> Self {
        Self {
            ids,
            final_name: tokenizer.token_to_id("final"),
            analysis_name: tokenizer.token_to_id("analysis"),
            commentary_name: tokenizer.token_to_id("commentary"),
            tokenizer: Some(tokenizer.clone()),
            state: GptOssHarmonyStreamState::Emitting,
            kept_output: false,
        }
    }

    #[cfg(test)]
    fn for_test(
        ids: GptOssHarmonyIds,
        final_name: Option<u32>,
        analysis_name: Option<u32>,
        commentary_name: Option<u32>,
    ) -> Self {
        Self {
            ids,
            final_name,
            analysis_name,
            commentary_name,
            tokenizer: None,
            state: GptOssHarmonyStreamState::Emitting,
            kept_output: false,
        }
    }

    fn channel_kind(&self, name_tokens: &[u32]) -> GptOssChannelKind {
        if let [token] = name_tokens {
            if self.final_name == Some(*token) {
                return GptOssChannelKind::Final;
            }
            if self.analysis_name == Some(*token) || self.commentary_name == Some(*token) {
                return GptOssChannelKind::Suppress;
            }
        }
        if let Some(tokenizer) = &self.tokenizer {
            let name = tokenizer
                .decode(name_tokens, true)
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();
            if name == "final" {
                return GptOssChannelKind::Final;
            }
        }
        // analysis, commentary, multi-token unknown, or empty → suppress
        GptOssChannelKind::Suppress
    }

    /// Partition a delta: returns tokens to stream (final body only).
    pub(crate) fn filter(&mut self, delta_tokens: &[u32]) -> Vec<u32> {
        let mut kept = Vec::with_capacity(delta_tokens.len());
        for &token in delta_tokens {
            match self.state {
                GptOssHarmonyStreamState::Emitting => {
                    if token == self.ids.channel {
                        self.state = GptOssHarmonyStreamState::Header { name: Vec::new() };
                    } else if self.ids.is_control(token) {
                        // Swallow Harmony control tokens; stay ready for a
                        // subsequent channel open after end/return/start.
                    } else {
                        kept.push(token);
                    }
                }
                GptOssHarmonyStreamState::Header { .. } => {
                    if token == self.ids.message {
                        let name = match std::mem::replace(
                            &mut self.state,
                            GptOssHarmonyStreamState::Emitting,
                        ) {
                            GptOssHarmonyStreamState::Header { name } => name,
                            other => {
                                self.state = other;
                                continue;
                            }
                        };
                        self.state = match self.channel_kind(&name) {
                            GptOssChannelKind::Final => GptOssHarmonyStreamState::Emitting,
                            GptOssChannelKind::Suppress => GptOssHarmonyStreamState::Suppressing,
                        };
                    } else if token == self.ids.channel {
                        self.state = GptOssHarmonyStreamState::Header { name: Vec::new() };
                    } else if !self.ids.is_control(token) {
                        if let GptOssHarmonyStreamState::Header { name } = &mut self.state {
                            name.push(token);
                        }
                    }
                }
                GptOssHarmonyStreamState::Suppressing => {
                    if token == self.ids.channel {
                        self.state = GptOssHarmonyStreamState::Header { name: Vec::new() };
                    } else if token == self.ids.end
                        || token == self.ids.return_tok
                        || self.ids.start == Some(token)
                    {
                        // Closer for a suppressed channel; next content may be
                        // plain final body (rare) or another channel open.
                        self.state = GptOssHarmonyStreamState::Emitting;
                    }
                    // else drop analysis/commentary body tokens
                }
            }
        }
        kept
    }

    /// Final body is streamed live; no deferred channel body to serve.
    pub(crate) fn take_fallback_tokens(&mut self) -> Option<Vec<u32>> {
        let _ = self.kept_output;
        None
    }
}

fn native_mlx_openai_stream_tokenizer(
    live: &LiveState,
) -> Result<Option<EngineTokenizer>, (StatusCode, Json<ErrorResponse>)> {
    if live.runtime_report.selected_backend != SelectedBackend::Mlx {
        return Ok(None);
    }
    let Some(model_dir) = live.session_config.mlx_model_artifacts_dir() else {
        return Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "native MLX OpenAI streaming requires mlx_model_artifacts_dir with tokenizer.json"
                .to_string(),
        ));
    };
    EngineTokenizer::from_model_dir_cached(model_dir)
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
pub(crate) struct IncrementalDecoder {
    tokenizer: EngineTokenizer,
    tokens: Vec<u32>,
    /// Start of the decode window; everything before it is already emitted.
    prefix_offset: usize,
    /// Boundary inside the window that ends on a complete codepoint — the text
    /// of `tokens[prefix_offset..read_offset]` has already been emitted.
    read_offset: usize,
}

impl IncrementalDecoder {
    pub(crate) fn new(tokenizer: EngineTokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
        }
    }

    pub(crate) fn push(&mut self, delta_tokens: &[u32]) -> Result<String, EngineTokenizerError> {
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
    cancel: &AtomicBool,
    request_id: u64,
    model_id: String,
    mut stream: MlxLmStreamHandle,
) {
    let mut chat_role_emitted = false;
    loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::debug!("stream cancelled: client disconnected");
            return;
        }
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
                        finish_reason_from_mlx_lm(Some(finish_reason.as_str())),
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
    cancel: &AtomicBool,
    request_id: u64,
    model_id: String,
    mut stream: LlamaCppStreamHandle,
) {
    let mut chat_role_emitted = false;
    loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::debug!("stream cancelled: client disconnected");
            return;
        }
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
                        finish_reason_from_llama_cpp_chat(chunk.stop_type.as_deref()),
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

#[cfg(test)]
mod gpt_oss_harmony_stream_filter_tests {
    use super::GptOssHarmonyStreamFilter;
    use crate::chat::GptOssHarmonyIds;

    /// Synthetic ids — no real GPT-OSS tokenizer required for the state machine.
    const IDS: GptOssHarmonyIds = GptOssHarmonyIds {
        channel: 1,
        message: 2,
        end: 3,
        return_tok: 4,
        start: Some(5),
        call: Some(6),
    };
    const FINAL_NAME: u32 = 10;
    const ANALYSIS_NAME: u32 = 11;

    fn filter() -> GptOssHarmonyStreamFilter {
        GptOssHarmonyStreamFilter::for_test(IDS, Some(FINAL_NAME), Some(ANALYSIS_NAME), Some(12))
    }

    #[test]
    fn prefilled_final_passes_content_tokens() {
        // No channel open: content streams immediately (generation prefill case).
        let mut filter = filter();
        assert_eq!(filter.filter(&[100, 101, 102]), vec![100, 101, 102]);
        assert!(filter.take_fallback_tokens().is_none());
    }

    #[test]
    fn suppresses_analysis_then_emits_final() {
        let mut filter = filter();
        let tokens = [
            IDS.channel,
            ANALYSIS_NAME,
            IDS.message,
            200,
            201,
            IDS.end,
            IDS.start.unwrap(),
            IDS.channel,
            FINAL_NAME,
            IDS.message,
            300,
            301,
            IDS.return_tok,
        ];
        assert_eq!(filter.filter(&tokens), vec![300, 301]);
    }

    #[test]
    fn drops_control_tokens_inside_final_body() {
        let mut filter = filter();
        assert_eq!(filter.filter(&[100, IDS.return_tok, 101]), vec![100, 101]);
    }

    #[test]
    fn unknown_channel_is_suppressed() {
        let mut filter = filter();
        let tokens = [
            IDS.channel,
            99, // unknown name token
            IDS.message,
            200,
            IDS.end,
            300,
        ];
        // Unknown channel body dropped; trailing content after end is emitted.
        assert_eq!(filter.filter(&tokens), vec![300]);
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
