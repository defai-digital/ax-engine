use std::path::Path;

use ax_engine_sdk::{
    EngineTokenizer, Gemma4UnifiedAudioInput, Gemma4UnifiedAudioRuntimeInput, Gemma4UnifiedError,
    Gemma4UnifiedImageInput, Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedModality,
    Gemma4UnifiedProcessorConfig, Gemma4UnifiedRuntimeInputs, Gemma4UnifiedSoftTokenRange,
    Gemma4UnifiedTokenSpan, Gemma4UnifiedVideoInput, Gemma4UnifiedVideoRuntimeInput,
    LlamaCppChatMessage, MlxLmChatMessage,
};
use axum::Json;
use axum::http::StatusCode;
use serde_json::Value;

use crate::chat;
use crate::errors::{ErrorResponse, error_response};
use crate::multimodal::{
    self, MediaError, MediaProcessors, PreprocessedAudio, PreprocessedImage, PreprocessedVideo,
    VideoFrame,
};
use crate::openai::schema::{
    OpenAiChatContent, OpenAiChatContentPart, OpenAiChatMessage, OpenAiStopInput,
};

type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);
type ChatMessagePairs = Vec<(String, String)>;

/// A native-MLX chat prompt with Gemma 4 unified media already expanded into
/// soft-token spans and preprocessed tensors.
pub(crate) struct Gemma4UnifiedChatPrompt {
    pub(crate) input_tokens: Vec<u32>,
    pub(crate) runtime_inputs: Gemma4UnifiedRuntimeInputs,
}

pub(crate) fn render_openai_chat_prompt(
    model_id: &str,
    messages: &[OpenAiChatMessage],
) -> Result<String, HttpErrorResponse> {
    let rendered_messages = render_openai_chat_message_pairs(messages)?;
    chat::render_prompt(model_id, &rendered_messages).map_err(chat_error_response)
}

pub(crate) fn build_mlx_lm_chat_messages(
    messages: &[OpenAiChatMessage],
) -> Result<Vec<MlxLmChatMessage>, HttpErrorResponse> {
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }

    messages
        .iter()
        .map(|message| {
            let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
            let content = render_openai_chat_content(&message.content)?;
            Ok(MlxLmChatMessage::new(role, content))
        })
        .collect()
}

pub(crate) fn build_llama_cpp_chat_messages(
    messages: &[OpenAiChatMessage],
) -> Result<Vec<LlamaCppChatMessage>, HttpErrorResponse> {
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }

    messages
        .iter()
        .map(|message| {
            let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
            let content = render_openai_chat_content(&message.content)?;
            Ok(LlamaCppChatMessage::new(role, content))
        })
        .collect()
}

pub(crate) fn chat_template_kwargs_for_model_id(model_id: &str) -> Option<serde_json::Value> {
    chat::template_kwargs_for_model_id(model_id)
}

pub(crate) fn openai_chat_stop_sequences(
    model_id: &str,
    stop: Option<OpenAiStopInput>,
) -> Vec<String> {
    chat::stop_sequences(
        model_id,
        stop.map(OpenAiStopInput::into_vec).unwrap_or_default(),
    )
}

fn render_openai_chat_message_pairs(
    messages: &[OpenAiChatMessage],
) -> Result<ChatMessagePairs, HttpErrorResponse> {
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }
    messages
        .iter()
        .map(|message| {
            let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
            let content = render_openai_chat_content(&message.content)?;
            Ok((role.to_string(), content))
        })
        .collect()
}

fn empty_chat_messages_error() -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        "chat.completions requires at least one message".to_string(),
    )
}

fn chat_error_response(message: String) -> HttpErrorResponse {
    error_response(StatusCode::BAD_REQUEST, "invalid_request", message)
}

fn render_openai_chat_content(content: &OpenAiChatContent) -> Result<String, HttpErrorResponse> {
    match content {
        OpenAiChatContent::Text(text) => Ok(text.clone()),
        OpenAiChatContent::Parts(parts) => {
            let mut rendered = String::new();
            for part in parts {
                match chat_content_part_kind(part) {
                    OpenAiChatContentPartKind::Text => {
                        let text = part.text.as_deref().ok_or_else(|| {
                            error_response(
                                StatusCode::BAD_REQUEST,
                                "invalid_request",
                                format!(
                                    "{} chat content parts require a text field",
                                    part.part_type
                                ),
                            )
                        })?;
                        rendered.push_str(text);
                    }
                    OpenAiChatContentPartKind::Media(kind) => {
                        return Err(openai_media_part_error(kind, part));
                    }
                    OpenAiChatContentPartKind::Unsupported => {
                        return Err(error_response(
                            StatusCode::BAD_REQUEST,
                            "invalid_request",
                            format!(
                                "unsupported chat content part type {}; AX preview currently accepts text-only OpenAI chat content plus processed multimodal_inputs on /v1/generate",
                                part.part_type
                            ),
                        ));
                    }
                }
            }
            Ok(rendered)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenAiChatContentPartKind {
    Text,
    Media(OpenAiChatMediaKind),
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenAiChatMediaKind {
    Image,
    Audio,
    Video,
}

fn chat_content_part_kind(part: &OpenAiChatContentPart) -> OpenAiChatContentPartKind {
    match part.part_type.as_str() {
        "text" | "input_text" => OpenAiChatContentPartKind::Text,
        "image_url" | "input_image" | "image" => {
            OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Image)
        }
        "input_audio" | "audio_url" | "audio" => {
            OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Audio)
        }
        "video_url" | "input_video" | "video" => {
            OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Video)
        }
        _ => OpenAiChatContentPartKind::Unsupported,
    }
}

fn openai_media_part_error(
    media_kind: OpenAiChatMediaKind,
    part: &OpenAiChatContentPart,
) -> HttpErrorResponse {
    let field_hint = match media_kind {
        OpenAiChatMediaKind::Image => {
            if part.image_url.is_some() {
                "image_url"
            } else {
                "image payload"
            }
        }
        OpenAiChatMediaKind::Audio => {
            if part.input_audio.is_some() {
                "input_audio"
            } else if part.audio_url.is_some() {
                "audio_url"
            } else {
                "audio payload"
            }
        }
        OpenAiChatMediaKind::Video => {
            if part.video_url.is_some() {
                "video_url"
            } else {
                "video payload"
            }
        }
    };
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!(
            "OpenAI chat content part type {} includes {field_hint}, but AX Engine does not yet decode raw OpenAI media into Gemma4UnifiedRuntimeInputs; send processed multimodal_inputs.gemma4_unified tensors through /v1/generate or use text-only chat",
            part.part_type
        ),
    )
}

/// True when any message carries an inline media content part (image/audio/video).
pub(crate) fn messages_contain_inline_media(messages: &[OpenAiChatMessage]) -> bool {
    messages.iter().any(|message| match &message.content {
        OpenAiChatContent::Text(_) => false,
        OpenAiChatContent::Parts(parts) => parts.iter().any(|part| {
            matches!(
                chat_content_part_kind(part),
                OpenAiChatContentPartKind::Media(_)
            )
        }),
    })
}

/// Build a native-MLX chat prompt when the messages carry inline base64 image or
/// audio parts, expanding them into Gemma 4 unified soft-token spans and the
/// encoder-free connector's preprocessed tensors. Returns `Ok(None)` when there
/// is no inline media, so the caller falls back to the text-only path.
pub(crate) fn render_gemma4_unified_chat_with_media(
    model_id: &str,
    model_dir: &Path,
    messages: &[OpenAiChatMessage],
) -> Result<Option<Gemma4UnifiedChatPrompt>, HttpErrorResponse> {
    if !messages_contain_inline_media(messages) {
        return Ok(None);
    }
    if messages.is_empty() {
        return Err(empty_chat_messages_error());
    }

    let tokenizer = EngineTokenizer::from_model_dir(model_dir).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("failed to load tokenizer for multimodal chat: {error}"),
        )
    })?;
    let MediaProcessors {
        config,
        normalization,
        video_vision,
        video_max_frames,
    } = multimodal::load_processor_config(model_dir).map_err(media_error_response)?;

    let image_placeholder = placeholder_string(&tokenizer, config.tokens.image_token_id, "image")?;
    let audio_placeholder = placeholder_string(&tokenizer, config.tokens.audio_token_id, "audio")?;
    // A video_token_id of 0 means the model declares no video token.
    let video_placeholder = if config.tokens.video_token_id != 0 {
        Some(placeholder_string(
            &tokenizer,
            config.tokens.video_token_id,
            "video",
        )?)
    } else {
        None
    };

    // Render the prompt with one placeholder token per media item, collecting the
    // raw bytes in document order so they line up with the placeholder positions.
    let mut collected = CollectedMedia::default();
    let mut pairs: ChatMessagePairs = Vec::with_capacity(messages.len());
    for message in messages {
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        let content = render_content_collecting_media(
            &message.content,
            &image_placeholder,
            &audio_placeholder,
            video_placeholder.as_deref(),
            &mut collected,
        )?;
        pairs.push((role.to_string(), content));
    }

    let prompt = chat::render_prompt(model_id, &pairs).map_err(chat_error_response)?;
    let tokens = tokenizer.encode(&prompt, false).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("failed to tokenize multimodal chat prompt: {error}"),
        )
    })?;

    // Preprocess raw media into the encoder-free connector's tensors.
    let images: Vec<PreprocessedImage> = collected
        .images
        .iter()
        .map(|bytes| multimodal::preprocess_image(bytes, &config.vision, &normalization))
        .collect::<Result<_, _>>()
        .map_err(media_error_response)?;
    let audios: Vec<PreprocessedAudio> = collected
        .audios
        .iter()
        .map(|bytes| match config.audio.as_ref() {
            Some(processor) => multimodal::preprocess_wav(bytes, processor),
            None => Err(MediaError::Unsupported(
                "model has no audio feature extractor; audio input is not supported".to_string(),
            )),
        })
        .collect::<Result<_, _>>()
        .map_err(media_error_response)?;
    // Video frames use a lower per-frame soft-token budget and carry mm:ss
    // timestamp tokens, matching the reference Gemma4UnifiedVideoProcessor.
    let mut videos: Vec<PreprocessedVideo> = Vec::with_capacity(collected.videos.len());
    let mut video_timestamp_tokens: Vec<Vec<Vec<u32>>> = Vec::with_capacity(collected.videos.len());
    for bytes in &collected.videos {
        let frames = multimodal::decode_video_frames(bytes, video_max_frames)
            .map_err(media_error_response)?;
        let timestamps = build_video_timestamp_tokens(&tokenizer, &frames)?;
        let preprocessed =
            multimodal::preprocess_video_frames(&frames, &video_vision, &normalization)
                .map_err(media_error_response)?;
        videos.push(preprocessed);
        video_timestamp_tokens.push(timestamps);
    }

    // Expand all placeholders in a single pass so every span indexes the final
    // token stream (separate passes would shift later positions).
    let image_inputs: Vec<Gemma4UnifiedImageInput> = images
        .iter()
        .map(|image| Gemma4UnifiedImageInput {
            width: image.width,
            height: image.height,
        })
        .collect();
    let audio_inputs: Vec<Gemma4UnifiedAudioInput> = audios
        .iter()
        .map(|audio| Gemma4UnifiedAudioInput {
            sample_count: audio.sample_count,
        })
        .collect();
    let video_inputs: Vec<Gemma4UnifiedVideoInput> = videos
        .iter()
        .zip(&video_timestamp_tokens)
        .map(|(video, timestamps)| Gemma4UnifiedVideoInput {
            frame_count: video.frame_count,
            soft_tokens_per_frame: video.soft_tokens_per_frame,
            timestamp_token_ids_per_frame: timestamps.clone(),
        })
        .collect();
    let expanded = expand_media(
        &config,
        &tokens,
        &image_inputs,
        &audio_inputs,
        &video_inputs,
    )?;

    let runtime_inputs = build_runtime_inputs(images, audios, videos, expanded.spans);
    runtime_inputs
        .validate_for_prompt_len(expanded.tokens.len())
        .map_err(|error| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("multimodal inputs do not match expanded prompt: {error}"),
            )
        })?;

    Ok(Some(Gemma4UnifiedChatPrompt {
        input_tokens: expanded.tokens,
        runtime_inputs,
    }))
}

/// Raw media bytes collected while rendering, in document order per modality.
#[derive(Default)]
struct CollectedMedia {
    images: Vec<Vec<u8>>,
    audios: Vec<Vec<u8>>,
    videos: Vec<Vec<u8>>,
}

fn placeholder_string(
    tokenizer: &EngineTokenizer,
    token_id: u32,
    label: &str,
) -> Result<String, HttpErrorResponse> {
    tokenizer.id_to_token(token_id).ok_or_else(|| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!(
                "tokenizer has no surface form for the Gemma4 {label} placeholder token {token_id}"
            ),
        )
    })
}

fn render_content_collecting_media(
    content: &OpenAiChatContent,
    image_placeholder: &str,
    audio_placeholder: &str,
    video_placeholder: Option<&str>,
    collected: &mut CollectedMedia,
) -> Result<String, HttpErrorResponse> {
    match content {
        OpenAiChatContent::Text(text) => Ok(text.clone()),
        OpenAiChatContent::Parts(parts) => {
            let mut rendered = String::new();
            for part in parts {
                match chat_content_part_kind(part) {
                    OpenAiChatContentPartKind::Text => {
                        let text = part.text.as_deref().ok_or_else(|| {
                            error_response(
                                StatusCode::BAD_REQUEST,
                                "invalid_request",
                                format!(
                                    "{} chat content parts require a text field",
                                    part.part_type
                                ),
                            )
                        })?;
                        rendered.push_str(text);
                    }
                    OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Image) => {
                        collected.images.push(image_part_bytes(part)?);
                        rendered.push_str(image_placeholder);
                    }
                    OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Audio) => {
                        collected.audios.push(audio_part_bytes(part)?);
                        rendered.push_str(audio_placeholder);
                    }
                    OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Video) => {
                        let placeholder = video_placeholder.ok_or_else(|| {
                            error_response(
                                StatusCode::BAD_REQUEST,
                                "invalid_request",
                                "model declares no video token; video chat content is not supported".to_string(),
                            )
                        })?;
                        collected.videos.push(video_part_bytes(part)?);
                        rendered.push_str(placeholder);
                    }
                    OpenAiChatContentPartKind::Unsupported => {
                        return Err(error_response(
                            StatusCode::BAD_REQUEST,
                            "invalid_request",
                            format!("unsupported chat content part type {}", part.part_type),
                        ));
                    }
                }
            }
            Ok(rendered)
        }
    }
}

fn image_part_bytes(part: &OpenAiChatContentPart) -> Result<Vec<u8>, HttpErrorResponse> {
    let value = part
        .image_url
        .as_ref()
        .ok_or_else(|| media_payload_missing("image", "image_url"))?;
    let url =
        data_value_url(value).ok_or_else(|| media_payload_missing("image", "image_url.url"))?;
    let (_mime, bytes) = multimodal::decode_data_uri(url).map_err(media_error_response)?;
    Ok(bytes)
}

fn audio_part_bytes(part: &OpenAiChatContentPart) -> Result<Vec<u8>, HttpErrorResponse> {
    if let Some(input_audio) = part.input_audio.as_ref() {
        let data = input_audio
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| media_payload_missing("audio", "input_audio.data"))?;
        return multimodal::decode_base64(data).map_err(media_error_response);
    }
    if let Some(audio_url) = part.audio_url.as_ref() {
        let url = data_value_url(audio_url)
            .ok_or_else(|| media_payload_missing("audio", "audio_url.url"))?;
        let (_mime, bytes) = multimodal::decode_data_uri(url).map_err(media_error_response)?;
        return Ok(bytes);
    }
    Err(media_payload_missing("audio", "input_audio or audio_url"))
}

fn video_part_bytes(part: &OpenAiChatContentPart) -> Result<Vec<u8>, HttpErrorResponse> {
    let value = part
        .video_url
        .as_ref()
        .ok_or_else(|| media_payload_missing("video", "video_url"))?;
    let url =
        data_value_url(value).ok_or_else(|| media_payload_missing("video", "video_url.url"))?;
    let (_mime, bytes) = multimodal::decode_data_uri(url).map_err(media_error_response)?;
    Ok(bytes)
}

/// Tokenize the `mm:ss` timestamp prefix for each video frame, matching the
/// reference `get_video_repl`: a leading space on every frame except the first,
/// and always a trailing space, e.g. `"00:00 "` then `" 00:02 "`.
fn build_video_timestamp_tokens(
    tokenizer: &EngineTokenizer,
    frames: &[VideoFrame],
) -> Result<Vec<Vec<u32>>, HttpErrorResponse> {
    frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let seconds = frame.timestamp_seconds.max(0.0);
            let minutes = (seconds / 60.0).floor() as u32;
            let secs = (seconds % 60.0).floor() as u32;
            let stamp = format!("{minutes:02}:{secs:02}");
            let prefix = if index == 0 {
                format!("{stamp} ")
            } else {
                format!(" {stamp} ")
            };
            tokenizer.encode(&prefix, false).map_err(|error| {
                error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    format!("failed to tokenize video timestamp: {error}"),
                )
            })
        })
        .collect()
}

/// OpenAI media URL fields are either a bare string or `{ "url": "..." }`.
fn data_value_url(value: &Value) -> Option<&str> {
    match value {
        Value::String(url) => Some(url.as_str()),
        _ => value.get("url").and_then(Value::as_str),
    }
}

/// A video span plus the per-frame soft-token ranges it occupies in the final
/// token stream.
type VideoSpan = (Gemma4UnifiedTokenSpan, Vec<Gemma4UnifiedSoftTokenRange>);

/// The expanded token stream and the spans for each modality, in document order.
#[derive(Debug)]
struct ExpandedMedia {
    tokens: Vec<u32>,
    spans: MediaSpans,
}

#[derive(Debug, Default)]
struct MediaSpans {
    image: Vec<Gemma4UnifiedTokenSpan>,
    audio: Vec<Gemma4UnifiedTokenSpan>,
    video: Vec<VideoSpan>,
}

/// Single-pass expansion of image, audio, and video placeholder tokens into
/// their boundary + soft-token replacements, recording spans against the final
/// stream. A single pass keeps every span (and video frame range) aligned with
/// the post-expansion token positions.
fn expand_media(
    config: &Gemma4UnifiedProcessorConfig,
    tokens: &[u32],
    image_inputs: &[Gemma4UnifiedImageInput],
    audio_inputs: &[Gemma4UnifiedAudioInput],
    video_inputs: &[Gemma4UnifiedVideoInput],
) -> Result<ExpandedMedia, HttpErrorResponse> {
    let image_token = config.tokens.image_token_id;
    let audio_token = config.tokens.audio_token_id;
    let video_token = config.tokens.video_token_id;

    let image_markers = tokens.iter().filter(|&&t| t == image_token).count();
    let audio_markers = tokens.iter().filter(|&&t| t == audio_token).count();
    if image_markers != image_inputs.len() {
        return Err(placeholder_count_error(
            "image",
            image_inputs.len(),
            image_markers,
        ));
    }
    if audio_markers != audio_inputs.len() {
        return Err(placeholder_count_error(
            "audio",
            audio_inputs.len(),
            audio_markers,
        ));
    }
    if video_token != 0 {
        let video_markers = tokens.iter().filter(|&&t| t == video_token).count();
        if video_markers != video_inputs.len() {
            return Err(placeholder_count_error(
                "video",
                video_inputs.len(),
                video_markers,
            ));
        }
    }

    let mut out = Vec::with_capacity(tokens.len());
    let mut spans = MediaSpans::default();
    let mut image_index = 0usize;
    let mut audio_index = 0usize;
    let mut video_index = 0usize;

    for (token_index, &token) in tokens.iter().enumerate() {
        if token == image_token {
            let input = image_inputs[image_index];
            image_index += 1;
            let replacement = config
                .image_replacement_tokens(input)
                .map_err(gemma4_unified_error_response)?;
            let soft = config
                .image_soft_tokens(input)
                .map_err(gemma4_unified_error_response)?;
            spans.image.push(Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Image,
                placeholder_index: token_index,
                replacement_start: out.len(),
                soft_token_count: soft,
                replacement_token_count: replacement.len() as u32,
            });
            out.extend_from_slice(&replacement);
        } else if token == audio_token {
            let input = audio_inputs[audio_index].clone();
            audio_index += 1;
            let replacement = config
                .audio_replacement_tokens(input.clone())
                .map_err(gemma4_unified_error_response)?;
            let soft = config
                .audio_soft_tokens(input)
                .map_err(gemma4_unified_error_response)?;
            spans.audio.push(Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Audio,
                placeholder_index: token_index,
                replacement_start: out.len(),
                soft_token_count: soft,
                replacement_token_count: replacement.len() as u32,
            });
            out.extend_from_slice(&replacement);
        } else if video_token != 0 && token == video_token {
            let input = video_inputs[video_index].clone();
            video_index += 1;
            let (replacement, relative_ranges) = config
                .video_replacement_tokens_with_ranges(input)
                .map_err(gemma4_unified_error_response)?;
            let replacement_start = out.len();
            let soft = relative_ranges
                .iter()
                .map(|range| range.soft_token_count)
                .sum();
            // The ranges are relative to the replacement block; offset them onto
            // the final stream.
            let ranges = relative_ranges
                .into_iter()
                .map(|range| Gemma4UnifiedSoftTokenRange {
                    start: replacement_start + range.start,
                    soft_token_count: range.soft_token_count,
                })
                .collect();
            spans.video.push((
                Gemma4UnifiedTokenSpan {
                    modality: Gemma4UnifiedModality::Video,
                    placeholder_index: token_index,
                    replacement_start,
                    soft_token_count: soft,
                    replacement_token_count: replacement.len() as u32,
                },
                ranges,
            ));
            out.extend_from_slice(&replacement);
        } else {
            out.push(token);
        }
    }

    Ok(ExpandedMedia { tokens: out, spans })
}

fn build_runtime_inputs(
    images: Vec<PreprocessedImage>,
    audios: Vec<PreprocessedAudio>,
    videos: Vec<PreprocessedVideo>,
    spans: MediaSpans,
) -> Gemma4UnifiedRuntimeInputs {
    let images = images
        .into_iter()
        .zip(spans.image)
        .map(|(image, span)| Gemma4UnifiedImageRuntimeInput {
            span,
            pixel_values: image.pixel_values,
            pixel_position_ids: image.pixel_position_ids,
        })
        .collect();
    let audios = audios
        .into_iter()
        .zip(spans.audio)
        .map(|(audio, span)| Gemma4UnifiedAudioRuntimeInput {
            span,
            input_features: audio.input_features,
            frame_count: audio.frame_count,
            feature_count: audio.feature_count,
        })
        .collect();
    let videos = videos
        .into_iter()
        .zip(spans.video)
        .map(
            |(video, (span, soft_token_ranges))| Gemma4UnifiedVideoRuntimeInput {
                span,
                soft_token_ranges,
                pixel_values: video.pixel_values,
                pixel_position_ids: video.pixel_position_ids,
                frame_count: video.frame_count,
            },
        )
        .collect();
    Gemma4UnifiedRuntimeInputs {
        images,
        audios,
        videos,
    }
}

fn media_payload_missing(media: &str, field: &str) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!("{media} chat content part is missing a {field} payload"),
    )
}

fn placeholder_count_error(media: &str, expected: usize, actual: usize) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!(
            "expected {expected} {media} placeholder token(s) after tokenization, found {actual}; the {media} placeholder may not round-trip through this tokenizer"
        ),
    )
}

fn media_error_response(error: MediaError) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        error.to_string(),
    )
}

fn gemma4_unified_error_response(error: Gemma4UnifiedError) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        error.to_string(),
    )
}

#[cfg(test)]
mod media_tests {
    use super::*;
    use serde_json::json;

    fn processor_config() -> Gemma4UnifiedProcessorConfig {
        let model = json!({
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "video_token_id": 258884,
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
            "boa_token_id": 256000,
            "eoa_token_index": 258883,
            "vision_config": {
                "patch_size": 16,
                "model_patch_size": 48,
                "pooling_kernel_size": 3,
                "num_soft_tokens": 280
            }
        });
        let processor = json!({
            "image_processor": {
                "patch_size": 16,
                "model_patch_size": 48,
                "pooling_kernel_size": 3,
                "max_soft_tokens": 280
            },
            "feature_extractor": {"sampling_rate": 16000},
            "audio_seq_length": 1500
        });
        Gemma4UnifiedProcessorConfig::from_model_and_processor_config(&model, &processor)
            .expect("processor config should parse")
    }

    #[test]
    fn single_pass_keeps_audio_span_aligned_after_image_expansion() {
        let config = processor_config();
        // text, <image>, text, <audio>, text
        let tokens = [10u32, 258880, 11, 258881, 12];
        let images = [Gemma4UnifiedImageInput {
            width: 224,
            height: 224,
        }];
        let audios = [Gemma4UnifiedAudioInput { sample_count: 1600 }];

        let expanded = expand_media(&config, &tokens, &images, &audios, &[]).expect("expansion");
        let image_spans = &expanded.spans.image;
        let audio_spans = &expanded.spans.audio;

        assert_eq!(image_spans.len(), 1);
        assert_eq!(audio_spans.len(), 1);

        // Image: boi + 256 soft + eoi = 258 tokens, starting right after token 10.
        let image = &image_spans[0];
        assert_eq!(image.replacement_start, 1);
        assert_eq!(image.soft_token_count, 256);
        assert_eq!(image.replacement_token_count, 258);

        // Audio span must account for the image expansion shift: 1 (token 10)
        // + 258 (image replacement) + 1 (token 11) = 260.
        let audio = &audio_spans[0];
        assert_eq!(audio.replacement_start, 260);
        assert_eq!(audio.soft_token_count, 3); // ceil(1600 / 640)
        assert_eq!(audio.replacement_token_count, 5); // boa + 3 soft + eoa

        // Total length: 3 plain tokens + 258 + 5 replacements.
        assert_eq!(expanded.tokens.len(), 3 + 258 + 5);
        // Boundary tokens landed where the spans say they did.
        assert_eq!(expanded.tokens[image.replacement_start], 255999); // boi
        assert_eq!(expanded.tokens[audio.replacement_start], 256000); // boa
    }

    #[test]
    fn video_expansion_aligns_per_frame_ranges() {
        let config = processor_config();
        // text, <video>, text
        let tokens = [10u32, 258884, 11];
        let videos = [Gemma4UnifiedVideoInput {
            frame_count: 2,
            soft_tokens_per_frame: 4,
            timestamp_token_ids_per_frame: Vec::new(),
        }];

        let expanded = expand_media(&config, &tokens, &[], &[], &videos).expect("expansion");
        assert_eq!(expanded.spans.video.len(), 1);
        let (span, ranges) = &expanded.spans.video[0];

        // Each frame contributes boi + 4 soft + eoi = 6 tokens; two frames = 12.
        assert_eq!(span.replacement_start, 1);
        assert_eq!(span.soft_token_count, 8); // 2 frames * 4
        assert_eq!(span.replacement_token_count, 12);
        assert_eq!(ranges.len(), 2);

        // Frame ranges point at the soft tokens (after each boi) in the final
        // stream: frame 0 at 1+1=2, frame 1 at 1+6+1=8.
        assert_eq!(ranges[0].start, 2);
        assert_eq!(ranges[0].soft_token_count, 4);
        assert_eq!(ranges[1].start, 8);
        assert_eq!(ranges[1].soft_token_count, 4);
        assert_eq!(expanded.tokens[span.replacement_start], 255999); // boi of frame 0
        for range in ranges {
            assert_eq!(expanded.tokens[range.start], 258884); // video soft token
        }
    }

    #[test]
    fn rejects_placeholder_count_mismatch() {
        let config = processor_config();
        let tokens = [10u32, 258880, 11]; // one image marker
        let error = expand_media(
            &config,
            &tokens,
            &[], // but zero image inputs
            &[],
            &[],
        )
        .expect_err("mismatch should fail");
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(error.1.error.message.contains("image placeholder"));
    }
}
