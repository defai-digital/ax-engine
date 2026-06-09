use std::path::Path;

use ax_engine_sdk::{
    EngineTokenizer, Gemma4UnifiedAudioInput, Gemma4UnifiedAudioRuntimeInput, Gemma4UnifiedError,
    Gemma4UnifiedImageInput, Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedModality,
    Gemma4UnifiedProcessorConfig, Gemma4UnifiedRuntimeInputs, Gemma4UnifiedTokenSpan,
    LlamaCppChatMessage, MlxLmChatMessage,
};
use axum::Json;
use axum::http::StatusCode;
use serde_json::Value;

use crate::chat;
use crate::errors::{ErrorResponse, error_response};
use crate::multimodal::{self, MediaError, PreprocessedAudio, PreprocessedImage};
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
    let (config, normalization) =
        multimodal::load_processor_config(model_dir).map_err(media_error_response)?;

    let image_placeholder = placeholder_string(&tokenizer, config.tokens.image_token_id, "image")?;
    let audio_placeholder = placeholder_string(&tokenizer, config.tokens.audio_token_id, "audio")?;

    // Render the prompt with one placeholder token per media item, collecting the
    // raw bytes in document order so they line up with the placeholder positions.
    let mut image_bytes: Vec<Vec<u8>> = Vec::new();
    let mut audio_bytes: Vec<Vec<u8>> = Vec::new();
    let mut pairs: ChatMessagePairs = Vec::with_capacity(messages.len());
    for message in messages {
        let role = chat::normalize_role(&message.role).map_err(chat_error_response)?;
        let content = render_content_collecting_media(
            &message.content,
            &image_placeholder,
            &audio_placeholder,
            &mut image_bytes,
            &mut audio_bytes,
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
    let images: Vec<PreprocessedImage> = image_bytes
        .iter()
        .map(|bytes| multimodal::preprocess_image(bytes, &config.vision, &normalization))
        .collect::<Result<_, _>>()
        .map_err(media_error_response)?;
    let audios: Vec<PreprocessedAudio> = audio_bytes
        .iter()
        .map(|bytes| match config.audio.as_ref() {
            Some(processor) => multimodal::preprocess_wav(bytes, processor),
            None => Err(MediaError::Unsupported(
                "model has no audio feature extractor; audio input is not supported".to_string(),
            )),
        })
        .collect::<Result<_, _>>()
        .map_err(media_error_response)?;

    // Expand image and audio placeholders in a single pass so every span indexes
    // the final token stream (separate passes would shift later positions).
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
    let (input_tokens, image_spans, audio_spans) =
        expand_image_and_audio(&config, &tokens, &image_inputs, &audio_inputs)?;

    let runtime_inputs = build_runtime_inputs(images, image_spans, audios, audio_spans);
    runtime_inputs
        .validate_for_prompt_len(input_tokens.len())
        .map_err(|error| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("multimodal inputs do not match expanded prompt: {error}"),
            )
        })?;

    Ok(Some(Gemma4UnifiedChatPrompt {
        input_tokens,
        runtime_inputs,
    }))
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
    image_bytes: &mut Vec<Vec<u8>>,
    audio_bytes: &mut Vec<Vec<u8>>,
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
                        image_bytes.push(image_part_bytes(part)?);
                        rendered.push_str(image_placeholder);
                    }
                    OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Audio) => {
                        audio_bytes.push(audio_part_bytes(part)?);
                        rendered.push_str(audio_placeholder);
                    }
                    OpenAiChatContentPartKind::Media(OpenAiChatMediaKind::Video) => {
                        return Err(error_response(
                            StatusCode::BAD_REQUEST,
                            "invalid_request",
                            "video chat content is not supported; send processed multimodal_inputs.gemma4_unified tensors through /v1/generate".to_string(),
                        ));
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

/// OpenAI media URL fields are either a bare string or `{ "url": "..." }`.
fn data_value_url(value: &Value) -> Option<&str> {
    match value {
        Value::String(url) => Some(url.as_str()),
        _ => value.get("url").and_then(Value::as_str),
    }
}

/// Expanded token stream plus the image and audio spans it produced.
type ExpandedMedia = (
    Vec<u32>,
    Vec<Gemma4UnifiedTokenSpan>,
    Vec<Gemma4UnifiedTokenSpan>,
);

/// Single-pass expansion of image and audio placeholder tokens into their
/// boundary + soft-token replacements, recording spans against the final stream.
fn expand_image_and_audio(
    config: &Gemma4UnifiedProcessorConfig,
    tokens: &[u32],
    image_inputs: &[Gemma4UnifiedImageInput],
    audio_inputs: &[Gemma4UnifiedAudioInput],
) -> Result<ExpandedMedia, HttpErrorResponse> {
    let image_token = config.tokens.image_token_id;
    let audio_token = config.tokens.audio_token_id;

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

    let mut out = Vec::with_capacity(tokens.len());
    let mut image_spans = Vec::with_capacity(image_inputs.len());
    let mut audio_spans = Vec::with_capacity(audio_inputs.len());
    let mut image_index = 0usize;
    let mut audio_index = 0usize;

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
            image_spans.push(Gemma4UnifiedTokenSpan {
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
            audio_spans.push(Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Audio,
                placeholder_index: token_index,
                replacement_start: out.len(),
                soft_token_count: soft,
                replacement_token_count: replacement.len() as u32,
            });
            out.extend_from_slice(&replacement);
        } else {
            out.push(token);
        }
    }

    Ok((out, image_spans, audio_spans))
}

fn build_runtime_inputs(
    images: Vec<PreprocessedImage>,
    image_spans: Vec<Gemma4UnifiedTokenSpan>,
    audios: Vec<PreprocessedAudio>,
    audio_spans: Vec<Gemma4UnifiedTokenSpan>,
) -> Gemma4UnifiedRuntimeInputs {
    let images = images
        .into_iter()
        .zip(image_spans)
        .map(|(image, span)| Gemma4UnifiedImageRuntimeInput {
            span,
            pixel_values: image.pixel_values,
            pixel_position_ids: image.pixel_position_ids,
        })
        .collect();
    let audios = audios
        .into_iter()
        .zip(audio_spans)
        .map(|(audio, span)| Gemma4UnifiedAudioRuntimeInput {
            span,
            input_features: audio.input_features,
            frame_count: audio.frame_count,
            feature_count: audio.feature_count,
        })
        .collect();
    Gemma4UnifiedRuntimeInputs {
        images,
        audios,
        videos: Vec::new(),
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

        let (expanded, image_spans, audio_spans) =
            expand_image_and_audio(&config, &tokens, &images, &audios).expect("expansion");

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
        assert_eq!(expanded.len(), 3 + 258 + 5);
        // Boundary tokens landed where the spans say they did.
        assert_eq!(expanded[image.replacement_start], 255999); // boi
        assert_eq!(expanded[audio.replacement_start], 256000); // boa
    }

    #[test]
    fn rejects_placeholder_count_mismatch() {
        let config = processor_config();
        let tokens = [10u32, 258880, 11]; // one image marker
        let error = expand_image_and_audio(
            &config,
            &tokens,
            &[], // but zero image inputs
            &[],
        )
        .expect_err("mismatch should fail");
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(error.1.error.message.contains("image placeholder"));
    }
}
