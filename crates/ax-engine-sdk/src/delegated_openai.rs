use std::fmt;
use std::io::{BufRead, BufReader, Cursor, Read};

use base64::Engine as _;
use image::ImageReader;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DEFAULT_MAX_DELEGATED_DATA_URI_ENCODED_BYTES: usize = 96 * 1024 * 1024;
pub const DEFAULT_MAX_DELEGATED_IMAGE_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_MAX_DELEGATED_IMAGES: usize = 40;
pub const DEFAULT_MAX_DELEGATED_IMAGE_PIXELS: u64 = 100_000_000;
pub const DEFAULT_MAX_DELEGATED_TOTAL_PIXELS: u64 = 1_000_000_000;
pub const DEFAULT_MAX_DELEGATED_SSE_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DelegatedImageLimits {
    pub max_encoded_bytes: usize,
    pub max_decoded_bytes: usize,
    pub max_images: usize,
    pub max_pixels_per_image: u64,
    pub max_total_pixels: u64,
}

impl Default for DelegatedImageLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: DEFAULT_MAX_DELEGATED_DATA_URI_ENCODED_BYTES,
            max_decoded_bytes: DEFAULT_MAX_DELEGATED_IMAGE_BYTES,
            max_images: DEFAULT_MAX_DELEGATED_IMAGES,
            max_pixels_per_image: DEFAULT_MAX_DELEGATED_IMAGE_PIXELS,
            max_total_pixels: DEFAULT_MAX_DELEGATED_TOTAL_PIXELS,
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ValidatedDataUri {
    canonical: String,
    mime: String,
    decoded_bytes: usize,
    width: u32,
    height: u32,
}

impl ValidatedDataUri {
    pub fn parse(
        value: impl Into<String>,
        limits: DelegatedImageLimits,
    ) -> Result<Self, DelegatedOpenAiValidationError> {
        let value = value.into();
        if value.len() > limits.max_encoded_bytes {
            return Err(DelegatedOpenAiValidationError::EncodedImageTooLarge {
                actual: value.len(),
                maximum: limits.max_encoded_bytes,
            });
        }
        if !value.starts_with("data:") {
            return Err(DelegatedOpenAiValidationError::RemoteImageUrl);
        }
        let (metadata, encoded) = value
            .split_once(',')
            .ok_or(DelegatedOpenAiValidationError::MalformedDataUri)?;
        let metadata = metadata
            .strip_prefix("data:")
            .ok_or(DelegatedOpenAiValidationError::RemoteImageUrl)?;
        let mut fields = metadata.split(';');
        let mime = fields
            .next()
            .filter(|value| !value.is_empty())
            .ok_or(DelegatedOpenAiValidationError::MalformedDataUri)?
            .to_string();
        if !matches!(mime.as_str(), "image/png" | "image/jpeg") {
            return Err(DelegatedOpenAiValidationError::UnsupportedImageMime { mime });
        }
        if fields.next() != Some("base64") || fields.next().is_some() {
            return Err(DelegatedOpenAiValidationError::DataUriMustUseBase64);
        }
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| DelegatedOpenAiValidationError::InvalidBase64)?;
        if decoded.is_empty() {
            return Err(DelegatedOpenAiValidationError::EmptyImage);
        }
        if decoded.len() > limits.max_decoded_bytes {
            return Err(DelegatedOpenAiValidationError::DecodedImageTooLarge {
                actual: decoded.len(),
                maximum: limits.max_decoded_bytes,
            });
        }

        let format = image::guess_format(&decoded)
            .map_err(|_| DelegatedOpenAiValidationError::UnrecognizedImage)?;
        let expected_format = match mime.as_str() {
            "image/png" => image::ImageFormat::Png,
            "image/jpeg" => image::ImageFormat::Jpeg,
            _ => {
                return Err(DelegatedOpenAiValidationError::UnsupportedImageMime { mime });
            }
        };
        if format != expected_format {
            return Err(DelegatedOpenAiValidationError::ImageMimeMismatch {
                declared: mime.clone(),
                detected: format!("{format:?}").to_ascii_lowercase(),
            });
        }
        let (width, height) = ImageReader::with_format(Cursor::new(&decoded), format)
            .into_dimensions()
            .map_err(|_| DelegatedOpenAiValidationError::InvalidImageDimensions)?;
        let pixels = u64::from(width).saturating_mul(u64::from(height));
        if pixels == 0 {
            return Err(DelegatedOpenAiValidationError::InvalidImageDimensions);
        }
        if pixels > limits.max_pixels_per_image {
            return Err(DelegatedOpenAiValidationError::ImagePixelLimit {
                actual: pixels,
                maximum: limits.max_pixels_per_image,
            });
        }

        Ok(Self {
            canonical: value,
            mime,
            decoded_bytes: decoded.len(),
            width,
            height,
        })
    }

    pub fn as_str(&self) -> &str {
        &self.canonical
    }

    pub fn mime(&self) -> &str {
        &self.mime
    }

    pub fn decoded_bytes(&self) -> usize {
        self.decoded_bytes
    }

    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width).saturating_mul(u64::from(self.height))
    }
}

impl fmt::Debug for ValidatedDataUri {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedDataUri")
            .field("mime", &self.mime)
            .field("decoded_bytes", &self.decoded_bytes)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl Serialize for ValidatedDataUri {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.canonical)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DelegatedChatRole {
    System,
    User,
    Assistant,
}

impl DelegatedChatRole {
    pub fn parse(value: &str) -> Result<Self, DelegatedOpenAiValidationError> {
        match value {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            other => Err(DelegatedOpenAiValidationError::UnsupportedRole {
                role: other.to_string(),
            }),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum DelegatedChatContent {
    Text(String),
    Parts(Vec<DelegatedChatContentPart>),
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DelegatedChatContentPart {
    Text { text: String },
    ImageUrl { image_url: DelegatedImageUrl },
}

#[derive(Clone, Debug, Serialize)]
pub struct DelegatedImageUrl {
    pub url: ValidatedDataUri,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DelegatedChatMessage {
    pub role: DelegatedChatRole,
    pub content: DelegatedChatContent,
}

impl DelegatedChatMessage {
    pub fn text(role: DelegatedChatRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: DelegatedChatContent::Text(text.into()),
        }
    }

    pub fn parts(
        role: DelegatedChatRole,
        parts: impl IntoIterator<Item = DelegatedChatContentPart>,
    ) -> Self {
        Self {
            role,
            content: DelegatedChatContent::Parts(parts.into_iter().collect()),
        }
    }
}

pub fn validate_delegated_image_budget(
    messages: &[DelegatedChatMessage],
    limits: DelegatedImageLimits,
) -> Result<(), DelegatedOpenAiValidationError> {
    let images = messages
        .iter()
        .filter_map(|message| match &message.content {
            DelegatedChatContent::Parts(parts) => Some(parts),
            DelegatedChatContent::Text(_) => None,
        })
        .flatten()
        .filter_map(|part| match part {
            DelegatedChatContentPart::ImageUrl { image_url } => Some(&image_url.url),
            DelegatedChatContentPart::Text { .. } => None,
        })
        .collect::<Vec<_>>();
    if images.len() > limits.max_images {
        return Err(DelegatedOpenAiValidationError::ImageCountLimit {
            actual: images.len(),
            maximum: limits.max_images,
        });
    }
    let total_pixels = images.iter().fold(0_u64, |total, image| {
        total.saturating_add(image.pixel_count())
    });
    if total_pixels > limits.max_total_pixels {
        return Err(DelegatedOpenAiValidationError::TotalPixelLimit {
            actual: total_pixels,
            maximum: limits.max_total_pixels,
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum DelegatedOpenAiValidationError {
    #[error("delegated image URL must be an inline data URI")]
    RemoteImageUrl,
    #[error("delegated image data URI is malformed")]
    MalformedDataUri,
    #[error("delegated image data URI must use exactly one base64 parameter")]
    DataUriMustUseBase64,
    #[error("delegated image MIME type {mime} is not certified; expected image/png or image/jpeg")]
    UnsupportedImageMime { mime: String },
    #[error("delegated image data URI contains invalid base64")]
    InvalidBase64,
    #[error("delegated image must not be empty")]
    EmptyImage,
    #[error("delegated encoded image has {actual} bytes; maximum is {maximum}")]
    EncodedImageTooLarge { actual: usize, maximum: usize },
    #[error("delegated decoded image has {actual} bytes; maximum is {maximum}")]
    DecodedImageTooLarge { actual: usize, maximum: usize },
    #[error("delegated image bytes are not a recognized image")]
    UnrecognizedImage,
    #[error("delegated image MIME {declared} does not match detected {detected}")]
    ImageMimeMismatch { declared: String, detected: String },
    #[error("delegated image dimensions are invalid")]
    InvalidImageDimensions,
    #[error("delegated image has {actual} pixels; per-image maximum is {maximum}")]
    ImagePixelLimit { actual: u64, maximum: u64 },
    #[error("delegated request has {actual} images; maximum is {maximum}")]
    ImageCountLimit { actual: usize, maximum: usize },
    #[error("delegated request has {actual} total pixels; maximum is {maximum}")]
    TotalPixelLimit { actual: u64, maximum: u64 },
    #[error("delegated chat role {role} is not supported")]
    UnsupportedRole { role: String },
}

#[derive(Debug)]
pub struct DelegatedOpenAiStreamChunk {
    pub text: String,
    pub finish_reason: Option<String>,
    pub prompt_token_count: Option<u32>,
    pub output_token_count: Option<u32>,
}

pub struct DelegatedOpenAiStreamHandle {
    endpoint: String,
    reader: BufReader<Box<dyn Read + Send>>,
    max_frame_bytes: usize,
    done: bool,
}

impl DelegatedOpenAiStreamHandle {
    pub fn new(
        endpoint: impl Into<String>,
        reader: Box<dyn Read + Send>,
        max_frame_bytes: usize,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            reader: BufReader::new(reader),
            max_frame_bytes,
            done: false,
        }
    }

    pub fn next_chunk(
        &mut self,
    ) -> Result<Option<DelegatedOpenAiStreamChunk>, DelegatedOpenAiSseError> {
        if self.done {
            return Ok(None);
        }
        loop {
            let mut line = Vec::new();
            let bytes_read = self.reader.read_until(b'\n', &mut line).map_err(|source| {
                DelegatedOpenAiSseError::Read {
                    endpoint: self.endpoint.clone(),
                    source,
                }
            })?;
            if bytes_read == 0 {
                return Err(DelegatedOpenAiSseError::EndedBeforeDone {
                    endpoint: self.endpoint.clone(),
                });
            }
            if line.len() > self.max_frame_bytes {
                return Err(DelegatedOpenAiSseError::FrameTooLarge {
                    endpoint: self.endpoint.clone(),
                    actual: line.len(),
                    maximum: self.max_frame_bytes,
                });
            }
            let line = std::str::from_utf8(&line)
                .map_err(|_| DelegatedOpenAiSseError::InvalidUtf8 {
                    endpoint: self.endpoint.clone(),
                })?
                .trim_end_matches(['\r', '\n']);
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.strip_prefix(' ').unwrap_or(data);
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                self.done = true;
                return Ok(None);
            }

            let event: DelegatedOpenAiSseEvent = serde_json::from_str(data).map_err(|source| {
                DelegatedOpenAiSseError::InvalidJson {
                    endpoint: self.endpoint.clone(),
                    source,
                }
            })?;
            if event.error.is_some() {
                return Err(DelegatedOpenAiSseError::ProviderError {
                    endpoint: self.endpoint.clone(),
                });
            }
            let has_usage = event.usage.is_some();
            let choice = event.choices.into_iter().next();
            if choice.is_none() && !has_usage {
                return Err(DelegatedOpenAiSseError::MissingChoice {
                    endpoint: self.endpoint.clone(),
                });
            }
            let choice = choice.unwrap_or_default();
            return Ok(Some(DelegatedOpenAiStreamChunk {
                text: choice
                    .delta
                    .and_then(|delta| delta.content)
                    .unwrap_or(choice.text),
                finish_reason: choice.finish_reason,
                prompt_token_count: event.usage.as_ref().map(|usage| usage.prompt_tokens),
                output_token_count: event.usage.as_ref().map(|usage| usage.completion_tokens),
            }));
        }
    }
}

impl fmt::Debug for DelegatedOpenAiStreamHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DelegatedOpenAiStreamHandle")
            .field("endpoint", &self.endpoint)
            .field("max_frame_bytes", &self.max_frame_bytes)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum DelegatedOpenAiSseError {
    #[error("delegated OpenAI SSE read from {endpoint} failed: {source}")]
    Read {
        endpoint: String,
        #[source]
        source: std::io::Error,
    },
    #[error("delegated OpenAI SSE frame from {endpoint} exceeded {maximum} bytes ({actual})")]
    FrameTooLarge {
        endpoint: String,
        actual: usize,
        maximum: usize,
    },
    #[error("delegated OpenAI SSE frame from {endpoint} was not valid UTF-8")]
    InvalidUtf8 { endpoint: String },
    #[error("delegated OpenAI SSE frame from {endpoint} was not valid JSON: {source}")]
    InvalidJson {
        endpoint: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("delegated OpenAI SSE provider at {endpoint} returned an error event")]
    ProviderError { endpoint: String },
    #[error("delegated OpenAI SSE event from {endpoint} had neither a choice nor usage")]
    MissingChoice { endpoint: String },
    #[error("delegated OpenAI SSE stream from {endpoint} ended before [DONE]")]
    EndedBeforeDone { endpoint: String },
}

#[derive(Debug, Deserialize)]
struct DelegatedOpenAiSseEvent {
    #[serde(default)]
    choices: Vec<DelegatedOpenAiSseChoice>,
    #[serde(default)]
    usage: Option<DelegatedOpenAiUsage>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
struct DelegatedOpenAiSseChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    delta: Option<DelegatedOpenAiSseDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DelegatedOpenAiSseDelta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DelegatedOpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use base64::Engine as _;
    use image::{DynamicImage, ImageFormat, RgbImage};

    use super::*;

    fn png_data_uri() -> String {
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 3, image::Rgb([1, 2, 3])))
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap_or_else(|error| panic!("test PNG should encode: {error}"));
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes.into_inner())
        )
    }

    #[test]
    fn validated_data_uri_keeps_exact_wire_value_without_debug_leak() {
        let value = png_data_uri();
        let image = ValidatedDataUri::parse(value.clone(), DelegatedImageLimits::default())
            .unwrap_or_else(|error| panic!("valid image should pass: {error}"));
        assert_eq!(image.as_str(), value);
        assert_eq!(image.pixel_count(), 6);
        assert!(!format!("{image:?}").contains("base64"));
        assert_eq!(
            serde_json::to_value(&image)
                .unwrap_or_else(|error| panic!("image should serialize: {error}")),
            value
        );
    }

    #[test]
    fn validated_data_uri_rejects_remote_and_mime_mismatch() {
        assert_eq!(
            ValidatedDataUri::parse(
                "https://example.test/image.png",
                DelegatedImageLimits::default()
            ),
            Err(DelegatedOpenAiValidationError::RemoteImageUrl)
        );
        let jpeg_label = png_data_uri().replacen("image/png", "image/jpeg", 1);
        assert!(matches!(
            ValidatedDataUri::parse(jpeg_label, DelegatedImageLimits::default()),
            Err(DelegatedOpenAiValidationError::ImageMimeMismatch { .. })
        ));
    }

    #[test]
    fn stream_accepts_role_usage_and_done_but_rejects_early_eof() {
        let body = concat!(
            ": heartbeat\n",
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data:{\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\r\n\r\n",
            "data: [DONE]\n\n"
        );
        let mut stream = DelegatedOpenAiStreamHandle::new(
            "http://127.0.0.1/v1/chat/completions",
            Box::new(Cursor::new(body.as_bytes().to_vec())),
            4096,
        );
        assert_eq!(
            stream
                .next_chunk()
                .unwrap_or_else(|error| panic!("role should parse: {error}"))
                .map(|chunk| chunk.text),
            Some(String::new())
        );
        assert_eq!(
            stream
                .next_chunk()
                .unwrap_or_else(|error| panic!("content should parse: {error}"))
                .map(|chunk| chunk.text),
            Some("ok".to_string())
        );
        let usage = stream
            .next_chunk()
            .unwrap_or_else(|error| panic!("usage should parse: {error}"))
            .unwrap_or_else(|| panic!("usage chunk should exist"));
        assert_eq!(usage.prompt_token_count, Some(2));
        assert_eq!(usage.output_token_count, Some(1));
        assert!(stream.next_chunk().is_ok_and(|chunk| chunk.is_none()));

        let mut truncated = DelegatedOpenAiStreamHandle::new(
            "http://127.0.0.1/v1/chat/completions",
            Box::new(Cursor::new(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n".to_vec(),
            )),
            4096,
        );
        assert!(truncated.next_chunk().is_ok());
        assert!(matches!(
            truncated.next_chunk(),
            Err(DelegatedOpenAiSseError::EndedBeforeDone { .. })
        ));
    }
}
