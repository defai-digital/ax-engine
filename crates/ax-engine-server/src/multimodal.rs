//! Gemma 4 unified multimodal preprocessing for the OpenAI-compatible endpoint.
//!
//! The Gemma 4 unified model graph is "encoder-free": there is no SigLIP/CLIP
//! vision tower and no mel-spectrogram audio front-end. Image patches are fed
//! directly into a LayerNorm/Linear/LayerNorm + factorized-position-embedding
//! connector, and audio is a raw 16 kHz waveform chunked into fixed-size frames.
//! This module turns raw media (base64 `data:` URIs) into the exact tensors the
//! [`ax_engine_core::gemma4_unified`] runtime expects, mirroring the reference
//! `processing_gemma4_unified.py`.
//!
//! Scope: image (PNG/JPEG), audio (PCM WAV or MP3), and video (animated GIF plus
//! MP4/WebM when `ffmpeg` is available on the server `PATH`). Video frames reuse
//! the image patchify path at a lower per-frame soft-token budget with `mm:ss`
//! timestamps, matching the reference `Gemma4UnifiedVideoProcessor`. Other audio
//! formats (AAC/OGG/FLAC) are out of scope — `/v1/generate` still accepts fully
//! pre-computed tensors for those.

use std::path::Path;

use ax_engine_sdk::{
    Gemma4UnifiedAudioProcessor, Gemma4UnifiedProcessorConfig, Gemma4UnifiedVisionProcessor,
};
use base64::Engine as _;
use serde_json::Value;

/// Failure modes for media decoding/preprocessing. Callers map these onto HTTP
/// 400 responses.
#[derive(Debug)]
pub(crate) enum MediaError {
    /// The bytes could not be decoded as the declared media type.
    Decode(String),
    /// The request asked for a modality/format this preview does not handle.
    Unsupported(String),
    /// The model's processor config could not be loaded or is incomplete.
    Config(String),
}

impl std::fmt::Display for MediaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MediaError::Decode(message) => write!(f, "{message}"),
            MediaError::Unsupported(message) => write!(f, "{message}"),
            MediaError::Config(message) => write!(f, "{message}"),
        }
    }
}

/// Pixel rescale/normalize parameters taken from `preprocessor_config.json`.
///
/// The encoder-free connector applies no normalization itself, so whatever the
/// reference HF processor does must happen here. Defaults match the Gemma 4
/// image processor (`do_rescale = true`, `rescale_factor = 1/255`,
/// `do_normalize = false`, `mean = std = 0.5`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ImageNormalization {
    pub do_rescale: bool,
    pub rescale_factor: f32,
    pub do_normalize: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ImageNormalization {
    fn default() -> Self {
        Self {
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: false,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

/// Image tensors ready to attach to a `Gemma4UnifiedImageRuntimeInput`.
#[derive(Debug)]
pub(crate) struct PreprocessedImage {
    /// Original decoded width/height — drives `compute_soft_tokens` and span
    /// expansion in core.
    pub width: u32,
    pub height: u32,
    /// `[num_patches * model_patch_size^2 * 3]`, row-major per patch as
    /// `[row][col][channel]`.
    pub pixel_values: Vec<f32>,
    /// `[num_patches]` of `[patch_x, patch_y]`. Every entry is valid (no `-1`
    /// padding) — we send exactly `soft_token_count` patches.
    pub pixel_position_ids: Vec<[i32; 2]>,
}

/// Reference video defaults (`Gemma4UnifiedVideoProcessor`): up to 32 frames at
/// 70 soft tokens each, with a 2 fps fallback when frame timing is unavailable.
pub(crate) const DEFAULT_VIDEO_MAX_FRAMES: usize = 32;
pub(crate) const DEFAULT_VIDEO_SOFT_TOKENS: u32 = 70;
pub(crate) const DEFAULT_VIDEO_FPS: f32 = 2.0;

/// Resource bounds for the ffmpeg inline-video path. Frames are downscaled to
/// at most this side length before piping (the patchify resize target never
/// exceeds it for realistic aspect ratios, so quality is unaffected), and the
/// piped PNG stream is capped so a pathological video cannot exhaust server
/// RAM — `Command::output` buffers the whole stream. Videos whose decoded
/// stream exceeds the cap are sampled from the decoded prefix.
const FFMPEG_VIDEO_MAX_FRAME_SIDE: u32 = 1600;
const FFMPEG_VIDEO_MAX_OUTPUT_BYTES: u64 = 512 * 1024 * 1024;

/// Strict decode bounds for client-supplied still images and GIF frames. The
/// patchify resize target is far below this side length, so quality is
/// unaffected; without a strict dimension cap a small highly-compressed file
/// can decode to an arbitrarily large buffer (`Limits::default()` only sets a
/// non-strict 512 MiB alloc cap that decoders may ignore).
const IMAGE_MAX_DECODE_SIDE: u32 = 8192;

fn image_decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(IMAGE_MAX_DECODE_SIDE);
    limits.max_image_height = Some(IMAGE_MAX_DECODE_SIDE);
    limits
}

/// One decoded video frame with its timestamp (seconds from the start).
#[derive(Debug)]
pub(crate) struct VideoFrame {
    pub image: image::RgbImage,
    pub timestamp_seconds: f32,
}

/// Video tensors ready to attach to a `Gemma4UnifiedVideoRuntimeInput`. The
/// per-frame patches are concatenated; every frame shares `soft_tokens_per_frame`.
#[derive(Debug)]
pub(crate) struct PreprocessedVideo {
    pub frame_count: u32,
    pub soft_tokens_per_frame: u32,
    /// `[frame_count * soft_tokens_per_frame * model_patch_size^2 * 3]`.
    pub pixel_values: Vec<f32>,
    pub pixel_position_ids: Vec<[i32; 2]>,
}

/// Everything needed to preprocess Gemma 4 unified chat media: the core processor
/// config (token ids + image/audio params), image normalization, and the
/// video-specific vision config (lower per-frame soft-token budget + frame cap).
#[derive(Debug)]
pub(crate) struct MediaProcessors {
    pub config: Gemma4UnifiedProcessorConfig,
    pub normalization: ImageNormalization,
    pub video_vision: Gemma4UnifiedVisionProcessor,
    pub video_max_frames: usize,
}

/// Audio tensors ready to attach to a `Gemma4UnifiedAudioRuntimeInput`.
#[derive(Debug)]
pub(crate) struct PreprocessedAudio {
    /// Sample count at the model's target sample rate — drives
    /// `audio_soft_tokens` / span expansion in core.
    pub sample_count: u32,
    /// `[frame_count * feature_count]`.
    pub input_features: Vec<f32>,
    pub frame_count: u32,
    pub feature_count: u32,
}

/// Load the Gemma 4 unified processor config, image normalization, and the
/// video processor params from a model artifacts directory (`config.json` +
/// `preprocessor_config.json`).
pub(crate) fn load_processor_config(model_dir: &Path) -> Result<MediaProcessors, MediaError> {
    let model_cfg = read_json(&model_dir.join("config.json"))?;
    let processor_cfg = read_json(&model_dir.join("preprocessor_config.json"))
        .or_else(|_| read_json(&model_dir.join("processor_config.json")))?;
    let config =
        Gemma4UnifiedProcessorConfig::from_model_and_processor_config(&model_cfg, &processor_cfg)
            .map_err(|error| MediaError::Config(error.to_string()))?;
    let normalization = image_normalization_from(&processor_cfg)?;
    let (video_vision, video_max_frames) = video_processor_from(&processor_cfg, &config.vision);
    Ok(MediaProcessors {
        config,
        normalization,
        video_vision,
        video_max_frames,
    })
}

/// Derive the video-specific vision processor (a lower per-frame soft-token
/// budget than images) and frame cap from `preprocessor_config.json`. Defaults
/// match the reference `Gemma4UnifiedVideoProcessor`: 70 soft tokens, 32 frames.
fn video_processor_from(
    processor_cfg: &Value,
    image_vision: &Gemma4UnifiedVisionProcessor,
) -> (Gemma4UnifiedVisionProcessor, usize) {
    let block = processor_cfg
        .get("video_processor")
        .filter(|value| value.is_object());
    let max_soft_tokens = block
        .and_then(|value| value.get("max_soft_tokens"))
        .and_then(Value::as_u64)
        .map(|value| value as u32)
        .unwrap_or(DEFAULT_VIDEO_SOFT_TOKENS);
    let max_frames = block
        .and_then(|value| value.get("num_frames"))
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(DEFAULT_VIDEO_MAX_FRAMES);
    let video_vision = Gemma4UnifiedVisionProcessor {
        patch_size: image_vision.patch_size,
        model_patch_size: image_vision.model_patch_size,
        pooling_kernel_size: image_vision.pooling_kernel_size,
        max_soft_tokens,
    };
    (video_vision, max_frames)
}

/// Decode an inline base64 `data:` URI into `(mime, bytes)`. Only base64 data
/// URIs are accepted — remote `http(s)` URLs are rejected so the server never
/// performs outbound fetches on a client's behalf.
pub(crate) fn decode_data_uri(uri: &str) -> Result<(String, Vec<u8>), MediaError> {
    let rest = uri.strip_prefix("data:").ok_or_else(|| {
        MediaError::Unsupported(
            "only inline base64 data: URIs are supported for media (no remote URLs)".to_string(),
        )
    })?;
    let (meta, data) = rest
        .split_once(',')
        .ok_or_else(|| MediaError::Decode("malformed data: URI (missing comma)".to_string()))?;
    if !meta.contains("base64") {
        return Err(MediaError::Unsupported(
            "data: URI media must be base64-encoded".to_string(),
        ));
    }
    let mime = meta.split(';').next().unwrap_or_default().to_string();
    let bytes = decode_base64(data)?;
    Ok((mime, bytes))
}

/// Decode a raw (non-URI) standard base64 payload, e.g. OpenAI `input_audio.data`.
pub(crate) fn decode_base64(data: &str) -> Result<Vec<u8>, MediaError> {
    base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .map_err(|error| MediaError::Decode(format!("invalid base64 media payload: {error}")))
}

fn read_json(path: &Path) -> Result<Value, MediaError> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| MediaError::Config(format!("{}: {error}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|error| MediaError::Config(format!("{}: {error}", path.display())))
}

fn image_normalization_from(processor_cfg: &Value) -> Result<ImageNormalization, MediaError> {
    let mut norm = ImageNormalization::default();
    let block = processor_cfg
        .get("image_processor")
        .filter(|value| value.is_object())
        .unwrap_or(processor_cfg);
    if let Some(value) = block.get("do_rescale").and_then(Value::as_bool) {
        norm.do_rescale = value;
    }
    if let Some(value) = block.get("rescale_factor").and_then(Value::as_f64) {
        norm.rescale_factor = value as f32;
    }
    if let Some(value) = block.get("do_normalize").and_then(Value::as_bool) {
        norm.do_normalize = value;
    }
    if let Some(mean) = rgb_triplet(block.get("image_mean")) {
        norm.mean = mean;
    }
    if let Some(std) = rgb_triplet(block.get("image_std")) {
        norm.std = std;
    }
    // A non-positive, subnormal, or non-finite std channel would corrupt
    // every pixel (inf/NaN, overflow, or sign-flipped values); reject the
    // checkpoint config up front instead. `is_normal` also excludes the
    // subnormal window (~1e-45..1e-38) where the value passes a `> 0` check
    // but `pixel / std` still overflows f32 to inf. normalize_channel
    // divides by std relying on this.
    if norm.do_normalize
        && let Some(channel) = norm
            .std
            .iter()
            .find(|value| !value.is_normal() || **value < 0.0)
    {
        return Err(MediaError::Config(format!(
            "preprocessor_config.json image_std contains a channel that is not a positive \
             normal float32 value (parsed as {channel}); cannot normalize image pixels"
        )));
    }
    Ok(norm)
}

fn rgb_triplet(value: Option<&Value>) -> Option<[f32; 3]> {
    let array = value?.as_array()?;
    if array.len() != 3 {
        return None;
    }
    let mut out = [0f32; 3];
    for (slot, item) in out.iter_mut().zip(array) {
        *slot = item.as_f64()? as f32;
    }
    Some(out)
}

/// Decode and patchify an image into the encoder-free vision connector's input.
///
/// Mirrors `processing_gemma4_unified.py`: aspect-ratio-preserving resize so the
/// patch grid fits the soft-token budget, rescale/normalize per config, then
/// reshape pixels directly into `model_patch_size` patches with a 2D position
/// grid. The patch count is kept identical to core's `compute_soft_tokens`.
pub(crate) fn preprocess_image(
    bytes: &[u8],
    vision: &Gemma4UnifiedVisionProcessor,
    normalization: &ImageNormalization,
) -> Result<PreprocessedImage, MediaError> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| MediaError::Decode(format!("failed to probe image format: {error}")))?;
    reader.limits(image_decode_limits());
    let decoded = reader
        .decode()
        .map_err(|error| MediaError::Decode(format!("failed to decode image: {error}")))?
        .to_rgb8();
    let (width, height) = (decoded.width(), decoded.height());
    let (pixel_values, pixel_position_ids) = patchify_rgb(&decoded, vision, normalization)?;
    Ok(PreprocessedImage {
        width,
        height,
        pixel_values,
        pixel_position_ids,
    })
}

/// Resize one already-decoded RGB frame, then reshape it into the connector's
/// patches and 2D positions. Shared by the image and video paths.
fn patchify_rgb(
    frame: &image::RgbImage,
    vision: &Gemma4UnifiedVisionProcessor,
    normalization: &ImageNormalization,
) -> Result<(Vec<f32>, Vec<[i32; 2]>), MediaError> {
    let (width, height) = (frame.width(), frame.height());
    if width == 0 || height == 0 {
        return Err(MediaError::Decode(
            "image has zero width or height".to_string(),
        ));
    }
    if vision.patch_size == 0 || vision.pooling_kernel_size == 0 || vision.max_soft_tokens == 0 {
        return Err(MediaError::Config(
            "vision processor has zero patch_size, pooling_kernel_size, or max_soft_tokens"
                .to_string(),
        ));
    }

    // `model_patch_size == patch_size * pooling_kernel_size`; use it for both the
    // resize unit (core parity) and patch extraction (internal consistency).
    let patch = vision.patch_size * vision.pooling_kernel_size;
    let (target_w, target_h) = vision.resize_target(width, height);
    let resized = if target_w == width && target_h == height {
        std::borrow::Cow::Borrowed(frame)
    } else {
        std::borrow::Cow::Owned(image::imageops::resize(
            frame,
            target_w,
            target_h,
            image::imageops::FilterType::CatmullRom,
        ))
    };

    let grid_w = target_w / patch;
    let grid_h = target_h / patch;
    let total_patches = (grid_w as usize) * (grid_h as usize);
    let keep = total_patches.min(vision.max_soft_tokens as usize);
    let patch_dim = (patch as usize) * (patch as usize) * 3;

    let mut pixel_values = Vec::with_capacity(keep * patch_dim);
    let mut pixel_position_ids = Vec::with_capacity(keep);
    'patches: for patch_y in 0..grid_h {
        for patch_x in 0..grid_w {
            if pixel_position_ids.len() >= keep {
                break 'patches;
            }
            for row in 0..patch {
                for col in 0..patch {
                    let pixel = resized.get_pixel(patch_x * patch + col, patch_y * patch + row);
                    for channel in 0..3 {
                        pixel_values.push(normalize_channel(
                            pixel.0[channel],
                            channel,
                            normalization,
                        ));
                    }
                }
            }
            pixel_position_ids.push([patch_x as i32, patch_y as i32]);
        }
    }

    Ok((pixel_values, pixel_position_ids))
}

/// Decode an animated container into RGB frames with timestamps, uniformly
/// sampled down to `max_frames`. GIF timestamps accumulate per-frame delays; if
/// the container carries no usable timing, frames fall back to
/// `DEFAULT_VIDEO_FPS`. MP4/WebM are decoded by an optional `ffmpeg` process:
/// video codecs are not MLX tensor kernels, while the downstream Gemma4 tensor
/// path stays native MLX.
pub(crate) fn decode_video_frames(
    bytes: &[u8],
    max_frames: usize,
) -> Result<Vec<VideoFrame>, MediaError> {
    if looks_like_gif(bytes) {
        return decode_gif_video_frames(bytes, max_frames);
    }
    if looks_like_ffmpeg_video(bytes) {
        return decode_video_frames_ffmpeg(bytes, max_frames);
    }
    Err(MediaError::Unsupported(
        "inline video must be GIF, MP4, or WebM; MP4/WebM require ffmpeg on PATH, or send pre-extracted frame tensors via /v1/generate".to_string(),
    ))
}

fn decode_gif_video_frames(bytes: &[u8], max_frames: usize) -> Result<Vec<VideoFrame>, MediaError> {
    use image::{AnimationDecoder, ImageDecoder};

    let mut decoder =
        image::codecs::gif::GifDecoder::new(std::io::Cursor::new(bytes)).map_err(|_| {
            MediaError::Unsupported(
                "inline GIF video could not be decoded; MP4/WebM require ffmpeg on PATH, or send pre-extracted frame tensors via /v1/generate".to_string(),
            )
        })?;
    decoder
        .set_limits(image_decode_limits())
        .map_err(|error| MediaError::Decode(format!("failed to decode video frames: {error}")))?;
    // Bound cumulative decoded bytes with the same budget as the ffmpeg path;
    // oversize GIFs are sampled from the decoded prefix instead of buffering
    // every frame.
    let mut frames = Vec::new();
    let mut decoded_bytes: u64 = 0;
    for frame in decoder.into_frames() {
        let frame = frame.map_err(|error| {
            MediaError::Decode(format!("failed to decode video frames: {error}"))
        })?;
        decoded_bytes += frame.buffer().as_raw().len() as u64;
        frames.push(frame);
        if decoded_bytes > FFMPEG_VIDEO_MAX_OUTPUT_BYTES {
            break;
        }
    }
    if frames.is_empty() {
        return Err(MediaError::Decode("video has no frames".to_string()));
    }

    // Cumulative timestamp (seconds) at the start of each frame.
    let mut timestamps = Vec::with_capacity(frames.len());
    let mut elapsed = 0.0f32;
    let mut images = Vec::with_capacity(frames.len());
    for frame in frames {
        let (numer, denom) = frame.delay().numer_denom_ms();
        timestamps.push(elapsed);
        elapsed += if denom == 0 {
            0.0
        } else {
            numer as f32 / denom as f32 / 1000.0
        };
        images.push(image::DynamicImage::ImageRgba8(frame.into_buffer()).to_rgb8());
    }
    // GIFs without frame delays report zero duration; fall back to a fixed fps.
    if elapsed <= 0.0 {
        for (index, timestamp) in timestamps.iter_mut().enumerate() {
            *timestamp = index as f32 / DEFAULT_VIDEO_FPS;
        }
    }

    Ok(sample_frame_indices(images.len(), max_frames)
        .into_iter()
        .map(|index| VideoFrame {
            image: images[index].clone(),
            timestamp_seconds: timestamps[index],
        })
        .collect())
}

fn decode_video_frames_ffmpeg(
    bytes: &[u8],
    max_frames: usize,
) -> Result<Vec<VideoFrame>, MediaError> {
    decode_video_frames_with_ffmpeg_path(bytes, max_frames, "ffmpeg")
}

fn decode_video_frames_with_ffmpeg_path(
    bytes: &[u8],
    max_frames: usize,
    ffmpeg_path: impl AsRef<Path>,
) -> Result<Vec<VideoFrame>, MediaError> {
    let input = TempVideoInput::new(bytes)?;
    // Downscale before piping: `min(iw, SIDE)` never upscales, and
    // `force_original_aspect_ratio=decrease` keeps the aspect ratio when one
    // axis hits the cap. `showinfo` must stay after `scale` in the chain so
    // its `pts_time` lines describe exactly the frames that reach the pipe.
    let video_filter = format!(
        "scale=w=min(iw\\,{side}):h=min(ih\\,{side}):force_original_aspect_ratio=decrease,showinfo",
        side = FFMPEG_VIDEO_MAX_FRAME_SIDE
    );
    let output = std::process::Command::new(ffmpeg_path.as_ref())
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("info")
        .arg("-i")
        .arg(input.path())
        .arg("-map")
        .arg("0:v:0")
        .arg("-an")
        .arg("-sn")
        .arg("-vf")
        .arg(video_filter)
        .arg("-vsync")
        .arg("0")
        .arg("-fs")
        .arg(FFMPEG_VIDEO_MAX_OUTPUT_BYTES.to_string())
        .arg("-f")
        .arg("image2pipe")
        .arg("-vcodec")
        .arg("png")
        .arg("pipe:1")
        .output()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                MediaError::Unsupported(
                    "inline MP4/WebM video requires ffmpeg on PATH to extract frames; install ffmpeg or send pre-extracted frame tensors via /v1/generate".to_string(),
                )
            } else {
                MediaError::Decode(format!("failed to run ffmpeg video decoder: {error}"))
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let message = stderr.lines().last().unwrap_or("ffmpeg failed");
        return Err(MediaError::Decode(format!(
            "failed to decode inline video with ffmpeg: {message}"
        )));
    }

    let pngs = split_png_stream(&output.stdout)?;
    if pngs.is_empty() {
        return Err(MediaError::Decode(
            "ffmpeg decoded no video frames".to_string(),
        ));
    }
    let timestamps = parse_ffmpeg_showinfo_timestamps(&output.stderr);
    // Sample before decoding: the pipe can carry thousands of PNGs while the
    // model consumes at most `max_frames`, so only the sampled frames are
    // worth the PNG-to-RGB decode (the GIF path samples the same way).
    let indices = sample_frame_indices(pngs.len(), max_frames);
    let mut frames = Vec::with_capacity(indices.len());
    for index in indices {
        let image = image::load_from_memory_with_format(pngs[index], image::ImageFormat::Png)
            .map_err(|error| {
                MediaError::Decode(format!("failed to decode ffmpeg PNG frame: {error}"))
            })?
            .to_rgb8();
        let timestamp_seconds = timestamps
            .get(index)
            .copied()
            .flatten()
            .unwrap_or(index as f32 / DEFAULT_VIDEO_FPS);
        frames.push(VideoFrame {
            image,
            timestamp_seconds,
        });
    }
    Ok(frames)
}

struct TempVideoInput {
    path: std::path::PathBuf,
}

impl TempVideoInput {
    fn new(bytes: &[u8]) -> Result<Self, MediaError> {
        // The timestamp alone is not unique: concurrent requests can observe
        // the same SystemTime tick (and a clock that went backwards collapses
        // it to 0), so a per-process counter breaks the tie — a collision
        // would let `fs::write` truncate another request's staged video.
        static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut path = std::env::temp_dir();
        let unique = format!(
            "ax-engine-inline-video-{}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0),
            SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        path.push(unique);
        std::fs::write(&path, bytes).map_err(|error| {
            MediaError::Decode(format!(
                "failed to stage inline video for decoding at {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self { path })
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempVideoInput {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn looks_like_gif(bytes: &[u8]) -> bool {
    bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")
}

fn looks_like_ffmpeg_video(bytes: &[u8]) -> bool {
    looks_like_isobmff(bytes) || looks_like_webm(bytes)
}

fn looks_like_isobmff(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[4..8] == b"ftyp"
}

fn looks_like_webm(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3])
}

fn split_png_stream(bytes: &[u8]) -> Result<Vec<&[u8]>, MediaError> {
    const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    let mut frames = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let Some(start_offset) = find_bytes(&bytes[pos..], PNG_SIGNATURE) else {
            break;
        };
        let start = pos + start_offset;
        let mut cursor = start + PNG_SIGNATURE.len();
        loop {
            if cursor + 12 > bytes.len() {
                return Err(MediaError::Decode(
                    "truncated PNG frame from ffmpeg".to_string(),
                ));
            }
            let len = u32::from_be_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]) as usize;
            let chunk_type_start = cursor + 4;
            let chunk_data_start = cursor + 8;
            let next = chunk_data_start
                .checked_add(len)
                .and_then(|value| value.checked_add(4))
                .ok_or_else(|| {
                    MediaError::Decode("PNG frame from ffmpeg is too large".to_string())
                })?;
            if next > bytes.len() {
                return Err(MediaError::Decode(
                    "truncated PNG frame from ffmpeg".to_string(),
                ));
            }
            let chunk_type = &bytes[chunk_type_start..chunk_type_start + 4];
            cursor = next;
            if chunk_type == b"IEND" {
                frames.push(&bytes[start..cursor]);
                pos = cursor;
                break;
            }
        }
    }
    Ok(frames)
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Parse per-frame `pts_time:` values from ffmpeg `showinfo` stderr, keyed by
/// the `n:` frame number on the same line. Keying by `n` (instead of line
/// position) means a malformed or dropped line yields a gap for that one frame
/// rather than silently shifting every later frame onto the wrong timestamp.
/// Only `Parsed_showinfo` lines are read so `-v info` metadata dumps (which
/// echo container strings) cannot inject entries.
fn parse_ffmpeg_showinfo_timestamps(stderr: &[u8]) -> Vec<Option<f32>> {
    // Backstop against absurd `n:` values resizing the Vec; the `-fs` output
    // cap keeps real streams far below this.
    const MAX_FRAME_INDEX: usize = 1 << 20;
    let mut timestamps: Vec<Option<f32>> = Vec::new();
    for line in String::from_utf8_lossy(stderr).lines() {
        let Some((prefix, fields)) = line.split_once(']') else {
            continue;
        };
        if !prefix.contains("Parsed_showinfo") {
            continue;
        }
        let Some(frame_index) = parse_showinfo_field(fields, "n:")
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|index| *index < MAX_FRAME_INDEX)
        else {
            continue;
        };
        let Some(pts_seconds) =
            parse_showinfo_field(fields, "pts_time:").and_then(|value| value.parse::<f32>().ok())
        else {
            continue;
        };
        if frame_index >= timestamps.len() {
            timestamps.resize(frame_index + 1, None);
        }
        timestamps[frame_index] = Some(pts_seconds);
    }
    timestamps
}

/// Value of a `label:value` field in a showinfo line, tolerating the padding
/// spaces ffmpeg inserts between the label and the value (`n:   4`).
fn parse_showinfo_field<'a>(fields: &'a str, label: &str) -> Option<&'a str> {
    let start = fields.find(label)? + label.len();
    let rest = fields[start..].trim_start();
    let end = rest
        .find(|ch: char| ch.is_ascii_whitespace())
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Uniformly pick at most `max` indices from `[0, len)`, anchored at the first
/// and last frame. Mirrors the reference `Gemma4UnifiedVideoProcessor._sample_frames`
/// (`np.linspace(0, len - 1, max).round()`, round-ties-to-even like numpy) so AX
/// selects the same frames the model was calibrated on, and agrees with the
/// Python SDK's `_sample_video_frames`. A plain `i * len / max` floor never
/// includes the last frame and shifts most indices.
fn sample_frame_indices(len: usize, max: usize) -> Vec<usize> {
    if len <= max {
        return (0..len).collect();
    }
    if max <= 1 {
        return vec![0];
    }
    (0..max)
        .map(|i| {
            let pos = i as f64 * (len - 1) as f64 / (max - 1) as f64;
            pos.round_ties_even() as usize
        })
        .collect()
}

/// Patchify each decoded frame (sharing one patch grid) into the video
/// connector's concatenated tensor. Mirrors the per-frame branch of
/// `Gemma4UnifiedVideoProcessor`.
pub(crate) fn preprocess_video_frames(
    frames: &[VideoFrame],
    vision: &Gemma4UnifiedVisionProcessor,
    normalization: &ImageNormalization,
) -> Result<PreprocessedVideo, MediaError> {
    if frames.is_empty() {
        return Err(MediaError::Decode("video has no frames".to_string()));
    }

    let mut pixel_values = Vec::new();
    let mut pixel_position_ids = Vec::new();
    let mut soft_tokens_per_frame: Option<u32> = None;
    for frame in frames {
        let (frame_values, frame_positions) = patchify_rgb(&frame.image, vision, normalization)?;
        let count = frame_positions.len() as u32;
        match soft_tokens_per_frame {
            None => soft_tokens_per_frame = Some(count),
            Some(expected) if expected != count => {
                return Err(MediaError::Unsupported(
                    "video frames must share dimensions so every frame yields the same soft-token count".to_string(),
                ));
            }
            _ => {}
        }
        pixel_values.extend(frame_values);
        pixel_position_ids.extend(frame_positions);
    }

    let soft_tokens_per_frame = soft_tokens_per_frame.unwrap_or(0);
    if soft_tokens_per_frame == 0 {
        return Err(MediaError::Decode(
            "video frames produced zero soft tokens".to_string(),
        ));
    }

    Ok(PreprocessedVideo {
        frame_count: frames.len() as u32,
        soft_tokens_per_frame,
        pixel_values,
        pixel_position_ids,
    })
}

fn normalize_channel(value: u8, channel: usize, normalization: &ImageNormalization) -> f32 {
    let mut pixel = value as f32;
    if normalization.do_rescale {
        pixel *= normalization.rescale_factor;
    }
    if normalization.do_normalize {
        // std is a positive normal float, guaranteed by the image
        // normalization config validation, so this division cannot overflow.
        pixel = (pixel - normalization.mean[channel]) / normalization.std[channel];
    }
    pixel
}

/// Decode inline chat audio into the encoder-free audio connector's fixed-size
/// frames. The container is sniffed from magic bytes rather than the caller's
/// declared format: `RIFF` → PCM WAV via hound, ID3 tag or MPEG frame sync →
/// MP3 via symphonia. Other formats (AAC/OGG/FLAC) stay unsupported.
pub(crate) fn preprocess_audio(
    bytes: &[u8],
    audio: &Gemma4UnifiedAudioProcessor,
) -> Result<PreprocessedAudio, MediaError> {
    if bytes.starts_with(b"RIFF") {
        return preprocess_wav(bytes, audio);
    }
    if looks_like_mp3(bytes) {
        return preprocess_mp3(bytes, audio);
    }
    Err(MediaError::Unsupported(
        "inline audio must be PCM WAV or MP3; other formats require pre-computed audio tensors via /v1/generate".to_string(),
    ))
}

/// MP3 streams start with an ID3v2 tag or directly with an MPEG audio frame
/// sync (11 set bits: `0xFF` then the top 3 bits of the next byte).
fn looks_like_mp3(bytes: &[u8]) -> bool {
    bytes.starts_with(b"ID3") || (bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0)
}

/// Decode a PCM WAV stream and chunk the (mono, resampled) waveform into the
/// encoder-free audio connector's fixed-size frames.
///
/// Mirrors `Gemma4UnifiedAudioFeatureExtractor`: downmix to mono, resample to
/// `sampling_rate`, zero-pad to a multiple of `audio_samples_per_token`, and
/// reshape into `[frames, audio_samples_per_token]`.
pub(crate) fn preprocess_wav(
    bytes: &[u8],
    audio: &Gemma4UnifiedAudioProcessor,
) -> Result<PreprocessedAudio, MediaError> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .map_err(|error| MediaError::Decode(format!("failed to decode WAV audio: {error}")))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved = read_wav_samples(reader, spec)?;
    let mono = downmix_to_mono(&interleaved, channels);
    let resampled = resample_linear(&mono, spec.sample_rate, audio.sampling_rate);
    frame_resampled_waveform(resampled, audio)
}

/// Decode an MP3 stream via symphonia and chunk the (mono, resampled) waveform
/// into the connector's frames, sharing the WAV path's downmix/resample/framing
/// semantics. MP3 is lossy and pads with encoder delay, so sample counts differ
/// slightly from the original PCM.
fn preprocess_mp3(
    bytes: &[u8],
    audio: &Gemma4UnifiedAudioProcessor,
) -> Result<PreprocessedAudio, MediaError> {
    let (mono, source_rate) = decode_mp3_mono(bytes, audio)?;
    let resampled = resample_linear(&mono, source_rate, audio.sampling_rate);
    frame_resampled_waveform(resampled, audio)
}

/// Decode an MP3 stream into a mono waveform at its source rate. Decoding stops
/// once enough source samples exist to fill the model's `audio_seq_length`
/// frame cap, so an oversized upload cannot expand unbounded in memory.
fn decode_mp3_mono(
    bytes: &[u8],
    audio: &Gemma4UnifiedAudioProcessor,
) -> Result<(Vec<f32>, u32), MediaError> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let decode_error = |error: SymphoniaError| -> MediaError {
        MediaError::Decode(format!("failed to decode MP3 audio: {error}"))
    };

    let stream = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes.to_vec())),
        Default::default(),
    );
    let mut hint = Hint::new();
    hint.with_extension("mp3");
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(decode_error)?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| MediaError::Decode("MP3 stream has no audio track".to_string()))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(decode_error)?;

    let mut source_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut mono = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            // End of stream surfaces as an UnexpectedEof I/O error.
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(error) => return Err(decode_error(error)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A corrupt frame is skippable; the decoder stays usable.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => return Err(decode_error(error)),
        };
        let spec = *decoded.spec();
        source_rate = spec.rate;
        let channels = spec.channels.count().max(1);
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);
        mono.extend(
            buffer
                .samples()
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32),
        );
        if let Some(cap) = mp3_decode_sample_cap(audio, source_rate)
            && mono.len() >= cap
        {
            break;
        }
    }

    if source_rate == 0 {
        return Err(MediaError::Decode(
            "MP3 stream declares no sample rate".to_string(),
        ));
    }
    Ok((mono, source_rate))
}

/// Source-rate sample count that fills the model's `audio_seq_length` frame cap
/// (plus one frame of slack); `None` when the model declares no cap.
fn mp3_decode_sample_cap(audio: &Gemma4UnifiedAudioProcessor, source_rate: u32) -> Option<usize> {
    let frames = audio.audio_seq_length? as u64;
    let per_token = audio.audio_samples_per_token.max(1) as u64;
    let target_samples = (frames + 1) * per_token;
    let target_rate = audio.sampling_rate.max(1) as u64;
    Some((target_samples * source_rate.max(1) as u64).div_ceil(target_rate) as usize)
}

/// Zero-pad a mono waveform (already at the model rate) to a multiple of
/// `audio_samples_per_token` and reshape it into
/// `[frames, audio_samples_per_token]`, capped at `audio_seq_length` frames.
fn frame_resampled_waveform(
    resampled: Vec<f32>,
    audio: &Gemma4UnifiedAudioProcessor,
) -> Result<PreprocessedAudio, MediaError> {
    let per_token = audio.audio_samples_per_token.max(1) as usize;
    if resampled.is_empty() {
        return Err(MediaError::Decode("audio contains no samples".to_string()));
    }
    let mut frame_count = resampled.len().div_ceil(per_token);
    if let Some(limit) = audio.audio_seq_length {
        frame_count = frame_count.min(limit as usize);
    }
    if frame_count == 0 {
        return Err(MediaError::Decode(
            "audio resolves to zero frames".to_string(),
        ));
    }

    let mut input_features = vec![0f32; frame_count * per_token];
    let copy_len = (frame_count * per_token).min(resampled.len());
    input_features[..copy_len].copy_from_slice(&resampled[..copy_len]);

    Ok(PreprocessedAudio {
        sample_count: resampled.len() as u32,
        input_features,
        frame_count: frame_count as u32,
        feature_count: per_token as u32,
    })
}

fn read_wav_samples(
    reader: hound::WavReader<std::io::Cursor<&[u8]>>,
    spec: hound::WavSpec,
) -> Result<Vec<f32>, MediaError> {
    let mut reader = reader;
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .filter_map(Result::ok)
            .collect::<Vec<_>>(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
            let scale = if scale == 0.0 { 1.0 } else { scale };
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|value| value as f32 / scale)
                .collect::<Vec<_>>()
        }
    };
    Ok(samples)
}

fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Linear-interpolation resampler. Adequate for the encoder-free connector's
/// learned projection; not a band-limited sinc resampler. Inputs already at the
/// target rate pass through untouched.
fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() || from_rate == 0 || to_rate == 0 {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let last = input.len() - 1;
    let mut out = Vec::with_capacity(out_len);
    for index in 0..out_len {
        let source = index as f64 / ratio;
        let lower = source.floor() as usize;
        let frac = (source - lower as f64) as f32;
        let a = input[lower.min(last)];
        let b = input[(lower + 1).min(last)];
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vision() -> Gemma4UnifiedVisionProcessor {
        Gemma4UnifiedVisionProcessor {
            patch_size: 16,
            model_patch_size: 48,
            pooling_kernel_size: 3,
            max_soft_tokens: 280,
        }
    }

    fn solid_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        let buffer = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
        let mut bytes: Vec<u8> = Vec::new();
        image::DynamicImage::ImageRgb8(buffer)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode png");
        bytes
    }

    #[test]
    fn resize_target_matches_core_soft_token_count() {
        // For each of these, the number of kept patches must equal what core's
        // compute_soft_tokens returns from the original dimensions.
        for (w, h, expected) in [(224, 224, 256), (1024, 256, 264), (3, 900, 280)] {
            let vision = vision();
            let core = vision.compute_soft_tokens(w, h).unwrap() as usize;
            assert_eq!(core, expected, "core soft tokens for {w}x{h}");

            let png = solid_png(w.max(1), h.max(1), [10, 20, 30]);
            let pre = preprocess_image(&png, &vision, &ImageNormalization::default()).unwrap();
            assert_eq!(
                pre.pixel_position_ids.len(),
                core,
                "kept patches must equal core soft tokens for {w}x{h}"
            );
            let patch_dim = (vision.patch_size * vision.pooling_kernel_size) as usize;
            let patch_dim = patch_dim * patch_dim * 3;
            assert_eq!(
                pre.pixel_values.len(),
                pre.pixel_position_ids.len() * patch_dim
            );
        }
    }

    #[test]
    fn patch_positions_are_row_major_x_then_y() {
        let vision = vision();
        // Small images are upscaled to fill the patch budget (reference
        // behavior), so derive the actual grid from the resize target and verify
        // the [x, y] ordering is y-outer / x-inner with no truncation.
        let png = solid_png(96, 48, [128, 128, 128]);
        let pre = preprocess_image(&png, &vision, &ImageNormalization::default()).unwrap();

        let patch = vision.patch_size * vision.pooling_kernel_size;
        let (target_w, target_h) = vision.resize_target(96, 48);
        let grid_w = (target_w / patch) as i32;
        let grid_h = (target_h / patch) as i32;
        assert!((grid_w as usize) * (grid_h as usize) <= vision.max_soft_tokens as usize);
        assert_eq!(pre.pixel_position_ids.len(), (grid_w * grid_h) as usize);

        for (index, position) in pre.pixel_position_ids.iter().enumerate() {
            let index = index as i32;
            assert_eq!(*position, [index % grid_w, index / grid_w]);
        }
    }

    #[test]
    fn normalization_config_rejects_non_positive_or_non_finite_std_channels() {
        // Zero divides every pixel into inf/NaN, a negative channel (sign
        // typo) silently flips pixel signs, a subnormal f32 (1e-40) passes a
        // naive > 0 check but overflows the division to inf, and a value too
        // large for f32 (1e40) casts to inf. All must be rejected at load
        // time. (JSON cannot express NaN/inf directly; the non-finite arm is
        // reachable only via the f64 -> f32 cast.)
        for bad_channel in [0.0, -0.5, 1e-40, 1e40] {
            let config = serde_json::json!({
                "do_normalize": true,
                "image_std": [0.5, bad_channel, 0.5]
            });
            let error = image_normalization_from(&config).expect_err(&format!(
                "image_std channel {bad_channel} must be rejected when do_normalize is set"
            ));
            assert!(format!("{error:?}").contains("image_std"));
        }

        // Without do_normalize the std values are unused, so the same config
        // stays loadable.
        let normalize_off = serde_json::json!({
            "do_normalize": false,
            "image_std": [0.5, 0.0, 0.5]
        });
        assert!(
            !image_normalization_from(&normalize_off)
                .unwrap()
                .do_normalize
        );

        let valid = serde_json::json!({
            "do_normalize": true,
            "image_std": [0.5, 0.5, 0.5]
        });
        assert!(image_normalization_from(&valid).unwrap().do_normalize);
    }

    #[test]
    fn default_normalization_is_rescale_only() {
        let norm = ImageNormalization::default();
        assert_eq!(normalize_channel(255, 0, &norm), 1.0);
        assert_eq!(normalize_channel(0, 0, &norm), 0.0);
        let normed = ImageNormalization {
            do_normalize: true,
            ..ImageNormalization::default()
        };
        assert_eq!(normalize_channel(255, 0, &normed), 1.0);
        assert_eq!(normalize_channel(0, 0, &normed), -1.0);
    }

    fn audio() -> Gemma4UnifiedAudioProcessor {
        Gemma4UnifiedAudioProcessor {
            sampling_rate: 16000,
            audio_samples_per_token: 640,
            audio_seq_length: Some(1500),
        }
    }

    fn wav_mono(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut bytes: Vec<u8> = Vec::new();
        {
            let mut writer =
                hound::WavWriter::new(std::io::Cursor::new(&mut bytes), spec).expect("writer");
            for &sample in samples {
                writer.write_sample(sample).expect("write");
            }
            writer.finalize().expect("finalize");
        }
        bytes
    }

    fn wav_16k_mono(samples: &[f32]) -> Vec<u8> {
        wav_mono(samples, 16000)
    }

    #[test]
    fn wav_frames_match_core_soft_tokens() {
        let audio = audio();
        // 1600 samples @ 16k -> ceil(1600/640) = 3 frames.
        let wav = wav_16k_mono(&vec![0.25f32; 1600]);
        let pre = preprocess_wav(&wav, &audio).unwrap();
        let core = audio.compute_soft_tokens(pre.sample_count);
        assert_eq!(pre.frame_count, core);
        assert_eq!(pre.frame_count, 3);
        assert_eq!(pre.feature_count, 640);
        assert_eq!(pre.input_features.len(), 3 * 640);
        // First samples preserved, tail zero-padded.
        assert_eq!(pre.input_features[0], 0.25);
        assert_eq!(*pre.input_features.last().unwrap(), 0.0);
    }

    #[test]
    fn wav_resampled_from_44100_to_16000_target_rate() {
        let audio = audio();
        // 1 second at 44.1kHz resamples to ~16000 samples at the model's 16kHz.
        let wav = wav_mono(&vec![0.1f32; 44_100], 44_100);
        let pre = preprocess_wav(&wav, &audio).unwrap();
        assert!(
            (pre.sample_count as i64 - 16_000).abs() <= 1,
            "expected ~16000 resampled samples, got {}",
            pre.sample_count
        );
        // Frame count follows the 16kHz sample count, not the source rate.
        assert_eq!(pre.frame_count, pre.sample_count.div_ceil(640));
        assert_eq!(pre.frame_count, audio.compute_soft_tokens(pre.sample_count));
        assert_eq!(pre.feature_count, 640);
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
    }

    #[test]
    fn mp3_mono_16k_decodes_and_frames_like_wav() {
        let audio = audio();
        // 0.5 s, 440 Hz tone at 0.5 amplitude, 16 kHz mono (lame, 64 kbps).
        let mp3 = include_bytes!("tests/fixtures/gemma4_golden/audio_tone_16k_mono.mp3");
        let pre = preprocess_audio(mp3, &audio).unwrap();

        // 8000 PCM samples plus MP3 encoder delay/padding (~0.576 s decoded).
        assert!(
            (7000..=11000).contains(&(pre.sample_count as usize)),
            "expected ~8000-9300 samples, got {}",
            pre.sample_count
        );
        assert_eq!(pre.frame_count, audio.compute_soft_tokens(pre.sample_count));
        assert_eq!(pre.feature_count, 640);

        // The decoded tone must carry real signal energy (a 0.5-amplitude sine
        // has RMS ~0.35; lossy decode and delay padding lower it slightly).
        let signal_len = (pre.sample_count as usize).min(pre.input_features.len());
        let energy = rms(&pre.input_features[..signal_len]);
        assert!(
            energy > 0.1 && energy < 1.0,
            "decoded MP3 tone RMS out of range: {energy}"
        );
    }

    #[test]
    fn mp3_stereo_44k_downmixes_and_resamples_to_16k() {
        let audio = audio();
        // 0.5 s stereo tone at 44.1 kHz resamples to ~8000 samples at 16 kHz.
        let mp3 = include_bytes!("tests/fixtures/gemma4_golden/audio_tone_44k_stereo.mp3");
        let pre = preprocess_audio(mp3, &audio).unwrap();

        assert!(
            (7000..=11000).contains(&(pre.sample_count as usize)),
            "expected ~8000-8900 resampled samples, got {}",
            pre.sample_count
        );
        assert_eq!(pre.frame_count, audio.compute_soft_tokens(pre.sample_count));
        assert_eq!(pre.feature_count, 640);
    }

    #[test]
    fn preprocess_audio_routes_wav_and_rejects_unknown_containers() {
        let audio = audio();

        // RIFF magic routes to the existing WAV path.
        let wav = wav_16k_mono(&vec![0.25f32; 1600]);
        let pre = preprocess_audio(&wav, &audio).unwrap();
        assert_eq!(pre.frame_count, 3);

        // Anything that is neither RIFF nor MP3 fails closed as unsupported.
        let error = preprocess_audio(b"OggS\x00\x02 not supported", &audio).unwrap_err();
        assert!(matches!(error, MediaError::Unsupported(_)));
        assert!(error.to_string().contains("PCM WAV or MP3"));
    }

    #[test]
    fn mp3_decode_cap_scales_with_source_rate() {
        let audio = audio(); // 16 kHz, 640 samples/token, 1500-frame cap
        let cap_16k = mp3_decode_sample_cap(&audio, 16000).unwrap();
        assert_eq!(cap_16k, 1501 * 640);
        let cap_32k = mp3_decode_sample_cap(&audio, 32000).unwrap();
        assert_eq!(cap_32k, 2 * cap_16k);

        let uncapped = Gemma4UnifiedAudioProcessor {
            sampling_rate: 16000,
            audio_samples_per_token: 640,
            audio_seq_length: None,
        };
        assert_eq!(mp3_decode_sample_cap(&uncapped, 16000), None);
    }

    #[test]
    fn image_decodes_jpeg_input() {
        // Fixtures elsewhere are PNG; confirm the JPEG decode path also works.
        let vision = Gemma4UnifiedVisionProcessor {
            patch_size: 4,
            model_patch_size: 8,
            pooling_kernel_size: 2,
            max_soft_tokens: 4,
        };
        let buffer = image::RgbImage::from_fn(16, 16, |x, y| {
            image::Rgb([(x * 8) as u8, (y * 8) as u8, 96])
        });
        let mut bytes: Vec<u8> = Vec::new();
        image::DynamicImage::ImageRgb8(buffer)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Jpeg,
            )
            .expect("encode jpeg");

        let pre = preprocess_image(&bytes, &vision, &ImageNormalization::default()).unwrap();
        let core = vision.compute_soft_tokens(16, 16).unwrap() as usize;
        assert_eq!(pre.pixel_position_ids.len(), core);
        let patch_dim = 8 * 8 * 3;
        assert_eq!(pre.pixel_values.len(), core * patch_dim);
        assert!(
            pre.pixel_values
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
    }

    fn animated_gif(frames: &[image::RgbImage]) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        {
            let mut encoder = image::codecs::gif::GifEncoder::new(std::io::Cursor::new(&mut bytes));
            for frame in frames {
                let rgba = image::DynamicImage::ImageRgb8(frame.clone()).to_rgba8();
                encoder
                    .encode_frame(image::Frame::new(rgba))
                    .expect("encode gif frame");
            }
        }
        bytes
    }

    #[test]
    fn video_frames_decode_and_concatenate_per_frame_patches() {
        let vision = vision();
        let frame_a = image::RgbImage::from_pixel(96, 48, image::Rgb([10, 20, 30]));
        let frame_b = image::RgbImage::from_pixel(96, 48, image::Rgb([40, 50, 60]));
        let gif = animated_gif(&[frame_a, frame_b]);

        let frames =
            decode_video_frames(&gif, DEFAULT_VIDEO_MAX_FRAMES).expect("decode gif frames");
        assert_eq!(frames.len(), 2);
        // First frame starts at t=0; second frame strictly later.
        assert_eq!(frames[0].timestamp_seconds, 0.0);
        assert!(frames[1].timestamp_seconds > 0.0);

        let pre =
            preprocess_video_frames(&frames, &vision, &ImageNormalization::default()).unwrap();
        assert_eq!(pre.frame_count, 2);
        assert!(pre.soft_tokens_per_frame > 0);

        let patch = (vision.patch_size * vision.pooling_kernel_size) as usize;
        let patch_dim = patch * patch * 3;
        let per_frame = pre.soft_tokens_per_frame as usize;
        assert_eq!(pre.pixel_position_ids.len(), 2 * per_frame);
        assert_eq!(pre.pixel_values.len(), 2 * per_frame * patch_dim);
    }

    #[test]
    fn png_pipe_and_showinfo_timestamps_are_parsed() {
        let frame_a = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        let frame_b = image::RgbImage::from_pixel(4, 4, image::Rgb([40, 50, 60]));
        let mut stream = Vec::new();
        for frame in [frame_a, frame_b] {
            let mut encoded = Vec::new();
            image::DynamicImage::ImageRgb8(frame)
                .write_to(
                    &mut std::io::Cursor::new(&mut encoded),
                    image::ImageFormat::Png,
                )
                .expect("encode png frame");
            stream.extend(encoded);
        }

        let frames = split_png_stream(&stream).expect("split PNG pipe");
        assert_eq!(frames.len(), 2);

        let timestamps = parse_ffmpeg_showinfo_timestamps(
            b"[Parsed_showinfo_0 @ x] n:0 pts:0 pts_time:0\n\
              [Parsed_showinfo_0 @ x] n:1 pts:512 pts_time:0.5\n",
        );
        assert_eq!(timestamps, vec![Some(0.0), Some(0.5)]);
    }

    #[test]
    fn showinfo_timestamps_key_by_frame_number_and_ignore_non_showinfo_lines() {
        // A dropped/malformed line must leave a gap at its own frame, not
        // shift later frames; metadata echoes of container strings must not
        // inject entries.
        let timestamps = parse_ffmpeg_showinfo_timestamps(
            b"[mov,mp4 @ x] title : n:0 pts_time:99\n\
              [Parsed_showinfo_0 @ x] n:   0 pts:0 pts_time:0\n\
              [Parsed_showinfo_0 @ x] n:   1 pts:512 pts_time:garbage\n\
              [Parsed_showinfo_0 @ x] n:   2 pts:1024 pts_time:1\n",
        );
        assert_eq!(timestamps, vec![Some(0.0), None, Some(1.0)]);
    }

    #[test]
    fn video_container_routing_rejects_unknown_formats() {
        let error = decode_video_frames(b"definitely not a video container", 4)
            .expect_err("unknown container bytes must be rejected");
        assert!(matches!(error, MediaError::Unsupported(_)));
        assert!(error.to_string().contains("GIF, MP4, or WebM"));

        // GIF magic routes to the in-process GIF decoder, never to ffmpeg
        // (the header may parse, so the failure can also surface from frame
        // decoding — both messages are GIF-path errors).
        let error = decode_video_frames(b"GIF89a then truncated garbage", 4)
            .expect_err("corrupt GIF must fail in the GIF decoder");
        let message = error.to_string();
        assert!(
            message.contains("inline GIF video could not be decoded")
                || message.contains("failed to decode video frames")
                || message.contains("video has no frames"),
            "expected a GIF-path error, got: {message}"
        );

        // ISO-BMFF magic routes to the ffmpeg path: the error mentions ffmpeg
        // (missing binary or failed decode) rather than the GIF decoder.
        let mut isobmff = vec![0u8; 16];
        isobmff[4..8].copy_from_slice(b"ftyp");
        let error = decode_video_frames(&isobmff, 4)
            .expect_err("truncated ISO-BMFF must fail in the ffmpeg path");
        let message = error.to_string();
        assert!(message.contains("ffmpeg"), "unexpected error: {message}");
        assert!(!message.contains("inline GIF video could not be decoded"));
    }

    #[cfg(unix)]
    #[test]
    fn ffmpeg_video_decode_path_samples_fake_png_pipe() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "ax-engine-fake-ffmpeg-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let frames_path = dir.join("frames.pngpipe");
        let ffmpeg_path = dir.join("ffmpeg");

        let mut stream = Vec::new();
        for rgb in [[10u8, 20, 30], [40, 50, 60], [70, 80, 90]] {
            let frame = image::RgbImage::from_pixel(4, 4, image::Rgb(rgb));
            let mut encoded = Vec::new();
            image::DynamicImage::ImageRgb8(frame)
                .write_to(
                    &mut std::io::Cursor::new(&mut encoded),
                    image::ImageFormat::Png,
                )
                .expect("encode png frame");
            stream.extend(encoded);
        }
        std::fs::write(&frames_path, stream).expect("write fake frame pipe");
        let script = format!(
            "#!/bin/sh\ncat '{}'\nprintf '%s\\n' '[Parsed_showinfo_0 @ x] n:0 pts_time:0' '[Parsed_showinfo_0 @ x] n:1 pts_time:0.5' '[Parsed_showinfo_0 @ x] n:2 pts_time:1' >&2\n",
            frames_path.display()
        );
        std::fs::write(&ffmpeg_path, script).expect("write fake ffmpeg");
        let mut permissions = std::fs::metadata(&ffmpeg_path)
            .expect("fake ffmpeg metadata")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&ffmpeg_path, permissions).expect("chmod fake ffmpeg");

        let frames = decode_video_frames_with_ffmpeg_path(b"fake mp4", 2, &ffmpeg_path)
            .expect("fake ffmpeg should decode frames");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].timestamp_seconds, 0.0);
        assert_eq!(frames[1].timestamp_seconds, 1.0);
        assert_eq!(frames[0].image.get_pixel(0, 0).0, [10, 20, 30]);
        assert_eq!(frames[1].image.get_pixel(0, 0).0, [70, 80, 90]);

        std::fs::remove_dir_all(dir).expect("cleanup fake ffmpeg dir");
    }

    #[test]
    fn video_uses_lower_soft_token_budget_than_images() {
        // A model with no video_processor block falls back to the 70-soft-token
        // video budget, which is lower than the image budget (280 here).
        let processor_cfg = serde_json::json!({
            "image_processor": {
                "patch_size": 16,
                "model_patch_size": 48,
                "pooling_kernel_size": 3,
                "max_soft_tokens": 280
            }
        });
        let image_vision = vision();
        let (video_vision, max_frames) = video_processor_from(&processor_cfg, &image_vision);
        assert_eq!(video_vision.max_soft_tokens, DEFAULT_VIDEO_SOFT_TOKENS);
        assert_eq!(max_frames, DEFAULT_VIDEO_MAX_FRAMES);
        assert!(video_vision.max_soft_tokens < image_vision.max_soft_tokens);
    }

    #[test]
    fn sample_frame_indices_caps_and_keeps_first() {
        assert_eq!(sample_frame_indices(3, 16), vec![0, 1, 2]);
        let sampled = sample_frame_indices(100, 16);
        assert_eq!(sampled.len(), 16);
        assert_eq!(sampled[0], 0);
        assert!(sampled.windows(2).all(|w| w[0] < w[1]));
        // Endpoint-anchored: the last sampled index is the final frame, matching
        // the reference linspace(0, len - 1, max).round().
        assert_eq!(*sampled.last().unwrap(), 99);
        assert_eq!(sample_frame_indices(5, 1), vec![0]);
    }

    #[test]
    fn sample_frame_indices_matches_reference_linspace() {
        // Reference: np.linspace(0, len - 1, max).round() (round ties to even).
        for (len, max) in [(100, 32), (60, 8), (7, 5), (101, 16)] {
            let got = sample_frame_indices(len, max);
            let expected: Vec<usize> = (0..max)
                .map(|i| {
                    let pos = i as f64 * (len - 1) as f64 / (max - 1) as f64;
                    pos.round_ties_even() as usize
                })
                .collect();
            assert_eq!(got, expected, "len={len} max={max}");
            assert_eq!(*got.first().unwrap(), 0);
            assert_eq!(*got.last().unwrap(), len - 1);
        }
    }

    fn parse_golden(
        golden: &Value,
    ) -> (Gemma4UnifiedVisionProcessor, Vec<f32>, Vec<[i32; 2]>, usize) {
        let cfg = &golden["config"];
        let vision = Gemma4UnifiedVisionProcessor {
            patch_size: cfg["patch_size"].as_u64().unwrap() as u32,
            model_patch_size: cfg["model_patch_size"].as_u64().unwrap() as u32,
            pooling_kernel_size: cfg["pooling_kernel_size"].as_u64().unwrap() as u32,
            max_soft_tokens: cfg["max_soft_tokens"].as_u64().unwrap() as u32,
        };
        let pixel_values = golden["pixel_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let positions = golden["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let pair = p.as_array().unwrap();
                [
                    pair[0].as_i64().unwrap() as i32,
                    pair[1].as_i64().unwrap() as i32,
                ]
            })
            .collect();
        // Image goldens carry `soft_tokens`; the video golden uses
        // `soft_tokens_per_frame` instead, so this is optional.
        let soft = golden
            .get("soft_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        (vision, pixel_values, positions, soft)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn golden_noresize_matches_reference_exactly() {
        let png = include_bytes!("tests/fixtures/gemma4_golden/image_noresize.png");
        let golden: Value = serde_json::from_str(include_str!(
            "tests/fixtures/gemma4_golden/golden_noresize.json"
        ))
        .unwrap();
        let (vision, expected_values, expected_positions, soft) = parse_golden(&golden);

        let pre = preprocess_image(png, &vision, &ImageNormalization::default()).unwrap();

        // No resize happens (scale == 1), so the patchify/normalize/position math
        // must match the reference numpy path bit-for-bit.
        assert_eq!(pre.pixel_position_ids.len(), soft);
        assert_eq!(pre.pixel_position_ids, expected_positions);
        assert_eq!(pre.pixel_values.len(), expected_values.len());
        let diff = max_abs_diff(&pre.pixel_values, &expected_values);
        assert!(diff < 1e-6, "max abs diff vs reference = {diff}");
    }

    #[test]
    fn golden_resize_matches_reference_within_tolerance() {
        let png = include_bytes!("tests/fixtures/gemma4_golden/image_resize.png");
        let golden: Value = serde_json::from_str(include_str!(
            "tests/fixtures/gemma4_golden/golden_resize.json"
        ))
        .unwrap();
        let (vision, expected_values, expected_positions, soft) = parse_golden(&golden);

        let pre = preprocess_image(png, &vision, &ImageNormalization::default()).unwrap();

        // Resize uses CatmullRom vs the reference's PIL BICUBIC: same patch grid
        // and positions, pixel values close but not bit-identical.
        assert_eq!(pre.pixel_position_ids.len(), soft);
        assert_eq!(pre.pixel_position_ids, expected_positions);
        let diff = max_abs_diff(&pre.pixel_values, &expected_values);
        assert!(
            diff < 0.12,
            "resize pixel diff vs PIL bicubic too large: {diff}"
        );
    }

    #[test]
    fn golden_audio_matches_reference_exactly() {
        let wav = include_bytes!("tests/fixtures/gemma4_golden/audio_noresize.wav");
        let golden: Value = serde_json::from_str(include_str!(
            "tests/fixtures/gemma4_golden/golden_audio_noresize.json"
        ))
        .unwrap();
        let audio = Gemma4UnifiedAudioProcessor {
            sampling_rate: golden["sample_rate"].as_u64().unwrap() as u32,
            audio_samples_per_token: golden["audio_samples_per_token"].as_u64().unwrap() as u32,
            audio_seq_length: None,
        };

        let pre = preprocess_wav(wav, &audio).unwrap();

        // Input is already at the model rate (no resample) and int16/32768
        // round-trips losslessly, so the framing must match the reference exactly.
        assert_eq!(
            pre.frame_count,
            golden["frame_count"].as_u64().unwrap() as u32
        );
        assert_eq!(
            pre.feature_count,
            golden["feature_count"].as_u64().unwrap() as u32
        );
        let expected: Vec<f32> = golden["input_features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(pre.input_features.len(), expected.len());
        let diff = max_abs_diff(&pre.input_features, &expected);
        assert!(diff < 1e-6, "audio diff vs reference = {diff}");
    }

    #[test]
    fn golden_video_matches_reference_exactly() {
        let frame0 = include_bytes!("tests/fixtures/gemma4_golden/video_noresize_frame0.png");
        let frame1 = include_bytes!("tests/fixtures/gemma4_golden/video_noresize_frame1.png");
        let golden: Value = serde_json::from_str(include_str!(
            "tests/fixtures/gemma4_golden/golden_video_noresize.json"
        ))
        .unwrap();
        let (vision, expected_values, expected_positions, _soft) = parse_golden(&golden);

        let to_frame = |bytes: &[u8]| VideoFrame {
            image: image::load_from_memory(bytes).unwrap().to_rgb8(),
            timestamp_seconds: 0.0,
        };
        let frames = vec![to_frame(frame0), to_frame(frame1)];
        let pre =
            preprocess_video_frames(&frames, &vision, &ImageNormalization::default()).unwrap();

        // Two distinct no-resize frames: per-frame patchify is exact, and this
        // validates frame concatenation order + per-frame position reset.
        assert_eq!(
            pre.frame_count,
            golden["frame_count"].as_u64().unwrap() as u32
        );
        assert_eq!(
            pre.soft_tokens_per_frame,
            golden["soft_tokens_per_frame"].as_u64().unwrap() as u32
        );
        assert_eq!(pre.pixel_position_ids, expected_positions);
        assert_eq!(pre.pixel_values.len(), expected_values.len());
        let diff = max_abs_diff(&pre.pixel_values, &expected_values);
        assert!(diff < 1e-6, "video diff vs reference = {diff}");
    }

    #[test]
    fn decode_data_uri_accepts_base64_and_rejects_remote() {
        let (mime, bytes) =
            decode_data_uri("data:image/png;base64,aGVsbG8=").expect("valid data uri");
        assert_eq!(mime, "image/png");
        assert_eq!(bytes, b"hello");

        let remote = decode_data_uri("https://example.com/cat.png").unwrap_err();
        assert!(matches!(remote, MediaError::Unsupported(_)));

        let not_base64 = decode_data_uri("data:image/png,hello").unwrap_err();
        assert!(matches!(not_base64, MediaError::Unsupported(_)));
    }

    #[test]
    fn resample_halves_length_when_downsampling() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = resample_linear(&input, 32000, 16000);
        assert_eq!(out.len(), 50);
        assert_eq!(out[0], 0.0);
    }
}
