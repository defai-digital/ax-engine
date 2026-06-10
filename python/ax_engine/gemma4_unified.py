from __future__ import annotations

import base64
import json
import math
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


_REMOTE_MEDIA_TIMEOUT_SECONDS = 10
_MAX_REMOTE_MEDIA_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Gemma4UnifiedMultimodalRequest:
    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    image_soft_token_counts: list[int]
    audio_soft_token_counts: list[int]
    video_soft_token_counts: list[int]
    video_frame_counts: list[int]


@dataclass(frozen=True)
class Gemma4UnifiedImageRequest:
    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    soft_token_counts: list[int]


@dataclass(frozen=True)
class Gemma4UnifiedAudioRequest:
    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    soft_token_counts: list[int]


@dataclass(frozen=True)
class Gemma4UnifiedVideoRequest:
    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    soft_token_counts: list[int]
    frame_counts: list[int]


@dataclass(frozen=True)
class _Gemma4UnifiedConfig:
    image_token_id: int
    audio_token_id: int
    video_token_id: int
    boi_token_id: int
    eoi_token_id: int
    boa_token_id: int
    eoa_token_id: int
    do_convert_rgb: bool
    do_resize: bool
    do_rescale: bool
    rescale_factor: float
    do_normalize: bool
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    patch_size: int
    model_patch_size: int
    pooling_kernel_size: int
    max_soft_tokens: int
    sampling_rate: int
    audio_samples_per_token: int
    audio_seq_length: int | None
    video_max_soft_tokens: int
    video_num_frames: int


def prepare_gemma4_unified_multimodal_request(
    model_dir: str | Path,
    input_tokens: list[int],
    *,
    images: list[Any] | None = None,
    audios: list[Any] | None = None,
    audio_sampling_rates: list[int] | None = None,
    videos: list[Any] | None = None,
    video_timestamp_token_ids: list[list[list[int]]] | None = None,
) -> Gemma4UnifiedMultimodalRequest:
    """Build a native Gemma4 unified request from tokenized placeholders.

    Raw media decoding stays outside AX's optimized model path. This helper
    prepares the processed tensor contract consumed by the native MLX runtime.
    Video timestamps are accepted as already-tokenized IDs so callers can keep
    tokenizer policy explicit. When provided, `video_timestamp_token_ids` must be
    shaped as `[video][frame][token]` and must contain as many video entries as
    videos provided, plus one frame entry per sampled frame after preprocessing.
    """

    config = _load_config(Path(model_dir))
    prepared_images = [] if images is None else list(images)
    prepared_audios = [] if audios is None else list(audios)
    prepared_videos = [] if videos is None else list(videos)
    processed_images = [_process_image(image, config) for image in prepared_images]
    processed_audios = _process_audios(prepared_audios, audio_sampling_rates, config)
    processed_videos = [_process_video(video, config) for video in prepared_videos]
    timestamp_ids = _normalize_video_timestamp_ids(
        video_timestamp_token_ids,
        [item["frame_count"] for item in processed_videos],
    )

    (
        expanded_tokens,
        image_spans,
        audio_spans,
        video_spans,
    ) = _expand_multimodal_placeholders(
        input_tokens,
        processed_images,
        processed_audios,
        processed_videos,
        timestamp_ids,
        config,
    )

    runtime_images = [
        {
            "span": span,
            "pixel_values": image["pixel_values"],
            "pixel_position_ids": image["pixel_position_ids"],
        }
        for span, image in zip(image_spans, processed_images)
    ]
    runtime_audios = [
        {
            "span": span,
            "input_features": audio["input_features"],
            "frame_count": audio["frame_count"],
            "feature_count": audio["feature_count"],
        }
        for span, audio in zip(audio_spans, processed_audios)
    ]
    runtime_videos = []
    for span, video in zip(video_spans, processed_videos):
        runtime_videos.append(
            {
                "span": span["span"],
                "soft_token_ranges": span["soft_token_ranges"],
                "pixel_values": [
                    value
                    for frame in video["frames"]
                    for value in frame["pixel_values"]
                ],
                "pixel_position_ids": [
                    position
                    for frame in video["frames"]
                    for position in frame["pixel_position_ids"]
                ],
                "frame_count": video["frame_count"],
            }
        )

    return Gemma4UnifiedMultimodalRequest(
        input_tokens=expanded_tokens,
        multimodal_inputs={
            "gemma4_unified": {
                "images": runtime_images,
                "audios": runtime_audios,
                "videos": runtime_videos,
            }
        },
        image_soft_token_counts=[item["soft_token_count"] for item in processed_images],
        audio_soft_token_counts=[item["frame_count"] for item in processed_audios],
        video_soft_token_counts=[
            sum(frame["soft_token_count"] for frame in item["frames"])
            for item in processed_videos
        ],
        video_frame_counts=[item["frame_count"] for item in processed_videos],
    )


def prepare_gemma4_unified_image_request(
    model_dir: str | Path,
    input_tokens: list[int],
    images: list[Any],
) -> Gemma4UnifiedImageRequest:
    request = prepare_gemma4_unified_multimodal_request(
        model_dir,
        input_tokens,
        images=images,
    )
    return Gemma4UnifiedImageRequest(
        input_tokens=request.input_tokens,
        multimodal_inputs=request.multimodal_inputs,
        soft_token_counts=request.image_soft_token_counts,
    )


def prepare_gemma4_unified_audio_request(
    model_dir: str | Path,
    input_tokens: list[int],
    audios: list[Any],
    *,
    sampling_rates: list[int] | None = None,
) -> Gemma4UnifiedAudioRequest:
    request = prepare_gemma4_unified_multimodal_request(
        model_dir,
        input_tokens,
        audios=audios,
        audio_sampling_rates=sampling_rates,
    )
    return Gemma4UnifiedAudioRequest(
        input_tokens=request.input_tokens,
        multimodal_inputs=request.multimodal_inputs,
        soft_token_counts=request.audio_soft_token_counts,
    )


def prepare_gemma4_unified_video_request(
    model_dir: str | Path,
    input_tokens: list[int],
    videos: list[Any],
    *,
    timestamp_token_ids: list[list[list[int]]] | None = None,
) -> Gemma4UnifiedVideoRequest:
    request = prepare_gemma4_unified_multimodal_request(
        model_dir,
        input_tokens,
        videos=videos,
        video_timestamp_token_ids=timestamp_token_ids,
    )
    return Gemma4UnifiedVideoRequest(
        input_tokens=request.input_tokens,
        multimodal_inputs=request.multimodal_inputs,
        soft_token_counts=request.video_soft_token_counts,
        frame_counts=request.video_frame_counts,
    )


def _load_config(model_dir: Path) -> _Gemma4UnifiedConfig:
    model_config_path = model_dir / "config.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Gemma4 unified config not found: {model_config_path}")

    # The image/audio processor params live in `preprocessor_config.json` for most
    # HF checkpoints, but combined processors save them to `processor_config.json`.
    # Accept either (preferring `preprocessor_config.json`, like the AX server's
    # `load_processor_config`) so the SDK loads the same models the server serves
    # and never silently reads the wrong file.
    processor_config_candidates = (
        model_dir / "preprocessor_config.json",
        model_dir / "processor_config.json",
    )
    processor_config_path = next(
        (path for path in processor_config_candidates if path.is_file()), None
    )
    if processor_config_path is None:
        raise FileNotFoundError(
            "Gemma4 unified processor config not found: expected one of "
            + ", ".join(str(path) for path in processor_config_candidates)
        )

    model_config = json.loads(model_config_path.read_text())
    processor_config = json.loads(processor_config_path.read_text())
    image_config = processor_config.get("image_processor") or {}
    video_config = processor_config.get("video_processor") or {}
    feature_config = (
        processor_config.get("feature_extractor")
        or processor_config.get("audio_feature_extractor")
        or {}
    )
    vision_config = model_config.get("vision_config") or {}
    audio_config = model_config.get("audio_config") or {}
    if image_config.get("image_processor_type") not in (
        None,
        "Gemma4UnifiedImageProcessor",
    ):
        raise ValueError(
            "Gemma4 unified image preprocessing requires Gemma4UnifiedImageProcessor"
        )
    if feature_config.get("feature_extractor_type") not in (
        None,
        "Gemma4UnifiedAudioFeatureExtractor",
    ):
        raise ValueError(
            "Gemma4 unified audio preprocessing requires "
            "Gemma4UnifiedAudioFeatureExtractor"
        )
    if video_config.get("video_processor_type") not in (
        None,
        "Gemma4UnifiedVideoProcessor",
    ):
        raise ValueError(
            "Gemma4 unified video preprocessing requires Gemma4UnifiedVideoProcessor"
        )

    sampling_rate = _optional_int(feature_config, "sampling_rate") or 16000
    audio_samples_per_token = (
        _optional_int(audio_config, "audio_samples_per_token")
        or _optional_int(feature_config, "audio_samples_per_token")
        or _optional_int(feature_config, "feature_size")
        or _optional_int(audio_config, "audio_embed_dim")
        or int(
            round(
                sampling_rate
                * float(processor_config.get("audio_ms_per_token", 40))
                / 1000
            )
        )
        or 640
    )

    do_normalize = bool(image_config.get("do_normalize", False))
    image_std = _triple(image_config.get("image_std", [0.5, 0.5, 0.5]))
    if do_normalize and any(
        not math.isfinite(channel) or channel == 0 for channel in image_std
    ):
        # A zero std channel would divide every pixel into inf/NaN and
        # silently corrupt the vision input; reject the checkpoint config.
        raise ValueError(
            "preprocessor_config.json image_std contains a zero or non-finite "
            f"channel {image_std!r}; cannot normalize image pixels"
        )

    return _Gemma4UnifiedConfig(
        image_token_id=_required_int(model_config, "image_token_id"),
        audio_token_id=_required_int(model_config, "audio_token_id"),
        video_token_id=_required_int(model_config, "video_token_id"),
        boi_token_id=_required_int(model_config, "boi_token_id"),
        eoi_token_id=_required_int(model_config, "eoi_token_id"),
        boa_token_id=_required_int(model_config, "boa_token_id"),
        eoa_token_id=_optional_int(model_config, "eoa_token_index")
        or _required_int(model_config, "eoa_token_id"),
        do_convert_rgb=bool(image_config.get("do_convert_rgb", True)),
        do_resize=bool(image_config.get("do_resize", True)),
        do_rescale=bool(image_config.get("do_rescale", True)),
        rescale_factor=float(image_config.get("rescale_factor", 1 / 255)),
        do_normalize=do_normalize,
        image_mean=_triple(image_config.get("image_mean", [0.5, 0.5, 0.5])),
        image_std=image_std,
        patch_size=_optional_int(image_config, "patch_size")
        or _required_int(vision_config, "patch_size"),
        model_patch_size=_optional_int(image_config, "model_patch_size")
        or _required_int(vision_config, "model_patch_size"),
        pooling_kernel_size=_optional_int(image_config, "pooling_kernel_size")
        or _required_int(vision_config, "pooling_kernel_size"),
        max_soft_tokens=_optional_int(image_config, "max_soft_tokens")
        or _optional_int(vision_config, "num_soft_tokens")
        or _required_int(vision_config, "default_output_length"),
        sampling_rate=sampling_rate,
        audio_samples_per_token=audio_samples_per_token,
        audio_seq_length=_optional_int(processor_config, "audio_seq_length") or 750,
        video_max_soft_tokens=_optional_int(video_config, "max_soft_tokens") or 70,
        video_num_frames=_optional_int(video_config, "num_frames") or 32,
    )


def _process_image(
    image: Any,
    config: _Gemma4UnifiedConfig,
    *,
    max_soft_tokens: int | None = None,
) -> dict[str, Any]:
    if max_soft_tokens is None:
        max_soft_tokens = config.max_soft_tokens
    pil_image = _load_pil_image(image)
    if config.do_convert_rgb:
        pil_image = pil_image.convert("RGB")
    elif pil_image.mode != "RGB":
        raise ValueError(
            "Gemma4 unified image preprocessing currently requires RGB images"
        )

    if config.do_resize:
        target_width, target_height = _resized_dimensions(
            pil_image.width,
            pil_image.height,
            config.patch_size,
            config.pooling_kernel_size,
            max_soft_tokens,
        )
        if target_width != pil_image.width or target_height != pil_image.height:
            from PIL import Image

            pil_image = pil_image.resize(
                (target_width, target_height),
                resample=Image.Resampling.BICUBIC,
            )

    if pil_image.width % config.model_patch_size != 0:
        raise ValueError(
            "Gemma4 unified image width must be divisible by model_patch_size "
            f"after resize: width={pil_image.width}, model_patch_size={config.model_patch_size}"
        )
    if pil_image.height % config.model_patch_size != 0:
        raise ValueError(
            "Gemma4 unified image height must be divisible by model_patch_size "
            f"after resize: height={pil_image.height}, model_patch_size={config.model_patch_size}"
        )

    pixels = _rgb_pixels(pil_image, config)
    patch_values, position_ids = _patchify_rgb(
        pixels, pil_image.width, pil_image.height, config
    )
    soft_token_count = len(position_ids)
    if soft_token_count > max_soft_tokens:
        patch_values = patch_values[:max_soft_tokens]
        position_ids = position_ids[:max_soft_tokens]
        soft_token_count = max_soft_tokens

    while len(position_ids) < max_soft_tokens:
        patch_values.append(
            [0.0] * (config.model_patch_size * config.model_patch_size * 3)
        )
        position_ids.append([-1, -1])

    return {
        "pixel_values": [value for patch in patch_values for value in patch],
        "pixel_position_ids": position_ids,
        "soft_token_count": soft_token_count,
    }


def _process_audios(
    audios: list[Any],
    sampling_rates: list[int] | None,
    config: _Gemma4UnifiedConfig,
) -> list[dict[str, Any]]:
    if sampling_rates is not None and len(sampling_rates) != len(audios):
        raise ValueError(
            "Gemma4 unified audio sampling_rates length must match audio count"
        )
    processed = []
    for idx, audio in enumerate(audios):
        waveform, sampling_rate = _load_audio_waveform(audio, config.sampling_rate)
        if sampling_rates is not None:
            sampling_rate = int(sampling_rates[idx])
        if sampling_rate != config.sampling_rate:
            waveform = _resample_waveform(waveform, sampling_rate, config.sampling_rate)
            sampling_rate = config.sampling_rate
        processed.append(_process_audio_waveform(waveform, sampling_rate, config))
    return processed


def _process_audio_waveform(
    waveform: Any,
    sampling_rate: int,
    config: _Gemma4UnifiedConfig,
) -> dict[str, Any]:
    if sampling_rate != config.sampling_rate:
        raise ValueError(
            "Gemma4 unified audio preprocessing requires waveform sampling_rate "
            f"{config.sampling_rate}; got {sampling_rate}. Resample before calling."
        )
    values = _flatten_audio_values(waveform)
    if not values:
        raise ValueError("Gemma4 unified audio waveform must not be empty")
    feature_count = config.audio_samples_per_token
    if feature_count <= 0:
        raise ValueError("Gemma4 unified audio_samples_per_token must be positive")
    pad_len = (-len(values)) % feature_count
    if pad_len:
        values.extend([0.0] * pad_len)
    frames = [
        values[start : start + feature_count]
        for start in range(0, len(values), feature_count)
    ]
    if config.audio_seq_length is not None:
        frames = frames[: config.audio_seq_length]
    if not frames:
        raise ValueError("Gemma4 unified audio produced no feature frames")
    return {
        "input_features": [value for frame in frames for value in frame],
        "frame_count": len(frames),
        "feature_count": feature_count,
    }


def _process_video(video: Any, config: _Gemma4UnifiedConfig) -> dict[str, Any]:
    frames = _sample_video_frames(_load_video_frames(video), config.video_num_frames)
    processed_frames = [
        _trim_video_frame(
            _process_image(
                frame,
                config,
                max_soft_tokens=config.video_max_soft_tokens,
            ),
            config,
        )
        for frame in frames
    ]
    first_soft_token_count = processed_frames[0]["soft_token_count"]
    if any(
        frame["soft_token_count"] != first_soft_token_count
        for frame in processed_frames
    ):
        raise ValueError(
            "Gemma4 unified video frames must produce a consistent soft-token count"
        )
    return {
        "frames": processed_frames,
        "frame_count": len(processed_frames),
    }


def _load_pil_image(image: Any):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Gemma4 unified image/video preprocessing requires Pillow; "
            "install ax-engine[multimodal] or pillow"
        ) from exc

    if isinstance(image, dict):
        image = _media_source_from_dict(image, "image")
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, bytes):
        return Image.open(BytesIO(image))
    if isinstance(image, str):
        if image.startswith("data:"):
            return Image.open(BytesIO(_decode_data_uri(image, ("image/",))))
        if _is_remote_url(image):
            return Image.open(BytesIO(_fetch_url_bytes(image)))
        if image.startswith("file://"):
            image = _file_url_to_path(image)
    return Image.open(image)


def _load_audio_waveform(
    audio: Any, default_sampling_rate: int
) -> tuple[list[float], int]:
    if isinstance(audio, dict):
        audio = _audio_source_from_dict(audio)
    if isinstance(audio, tuple) and len(audio) == 2:
        waveform, sampling_rate = audio
        return _flatten_audio_values(waveform), int(sampling_rate)
    if isinstance(audio, bytes):
        return _load_wav(BytesIO(audio))
    if isinstance(audio, str):
        if audio.startswith("data:"):
            return _load_wav(
                BytesIO(_decode_data_uri(audio, ("audio/", "application/")))
            )
        if _is_remote_url(audio):
            return _load_wav(BytesIO(_fetch_url_bytes(audio)))
        if audio.startswith("file://"):
            audio = _file_url_to_path(audio)
    if isinstance(audio, Path) or isinstance(audio, str):
        with Path(audio).open("rb") as handle:
            return _load_wav(handle)
    return _flatten_audio_values(audio), default_sampling_rate


def _load_wav(handle: Any) -> tuple[list[float], int]:
    try:
        with wave.open(handle, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            sampling_rate = wav.getframerate()
            raw = wav.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError(
            "Gemma4 unified audio preprocessing expects waveform samples or WAV audio"
        ) from exc
    if channels <= 0:
        raise ValueError("WAV audio must have at least one channel")
    samples = _decode_pcm_samples(raw, sample_width)
    if channels == 1:
        return samples, sampling_rate

    mono = []
    for index in range(0, len(samples), channels):
        frame = samples[index : index + channels]
        mono.append(sum(frame) / len(frame))
    return mono, sampling_rate


def _decode_pcm_samples(raw: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(value - 128) / 128.0 for value in raw]
    if sample_width == 2:
        return [
            int.from_bytes(raw[index : index + 2], "little", signed=True) / 32768.0
            for index in range(0, len(raw), 2)
        ]
    if sample_width == 4:
        return [
            int.from_bytes(raw[index : index + 4], "little", signed=True) / 2147483648.0
            for index in range(0, len(raw), 4)
        ]
    raise ValueError(f"unsupported WAV sample width: {sample_width}")


def _resample_waveform(
    waveform: list[float],
    source_rate: int,
    target_rate: int,
) -> list[float]:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("audio sampling rates must be positive")
    if source_rate == target_rate or not waveform:
        return waveform
    target_len = max(1, round(len(waveform) * target_rate / source_rate))
    if target_len == 1:
        return [waveform[0]]
    scale = (len(waveform) - 1) / (target_len - 1)
    output = []
    for index in range(target_len):
        source_pos = index * scale
        left = int(math.floor(source_pos))
        right = min(left + 1, len(waveform) - 1)
        frac = source_pos - left
        output.append(waveform[left] * (1.0 - frac) + waveform[right] * frac)
    return output


def _load_video_frames(video: Any) -> list[Any]:
    if isinstance(video, tuple) and len(video) == 2:
        video = video[0]
    if isinstance(video, (str, Path, bytes)):
        raise ValueError(
            "Gemma4 unified video preprocessing expects a sequence of decoded frames; "
            "encoded video files are not decoded by this helper"
        )
    try:
        frames = list(video)
    except TypeError as exc:
        raise ValueError(
            "Gemma4 unified video input must be a sequence of Pillow images, paths, "
            "or image bytes"
        ) from exc
    if not frames:
        raise ValueError("Gemma4 unified video input must contain at least one frame")
    return frames


def _media_source_from_dict(source: dict[str, Any], modality: str) -> Any:
    if "url" in source:
        return source["url"]
    nested = source.get(f"{modality}_url")
    if isinstance(nested, dict) and "url" in nested:
        return nested["url"]
    if isinstance(nested, str):
        return nested
    raise ValueError(
        f"Gemma4 unified {modality} source dict must contain url or {modality}_url.url"
    )


def _audio_source_from_dict(source: dict[str, Any]) -> Any:
    if "data" in source:
        audio_format = str(source.get("format", "wav")).lower()
        if audio_format not in {"wav", "wave"}:
            raise ValueError(
                "Gemma4 unified input_audio data currently supports WAV format only"
            )
        return base64.b64decode(str(source["data"]), validate=True)
    if "input_audio" in source and isinstance(source["input_audio"], dict):
        return _audio_source_from_dict(source["input_audio"])
    if "url" in source:
        return source["url"]
    audio_url = source.get("audio_url")
    if isinstance(audio_url, dict) and "url" in audio_url:
        return audio_url["url"]
    if isinstance(audio_url, str):
        return audio_url
    raise ValueError(
        "Gemma4 unified audio source dict must contain data, input_audio, url, "
        "or audio_url.url"
    )


def _decode_data_uri(uri: str, accepted_media_prefixes: tuple[str, ...]) -> bytes:
    if "," not in uri:
        raise ValueError("Gemma4 unified media data URI is missing a comma separator")
    header, payload = uri.split(",", 1)
    header_lower = header.lower()
    media_type = header_lower[5:].split(";", 1)[0]
    if accepted_media_prefixes and not any(
        media_type.startswith(prefix) for prefix in accepted_media_prefixes
    ):
        accepted = ", ".join(accepted_media_prefixes)
        raise ValueError(
            f"Gemma4 unified media data URI has unsupported media type "
            f"{media_type!r}; expected one of {accepted}"
        )
    if ";base64" in header_lower:
        return base64.b64decode(payload, validate=True)
    return urllib.parse.unquote_to_bytes(payload)


def _is_remote_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _fetch_url_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(
            url, timeout=_REMOTE_MEDIA_TIMEOUT_SECONDS
        ) as response:
            # Read at most the limit, then probe a single byte to detect overflow
            # rather than pulling the whole oversized body into memory first.
            data = response.read(_MAX_REMOTE_MEDIA_BYTES)
            overflowed = bool(response.read(1))
    except Exception as exc:
        raise ValueError(f"Gemma4 unified media URL fetch failed: {url}") from exc
    if overflowed:
        raise ValueError(
            "Gemma4 unified media URL response exceeded "
            f"{_MAX_REMOTE_MEDIA_BYTES} bytes"
        )
    return data


def _file_url_to_path(url: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file":
        raise ValueError(f"Gemma4 unified expected a file URL, got {url}")
    if parsed.netloc not in ("", "localhost"):
        raise ValueError("Gemma4 unified file URLs must be local")
    return Path(urllib.parse.unquote(parsed.path))


def _sample_video_frames(frames: list[Any], max_frames: int) -> list[Any]:
    if max_frames <= 0:
        raise ValueError("Gemma4 unified video num_frames must be positive")
    if len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]
    sampled = []
    for idx in range(max_frames):
        source_idx = round(idx * (len(frames) - 1) / (max_frames - 1))
        sampled.append(frames[source_idx])
    return sampled


def _trim_video_frame(
    frame: dict[str, Any],
    config: _Gemma4UnifiedConfig,
) -> dict[str, Any]:
    soft_token_count = frame["soft_token_count"]
    patch_dim = config.model_patch_size * config.model_patch_size * 3
    return {
        "pixel_values": frame["pixel_values"][: soft_token_count * patch_dim],
        "pixel_position_ids": frame["pixel_position_ids"][:soft_token_count],
        "soft_token_count": soft_token_count,
    }


def _flatten_audio_values(waveform: Any) -> list[float]:
    if isinstance(waveform, (bytes, str, Path)):
        raise ValueError(
            "Gemma4 unified audio preprocessing expects waveform samples; "
            "decode and resample audio before calling"
        )
    if hasattr(waveform, "tolist"):
        waveform = waveform.tolist()
    values: list[float] = []

    def extend(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                extend(item)
        else:
            values.append(float(value))

    extend(waveform)
    return values


def _normalize_video_timestamp_ids(
    timestamp_ids: list[list[list[int]]] | None,
    frame_counts: list[int],
) -> list[list[list[int]]]:
    if timestamp_ids is None:
        return [[[] for _ in range(frame_count)] for frame_count in frame_counts]
    if not isinstance(timestamp_ids, list):
        raise ValueError(
            "Gemma4 unified video timestamp_token_ids must be a list of videos"
        )
    if len(timestamp_ids) != len(frame_counts):
        raise ValueError(
            "Gemma4 unified video timestamp_token_ids length must match video count"
        )
    normalized = []
    for idx, (video_timestamps, frame_count) in enumerate(
        zip(timestamp_ids, frame_counts)
    ):
        if not isinstance(video_timestamps, list):
            raise ValueError(
                "Gemma4 unified video timestamp_token_ids entry "
                f"for video {idx} must be a list of frame token lists"
            )
        if len(video_timestamps) != frame_count:
            raise ValueError(
                "Gemma4 unified video timestamp_token_ids frame count mismatch: "
                f"video {idx} expected {frame_count}, found {len(video_timestamps)}"
            )
        normalized_frame_tokens: list[list[int]] = []
        for frame_idx, frame_tokens in enumerate(video_timestamps):
            if not isinstance(frame_tokens, list):
                raise ValueError(
                    "Gemma4 unified video timestamp_token_ids frame entry "
                    f"for video {idx}, frame {frame_idx} must be a list"
                )
            normalized_tokens: list[int] = []
            for token_idx, token in enumerate(frame_tokens):
                if isinstance(token, bool):
                    raise ValueError(
                        "Gemma4 unified video timestamp_token_ids must contain integer token ids; "
                        f"found bool at video {idx}, frame {frame_idx}, index {token_idx}"
                    )
                try:
                    int_token = int(token)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Gemma4 unified video timestamp_token_ids must contain integer token ids; "
                        f"found non-integer at video {idx}, frame {frame_idx}, index {token_idx}"
                    ) from exc
                if isinstance(token, float) and not token.is_integer():
                    raise ValueError(
                        "Gemma4 unified video timestamp_token_ids must contain integer token ids; "
                        f"found non-integer {token!r} at video {idx}, frame {frame_idx}, index {token_idx}"
                    )
                if int_token < 0:
                    raise ValueError(
                        "Gemma4 unified video timestamp_token_ids must contain non-negative token ids; "
                        f"found {int_token} at video {idx}, frame {frame_idx}, index {token_idx}"
                    )
                normalized_tokens.append(int_token)
            normalized_frame_tokens.append(normalized_tokens)
        normalized.append(normalized_frame_tokens)
    return normalized


def _expand_multimodal_placeholders(
    input_tokens: list[int],
    images: list[dict[str, Any]],
    audios: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    video_timestamp_ids: list[list[list[int]]],
    config: _Gemma4UnifiedConfig,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_placeholder_count(
        "image", input_tokens, config.image_token_id, len(images)
    )
    _validate_placeholder_count(
        "audio", input_tokens, config.audio_token_id, len(audios)
    )
    _validate_placeholder_count(
        "video", input_tokens, config.video_token_id, len(videos)
    )

    output: list[int] = []
    image_spans: list[dict[str, Any]] = []
    audio_spans: list[dict[str, Any]] = []
    video_spans: list[dict[str, Any]] = []
    image_index = 0
    audio_index = 0
    video_index = 0
    for placeholder_index, token in enumerate(input_tokens):
        if token == config.image_token_id:
            image = images[image_index]
            replacement_start = len(output)
            soft_token_count = image["soft_token_count"]
            output.append(config.boi_token_id)
            output.extend([config.image_token_id] * soft_token_count)
            output.append(config.eoi_token_id)
            image_spans.append(
                _span(
                    "image",
                    placeholder_index,
                    replacement_start,
                    soft_token_count,
                    soft_token_count + 2,
                )
            )
            image_index += 1
        elif token == config.audio_token_id:
            audio = audios[audio_index]
            replacement_start = len(output)
            soft_token_count = audio["frame_count"]
            output.append(config.boa_token_id)
            output.extend([config.audio_token_id] * soft_token_count)
            output.append(config.eoa_token_id)
            audio_spans.append(
                _span(
                    "audio",
                    placeholder_index,
                    replacement_start,
                    soft_token_count,
                    soft_token_count + 2,
                )
            )
            audio_index += 1
        elif token == config.video_token_id:
            video = videos[video_index]
            timestamps = video_timestamp_ids[video_index]
            replacement_start = len(output)
            soft_token_ranges = []
            soft_token_count = 0
            for frame, timestamp_tokens in zip(video["frames"], timestamps):
                output.extend(timestamp_tokens)
                output.append(config.boi_token_id)
                range_start = len(output)
                frame_soft_tokens = frame["soft_token_count"]
                output.extend([config.video_token_id] * frame_soft_tokens)
                soft_token_ranges.append(
                    {
                        "start": range_start,
                        "soft_token_count": frame_soft_tokens,
                    }
                )
                soft_token_count += frame_soft_tokens
                output.append(config.eoi_token_id)
            video_spans.append(
                {
                    "span": _span(
                        "video",
                        placeholder_index,
                        replacement_start,
                        soft_token_count,
                        len(output) - replacement_start,
                    ),
                    "soft_token_ranges": soft_token_ranges,
                }
            )
            video_index += 1
        else:
            output.append(token)

    return output, image_spans, audio_spans, video_spans


def _validate_placeholder_count(
    name: str,
    input_tokens: list[int],
    token_id: int,
    expected: int,
) -> None:
    actual = sum(1 for token in input_tokens if token == token_id)
    if actual != expected:
        raise ValueError(
            f"Gemma4 unified {name} placeholder count mismatch: "
            f"expected {expected}, found {actual}"
        )


def _span(
    modality: str,
    placeholder_index: int,
    replacement_start: int,
    soft_token_count: int,
    replacement_token_count: int,
) -> dict[str, Any]:
    return {
        "modality": modality,
        "placeholder_index": placeholder_index,
        "replacement_start": replacement_start,
        "soft_token_count": soft_token_count,
        "replacement_token_count": replacement_token_count,
    }


def _resized_dimensions(
    width: int,
    height: int,
    patch_size: int,
    pooling_kernel_size: int,
    max_soft_tokens: int,
) -> tuple[int, int]:
    max_patches = max_soft_tokens * pooling_kernel_size**2
    target_px = max_patches * patch_size**2
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    factor = math.sqrt(target_px / (height * width))
    side_mult = pooling_kernel_size * patch_size
    target_height = math.floor(factor * height / side_mult) * side_mult
    target_width = math.floor(factor * width / side_mult) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("attempting to resize to a 0 x 0 image")

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(math.floor(width / height) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(math.floor(height / width) * side_mult, max_side_length)
    return int(target_width), int(target_height)


def _rgb_pixels(
    image: Any,
    config: _Gemma4UnifiedConfig,
) -> list[tuple[float, float, float]]:
    values = []
    data = image.tobytes()
    for index in range(0, len(data), 3):
        red, green, blue = data[index], data[index + 1], data[index + 2]
        pixel = [float(red), float(green), float(blue)]
        if config.do_rescale:
            pixel = [channel * config.rescale_factor for channel in pixel]
        if config.do_normalize:
            pixel = [
                (channel - mean) / max(std, 1e-12)
                for channel, mean, std in zip(
                    pixel, config.image_mean, config.image_std
                )
            ]
        values.append((pixel[0], pixel[1], pixel[2]))
    return values


def _patchify_rgb(
    pixels: list[tuple[float, float, float]],
    width: int,
    height: int,
    config: _Gemma4UnifiedConfig,
) -> tuple[list[list[float]], list[list[int]]]:
    patch = config.model_patch_size
    patch_height = height // patch
    patch_width = width // patch
    patch_values: list[list[float]] = []
    position_ids: list[list[int]] = []
    for patch_y in range(patch_height):
        for patch_x in range(patch_width):
            values: list[float] = []
            y0 = patch_y * patch
            x0 = patch_x * patch
            for dy in range(patch):
                row_offset = (y0 + dy) * width
                for dx in range(patch):
                    values.extend(pixels[row_offset + x0 + dx])
            patch_values.append(values)
            position_ids.append([patch_x, patch_y])
    return patch_values, position_ids


def _required_int(config: dict[str, Any], key: str) -> int:
    value = _optional_int(config, key)
    if value is None:
        raise ValueError(f"missing Gemma4 unified config field {key}")
    return value


def _optional_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    return int(value)


def _triple(value: Any) -> tuple[float, float, float]:
    items = list(value)
    if len(items) != 3:
        raise ValueError("Gemma4 unified image mean/std must contain three values")
    return float(items[0]), float(items[1]), float(items[2])
