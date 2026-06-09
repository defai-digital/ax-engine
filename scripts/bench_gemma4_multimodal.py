#!/usr/bin/env python3
"""Benchmark Gemma 4 12B multimodal prefill/chat paths.

The runner produces an artifact-first matrix. It prepares deterministic image,
audio, and video fixtures, runs AX Engine through a live server, and records
llama.cpp peer rows only when the model/projector/modality contract is explicit.
Unavailable peer coverage is emitted as skipped rows, not silently omitted.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import importlib.util
import json
import math
import platform
import shlex
import statistics
import struct
import subprocess
import sys
import time
import urllib.parse
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from PIL import Image
except ImportError:  # pragma: no cover - environment dependent
    Image = None  # type: ignore[assignment]

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover - environment dependent
    Tokenizer = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "python" / "ax_engine" / "gemma4_unified.py"
SCHEMA = "ax.gemma4_multimodal_benchmark.v1"
DEFAULT_MODEL = "gemma-4-12B-it"
DEFAULT_URL = "http://127.0.0.1:18080"
DEFAULT_CASES = "all"
DEFAULT_LAYERS = "native_runtime_prefill,openai_chat_e2e"
DEFAULT_TIMEOUT_S = 300


@dataclass(frozen=True)
class MediaFixture:
    fixture_id: str
    modality: str
    payload: Any
    chat_payload: bytes | None
    mime: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    modalities: list[str]
    native_prompt: str
    chat_text: str
    fixture_ids: list[str]
    video_timestamp_seconds: list[list[float]]
    chat_enabled: bool = True


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    description: str
    modalities: list[str]
    fixture_ids: list[str]
    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    original_prompt_tokens: int
    expanded_prompt_tokens: int
    image_soft_tokens: list[int]
    audio_soft_tokens: list[int]
    video_soft_tokens: list[int]
    video_frame_counts: list[int]
    span_order: list[str]
    video_timestamp_seconds: list[list[float]]
    chat_content: list[dict[str, Any]]
    chat_enabled: bool


@dataclass(frozen=True)
class PeerDecision:
    status: str
    reason: str | None
    detail: str | None
    capability: dict[str, Any]


def require_pillow() -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for Gemma 4 multimodal benchmark fixtures")


def require_tokenizers() -> None:
    if Tokenizer is None:
        raise RuntimeError("tokenizers is required to prepare Gemma 4 multimodal requests")


def load_gemma4_helper():
    spec = importlib.util.spec_from_file_location("ax_engine_gemma4_unified_bench", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load Gemma4 helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    try:
        return [int(token) for token in tokenizer.encode(text, add_special_tokens=False).ids]
    except TypeError:
        return [int(token) for token in tokenizer.encode(text).ids]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def png_solid(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    require_pillow()
    buffer = BytesIO()
    Image.new("RGB", (width, height), rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def png_gradient(width: int, height: int) -> bytes:
    require_pillow()
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((x * 7) % 256, (y * 5) % 256, 128)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def wav_tone(seconds: float, sample_rate: int = 16000, freq: float = 220.0) -> bytes:
    n = int(seconds * sample_rate)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        samples = bytearray()
        for i in range(n):
            value = int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            samples += struct.pack("<h", value)
        writer.writeframes(bytes(samples))
    return buffer.getvalue()


def decoded_video_frames(
    colors: list[tuple[int, int, int]],
    *,
    width: int = 32,
    height: int = 32,
) -> list[Any]:
    require_pillow()
    return [Image.new("RGB", (width, height), color) for color in colors]


def gif_frames(
    width: int,
    height: int,
    colors: list[tuple[int, int, int]],
    *,
    duration_ms: int,
) -> bytes:
    require_pillow()
    frames = [Image.new("RGB", (width, height), color) for color in colors]
    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return buffer.getvalue()


def fixture_record(fixture: MediaFixture) -> dict[str, Any]:
    if fixture.modality == "video":
        payload_hash = sha256_bytes(fixture.chat_payload or b"")
        native_hashes = []
        for frame in fixture.payload:
            buffer = BytesIO()
            frame.save(buffer, format="PNG")
            native_hashes.append(sha256_bytes(buffer.getvalue()))
        raw = {**fixture.raw, "native_frame_sha256": native_hashes}
    else:
        payload_hash = sha256_bytes(fixture.payload)
        raw = fixture.raw
    return {
        "id": fixture.fixture_id,
        "modality": fixture.modality,
        "source": "generated",
        "sha256": payload_hash,
        "mime": fixture.mime,
        "raw": raw,
    }


def image_fixture(
    fixture_id: str,
    data: bytes,
    *,
    width: int,
    height: int,
    generator: str,
    color: tuple[int, int, int] | None = None,
) -> MediaFixture:
    raw: dict[str, Any] = {"width": width, "height": height, "generator": generator}
    if color is not None:
        raw["color"] = list(color)
    return MediaFixture(fixture_id, "image", data, data, "image/png", raw)


def audio_fixture(fixture_id: str, seconds: float, *, cap_case: bool = False) -> MediaFixture:
    sample_rate = 16000
    data = wav_tone(seconds, sample_rate=sample_rate)
    return MediaFixture(
        fixture_id,
        "audio",
        {"input_audio": {"data": b64(data), "format": "wav"}},
        data,
        "audio/wav",
        {
            "duration_s": seconds,
            "sample_rate": sample_rate,
            "sample_count": int(seconds * sample_rate),
            "sample_format": "int16",
            "generator": "sine_wave",
            "frequency_hz": 220.0,
            "cap_case": cap_case,
        },
    )


def video_fixture(
    fixture_id: str,
    colors: list[tuple[int, int, int]],
    *,
    duration_ms: int,
    cap_case: bool = False,
) -> MediaFixture:
    width = 32
    height = 32
    frames = decoded_video_frames(colors, width=width, height=height)
    gif = gif_frames(width, height, colors, duration_ms=duration_ms)
    timestamps = [index * duration_ms / 1000.0 for index in range(len(colors))]
    return MediaFixture(
        fixture_id,
        "video",
        frames,
        gif,
        "image/gif",
        {
            "width": width,
            "height": height,
            "source_frame_count": len(colors),
            "duration_ms_per_frame": duration_ms,
            "timestamp_seconds": timestamps,
            "generator": "solid_color_frames",
            "colors": [list(color) for color in colors],
            "cap_case": cap_case,
        },
    )


def build_fixture_registry() -> dict[str, MediaFixture]:
    red = (220, 20, 20)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 128, 128),
        (255, 128, 0),
    ]
    cap_palette = [palette[index % len(palette)] for index in range(40)]
    fixtures = [
        image_fixture(
            "image_red_64",
            png_solid(64, 64, red),
            width=64,
            height=64,
            generator="solid_rgb",
            color=red,
        ),
        image_fixture(
            "image_gradient_96",
            png_gradient(96, 96),
            width=96,
            height=96,
            generator="xy_gradient",
        ),
        image_fixture(
            "image_portrait_64x128",
            png_gradient(64, 128),
            width=64,
            height=128,
            generator="xy_gradient",
        ),
        image_fixture(
            "image_landscape_128x64",
            png_gradient(128, 64),
            width=128,
            height=64,
            generator="xy_gradient",
        ),
        image_fixture(
            "image_large_512",
            png_gradient(512, 512),
            width=512,
            height=512,
            generator="xy_gradient",
        ),
        audio_fixture("audio_tone_0_5s", 0.5),
        audio_fixture("audio_tone_2s", 2.0),
        audio_fixture("audio_tone_10s", 10.0),
        audio_fixture("audio_tone_40s_cap", 40.0, cap_case=True),
        video_fixture("video_1frame_red", [red], duration_ms=2000),
        video_fixture("video_2frame_red_green", [red, green], duration_ms=2000),
        video_fixture("video_8frame_palette", palette, duration_ms=1000),
        video_fixture("video_40frame_cap", cap_palette, duration_ms=1000, cap_case=True),
        video_fixture("video_2frame_blue_green", [blue, green], duration_ms=2000),
    ]
    return {fixture.fixture_id: fixture for fixture in fixtures}


def build_case_registry() -> dict[str, BenchmarkCase]:
    fixtures = build_fixture_registry()

    def prompt(question: str, placeholders: str) -> str:
        return (
            "<bos><|turn>user\n"
            f"{question}{placeholders}<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>"
        )

    def video_timestamps(fixture_id: str) -> list[float]:
        raw = fixtures[fixture_id].raw
        return [float(value) for value in raw["timestamp_seconds"]]

    cases = [
        BenchmarkCase(
            "image_single_256soft",
            "one deterministic PNG image plus a short question",
            ["image"],
            prompt("What color is this image? Answer in one word.", "<|image|>"),
            "What color is this image? Answer in one word.",
            ["image_red_64"],
            [],
        ),
        BenchmarkCase(
            "image_multi_2x256soft",
            "two distinct PNG images to catch placeholder ordering bugs",
            ["image"],
            prompt("Compare these two images briefly.", "<|image|><|image|>"),
            "Compare these two images briefly.",
            ["image_red_64", "image_gradient_96"],
            [],
        ),
        BenchmarkCase(
            "image_aspect_portrait",
            "portrait image resize and pad behavior",
            ["image"],
            prompt("Describe this portrait-oriented image.", "<|image|>"),
            "Describe this portrait-oriented image.",
            ["image_portrait_64x128"],
            [],
        ),
        BenchmarkCase(
            "image_aspect_landscape",
            "landscape image resize and pad behavior",
            ["image"],
            prompt("Describe this landscape-oriented image.", "<|image|>"),
            "Describe this landscape-oriented image.",
            ["image_landscape_128x64"],
            [],
        ),
        BenchmarkCase(
            "image_max_soft_tokens",
            "large generated image exercising the configured image soft-token ceiling",
            ["image"],
            prompt("Describe this large generated image.", "<|image|>"),
            "Describe this large generated image.",
            ["image_large_512"],
            [],
        ),
        BenchmarkCase(
            "audio_0_5s",
            "0.5 second 16 kHz int16 sine wave",
            ["audio"],
            prompt("Describe this audio.", "<|audio|>"),
            "Describe this audio.",
            ["audio_tone_0_5s"],
            [],
        ),
        BenchmarkCase(
            "audio_2s",
            "2 second 16 kHz int16 sine wave",
            ["audio"],
            prompt("Describe this short audio.", "<|audio|>"),
            "Describe this short audio.",
            ["audio_tone_2s"],
            [],
        ),
        BenchmarkCase(
            "audio_10s",
            "10 second 16 kHz int16 sine wave",
            ["audio"],
            prompt("Describe this longer audio.", "<|audio|>"),
            "Describe this longer audio.",
            ["audio_tone_10s"],
            [],
        ),
        BenchmarkCase(
            "audio_cap",
            "longer generated audio cap behavior",
            ["audio"],
            prompt("Describe this capped audio input.", "<|audio|>"),
            "Describe this capped audio input.",
            ["audio_tone_40s_cap"],
            [],
        ),
        BenchmarkCase(
            "video_1frame",
            "single-frame video path",
            ["video"],
            prompt("Describe this single-frame clip.", "<|video|>"),
            "Describe this single-frame clip.",
            ["video_1frame_red"],
            [video_timestamps("video_1frame_red")],
        ),
        BenchmarkCase(
            "video_2frame_distinct",
            "two distinct frames with timestamp prefixes",
            ["video"],
            prompt("How many distinct frames are in this clip?", "<|video|>"),
            "How many distinct frames are in this clip?",
            ["video_2frame_red_green"],
            [video_timestamps("video_2frame_red_green")],
        ),
        BenchmarkCase(
            "video_8frame",
            "eight-frame generated color sequence",
            ["video"],
            prompt("Summarize the color changes in this clip.", "<|video|>"),
            "Summarize the color changes in this clip.",
            ["video_8frame_palette"],
            [video_timestamps("video_8frame_palette")],
        ),
        BenchmarkCase(
            "video_32frame_cap",
            "forty source frames exercising the 32-frame cap",
            ["video"],
            prompt("Summarize this long generated clip.", "<|video|>"),
            "Summarize this long generated clip.",
            ["video_40frame_cap"],
            [video_timestamps("video_40frame_cap")],
        ),
        BenchmarkCase(
            "image_audio",
            "combined image and audio request",
            ["image", "audio"],
            prompt("Summarize the image and audio together.", "<|image|><|audio|>"),
            "Summarize the image and audio together.",
            ["image_gradient_96", "audio_tone_0_5s"],
            [],
        ),
        BenchmarkCase(
            "image_video",
            "combined image and video request",
            ["image", "video"],
            prompt("Summarize the image and video together.", "<|image|><|video|>"),
            "Summarize the image and video together.",
            ["image_gradient_96", "video_2frame_blue_green"],
            [video_timestamps("video_2frame_blue_green")],
        ),
        BenchmarkCase(
            "audio_video",
            "combined audio and video request",
            ["audio", "video"],
            prompt("Summarize the audio and video together.", "<|audio|><|video|>"),
            "Summarize the audio and video together.",
            ["audio_tone_0_5s", "video_2frame_blue_green"],
            [video_timestamps("video_2frame_blue_green")],
        ),
        BenchmarkCase(
            "image_audio_video",
            "combined image, audio, and two-frame video request",
            ["image", "audio", "video"],
            prompt(
                "Summarize the image, audio, and video together.",
                "<|image|><|audio|><|video|>",
            ),
            "Summarize the image, audio, and video together.",
            ["image_gradient_96", "audio_tone_0_5s", "video_2frame_blue_green"],
            [video_timestamps("video_2frame_blue_green")],
        ),
    ]
    return {case.case_id: case for case in cases}


def select_cases(selection: str) -> list[BenchmarkCase]:
    cases = build_case_registry()
    names = [name.strip() for name in selection.split(",") if name.strip()]
    if not names or names == ["all"]:
        names = list(cases)
    unknown = [name for name in names if name not in cases]
    if unknown:
        raise ValueError(f"unknown case(s): {', '.join(unknown)}")
    return [cases[name] for name in names]


def fixtures_for_case(
    fixtures: dict[str, MediaFixture],
    case: BenchmarkCase,
) -> tuple[list[bytes], list[dict[str, Any]], list[list[Any]]]:
    images: list[bytes] = []
    audios: list[dict[str, Any]] = []
    videos: list[list[Any]] = []
    for fixture_id in case.fixture_ids:
        fixture = fixtures[fixture_id]
        if fixture.modality == "image":
            images.append(fixture.payload)
        elif fixture.modality == "audio":
            audios.append(fixture.payload)
        elif fixture.modality == "video":
            videos.append(fixture.payload)
    return images, audios, videos


def chat_content_for_case(
    fixtures: dict[str, MediaFixture],
    case: BenchmarkCase,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": case.chat_text}]
    for fixture_id in case.fixture_ids:
        fixture = fixtures[fixture_id]
        if fixture.modality == "image":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{fixture.mime};base64,{b64(fixture.chat_payload or b'')}"
                    },
                }
            )
        elif fixture.modality == "audio":
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": b64(fixture.chat_payload or b""), "format": "wav"},
                }
            )
        elif fixture.modality == "video":
            content.append(
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:{fixture.mime};base64,{b64(fixture.chat_payload or b'')}"
                    },
                }
            )
    return content


def video_timestamp_token_ids(tokenizer: Any, seconds_per_video: list[list[float]]) -> list[list[list[int]]]:
    out: list[list[list[int]]] = []
    for frame_seconds in seconds_per_video:
        video_ids = []
        for index, seconds in enumerate(frame_seconds):
            minutes = int(max(seconds, 0.0) // 60)
            secs = int(max(seconds, 0.0) % 60)
            stamp = f"{minutes:02}:{secs:02}"
            prefix = f"{stamp} " if index == 0 else f" {stamp} "
            video_ids.append(encode_without_special_tokens(tokenizer, prefix))
        out.append(video_ids)
    return out


def prepare_case(
    model_dir: Path,
    fixtures: dict[str, MediaFixture],
    case: BenchmarkCase,
) -> PreparedCase:
    require_tokenizers()
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt_tokens = encode_without_special_tokens(tokenizer, case.native_prompt)
    images, audios, videos = fixtures_for_case(fixtures, case)
    helper = load_gemma4_helper()
    timestamps = (
        video_timestamp_token_ids(tokenizer, case.video_timestamp_seconds)
        if videos
        else None
    )
    request = helper.prepare_gemma4_unified_multimodal_request(
        model_dir,
        prompt_tokens,
        images=images,
        audios=audios,
        videos=videos,
        video_timestamp_token_ids=timestamps,
    )
    return PreparedCase(
        case_id=case.case_id,
        description=case.description,
        modalities=case.modalities,
        fixture_ids=case.fixture_ids,
        input_tokens=[int(token) for token in request.input_tokens],
        multimodal_inputs=request.multimodal_inputs,
        original_prompt_tokens=len(prompt_tokens),
        expanded_prompt_tokens=len(request.input_tokens),
        image_soft_tokens=[int(value) for value in request.image_soft_token_counts],
        audio_soft_tokens=[int(value) for value in request.audio_soft_token_counts],
        video_soft_tokens=[int(value) for value in request.video_soft_token_counts],
        video_frame_counts=[int(value) for value in request.video_frame_counts],
        span_order=case.modalities,
        video_timestamp_seconds=case.video_timestamp_seconds,
        chat_content=chat_content_for_case(fixtures, case),
        chat_enabled=case.chat_enabled,
    )


def iter_sse_json_events(lines: Iterable[str]) -> Iterator[tuple[str, Any]]:
    event_name = ""
    data_parts: list[str] = []

    def flush() -> tuple[str, Any] | None:
        nonlocal event_name, data_parts
        name = event_name
        data = "\n".join(data_parts)
        event_name = ""
        data_parts = []
        if not data:
            return None
        return name, json.loads(data)

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if line == "":
            frame = flush()
            if frame is not None:
                yield frame
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            value = line[len("data:") :]
            data_parts.append(value[1:] if value.startswith(" ") else value)

    frame = flush()
    if frame is not None:
        yield frame


def is_prefill_step(step: dict[str, Any], *, seen_prefill: bool) -> bool:
    route = step.get("route") or {}
    labels = [
        str(route.get("execution_plan") or "").lower(),
        str(route.get("attention_route") or "").lower(),
    ]
    if any("prefill" in label for label in labels):
        return True
    if any("decode" in label for label in labels):
        return False
    return not seen_prefill or int(step.get("scheduled_tokens") or 0) > 1


def http_connection(url: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[http.client.HTTPConnection, str]:
    parsed = urllib.parse.urlsplit(url.rstrip("/"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(host, port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    base_path = parsed.path.rstrip("/")
    return conn, base_path


def run_native_one(
    url: str,
    model: str,
    request: PreparedCase,
    max_output_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    conn, base_path = http_connection(url, timeout=timeout_s)
    payload = json.dumps(
        {
            "model": model,
            "input_tokens": request.input_tokens,
            "multimodal_inputs": request.multimodal_inputs,
            "max_output_tokens": max_output_tokens,
            "sampling": {"temperature": 0.0, "ignore_eos": True},
        }
    ).encode("utf-8")
    started = time.perf_counter()
    first_output_wall_s: float | None = None
    try:
        conn.request(
            "POST",
            f"{base_path}/v1/generate/stream",
            body=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {response.read(500).decode(errors='replace')}"
            )

        prefill_us = 0
        seen_prefill = False
        output_tokens = 0
        execution_plan = None
        route_metadata: dict[str, Any] = {}
        decoded_lines = (line.decode("utf-8", errors="replace") for line in response)
        for event_name, obj in iter_sse_json_events(decoded_lines):
            if event_name == "step":
                step = obj.get("step") or {}
                current_output_len = obj.get("request", {}).get("output_len")
                if current_output_len:
                    output_tokens = int(current_output_len)
                    if first_output_wall_s is None:
                        first_output_wall_s = time.perf_counter() - started
                if is_prefill_step(step, seen_prefill=seen_prefill):
                    prefill_us += int(step.get("runner_time_us") or 0)
                    seen_prefill = True
                    route = step.get("route") or {}
                    route_metadata = route or route_metadata
                    execution_plan = route.get("execution_plan") or execution_plan
            elif event_name == "response":
                response_obj = obj.get("response") or {}
                tokens = response_obj.get("output_tokens")
                if isinstance(tokens, list):
                    output_tokens = len(tokens)
                    if tokens and first_output_wall_s is None:
                        first_output_wall_s = time.perf_counter() - started
                route = response_obj.get("route") or {}
                route_metadata = route_metadata or route
                execution_plan = execution_plan or route.get("execution_plan")
    finally:
        conn.close()

    total_wall_ms = (time.perf_counter() - started) * 1000.0
    runner_prefill_ms = prefill_us / 1000.0
    return {
        "runner_prefill_ttft_ms": runner_prefill_ms,
        "client_wall_ttft_ms": (
            first_output_wall_s * 1000.0 if first_output_wall_s is not None else None
        ),
        "client_wall_total_ms": total_wall_ms,
        "prefill_tok_s": (
            request.expanded_prompt_tokens / (runner_prefill_ms / 1000.0)
            if runner_prefill_ms > 0
            else None
        ),
        "output_tokens": output_tokens,
        "execution_plan": execution_plan,
        "route": route_metadata,
    }


def run_chat_one(
    url: str,
    model: str,
    case: PreparedCase,
    max_output_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    conn, base_path = http_connection(url, timeout=timeout_s)
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": case.chat_content}],
            "max_tokens": max_output_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    started = time.perf_counter()
    try:
        conn.request(
            "POST",
            f"{base_path}/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = response.read()
        wall_ms = (time.perf_counter() - started) * 1000.0
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:500].decode(errors='replace')}")
        obj = json.loads(body.decode("utf-8"))
    finally:
        conn.close()

    content = ""
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
    usage = obj.get("usage") or {}
    return {
        "non_streaming_total_ms": wall_ms,
        "client_wall_ms": wall_ms,
        "client_wall_ttft_ms": None,
        "output_tokens": usage.get("completion_tokens"),
        "prompt_tokens_reported": usage.get("prompt_tokens"),
        "content_chars": len(content),
        "payload_bytes": len(payload),
    }


def summarize(values: list[float | int | None]) -> dict[str, float | None]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": sum(present) / len(present),
        "median": statistics.median(present),
        "min": min(present),
        "max": max(present),
    }


def summarize_runs(runs: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float | None]]:
    return {key: summarize([run.get(key) for run in runs]) for key in keys}


def prompt_block(case: PreparedCase) -> dict[str, Any]:
    soft_tokens = {
        "image": sum(case.image_soft_tokens),
        "audio": sum(case.audio_soft_tokens),
        "video": sum(case.video_soft_tokens),
    }
    return {
        "original_tokens": case.original_prompt_tokens,
        "expanded_tokens": case.expanded_prompt_tokens,
        "soft_tokens": soft_tokens,
        "span_order": case.span_order,
        "fixture_ids": case.fixture_ids,
        "image_soft_tokens": case.image_soft_tokens,
        "audio_soft_tokens": case.audio_soft_tokens,
        "video_soft_tokens": case.video_soft_tokens,
        "video_frame_counts": case.video_frame_counts,
        "video_timestamp_seconds": case.video_timestamp_seconds,
    }


def row_common(
    *,
    engine: str,
    backend: str,
    layer: str,
    endpoint: str | None,
    case: PreparedCase,
) -> dict[str, Any]:
    return {
        "row_id": f"{engine}.{layer}.{case.case_id}",
        "engine": engine,
        "backend": backend,
        "layer": layer,
        "endpoint": endpoint,
        "case_id": case.case_id,
        "description": case.description,
        "modalities": case.modalities,
        "modality_set": case.modalities,
        "fixture_ids": case.fixture_ids,
        "prompt": prompt_block(case),
    }


def measured_row(
    *,
    engine: str,
    backend: str,
    layer: str,
    endpoint: str,
    case: PreparedCase,
    runs: list[dict[str, Any]],
    metric_keys: list[str],
    max_output_tokens: int,
    warmup: int,
    repetitions: int,
    capability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = row_common(engine=engine, backend=backend, layer=layer, endpoint=endpoint, case=case)
    row.update(
        {
            "status": "measured",
            "sampling": {"temperature": 0.0, "ignore_eos": layer == "native_runtime_prefill"},
            "max_output_tokens": max_output_tokens,
            "warmup": warmup,
            "repetitions": repetitions,
            "runs": runs,
            "summary": summarize_runs(runs, metric_keys),
        }
    )
    if capability is not None:
        row["capability"] = capability
    return row


def skipped_row(
    *,
    engine: str,
    backend: str,
    layer: str,
    endpoint: str | None,
    case: PreparedCase,
    reason: str,
    detail: str,
    capability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = row_common(engine=engine, backend=backend, layer=layer, endpoint=endpoint, case=case)
    row.update(
        {
            "status": "skipped",
            "skip_reason": reason,
            "skip_detail": detail,
            "runs": [],
            "summary": {},
        }
    )
    if capability is not None:
        row["capability"] = capability
    return row


def run_repetitions(
    *,
    warmup: int,
    repetitions: int,
    cooldown_s: float,
    run,
) -> list[dict[str, Any]]:
    for _ in range(warmup):
        run()
        if cooldown_s > 0:
            time.sleep(cooldown_s)
    runs = []
    for _ in range(repetitions):
        runs.append(run())
        if cooldown_s > 0:
            time.sleep(cooldown_s)
    return runs


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_block() -> dict[str, Any]:
    status = run_git(["status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "build_profile": "unknown",
        "git_tracked_dirty": bool(status),
        "git_tracked_status": status.splitlines() if status else [],
        "git_tracked_dirty_accepted": False,
    }


def host_block() -> dict[str, Any]:
    return {
        "platform": sys.platform,
        "platform_detail": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "chip": platform.processor() or "unknown",
        "memory_gb": None,
        "os_version": platform.platform(),
    }


def model_block(model: str, model_dir: Path) -> dict[str, Any]:
    return {
        "id": model,
        "model_dir": str(model_dir),
        "model_type": "gemma4_unified",
        "model_manifest_sha256": sha256_file(model_dir / "model-manifest.json"),
        "config_sha256": sha256_file(model_dir / "config.json"),
        "processor_config_sha256": sha256_file(model_dir / "processor_config.json"),
        "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json"),
    }


def server_block(args: argparse.Namespace) -> dict[str, Any]:
    command = shlex.split(args.server_command) if args.server_command else None
    return {
        "url": args.url,
        "binary": args.server_binary,
        "command": command,
        "command_source": "cli" if command else "user_supplied_running_server",
        "endpoint_layers": ["/v1/generate/stream", "/v1/chat/completions"],
        "request_timeout_s": args.timeout,
    }


def benchmark_block(args: argparse.Namespace, cases: list[PreparedCase], layers: set[str]) -> dict[str, Any]:
    return {
        "name": "gemma4_12b_multimodal",
        "model": args.model,
        "model_dir": str(args.model_dir),
        "layers": sorted(layers),
        "cases": [case.case_id for case in cases],
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "cooldown_s": args.cooldown,
        "max_output_tokens": args.max_output_tokens,
        "timeout_s": args.timeout,
        "sampler": {"temperature": 0.0},
    }


def command_json(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "url": args.url,
        "model": args.model,
        "model_dir": str(args.model_dir),
        "llama_url": args.llama_url,
        "llama_gguf": str(args.llama_gguf) if args.llama_gguf else None,
        "llama_mmproj": str(args.llama_mmproj) if args.llama_mmproj else None,
    }


def peer_capability(args: argparse.Namespace, case: PreparedCase) -> PeerDecision:
    gguf_exists = bool(args.llama_gguf and args.llama_gguf.is_file())
    mmproj_exists = bool(args.llama_mmproj and args.llama_mmproj.is_file())
    supports_video = False
    supports_image = True
    supports_audio = True
    capability = {
        "url": args.llama_url,
        "binary": args.llama_binary,
        "text_gguf": str(args.llama_gguf) if args.llama_gguf else None,
        "text_gguf_sha256": sha256_file(args.llama_gguf),
        "mmproj": str(args.llama_mmproj) if args.llama_mmproj else None,
        "mmproj_sha256": sha256_file(args.llama_mmproj),
        "supports_image": supports_image,
        "supports_audio": supports_audio,
        "supports_video": supports_video,
        "prompt_contract": "openai_chat_completions",
        "proof": False,
    }
    if "video" in case.modalities:
        return PeerDecision(
            "skipped",
            "llama_cpp_video_not_supported",
            "local llama.cpp multimodal support covers image/audio but not video",
            capability,
        )
    if not args.llama_url:
        return PeerDecision(
            "skipped",
            "no_llama_cpp_server_url",
            "pass --llama-url for an OpenAI-compatible llama.cpp multimodal server",
            capability,
        )
    if not gguf_exists:
        return PeerDecision(
            "skipped",
            "missing_llama_cpp_gguf_for_gemma4_12b",
            "pass --llama-gguf with the Gemma 4 12B text GGUF used by the peer server",
            capability,
        )
    if not mmproj_exists:
        return PeerDecision(
            "skipped",
            "missing_llama_cpp_mmproj_for_gemma4_12b",
            "pass --llama-mmproj with the matching Gemma 4 12B multimodal projector",
            capability,
        )
    capability["proof"] = True
    return PeerDecision("measured", None, None, capability)


def artifact_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row.get("status") == "measured"]
    skipped = [row for row in rows if row.get("status") == "skipped"]
    measured_modalities = sorted(
        {
            modality
            for row in measured
            for modality in row.get("modalities", [])
            if isinstance(modality, str)
        }
    )
    return {
        "measured_rows": len(measured),
        "skipped_rows": len(skipped),
        "measured_modalities": measured_modalities,
        "skip_reasons": sorted(
            {row.get("skip_reason") for row in skipped if isinstance(row.get("skip_reason"), str)}
        ),
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    fixtures = build_fixture_registry()
    selected_cases = select_cases(args.cases)
    prepared_cases = [prepare_case(args.model_dir, fixtures, case) for case in selected_cases]
    layers = {layer.strip() for layer in args.layers.split(",") if layer.strip()}
    unknown_layers = layers - {"native_runtime_prefill", "openai_chat_e2e"}
    if unknown_layers:
        raise ValueError(f"unknown layer(s): {', '.join(sorted(unknown_layers))}")

    rows: list[dict[str, Any]] = []
    for case in prepared_cases:
        if "native_runtime_prefill" in layers:
            runs = run_repetitions(
                warmup=args.warmup,
                repetitions=args.repetitions,
                cooldown_s=args.cooldown,
                run=lambda case=case: run_native_one(
                    args.url, args.model, case, args.max_output_tokens, args.timeout
                ),
            )
            rows.append(
                measured_row(
                    engine="ax_engine_mlx",
                    backend="mlx",
                    layer="native_runtime_prefill",
                    endpoint="/v1/generate/stream",
                    case=case,
                    runs=runs,
                    metric_keys=[
                        "runner_prefill_ttft_ms",
                        "client_wall_ttft_ms",
                        "client_wall_total_ms",
                        "prefill_tok_s",
                    ],
                    max_output_tokens=args.max_output_tokens,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
            )

        if "openai_chat_e2e" in layers:
            runs = run_repetitions(
                warmup=args.warmup,
                repetitions=args.repetitions,
                cooldown_s=args.cooldown,
                run=lambda case=case: run_chat_one(
                    args.url, args.model, case, args.max_output_tokens, args.timeout
                ),
            )
            rows.append(
                measured_row(
                    engine="ax_engine_mlx",
                    backend="mlx",
                    layer="openai_chat_e2e",
                    endpoint="/v1/chat/completions",
                    case=case,
                    runs=runs,
                    metric_keys=[
                        "non_streaming_total_ms",
                        "client_wall_ms",
                        "content_chars",
                        "payload_bytes",
                        "output_tokens",
                    ],
                    max_output_tokens=args.max_output_tokens,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
            )

        if not args.no_llama_cpp:
            decision = peer_capability(args, case)
            if decision.status == "measured":
                runs = run_repetitions(
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    cooldown_s=args.cooldown,
                    run=lambda case=case: run_chat_one(
                        args.llama_url, args.model, case, args.max_output_tokens, args.timeout
                    ),
                )
                rows.append(
                    measured_row(
                        engine="llama_cpp_metal",
                        backend="metal",
                        layer="peer_comparison",
                        endpoint="/v1/chat/completions",
                        case=case,
                        runs=runs,
                        metric_keys=[
                            "non_streaming_total_ms",
                            "client_wall_ms",
                            "content_chars",
                            "payload_bytes",
                            "output_tokens",
                        ],
                        max_output_tokens=args.max_output_tokens,
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        capability=decision.capability,
                    )
                )
            else:
                rows.append(
                    skipped_row(
                        engine="llama_cpp_metal",
                        backend="metal",
                        layer="peer_comparison",
                        endpoint="/v1/chat/completions" if args.llama_url else None,
                        case=case,
                        reason=decision.reason or "unsupported_modality",
                        detail=decision.detail or "peer comparison unavailable",
                        capability=decision.capability,
                    )
                )

    fixture_ids = {fixture_id for case in prepared_cases for fixture_id in case.fixture_ids}
    fixtures_used = [fixture_record(fixtures[fixture_id]) for fixture_id in sorted(fixture_ids)]
    artifact = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": host_block(),
        "build": build_block(),
        "server": server_block(args),
        "model": model_block(args.model, args.model_dir),
        "benchmark": benchmark_block(args, prepared_cases, layers),
        "fixtures": fixtures_used,
        "rows": rows,
        "summary": artifact_summary(rows),
        "command": command_json(args),
    }
    # Compatibility with the first implementation and existing renderer tests.
    artifact["provenance"] = {
        "created_at": artifact["created_at"],
        "host": artifact["host"],
        "git": {
            "commit": artifact["build"]["commit"],
            "tracked_dirty": artifact["build"]["git_tracked_dirty"],
            "tracked_dirty_paths": artifact["build"]["git_tracked_status"],
        },
        "model_fingerprints": {
            "config.json": artifact["model"]["config_sha256"],
            "processor_config.json": artifact["model"]["processor_config_sha256"],
            "tokenizer.json": artifact["model"]["tokenizer_sha256"],
            "model-manifest.json": artifact["model"]["model_manifest_sha256"],
        },
    }
    return artifact


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    return REPO_ROOT / "benchmarks" / "results" / "gemma4-multimodal" / f"{stamp}-matrix.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / ".internal/models/gemma-4-12B-it-4bit",
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-binary")
    parser.add_argument("--server-command")
    parser.add_argument("--llama-url")
    parser.add_argument("--llama-binary")
    parser.add_argument("--llama-gguf", type=Path)
    parser.add_argument("--llama-mmproj", type=Path)
    parser.add_argument("--no-llama-cpp", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    artifact = build_artifact(args)
    text = json.dumps(artifact, indent=2) + "\n"
    output = args.output or default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
