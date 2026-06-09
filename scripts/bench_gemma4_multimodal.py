#!/usr/bin/env python3
"""Benchmark Gemma 4 12B multimodal prefill/chat paths.

The runner is intentionally artifact-first. It prepares deterministic image,
audio, and video fixtures with the same Gemma 4 unified preprocessing helper
used by the SDK path, measures AX Engine through a running server, and records
llama.cpp peer rows when an OpenAI-compatible llama.cpp server URL is supplied.
Rows that cannot be measured are written as explicit skips instead of silently
dropping coverage.
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
DEFAULT_CASES = "image_single_256soft,audio_0_5s,video_2frame_distinct,image_audio_video"
DEFAULT_LAYERS = "native_runtime_prefill,openai_chat_e2e"


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    modalities: list[str]
    native_prompt: str
    chat_text: str
    images: list[bytes]
    audios: list[dict[str, Any]]
    videos: list[list[Any]]
    video_timestamp_seconds: list[list[float]]
    chat_content: list[dict[str, Any]]


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    description: str
    modalities: list[str]
    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    original_prompt_tokens: int
    expanded_prompt_tokens: int
    image_soft_tokens: list[int]
    audio_soft_tokens: list[int]
    video_soft_tokens: list[int]
    video_frame_counts: list[int]
    chat_content: list[dict[str, Any]]


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


def gif_frames(width: int, height: int, colors: list[tuple[int, int, int]]) -> bytes:
    require_pillow()
    frames = [Image.new("RGB", (width, height), color) for color in colors]
    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    return buffer.getvalue()


def decoded_video_frames(colors: list[tuple[int, int, int]]) -> list[Any]:
    require_pillow()
    return [Image.new("RGB", (32, 32), color) for color in colors]


def wav_tone(seconds: float = 0.5, sample_rate: int = 16000, freq: float = 220.0) -> bytes:
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


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def image_part(data: bytes) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(data)}"}}


def audio_part(data: bytes) -> dict[str, Any]:
    return {"type": "input_audio", "input_audio": {"data": b64(data), "format": "wav"}}


def video_part(data: bytes) -> dict[str, Any]:
    return {"type": "video_url", "video_url": {"url": f"data:image/gif;base64,{b64(data)}"}}


def build_cases() -> dict[str, BenchmarkCase]:
    red_png = png_solid(64, 64, (220, 20, 20))
    gradient_png = png_gradient(96, 96)
    tone_wav = wav_tone()
    video_colors = [(255, 0, 0), (0, 255, 0)]
    video_gif = gif_frames(32, 32, video_colors)
    video_frames = decoded_video_frames(video_colors)

    return {
        "image_single_256soft": BenchmarkCase(
            case_id="image_single_256soft",
            description="one deterministic PNG image plus a short question",
            modalities=["image"],
            native_prompt=(
                "<bos><|turn>user\n"
                "What color is this image? Answer in one word.<|image|><turn|>\n"
                "<|turn>model\n<|channel>thought\n<channel|>"
            ),
            chat_text="What color is this image? Answer in one word.",
            images=[red_png],
            audios=[],
            videos=[],
            video_timestamp_seconds=[],
            chat_content=[{"type": "text", "text": "What color is this image? Answer in one word."}, image_part(red_png)],
        ),
        "image_multi_2x256soft": BenchmarkCase(
            case_id="image_multi_2x256soft",
            description="two distinct PNG images to catch placeholder ordering bugs",
            modalities=["image"],
            native_prompt=(
                "<bos><|turn>user\n"
                "Compare these two images briefly.<|image|><|image|><turn|>\n"
                "<|turn>model\n<|channel>thought\n<channel|>"
            ),
            chat_text="Compare these two images briefly.",
            images=[red_png, gradient_png],
            audios=[],
            videos=[],
            video_timestamp_seconds=[],
            chat_content=[
                {"type": "text", "text": "Compare these two images briefly."},
                image_part(red_png),
                image_part(gradient_png),
            ],
        ),
        "audio_0_5s": BenchmarkCase(
            case_id="audio_0_5s",
            description="0.5 second 16 kHz int16 sine wave",
            modalities=["audio"],
            native_prompt=(
                "<bos><|turn>user\n"
                "Describe this audio.<|audio|><turn|>\n"
                "<|turn>model\n<|channel>thought\n<channel|>"
            ),
            chat_text="Describe this audio.",
            images=[],
            audios=[{"input_audio": {"data": b64(tone_wav), "format": "wav"}}],
            videos=[],
            video_timestamp_seconds=[],
            chat_content=[{"type": "text", "text": "Describe this audio."}, audio_part(tone_wav)],
        ),
        "video_2frame_distinct": BenchmarkCase(
            case_id="video_2frame_distinct",
            description="two distinct decoded frames with timestamp prefixes",
            modalities=["video"],
            native_prompt=(
                "<bos><|turn>user\n"
                "How many distinct frames are in this clip?<|video|><turn|>\n"
                "<|turn>model\n<|channel>thought\n<channel|>"
            ),
            chat_text="How many distinct frames are in this clip?",
            images=[],
            audios=[],
            videos=[video_frames],
            video_timestamp_seconds=[[0.0, 2.0]],
            chat_content=[
                {"type": "text", "text": "How many distinct frames are in this clip?"},
                video_part(video_gif),
            ],
        ),
        "image_audio_video": BenchmarkCase(
            case_id="image_audio_video",
            description="combined image, audio, and two-frame video request",
            modalities=["image", "audio", "video"],
            native_prompt=(
                "<bos><|turn>user\n"
                "Summarize the image, audio, and video together."
                "<|image|><|audio|><|video|><turn|>\n"
                "<|turn>model\n<|channel>thought\n<channel|>"
            ),
            chat_text="Summarize the image, audio, and video together.",
            images=[gradient_png],
            audios=[{"input_audio": {"data": b64(tone_wav), "format": "wav"}}],
            videos=[video_frames],
            video_timestamp_seconds=[[0.0, 2.0]],
            chat_content=[
                {"type": "text", "text": "Summarize the image, audio, and video together."},
                image_part(gradient_png),
                audio_part(tone_wav),
                video_part(video_gif),
            ],
        ),
    }


def select_cases(selection: str) -> list[BenchmarkCase]:
    cases = build_cases()
    names = [name.strip() for name in selection.split(",") if name.strip()]
    if not names or names == ["all"]:
        names = list(cases)
    unknown = [name for name in names if name not in cases]
    if unknown:
        raise ValueError(f"unknown case(s): {', '.join(unknown)}")
    return [cases[name] for name in names]


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


def prepare_case(model_dir: Path, case: BenchmarkCase) -> PreparedCase:
    require_tokenizers()
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt_tokens = encode_without_special_tokens(tokenizer, case.native_prompt)
    helper = load_gemma4_helper()
    timestamps = (
        video_timestamp_token_ids(tokenizer, case.video_timestamp_seconds)
        if case.videos
        else None
    )
    request = helper.prepare_gemma4_unified_multimodal_request(
        model_dir,
        prompt_tokens,
        images=case.images,
        audios=case.audios,
        videos=case.videos,
        video_timestamp_token_ids=timestamps,
    )
    return PreparedCase(
        case_id=case.case_id,
        description=case.description,
        modalities=case.modalities,
        input_tokens=[int(token) for token in request.input_tokens],
        multimodal_inputs=request.multimodal_inputs,
        original_prompt_tokens=len(prompt_tokens),
        expanded_prompt_tokens=len(request.input_tokens),
        image_soft_tokens=[int(value) for value in request.image_soft_token_counts],
        audio_soft_tokens=[int(value) for value in request.audio_soft_token_counts],
        video_soft_tokens=[int(value) for value in request.video_soft_token_counts],
        video_frame_counts=[int(value) for value in request.video_frame_counts],
        chat_content=case.chat_content,
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


def http_connection(url: str, timeout: int = 300) -> tuple[http.client.HTTPConnection, str]:
    parsed = urllib.parse.urlsplit(url.rstrip("/"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(host, port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    base_path = parsed.path.rstrip("/")
    return conn, base_path


def run_native_one(url: str, model: str, request: PreparedCase, max_output_tokens: int) -> dict[str, Any]:
    conn, base_path = http_connection(url)
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
                    execution_plan = route.get("execution_plan") or execution_plan
            elif event_name == "response":
                response_obj = obj.get("response") or {}
                tokens = response_obj.get("output_tokens")
                if isinstance(tokens, list):
                    output_tokens = len(tokens)
                    if tokens and first_output_wall_s is None:
                        first_output_wall_s = time.perf_counter() - started
                route = response_obj.get("route") or {}
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
    }


def run_chat_one(url: str, model: str, case: PreparedCase, max_output_tokens: int) -> dict[str, Any]:
    conn, base_path = http_connection(url)
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": case.chat_content}],
            "max_tokens": max_output_tokens,
            "temperature": 0.0,
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
        "client_wall_ms": wall_ms,
        "output_tokens": usage.get("completion_tokens"),
        "prompt_tokens_reported": usage.get("prompt_tokens"),
        "content_chars": len(content),
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


def case_prompt_block(case: PreparedCase) -> dict[str, Any]:
    return {
        "original_tokens": case.original_prompt_tokens,
        "expanded_tokens": case.expanded_prompt_tokens,
        "image_soft_tokens": case.image_soft_tokens,
        "audio_soft_tokens": case.audio_soft_tokens,
        "video_soft_tokens": case.video_soft_tokens,
        "video_frame_counts": case.video_frame_counts,
    }


def measured_row(
    *,
    engine: str,
    backend: str,
    layer: str,
    case: PreparedCase,
    runs: list[dict[str, Any]],
    metric_keys: list[str],
) -> dict[str, Any]:
    return {
        "engine": engine,
        "backend": backend,
        "layer": layer,
        "case_id": case.case_id,
        "description": case.description,
        "modalities": case.modalities,
        "status": "measured",
        "prompt": case_prompt_block(case),
        "runs": runs,
        "summary": summarize_runs(runs, metric_keys),
    }


def skipped_row(
    *,
    engine: str,
    backend: str,
    layer: str,
    case: PreparedCase,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "engine": engine,
        "backend": backend,
        "layer": layer,
        "case_id": case.case_id,
        "description": case.description,
        "modalities": case.modalities,
        "status": "skipped",
        "skip_reason": reason,
        "skip_detail": detail,
        "prompt": case_prompt_block(case),
        "runs": [],
        "summary": {},
    }


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


def git_provenance() -> dict[str, Any]:
    status = run_git(["status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "tracked_dirty": bool(status),
        "tracked_dirty_paths": status.splitlines() if status else [],
    }


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprints(model_dir: Path) -> dict[str, str | None]:
    names = ["config.json", "processor_config.json", "tokenizer.json", "model-manifest.json"]
    return {name: sha256_file(model_dir / name) for name in names}


def build_provenance(model_dir: Path) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "git": git_provenance(),
        "model_fingerprints": model_fingerprints(model_dir),
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    cases = [prepare_case(args.model_dir, case) for case in select_cases(args.cases)]
    layers = {layer.strip() for layer in args.layers.split(",") if layer.strip()}
    unknown_layers = layers - {"native_runtime_prefill", "openai_chat_e2e"}
    if unknown_layers:
        raise ValueError(f"unknown layer(s): {', '.join(sorted(unknown_layers))}")

    rows: list[dict[str, Any]] = []
    for case in cases:
        if "native_runtime_prefill" in layers:
            runs = run_repetitions(
                warmup=args.warmup,
                repetitions=args.repetitions,
                cooldown_s=args.cooldown,
                run=lambda case=case: run_native_one(
                    args.url, args.model, case, args.max_output_tokens
                ),
            )
            rows.append(
                measured_row(
                    engine="ax_engine",
                    backend="mlx",
                    layer="native_runtime_prefill",
                    case=case,
                    runs=runs,
                    metric_keys=[
                        "runner_prefill_ttft_ms",
                        "client_wall_ttft_ms",
                        "client_wall_total_ms",
                        "prefill_tok_s",
                    ],
                )
            )

        if "openai_chat_e2e" in layers:
            runs = run_repetitions(
                warmup=args.warmup,
                repetitions=args.repetitions,
                cooldown_s=args.cooldown,
                run=lambda case=case: run_chat_one(
                    args.url, args.model, case, args.max_output_tokens
                ),
            )
            rows.append(
                measured_row(
                    engine="ax_engine",
                    backend="mlx",
                    layer="openai_chat_e2e",
                    case=case,
                    runs=runs,
                    metric_keys=["client_wall_ms", "content_chars", "output_tokens"],
                )
            )

        if not args.no_llama_cpp:
            if "video" in case.modalities:
                rows.append(
                    skipped_row(
                        engine="llama_cpp",
                        backend="metal",
                        layer="openai_chat_e2e",
                        case=case,
                        reason="llama_cpp_video_not_supported",
                        detail="local llama.cpp multimodal documentation covers image/audio but not video",
                    )
                )
            elif not args.llama_url:
                rows.append(
                    skipped_row(
                        engine="llama_cpp",
                        backend="metal",
                        layer="openai_chat_e2e",
                        case=case,
                        reason="no_llama_cpp_server_url",
                        detail="pass --llama-url for an OpenAI-compatible llama.cpp multimodal server",
                    )
                )
            else:
                runs = run_repetitions(
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    cooldown_s=args.cooldown,
                    run=lambda case=case: run_chat_one(
                        args.llama_url, args.model, case, args.max_output_tokens
                    ),
                )
                rows.append(
                    measured_row(
                        engine="llama_cpp",
                        backend="metal",
                        layer="openai_chat_e2e",
                        case=case,
                        runs=runs,
                        metric_keys=["client_wall_ms", "content_chars", "output_tokens"],
                    )
                )

    return {
        "schema": SCHEMA,
        "benchmark": {
            "name": "gemma4_12b_multimodal",
            "model": args.model,
            "model_dir": str(args.model_dir),
            "layers": sorted(layers),
            "cases": [case.case_id for case in cases],
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "cooldown_s": args.cooldown,
            "max_output_tokens": args.max_output_tokens,
        },
        "server": {"ax_engine_url": args.url, "llama_cpp_url": args.llama_url},
        "peer": {
            "llama_cpp": {
                "url": args.llama_url,
                "gguf": str(args.llama_gguf) if args.llama_gguf else None,
                "mmproj": str(args.llama_mmproj) if args.llama_mmproj else None,
                "gguf_sha256": sha256_file(args.llama_gguf) if args.llama_gguf else None,
                "mmproj_sha256": sha256_file(args.llama_mmproj) if args.llama_mmproj else None,
            }
        },
        "command": command_json(args),
        "provenance": build_provenance(args.model_dir),
        "rows": rows,
    }


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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--llama-url")
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
    artifact = build_artifact(args)
    text = json.dumps(artifact, indent=2) + "\n"
    output = args.output or default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
