#!/usr/bin/env python3
"""Run an evidence-producing soak against AX Engine's delegated vLLM route.

The runner intentionally sends each generation POST exactly once. A failed
request is recorded and the next scheduled sample is a new request; ambiguous
transport failures are never retried.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import ipaddress
import json
import math
import mimetypes
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA_VERSION = "com.automatosx.ax-engine.vllm-soak.v1"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_ERROR_CHARS = 512
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}


@dataclass(frozen=True)
class ImageInput:
    data_uri: str
    mime_type: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class ProcessTarget:
    label: str
    pid: int


UrlOpen = Callable[..., BinaryIO]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def bounded_error(error: BaseException | str) -> str:
    text = str(error).replace("\r", " ").replace("\n", " ")
    return text[:MAX_ERROR_CHARS]


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def normalize_base_url(raw: str, *, allow_insecure_http: bool) -> str:
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base URL scheme must be http or https")
    if not parsed.hostname:
        raise ValueError("base URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a query or fragment")
    if parsed.scheme == "http" and not allow_insecure_http:
        host = parsed.hostname
        loopback = host == "localhost"
        with suppress(ValueError):
            loopback = loopback or ipaddress.ip_address(host).is_loopback
        if not loopback:
            raise ValueError(
                "plaintext HTTP is restricted to loopback; use HTTPS or --allow-insecure-http"
            )
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    elif not path.endswith("/v1"):
        path = f"{path}/v1"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def public_endpoint_identity(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def parse_process_target(value: str) -> ProcessTarget:
    label, separator, raw_pid = value.partition("=")
    if not separator or not label or not raw_pid:
        raise argparse.ArgumentTypeError("process target must use LABEL=PID")
    if not label.replace("-", "").replace("_", "").isalnum():
        raise argparse.ArgumentTypeError(
            "process label may contain letters, digits, '-' and '_' only"
        )
    try:
        pid = int(raw_pid)
    except ValueError as error:
        raise argparse.ArgumentTypeError("process PID must be an integer") from error
    if pid <= 0:
        raise argparse.ArgumentTypeError("process PID must be positive")
    return ProcessTarget(label=label, pid=pid)


def load_image(path: Path) -> ImageInput:
    if not path.is_file():
        raise ValueError(f"image is not a regular file: {path}")
    byte_count = path.stat().st_size
    if byte_count <= 0:
        raise ValueError("image must not be empty")
    if byte_count > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} byte soak-runner limit")
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ValueError(f"unsupported image MIME type: {mime_type!r}")
    raw = path.read_bytes()
    return ImageInput(
        data_uri=f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}",
        mime_type=mime_type,
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_payload(
    *,
    model: str,
    prompt: str,
    image: ImageInput | None,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    if image is None:
        content: str | list[dict[str, Any]] = prompt
    else:
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": image.data_uri},
            },
        ]
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def open_request(
    url: str,
    *,
    payload: Mapping[str, Any] | None,
    api_key: str | None,
    timeout: float,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> BinaryIO:
    data = None
    headers = request_headers(api_key)
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        method = "POST"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    # Deliberately one call and no retry loop.
    return urlopen(request, timeout=timeout)


def read_json_response(response: BinaryIO) -> dict[str, Any]:
    payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("response body must be a JSON object")
    return payload


def verify_model_identity(
    base_url: str,
    *,
    model: str,
    api_key: str | None,
    timeout: float,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    with open_request(
        f"{base_url}/models",
        payload=None,
        api_key=api_key,
        timeout=timeout,
        urlopen=urlopen,
    ) as response:
        payload = read_json_response(response)
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("/v1/models response is missing its data array")
    identities = {
        item.get("id")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if model not in identities:
        raise ValueError(f"AX Engine does not advertise the exact model identity {model!r}")
    return {
        "model": model,
        "advertised_model_count": len(identities),
    }


def response_text_and_usage(
    payload: Mapping[str, Any],
) -> tuple[str, str | None, dict[str, Any] | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response is missing a completion choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("completion choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("completion choice is missing its message")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("completion response must contain non-empty text")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ValueError("finish_reason must be a string or null")
    usage = payload.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise ValueError("usage must be an object or null")
    return content, finish_reason, usage


def iter_sse_data(lines: Iterable[bytes]) -> Iterator[str]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data":
            continue
        if separator and value.startswith(" "):
            value = value[1:]
        data_lines.append(value)
    if data_lines:
        yield "\n".join(data_lines)


def read_stream_response(
    response: BinaryIO,
    *,
    request_started: float,
) -> dict[str, Any]:
    parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    chunks = 0
    done = False
    ttft_seconds: float | None = None
    for data in iter_sse_data(response):
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("SSE data must contain a JSON object")
        chunks += 1
        candidate_usage = event.get("usage")
        if candidate_usage is not None:
            if not isinstance(candidate_usage, dict):
                raise ValueError("stream usage must be an object")
            usage = candidate_usage
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError("stream choices must be an array")
        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("stream choice must be an object")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ValueError("stream delta must be an object")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("stream content delta must be a string")
                if content and ttft_seconds is None:
                    ttft_seconds = time.perf_counter() - request_started
                parts.append(content)
            candidate_reason = choice.get("finish_reason")
            if candidate_reason is not None:
                if not isinstance(candidate_reason, str):
                    raise ValueError("stream finish_reason must be a string")
                finish_reason = candidate_reason
    text = "".join(parts)
    if not done:
        raise ValueError("stream ended before [DONE]")
    if not text:
        raise ValueError("stream response must contain non-empty text")
    return {
        "text": text,
        "finish_reason": finish_reason,
        "usage": usage,
        "chunks": chunks,
        "ttft_seconds": ttft_seconds,
        "done": done,
    }


def execute_generation(
    base_url: str,
    *,
    payload: Mapping[str, Any],
    api_key: str | None,
    timeout: float,
    stream: bool,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    started = time.perf_counter()
    with open_request(
        f"{base_url}/chat/completions",
        payload=payload,
        api_key=api_key,
        timeout=timeout,
        urlopen=urlopen,
    ) as response:
        status = int(getattr(response, "status", 200))
        if stream:
            result = read_stream_response(response, request_started=started)
        else:
            body = read_json_response(response)
            text, finish_reason, usage = response_text_and_usage(body)
            result = {
                "text": text,
                "finish_reason": finish_reason,
                "usage": usage,
                "chunks": None,
                "ttft_seconds": None,
                "done": None,
            }
    elapsed = time.perf_counter() - started
    text = result.pop("text")
    return {
        "ok": True,
        "http_status": status,
        "elapsed_seconds": elapsed,
        "response_bytes": len(text.encode("utf-8")),
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        **result,
    }


def process_rss_mib(target: ProcessTarget) -> float:
    status = Path(f"/proc/{target.pid}/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                break
            return int(fields[1]) / 1024.0
    raise RuntimeError(f"VmRSS is unavailable for PID {target.pid}")


def optional_nvidia_float(value: str) -> float | None:
    if value.strip().lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    return float(value)


def gpu_telemetry(gpu_index: int) -> dict[str, float | None]:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError("unexpected nvidia-smi telemetry shape")
    values = [optional_nvidia_float(field) for field in fields]
    return {
        "memory_used_mib": values[0],
        "memory_free_mib": values[1],
        "utilization_percent": values[2],
        "power_watts": values[3],
        "temperature_celsius": values[4],
        "memory_telemetry_supported": (values[0] is not None and values[1] is not None),
    }


def system_memory_available_mib() -> float:
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            return int(fields[1]) / 1024.0
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def collect_telemetry(
    *,
    processes: Iterable[ProcessTarget],
    gpu_index: int | None,
) -> dict[str, Any]:
    telemetry: dict[str, Any] = {"process_rss_mib": {}}
    errors: list[str] = []
    for target in processes:
        try:
            telemetry["process_rss_mib"][target.label] = process_rss_mib(target)
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"{target.label}: {bounded_error(error)}")
    try:
        telemetry["system_memory_available_mib"] = system_memory_available_mib()
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"system memory: {bounded_error(error)}")
    if gpu_index is not None:
        try:
            telemetry["gpu"] = gpu_telemetry(gpu_index)
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
            errors.append(f"GPU: {bounded_error(error)}")
    if errors:
        telemetry["errors"] = errors
    return telemetry


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def window_growth(values: list[float], *, window: int = 5) -> float | None:
    if len(values) < 2:
        return None
    width = min(window, max(1, len(values) // 2))
    return statistics.median(values[-width:]) - statistics.median(values[:width])


def longest_failure_streak(samples: Iterable[Mapping[str, Any]]) -> int:
    longest = 0
    current = 0
    for sample in samples:
        if sample.get("ok") is True:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def summarize(
    samples: list[dict[str, Any]],
    *,
    duration_completed: bool,
    warmup_samples: int,
    max_failures: int,
    max_consecutive_failures: int,
    require_stable_response: bool,
    require_telemetry: bool,
    max_process_rss_growth_mib: float | None,
    max_gpu_memory_growth_mib: float | None,
) -> dict[str, Any]:
    considered = samples[warmup_samples:]
    successes = [sample for sample in considered if sample.get("ok") is True]
    failures = [sample for sample in considered if sample.get("ok") is not True]
    latency = [float(sample["elapsed_seconds"]) for sample in successes]
    ttft = [
        float(sample["ttft_seconds"])
        for sample in successes
        if sample.get("ttft_seconds") is not None
    ]
    response_hashes = {
        str(sample["response_sha256"]) for sample in successes if sample.get("response_sha256")
    }
    process_values: dict[str, list[float]] = {}
    gpu_memory: list[float] = []
    telemetry_errors = 0
    for sample in considered:
        telemetry = sample.get("telemetry")
        if not isinstance(telemetry, dict):
            telemetry_errors += 1
            continue
        if telemetry.get("errors"):
            telemetry_errors += 1
        rss = telemetry.get("process_rss_mib")
        if isinstance(rss, dict):
            for label, value in rss.items():
                process_values.setdefault(str(label), []).append(float(value))
        gpu = telemetry.get("gpu")
        if isinstance(gpu, dict) and gpu.get("memory_used_mib") is not None:
            gpu_memory.append(float(gpu["memory_used_mib"]))

    process_growth = {
        label: window_growth(values) for label, values in sorted(process_values.items())
    }
    gpu_growth = window_growth(gpu_memory)
    reasons: list[str] = []
    if not duration_completed:
        reasons.append("requested duration did not complete")
    if not considered:
        reasons.append("no post-warmup samples were collected")
    if len(failures) > max_failures:
        reasons.append(f"post-warmup failures {len(failures)} exceed {max_failures}")
    failure_streak = longest_failure_streak(considered)
    if failure_streak > max_consecutive_failures:
        reasons.append(
            "longest consecutive failure streak "
            f"{failure_streak} exceeds {max_consecutive_failures}"
        )
    if require_stable_response and len(response_hashes) > 1:
        reasons.append("successful responses were not semantically stable")
    if require_telemetry and telemetry_errors:
        reasons.append(f"{telemetry_errors} samples have telemetry errors")
    if max_process_rss_growth_mib is not None:
        for label, growth in process_growth.items():
            if growth is not None and growth > max_process_rss_growth_mib:
                reasons.append(
                    f"{label} RSS growth {growth:.3f} MiB exceeds "
                    f"{max_process_rss_growth_mib:.3f} MiB"
                )
    if (
        max_gpu_memory_growth_mib is not None
        and gpu_growth is not None
        and gpu_growth > max_gpu_memory_growth_mib
    ):
        reasons.append(
            f"GPU memory growth {gpu_growth:.3f} MiB exceeds {max_gpu_memory_growth_mib:.3f} MiB"
        )
    return {
        "schema": SCHEMA_VERSION,
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "duration_completed": duration_completed,
        "samples_total": len(samples),
        "warmup_samples_excluded": min(warmup_samples, len(samples)),
        "samples_evaluated": len(considered),
        "successes": len(successes),
        "failures": len(failures),
        "longest_consecutive_failures": failure_streak,
        "response_sha256_count": len(response_hashes),
        "latency_seconds": {
            "p50": percentile(latency, 0.50),
            "p95": percentile(latency, 0.95),
            "p99": percentile(latency, 0.99),
            "max": max(latency, default=None),
        },
        "ttft_seconds": {
            "p50": percentile(ttft, 0.50),
            "p95": percentile(ttft, 0.95),
            "p99": percentile(ttft, 0.99),
            "max": max(ttft, default=None),
        },
        "process_rss_growth_mib": process_growth,
        "gpu_memory_growth_mib": gpu_growth,
        "telemetry_error_samples": telemetry_errors,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def mode_for_iteration(mode: str, iteration: int) -> str:
    if mode != "alternate":
        return mode
    return "non-stream" if iteration % 2 == 0 else "stream"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt content; defaults to the canonical document OCR prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=("alternate", "stream", "non-stream"),
        default="alternate",
    )
    parser.add_argument(
        "--duration-seconds",
        type=positive_float,
        default=24 * 60 * 60,
    )
    parser.add_argument("--interval-seconds", type=positive_float, default=60.0)
    parser.add_argument("--timeout-seconds", type=positive_float, default=120.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="AX_ENGINE_API_KEY")
    parser.add_argument("--allow-insecure-http", action="store_true")
    parser.add_argument(
        "--rss-pid",
        action="append",
        default=[],
        type=parse_process_target,
        metavar="LABEL=PID",
    )
    parser.add_argument("--gpu-index", type=non_negative_int)
    parser.add_argument("--require-telemetry", action="store_true")
    parser.add_argument("--warmup-samples", type=non_negative_int, default=5)
    parser.add_argument("--max-failures", type=non_negative_int, default=0)
    parser.add_argument(
        "--max-consecutive-failures",
        type=non_negative_int,
        default=0,
    )
    parser.add_argument("--require-stable-response", action="store_true")
    parser.add_argument(
        "--max-process-rss-growth-mib",
        type=positive_float,
    )
    parser.add_argument(
        "--max-gpu-memory-growth-mib",
        type=positive_float,
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    base_url = normalize_base_url(
        args.base_url,
        allow_insecure_http=args.allow_insecure_http,
    )
    image = load_image(args.image) if args.image is not None else None
    prompt = args.prompt
    if prompt is None:
        prompt = (
            "<image>document parsing."
            if image is not None
            else "Reply with a short health-check token."
        )
    if image is not None and prompt.count("<image>") != 1:
        raise ValueError("image soak prompt must contain one literal '<image>'")
    if image is None and "<image>" in prompt:
        raise ValueError("text soak prompt must not contain '<image>'")
    api_key = os.environ.get(args.api_key_env)
    if api_key is not None and not api_key:
        api_key = None

    args.output_dir.mkdir(parents=True, exist_ok=False)
    readiness = verify_model_identity(
        base_url,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout_seconds,
    )
    manifest = {
        "schema": SCHEMA_VERSION,
        "started_at": utc_now(),
        "endpoint": public_endpoint_identity(base_url),
        "model": args.model,
        "mode": args.mode,
        "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "timeout_seconds": args.timeout_seconds,
        "max_tokens": args.max_tokens,
        "api_key_configured": api_key is not None,
        "readiness": readiness,
        "generation_retry_policy": "never",
        "process_targets": [{"label": target.label, "pid": target.pid} for target in args.rss_pid],
        "gpu_index": args.gpu_index,
        "image": (
            {
                "mime_type": image.mime_type,
                "byte_count": image.byte_count,
                "sha256": image.sha256,
            }
            if image is not None
            else None
        ),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)

    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    deadline = started + args.duration_seconds
    iteration = 0
    samples: list[dict[str, Any]] = []
    samples_path = args.output_dir / "samples.jsonl"
    try:
        with samples_path.open("a", encoding="utf-8") as handle:
            while not stopping and time.monotonic() < deadline:
                scheduled_at = started + iteration * args.interval_seconds
                delay = scheduled_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if stopping or time.monotonic() >= deadline:
                    break
                selected_mode = mode_for_iteration(args.mode, iteration)
                stream = selected_mode == "stream"
                payload = build_payload(
                    model=args.model,
                    prompt=prompt,
                    image=image,
                    max_tokens=args.max_tokens,
                    stream=stream,
                )
                sample: dict[str, Any] = {
                    "schema": SCHEMA_VERSION,
                    "iteration": iteration,
                    "timestamp": utc_now(),
                    "elapsed_since_start_seconds": time.monotonic() - started,
                    "mode": selected_mode,
                }
                try:
                    sample.update(
                        execute_generation(
                            base_url,
                            payload=payload,
                            api_key=api_key,
                            timeout=args.timeout_seconds,
                            stream=stream,
                        )
                    )
                except urllib.error.HTTPError as error:
                    body = error.read(MAX_ERROR_CHARS).decode("utf-8", errors="replace")
                    sample.update(
                        {
                            "ok": False,
                            "http_status": error.code,
                            "error_type": type(error).__name__,
                            "error": bounded_error(body or error),
                        }
                    )
                except (
                    OSError,
                    TimeoutError,
                    ValueError,
                    json.JSONDecodeError,
                ) as error:
                    sample.update(
                        {
                            "ok": False,
                            "http_status": None,
                            "error_type": type(error).__name__,
                            "error": bounded_error(error),
                        }
                    )
                sample["telemetry"] = collect_telemetry(
                    processes=args.rss_pid,
                    gpu_index=args.gpu_index,
                )
                samples.append(sample)
                append_jsonl(handle, sample)
                iteration += 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)

    duration_completed = not stopping and time.monotonic() >= deadline
    summary = summarize(
        samples,
        duration_completed=duration_completed,
        warmup_samples=args.warmup_samples,
        max_failures=args.max_failures,
        max_consecutive_failures=args.max_consecutive_failures,
        require_stable_response=args.require_stable_response,
        require_telemetry=args.require_telemetry,
        max_process_rss_growth_mib=args.max_process_rss_growth_mib,
        max_gpu_memory_growth_mib=args.max_gpu_memory_growth_mib,
    )
    summary.update(
        {
            "started_at": manifest["started_at"],
            "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "manifest": "manifest.json",
            "samples": "samples.jsonl",
        }
    )
    atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"soak setup failed: {bounded_error(error)}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
