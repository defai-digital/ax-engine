#!/usr/bin/env python3
"""Benchmark AX Engine and peer MLX servers through streaming OpenAI chat."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.cross_runtime.single_client.v2"
DEFAULT_PROMPT_TARGETS = (512, 2048)
DEFAULT_GENERATION_TOKENS = 256
DEFAULT_REPETITIONS = 3
DEFAULT_COOLDOWN_S = 15.0
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_QUALITY_SEED = 20_260_728
AX_COUNTER_METRICS = (
    "ax_engine_steps_total",
    "ax_engine_scheduled_tokens_total",
    "ax_engine_mtp_draft_tokens_total",
    "ax_engine_mtp_accepted_tokens_total",
    "ax_engine_mtp_direct_fallback_steps_total",
)
AX_GAUGE_METRICS = ("ax_engine_mtp_accept_rate_ewma_x1000",)

# Intentionally includes words with different tokenizer shapes. The prompt target
# is a deterministic word count, matching the historical single-client session's
# nominal p512/p2048 convention; authoritative server usage is always recorded.
PROMPT_WORDS = (
    "adapter",
    "allocation",
    "analysis",
    "async",
    "attention",
    "benchmark",
    "buffer",
    "cache",
    "checkpoint",
    "compiler",
    "concurrency",
    "configuration",
    "context",
    "correctness",
    "cache_coherency",
    "decode",
    "dequantization",
    "deserialization",
    "deterministic",
    "diagnostic",
    "dispatch",
    "embedding",
    "endpoint",
    "engine",
    "evaluation",
    "generation",
    "gradient",
    "inference",
    "interoperability",
    "kernel_dispatch",
    "latency",
    "manifest",
    "matrix",
    "memory",
    "metadata",
    "microarchitecture",
    "model",
    "multimodality",
    "multimodal",
    "nondeterministic",
    "operator",
    "optimization",
    "parallel",
    "parallelization",
    "pipeline",
    "prefill",
    "prefix_snapshot",
    "prompt",
    "quantization",
    "quantized_matmul",
    "repetition",
    "reproducibility",
    "request",
    "request_identifier",
    "runtime",
    "scheduler",
    "sequence",
    "server",
    "snapshot",
    "speculative",
    "streaming",
    "telemetry",
    "tensor_layout",
    "throughput",
    "tokenizer",
    "validation",
    "vector",
    "vectorization",
    "verification",
    "warmup",
    "workload",
    "autoregressive",
    "backpressure",
    "benchmark_case",
    "observability",
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class EngineSpec:
    key: str
    binary: Path


@dataclass(frozen=True)
class EngineModelOverride:
    engine: str
    label: str
    path: Path


@dataclass(frozen=True)
class QualityTask:
    task_id: str
    profile: str
    category: str
    prompt: str
    max_tokens: int
    checks: tuple[dict[str, Any], ...]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_csv_ints(raw: str, *, field: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{field} must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{field} values must be positive")
    return values


def parse_model(raw: str) -> ModelSpec:
    label, separator, path_raw = raw.partition("=")
    if not separator or not label.strip() or not path_raw.strip():
        raise argparse.ArgumentTypeError("--model must use LABEL=/absolute/model/path")
    path = Path(path_raw).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise argparse.ArgumentTypeError(f"model directory has no config.json: {path}")
    return ModelSpec(label=label.strip(), path=path)


def parse_engine_model(raw: str) -> EngineModelOverride:
    engine_and_label, separator, path_raw = raw.partition("=")
    engine, label_separator, label = engine_and_label.partition(":")
    if (
        not separator
        or not label_separator
        or not engine.strip()
        or not label.strip()
        or not path_raw.strip()
    ):
        raise argparse.ArgumentTypeError(
            "--engine-model must use ENGINE:LABEL=/absolute/model/path"
        )
    path = Path(path_raw).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise argparse.ArgumentTypeError(f"model directory has no config.json: {path}")
    return EngineModelOverride(engine=engine.strip(), label=label.strip(), path=path)


QUALITY_PROFILES = frozenset({"agent-coding", "general"})
QUALITY_CHECK_KINDS = frozenset(
    {"contains", "exact", "json-equals", "json-keys", "json-valid", "python-syntax"}
)
CONTROL_TOKEN_MARKERS = (
    "<|eot|>",
    "<|endoftext|>",
    "<|im_end|>",
    "<｜User｜>",
    "<｜end▁of▁sentence｜>",
)
PYTHON_FENCE = re.compile(r"```(?:python|py)\b[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ANY_FENCE = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)


def require_quality_string(
    payload: dict[str, Any], field: str, path: Path, line_number: int
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"quality task at {path}:{line_number} needs non-empty {field}")
    return value


def load_quality_tasks(path: Path) -> tuple[QualityTask, ...]:
    """Load a small checksum-bound quality suite using AXQuant-style checks."""
    tasks: list[QualityTask] = []
    with path.expanduser().resolve().open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid quality JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"quality task at {path}:{line_number} must be an object")
            task_id = require_quality_string(payload, "task_id", path, line_number)
            profile = require_quality_string(payload, "profile", path, line_number)
            category = require_quality_string(payload, "category", path, line_number)
            prompt = require_quality_string(payload, "prompt", path, line_number)
            max_tokens = payload.get("max_tokens")
            checks = payload.get("checks")
            if profile not in QUALITY_PROFILES:
                raise ValueError(
                    f"quality task at {path}:{line_number} has unsupported profile {profile!r}"
                )
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
                raise ValueError(
                    f"quality task at {path}:{line_number} needs positive integer max_tokens"
                )
            if not isinstance(checks, list) or not checks:
                raise ValueError(f"quality task at {path}:{line_number} needs checks")
            validated_checks: list[dict[str, Any]] = []
            for check_index, check in enumerate(checks):
                if not isinstance(check, dict):
                    raise ValueError(
                        f"quality check {check_index} at {path}:{line_number} must be an object"
                    )
                kind = check.get("kind")
                if kind not in QUALITY_CHECK_KINDS:
                    raise ValueError(
                        f"quality check {check_index} at {path}:{line_number} "
                        f"has unsupported kind {kind!r}"
                    )
                value = check.get("value")
                if kind in {"contains", "exact"} and not isinstance(value, str):
                    raise ValueError(
                        f"quality check {check_index} at {path}:{line_number} "
                        f"requires a string value"
                    )
                if kind == "json-keys" and (
                    not isinstance(value, list) or not all(isinstance(key, str) for key in value)
                ):
                    raise ValueError(
                        f"quality check {check_index} at {path}:{line_number} "
                        f"requires a string-list value"
                    )
                validated_checks.append(dict(check))
            tasks.append(
                QualityTask(
                    task_id=task_id,
                    profile=profile,
                    category=category,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    checks=tuple(validated_checks),
                )
            )
    if not tasks:
        raise ValueError(f"quality dataset contains no tasks: {path}")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"quality task IDs must be unique: {path}")
    if {task.profile for task in tasks} != QUALITY_PROFILES:
        raise ValueError("quality dataset must contain both agent-coding and general profiles")
    return tuple(tasks)


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def strip_control_tokens(value: str) -> str:
    text = value.strip()
    while text.startswith("</think>"):
        text = text.removeprefix("</think>").lstrip()
    cut = len(text)
    for marker in CONTROL_TOKEN_MARKERS:
        index = text.find(marker)
        if 0 <= index < cut:
            cut = index
    return text[:cut].strip()


def unfenced(value: str) -> str:
    stripped = strip_control_tokens(value)
    match = re.fullmatch(r"```(?:json|python|py)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    embedded = re.search(
        r"```(?:json|python|py)?\s*(.*?)```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return embedded.group(1).strip() if embedded else stripped


def python_source(value: str) -> str:
    text = strip_control_tokens(value)
    python_fence = PYTHON_FENCE.search(text)
    if python_fence:
        return python_fence.group(1).strip()
    any_fence = ANY_FENCE.search(text)
    if any_fence:
        return any_fence.group(1).strip()
    return text


def parse_json_value(value: str) -> Any:
    text = unfenced(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        for start, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            return parsed
        raise original_error


def score_quality_task(task: QualityTask, output: str) -> tuple[float, dict[str, float]]:
    scored = strip_control_tokens(output)
    check_scores: dict[str, float] = {}
    for index, check in enumerate(task.checks):
        kind = str(check["kind"])
        value = check.get("value")
        if kind == "exact":
            passed = normalized_text(scored) == normalized_text(str(value))
        elif kind == "contains":
            passed = normalized_text(str(value)) in normalized_text(scored)
        elif kind == "python-syntax":
            try:
                ast.parse(python_source(output))
                passed = True
            except SyntaxError:
                passed = False
        elif kind == "json-valid":
            try:
                parse_json_value(output)
                passed = True
            except (json.JSONDecodeError, ValueError):
                passed = False
        elif kind == "json-equals":
            try:
                passed = parse_json_value(output) == value
            except (json.JSONDecodeError, ValueError):
                passed = False
        elif kind == "json-keys":
            try:
                parsed = parse_json_value(output)
                passed = (
                    isinstance(parsed, dict)
                    and isinstance(value, list)
                    and all(key in parsed for key in value)
                )
            except (json.JSONDecodeError, ValueError):
                passed = False
        else:  # pragma: no cover - load_quality_tasks rejects unsupported checks.
            raise AssertionError(f"unsupported quality check: {kind}")
        check_scores[f"{kind}:{index}"] = float(passed)
    return sum(check_scores.values()) / len(check_scores), check_scores


def model_for_engine(
    model: ModelSpec,
    engine: str,
    overrides: dict[tuple[str, str], Path],
) -> ModelSpec:
    return ModelSpec(model.label, overrides.get((engine, model.label), model.path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model: ModelSpec) -> dict[str, Any]:
    identity_files: dict[str, str] = {}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        candidate = model.path / name
        if candidate.is_file():
            identity_files[name] = file_sha256(candidate)
    weights = sorted(model.path.glob("*.safetensors"))
    return {
        "label": model.label,
        "path": str(model.path),
        "identity_file_sha256": identity_files,
        "safetensors_files": len(weights),
        "safetensors_bytes": sum(path.stat().st_size for path in weights),
    }


def omlx_base_path(model: ModelSpec) -> Path:
    return model.path.parent.parent / ".omlx-benchmark-state"


def configure_omlx_mtp(model: ModelSpec, *, depth: int = 3) -> Path:
    """Persist oMLX's per-model Lightning-MTP toggle in its isolated state."""
    base_path = omlx_base_path(model)
    settings_path = base_path / "model_settings.json"
    if settings_path.is_file():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}
    data.setdefault("version", 1)
    models = data.setdefault("models", {})
    current = models.setdefault(model.path.name, {})
    current.update(
        {
            "dflash_enabled": False,
            "mtp_enabled": True,
            "mtp_num_draft_tokens": depth,
            "vlm_mtp_enabled": False,
        }
    )
    base_path.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return settings_path


def deterministic_prompt(target: int, rep: int) -> tuple[int, str]:
    seed = 100_000 + target * 100 + rep
    generator = random.Random(seed)
    words = [generator.choice(PROMPT_WORDS) for _ in range(target)]
    prefix = "Notes:\n"
    return seed, prefix + " ".join(words)


def binary_version(binary: Path) -> str:
    result = subprocess.run(
        [str(binary), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = (result.stdout or result.stderr).strip()
    return output or f"version command exited {result.returncode}"


def runner_identity() -> dict[str, Any]:
    path = Path(__file__).resolve()
    repository = path.parents[1]
    relative_path = path.relative_to(repository).as_posix()
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    blob = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", f"HEAD:{relative_path}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--", relative_path],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "git_commit": commit.stdout.strip() or None,
        "git_blob": blob.stdout.strip() or None,
        "clean_at_start": blob.returncode == 0 and not status.stdout.strip(),
    }


def engine_command(
    engine: EngineSpec,
    *,
    model: ModelSpec,
    port: int,
    mlxcel_draft_model: Path | None = None,
    mtplx_force_unverified: bool = False,
) -> list[str]:
    common = ["--host", "127.0.0.1", "--port", str(port)]
    if engine.key == "ax-engine":
        return [
            str(engine.binary),
            "--model-id",
            "local",
            "--mlx",
            "--mlx-model-artifacts-dir",
            str(model.path),
            *common,
            "--max-concurrent-requests",
            "1",
            "--max-concurrent-requests-per-model",
            "1",
        ]
    if engine.key == "mlxcel":
        speculative = (
            []
            if mlxcel_draft_model is None
            else [
                "--model-draft",
                str(mlxcel_draft_model),
                "--draft-kind",
                "mtp",
                "--draft-block-size",
                "3",
            ]
        )
        return [
            str(engine.binary),
            "-m",
            str(model.path),
            "--alias",
            "local",
            *speculative,
            *common,
            "--parallel",
            "1",
            "--max-batch-prefill",
            "1",
            "--ctx-size",
            "65536",
            "--no-prompt-cache",
        ]
    if engine.key == "omlx":
        return [
            str(engine.binary),
            "serve",
            "--model-dir",
            str(model.path.parent),
            *common,
            "--max-concurrent-requests",
            "1",
            "--memory-guard",
            "off",
            "--no-cache",
            "--base-path",
            str(omlx_base_path(model)),
        ]
    if engine.key == "mtplx":
        admission_override = (
            ["--unsafe-force-unverified", "--yes"] if mtplx_force_unverified else []
        )
        return [
            str(engine.binary),
            "serve",
            "--model",
            str(model.path),
            "--model-id",
            "local",
            *common,
            "--no-auth",
            "--profile",
            "turbo",
            "--depth",
            "3",
            "--mtp",
            *admission_override,
            "--scheduler-mode",
            "serial",
            "--max-active-requests",
            "1",
            "--ssd-session-cache",
            "off",
            "--reasoning",
            "off",
            "--stream-interval",
            "1",
            "--warmup-tokens",
            "0",
            "--no-stats-footer",
            "--fan-mode",
            "default",
        ]
    raise ValueError(f"unsupported engine: {engine.key}")


def wait_for_server(port: int, process: subprocess.Popen[bytes], timeout_s: float) -> float:
    started = time.perf_counter()
    deadline = started + timeout_s
    urls = (
        f"http://127.0.0.1:{port}/health",
        f"http://127.0.0.1:{port}/v1/models",
    )
    while time.perf_counter() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"server exited before readiness with code {exit_code}")
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if 200 <= response.status < 300:
                        return time.perf_counter() - started
            except (OSError, urllib.error.URLError):
                pass
        time.sleep(1)
    raise TimeoutError(f"server did not become ready on port {port} in {timeout_s:.0f}s")


def stop_process(process: subprocess.Popen[bytes], timeout_s: float = 30.0) -> tuple[int, bool]:
    forced = False
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            forced = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=timeout_s)
    return int(process.returncode or 0), forced


def decode_sse_frames(
    lines: Iterable[bytes],
    *,
    started_at: float | None = None,
) -> Iterable[tuple[dict[str, Any] | None, float]]:
    started = time.perf_counter() if started_at is None else started_at
    data_lines: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line:
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            continue
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        data_lines.clear()
        elapsed = time.perf_counter() - started
        if data == "[DONE]":
            yield None, elapsed
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid SSE JSON: {data[:200]}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("SSE payload must be an object")
        yield payload, elapsed


def run_request(
    *,
    port: int,
    prompt: str,
    seed: int,
    generation_tokens: int,
    timeout_s: float,
    model_id: str = "local",
    capture_text: bool = False,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": generation_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": 0,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    first_chunk_s: float | None = None
    last_chunk_s: float | None = None
    finish_reason: str | None = None
    done = False
    visible_chars = 0
    visible_parts: list[str] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    runtime_stats: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"chat request returned HTTP {response.status}")
        for payload, elapsed_s in decode_sse_frames(response, started_at=started_at):
            if payload is None:
                done = True
                continue
            error = payload.get("error")
            if error is not None:
                raise RuntimeError(f"server stream error: {json.dumps(error, sort_keys=True)}")
            # MTPLX publishes its effective route and speculation counters on
            # the terminal stream frame.  Preserve that evidence in the raw
            # artifact instead of silently discarding it; effective depth and
            # acceptance must be audited independently from requested flags.
            mtplx_stats = payload.get("mtplx_stats")
            if isinstance(mtplx_stats, dict):
                runtime_stats["mtplx_stats"] = mtplx_stats
            usage = payload.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("prompt_tokens"), int):
                    prompt_tokens = int(usage["prompt_tokens"])
                if isinstance(usage.get("completion_tokens"), int):
                    completion_tokens = int(usage["completion_tokens"])
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            emitted = False
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                if isinstance(choice.get("finish_reason"), str):
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                for key in ("reasoning_content", "content"):
                    value = delta.get(key)
                    if isinstance(value, str) and value:
                        visible_chars += len(value)
                        visible_parts.append(value)
                        if key == "content":
                            content_parts.append(value)
                        else:
                            reasoning_parts.append(value)
                        emitted = True
            if emitted:
                first_chunk_s = elapsed_s if first_chunk_s is None else first_chunk_s
                last_chunk_s = elapsed_s
    e2e_s = time.perf_counter() - started_at
    if not done:
        raise RuntimeError("chat stream ended without [DONE]")
    if prompt_tokens is None or completion_tokens is None:
        raise RuntimeError("chat stream ended without authoritative token usage")
    if first_chunk_s is None or last_chunk_s is None:
        raise RuntimeError("chat stream emitted no content or reasoning content")
    decode_s = max(last_chunk_s - first_chunk_s, 0.0)
    decode_tps = (
        (completion_tokens - 1) / decode_s if completion_tokens > 1 and decode_s > 0 else None
    )
    visible_text = "".join(visible_parts)
    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": first_chunk_s,
        "prefill_tps": prompt_tokens / first_chunk_s if first_chunk_s > 0 else None,
        "decode_tps": decode_tps,
        "decode_window_s": decode_s,
        "e2e_s": e2e_s,
        "visible_chars": visible_chars,
        "text_sha256": hashlib.sha256(visible_text.encode()).hexdigest(),
        "finish_reason": finish_reason,
    }
    if capture_text:
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        result.update(
            {
                "content": content,
                "reasoning_content": reasoning,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
            }
        )
    result.update(runtime_stats)
    return result


def parse_unlabelled_prometheus_metrics(text: str) -> dict[str, float]:
    """Parse only aggregate, unlabelled samples from AX's metrics endpoint."""
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or "{" in fields[0]:
            continue
        try:
            values[fields[0]] = float(fields[1])
        except ValueError:
            continue
    return values


def fetch_ax_metrics(port: int, timeout_s: float = 5.0) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=timeout_s) as response:
        body = response.read().decode("utf-8", errors="replace")
    available = parse_unlabelled_prometheus_metrics(body)
    selected = (*AX_COUNTER_METRICS, *AX_GAUGE_METRICS)
    return {name: available[name] for name in selected if name in available}


def ax_metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    delta = {
        name: after[name] - before.get(name, 0.0) for name in AX_COUNTER_METRICS if name in after
    }
    delta.update({name: after[name] for name in AX_GAUGE_METRICS if name in after})
    return delta


def median(values: Iterable[float | int | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return statistics.median(available) if available else None


def require_complete_generation(observed: dict[str, Any], expected_tokens: int) -> None:
    completion_tokens = observed.get("completion_tokens")
    if completion_tokens != expected_tokens:
        raise RuntimeError(
            "generation ended before the fixed-length benchmark window: "
            f"expected {expected_tokens} completion tokens, observed {completion_tokens}"
        )


def summarize_measurements(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {(str(row["engine"]), str(row["model"]), int(row["prompt_target"])) for row in measurements}
    )
    rows: list[dict[str, Any]] = []
    for engine, model, prompt_target in keys:
        selected = [
            row
            for row in measurements
            if row["engine"] == engine
            and row["model"] == model
            and row["prompt_target"] == prompt_target
        ]
        rows.append(
            {
                "engine": engine,
                "model": model,
                "prompt_target": prompt_target,
                "prompt_tokens": round(median(row["prompt_tokens"] for row in selected) or 0),
                "completion_tokens": round(
                    median(row["completion_tokens"] for row in selected) or 0
                ),
                "ttft_s": median(row["ttft_s"] for row in selected),
                "prefill_tps": median(row["prefill_tps"] for row in selected),
                "decode_tps": median(row["decode_tps"] for row in selected),
                "reps": len(selected),
            }
        )
    return rows


def summarize_quality_measurements(
    measurements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted(
        {(str(row["engine"]), str(row["model"]), str(row["profile"])) for row in measurements}
    )
    rows: list[dict[str, Any]] = []
    for engine, model, profile in keys:
        selected = [
            row
            for row in measurements
            if row["engine"] == engine and row["model"] == model and row["profile"] == profile
        ]
        task_ids = sorted({str(row["task_id"]) for row in selected})
        content_hashes = {
            task_id: sorted(
                {str(row["content_sha256"]) for row in selected if row["task_id"] == task_id}
            )
            for task_id in task_ids
        }
        rows.append(
            {
                "engine": engine,
                "model": model,
                "profile": profile,
                "tasks": len(task_ids),
                "measurements": len(selected),
                "score": sum(float(row["score"]) for row in selected) / len(selected),
                "pass_rate": sum(bool(row["passed"]) for row in selected) / len(selected),
                "all_pass": all(bool(row["passed"]) for row in selected),
                "deterministic_across_repetitions": all(
                    len(content_hashes[task_id]) == 1 for task_id in task_ids
                ),
            }
        )
    return rows


def summarize_quality_consensus(
    measurements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted({(str(row["model"]), str(row["task_id"])) for row in measurements})
    rows: list[dict[str, Any]] = []
    for model, task_id in keys:
        selected = [
            row for row in measurements if row["model"] == model and row["task_id"] == task_id
        ]
        rows.append(
            {
                "model": model,
                "task_id": task_id,
                "profile": selected[0]["profile"],
                "all_runtimes_pass": all(bool(row["passed"]) for row in selected),
                "exact_output_match": len({str(row["content_sha256"]) for row in selected}) == 1,
                "distinct_outputs": len({str(row["content_sha256"]) for row in selected}),
            }
        )
    return rows


def engine_order(
    model_index: int,
    rep: int,
    engines: tuple[str, ...] = ("ax-engine", "mlxcel"),
) -> tuple[str, ...]:
    offset = (model_index + rep) % len(engines)
    return engines[offset:] + engines[:offset]


def sanitize_hardware_profile(profile: str) -> str:
    sensitive_fields = (
        "Serial Number",
        "Hardware UUID",
        "Provisioning UDID",
        "Activation Lock Status",
    )
    return "\n".join(
        line for line in profile.splitlines() if not line.strip().startswith(sensitive_fields)
    )


def collect_host() -> dict[str, Any]:
    def command(*parts: str) -> str:
        result = subprocess.run(parts, check=False, capture_output=True, text=True, timeout=30)
        return (result.stdout or result.stderr).strip()

    return {
        "platform": platform.system().lower(),
        "machine": platform.machine(),
        "macos": command("sw_vers"),
        "hardware": sanitize_hardware_profile(command("system_profiler", "SPHardwareDataType")),
        "power": command("pmset", "-g", "batt"),
        "load_average": os.getloadavg(),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    engines = {"ax-engine": EngineSpec("ax-engine", args.ax_server.resolve())}
    for key, binary in (
        ("mlxcel", args.mlxcel_server),
        ("omlx", args.omlx_server),
        ("mtplx", args.mtplx_server),
    ):
        if binary is not None:
            engines[key] = EngineSpec(key, binary.resolve())
    if len(engines) < 2:
        raise ValueError("at least one peer server must be configured")
    for engine in engines.values():
        if not engine.binary.is_file():
            raise FileNotFoundError(f"server binary does not exist: {engine.binary}")

    model_overrides = {
        (override.engine, override.label): override.path for override in args.engine_model
    }
    quality_tasks = (
        load_quality_tasks(args.quality_dataset) if args.quality_dataset is not None else ()
    )
    if args.omlx_mtp:
        for model in args.model:
            configure_omlx_mtp(model_for_engine(model, "omlx", model_overrides))

    runner = runner_identity()
    binary_identities = {
        key: {
            "path": str(engine.binary),
            "sha256": file_sha256(engine.binary),
            "version": binary_version(engine.binary),
        }
        for key, engine in engines.items()
    }
    model_identities = [model_identity(model) for model in args.model]
    engine_model_identities = {
        engine_key: [
            model_identity(model_for_engine(model, engine_key, model_overrides))
            for model in args.model
        ]
        for engine_key in engines
    }
    output = args.output.resolve()
    log_dir = (args.log_dir or output.parent / "logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    measurements: list[dict[str, Any]] = []
    quality_measurements: list[dict[str, Any]] = []
    process_audit: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started_at = utc_now()
    performance_conditions_start = collect_host()

    for model_index, model in enumerate(args.model):
        for rep in range(args.repetitions):
            for engine_key in engine_order(model_index, rep, tuple(engines)):
                engine = engines[engine_key]
                runtime_model = model_for_engine(model, engine_key, model_overrides)
                command = engine_command(
                    engine,
                    model=runtime_model,
                    port=args.port,
                    mlxcel_draft_model=args.mlxcel_draft_model,
                    mtplx_force_unverified=args.mtplx_force_unverified,
                )
                api_model_id = runtime_model.path.name if engine_key == "omlx" else "local"
                log_path = log_dir / f"{model_index:02d}-{model.label}-{rep}-{engine_key}.log"
                environment = os.environ.copy()
                if engine_key == "ax-engine":
                    environment["AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY"] = "1"
                    # Peers are started with their request/prefix caches off.
                    # Disable both AX cache layers too: core retained-prefix
                    # reuse and the MLX runner's portable snapshot cache.
                    environment["AX_ENGINE_PREFIX_REUSE_DISABLED"] = "1"
                    environment["AX_MLX_PREFIX_CACHE_MAX_BYTES"] = "0"
                    environment["AX_MLX_PREFIX_CACHE_MAX_ENTRIES"] = "0"
                    environment["AX_MLX_PREFIX_CACHE_DISK_DISABLED"] = "1"
                    if not args.ax_speculative:
                        environment["AX_NO_SPEC"] = "1"
                    if args.ax_force_mtp:
                        environment["AX_MLX_MTP_FORCE_REQUESTED"] = "1"
                    if args.ax_projected_replay:
                        environment["AX_MLX_MTP_LINEAR_PROJECTED_REPLAY"] = "1"
                    if args.ax_relaxed_verify:
                        # Keep the exact Qwen MTP drafter active while routing
                        # only the target verifier through the stock/oMLX-style
                        # multi-row arithmetic.  Clearing the exact profile here
                        # bypassed both the intended hybrid policy and the
                        # verifier-only QMM kernels, understating AX decode.
                        environment["AX_MLX_QWEN_LINEAR_MTP_EXACT"] = "1"
                        environment["AX_MLX_MTP_RELAXED_TARGET_VERIFY"] = "1"
                elif engine_key == "mtplx":
                    environment["MTPLX_STREAM_COALESCE"] = "0"
                elif engine_key == "mlxcel" and args.mlxcel_relaxed_verify:
                    # MLXcel's M5 exactness probe rejects Qwen's multi-row
                    # verifier even after its qmv_wide fallback.  oMLX and the
                    # admitted AX profile use the same target-verified,
                    # singleton-non-bit-exact arithmetic class, so opt into
                    # MLXcel's equivalent route for the matched comparison.
                    environment["MLXCEL_MTP_ALLOW_INEXACT"] = "1"
                launched_at = utc_now()
                with log_path.open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                    audit: dict[str, Any] = {
                        "engine": engine_key,
                        "model": model.label,
                        "rep": rep,
                        "pid": process.pid,
                        "command": command,
                        "log_path": str(log_path),
                        "launched_at": launched_at,
                    }
                    try:
                        audit["startup_s"] = wait_for_server(
                            args.port, process, args.startup_timeout
                        )
                        if args.warmup_tokens > 0:
                            warmup_seed, warmup_prompt = deterministic_prompt(128, rep)
                            run_request(
                                port=args.port,
                                prompt=warmup_prompt,
                                seed=warmup_seed,
                                generation_tokens=args.warmup_tokens,
                                timeout_s=args.timeout,
                                model_id=api_model_id,
                            )
                        for prompt_target in args.prompt_targets:
                            seed, prompt = deterministic_prompt(prompt_target, rep)
                            measured_at = utc_now()
                            try:
                                ax_metrics_before = (
                                    fetch_ax_metrics(args.port) if engine_key == "ax-engine" else {}
                                )
                                observed = run_request(
                                    port=args.port,
                                    prompt=prompt,
                                    seed=seed,
                                    generation_tokens=args.generation_tokens,
                                    timeout_s=args.timeout,
                                    model_id=api_model_id,
                                )
                                if engine_key == "ax-engine":
                                    observed["ax_metrics"] = ax_metric_delta(
                                        ax_metrics_before, fetch_ax_metrics(args.port)
                                    )
                                require_complete_generation(observed, args.generation_tokens)
                            except Exception as exc:  # noqa: BLE001 - preserve failed evidence.
                                errors.append(
                                    {
                                        "kind": "performance",
                                        "engine": engine_key,
                                        "model": model.label,
                                        "prompt_target": prompt_target,
                                        "rep": rep,
                                        "seed": seed,
                                        "error": str(exc),
                                    }
                                )
                                continue
                            measurements.append(
                                {
                                    "engine": engine_key,
                                    "model": model.label,
                                    "model_dir": str(runtime_model.path),
                                    "prompt_target": prompt_target,
                                    "rep": rep,
                                    "seed": seed,
                                    "measured_at": measured_at,
                                    **observed,
                                }
                            )
                        for task_index, task in enumerate(quality_tasks):
                            seed = args.quality_seed + task_index
                            measured_at = utc_now()
                            try:
                                ax_metrics_before = (
                                    fetch_ax_metrics(args.port) if engine_key == "ax-engine" else {}
                                )
                                observed = run_request(
                                    port=args.port,
                                    prompt=task.prompt,
                                    seed=seed,
                                    generation_tokens=task.max_tokens,
                                    timeout_s=args.timeout,
                                    model_id=api_model_id,
                                    capture_text=True,
                                )
                                if engine_key == "ax-engine":
                                    observed["ax_metrics"] = ax_metric_delta(
                                        ax_metrics_before, fetch_ax_metrics(args.port)
                                    )
                                score, check_scores = score_quality_task(
                                    task, str(observed["content"])
                                )
                            except Exception as exc:  # noqa: BLE001 - preserve failed evidence.
                                errors.append(
                                    {
                                        "kind": "quality",
                                        "engine": engine_key,
                                        "model": model.label,
                                        "task_id": task.task_id,
                                        "profile": task.profile,
                                        "rep": rep,
                                        "seed": seed,
                                        "error": str(exc),
                                    }
                                )
                                continue
                            quality_measurements.append(
                                {
                                    "engine": engine_key,
                                    "model": model.label,
                                    "model_dir": str(runtime_model.path),
                                    "task_id": task.task_id,
                                    "profile": task.profile,
                                    "category": task.category,
                                    "rep": rep,
                                    "seed": seed,
                                    "measured_at": measured_at,
                                    "prompt_sha256": hashlib.sha256(
                                        task.prompt.encode()
                                    ).hexdigest(),
                                    "checks": list(task.checks),
                                    "check_scores": check_scores,
                                    "score": score,
                                    "passed": all(value == 1.0 for value in check_scores.values()),
                                    **observed,
                                }
                            )
                    except Exception as exc:  # noqa: BLE001 - preserve failed process evidence.
                        errors.append(
                            {
                                "kind": "process",
                                "engine": engine_key,
                                "model": model.label,
                                "rep": rep,
                                "error": str(exc),
                            }
                        )
                    finally:
                        exit_code, forced = stop_process(process)
                        audit["exit_code"] = exit_code
                        audit["forced_kill"] = forced
                        process_audit.append(audit)
                if args.cooldown > 0:
                    time.sleep(args.cooldown)

    expected = len(args.model) * len(args.prompt_targets) * args.repetitions * len(engines)
    expected_quality = len(args.model) * len(quality_tasks) * args.repetitions * len(engines)
    status = (
        "complete"
        if not errors
        and len(measurements) == expected
        and len(quality_measurements) == expected_quality
        else "incomplete"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "started_at": started_at,
        "updated_at": utc_now(),
        "config": {
            "models": [model.label for model in args.model],
            "prompt_targets": list(args.prompt_targets),
            "generation_tokens": args.generation_tokens,
            "repetitions": args.repetitions,
            "warmup_tokens_per_process": args.warmup_tokens,
            "cooldown_s": args.cooldown,
            "engines": list(engines),
            "engine_model_overrides": {
                f"{engine}:{label}": str(path)
                for (engine, label), path in sorted(model_overrides.items())
            },
            "engine_versions": {
                key: binary_version(engine.binary) for key, engine in engines.items()
            },
            "mlxcel_draft_model": (
                str(args.mlxcel_draft_model.resolve())
                if args.mlxcel_draft_model is not None
                else None
            ),
            "mlxcel_relaxed_verify": bool(args.mlxcel_relaxed_verify),
            "mtplx_force_unverified": bool(args.mtplx_force_unverified),
            "omlx_mtp": bool(args.omlx_mtp),
            "ax_speculative": bool(args.ax_speculative),
            "ax_force_mtp": bool(args.ax_force_mtp),
            "ax_projected_replay": bool(args.ax_projected_replay),
            "ax_relaxed_verify": bool(args.ax_relaxed_verify),
            "quality": (
                {
                    "dataset": str(args.quality_dataset.resolve()),
                    "dataset_sha256": file_sha256(args.quality_dataset.resolve()),
                    "profiles": sorted({task.profile for task in quality_tasks}),
                    "tasks": [task.task_id for task in quality_tasks],
                    "seed": args.quality_seed,
                    "raw_outputs_retained": True,
                }
                if args.quality_dataset is not None
                else None
            ),
            "methodology": {
                "endpoint": "/v1/chat/completions",
                "streaming": True,
                "temperature": 0,
                "ttft": "request send to first content or reasoning chunk",
                "prefill_tps": "authoritative prompt token count divided by client TTFT",
                "decode_tps": (
                    "(completion_tokens - 1) / (last content chunk - first content chunk)"
                ),
                "cache_policy": "fresh server process per engine/model/repetition",
                "engine_order": "rotated by model index and repetition",
                "prompt_policy": "deterministic nominal word count; same text and seed per engine",
                "quality_policy": (
                    "AXQuant-style objective checks; same prompt, greedy controls, and seed per "
                    "engine; quality requests run after timed throughput requests"
                ),
            },
        },
        "host": {
            "start": performance_conditions_start,
            "end": collect_host(),
        },
        "binaries": binary_identities,
        "runner": runner,
        "models": model_identities,
        "engine_models": engine_model_identities,
        "mlxcel_draft_model": (
            model_identity(ModelSpec("qwen-mtp-drafter", args.mlxcel_draft_model.resolve()))
            if args.mlxcel_draft_model is not None
            else None
        ),
        "expected_measurements": expected,
        "measurements": measurements,
        "results": summarize_measurements(measurements),
        "expected_quality_measurements": expected_quality,
        "quality_measurements": quality_measurements,
        "quality_results": summarize_quality_measurements(quality_measurements),
        "quality_consensus": summarize_quality_consensus(quality_measurements),
        "quality_gate_pass": (
            (
                len(quality_measurements) == expected_quality
                and all(bool(row["passed"]) for row in quality_measurements)
            )
            if quality_tasks
            else None
        ),
        "process_audit": process_audit,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="Repeat LABEL=/absolute/model/path for every benchmark checkpoint.",
    )
    parser.add_argument(
        "--engine-model",
        action="append",
        type=parse_engine_model,
        default=[],
        help=(
            "Override one runtime's compatible view of a checkpoint with "
            "ENGINE:LABEL=/absolute/model/path."
        ),
    )
    parser.add_argument("--ax-server", type=Path, required=True)
    parser.add_argument("--mlxcel-server", type=Path)
    parser.add_argument(
        "--mlxcel-draft-model",
        type=Path,
        help=(
            "Qwen MTP drafter directory for MLXcel; enables --draft-kind mtp "
            "with a three-position verify block."
        ),
    )
    parser.add_argument(
        "--mlxcel-relaxed-verify",
        action="store_true",
        help=(
            "Set MLXCEL_MTP_ALLOW_INEXACT=1 so its target-verified Qwen MTP "
            "route may use multi-row arithmetic not bit-identical to singleton decode."
        ),
    )
    parser.add_argument("--omlx-server", type=Path)
    parser.add_argument(
        "--omlx-mtp",
        action="store_true",
        help=(
            "Enable oMLX Lightning MTP at depth 3 in the benchmark's isolated "
            "per-model settings store."
        ),
    )
    parser.add_argument("--mtplx-server", type=Path)
    parser.add_argument(
        "--mtplx-force-unverified",
        action="store_true",
        help=(
            "Pass MTPLX's explicit --unsafe-force-unverified --yes admission override; "
            "the raw command is retained in the benchmark artifact."
        ),
    )
    parser.add_argument(
        "--ax-speculative",
        action="store_true",
        help="Keep AX speculative decoding enabled; the historical default sets AX_NO_SPEC=1.",
    )
    parser.add_argument(
        "--ax-force-mtp",
        action="store_true",
        help=(
            "Request AX model MTP for an uncertified third-party pack by setting "
            "AX_MLX_MTP_FORCE_REQUESTED=1."
        ),
    )
    parser.add_argument(
        "--ax-projected-replay",
        action="store_true",
        help="Enable AX's opt-in Qwen gated-delta projected-prefix rollback.",
    )
    parser.add_argument(
        "--ax-relaxed-verify",
        action="store_true",
        help=(
            "Use target-verified batched Qwen arithmetic that is not guaranteed "
            "bit-identical to singleton direct decode."
        ),
    )
    parser.add_argument(
        "--prompt-targets",
        type=lambda value: parse_csv_ints(value, field="prompt targets"),
        default=DEFAULT_PROMPT_TARGETS,
    )
    parser.add_argument(
        "--quality-dataset",
        type=Path,
        help=(
            "Optional JSONL suite containing both general and agent-coding tasks with "
            "AXQuant-style objective checks."
        ),
    )
    parser.add_argument("--quality-seed", type=int, default=DEFAULT_QUALITY_SEED)
    parser.add_argument("--generation-tokens", type=int, default=DEFAULT_GENERATION_TOKENS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--port", type=int, default=31910)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.generation_tokens <= 0 or args.repetitions <= 0:
        parser.error("--generation-tokens and --repetitions must be positive")
    if args.quality_seed < 0:
        parser.error("--quality-seed must be non-negative")
    if args.quality_dataset is not None and not args.quality_dataset.is_file():
        parser.error("--quality-dataset must be a JSONL file")
    if args.ax_force_mtp and not args.ax_speculative:
        parser.error("--ax-force-mtp requires --ax-speculative")
    if args.ax_projected_replay and not args.ax_force_mtp:
        parser.error("AX MTP optimization flags require --ax-force-mtp")
    if args.ax_relaxed_verify and not args.ax_projected_replay:
        parser.error("--ax-relaxed-verify requires --ax-projected-replay")
    if args.mlxcel_draft_model is not None:
        if args.mlxcel_server is None:
            parser.error("--mlxcel-draft-model requires --mlxcel-server")
        if not args.mlxcel_draft_model.is_dir():
            parser.error("--mlxcel-draft-model must be a model directory")
        if not (args.mlxcel_draft_model / "config.json").is_file():
            parser.error("--mlxcel-draft-model directory has no config.json")
    if args.mlxcel_relaxed_verify and args.mlxcel_draft_model is None:
        parser.error("--mlxcel-relaxed-verify requires --mlxcel-draft-model")
    if args.mtplx_force_unverified and args.mtplx_server is None:
        parser.error("--mtplx-force-unverified requires --mtplx-server")
    if args.omlx_mtp and args.omlx_server is None:
        parser.error("--omlx-mtp requires --omlx-server")
    if args.warmup_tokens < 0 or args.cooldown < 0:
        parser.error("--warmup-tokens and --cooldown must be non-negative")
    if args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error("--timeout and --startup-timeout must be positive")
    if not 0 < args.port < 65536:
        parser.error("--port must be in 1..65535")
    if len(args.prompt_targets) != len(set(args.prompt_targets)):
        parser.error("--prompt-targets values must be unique")
    labels = [model.label for model in args.model]
    if len(labels) != len(set(labels)):
        parser.error("--model labels must be unique")
    configured_engines = {
        "ax-engine",
        *(key for key in ("mlxcel", "omlx", "mtplx") if getattr(args, f"{key}_server")),
    }
    override_keys = [(override.engine, override.label) for override in args.engine_model]
    if len(override_keys) != len(set(override_keys)):
        parser.error("--engine-model values must be unique per engine and label")
    for override in args.engine_model:
        if override.engine not in configured_engines:
            parser.error(f"--engine-model names an unconfigured engine: {override.engine}")
        if override.label not in labels:
            parser.error(f"--engine-model names an unknown model label: {override.label}")

    artifact = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({artifact['status']})")
    return 0 if artifact["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
