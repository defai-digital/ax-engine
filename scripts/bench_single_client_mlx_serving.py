#!/usr/bin/env python3
"""Benchmark AX Engine against a peer MLX server through streaming OpenAI chat."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.vs_mlxcel.single_client.v2"
DEFAULT_PROMPT_TARGETS = (512, 2048)
DEFAULT_GENERATION_TOKENS = 256
DEFAULT_REPETITIONS = 3
DEFAULT_COOLDOWN_S = 15.0
DEFAULT_TIMEOUT_S = 900.0

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        return [
            str(engine.binary),
            "-m",
            str(model.path),
            "--alias",
            "local",
            *common,
            "--parallel",
            "1",
            "--max-batch-prefill",
            "1",
            "--ctx-size",
            "65536",
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
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": generation_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": 0,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
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
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": first_chunk_s,
        "prefill_tps": prompt_tokens / first_chunk_s if first_chunk_s > 0 else None,
        "decode_tps": decode_tps,
        "decode_window_s": decode_s,
        "e2e_s": e2e_s,
        "visible_chars": visible_chars,
        "finish_reason": finish_reason,
    }


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


def engine_order(model_index: int, rep: int) -> tuple[str, str]:
    ax_first = (model_index + rep) % 2 == 0
    return ("ax-engine", "mlxcel") if ax_first else ("mlxcel", "ax-engine")


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
    engines = {
        "ax-engine": EngineSpec("ax-engine", args.ax_server.resolve()),
        "mlxcel": EngineSpec("mlxcel", args.mlxcel_server.resolve()),
    }
    for engine in engines.values():
        if not engine.binary.is_file():
            raise FileNotFoundError(f"server binary does not exist: {engine.binary}")

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
    output = args.output.resolve()
    log_dir = (args.log_dir or output.parent / "logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    measurements: list[dict[str, Any]] = []
    process_audit: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started_at = utc_now()
    performance_conditions_start = collect_host()

    for model_index, model in enumerate(args.model):
        for rep in range(args.repetitions):
            for engine_key in engine_order(model_index, rep):
                engine = engines[engine_key]
                command = engine_command(engine, model=model, port=args.port)
                log_path = log_dir / f"{model_index:02d}-{model.label}-{rep}-{engine_key}.log"
                environment = os.environ.copy()
                if engine_key == "ax-engine":
                    environment.update(
                        {
                            "AX_NO_SPEC": "1",
                            "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY": "1",
                        }
                    )
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
                            )
                        for prompt_target in args.prompt_targets:
                            seed, prompt = deterministic_prompt(prompt_target, rep)
                            measured_at = utc_now()
                            try:
                                observed = run_request(
                                    port=args.port,
                                    prompt=prompt,
                                    seed=seed,
                                    generation_tokens=args.generation_tokens,
                                    timeout_s=args.timeout,
                                )
                                require_complete_generation(observed, args.generation_tokens)
                            except Exception as exc:  # noqa: BLE001 - preserve failed evidence.
                                errors.append(
                                    {
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
                                    "model_dir": str(model.path),
                                    "prompt_target": prompt_target,
                                    "rep": rep,
                                    "seed": seed,
                                    "measured_at": measured_at,
                                    **observed,
                                }
                            )
                    except Exception as exc:  # noqa: BLE001 - preserve failed process evidence.
                        errors.append(
                            {
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
    status = "complete" if not errors and len(measurements) == expected else "incomplete"
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
            "ax_engine_version": binary_version(engines["ax-engine"].binary),
            "mlxcel_version": binary_version(engines["mlxcel"].binary),
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
                "engine_order": "balanced by model index and repetition",
                "prompt_policy": "deterministic nominal word count; same text and seed per engine",
            },
        },
        "host": {
            "start": performance_conditions_start,
            "end": collect_host(),
        },
        "binaries": binary_identities,
        "runner": runner,
        "models": model_identities,
        "expected_measurements": expected,
        "measurements": measurements,
        "results": summarize_measurements(measurements),
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
    parser.add_argument("--ax-server", type=Path, required=True)
    parser.add_argument("--mlxcel-server", type=Path, required=True)
    parser.add_argument(
        "--prompt-targets",
        type=lambda value: parse_csv_ints(value, field="prompt targets"),
        default=DEFAULT_PROMPT_TARGETS,
    )
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

    artifact = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({artifact['status']})")
    return 0 if artifact["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
