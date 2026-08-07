#!/usr/bin/env python3
"""Run a low-rate, no-restart AX Engine MLX endurance workload.

The runner owns one AX Engine server process for the entire test.  It sends a
deterministic, single-client streaming request every fixed interval, persists
each outcome as JSONL, and writes an atomic checkpoint every four hours by
default.  It deliberately does not automatically restart a failed server:
that would hide the availability failure this test is intended to detect.

This is an endurance/soak test, not a maximum-throughput benchmark.  The
default workload is intentionally light: one in-flight request, 30-second
cadence, short prompts most of the time, and bounded medium/long prompt
coverage to exercise prefill and allocator cleanup over a long server lifetime.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bench_ax_serving as serving_bench

SCHEMA_VERSION = "ax.axq_endurance_soak.v1"
CHECKPOINT_SCHEMA_VERSION = "ax.axq_endurance_checkpoint.v1"
DEFAULT_DURATION_HOURS = 72.0
DEFAULT_REPORT_INTERVAL_HOURS = 4.0
DEFAULT_REQUEST_INTERVAL_S = 30.0
DEFAULT_REQUEST_TIMEOUT_S = 180.0
DEFAULT_STARTUP_TIMEOUT_S = 1_200.0
DEFAULT_WARMUP_REQUESTS = 2
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_MAX_RSS_GROWTH_MIB = 4_096.0

SENSITIVE_HARDWARE_PREFIXES = (
    "Serial Number",
    "Hardware UUID",
    "Provisioning UDID",
    "Activation Lock Status",
)
INTERESTING_METRICS = (
    "ax_engine_jobs_in_flight",
    "ax_engine_http_requests_in_flight",
    "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued",
    "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
    "ax_engine_generation_saturated_commands_total",
    "ax_engine_generation_stream_backlog_overflows_total",
    "ax_engine_memory_pressure_level",
)
PROMPT_WORDS = (
    "adapter",
    "allocation",
    "analysis",
    "attention",
    "cache",
    "checkpoint",
    "context",
    "decode",
    "deterministic",
    "endurance",
    "generation",
    "health",
    "inference",
    "latency",
    "memory",
    "model",
    "monitor",
    "prefill",
    "request",
    "runtime",
    "scheduler",
    "streaming",
    "telemetry",
    "throughput",
    "verification",
    "workload",
)


@dataclass(frozen=True)
class WorkloadShape:
    """One bounded request shape in the deterministic endurance mix."""

    name: str
    nominal_input_words: int
    max_output_tokens: int


WORKLOAD_SHAPES = {
    "short": WorkloadShape("short", nominal_input_words=96, max_output_tokens=48),
    "medium": WorkloadShape("medium", nominal_input_words=512, max_output_tokens=64),
    "long": WorkloadShape("long", nominal_input_words=2_048, max_output_tokens=64),
}


@dataclass
class RunState:
    """Mutable state kept compact enough for a multi-day run."""

    started_wall: str
    started_monotonic: float
    server_pid: int
    baseline_rss_kb: int | None = None
    max_rss_kb: int | None = None
    requests_attempted: int = 0
    requests_ok: int = 0
    requests_failed: int = 0
    health_failures: int = 0
    consecutive_request_failures: int = 0
    rss_growth_alerts: int = 0
    last_error: str | None = None


def utc_now() -> str:
    """Return a stable, timezone-explicit timestamp for artifacts."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def utc_run_id() -> str:
    """Return a filesystem-safe default run id."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def positive_float(value: str) -> float:
    """Parse a strictly positive CLI float."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative CLI integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_int(value: str) -> int:
    """Parse a strictly positive CLI integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def sha256_file(path: Path) -> str:
    """Hash a small identity file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model_dir: Path) -> dict[str, Any]:
    """Return cheap, auditable identity data for a local AX model package."""
    required = ("config.json", "model-manifest.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    weights = sorted(model_dir.glob("*.safetensors"))
    if missing or not weights:
        missing_text = ", ".join([*missing, "*.safetensors" if not weights else ""])
        raise FileNotFoundError(f"AX model package is incomplete at {model_dir}: {missing_text}")
    identity_files = {
        name: sha256_file(model_dir / name)
        for name in ("config.json", "model-manifest.json", "tokenizer.json")
        if (model_dir / name).is_file()
    }
    return {
        "path": str(model_dir),
        "identity_files_sha256": identity_files,
        "safetensors_files": len(weights),
        "safetensors_bytes": sum(path.stat().st_size for path in weights),
    }


def command_output(*command: str) -> str:
    """Collect non-sensitive diagnostic command output without failing a run."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    return (result.stdout or result.stderr).strip()


def sanitized_hardware_profile() -> str:
    """Collect hardware context while excluding durable machine identifiers."""
    profile = command_output("system_profiler", "SPHardwareDataType")
    return "\n".join(
        line
        for line in profile.splitlines()
        if not line.strip().startswith(SENSITIVE_HARDWARE_PREFIXES)
    )


def runtime_metadata(server_path: Path) -> dict[str, Any]:
    """Capture versions relevant to a reproducible endurance result."""
    packages: dict[str, str | None] = {}
    for package in ("mlx", "mlx-lm", "huggingface-hub"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": command_output("sw_vers"),
        "hardware": sanitized_hardware_profile(),
        "power": command_output("pmset", "-g", "batt"),
        "mlx_packages": packages,
        "server_path": str(server_path),
        "server_sha256": sha256_file(server_path),
        "server_version": command_output(str(server_path), "--version"),
    }


def build_server_command(args: argparse.Namespace) -> list[str]:
    """Build the one-server command used for the full test lifetime."""
    return [
        str(args.server),
        "--model-id",
        args.model_id,
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(args.model_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-concurrent-requests",
        "1",
        "--max-concurrent-requests-per-model",
        "1",
        *args.server_extra_arg,
    ]


def request_json(url: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
    """Fetch a JSON endpoint and require an object response."""
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"expected JSON object from {url}")
        return response.status, payload


def health_check(base_url: str, timeout_s: float) -> dict[str, Any]:
    """Perform a non-invasive health check and retain failures as evidence."""
    started = time.perf_counter()
    try:
        status, payload = request_json(f"{base_url}/health", timeout_s)
        reported_status = payload.get("status")
        ok = 200 <= status < 300 and reported_status in {None, "ok"}
        return {
            "ok": ok,
            "http_status": status,
            "reported_status": reported_status,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "error": None if ok else f"health returned status={status}, body_status={reported_status}",
        }
    except Exception as error:  # noqa: BLE001 - preserve endpoint errors in artifacts.
        return {
            "ok": False,
            "http_status": None,
            "reported_status": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "error": str(error),
        }


def wait_for_server(
    process: subprocess.Popen[bytes],
    *,
    base_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Wait for the owned server to report ready, or preserve its early exit."""
    deadline = time.monotonic() + timeout_s
    last_health: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"server exited before readiness with code {exit_code}")
        last_health = health_check(base_url, timeout_s=min(10.0, timeout_s))
        if last_health["ok"]:
            return last_health
        time.sleep(2.0)
    detail = last_health.get("error") if last_health else "no health response"
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s: {detail}")


def stop_server(process: subprocess.Popen[bytes], timeout_s: float = 60.0) -> dict[str, Any]:
    """Stop the owned server gracefully, escalating only after a bounded wait."""
    forced = False
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            forced = True
            process.kill()
            process.wait(timeout=30.0)
    return {"exit_code": process.returncode, "forced": forced}


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Parse unlabeled Prometheus samples while ignoring comments and labels."""
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) != 2 or not fields[0] or fields[0].startswith("#"):
            continue
        if "{" in fields[0]:
            continue
        try:
            metrics[fields[0]] = float(fields[1])
        except ValueError:
            continue
    return metrics


def collect_metrics(base_url: str, timeout_s: float) -> dict[str, Any]:
    """Read a compact subset of server metrics without making it a hard gate."""
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout_s) as response:
            values = parse_prometheus_metrics(response.read().decode("utf-8"))
        return {
            "ok": True,
            "values": {name: values[name] for name in INTERESTING_METRICS if name in values},
        }
    except Exception as error:  # noqa: BLE001 - metrics must not mask workload evidence.
        return {"ok": False, "error": str(error), "values": {}}


def process_snapshot(pid: int) -> dict[str, Any]:
    """Collect RSS/CPU from the exact server PID, not a broad process match."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,rss=,%cpu=,etime="],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = result.stdout.strip().split(maxsplit=3)
    if len(fields) != 4:
        return {"alive": False, "pid": pid}
    try:
        return {
            "alive": True,
            "pid": int(fields[0]),
            "rss_kb": int(fields[1]),
            "cpu_percent": float(fields[2]),
            "elapsed": fields[3],
        }
    except ValueError:
        return {"alive": False, "pid": pid, "parse_error": result.stdout.strip()}


def collect_host_snapshot(output_dir: Path) -> dict[str, Any]:
    """Collect lightweight host context at four-hour checkpoint boundaries."""
    disk = shutil.disk_usage(output_dir)
    return {
        "load_average": list(os.getloadavg()),
        "swap_usage": command_output("sysctl", "-n", "vm.swapusage"),
        "memory_pressure": command_output("memory_pressure"),
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "power": command_output("pmset", "-g", "batt"),
    }


def select_shape(request_index: int) -> WorkloadShape:
    """Return the fixed 21:2:1 short/medium/long request mix."""
    if request_index % 24 == 0:
        return WORKLOAD_SHAPES["long"]
    if request_index % 8 == 0:
        return WORKLOAD_SHAPES["medium"]
    return WORKLOAD_SHAPES["short"]


def deterministic_prompt(shape: WorkloadShape, request_index: int) -> str:
    """Create a unique, tokenizer-exercising raw-text prompt without user data."""
    offset = request_index % len(PROMPT_WORDS)
    words = [PROMPT_WORDS[(offset + index) % len(PROMPT_WORDS)] for index in range(shape.nominal_input_words)]
    header = (
        f"Endurance request {request_index}; this is a synthetic runtime-health probe. "
        "Return a concise acknowledgement after processing the following sequence.\n"
    )
    return header + " ".join(words)


def make_prompt_item(shape: WorkloadShape, request_index: int) -> serving_bench.PromptItem:
    """Adapt a shape into the existing serving benchmark's request type."""
    return serving_bench.PromptItem(
        id=f"{shape.name}-{request_index:08d}",
        category=shape.name,
        input_text=deterministic_prompt(shape, request_index),
        input_tokens=None,
        input_tokens_count=shape.nominal_input_words,
        max_output_tokens=shape.max_output_tokens,
        metadata={"nominal_input_words": shape.nominal_input_words},
    )


def run_stream_request(
    *,
    prompt: serving_bench.PromptItem,
    model_id: str,
    base_url: str,
    timeout_s: float,
    stream_func: Callable[..., Any] = serving_bench.http_sse_events,
) -> dict[str, Any]:
    """Run one native streaming request with fixed-length greedy sampling."""
    started = time.perf_counter()
    payload = serving_bench.build_payload(
        prompt,
        model_id=model_id,
        input_kind="text",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=0,
    )
    sampling = payload.get("sampling")
    if isinstance(sampling, dict):
        # This is an AX-native control used by the existing fault soak. It keeps
        # each request bounded and prevents accidental early-EOS-only cycles.
        sampling["ignore_eos"] = True
    try:
        events = list(stream_func(f"{base_url}/v1/generate/stream", payload, timeout_s))
        return serving_bench.observe_stream(
            events,
            prompt=prompt,
            scheduled_at_s=0.0,
            started_at_s=0.0,
            completed_at_s=time.perf_counter() - started,
        )
    except Exception as error:  # noqa: BLE001 - workload failures are first-class evidence.
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "prompt_id": prompt.id,
            "category": prompt.category,
            "phase": "endurance",
            "status": None,
            "ok": False,
            "error": str(error),
            "e2e_latency_ms": elapsed_ms,
            "ttft_ms": None,
            "client_tpot_ms": None,
            "output_tokens": None,
            "route_decisions": {},
        }


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linear-interpolated percentile for checkpoint windows."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize_values(values: list[float]) -> dict[str, float] | None:
    """Produce small, high-signal latency summaries for one reporting window."""
    if not values:
        return None
    return {
        "count": float(len(values)),
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50) or 0.0,
        "p95": percentile(values, 0.95) or 0.0,
        "p99": percentile(values, 0.99) or 0.0,
        "max": max(values),
    }


def summarize_window(records: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    """Summarize a bounded in-memory checkpoint window only."""
    successes = [record for record in records if record.get("request", {}).get("ok")]
    failures = len(records) - len(successes)
    ttft = [
        float(record["request"]["ttft_ms"])
        for record in successes
        if isinstance(record["request"].get("ttft_ms"), (int, float))
    ]
    tpot = [
        float(record["request"]["client_tpot_ms"])
        for record in successes
        if isinstance(record["request"].get("client_tpot_ms"), (int, float))
    ]
    e2e = [
        float(record["request"]["e2e_latency_ms"])
        for record in successes
        if isinstance(record["request"].get("e2e_latency_ms"), (int, float))
    ]
    output_tokens = [
        float(record["request"]["output_tokens"])
        for record in successes
        if isinstance(record["request"].get("output_tokens"), (int, float))
    ]
    route_decisions = serving_bench.summarize_route_decisions(
        [record["request"] for record in successes]
    )
    window_duration = max(elapsed_s, 0.001)
    return {
        "requests": len(records),
        "successful_requests": len(successes),
        "failed_requests": failures,
        "success_ratio": len(successes) / len(records) if records else 0.0,
        "request_throughput_rps": len(successes) / window_duration,
        "output_token_throughput_tok_s": sum(output_tokens) / window_duration,
        "ttft_ms": summarize_values(ttft),
        "client_tpot_ms": summarize_values(tpot),
        "e2e_latency_ms": summarize_values(e2e),
        "output_tokens": summarize_values(output_tokens),
        "route_decisions": route_decisions,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Persist an inspectable checkpoint without leaving a partial JSON file."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append and flush one event so a sudden host failure retains evidence."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_summary(
    *,
    state: RunState,
    status: str,
    elapsed_s: float,
    target_duration_s: float,
    latest_window: dict[str, Any],
    latest_server: dict[str, Any],
    latest_host: dict[str, Any],
    latest_metrics: dict[str, Any],
    alerts: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Build the current durable status used by operators and four-hour reports."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "updated_at": utc_now(),
        "elapsed_seconds": elapsed_s,
        "target_duration_seconds": target_duration_s,
        "target_end_at": (
            dt.datetime.fromisoformat(state.started_wall)
            + dt.timedelta(seconds=target_duration_s)
        ).isoformat(),
        "server": latest_server,
        "requests": {
            "attempted": state.requests_attempted,
            "successful": state.requests_ok,
            "failed": state.requests_failed,
            "health_failures": state.health_failures,
            "consecutive_request_failures": state.consecutive_request_failures,
        },
        "memory": {
            "baseline_rss_kb": state.baseline_rss_kb,
            "max_rss_kb": state.max_rss_kb,
            "rss_growth_alerts": state.rss_growth_alerts,
        },
        "latest_window": latest_window,
        "host": latest_host,
        "metrics": latest_metrics,
        "alerts": alerts[-20:],
        "last_error": state.last_error,
        "artifacts": {
            "output_dir": str(output_dir),
            "events": str(output_dir / "events.jsonl"),
            "server_log": str(output_dir / "server.log"),
            "checkpoints_dir": str(output_dir / "checkpoints"),
        },
    }


def write_checkpoint(
    *,
    output_dir: Path,
    reason: str,
    state: RunState,
    status: str,
    elapsed_s: float,
    window_elapsed_s: float,
    target_duration_s: float,
    records: list[dict[str, Any]],
    latest_server: dict[str, Any],
    latest_host: dict[str, Any],
    latest_metrics: dict[str, Any],
    alerts: list[str],
) -> dict[str, Any]:
    """Write one immutable checkpoint plus the mutable summary.json view."""
    window = summarize_window(records, elapsed_s=max(window_elapsed_s, 0.001))
    summary = run_summary(
        state=state,
        status=status,
        elapsed_s=elapsed_s,
        target_duration_s=target_duration_s,
        latest_window=window,
        latest_server=latest_server,
        latest_host=latest_host,
        latest_metrics=latest_metrics,
        alerts=alerts,
        output_dir=output_dir,
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "reason": reason,
        "summary": summary,
    }
    checkpoint_name = f"{utc_now().replace(':', '').replace('+00:00', 'Z')}-{reason}.json"
    write_json_atomic(output_dir / "checkpoints" / checkpoint_name, checkpoint)
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def prepare_output_dir(path: Path) -> None:
    """Refuse a non-empty output path so long-run evidence is never overwritten."""
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    """Validate exact local inputs before spawning a multi-day process."""
    args.server = args.server.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.server.is_file() or not os.access(args.server, os.X_OK):
        raise FileNotFoundError(f"server binary is not executable: {args.server}")
    model_identity(args.model_dir)
    if not 0 < args.port < 65_536:
        raise ValueError("--port must be in 1..65535")
    prepare_output_dir(args.output_dir)


def run_endurance(args: argparse.Namespace) -> int:
    """Run the soak while converting SIGTERM into a checkpointed interrupt."""
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        return _run_endurance(args)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def _run_endurance(args: argparse.Namespace) -> int:
    """Launch one server and keep it exercised until duration or a real failure."""
    validate_args(args)
    base_url = f"http://{args.host}:{args.port}"
    target_duration_s = args.duration_hours * 3_600.0
    report_interval_s = args.report_interval_hours * 3_600.0
    output_dir = args.output_dir
    server_command = build_server_command(args)
    server_log = output_dir / "server.log"
    events_path = output_dir / "events.jsonl"
    process: subprocess.Popen[bytes] | None = None
    state: RunState | None = None
    status = "failed"
    failure: str | None = None
    alerts: list[str] = []
    records: list[dict[str, Any]] = []
    latest_server: dict[str, Any] = {}
    latest_host: dict[str, Any] = {}
    latest_metrics: dict[str, Any] = {}
    last_checkpoint_monotonic: float | None = None

    try:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "methodology": {
                "scope": "72-hour low-rate no-restart AX Engine MLX endurance soak",
                "server_lifetime": "one owned process; automatic restart is forbidden",
                "concurrency": 1,
                "cadence_seconds": args.request_interval_seconds,
                "workload_mix": "21 short : 2 medium : 1 long synthetic raw-text streams",
                "stream_validation": "HTTP success, terminal response, and non-empty output",
                "reporting": "atomic summary plus immutable checkpoint every report interval",
            },
            "target": {
                "model_id": args.model_id,
                "model": model_identity(args.model_dir),
                "server_command": server_command,
                "base_url": base_url,
            },
            "limits": {
                "duration_hours": args.duration_hours,
                "report_interval_hours": args.report_interval_hours,
                "request_timeout_seconds": args.request_timeout_seconds,
                "max_consecutive_request_failures": args.max_consecutive_request_failures,
                "rss_growth_alert_mib": args.max_rss_growth_mib,
            },
            "runtime": runtime_metadata(args.server),
        }
        write_json_atomic(output_dir / "manifest.json", manifest)

        with server_log.open("wb") as log:
            process = subprocess.Popen(
                server_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state = RunState(
            started_wall=utc_now(),
            started_monotonic=time.monotonic(),
            server_pid=process.pid,
        )
        readiness = wait_for_server(
            process,
            base_url=base_url,
            timeout_s=args.startup_timeout_seconds,
        )
        append_jsonl(
            events_path,
            {"timestamp": utc_now(), "kind": "readiness", "health": readiness, "pid": process.pid},
        )

        for warmup_index in range(args.warmup_requests):
            shape = select_shape(warmup_index + 1)
            observation = run_stream_request(
                prompt=make_prompt_item(shape, warmup_index + 1),
                model_id=args.model_id,
                base_url=base_url,
                timeout_s=args.request_timeout_seconds,
            )
            observation["phase"] = "warmup"
            append_jsonl(
                events_path,
                {
                    "timestamp": utc_now(),
                    "kind": "warmup_request",
                    "request_index": warmup_index + 1,
                    "request": observation,
                },
            )
            if not observation.get("ok"):
                raise RuntimeError(f"warmup request {warmup_index + 1} failed: {observation.get('error')}")

        # Warmups are setup, not part of the promised endurance duration or
        # fixed-rate schedule. Start the measured clock only after they finish
        # so a slow first model load cannot make the runner catch up by issuing
        # several measured requests back-to-back.
        started_monotonic = time.monotonic()
        state.started_wall = utc_now()
        state.started_monotonic = started_monotonic
        latest_server = process_snapshot(process.pid)
        state.baseline_rss_kb = latest_server.get("rss_kb")
        state.max_rss_kb = state.baseline_rss_kb
        latest_host = collect_host_snapshot(output_dir)
        latest_metrics = collect_metrics(base_url, timeout_s=10.0)
        next_checkpoint = started_monotonic + report_interval_s
        last_checkpoint_monotonic = started_monotonic
        write_checkpoint(
            output_dir=output_dir,
            reason="started",
            state=state,
            status="running",
            elapsed_s=0.0,
            window_elapsed_s=0.0,
            target_duration_s=target_duration_s,
            records=records,
            latest_server=latest_server,
            latest_host=latest_host,
            latest_metrics=latest_metrics,
            alerts=alerts,
        )

        # Warmups use indices 1..N, so measured requests continue at N+1
        # instead of replaying the same prompt ids and accidentally exercising
        # retained-prefix reuse.
        request_index = args.warmup_requests
        cadence_index = 0
        while True:
            now = time.monotonic()
            elapsed_s = now - started_monotonic
            if elapsed_s >= target_duration_s:
                status = "completed"
                break
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(f"server exited during endurance run with code {exit_code}")

            request_index += 1
            cadence_index += 1
            shape = select_shape(request_index)
            health = health_check(base_url, timeout_s=10.0)
            if not health["ok"]:
                state.health_failures += 1
                state.last_error = f"health check failed: {health['error']}"
                alerts.append(state.last_error)
            observation = run_stream_request(
                prompt=make_prompt_item(shape, request_index),
                model_id=args.model_id,
                base_url=base_url,
                timeout_s=args.request_timeout_seconds,
            )
            observation["phase"] = "endurance"
            state.requests_attempted += 1
            if observation.get("ok"):
                state.requests_ok += 1
                state.consecutive_request_failures = 0
            else:
                state.requests_failed += 1
                state.consecutive_request_failures += 1
                state.last_error = str(observation.get("error") or "stream request failed")
                alerts.append(state.last_error)

            latest_server = process_snapshot(process.pid)
            rss_kb = latest_server.get("rss_kb")
            if isinstance(rss_kb, int):
                state.max_rss_kb = max(state.max_rss_kb or rss_kb, rss_kb)
                baseline = state.baseline_rss_kb
                growth_limit_kb = int(args.max_rss_growth_mib * 1024.0)
                if baseline is not None and rss_kb - baseline > growth_limit_kb:
                    state.rss_growth_alerts += 1
                    alerts.append(
                        f"server RSS grew {(rss_kb - baseline) / 1024.0:.1f} MiB above warm baseline"
                    )
            if not latest_server.get("alive"):
                raise RuntimeError("server PID disappeared during an endurance request")

            record = {
                "timestamp": utc_now(),
                "kind": "endurance_request",
                "request_index": request_index,
                "shape": shape.name,
                "health": health,
                "request": observation,
                "server": latest_server,
            }
            records.append(record)
            append_jsonl(events_path, record)
            if state.consecutive_request_failures >= args.max_consecutive_request_failures:
                raise RuntimeError(
                    "consecutive request failure limit reached: "
                    f"{state.consecutive_request_failures}/{args.max_consecutive_request_failures}"
                )

            now = time.monotonic()
            if now >= next_checkpoint:
                latest_host = collect_host_snapshot(output_dir)
                latest_metrics = collect_metrics(base_url, timeout_s=10.0)
                write_checkpoint(
                    output_dir=output_dir,
                    reason="periodic",
                    state=state,
                    status="running",
                    elapsed_s=now - started_monotonic,
                    window_elapsed_s=now - last_checkpoint_monotonic,
                    target_duration_s=target_duration_s,
                    records=records,
                    latest_server=latest_server,
                    latest_host=latest_host,
                    latest_metrics=latest_metrics,
                    alerts=alerts,
                )
                records.clear()
                last_checkpoint_monotonic = now
                while next_checkpoint <= now:
                    next_checkpoint += report_interval_s

            scheduled_next = (
                started_monotonic
                + cadence_index * args.request_interval_seconds
            )
            delay = max(0.0, scheduled_next - time.monotonic())
            if delay > 0.0:
                time.sleep(delay)

    except KeyboardInterrupt:
        status = "interrupted"
        failure = "runner received an interrupt signal"
    except Exception as error:  # noqa: BLE001 - a failed soak must retain its exact cause.
        status = "failed"
        failure = str(error)
    finally:
        if state is not None:
            if failure:
                state.last_error = failure
                alerts.append(failure)
            elapsed_s = time.monotonic() - state.started_monotonic
            window_elapsed_s = (
                time.monotonic() - last_checkpoint_monotonic
                if last_checkpoint_monotonic is not None
                else elapsed_s
            )
            if process is not None:
                latest_server = process_snapshot(process.pid)
            if not latest_host:
                latest_host = collect_host_snapshot(output_dir)
            if not latest_metrics:
                latest_metrics = collect_metrics(base_url, timeout_s=10.0)
            append_jsonl(
                events_path,
                {
                    "timestamp": utc_now(),
                    "kind": "run_terminal",
                    "status": status,
                    "error": failure,
                },
            )
            write_checkpoint(
                output_dir=output_dir,
                reason="final",
                state=state,
                status=status,
                elapsed_s=elapsed_s,
                window_elapsed_s=window_elapsed_s,
                target_duration_s=target_duration_s,
                records=records,
                latest_server=latest_server,
                latest_host=latest_host,
                latest_metrics=latest_metrics,
                alerts=alerts,
            )
        if process is not None:
            stop_result = stop_server(process)
            append_jsonl(
                events_path,
                {"timestamp": utc_now(), "kind": "server_stopped", "result": stop_result},
            )

    if failure:
        print(f"AXQ endurance run {status}: {failure}", file=sys.stderr)
    else:
        print(f"AXQ endurance run {status}: {output_dir}")
    return 0 if status == "completed" else 130 if status == "interrupted" else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit, conservative endurance runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen3.6-27b-axq-6bit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31418)
    parser.add_argument("--duration-hours", type=positive_float, default=DEFAULT_DURATION_HOURS)
    parser.add_argument(
        "--report-interval-hours",
        type=positive_float,
        default=DEFAULT_REPORT_INTERVAL_HOURS,
    )
    parser.add_argument(
        "--request-interval-seconds",
        type=positive_float,
        default=DEFAULT_REQUEST_INTERVAL_S,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=positive_float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=positive_float,
        default=DEFAULT_STARTUP_TIMEOUT_S,
    )
    parser.add_argument("--warmup-requests", type=non_negative_int, default=DEFAULT_WARMUP_REQUESTS)
    parser.add_argument(
        "--max-consecutive-request-failures",
        type=positive_int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
    )
    parser.add_argument(
        "--max-rss-growth-mib",
        type=positive_float,
        default=DEFAULT_MAX_RSS_GROWTH_MIB,
        help="Alert threshold versus the warm baseline; it does not restart the server.",
    )
    parser.add_argument(
        "--server-extra-arg",
        action="append",
        default=[],
        help="Repeat to pass a safe additional ax-engine-server argument.",
    )
    return parser


def main_with_args_for_test(argv: list[str]) -> int:
    """Entrypoint retained separately so unit tests can parse realistic args."""
    return run_endurance(build_parser().parse_args(argv))


def main() -> None:
    """Run the command-line entrypoint."""
    raise SystemExit(main_with_args_for_test(sys.argv[1:]))


if __name__ == "__main__":
    main()
