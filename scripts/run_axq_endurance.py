#!/usr/bin/env python3
"""Run a light, no-restart AX Engine MLX endurance/soak workload.

This runner is designed for evidence, not peak throughput.  It owns one AX
Engine server process for the entire run, sends one bounded native streaming
request at a fixed cadence, and never restarts the server after a failure.  It
writes raw request/resource evidence continuously and an atomic, human-readable
status report every four hours by default.

The default workload deliberately uses one in-flight request.  It combines
unique prompts (allocator/KV retirement coverage) with a small shared-prefix
slice (prefix-cache coverage), while leaving headroom so that a queueing stress
test cannot mask an endurance defect.
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
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bench_ax_serving as serving_bench

SCHEMA_VERSION = "ax.axq_endurance_soak.v4"
CHECKPOINT_SCHEMA_VERSION = "ax.axq_endurance_checkpoint.v4"
DEFAULT_DURATION_HOURS = 72.0
DEFAULT_REPORT_INTERVAL_HOURS = 4.0
DEFAULT_BASELINE_HOURS = 4.0
DEFAULT_REQUEST_INTERVAL_S = 60.0
DEFAULT_RESOURCE_INTERVAL_S = 60.0
DEFAULT_REQUEST_TIMEOUT_S = 300.0
DEFAULT_STARTUP_TIMEOUT_S = 1_200.0
DEFAULT_DRAIN_TIMEOUT_S = 10.0
DEFAULT_MAX_BATCH_TOKENS = 2_048
DEFAULT_BLOCK_SIZE_TOKENS = 16
DEFAULT_TOTAL_BLOCKS = 1_024
DEFAULT_MAX_CAPACITY_FILL_REQUESTS = 4
# The measured baseline must begin after each request class has caused its
# normal first-use allocations.  One complete interleaved cycle includes the
# long prefill and prefix-cache probes as well as the short steady-state
# traffic; warming only the first few short prompts would make the first
# baseline window absorb those expected allocation steps.
DEFAULT_WARMUP_REQUESTS = 20
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_MAX_ERROR_RATE = 0.001
DEFAULT_MAX_TTFT_P95_RATIO = 1.50
DEFAULT_MIN_DECODE_P05_RATIO = 0.75
DEFAULT_MIN_PREFILL_P05_RATIO = 0.75
DEFAULT_MIN_PERFORMANCE_SAMPLES = 8
DEFAULT_MEMORY_GROWTH_MIB = 4_096.0
# A 4 GiB retained increase spread across the full 68-hour post-baseline
# window is only about 60 MiB/hour.  The slope guard therefore cannot be as
# high as the short-baseline settling guard or a slow, material leak would be
# silently classified as normal allocator noise.
DEFAULT_MEMORY_SLOPE_MIB_PER_HOUR = 64.0
DEFAULT_EARLY_MEMORY_GROWTH_MIB = 256.0
DEFAULT_EARLY_MEMORY_SLOPE_MIB_PER_HOUR = 32.0
DEFAULT_EARLY_MEMORY_POSITIVE_STEP_RATIO = 0.70
DEFAULT_BASELINE_STABILITY_GROWTH_MIB = 1_024.0
DEFAULT_BASELINE_STABILITY_SLOPE_MIB_PER_HOUR = 256.0
DEFAULT_MAX_SWAP_GROWTH_MIB = 512.0
DEFAULT_MAX_QUIESCENT_KV_LOGICAL_MIB = 1_024.0
DEFAULT_MAX_SAMPLING_GAP_S = 140.0
# A healthy single-client request may take several scheduler prefill turns;
# this deliberately roomy limit detects the pathological one-token turn loop
# without treating ordinary chunking as a failure.  It is evaluated only by
# fresh, post-warm-up capacity probes, where a cache hit is not expected.
DEFAULT_MAX_PREFILL_STEPS_PER_1K_TOKENS = 64.0

MEBIBYTE = 1024 * 1024
SENSITIVE_HARDWARE_PREFIXES = (
    "Serial Number",
    "Hardware UUID",
    "Provisioning UDID",
    "Activation Lock Status",
)

# These lifecycle gauges must all be zero once a completed response has drained.
# A missing gauge makes the drain verdict inconclusive, rather than silently
# claiming that KV/cache cleanup was verified.
LIFECYCLE_METRICS = (
    "ax_engine_jobs_in_flight",
    "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued",
    "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
)
SERVER_METRICS = (
    *LIFECYCLE_METRICS,
    # `/metrics` reports its own HTTP request as in flight, so this stays
    # observable but is deliberately not a post-request drain condition.
    "ax_engine_http_requests_in_flight",
    "ax_engine_generation_worker_ready",
    "ax_engine_generation_saturated_commands_total",
    "ax_engine_generation_stream_backlog_overflows_total",
    "ax_engine_generation_completed_requests_total",
    "ax_engine_http_status_2xx_total",
    "ax_engine_http_status_4xx_total",
    "ax_engine_http_status_5xx_total",
    "ax_engine_memory_mlx_active_bytes",
    "ax_engine_memory_mlx_cache_bytes",
    "ax_engine_memory_mlx_peak_bytes",
    "ax_engine_memory_host_resident_bytes",
    "ax_engine_memory_unattributed_active_bytes",
    "ax_engine_memory_attribution_excess_bytes",
    "ax_engine_steps_total",
    "ax_engine_scheduled_requests_total",
    "ax_engine_scheduled_tokens_total",
    "ax_engine_kv_allocated_blocks_total",
    "ax_engine_kv_released_blocks_total",
    "ax_engine_kv_cache_evictions_total",
    "ax_engine_kv_free_blocks",
    "ax_engine_kv_block_tables",
    "ax_engine_kv_prompt_entries",
    "ax_engine_kv_block_ref_entries",
    "ax_engine_kv_live_prefix_index_keys",
    "ax_engine_kv_live_prefix_request_refs",
    "ax_engine_kv_cached_blocks",
    "ax_engine_kv_cached_child_index_keys",
    "ax_engine_kv_cached_child_edges",
    "ax_engine_request_active_records",
    "ax_engine_request_terminal_snapshots",
    "ax_engine_request_terminal_snapshot_order",
    "ax_engine_request_terminal_snapshot_bytes",
    "ax_engine_request_owners",
    "ax_engine_request_active_owners",
    "ax_engine_request_terminal_owners",
    "ax_engine_request_terminal_owner_queue_entries",
    "ax_engine_request_terminal_owner_queues",
    "ax_runtime_ttft_p95_ms",
    "ax_runtime_decode_tok_per_sec",
    "ax_runtime_error_rate",
    # The unlabelled value is the one-model logical KV ledger in this test.
    # It distinguishes a full reusable block cache from model-side KV bytes.
    "ax_engine_step_kv_usage_blocks",
    "ax_runtime_kv_pages_total",
    "ax_runtime_kv_utilization",
    "ax_runtime_queue_depth",
)
MODEL_METRICS = (
    "ax_engine_model_memory_kv_report_available",
    "ax_engine_model_memory_kv_logical_bytes",
    "ax_engine_model_memory_kv_capacity_bytes",
    "ax_engine_model_memory_kv_linear_state_bytes",
    "ax_engine_model_memory_kv_paged_pool_slab_bytes",
    "ax_engine_model_memory_kv_physical_bytes",
    "ax_engine_model_memory_prefix_cache_payload_bytes",
)
COUNTER_METRICS = (
    "ax_engine_generation_saturated_commands_total",
    "ax_engine_generation_stream_backlog_overflows_total",
    "ax_engine_http_status_2xx_total",
    "ax_engine_http_status_4xx_total",
    "ax_engine_http_status_5xx_total",
    "ax_engine_generation_completed_requests_total",
    "ax_engine_steps_total",
    "ax_engine_scheduled_requests_total",
    "ax_engine_scheduled_tokens_total",
    "ax_engine_kv_allocated_blocks_total",
    "ax_engine_kv_released_blocks_total",
    "ax_engine_kv_cache_evictions_total",
)
BOUNDED_STATE_METRICS = (
    "ax_engine_kv_free_blocks",
    "ax_engine_kv_block_tables",
    "ax_engine_kv_prompt_entries",
    "ax_engine_kv_block_ref_entries",
    "ax_engine_kv_live_prefix_index_keys",
    "ax_engine_kv_live_prefix_request_refs",
    "ax_engine_kv_cached_blocks",
    "ax_engine_kv_cached_child_index_keys",
    "ax_engine_kv_cached_child_edges",
    "ax_engine_request_active_records",
    "ax_engine_request_terminal_snapshots",
    "ax_engine_request_terminal_snapshot_order",
    "ax_engine_request_terminal_snapshot_bytes",
    "ax_engine_request_owners",
    "ax_engine_request_active_owners",
    "ax_engine_request_terminal_owners",
    "ax_engine_request_terminal_owner_queue_entries",
    "ax_engine_request_terminal_owner_queues",
)
RETENTION_OBSERVABILITY_METRICS = (
    "ax_engine_generation_completed_requests_total",
    "ax_engine_steps_total",
    "ax_engine_scheduled_tokens_total",
    "ax_engine_kv_allocated_blocks_total",
    "ax_engine_kv_released_blocks_total",
    "ax_engine_kv_cache_evictions_total",
    *BOUNDED_STATE_METRICS,
)
MEMORY_SERIES_PATHS = {
    "server_rss_bytes": ("process", "rss_bytes"),
    "server_host_resident_metric_bytes": (
        "metrics",
        "values",
        "ax_engine_memory_host_resident_bytes",
    ),
    "host_wired_bytes": ("host", "wired_bytes"),
    "host_compressor_bytes": ("host", "compressor_bytes"),
    "host_swap_used_bytes": ("host", "swap", "used_bytes"),
    "iogpu_alloc_system_memory_bytes": ("host", "iogpu", "alloc_system_memory_bytes"),
    "iogpu_in_use_system_memory_bytes": ("host", "iogpu", "in_use_system_memory_bytes"),
    "mlx_active_bytes": ("metrics", "values", "ax_engine_memory_mlx_active_bytes"),
    "mlx_cache_bytes": ("metrics", "values", "ax_engine_memory_mlx_cache_bytes"),
    "mlx_unattributed_active_bytes": (
        "metrics",
        "values",
        "ax_engine_memory_unattributed_active_bytes",
    ),
    "request_terminal_snapshot_bytes": (
        "metrics",
        "values",
        "ax_engine_request_terminal_snapshot_bytes",
    ),
    "model_kv_logical_bytes": (
        "metrics",
        "values",
        "ax_engine_model_memory_kv_logical_bytes",
    ),
    "model_kv_physical_bytes": (
        "metrics",
        "values",
        "ax_engine_model_memory_kv_physical_bytes",
    ),
    "model_prefix_cache_payload_bytes": (
        "metrics",
        "values",
        "ax_engine_model_memory_prefix_cache_payload_bytes",
    ),
}
WORK_COUNTER_PATHS = {
    "completed_requests": (
        "metrics",
        "values",
        "ax_engine_generation_completed_requests_total",
    ),
    "engine_steps": ("metrics", "values", "ax_engine_steps_total"),
    "scheduled_tokens": ("metrics", "values", "ax_engine_scheduled_tokens_total"),
    "kv_allocated_blocks": (
        "metrics",
        "values",
        "ax_engine_kv_allocated_blocks_total",
    ),
}
PREFIX_CACHE_ROUTE_DECISION_KEYS = (
    "ax_mlx_prefix_cache_hits",
    "ax_mlx_prefix_cache_misses",
    "ax_mlx_prefix_cache_blocked",
    "ax_mlx_prefix_cache_stores",
    "ax_mlx_prefix_cache_evictions",
    "ax_mlx_prefix_cache_reused_tokens",
    "ax_mlx_prefix_cache_warmup_tokens",
    "retained_cache_hits",
    "prefix_reused_requests",
    "prefix_reused_tokens",
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
    """A bounded deterministic request shape in the endurance mix."""

    name: str
    mode: str
    unique_words: int
    shared_prefix_words: int
    max_output_tokens: int

    @property
    def nominal_input_words(self) -> int:
        return self.unique_words + self.shared_prefix_words


WORKLOAD_SHAPES = {
    "short_unique": WorkloadShape("short_unique", "unique", 128, 0, 96),
    "medium_unique": WorkloadShape("medium_unique", "unique", 1_024, 0, 128),
    "shared_prefix": WorkloadShape("shared_prefix", "shared_prefix", 96, 1_024, 96),
    "long_unique": WorkloadShape("long_unique", "unique", 4_096, 0, 128),
}
# One 20-request cycle is 70% short, 15% medium, 10% deliberate prefix reuse,
# and 5% long prefill.  The cases are interleaved rather than bursty.
WORKLOAD_SEQUENCE = (
    "short_unique",
    "short_unique",
    "short_unique",
    "medium_unique",
    "short_unique",
    "short_unique",
    "shared_prefix",
    "short_unique",
    "short_unique",
    "medium_unique",
    "short_unique",
    "short_unique",
    "short_unique",
    "long_unique",
    "short_unique",
    "short_unique",
    "medium_unique",
    "short_unique",
    "short_unique",
    "shared_prefix",
)
# Run after every full warm-up with fresh nonces.  These probes force the
# logical-KV cache to reclaim entries before the measured interval begins;
# otherwise a pristine first long prefill can mask a cache-pressure scheduler
# regression until hours into a soak.
CACHE_CAPACITY_PROBE_SHAPES = ("long_unique", "medium_unique")


class PreflightComplete(Exception):
    """Signal that an explicitly requested preflight-only run passed."""


@dataclass
class RunState:
    """Small mutable state; detailed evidence lives in JSONL artifacts."""

    started_wall: str
    started_monotonic: float
    server_pid: int
    requests_attempted: int = 0
    requests_ok: int = 0
    requests_failed: int = 0
    health_failures: int = 0
    consecutive_request_failures: int = 0
    lifecycle_timeouts: int = 0
    lifecycle_inconclusive: int = 0
    kv_report_unavailable: int = 0
    metric_scrape_failures: int = 0
    resource_sampler_stop_timeouts: int = 0
    quiescent_kv_logical_exceedances: int = 0
    baseline_completed_at: str | None = None
    baseline_completed_elapsed_s: float | None = None
    baseline: dict[str, Any] | None = None
    resource_baseline: dict[str, float] = field(default_factory=dict)
    baseline_coverage_concerns: list[str] = field(default_factory=list)
    baseline_stability_alerts: list[str] = field(default_factory=list)
    sampling_continuity_concerns: list[str] = field(default_factory=list)
    counter_baseline: dict[str, float] = field(default_factory=dict)
    performance_alerts: int = 0
    memory_alerts: int = 0
    # Defaults preserve the historical meaning for callers that construct a
    # state for the measured interval.  The runner explicitly marks the
    # preflight state before warm-up so a rejected run is not mistaken for a
    # zero-duration measurement.
    preflight_status: str = "passed"
    preflight_elapsed_seconds: float | None = None
    measurement_started: bool = True
    cache_capacity_rehearsal: list[dict[str, Any]] = field(default_factory=list)
    last_error: str | None = None


JSONL_LOCK = threading.Lock()


def utc_now() -> str:
    """Return a stable, timezone-explicit timestamp for evidence."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def utc_run_id() -> str:
    """Return a filesystem-safe default run identifier."""
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def positive_float(value: str) -> float:
    """Parse a strictly positive finite CLI float."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def non_negative_float(value: str) -> float:
    """Parse a non-negative finite CLI float."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
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
    """Hash an identity file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model_dir: Path) -> dict[str, Any]:
    """Return compact, auditable identity data for a local AX model package."""
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


def command_output(*command: str, timeout_s: float = 30.0) -> str:
    """Collect a diagnostic command result without obscuring test evidence."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    return (result.stdout or result.stderr).strip()


def sanitized_hardware_profile() -> str:
    """Collect hardware context while excluding durable machine identifiers."""
    profile = command_output("/usr/sbin/system_profiler", "SPHardwareDataType", timeout_s=45.0)
    return "\n".join(
        line
        for line in profile.splitlines()
        if not line.strip().startswith(SENSITIVE_HARDWARE_PREFIXES)
    )


def runtime_metadata(server_path: Path) -> dict[str, Any]:
    """Capture reproducibility context once, before the server starts."""
    packages: dict[str, str | None] = {}
    for package in ("mlx", "mlx-lm", "huggingface-hub", "tokenizers"):
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
        "power_settings": command_output("pmset", "-g"),
        "mlx_packages": packages,
        "server_path": str(server_path),
        "server_sha256": sha256_file(server_path),
        "server_version": command_output(str(server_path), "--version"),
        "launch_context": {
            "runner_pid": os.getpid(),
            "parent_pid": os.getppid(),
            "session_id": os.getsid(0),
            "process_group_id": os.getpgrp(),
            "stdin_isatty": sys.stdin.isatty(),
            "stdout_isatty": sys.stdout.isatty(),
            "working_directory": str(Path.cwd()),
        },
    }


def validate_server_version(server_path: Path, expected_version: str | None) -> str:
    """Fail before a multi-day run when the executable version is not the pinned target."""
    observed = command_output(str(server_path), "--version")
    if observed.startswith("unavailable:"):
        raise RuntimeError(f"could not read server version: {observed}")
    if expected_version is not None:
        pattern = rf"(?<![0-9A-Za-z.])v?{re.escape(expected_version)}(?![0-9A-Za-z.])"
        if re.search(pattern, observed) is None:
            raise RuntimeError(
                f"server version mismatch: expected {expected_version!r}, observed {observed!r}"
            )
    return observed


def build_server_command(args: argparse.Namespace) -> list[str]:
    """Build the one-server command used for the complete endurance lifetime."""
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
        # Pin the logical-KV geometry as well as the scheduling budget.  A
        # cache-pressure result is uninterpretable if an upgraded server has
        # silently changed either default.
        "--block-size-tokens",
        str(args.block_size_tokens),
        "--total-blocks",
        str(args.total_blocks),
        # Pin the scheduler's prefill-step budget instead of depending on a
        # server-version default. Text prompts above this limit are chunked,
        # which keeps the periodic long probe bounded without turning this
        # single-client soak into a batch-load experiment.
        "--max-batch-tokens",
        str(args.max_batch_tokens),
        *args.server_extra_arg,
    ]


def assert_port_available(host: str, port: int) -> None:
    """Avoid accidentally accepting a different pre-existing server as ours."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
    except OSError as error:
        raise RuntimeError(f"refusing to use occupied endpoint {host}:{port}: {error}") from error


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
            "error": (
                None if ok else f"health returned status={status}, body_status={reported_status}"
            ),
        }
    except Exception as error:  # noqa: BLE001 - endpoint failures are evidence.
        return {
            "ok": False,
            "http_status": None,
            "reported_status": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "error": str(error),
        }


def wait_for_server(
    process: subprocess.Popen[bytes], *, base_url: str, timeout_s: float
) -> dict[str, Any]:
    """Wait for the owned process to report ready, preserving early exits."""
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
    """Stop only the process group created by this runner, then record outcome."""
    forced = False
    if process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            forced = True
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30.0)
    return {"exit_code": process.returncode, "forced": forced}


def parse_prometheus_labels(raw: str) -> dict[str, str]:
    """Parse the simple quoted-label form emitted by AX Engine metrics."""
    labels: dict[str, str] = {}
    for match in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"', raw):
        labels[match.group(1)] = bytes(match.group(2), "utf-8").decode("unicode_escape")
    return labels


def parse_prometheus_samples(text: str) -> list[dict[str, Any]]:
    """Parse numeric Prometheus samples, retaining labels needed for model KV."""
    samples: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) < 2 or not fields[0] or fields[0].startswith("#"):
            continue
        metric_and_labels = fields[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        if "{" in metric_and_labels:
            name, remainder = metric_and_labels.split("{", 1)
            if not remainder.endswith("}"):
                continue
            labels = parse_prometheus_labels(remainder[:-1])
        else:
            name = metric_and_labels
            labels = {}
        samples.append({"name": name, "labels": labels, "value": value})
    return samples


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Backward-compatible unlabeled metric view used by focused unit tests."""
    return {
        str(sample["name"]): float(sample["value"])
        for sample in parse_prometheus_samples(text)
        if not sample["labels"]
    }


def select_ax_metrics(samples: Iterable[dict[str, Any]], model_id: str) -> dict[str, float]:
    """Select server-wide values plus model-labelled KV/cache values."""
    values: dict[str, float] = {}
    for sample in samples:
        name = sample.get("name")
        labels = sample.get("labels")
        value = sample.get("value")
        if not isinstance(name, str) or not isinstance(labels, dict):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        is_server_metric = name in SERVER_METRICS and not labels
        is_target_model_metric = name in MODEL_METRICS and labels.get("model") == model_id
        if is_server_metric or is_target_model_metric:
            values[name] = float(value)
    return values


def collect_metrics(base_url: str, model_id: str, timeout_s: float) -> dict[str, Any]:
    """Read relevant metrics; a failed scrape is evidence, not a thrown error."""
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
        values = select_ax_metrics(parse_prometheus_samples(body), model_id)
        return {
            "ok": True,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "values": values,
            "missing_lifecycle_metrics": [name for name in LIFECYCLE_METRICS if name not in values],
            "missing_model_memory_metrics": [name for name in MODEL_METRICS if name not in values],
        }
    except Exception as error:  # noqa: BLE001 - observability cannot hide request evidence.
        return {
            "ok": False,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "error": str(error),
            "values": {},
            "missing_lifecycle_metrics": list(LIFECYCLE_METRICS),
            "missing_model_memory_metrics": list(MODEL_METRICS),
        }


def lifecycle_state(metrics: dict[str, Any]) -> dict[str, Any]:
    """Classify whether native generation lifecycle queues have fully drained."""
    if not metrics.get("ok"):
        return {"state": "inconclusive", "missing": list(LIFECYCLE_METRICS), "nonzero": {}}
    values = metrics.get("values", {})
    if not isinstance(values, dict):
        return {"state": "inconclusive", "missing": list(LIFECYCLE_METRICS), "nonzero": {}}
    missing = [name for name in LIFECYCLE_METRICS if name not in values]
    if missing:
        return {"state": "inconclusive", "missing": missing, "nonzero": {}}
    nonzero = {
        name: float(values[name])
        for name in LIFECYCLE_METRICS
        if isinstance(values.get(name), (int, float)) and float(values[name]) > 0.0
    }
    return {"state": "drained" if not nonzero else "busy", "missing": [], "nonzero": nonzero}


def wait_for_quiescence(
    *, base_url: str, model_id: str, timeout_s: float, poll_interval_s: float = 0.25
) -> dict[str, Any]:
    """Poll after a response until all lifecycle gauges are zero or evidence says why not."""
    started = time.monotonic()
    attempts = 0
    latest: dict[str, Any] = {}
    while True:
        attempts += 1
        latest = collect_metrics(base_url, model_id, timeout_s=min(5.0, timeout_s))
        state = lifecycle_state(latest)
        if state["state"] != "busy":
            return {
                "state": state["state"],
                "attempts": attempts,
                "elapsed_ms": (time.monotonic() - started) * 1000.0,
                "missing": state["missing"],
                "nonzero": state["nonzero"],
                "metrics": latest,
            }
        if time.monotonic() - started >= timeout_s:
            return {
                "state": "timeout",
                "attempts": attempts,
                "elapsed_ms": (time.monotonic() - started) * 1000.0,
                "missing": [],
                "nonzero": state["nonzero"],
                "metrics": latest,
            }
        time.sleep(min(poll_interval_s, max(0.01, timeout_s)))


def process_snapshot(pid: int) -> dict[str, Any]:
    """Collect RSS/CPU for the exact owned server PID, not a process-name match."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,rss=,%cpu=,etime="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"alive": False, "pid": pid, "error": str(error)}
    fields = result.stdout.strip().split(maxsplit=3)
    if len(fields) != 4:
        return {"alive": False, "pid": pid}
    try:
        rss_kb = int(fields[1])
        return {
            "alive": True,
            "pid": int(fields[0]),
            "rss_kb": rss_kb,
            "rss_bytes": rss_kb * 1024,
            "cpu_percent": float(fields[2]),
            "elapsed": fields[3],
        }
    except ValueError:
        return {"alive": False, "pid": pid, "parse_error": result.stdout.strip()}


VM_STAT_FIELDS = {
    "Pages wired down": "wired_pages",
    "Pages active": "active_pages",
    "Pages inactive": "inactive_pages",
    "Pages speculative": "speculative_pages",
    "Pages occupied by compressor": "compressor_pages",
    "Pages purgeable": "purgeable_pages",
    "File-backed pages": "file_backed_pages",
    "Anonymous pages": "anonymous_pages",
}


def parse_vm_stat(text: str) -> dict[str, int]:
    """Extract relevant page counts from macOS ``vm_stat`` output."""
    result: dict[str, int] = {}
    for raw_name, key in VM_STAT_FIELDS.items():
        match = re.search(rf"(?m)^\s*{re.escape(raw_name)}:\s*([0-9,]+)\.", text)
        if match:
            result[key] = int(match.group(1).replace(",", ""))
    return result


def parse_memory_size(value: str) -> int | None:
    """Parse a compact macOS memory size such as ``128.00M`` into bytes."""
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])(?:B)?\s*", value, re.I)
    if not match:
        return None
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    return int(float(match.group(1)) * multipliers[match.group(2).upper()])


def parse_swap_usage(text: str) -> dict[str, int]:
    """Extract total/used/free swap bytes from ``sysctl vm.swapusage``."""
    result: dict[str, int] = {}
    for key in ("total", "used", "free"):
        match = re.search(rf"\b{key}\s*=\s*([0-9.]+\s*[KMGTP](?:B)?)", text, re.I)
        if match:
            parsed = parse_memory_size(match.group(1))
            if parsed is not None:
                result[f"{key}_bytes"] = parsed
    return result


def parse_iogpu_memory(text: str) -> dict[str, int]:
    """Extract Apple GPU driver memory counters when the host exposes them.

    These are driver-wide unified-memory observations, not per-process wired
    allocations.  They complement `vm_stat` and AX's own MLX gauges when
    diagnosing a Metal-side growth pattern on Apple Silicon.
    """
    fields = {
        "Alloc system memory": "alloc_system_memory_bytes",
        "In use system memory": "in_use_system_memory_bytes",
        "In use system memory (driver)": "in_use_driver_system_memory_bytes",
    }
    result: dict[str, int] = {}
    for raw_name, key in fields.items():
        match = re.search(rf'"{re.escape(raw_name)}"\s*=\s*([0-9]+)', text)
        if match:
            result[key] = int(match.group(1))
    return result


def parse_int_output(text: str) -> int | None:
    """Parse a clean non-negative command value, otherwise return unavailable."""
    try:
        value = int(text.strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def collect_light_host_snapshot(output_dir: Path) -> dict[str, Any]:
    """Collect low-overhead OS context suitable for minute sampling.

    Thermal state is deliberately captured with the memory gauges.  A thermal
    limit is a competing explanation for a later token/s or TTFT change; it
    should be visible in the same evidence stream instead of inferred after
    the fact from an unrelated machine snapshot.
    """
    page_size = parse_int_output(
        command_output("/usr/sbin/sysctl", "-n", "hw.pagesize", timeout_s=10.0)
    )
    page_size = page_size or 16_384
    page_counts = parse_vm_stat(command_output("vm_stat", timeout_s=10.0))
    pages = dict(page_counts)
    host: dict[str, Any] = {
        "page_size_bytes": page_size,
        "vm_pages": pages,
        "swap": parse_swap_usage(
            command_output("/usr/sbin/sysctl", "-n", "vm.swapusage", timeout_s=10.0)
        ),
        "load_average": list(os.getloadavg()),
        "thermal": command_output("pmset", "-g", "therm", timeout_s=10.0),
    }
    for page_key, value in pages.items():
        host[page_key.removesuffix("_pages") + "_bytes"] = value * page_size
    wired_limit_mib = parse_int_output(
        command_output("/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb", timeout_s=10.0)
    )
    if wired_limit_mib is not None:
        host["iogpu_wired_limit_bytes"] = wired_limit_mib * MEBIBYTE
    iogpu = parse_iogpu_memory(
        command_output("/usr/sbin/ioreg", "-l", "-w0", "-r", "-c", "IOGPU", timeout_s=10.0)
    )
    if iogpu:
        host["iogpu"] = iogpu
    try:
        disk = shutil.disk_usage(output_dir)
        host["disk_free_bytes"] = disk.free
        host["disk_total_bytes"] = disk.total
    except OSError as error:
        host["disk_error"] = str(error)
    return host


def collect_process_diagnostics(
    output_dir: Path, *, server_pid: int, reason: str
) -> dict[str, Any]:
    """Persist low-frequency process VM summaries without allocation tracing.

    `vmmap` and `footprint` provide region/category attribution if RSS later
    grows while MLX/KV gauges remain flat. They run only at checkpoints so the
    endurance workload is not continuously perturbed.
    """
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    result: dict[str, Any] = {"pid": server_pid, "reason": reason, "timestamp": utc_now()}
    for name, command in (
        ("vmmap-summary", ("/usr/bin/vmmap", "-summary", str(server_pid))),
        ("footprint", ("/usr/bin/footprint", "-p", str(server_pid), "--wide", "--swapped")),
    ):
        return_code: int | None = None
        error: str | None = None
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=45.0,
            )
            return_code = completed.returncode
            content = (completed.stdout or completed.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as command_error:
            error = str(command_error)
            content = f"unavailable: {command_error}"
        path = diagnostics_dir / f"{timestamp}-{reason}-{name}.txt"
        write_text_atomic(path, content + "\n")
        result[name.replace("-", "_")] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "available": return_code == 0 and bool(content),
            "return_code": return_code,
            "error": error,
        }
    return result


def collect_checkpoint_host_snapshot(
    output_dir: Path,
    *,
    server_pid: int | None = None,
    reason: str = "checkpoint",
) -> dict[str, Any]:
    """Add less-frequent diagnostic context to a lightweight host sample."""
    snapshot = collect_light_host_snapshot(output_dir)
    snapshot["memory_pressure"] = command_output("memory_pressure", timeout_s=30.0)
    snapshot["power"] = command_output("pmset", "-g", "batt", timeout_s=10.0)
    if server_pid is not None and process_snapshot(server_pid).get("alive"):
        snapshot["process_diagnostics"] = collect_process_diagnostics(
            output_dir,
            server_pid=server_pid,
            reason=reason,
        )
    return snapshot


class ResourceSampler:
    """Independently sample the fixed server and OS once per minute by default."""

    def __init__(
        self,
        *,
        output_dir: Path,
        events_path: Path,
        base_url: str,
        model_id: str,
        server_pid: int,
        started_monotonic: float,
        interval_s: float,
    ) -> None:
        self.output_dir = output_dir
        self.events_path = events_path
        self.base_url = base_url
        self.model_id = model_id
        self.server_pid = server_pid
        self.started_monotonic = started_monotonic
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="ax-endurance-resource-sampler", daemon=True
        )

    def sample_once(self) -> dict[str, Any]:
        """Capture one resource sample and synchronously persist it."""
        sampled_wall_unix_s = time.time()
        sample = {
            "timestamp": utc_now(),
            "kind": "resource_sample",
            # Keep an independent wall-clock marker in addition to elapsed
            # monotonic time.  Their difference lets the checkpoint logic
            # expose a sleep/scheduling gap rather than counting paused time
            # as a continuous endurance interval.
            "sampled_wall_unix_seconds": sampled_wall_unix_s,
            "elapsed_seconds": max(0.0, time.monotonic() - self.started_monotonic),
            "process": process_snapshot(self.server_pid),
            "host": collect_light_host_snapshot(self.output_dir),
            "metrics": collect_metrics(self.base_url, self.model_id, timeout_s=10.0),
        }
        append_jsonl(self.events_path, sample)
        with self._lock:
            self._samples.append(sample)
        return sample

    def start(self) -> None:
        """Record an initial sample before beginning periodic collection."""
        self.sample_once()
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.sample_once()
            except Exception as error:  # noqa: BLE001 - do not kill workload sampling thread.
                append_jsonl(
                    self.events_path,
                    {"timestamp": utc_now(), "kind": "resource_sampler_error", "error": str(error)},
                )

    def stop(self) -> bool:
        """Stop sampling and report whether the sampler joined cleanly."""
        self._stop.set()
        self._thread.join(timeout=60.0)
        return not self._thread.is_alive()

    def samples(self) -> list[dict[str, Any]]:
        """Return a stable shallow copy of collected samples."""
        with self._lock:
            return list(self._samples)


def select_shape(request_index: int) -> WorkloadShape:
    """Return a deterministic, interleaved shape from the 20-request cycle."""
    if request_index <= 0:
        raise ValueError("request_index must be positive")
    return WORKLOAD_SHAPES[WORKLOAD_SEQUENCE[(request_index - 1) % len(WORKLOAD_SEQUENCE)]]


def deterministic_words(count: int, salt: int) -> str:
    """Generate deterministic synthetic text without model/user data."""
    offset = salt % len(PROMPT_WORDS)
    return " ".join(PROMPT_WORDS[(offset + index) % len(PROMPT_WORDS)] for index in range(count))


def deterministic_prompt(shape: WorkloadShape, request_index: int) -> str:
    """Create a unique or deliberately shared-prefix raw-text request."""
    if shape.mode == "shared_prefix":
        # The long common prefix intentionally exercises the prefix-cache path.
        # A unique tail keeps the request semantically distinct and bounded.
        shared = deterministic_words(shape.shared_prefix_words, salt=0)
        tail = deterministic_words(shape.unique_words, salt=request_index)
        return (
            "Shared AX Engine endurance prefix follows.\n"
            f"{shared}\n"
            f"unique_tail_nonce={request_index:08d}; acknowledge this synthetic probe.\n{tail}"
        )
    # Put the nonce first so a normal prefix cache cannot make unique workload
    # requests look artificially warm after the initial header tokens.
    body = deterministic_words(shape.unique_words, salt=request_index)
    return (
        f"unique_nonce_{request_index:08d} AX Engine endurance probe. "
        "Process the synthetic sequence and return a concise acknowledgement.\n"
        f"{body}"
    )


def load_local_tokenizer(model_dir: Path) -> Callable[[str], list[int]]:
    """Load the exact artifact tokenizer for native MLX's token-only route.

    The native `/v1/generate/stream` MLX preview endpoint intentionally accepts
    pre-tokenized input only.  Tokenizing locally makes that restriction
    explicit, avoids timing client text tokenization as server prefill, and
    gives the runner an exact sent-token count to reconcile with the native
    request event.  `tokenizers` reads the model's bundled `tokenizer.json`
    directly, matching AX's non-special-token native prompt encoding.
    """
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"native MLX token input requires {tokenizer_path}")
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "native MLX endurance input needs the Python `tokenizers` package"
        ) from error
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as error:  # noqa: BLE001 - artifact tokenizer errors are preflight evidence.
        raise RuntimeError(f"failed to load tokenizer.json at {tokenizer_path}: {error}") from error

    def encode(text: str) -> list[int]:
        try:
            token_ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
        except Exception as error:  # noqa: BLE001 - preserve artifact encoding failure context.
            raise RuntimeError(f"failed to tokenize endurance prompt: {error}") from error
        if not token_ids:
            raise RuntimeError("endurance tokenizer returned an empty prompt")
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in token_ids
        ):
            raise RuntimeError("endurance tokenizer returned an invalid token id")
        return token_ids

    return encode


def make_prompt_item(
    shape: WorkloadShape,
    request_index: int,
    *,
    tokenize: Callable[[str], list[int]] | None = None,
) -> serving_bench.PromptItem:
    """Adapt a shape into the existing native-serving benchmark request type."""
    input_text = deterministic_prompt(shape, request_index)
    input_tokens = tokenize(input_text) if tokenize is not None else None
    return serving_bench.PromptItem(
        id=f"{shape.name}-{request_index:08d}",
        category=shape.name,
        input_text=input_text,
        input_tokens=input_tokens,
        input_tokens_count=len(input_tokens)
        if input_tokens is not None
        else shape.nominal_input_words,
        max_output_tokens=shape.max_output_tokens,
        metadata={
            "mode": shape.mode,
            "nominal_input_words": shape.nominal_input_words,
            "shared_prefix_words": shape.shared_prefix_words,
            "input_encoding": "tokenizer.json" if input_tokens is not None else "raw_text",
        },
    )


def run_stream_request(
    *,
    prompt: serving_bench.PromptItem,
    model_id: str,
    base_url: str,
    timeout_s: float,
    input_kind: str = "tokens",
    stream_func: Callable[..., Any] = serving_bench.http_sse_events,
) -> dict[str, Any]:
    """Run one bounded greedy native stream and retain only timing/token evidence."""
    started = time.perf_counter()
    try:
        payload = serving_bench.build_payload(
            prompt,
            model_id=model_id,
            input_kind=input_kind,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=0,
        )
        sampling = payload.get("sampling")
        if isinstance(sampling, dict):
            # AX-native control used by the existing fault soak: bounded fixed-length
            # decode avoids a mostly-empty early-EOS workload.
            sampling["ignore_eos"] = True
        events = list(stream_func(f"{base_url}/v1/generate/stream", payload, timeout_s))
        observation = serving_bench.observe_stream(
            events,
            prompt=prompt,
            scheduled_at_s=0.0,
            started_at_s=0.0,
            completed_at_s=time.perf_counter() - started,
        )
        tpot_ms = observation.get("client_tpot_ms")
        if isinstance(tpot_ms, (int, float)) and tpot_ms > 0.0:
            observation["client_decode_tok_s"] = 1_000.0 / float(tpot_ms)
        else:
            observation["client_decode_tok_s"] = None
        prompt_token_count = native_prompt_token_count(events)
        observation["prompt_token_count"] = prompt_token_count
        observation["client_input_token_count"] = prompt.input_tokens_count
        ttft_ms = observation.get("ttft_ms")
        if (
            isinstance(prompt_token_count, int)
            and prompt_token_count > 0
            and isinstance(ttft_ms, (int, float))
            and ttft_ms > 0.0
        ):
            # Effective prefill includes native admission/tokenization and the
            # first-token step. Exact native prompt length avoids treating
            # synthetic words as tokenizer tokens.
            observation["effective_prefill_tok_s"] = prompt_token_count * 1_000.0 / ttft_ms
        else:
            observation["effective_prefill_tok_s"] = None
        return observation
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
            "client_decode_tok_s": None,
            "prompt_token_count": None,
            "client_input_token_count": prompt.input_tokens_count,
            "effective_prefill_tok_s": None,
            "output_tokens": None,
            "route_decisions": {},
        }


def native_prompt_token_count(events: Iterable[tuple[str | None, Any, float]]) -> int | None:
    """Return the exact native prompt length carried by the SSE request event."""
    for event_name, payload, _elapsed_s in events:
        if event_name not in {"request", "step"} or not isinstance(payload, dict):
            continue
        request = payload.get("request")
        if not isinstance(request, dict):
            continue
        prompt_len = request.get("prompt_len")
        if isinstance(prompt_len, int) and not isinstance(prompt_len, bool) and prompt_len > 0:
            return prompt_len
    return None


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a linear-interpolated percentile for concise window reports."""
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    index = (len(clean) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(clean) - 1)
    return clean[lower] + (clean[upper] - clean[lower]) * (index - lower)


def summarize_values(values: Iterable[float | int | None]) -> dict[str, float] | None:
    """Produce a compact latency/throughput distribution."""
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    clean = [value for value in clean if math.isfinite(value)]
    if not clean:
        return None
    return {
        "count": float(len(clean)),
        "min": min(clean),
        "mean": statistics.fmean(clean),
        "p05": percentile(clean, 0.05),
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean),
    }


def request_successes(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records containing a successful stream observation."""
    return [record for record in records if record.get("request", {}).get("ok")]


def summarize_requests(records: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    """Summarize one reporting window and break performance out by request shape."""
    successes = request_successes(records)
    failures = len(records) - len(successes)
    shapes = sorted({str(record.get("shape", "unknown")) for record in records})

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        observations = [record["request"] for record in request_successes(group)]
        return {
            "requests": len(group),
            "successful_requests": len(observations),
            "failed_requests": len(group) - len(observations),
            "ttft_ms": summarize_values([item.get("ttft_ms") for item in observations]),
            "client_tpot_ms": summarize_values(
                [item.get("client_tpot_ms") for item in observations]
            ),
            "client_decode_tok_s": summarize_values(
                [item.get("client_decode_tok_s") for item in observations]
            ),
            "effective_prefill_tok_s": summarize_values(
                [item.get("effective_prefill_tok_s") for item in observations]
            ),
            "e2e_latency_ms": summarize_values(
                [item.get("e2e_latency_ms") for item in observations]
            ),
            "output_tokens": summarize_values([item.get("output_tokens") for item in observations]),
            "route_decisions": serving_bench.summarize_route_decisions(observations),
        }

    duration = max(elapsed_s, 0.001)
    overall = summarize_group(records)
    output_tokens = [
        float(record["request"]["output_tokens"])
        for record in successes
        if isinstance(record["request"].get("output_tokens"), (int, float))
    ]
    by_shape = {
        shape: summarize_group([record for record in records if record.get("shape") == shape])
        for shape in shapes
    }
    shared_prefix = by_shape.get("shared_prefix", {})
    shared_route_decisions = (
        shared_prefix.get("route_decisions", {}) if isinstance(shared_prefix, dict) else {}
    )
    prefix_cache_route_evidence = {
        key: shared_route_decisions[key]
        for key in PREFIX_CACHE_ROUTE_DECISION_KEYS
        if isinstance(shared_route_decisions, dict)
        and isinstance(shared_route_decisions.get(key), (int, float))
    }
    return {
        "requests": len(records),
        "successful_requests": len(successes),
        "failed_requests": failures,
        "success_ratio": len(successes) / len(records) if records else 0.0,
        "request_throughput_rps": len(successes) / duration,
        "output_token_throughput_tok_s": sum(output_tokens) / duration,
        "overall": overall,
        "by_shape": by_shape,
        # Only shared-prefix requests intentionally exercise cross-request
        # reuse.  Keep their cache telemetry separate from all-route totals so
        # a four-hour handoff can say whether this part of the workload really
        # touched the cache, without treating a warm allocation as a leak.
        "shared_prefix_cache_route_evidence": prefix_cache_route_evidence,
    }


def summarize_window(records: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    """Compatibility alias for tests and callers expecting a window summary."""
    return summarize_requests(records, elapsed_s)


def get_nested_number(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    """Get one finite numeric value from a nested evidence record."""
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, (int, float)) and math.isfinite(float(current)):
        return float(current)
    return None


def resource_sample_is_quiescent(sample: dict[str, Any]) -> bool | None:
    """Classify a resource sample using the same native drain gauges as requests.

    On Apple unified memory, wired pages and MLX active memory can legitimately
    rise while a request is executing and fall after it drains.  Comparing an
    active sample with an idle baseline would look like a multi-gigabyte leak.
    A definitive quiescent classification therefore needs every lifecycle
    gauge; unknown instrumentation remains unclassified rather than assumed
    idle.
    """
    metrics = sample.get("metrics", {})
    values = metrics.get("values", {}) if isinstance(metrics, dict) else {}
    if not isinstance(values, dict):
        return None
    lifecycle_values: list[float] = []
    for name in LIFECYCLE_METRICS:
        value = values.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        if not math.isfinite(float(value)):
            return None
        lifecycle_values.append(float(value))
    return all(value <= 0.0 for value in lifecycle_values)


def memory_points(
    samples: Iterable[dict[str, Any]],
    path: tuple[str, ...],
    *,
    quiescent: bool | None = None,
) -> list[tuple[float, float]]:
    """Select chronologically ordered memory points, optionally by lifecycle phase."""
    points: list[tuple[float, float]] = []
    for sample in samples:
        if quiescent is not None and resource_sample_is_quiescent(sample) is not quiescent:
            continue
        elapsed = sample.get("elapsed_seconds")
        value = get_nested_number(sample, path)
        if not isinstance(elapsed, (int, float)) or value is None:
            continue
        if math.isfinite(float(elapsed)):
            points.append((float(elapsed), value))
    return points


def linear_slope_per_hour(points: Iterable[tuple[float, float]]) -> float | None:
    """Return ordinary-least-squares value slope per hour, or inconclusive."""
    slope = linear_slope_y_per_x(points)
    return None if slope is None else slope * 3_600.0


def linear_slope_y_per_x(points: Iterable[tuple[float, float]]) -> float | None:
    """Return the ordinary-least-squares Y/X slope, or inconclusive."""
    clean = [(float(x), float(y)) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if len(clean) < 3:
        return None
    mean_x = statistics.fmean(point[0] for point in clean)
    mean_y = statistics.fmean(point[1] for point in clean)
    denominator = sum((x - mean_x) ** 2 for x, _ in clean)
    if denominator <= 0.0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in clean)
    return numerator / denominator


def positive_step_ratio(points: list[tuple[float, float]]) -> float | None:
    """Fraction of consecutive observations whose retained value increased."""
    if len(points) < 2:
        return None
    deltas = [
        current[1] - previous[1] for previous, current in zip(points, points[1:], strict=False)
    ]
    return sum(delta > 0.0 for delta in deltas) / len(deltas)


def bounded_state_analysis(
    *,
    samples: list[dict[str, Any]],
    baseline_end_elapsed_s: float | None,
) -> dict[str, Any]:
    """Preserve trajectories for internal containers that can grow per request.

    These gauges include intentionally bounded caches and terminal-record stores,
    so their growth is diagnostic rather than automatically classified as a leak.
    Plateau behavior and bytes per completed request make a producer/consumer
    imbalance visible without relying on process RSS alone.
    """
    output: dict[str, Any] = {}
    completed_path = ("metrics", "values", "ax_engine_generation_completed_requests_total")
    for name in BOUNDED_STATE_METRICS:
        path = ("metrics", "values", name)
        quiescent = memory_points(samples, path, quiescent=True)
        points = quiescent if len(quiescent) >= 3 else memory_points(samples, path)
        if not points:
            continue
        post_baseline = [
            point
            for point in points
            if baseline_end_elapsed_s is None or point[0] >= baseline_end_elapsed_s
        ]
        entry: dict[str, Any] = {
            "current": points[-1][1],
            "peak": max(value for _elapsed, value in points),
            "samples": len(points),
            "quiescent_samples": len(quiescent),
        }
        if len(post_baseline) >= 2:
            entry.update(
                {
                    "post_baseline_growth": post_baseline[-1][1] - post_baseline[0][1],
                    "post_baseline_slope_per_hour": linear_slope_per_hour(post_baseline),
                    "post_baseline_positive_step_ratio": positive_step_ratio(post_baseline),
                    "post_baseline_span_hours": (post_baseline[-1][0] - post_baseline[0][0])
                    / 3_600.0,
                }
            )
            completed_by_elapsed = {
                elapsed: value
                for elapsed, value in memory_points(samples, completed_path)
                if baseline_end_elapsed_s is None or elapsed >= baseline_end_elapsed_s
            }
            normalized_points = [
                (completed_by_elapsed[elapsed], value)
                for elapsed, value in post_baseline
                if elapsed in completed_by_elapsed
            ]
            entry["ols_units_per_completed_request"] = linear_slope_y_per_x(normalized_points)
        output[name] = entry
    return output


def work_normalized_memory_slopes(
    *,
    samples: list[dict[str, Any]],
    memory_path: tuple[str, ...],
    baseline_end_elapsed_s: float | None,
) -> dict[str, Any]:
    """Relate retained bytes to completed work, not wall time alone."""
    if baseline_end_elapsed_s is None:
        return {}
    output: dict[str, Any] = {}
    for counter_name, counter_path in WORK_COUNTER_PATHS.items():
        points: list[tuple[float, float]] = []
        for sample in samples:
            elapsed = sample.get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)) or float(elapsed) < baseline_end_elapsed_s:
                continue
            memory_value = get_nested_number(sample, memory_path)
            counter_value = get_nested_number(sample, counter_path)
            if memory_value is None or counter_value is None:
                continue
            points.append((counter_value, memory_value))
        if len(points) < 3:
            continue
        counter_delta = points[-1][0] - points[0][0]
        if counter_delta <= 0.0:
            continue
        slope = linear_slope_y_per_x(points)
        if slope is None:
            continue
        output[counter_name] = {
            "samples": len(points),
            "counter_delta": counter_delta,
            "ols_bytes_per_unit": slope,
            "endpoint_bytes_per_unit": (points[-1][1] - points[0][1]) / counter_delta,
        }
    return output


def default_max_sampling_gap_seconds(resource_interval_s: float) -> float:
    """Set a forgiving but finite continuity bound from the sample cadence."""
    return max(60.0, resource_interval_s * 2.0 + 20.0)


def evaluate_sampling_continuity(
    samples: Iterable[dict[str, Any]], *, max_gap_seconds: float
) -> list[str]:
    """Find observation gaps that would weaken a non-stop endurance claim.

    A resource sampler runs independently of the request loop, so a gap larger
    than its bounded cadence signals a paused runner, host sleep, or severe
    host scheduling stall.  Wall-minus-monotonic divergence catches a shorter
    sleep-like gap on platforms whose monotonic counter pauses while asleep.
    Both are watch conditions: the primary workload result is retained, but it
    must not be reported as uninterrupted evidence.
    """
    ordered = list(samples)
    messages: list[str] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        previous_wall = previous.get("sampled_wall_unix_seconds")
        current_wall = current.get("sampled_wall_unix_seconds")
        if not isinstance(previous_wall, (int, float)) or not isinstance(
            current_wall, (int, float)
        ):
            continue
        wall_gap = float(current_wall) - float(previous_wall)
        if not math.isfinite(wall_gap) or wall_gap < 0.0:
            continue
        previous_elapsed = previous.get("elapsed_seconds")
        current_elapsed = current.get("elapsed_seconds")
        monotonic_gap: float | None = None
        if isinstance(previous_elapsed, (int, float)) and isinstance(current_elapsed, (int, float)):
            candidate = float(current_elapsed) - float(previous_elapsed)
            if math.isfinite(candidate) and candidate >= 0.0:
                monotonic_gap = candidate

        elapsed_label = (
            f" near elapsed {float(current_elapsed) / 3600.0:.2f} h"
            if isinstance(current_elapsed, (int, float))
            else ""
        )
        if wall_gap > max_gap_seconds:
            messages.append(
                f"resource sampling gap {wall_gap:.1f} s exceeds "
                f"{max_gap_seconds:.1f} s{elapsed_label}"
            )
        if monotonic_gap is not None:
            divergence = wall_gap - monotonic_gap
            sleep_threshold = max(15.0, max_gap_seconds / 4.0)
            if divergence > sleep_threshold:
                messages.append(
                    f"wall/monotonic sampling divergence {divergence:.1f} s "
                    f"suggests host sleep or clock change{elapsed_label}"
                )
    return messages


def memory_analysis(
    *,
    samples: list[dict[str, Any]],
    resource_baseline: dict[str, float],
    window_start_elapsed_s: float,
    baseline_end_elapsed_s: float | None = None,
) -> dict[str, Any]:
    """Report memory current/peak/growth/slope without mistaking capacity for a leak."""
    analysis: dict[str, Any] = {"sample_count": len(samples), "series": {}}
    for name, path in MEMORY_SERIES_PATHS.items():
        points = memory_points(samples, path)
        if not points:
            continue
        values = [point[1] for point in points]
        current = values[-1]
        baseline = resource_baseline.get(name)
        recent_points = [point for point in points if point[0] >= window_start_elapsed_s]
        after_baseline_points = [
            point
            for point in points
            if baseline is None
            or baseline_end_elapsed_s is None
            or point[0] >= baseline_end_elapsed_s
        ]
        entry: dict[str, Any] = {
            "current_bytes": current,
            "peak_bytes": max(values),
            "samples": len(points),
            "window_slope_mib_per_hour": _bytes_slope_to_mib(linear_slope_per_hour(recent_points)),
            "lifetime_slope_mib_per_hour": _bytes_slope_to_mib(
                linear_slope_per_hour(after_baseline_points)
            ),
        }
        if baseline is not None:
            entry["baseline_median_bytes"] = baseline
            entry["growth_mib"] = (current - baseline) / MEBIBYTE

        # Keep active/all-sample observations for forensic diagnosis, but
        # assess retained growth with like-for-like idle samples whenever
        # native lifecycle instrumentation allows it.
        quiescent_points = memory_points(samples, path, quiescent=True)
        active_points = memory_points(samples, path, quiescent=False)
        entry["quiescent_samples"] = len(quiescent_points)
        entry["active_samples"] = len(active_points)
        if active_points:
            entry["active_peak_bytes"] = max(value for _elapsed, value in active_points)
        if quiescent_points:
            quiescent_values = [value for _elapsed, value in quiescent_points]
            quiescent_recent = [
                point for point in quiescent_points if point[0] >= window_start_elapsed_s
            ]
            quiescent_after_baseline = [
                point
                for point in quiescent_points
                if baseline is None
                or baseline_end_elapsed_s is None
                or point[0] >= baseline_end_elapsed_s
            ]
            entry.update(
                {
                    "quiescent_current_bytes": quiescent_values[-1],
                    "quiescent_peak_bytes": max(quiescent_values),
                    "quiescent_window_slope_mib_per_hour": _bytes_slope_to_mib(
                        linear_slope_per_hour(quiescent_recent)
                    ),
                    "quiescent_lifetime_slope_mib_per_hour": _bytes_slope_to_mib(
                        linear_slope_per_hour(quiescent_after_baseline)
                    ),
                    "quiescent_positive_step_ratio": positive_step_ratio(quiescent_after_baseline),
                    "quiescent_post_baseline_span_hours": (
                        (quiescent_after_baseline[-1][0] - quiescent_after_baseline[0][0]) / 3_600.0
                        if len(quiescent_after_baseline) >= 2
                        else 0.0
                    ),
                }
            )
            if baseline is not None:
                entry["quiescent_growth_mib"] = (quiescent_values[-1] - baseline) / MEBIBYTE
        entry["work_normalized"] = work_normalized_memory_slopes(
            samples=samples,
            memory_path=path,
            baseline_end_elapsed_s=baseline_end_elapsed_s,
        )
        analysis["series"][name] = entry
    return analysis


def _bytes_slope_to_mib(slope: float | None) -> float | None:
    return None if slope is None else slope / MEBIBYTE


def build_resource_baseline(samples: list[dict[str, Any]], baseline_s: float) -> dict[str, float]:
    """Use baseline-window medians to avoid classifying normal allocator noise as drift."""
    baseline: dict[str, float] = {}
    baseline_samples = [
        sample
        for sample in samples
        if isinstance(sample.get("elapsed_seconds"), (int, float))
        and float(sample["elapsed_seconds"]) <= baseline_s
    ]
    quiescent_baseline_samples = [
        sample for sample in baseline_samples if resource_sample_is_quiescent(sample) is True
    ]
    # A 4-hour run has many idle samples. A short pilot or an observability
    # gap may not, in which case retaining all samples is more honest than
    # dropping the memory reference entirely.
    selected_samples = (
        quiescent_baseline_samples if len(quiescent_baseline_samples) >= 3 else baseline_samples
    )
    for name, path in MEMORY_SERIES_PATHS.items():
        values = [
            value
            for sample in selected_samples
            for value in [get_nested_number(sample, path)]
            if value is not None
        ]
        median = percentile(values, 0.50)
        if median is not None:
            baseline[name] = median
    return baseline


def evaluate_baseline_stability(
    *,
    samples: list[dict[str, Any]],
    baseline_s: float,
    baseline_growth_mib: float,
    max_slope_mib_per_hour: float,
    max_swap_growth_mib: float,
) -> list[str]:
    """Flag a baseline that is still climbing instead of treating it as a reference.

    The first and last quartiles establish material growth, while a separate
    latter-half slope establishes that the baseline is still rising.  This
    avoids mistaking a one-time warm-cache allocation for ongoing drift.  It is
    deliberately a watch condition: it says the run needs investigation, not
    that a warm cache is a leak.
    """
    baseline_samples = [
        sample
        for sample in samples
        if isinstance(sample.get("elapsed_seconds"), (int, float))
        and float(sample["elapsed_seconds"]) <= baseline_s
    ]
    quiescent_baseline_samples = [
        sample for sample in baseline_samples if resource_sample_is_quiescent(sample) is True
    ]
    trend_samples = (
        quiescent_baseline_samples if len(quiescent_baseline_samples) >= 3 else baseline_samples
    )
    if len(trend_samples) < 3:
        return ["baseline has fewer than three resource samples; resource trend is inconclusive"]

    messages: list[str] = []
    for name, path in MEMORY_SERIES_PATHS.items():
        points = [
            (float(sample["elapsed_seconds"]), value)
            for sample in trend_samples
            for value in [get_nested_number(sample, path)]
            if value is not None
        ]
        if len(points) < 3:
            continue
        endpoint_count = max(1, len(points) // 4)
        initial = percentile([value for _elapsed, value in points[:endpoint_count]], 0.50)
        current = percentile([value for _elapsed, value in points[-endpoint_count:]], 0.50)
        latter_half = points[len(points) // 2 :]
        slope = _bytes_slope_to_mib(linear_slope_per_hour(latter_half))
        if initial is None or current is None:
            continue
        growth_mib = (current - initial) / MEBIBYTE
        if name == "host_swap_used_bytes":
            if growth_mib >= max_swap_growth_mib:
                messages.append(
                    "baseline did not settle: host swap used rose "
                    f"{growth_mib:.1f} MiB (guardrail {max_swap_growth_mib:.1f} MiB)"
                )
        elif (
            growth_mib >= baseline_growth_mib
            and isinstance(slope, (int, float))
            and slope >= max_slope_mib_per_hour
        ):
            messages.append(
                f"baseline did not settle: {name} rose {growth_mib:.1f} MiB with "
                f"{slope:.1f} MiB/h slope"
            )
    return messages


def counter_deltas(values: dict[str, Any], baseline: dict[str, float]) -> dict[str, float]:
    """Calculate non-negative deltas for server counters sampled after warmup."""
    output: dict[str, float] = {}
    for name in COUNTER_METRICS:
        current = values.get(name)
        initial = baseline.get(name)
        if isinstance(current, (int, float)) and initial is not None:
            output[name] = max(0.0, float(current) - initial)
    return output


def evaluate_performance_regression(
    *,
    baseline: dict[str, Any] | None,
    window: dict[str, Any],
    min_samples: int,
    max_ttft_p95_ratio: float,
    min_decode_p05_ratio: float,
    min_prefill_p05_ratio: float = DEFAULT_MIN_PREFILL_P05_RATIO,
) -> list[str]:
    """Compare same-shape p95 TTFT, p05 decode, and p05 prefill to baseline."""
    if baseline is None:
        return []
    messages: list[str] = []
    baseline_shapes = baseline.get("by_shape", {})
    window_shapes = window.get("by_shape", {})
    if not isinstance(baseline_shapes, dict) or not isinstance(window_shapes, dict):
        return messages
    for shape, current in window_shapes.items():
        reference = baseline_shapes.get(shape)
        if not isinstance(current, dict) or not isinstance(reference, dict):
            continue
        if (
            int(current.get("successful_requests", 0)) < min_samples
            or int(reference.get("successful_requests", 0)) < min_samples
        ):
            continue
        current_ttft = get_distribution_value(current, "ttft_ms", "p95")
        baseline_ttft = get_distribution_value(reference, "ttft_ms", "p95")
        if current_ttft and baseline_ttft and current_ttft > baseline_ttft * max_ttft_p95_ratio:
            messages.append(
                f"{shape}: p95 TTFT {current_ttft:.1f} ms is "
                f"{current_ttft / baseline_ttft:.2f}x baseline ({baseline_ttft:.1f} ms)"
            )
        current_decode = get_distribution_value(current, "client_decode_tok_s", "p05")
        baseline_decode = get_distribution_value(reference, "client_decode_tok_s", "p05")
        if (
            current_decode
            and baseline_decode
            and current_decode < baseline_decode * min_decode_p05_ratio
        ):
            messages.append(
                f"{shape}: p05 decode {current_decode:.2f} tok/s is "
                f"{current_decode / baseline_decode:.2f}x baseline ({baseline_decode:.2f} tok/s)"
            )
        current_prefill = get_distribution_value(current, "effective_prefill_tok_s", "p05")
        baseline_prefill = get_distribution_value(reference, "effective_prefill_tok_s", "p05")
        if (
            current_prefill
            and baseline_prefill
            and current_prefill < baseline_prefill * min_prefill_p05_ratio
        ):
            messages.append(
                f"{shape}: p05 effective prefill {current_prefill:.2f} tok/s is "
                f"{current_prefill / baseline_prefill:.2f}x baseline "
                f"({baseline_prefill:.2f} tok/s)"
            )
    return messages


def get_distribution_value(summary: dict[str, Any], metric: str, statistic: str) -> float | None:
    """Read a finite statistic from a nested request distribution."""
    distribution = summary.get(metric)
    if not isinstance(distribution, dict):
        return None
    value = distribution.get(statistic)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def evaluate_baseline_coverage(baseline: dict[str, Any] | None, min_samples: int) -> list[str]:
    """Require enough same-shape client evidence before calling a run clean.

    A successful HTTP stream without a measured first token or native prompt length
    cannot substantiate TTFT or effective-prefill stability.  Keep that gap
    explicit rather than silently skipping the corresponding regression check.
    """
    if baseline is None:
        return ["performance/resource baseline was not finalized"]
    by_shape = baseline.get("by_shape")
    if not isinstance(by_shape, dict):
        return ["baseline has no per-shape request summaries"]

    messages: list[str] = []
    required_metrics = (
        ("ttft_ms", "TTFT"),
        ("client_decode_tok_s", "decode token/s"),
        ("effective_prefill_tok_s", "effective prefill token/s"),
    )
    for shape in WORKLOAD_SHAPES:
        summary = by_shape.get(shape)
        if not isinstance(summary, dict):
            messages.append(f"baseline has no {shape} requests")
            continue
        successful = int(summary.get("successful_requests", 0))
        if successful < min_samples:
            messages.append(
                f"baseline {shape} has {successful} successful request(s); need {min_samples}"
            )
        for metric, label in required_metrics:
            available = get_distribution_value(summary, metric, "count")
            count = int(available) if available is not None else 0
            if count < min_samples:
                messages.append(
                    f"baseline {shape} has {count} {label} sample(s); need {min_samples}"
                )
    return messages


def evaluate_memory_alerts(
    *,
    analysis: dict[str, Any],
    max_growth_mib: float,
    max_slope_mib_per_hour: float,
    max_swap_growth_mib: float = DEFAULT_MAX_SWAP_GROWTH_MIB,
) -> list[str]:
    """Flag persistent-looking growth, retaining cache-capacity ambiguity in wording."""
    messages: list[str] = []
    series = analysis.get("series", {})
    if not isinstance(series, dict):
        return messages
    for name, value in series.items():
        if name == "host_swap_used_bytes":
            # Swap has its own lower absolute-growth guardrail below. Reporting
            # it here as well would duplicate the same concern when both
            # thresholds happen to trigger.
            continue
        if not isinstance(value, dict):
            continue
        quiescent_growth = value.get("quiescent_growth_mib")
        quiescent_slope = value.get("quiescent_lifetime_slope_mib_per_hour")
        uses_quiescent = isinstance(quiescent_growth, (int, float)) and isinstance(
            quiescent_slope, (int, float)
        )
        growth = quiescent_growth if uses_quiescent else value.get("growth_mib")
        slope = quiescent_slope if uses_quiescent else value.get("lifetime_slope_mib_per_hour")
        if (
            isinstance(growth, (int, float))
            and isinstance(slope, (int, float))
            and growth >= max_growth_mib
            and slope >= max_slope_mib_per_hour
        ):
            scope = "quiescent " if uses_quiescent else ""
            messages.append(
                f"{scope}{name} rose {growth:.1f} MiB with {slope:.1f} MiB/h slope after baseline"
            )
    swap = series.get("host_swap_used_bytes")
    if isinstance(swap, dict):
        swap_growth = swap.get("quiescent_growth_mib", swap.get("growth_mib"))
        if isinstance(swap_growth, (int, float)) and swap_growth >= max_swap_growth_mib:
            messages.append(
                f"host swap used rose {swap_growth:.1f} MiB after baseline "
                f"(guardrail {max_swap_growth_mib:.1f} MiB)"
            )
    return messages


def evaluate_memory_early_warnings(
    analysis: dict[str, Any],
    *,
    min_growth_mib: float = DEFAULT_EARLY_MEMORY_GROWTH_MIB,
    min_slope_mib_per_hour: float = DEFAULT_EARLY_MEMORY_SLOPE_MIB_PER_HOUR,
    min_positive_step_ratio: float = DEFAULT_EARLY_MEMORY_POSITIVE_STEP_RATIO,
) -> list[str]:
    """Flag smaller monotonic retained growth before the material failure gate.

    Only process RSS and AX's unattributed-active gauge participate. Bounded
    allocator and KV caches are reported diagnostically but do not trigger this
    early signal while they fill toward a fixed capacity.
    """
    series = analysis.get("series", {})
    if not isinstance(series, dict):
        return []
    messages: list[str] = []
    for name in ("server_rss_bytes", "mlx_unattributed_active_bytes"):
        values = series.get(name)
        if not isinstance(values, dict):
            continue
        growth = values.get("quiescent_growth_mib", values.get("growth_mib"))
        slope = values.get(
            "quiescent_lifetime_slope_mib_per_hour",
            values.get("lifetime_slope_mib_per_hour"),
        )
        ratio = values.get("quiescent_positive_step_ratio")
        span = values.get("quiescent_post_baseline_span_hours")
        if (
            isinstance(growth, (int, float))
            and isinstance(slope, (int, float))
            and isinstance(ratio, (int, float))
            and isinstance(span, (int, float))
            and span >= 4.0
            and growth >= min_growth_mib
            and slope >= min_slope_mib_per_hour
            and ratio >= min_positive_step_ratio
        ):
            messages.append(
                f"early retained-memory warning: {name} rose {growth:.1f} MiB with "
                f"{slope:.1f} MiB/h slope and {ratio:.1%} positive steps over {span:.1f} h"
            )
    return messages


def evaluate_window_guardrails(
    *,
    state: RunState,
    records: list[dict[str, Any]],
    window_elapsed_s: float,
    latest_metrics: dict[str, Any],
    resource_samples: list[dict[str, Any]],
    window_start_elapsed_s: float,
    min_performance_samples: int,
    max_ttft_p95_ratio: float,
    min_decode_p05_ratio: float,
    max_client_error_rate: float,
    memory_growth_mib: float,
    memory_slope_mib_per_hour: float,
    early_memory_growth_mib: float = DEFAULT_EARLY_MEMORY_GROWTH_MIB,
    early_memory_slope_mib_per_hour: float = DEFAULT_EARLY_MEMORY_SLOPE_MIB_PER_HOUR,
    early_memory_positive_step_ratio: float = DEFAULT_EARLY_MEMORY_POSITIVE_STEP_RATIO,
    max_swap_growth_mib: float = DEFAULT_MAX_SWAP_GROWTH_MIB,
    min_prefill_p05_ratio: float = DEFAULT_MIN_PREFILL_P05_RATIO,
) -> tuple[list[str], list[str]]:
    """Evaluate one complete or terminal partial window with identical gates."""
    current_window = summarize_requests(records, max(window_elapsed_s, 0.001))
    performance_alerts = evaluate_performance_regression(
        baseline=state.baseline,
        window=current_window,
        min_samples=min_performance_samples,
        max_ttft_p95_ratio=max_ttft_p95_ratio,
        min_decode_p05_ratio=min_decode_p05_ratio,
        min_prefill_p05_ratio=min_prefill_p05_ratio,
    )
    if state.requests_attempted:
        client_error_rate = state.requests_failed / state.requests_attempted
        if client_error_rate > max_client_error_rate:
            performance_alerts.append(
                f"client error rate {client_error_rate:.4%} exceeds "
                f"{max_client_error_rate:.4%} guardrail"
            )
    metric_values = latest_metrics.get("values", {})
    if isinstance(metric_values, dict):
        server_error_rate = metric_values.get("ax_runtime_error_rate")
        if (
            isinstance(server_error_rate, (int, float))
            and not isinstance(server_error_rate, bool)
            and math.isfinite(float(server_error_rate))
            and float(server_error_rate) > max_client_error_rate
        ):
            performance_alerts.append(
                f"server error rate {float(server_error_rate):.4%} exceeds "
                f"{max_client_error_rate:.4%} guardrail"
            )
        deltas = counter_deltas(metric_values, state.counter_baseline)
        for counter in (
            "ax_engine_generation_saturated_commands_total",
            "ax_engine_generation_stream_backlog_overflows_total",
            "ax_engine_http_status_5xx_total",
        ):
            if deltas.get(counter, 0.0) > 0.0:
                performance_alerts.append(
                    f"server counter {counter} increased by {deltas[counter]:.0f}"
                )

    memory = memory_analysis(
        samples=resource_samples,
        resource_baseline=state.resource_baseline,
        window_start_elapsed_s=window_start_elapsed_s,
        baseline_end_elapsed_s=state.baseline_completed_elapsed_s,
    )
    memory_alerts = evaluate_memory_alerts(
        analysis=memory,
        max_growth_mib=memory_growth_mib,
        max_slope_mib_per_hour=memory_slope_mib_per_hour,
        max_swap_growth_mib=max_swap_growth_mib,
    )
    memory_alerts.extend(
        evaluate_memory_early_warnings(
            memory,
            min_growth_mib=early_memory_growth_mib,
            min_slope_mib_per_hour=early_memory_slope_mib_per_hour,
            min_positive_step_ratio=early_memory_positive_step_ratio,
        )
    )
    return performance_alerts, memory_alerts


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Persist an inspectable JSON checkpoint without a partial visible file."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text_atomic(path: Path, content: str) -> None:
    """Atomically publish the human-readable checkpoint report."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append+fsync one evidence event; safe across workload and sampler threads."""
    with JSONL_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def assessment(
    *, state: RunState, terminal_status: str, baseline_required: bool = True
) -> tuple[str, list[str]]:
    """Create a conservative high-level verdict from independent evidence classes."""
    concerns: list[str] = []
    if state.requests_failed:
        concerns.append(f"{state.requests_failed} client stream request(s) failed")
    if state.health_failures:
        concerns.append(f"{state.health_failures} health check(s) failed")
    if state.lifecycle_timeouts:
        concerns.append(f"{state.lifecycle_timeouts} post-request lifecycle drain timeout(s)")
    if state.lifecycle_inconclusive:
        concerns.append(
            f"{state.lifecycle_inconclusive} lifecycle drain verdict(s) were inconclusive"
        )
    if state.kv_report_unavailable:
        concerns.append(
            f"{state.kv_report_unavailable} post-request native KV report(s) were unavailable"
        )
    if state.metric_scrape_failures:
        concerns.append(f"{state.metric_scrape_failures} metric scrape(s) failed")
    if state.resource_sampler_stop_timeouts:
        concerns.append(f"{state.resource_sampler_stop_timeouts} resource sampler stop timed out")
    concerns.extend(state.sampling_continuity_concerns)
    if state.quiescent_kv_logical_exceedances:
        concerns.append(
            f"{state.quiescent_kv_logical_exceedances} drained sample(s) exceeded "
            "the logical-KV guardrail"
        )
    if state.performance_alerts:
        concerns.append(f"{state.performance_alerts} performance-regression alert(s)")
    if state.memory_alerts:
        concerns.append(f"{state.memory_alerts} persistent-memory-growth alert(s)")
    if terminal_status == "preflight_passed":
        return ("watch" if concerns else "pass"), concerns
    if not state.measurement_started:
        concerns.append(
            "endurance preflight did not pass; the 72-hour measured interval never started"
        )
    if terminal_status == "failed":
        if state.last_error:
            concerns.append(f"terminal failure: {state.last_error}")
        return "fail", concerns
    if terminal_status == "interrupted":
        concerns.append("run was interrupted before the target duration")
    if baseline_required and state.baseline is None:
        concerns.append("performance/memory baseline not yet complete")
    concerns.extend(state.baseline_coverage_concerns)
    concerns.extend(state.baseline_stability_alerts)
    if concerns:
        return "watch", concerns
    return "pass", concerns


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
    memory: dict[str, Any],
    counter_deltas_view: dict[str, float],
    performance_alerts: list[str],
    memory_alerts: list[str],
    alerts: list[str],
    output_dir: Path,
    bounded_state_trends: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a durable current status used for both automation and operator review."""
    verdict, concerns = assessment(state=state, terminal_status=status)
    measurement_elapsed_s = elapsed_s if state.measurement_started else 0.0
    metric_values = latest_metrics.get("values", {}) if isinstance(latest_metrics, dict) else {}
    metric_values = metric_values if isinstance(metric_values, dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "verdict": verdict,
        "updated_at": utc_now(),
        # This counter represents only the configured 72-hour evidence
        # interval.  A failed capacity rehearsal remains visible below as
        # preflight time instead of looking like a very short endurance run.
        "elapsed_seconds": measurement_elapsed_s,
        "target_duration_seconds": target_duration_s,
        "target_end_at": (
            (
                dt.datetime.fromisoformat(state.started_wall)
                + dt.timedelta(seconds=target_duration_s)
            ).isoformat()
            if state.measurement_started
            else None
        ),
        "server": latest_server,
        "requests": {
            "attempted": state.requests_attempted,
            "successful": state.requests_ok,
            "failed": state.requests_failed,
            "client_error_rate": state.requests_failed / state.requests_attempted
            if state.requests_attempted
            else 0.0,
            "health_failures": state.health_failures,
            "consecutive_request_failures": state.consecutive_request_failures,
        },
        "lifecycle": {
            "drain_timeouts": state.lifecycle_timeouts,
            "inconclusive_drains": state.lifecycle_inconclusive,
            "kv_reports_unavailable": state.kv_report_unavailable,
            "quiescent_kv_logical_exceedances": state.quiescent_kv_logical_exceedances,
        },
        "preflight": {
            "status": state.preflight_status,
            "elapsed_seconds": state.preflight_elapsed_seconds,
            "measurement_started": state.measurement_started,
            "cache_capacity_rehearsal": state.cache_capacity_rehearsal,
        },
        "resource_sampler_stop_timeouts": state.resource_sampler_stop_timeouts,
        "observation_continuity": {
            "concerns": state.sampling_continuity_concerns,
        },
        "baseline": {
            "completed_at": state.baseline_completed_at,
            "performance": state.baseline,
            "resource_medians_bytes": state.resource_baseline,
            "coverage_concerns": state.baseline_coverage_concerns,
            "stability_alerts": state.baseline_stability_alerts,
        },
        "latest_window": latest_window,
        "memory": memory,
        "metrics": latest_metrics,
        "metric_counter_deltas_since_warmup": counter_deltas_view,
        "bounded_state": {
            name: metric_values[name]
            for name in BOUNDED_STATE_METRICS
            if isinstance(metric_values.get(name), (int, float))
        },
        "bounded_state_trends": bounded_state_trends or {},
        "host": latest_host,
        "performance_alerts": performance_alerts,
        "memory_alerts": memory_alerts,
        "assessment_concerns": concerns,
        "alerts": alerts[-50:],
        "last_error": state.last_error,
        "artifacts": {
            "output_dir": str(output_dir),
            "events": str(output_dir / "events.jsonl"),
            "server_log": str(output_dir / "server.log"),
            "summary": str(output_dir / "summary.json"),
            "checkpoints_dir": str(output_dir / "checkpoints"),
            "reports_dir": str(output_dir / "reports"),
            "diagnostics_dir": str(output_dir / "diagnostics"),
        },
    }


def render_checkpoint_markdown(reason: str, summary: dict[str, Any]) -> str:
    """Render a short status readout suitable for an every-four-hour handoff."""
    requests = summary["requests"]
    window = summary["latest_window"]
    overall = window.get("overall", {}) if isinstance(window, dict) else {}
    ttft = get_distribution_value(overall, "ttft_ms", "p95") if isinstance(overall, dict) else None
    decode = (
        get_distribution_value(overall, "client_decode_tok_s", "p05")
        if isinstance(overall, dict)
        else None
    )
    prefill = (
        get_distribution_value(overall, "effective_prefill_tok_s", "p05")
        if isinstance(overall, dict)
        else None
    )
    prefix_cache_evidence = (
        window.get("shared_prefix_cache_route_evidence", {}) if isinstance(window, dict) else {}
    )
    prefix_cache_text = (
        ", ".join(f"{key}={value:g}" for key, value in prefix_cache_evidence.items())
        if isinstance(prefix_cache_evidence, dict) and prefix_cache_evidence
        else "n/a"
    )
    baseline = summary["baseline"]
    baseline_concerns = [
        *baseline.get("coverage_concerns", []),
        *baseline.get("stability_alerts", []),
    ]
    continuity = summary.get("observation_continuity", {})
    continuity_concerns = continuity.get("concerns", []) if isinstance(continuity, dict) else []
    preflight = summary.get("preflight", {})
    rehearsal = preflight.get("cache_capacity_rehearsal", []) if isinstance(preflight, dict) else []
    preflight_status = (
        preflight.get("status", "unknown") if isinstance(preflight, dict) else "unknown"
    )
    measurement_started = (
        bool(preflight.get("measurement_started")) if isinstance(preflight, dict) else True
    )
    preflight_elapsed = preflight.get("elapsed_seconds") if isinstance(preflight, dict) else None
    rehearsal_text = "n/a"
    if isinstance(rehearsal, list) and rehearsal:
        parts = []
        for probe in rehearsal:
            if not isinstance(probe, dict):
                continue
            shape = probe.get("shape", "unknown")
            phase = probe.get("phase", "cache_capacity_probe")
            verdict = probe.get("verdict", "unknown")
            progress = probe.get("prefill_progress", {})
            capacity = probe.get("logical_kv", {})
            pressure = probe.get("capacity_pressure", {})
            if not isinstance(progress, dict):
                progress = {}
            if not isinstance(capacity, dict):
                capacity = {}
            if not isinstance(pressure, dict):
                pressure = {}
            ratio = format_metric(progress.get("prefill_steps_per_1k_tokens"), "/1k tokens")
            utilization_value = capacity.get("utilization")
            utilization = (
                f"{float(utilization_value):.1%}"
                if isinstance(utilization_value, (int, float))
                and math.isfinite(float(utilization_value))
                else "n/a"
            )
            if phase == "cache_capacity_anchor":
                free_blocks = format_metric(pressure.get("free_blocks"), "blocks")
                fresh_blocks = format_metric(pressure.get("fresh_prompt_blocks"), "blocks")
                reclamation = pressure.get("reclamation_required")
                reclamation_text = (
                    "required"
                    if reclamation is True
                    else "not yet required"
                    if reclamation is False
                    else "unknown"
                )
                parts.append(
                    f"anchor/{shape} ({verdict}): free {free_blocks}, "
                    f"fresh-long {fresh_blocks}, reclamation {reclamation_text}, "
                    f"KV utilization {utilization}"
                )
            else:
                parts.append(f"{phase}/{shape} ({verdict}): {ratio}, KV utilization {utilization}")
        if parts:
            rehearsal_text = "; ".join(parts)
    timing_line = (
        f"- Elapsed: `{summary['elapsed_seconds'] / 3600.0:.2f} h` / "
        f"`{summary['target_duration_seconds'] / 3600.0:.2f} h`"
        if measurement_started
        else (
            f"- Measurement: `not started`; preflight status: `{preflight_status}`; "
            f"preflight elapsed: `{format_metric(preflight_elapsed, 's')}`"
        )
    )
    lines = [
        f"# AXQ endurance checkpoint: {summary['status']}",
        "",
        f"- Reason: `{reason}`",
        f"- Updated: `{summary['updated_at']}`",
        timing_line,
        f"- Server PID alive: `{summary['server'].get('alive')}` "
        f"(PID `{summary['server'].get('pid')}`)",
        f"- Requests: `{requests['successful']}/{requests['attempted']}` successful; "
        f"client error rate `{requests['client_error_rate']:.4%}`",
        f"- Window p95 TTFT: `{format_metric(ttft, 'ms')}`; "
        f"p05 decode: `{format_metric(decode, 'tok/s')}`; "
        f"p05 effective prefill: `{format_metric(prefill, 'tok/s')}`",
        f"- Shared-prefix cache route evidence: `{prefix_cache_text}`",
        f"- Post-warm-up cache-capacity rehearsal: `{rehearsal_text}`",
        f"- Baseline: `{'complete' if baseline['performance'] else 'pending'}`; "
        f"quality: `{'ready' if baseline['performance'] and not baseline_concerns else 'watch'}`",
        f"- Observation continuity: `{'watch' if continuity_concerns else 'continuous'}`",
        "",
        "## Assessment",
        "",
    ]
    concerns = summary.get("assessment_concerns", [])
    if concerns:
        lines.extend(f"- {concern}" for concern in concerns)
    else:
        lines.append("- No current endurance concern was detected by configured guardrails.")
    by_shape = window.get("by_shape", {}) if isinstance(window, dict) else {}
    if isinstance(by_shape, dict) and by_shape:
        lines.extend(
            [
                "",
                "## Per-shape serving metrics",
                "",
                "| Shape | Successful / attempted | p95 TTFT | p05 decode | "
                "p05 effective prefill |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for shape in WORKLOAD_SHAPES:
            shape_summary = by_shape.get(shape)
            if not isinstance(shape_summary, dict):
                continue
            successful = int(shape_summary.get("successful_requests", 0))
            attempted = int(shape_summary.get("requests", 0))
            shape_ttft = format_metric(
                get_distribution_value(shape_summary, "ttft_ms", "p95"), "ms"
            )
            shape_decode = format_metric(
                get_distribution_value(shape_summary, "client_decode_tok_s", "p05"), "tok/s"
            )
            shape_prefill = format_metric(
                get_distribution_value(shape_summary, "effective_prefill_tok_s", "p05"),
                "tok/s",
            )
            lines.append(
                f"| `{shape}` | {successful} / {attempted} | "
                f"{shape_ttft} | {shape_decode} | {shape_prefill} |"
            )
    memory_series = summary.get("memory", {}).get("series", {})
    if isinstance(memory_series, dict) and memory_series:
        lines.extend(["", "## Memory trend", ""])
        for name, values in memory_series.items():
            if not isinstance(values, dict):
                continue
            growth = values.get("quiescent_growth_mib", values.get("growth_mib"))
            slope = values.get(
                "quiescent_lifetime_slope_mib_per_hour",
                values.get("lifetime_slope_mib_per_hour"),
            )
            lines.append(
                f"- `{name}`: current `{format_bytes(values.get('current_bytes'))}`; "
                f"quiescent current `{format_bytes(values.get('quiescent_current_bytes'))}`; "
                f"growth `{format_metric(growth, 'MiB')}`; "
                f"slope `{format_metric(slope, 'MiB/h')}`"
            )
        normalized_lines = []
        for name in ("server_rss_bytes", "mlx_unattributed_active_bytes"):
            values = memory_series.get(name)
            if not isinstance(values, dict):
                continue
            normalized = values.get("work_normalized", {})
            if not isinstance(normalized, dict):
                continue
            requests = normalized.get("completed_requests")
            allocations = normalized.get("kv_allocated_blocks")
            request_rate = (
                requests.get("ols_bytes_per_unit") if isinstance(requests, dict) else None
            )
            allocation_rate = (
                allocations.get("ols_bytes_per_unit") if isinstance(allocations, dict) else None
            )
            normalized_lines.append(
                f"- `{name}`: "
                f"`{format_metric_per_unit(request_rate, 'KiB/completed request')}`; "
                f"`{format_metric_per_unit(allocation_rate, 'KiB/KV block allocation')}`"
            )
        if normalized_lines:
            lines.extend(["", "## Work-normalized retained-memory diagnostics", ""])
            lines.extend(normalized_lines)
    bounded_state = summary.get("bounded_state", {})
    if isinstance(bounded_state, dict) and bounded_state:
        lines.extend(["", "## Bounded internal state", ""])
        lines.extend(f"- `{name}`: `{value:g}`" for name, value in bounded_state.items())
    bounded_trends = summary.get("bounded_state_trends", {})
    if isinstance(bounded_trends, dict) and bounded_trends:
        lines.extend(["", "## Bounded-state trajectories", ""])
        for name, values in bounded_trends.items():
            if not isinstance(values, dict):
                continue
            lines.append(
                f"- `{name}`: growth "
                f"`{format_metric(values.get('post_baseline_growth'), 'units')}`; "
                f"slope `"
                f"{format_metric(values.get('post_baseline_slope_per_hour'), 'units/h')}`; "
                f"per completed request `"
                f"{format_metric(values.get('ols_units_per_completed_request'), 'units/request')}`"
            )
    lines.append("")
    return "\n".join(lines)


def format_metric(value: Any, unit: str) -> str:
    """Render an optional finite metric for the human checkpoint."""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.2f} {unit}"


def format_bytes(value: Any) -> str:
    """Render an optional byte value in MiB."""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value) / MEBIBYTE:.1f} MiB"


def format_metric_per_unit(value: Any, unit: str) -> str:
    """Render a byte-per-work-unit slope as KiB per unit."""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value) / 1024.0:.3f} {unit}"


def write_checkpoint(
    *,
    output_dir: Path,
    reason: str,
    state: RunState,
    status: str,
    elapsed_s: float,
    target_duration_s: float,
    records: list[dict[str, Any]],
    window_elapsed_s: float,
    latest_server: dict[str, Any],
    latest_host: dict[str, Any],
    latest_metrics: dict[str, Any],
    resource_samples: list[dict[str, Any]],
    window_start_elapsed_s: float,
    performance_alerts: list[str],
    memory_alerts: list[str],
    alerts: list[str],
    max_sampling_gap_seconds: float = DEFAULT_MAX_SAMPLING_GAP_S,
) -> dict[str, Any]:
    """Publish immutable JSON/Markdown checkpoint and update mutable summary.json."""
    for concern in evaluate_sampling_continuity(
        resource_samples,
        max_gap_seconds=max_sampling_gap_seconds,
    ):
        if concern not in state.sampling_continuity_concerns:
            state.sampling_continuity_concerns.append(concern)
            add_alert(alerts, concern)
    window = summarize_requests(records, elapsed_s=max(window_elapsed_s, 0.001))
    values = latest_metrics.get("values", {}) if isinstance(latest_metrics, dict) else {}
    values = values if isinstance(values, dict) else {}
    memory = memory_analysis(
        samples=resource_samples,
        resource_baseline=state.resource_baseline,
        window_start_elapsed_s=window_start_elapsed_s,
        baseline_end_elapsed_s=state.baseline_completed_elapsed_s,
    )
    bounded_trends = bounded_state_analysis(
        samples=resource_samples,
        baseline_end_elapsed_s=state.baseline_completed_elapsed_s,
    )
    summary = run_summary(
        state=state,
        status=status,
        elapsed_s=elapsed_s,
        target_duration_s=target_duration_s,
        latest_window=window,
        latest_server=latest_server,
        latest_host=latest_host,
        latest_metrics=latest_metrics,
        memory=memory,
        counter_deltas_view=counter_deltas(values, state.counter_baseline),
        performance_alerts=performance_alerts,
        memory_alerts=memory_alerts,
        alerts=alerts,
        output_dir=output_dir,
        bounded_state_trends=bounded_trends,
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "reason": reason,
        "summary": summary,
    }
    timestamp = utc_now().replace(":", "").replace("+00:00", "Z")
    checkpoint_name = f"{timestamp}-{reason}"
    write_json_atomic(output_dir / "checkpoints" / f"{checkpoint_name}.json", checkpoint)
    write_text_atomic(
        output_dir / "reports" / f"{checkpoint_name}.md",
        render_checkpoint_markdown(reason, summary),
    )
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def prepare_output_dir(path: Path) -> None:
    """Refuse non-empty evidence directories so a run can never overwrite another."""
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    (path / "reports").mkdir(exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    """Validate exact local inputs before creating a multi-day server process."""
    args.server = args.server.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.server.is_file() or not os.access(args.server, os.X_OK):
        raise FileNotFoundError(f"server binary is not executable: {args.server}")
    validate_server_version(args.server, args.expected_server_version)
    model_identity(args.model_dir)
    if not 0 < args.port < 65_536:
        raise ValueError("--port must be in 1..65535")
    if args.baseline_hours >= args.duration_hours:
        raise ValueError("--baseline-hours must be smaller than --duration-hours")
    if args.early_memory_positive_step_ratio > 1.0:
        raise ValueError("--early-memory-positive-step-ratio must be at most 1")
    if args.warmup_requests < len(WORKLOAD_SEQUENCE):
        raise ValueError(
            "--warmup-requests must cover one complete workload cycle "
            f"({len(WORKLOAD_SEQUENCE)} requests)"
        )
    assert_port_available(args.host, args.port)
    prepare_output_dir(args.output_dir)


def add_alert(alerts: list[str], message: str) -> None:
    """Retain bounded deduplicated alerts in the summary while JSONL keeps all events."""
    if not alerts or alerts[-1] != message:
        alerts.append(message)
    if len(alerts) > 500:
        del alerts[:-500]


def maybe_finalize_baseline(
    *,
    state: RunState,
    baseline_records: list[dict[str, Any]],
    resource_samples: list[dict[str, Any]],
    elapsed_s: float,
    baseline_s: float,
    min_performance_samples: int,
    baseline_stability_growth_mib: float,
    baseline_stability_slope_mib_per_hour: float,
    max_swap_growth_mib: float,
) -> bool:
    """Finalize stable references after the configured warm baseline window."""
    if state.baseline is not None or elapsed_s < baseline_s:
        return False
    state.baseline = summarize_requests(baseline_records, elapsed_s=baseline_s)
    state.resource_baseline = build_resource_baseline(resource_samples, baseline_s)
    state.baseline_coverage_concerns = evaluate_baseline_coverage(
        state.baseline,
        min_samples=min_performance_samples,
    )
    state.baseline_stability_alerts = evaluate_baseline_stability(
        samples=resource_samples,
        baseline_s=baseline_s,
        baseline_growth_mib=baseline_stability_growth_mib,
        max_slope_mib_per_hour=baseline_stability_slope_mib_per_hour,
        max_swap_growth_mib=max_swap_growth_mib,
    )
    state.baseline_completed_at = utc_now()
    state.baseline_completed_elapsed_s = elapsed_s
    return True


def guard_lifecycle(
    *, state: RunState, drain: dict[str, Any], max_quiescent_kv_logical_mib: float
) -> list[str]:
    """Turn post-request lifecycle/KV evidence into conservative diagnostics."""
    messages: list[str] = []
    drain_state = drain.get("state")
    if drain_state == "timeout":
        state.lifecycle_timeouts += 1
        messages.append(f"post-request lifecycle drain timed out with {drain.get('nonzero', {})}")
    elif drain_state == "inconclusive":
        state.lifecycle_inconclusive += 1
        messages.append(
            f"post-request lifecycle drain is inconclusive; missing {drain.get('missing', [])}"
        )
    metrics = drain.get("metrics", {})
    values = metrics.get("values", {}) if isinstance(metrics, dict) else {}
    report_available = (
        values.get("ax_engine_model_memory_kv_report_available")
        if isinstance(values, dict)
        else None
    )
    if report_available != 1.0:
        state.kv_report_unavailable += 1
        messages.append("post-request native model KV memory report is unavailable")
    logical = (
        values.get("ax_engine_model_memory_kv_logical_bytes") if isinstance(values, dict) else None
    )
    if isinstance(logical, (int, float)) and logical > max_quiescent_kv_logical_mib * MEBIBYTE:
        state.quiescent_kv_logical_exceedances += 1
        messages.append(
            "logical model KV remained "
            f"{float(logical) / MEBIBYTE:.1f} MiB after lifecycle drain "
            f"(guardrail {max_quiescent_kv_logical_mib:.1f} MiB)"
        )
    return messages


def warmup_preflight_concerns(
    *, observation: dict[str, Any], drain: dict[str, Any], max_quiescent_kv_logical_mib: float
) -> list[str]:
    """Require the probes and observability needed for a meaningful measured run.

    Warm-up is the last safe point to discover that the exact server cannot
    produce the native token/KV/lifecycle evidence required by this soak.  A
    72-hour request stream without that evidence would show liveness, but
    could not answer the retention and prefill questions the run is intended
    to resolve.
    """
    concerns: list[str] = []
    if not observation.get("ok"):
        concerns.append(f"stream failed: {observation.get('error')}")
    prompt_tokens = observation.get("prompt_token_count")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens <= 0:
        concerns.append("native prompt token count was unavailable")
    sent_tokens = observation.get("client_input_token_count")
    if not isinstance(sent_tokens, int) or isinstance(sent_tokens, bool) or sent_tokens <= 0:
        concerns.append("client pre-tokenized input count was unavailable")
    elif isinstance(prompt_tokens, int) and prompt_tokens > 0 and prompt_tokens != sent_tokens:
        concerns.append(
            f"native prompt token count {prompt_tokens} did not match sent token count "
            f"{sent_tokens}"
        )

    drain_state = drain.get("state")
    if drain_state != "drained":
        concerns.append(
            f"post-response lifecycle did not drain ({drain_state}; "
            f"missing={drain.get('missing', [])}; nonzero={drain.get('nonzero', {})})"
        )
    metrics = drain.get("metrics", {})
    values = metrics.get("values", {}) if isinstance(metrics, dict) else {}
    if not isinstance(values, dict):
        values = {}
    if values.get("ax_engine_model_memory_kv_report_available") != 1.0:
        concerns.append("native model KV memory report was unavailable")
    missing_model_memory = metrics.get("missing_model_memory_metrics", [])
    if isinstance(missing_model_memory, list) and missing_model_memory:
        concerns.append(f"required model-memory metrics were missing: {missing_model_memory}")
    missing_retention = [
        name
        for name in RETENTION_OBSERVABILITY_METRICS
        if not isinstance(values.get(name), (int, float))
    ]
    if missing_retention:
        concerns.append(
            "required retention/churn metrics were missing: " + ", ".join(missing_retention)
        )
    logical = values.get("ax_engine_model_memory_kv_logical_bytes")
    if isinstance(logical, (int, float)) and logical > max_quiescent_kv_logical_mib * MEBIBYTE:
        concerns.append(
            "logical model KV remained "
            f"{float(logical) / MEBIBYTE:.1f} MiB after lifecycle drain "
            f"(guardrail {max_quiescent_kv_logical_mib:.1f} MiB)"
        )
    return concerns


def prefill_progress_evidence(observation: dict[str, Any]) -> dict[str, float | None]:
    """Extract scheduler-fragmentation evidence from one completed fresh prompt.

    A request can legitimately be divided into a few prefill items.  The
    relevant failure signature is a ratio near one item per prompt token, not
    a nonzero continuation count.  Route telemetry is attached to the native
    terminal response, so it survives independently of periodic metric
    sampling and makes the capacity rehearsal auditable after a failed run.
    """
    prompt_tokens = observation.get("prompt_token_count")
    route = observation.get("route_decisions")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens <= 0:
        prompt_tokens = None
    if not isinstance(route, dict):
        route = {}

    def number(name: str) -> float | None:
        value = route.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric) and numeric >= 0.0:
                return numeric
        return None

    steps = number("ax_mlx_prefill_steps")
    continuations = number("ax_mlx_prefill_cache_only_continuations")
    per_1k = (1_000.0 / prompt_tokens) if prompt_tokens is not None else None
    return {
        "prompt_tokens": float(prompt_tokens) if prompt_tokens is not None else None,
        "prefill_steps": steps,
        "cache_only_continuations": continuations,
        "prefill_steps_per_1k_tokens": steps * per_1k
        if steps is not None and per_1k is not None
        else None,
        "cache_only_continuations_per_1k_tokens": continuations * per_1k
        if continuations is not None and per_1k is not None
        else None,
    }


def prefill_progress_concerns(
    *, observation: dict[str, Any], max_steps_per_1k_tokens: float
) -> list[str]:
    """Fail closed when a fresh capacity probe degrades into token-by-token prefill."""
    evidence = prefill_progress_evidence(observation)
    steps = evidence["prefill_steps"]
    continuations = evidence["cache_only_continuations"]
    steps_per_1k = evidence["prefill_steps_per_1k_tokens"]
    continuations_per_1k = evidence["cache_only_continuations_per_1k_tokens"]
    if steps is None or continuations is None or steps_per_1k is None:
        return [
            "native prefill-progress telemetry was unavailable "
            "(need ax_mlx_prefill_steps and ax_mlx_prefill_cache_only_continuations)"
        ]
    if steps_per_1k > max_steps_per_1k_tokens or (
        continuations_per_1k is not None and continuations_per_1k > max_steps_per_1k_tokens
    ):
        return [
            "prefill fragmentation exceeded the cache-capacity guardrail: "
            f"steps={steps:.0f} ({steps_per_1k:.1f}/1k tokens), "
            f"cache-only continuations={continuations:.0f} "
            f"({continuations_per_1k or 0.0:.1f}/1k tokens), "
            f"guardrail={max_steps_per_1k_tokens:.1f}/1k tokens"
        ]
    return []


def cache_capacity_evidence(drain: dict[str, Any]) -> dict[str, float | None]:
    """Return the logical-KV occupancy evidence collected after a capacity probe."""
    metrics = drain.get("metrics", {})
    values = metrics.get("values", {}) if isinstance(metrics, dict) else {}
    values = values if isinstance(values, dict) else {}

    def number(name: str) -> float | None:
        value = values.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
        return None

    return {
        "used_blocks": number("ax_engine_step_kv_usage_blocks"),
        "total_blocks": number("ax_runtime_kv_pages_total"),
        "utilization": number("ax_runtime_kv_utilization"),
    }


def cache_capacity_telemetry_concerns(evidence: dict[str, float | None]) -> list[str]:
    """Validate the core logical-KV gauges required to interpret a rehearsal."""
    concerns: list[str] = []
    missing = [name for name, value in evidence.items() if value is None]
    if missing:
        concerns.append(f"cache-capacity telemetry was unavailable: {', '.join(missing)}")
    total = evidence["total_blocks"]
    used = evidence["used_blocks"]
    utilization = evidence["utilization"]
    if total is not None and total <= 0.0:
        concerns.append(f"cache-capacity telemetry reported invalid total_blocks={total:g}")
    if total is not None and used is not None and total > 0.0 and (used < 0.0 or used > total):
        concerns.append(
            "cache-capacity telemetry was internally inconsistent: "
            f"used_blocks={used:g}, total_blocks={total:g}"
        )
    if utilization is not None and not 0.0 <= utilization <= 1.0:
        concerns.append(f"cache-capacity telemetry reported invalid utilization={utilization:g}")
    return concerns


def cache_capacity_pressure_evidence(
    *,
    drain: dict[str, Any],
    fresh_prompt_tokens: int,
    block_size_tokens: int,
) -> dict[str, float | bool | None]:
    """Show whether the next fresh long prompt must reclaim logical KV blocks.

    A successful request at low occupancy says nothing about the cache-full
    path.  The rehearsal therefore records the number of free logical blocks
    before its fresh long prompt and requires that the prompt's own block
    demand exceed that free space.  This is a conservative pressure proof;
    it does not assume an intentionally warm cache is a leak.
    """
    evidence = cache_capacity_evidence(drain)
    prompt_blocks = (
        float(math.ceil(fresh_prompt_tokens / block_size_tokens))
        if fresh_prompt_tokens > 0 and block_size_tokens > 0
        else None
    )
    used = evidence["used_blocks"]
    total = evidence["total_blocks"]
    free_blocks = total - used if total is not None and used is not None else None
    reclamation_required = (
        used + prompt_blocks > total
        if used is not None and prompt_blocks is not None and total is not None and total > 0.0
        else None
    )
    return {
        **evidence,
        "fresh_prompt_tokens": float(fresh_prompt_tokens),
        "fresh_prompt_blocks": prompt_blocks,
        "free_blocks": free_blocks,
        "reclamation_required": reclamation_required,
    }


def cache_capacity_probe_concerns(
    *,
    observation: dict[str, Any],
    drain: dict[str, Any],
    max_quiescent_kv_logical_mib: float,
    max_prefill_steps_per_1k_tokens: float,
) -> list[str]:
    """Combine ordinary warm-up checks with fresh-cache pressure evidence.

    High logical-KV utilization is intentionally *not* an automatic failure:
    a cache is allowed to fill and reclaim.  The gate is whether a fresh prompt
    still makes productive prefill progress and whether the occupancy telemetry
    needed to interpret that result exists.
    """
    concerns = warmup_preflight_concerns(
        observation=observation,
        drain=drain,
        max_quiescent_kv_logical_mib=max_quiescent_kv_logical_mib,
    )
    concerns.extend(
        prefill_progress_concerns(
            observation=observation,
            max_steps_per_1k_tokens=max_prefill_steps_per_1k_tokens,
        )
    )
    concerns.extend(cache_capacity_telemetry_concerns(cache_capacity_evidence(drain)))
    return concerns


def run_endurance(args: argparse.Namespace) -> int:
    """Convert SIGTERM into a normal interrupted run with a final checkpoint."""
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        return _run_endurance(args)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def _run_endurance(args: argparse.Namespace) -> int:
    """Launch one server and exercise it continuously until duration or hard failure."""
    validate_args(args)
    tokenize_prompt = load_local_tokenizer(args.model_dir)
    base_url = f"http://{args.host}:{args.port}"
    target_duration_s = args.duration_hours * 3_600.0
    report_interval_s = args.report_interval_hours * 3_600.0
    baseline_s = args.baseline_hours * 3_600.0
    max_sampling_gap_s = (
        args.max_sampling_gap_seconds
        if args.max_sampling_gap_seconds is not None
        else default_max_sampling_gap_seconds(args.resource_interval_seconds)
    )
    output_dir = args.output_dir
    server_command = build_server_command(args)
    server_log = output_dir / "server.log"
    events_path = output_dir / "events.jsonl"
    process: subprocess.Popen[bytes] | None = None
    log_handle: Any = None
    sampler: ResourceSampler | None = None
    state: RunState | None = None
    status = "failed"
    failure: str | None = None
    alerts: list[str] = []
    baseline_records: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    latest_server: dict[str, Any] = {}
    latest_host: dict[str, Any] = {}
    latest_metrics: dict[str, Any] = {}
    last_checkpoint_monotonic: float | None = None
    last_performance_alerts: list[str] = []
    last_memory_alerts: list[str] = []

    try:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "methodology": {
                "scope": "low-rate no-restart AX Engine MLX endurance/soak",
                "server_lifetime": "one owned process; automatic restart is forbidden",
                "concurrency": 1,
                "max_batch_tokens": args.max_batch_tokens,
                "block_size_tokens": args.block_size_tokens,
                "total_blocks": args.total_blocks,
                "max_capacity_fill_requests": args.max_capacity_fill_requests,
                "minimum_idle_seconds_after_request": args.request_interval_seconds,
                "resource_sample_seconds": args.resource_interval_seconds,
                "workload_mix": (
                    "20-request interleaved cycle: 14 short unique, 3 medium unique, "
                    "2 shared-prefix, 1 long unique"
                ),
                "cache_capacity_rehearsal": (
                    "after one complete warm-up cycle, fresh long and medium unique prompts "
                    "must reclaim logical-KV capacity without token-by-token prefill"
                ),
                "input_encoding": (
                    "client-local tokenizer.json with add_special_tokens=false; "
                    "native MLX input_tokens only"
                ),
                "stream_validation": "HTTP success, terminal response, non-empty output",
                "lifecycle_validation": "post-response native lifecycle gauges must drain to zero",
                "cache_interpretation": (
                    "non-zero KV capacity/paged pool/prefix-cache is not itself a leak; "
                    "the test evaluates drain state, post-baseline growth, and slope"
                ),
                "performance_validation": (
                    "same-shape client p95 TTFT plus p05 decode/effective-prefill token/s "
                    "compared with the first baseline window; server runtime gauges are "
                    "retained as corroboration"
                ),
                "reporting": "atomic current summary plus immutable JSON and Markdown checkpoints",
                "retention_diagnostics": (
                    "request/KV bounded-state gauges, allocation/release counters, "
                    "work-normalized memory slopes, and vmmap/footprint checkpoint captures"
                ),
                "detached_execution": (
                    "runner records its process/session/TTY context; use the repository "
                    "detached launcher so the run does not depend on an SSH client"
                ),
            },
            "target": {
                "model_id": args.model_id,
                "model": model_identity(args.model_dir),
                "server_command": server_command,
                "base_url": base_url,
            },
            "guardrails": {
                "duration_hours": args.duration_hours,
                "baseline_hours": args.baseline_hours,
                "report_interval_hours": args.report_interval_hours,
                "request_timeout_seconds": args.request_timeout_seconds,
                "drain_timeout_seconds": args.drain_timeout_seconds,
                "max_consecutive_request_failures": args.max_consecutive_request_failures,
                "max_client_error_rate": args.max_client_error_rate,
                "max_ttft_p95_ratio": args.max_ttft_p95_ratio,
                "min_decode_p05_ratio": args.min_decode_p05_ratio,
                "min_prefill_p05_ratio": args.min_prefill_p05_ratio,
                "min_performance_samples": args.min_performance_samples,
                "memory_growth_mib": args.memory_growth_mib,
                "memory_slope_mib_per_hour": args.memory_slope_mib_per_hour,
                "early_memory_growth_mib": args.early_memory_growth_mib,
                "early_memory_slope_mib_per_hour": args.early_memory_slope_mib_per_hour,
                "early_memory_positive_step_ratio": args.early_memory_positive_step_ratio,
                "baseline_stability_growth_mib": args.baseline_stability_growth_mib,
                "baseline_stability_slope_mib_per_hour": (
                    args.baseline_stability_slope_mib_per_hour
                ),
                "max_swap_growth_mib": args.max_swap_growth_mib,
                "max_quiescent_kv_logical_mib": args.max_quiescent_kv_logical_mib,
                "max_prefill_steps_per_1k_tokens": args.max_prefill_steps_per_1k_tokens,
                "max_capacity_fill_requests": args.max_capacity_fill_requests,
                "max_sampling_gap_seconds": max_sampling_gap_s,
                "preflight_only": args.preflight_only,
            },
            "pre_server_host": collect_checkpoint_host_snapshot(output_dir),
            "runtime": runtime_metadata(args.server),
        }
        write_json_atomic(output_dir / "manifest.json", manifest)

        log_handle = server_log.open("wb")
        process = subprocess.Popen(
            server_command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
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
        # Create a state before warm-up so a preflight rejection receives the
        # same durable summary/checkpoint treatment as a measured-run failure.
        # Its clock is reset only after all preflight gates pass.
        state = RunState(
            started_wall=utc_now(),
            started_monotonic=time.monotonic(),
            server_pid=process.pid,
            preflight_status="running",
            measurement_started=False,
        )

        for warmup_index in range(args.warmup_requests):
            shape = select_shape(warmup_index + 1)
            warmup_request_index = 1_000_000 + warmup_index + 1
            observation = run_stream_request(
                prompt=make_prompt_item(shape, warmup_request_index, tokenize=tokenize_prompt),
                model_id=args.model_id,
                base_url=base_url,
                timeout_s=args.request_timeout_seconds,
            )
            observation["phase"] = "warmup"
            drain = wait_for_quiescence(
                base_url=base_url,
                model_id=args.model_id,
                timeout_s=args.drain_timeout_seconds,
            )
            append_jsonl(
                events_path,
                {
                    "timestamp": utc_now(),
                    "kind": "warmup_request",
                    "request_index": warmup_index + 1,
                    "prompt_request_index": warmup_request_index,
                    "shape": shape.name,
                    "request": observation,
                    "lifecycle": drain,
                },
            )
            preflight_concerns = warmup_preflight_concerns(
                observation=observation,
                drain=drain,
                max_quiescent_kv_logical_mib=args.max_quiescent_kv_logical_mib,
            )
            if preflight_concerns:
                raise RuntimeError(
                    f"warmup request {warmup_index + 1} did not satisfy the endurance "
                    f"preflight: {'; '.join(preflight_concerns)}"
                )

        # A full warm-up can fill the logical KV cache while every warm-up
        # request still looks healthy. Prove that the fresh long probe will
        # need to reclaim logical KV capacity before using it as the gate.
        # This avoids treating a healthy low-occupancy long prompt as evidence
        # of the cache-full path that caused the pilot regression.
        capacity_probe_items = [
            (
                probe_index,
                WORKLOAD_SHAPES[shape_name],
                2_000_000 + probe_index,
                make_prompt_item(
                    WORKLOAD_SHAPES[shape_name],
                    2_000_000 + probe_index,
                    tokenize=tokenize_prompt,
                ),
            )
            for probe_index, shape_name in enumerate(CACHE_CAPACITY_PROBE_SHAPES, start=1)
        ]
        _, long_probe_shape, long_probe_request_index, long_probe = capacity_probe_items[0]
        for fill_index in range(args.max_capacity_fill_requests + 1):
            drain = wait_for_quiescence(
                base_url=base_url,
                model_id=args.model_id,
                timeout_s=args.drain_timeout_seconds,
            )
            logical_kv = cache_capacity_evidence(drain)
            pressure = cache_capacity_pressure_evidence(
                drain=drain,
                fresh_prompt_tokens=long_probe.input_tokens_count,
                block_size_tokens=args.block_size_tokens,
            )
            anchor_concerns: list[str] = []
            if drain.get("state") != "drained":
                anchor_concerns.append(
                    "post-warm-up cache-capacity anchor did not drain "
                    f"({drain.get('state')}; missing={drain.get('missing', [])}; "
                    f"nonzero={drain.get('nonzero', {})})"
                )
            anchor_concerns.extend(cache_capacity_telemetry_concerns(logical_kv))
            reclamation_required = pressure["reclamation_required"]
            if not anchor_concerns and reclamation_required is not True:
                if fill_index >= args.max_capacity_fill_requests:
                    anchor_concerns.append(
                        "cache-capacity rehearsal could not force logical-KV reclamation: "
                        f"used_blocks={pressure['used_blocks']}, "
                        f"free_blocks={pressure['free_blocks']}, "
                        f"fresh_prompt_blocks={pressure['fresh_prompt_blocks']}"
                    )
                anchor_verdict = "filling"
            else:
                anchor_verdict = "pressure_reached" if not anchor_concerns else "fail"
            anchor_record = {
                "phase": "cache_capacity_anchor",
                "fill_index": fill_index,
                "shape": long_probe_shape.name,
                "prompt_request_index": long_probe_request_index,
                "lifecycle": drain,
                "logical_kv": logical_kv,
                "capacity_pressure": pressure,
                "server": process_snapshot(process.pid),
                "host": collect_light_host_snapshot(output_dir),
                "verdict": anchor_verdict,
                "concerns": anchor_concerns,
            }
            state.cache_capacity_rehearsal.append(anchor_record)
            append_jsonl(
                events_path,
                {"timestamp": utc_now(), "kind": "cache_capacity_anchor", **anchor_record},
            )
            if anchor_concerns:
                raise RuntimeError(
                    "cache-capacity anchor did not satisfy the endurance preflight: "
                    f"{'; '.join(anchor_concerns)}"
                )
            if reclamation_required is True:
                break

            fill_request_index = 1_900_000 + fill_index + 1
            fill_shape = WORKLOAD_SHAPES["long_unique"]
            observation = run_stream_request(
                prompt=make_prompt_item(fill_shape, fill_request_index, tokenize=tokenize_prompt),
                model_id=args.model_id,
                base_url=base_url,
                timeout_s=args.request_timeout_seconds,
            )
            observation["phase"] = "cache_capacity_fill"
            drain = wait_for_quiescence(
                base_url=base_url,
                model_id=args.model_id,
                timeout_s=args.drain_timeout_seconds,
            )
            fill_concerns = cache_capacity_probe_concerns(
                observation=observation,
                drain=drain,
                max_quiescent_kv_logical_mib=args.max_quiescent_kv_logical_mib,
                max_prefill_steps_per_1k_tokens=args.max_prefill_steps_per_1k_tokens,
            )
            fill_record = {
                "phase": "cache_capacity_fill",
                "fill_index": fill_index + 1,
                "shape": fill_shape.name,
                "prompt_request_index": fill_request_index,
                "request": observation,
                "lifecycle": drain,
                "prefill_progress": prefill_progress_evidence(observation),
                "logical_kv": cache_capacity_evidence(drain),
                "server": process_snapshot(process.pid),
                "host": collect_light_host_snapshot(output_dir),
                "verdict": "pass" if not fill_concerns else "fail",
                "concerns": fill_concerns,
            }
            state.cache_capacity_rehearsal.append(fill_record)
            append_jsonl(
                events_path,
                {"timestamp": utc_now(), "kind": "cache_capacity_fill", **fill_record},
            )
            if fill_concerns:
                raise RuntimeError(
                    f"cache-capacity fill {fill_index + 1} did not satisfy the endurance "
                    f"preflight: {'; '.join(fill_concerns)}"
                )

        for probe_index, shape, probe_request_index, prompt in capacity_probe_items:
            observation = run_stream_request(
                prompt=prompt,
                model_id=args.model_id,
                base_url=base_url,
                timeout_s=args.request_timeout_seconds,
            )
            observation["phase"] = "cache_capacity_rehearsal"
            drain = wait_for_quiescence(
                base_url=base_url,
                model_id=args.model_id,
                timeout_s=args.drain_timeout_seconds,
            )
            server_snapshot = process_snapshot(process.pid)
            host_snapshot = collect_light_host_snapshot(output_dir)
            preflight_concerns = cache_capacity_probe_concerns(
                observation=observation,
                drain=drain,
                max_quiescent_kv_logical_mib=args.max_quiescent_kv_logical_mib,
                max_prefill_steps_per_1k_tokens=args.max_prefill_steps_per_1k_tokens,
            )
            rehearsal_record = {
                "phase": "cache_capacity_probe",
                "probe_index": probe_index,
                "shape": shape.name,
                "prompt_request_index": probe_request_index,
                "request": observation,
                "lifecycle": drain,
                "prefill_progress": prefill_progress_evidence(observation),
                "logical_kv": cache_capacity_evidence(drain),
                "server": server_snapshot,
                "host": host_snapshot,
                "verdict": "pass" if not preflight_concerns else "fail",
                "concerns": preflight_concerns,
            }
            state.cache_capacity_rehearsal.append(rehearsal_record)
            append_jsonl(
                events_path,
                {
                    "timestamp": utc_now(),
                    "kind": "cache_capacity_probe",
                    **rehearsal_record,
                },
            )
            if preflight_concerns:
                raise RuntimeError(
                    f"cache-capacity probe {probe_index} ({shape.name}) did not satisfy "
                    f"the endurance preflight: {'; '.join(preflight_concerns)}"
                )

        started_monotonic = time.monotonic()
        state.preflight_status = "passed"
        state.preflight_elapsed_seconds = started_monotonic - state.started_monotonic
        if args.preflight_only:
            raise PreflightComplete
        state.measurement_started = True
        state.started_wall = utc_now()
        state.started_monotonic = started_monotonic
        latest_metrics = collect_metrics(base_url, args.model_id, timeout_s=10.0)
        if not latest_metrics.get("ok"):
            state.metric_scrape_failures += 1
        values = latest_metrics.get("values", {})
        if isinstance(values, dict):
            state.counter_baseline = {
                name: float(values[name])
                for name in COUNTER_METRICS
                if isinstance(values.get(name), (int, float))
            }
        sampler = ResourceSampler(
            output_dir=output_dir,
            events_path=events_path,
            base_url=base_url,
            model_id=args.model_id,
            server_pid=process.pid,
            started_monotonic=started_monotonic,
            interval_s=args.resource_interval_seconds,
        )
        sampler.start()
        latest_server = process_snapshot(process.pid)
        latest_host = collect_checkpoint_host_snapshot(
            output_dir,
            server_pid=process.pid,
            reason="started",
        )
        next_checkpoint = started_monotonic + report_interval_s
        last_checkpoint_monotonic = started_monotonic
        write_checkpoint(
            output_dir=output_dir,
            reason="started",
            state=state,
            status="running",
            elapsed_s=0.0,
            target_duration_s=target_duration_s,
            records=[],
            window_elapsed_s=0.001,
            latest_server=latest_server,
            latest_host=latest_host,
            latest_metrics=latest_metrics,
            resource_samples=sampler.samples(),
            window_start_elapsed_s=0.0,
            performance_alerts=[],
            memory_alerts=[],
            alerts=alerts,
            max_sampling_gap_seconds=max_sampling_gap_s,
        )

        request_index = 0
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
            shape = select_shape(request_index)
            phase = "baseline" if elapsed_s < baseline_s else "endurance"
            health = health_check(base_url, timeout_s=10.0)
            if not health["ok"]:
                state.health_failures += 1
                state.last_error = f"health check failed: {health['error']}"
                add_alert(alerts, state.last_error)
            observation = run_stream_request(
                prompt=make_prompt_item(shape, request_index, tokenize=tokenize_prompt),
                model_id=args.model_id,
                base_url=base_url,
                timeout_s=args.request_timeout_seconds,
            )
            observation["phase"] = phase
            state.requests_attempted += 1
            if observation.get("ok"):
                state.requests_ok += 1
                state.consecutive_request_failures = 0
            else:
                state.requests_failed += 1
                state.consecutive_request_failures += 1
                state.last_error = str(observation.get("error") or "stream request failed")
                add_alert(alerts, state.last_error)

            drain = wait_for_quiescence(
                base_url=base_url,
                model_id=args.model_id,
                timeout_s=args.drain_timeout_seconds,
            )
            latest_metrics = drain["metrics"]
            if not latest_metrics.get("ok"):
                state.metric_scrape_failures += 1
            for message in guard_lifecycle(
                state=state,
                drain=drain,
                max_quiescent_kv_logical_mib=args.max_quiescent_kv_logical_mib,
            ):
                add_alert(alerts, message)
            latest_server = process_snapshot(process.pid)
            if not latest_server.get("alive"):
                raise RuntimeError("server PID disappeared during an endurance request")

            record = {
                "timestamp": utc_now(),
                "kind": "endurance_request",
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "request_index": request_index,
                "phase": phase,
                "shape": shape.name,
                "health": health,
                "request": observation,
                "lifecycle": drain,
                "server": latest_server,
            }
            window_records.append(record)
            if phase == "baseline":
                baseline_records.append(record)
            append_jsonl(events_path, record)

            if state.consecutive_request_failures >= args.max_consecutive_request_failures:
                raise RuntimeError(
                    "consecutive request failure limit reached: "
                    f"{state.consecutive_request_failures}/{args.max_consecutive_request_failures}"
                )

            now = time.monotonic()
            resource_samples = sampler.samples()
            if maybe_finalize_baseline(
                state=state,
                baseline_records=baseline_records,
                resource_samples=resource_samples,
                elapsed_s=now - started_monotonic,
                baseline_s=baseline_s,
                min_performance_samples=args.min_performance_samples,
                baseline_stability_growth_mib=args.baseline_stability_growth_mib,
                baseline_stability_slope_mib_per_hour=(args.baseline_stability_slope_mib_per_hour),
                max_swap_growth_mib=args.max_swap_growth_mib,
            ):
                for message in [
                    *state.baseline_coverage_concerns,
                    *state.baseline_stability_alerts,
                ]:
                    add_alert(alerts, message)
                append_jsonl(
                    events_path,
                    {
                        "timestamp": utc_now(),
                        "kind": "baseline_finalized",
                        "elapsed_seconds": now - started_monotonic,
                        "baseline": state.baseline,
                        "resource_medians_bytes": state.resource_baseline,
                    },
                )

            if now >= next_checkpoint:
                latest_host = collect_checkpoint_host_snapshot(
                    output_dir,
                    server_pid=process.pid,
                    reason="periodic",
                )
                latest_metrics = collect_metrics(base_url, args.model_id, timeout_s=10.0)
                if not latest_metrics.get("ok"):
                    state.metric_scrape_failures += 1
                checkpoint_start = last_checkpoint_monotonic or started_monotonic
                window_elapsed_s = now - checkpoint_start
                last_performance_alerts, last_memory_alerts = evaluate_window_guardrails(
                    state=state,
                    records=window_records,
                    window_elapsed_s=window_elapsed_s,
                    latest_metrics=latest_metrics,
                    resource_samples=resource_samples,
                    window_start_elapsed_s=checkpoint_start - started_monotonic,
                    min_performance_samples=args.min_performance_samples,
                    max_ttft_p95_ratio=args.max_ttft_p95_ratio,
                    min_decode_p05_ratio=args.min_decode_p05_ratio,
                    min_prefill_p05_ratio=args.min_prefill_p05_ratio,
                    max_client_error_rate=args.max_client_error_rate,
                    memory_growth_mib=args.memory_growth_mib,
                    memory_slope_mib_per_hour=args.memory_slope_mib_per_hour,
                    early_memory_growth_mib=args.early_memory_growth_mib,
                    early_memory_slope_mib_per_hour=args.early_memory_slope_mib_per_hour,
                    early_memory_positive_step_ratio=args.early_memory_positive_step_ratio,
                    max_swap_growth_mib=args.max_swap_growth_mib,
                )
                for message in [*last_performance_alerts, *last_memory_alerts]:
                    add_alert(alerts, message)
                state.performance_alerts += len(last_performance_alerts)
                state.memory_alerts += len(last_memory_alerts)
                write_checkpoint(
                    output_dir=output_dir,
                    reason="periodic",
                    state=state,
                    status="running",
                    elapsed_s=now - started_monotonic,
                    target_duration_s=target_duration_s,
                    records=window_records,
                    window_elapsed_s=window_elapsed_s,
                    latest_server=latest_server,
                    latest_host=latest_host,
                    latest_metrics=latest_metrics,
                    resource_samples=resource_samples,
                    window_start_elapsed_s=checkpoint_start - started_monotonic,
                    performance_alerts=last_performance_alerts,
                    memory_alerts=last_memory_alerts,
                    alerts=alerts,
                    max_sampling_gap_seconds=max_sampling_gap_s,
                )
                window_records.clear()
                last_checkpoint_monotonic = now
                while next_checkpoint <= now:
                    next_checkpoint += report_interval_s

            # Completion-based pacing avoids catch-up bursts after an unusually
            # slow prefill/decode.  This remains an endurance test, not a load test.
            delay = args.request_interval_seconds
            if delay > 0.0:
                time.sleep(delay)

    except PreflightComplete:
        status = "preflight_passed"
    except KeyboardInterrupt:
        status = "interrupted"
        failure = "runner received an interrupt signal"
    except Exception as error:  # noqa: BLE001 - failure cause is the primary result of a soak.
        status = "failed"
        failure = str(error)
    finally:
        try:
            if state is not None and not state.measurement_started:
                if status == "preflight_passed":
                    state.preflight_status = "passed"
                else:
                    state.preflight_status = "interrupted" if status == "interrupted" else "failed"
                state.preflight_elapsed_seconds = time.monotonic() - state.started_monotonic
            if sampler is not None and not sampler.stop() and state is not None:
                state.resource_sampler_stop_timeouts += 1
                add_alert(
                    alerts,
                    "resource sampler did not stop before the final checkpoint",
                )
            if state is not None:
                if failure:
                    state.last_error = failure
                    add_alert(alerts, failure)
                elapsed_s = time.monotonic() - state.started_monotonic
                if process is not None:
                    latest_server = process_snapshot(process.pid)
                latest_host = collect_checkpoint_host_snapshot(
                    output_dir,
                    server_pid=process.pid if process is not None else None,
                    reason="final",
                )
                latest_metrics = collect_metrics(base_url, args.model_id, timeout_s=10.0)
                if not latest_metrics.get("ok"):
                    state.metric_scrape_failures += 1
                resource_samples = sampler.samples() if sampler is not None else []
                if state.measurement_started and maybe_finalize_baseline(
                    state=state,
                    baseline_records=baseline_records,
                    resource_samples=resource_samples,
                    elapsed_s=elapsed_s,
                    baseline_s=baseline_s,
                    min_performance_samples=args.min_performance_samples,
                    baseline_stability_growth_mib=args.baseline_stability_growth_mib,
                    baseline_stability_slope_mib_per_hour=(
                        args.baseline_stability_slope_mib_per_hour
                    ),
                    max_swap_growth_mib=args.max_swap_growth_mib,
                ):
                    for message in [
                        *state.baseline_coverage_concerns,
                        *state.baseline_stability_alerts,
                    ]:
                        add_alert(alerts, message)
                checkpoint_start = last_checkpoint_monotonic or state.started_monotonic
                window_elapsed_s = max(
                    0.001,
                    elapsed_s - (checkpoint_start - state.started_monotonic),
                )
                window_start_elapsed_s = checkpoint_start - state.started_monotonic
                if window_records:
                    last_performance_alerts, last_memory_alerts = evaluate_window_guardrails(
                        state=state,
                        records=window_records,
                        window_elapsed_s=window_elapsed_s,
                        latest_metrics=latest_metrics,
                        resource_samples=resource_samples,
                        window_start_elapsed_s=window_start_elapsed_s,
                        min_performance_samples=args.min_performance_samples,
                        max_ttft_p95_ratio=args.max_ttft_p95_ratio,
                        min_decode_p05_ratio=args.min_decode_p05_ratio,
                        min_prefill_p05_ratio=args.min_prefill_p05_ratio,
                        max_client_error_rate=args.max_client_error_rate,
                        memory_growth_mib=args.memory_growth_mib,
                        memory_slope_mib_per_hour=args.memory_slope_mib_per_hour,
                        early_memory_growth_mib=args.early_memory_growth_mib,
                        early_memory_slope_mib_per_hour=args.early_memory_slope_mib_per_hour,
                        early_memory_positive_step_ratio=args.early_memory_positive_step_ratio,
                        max_swap_growth_mib=args.max_swap_growth_mib,
                    )
                    for message in [
                        *last_performance_alerts,
                        *last_memory_alerts,
                    ]:
                        add_alert(alerts, message)
                    state.performance_alerts += len(last_performance_alerts)
                    state.memory_alerts += len(last_memory_alerts)
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
                    target_duration_s=target_duration_s,
                    records=window_records,
                    window_elapsed_s=window_elapsed_s,
                    latest_server=latest_server,
                    latest_host=latest_host,
                    latest_metrics=latest_metrics,
                    resource_samples=resource_samples,
                    window_start_elapsed_s=window_start_elapsed_s,
                    performance_alerts=last_performance_alerts,
                    memory_alerts=last_memory_alerts,
                    alerts=alerts,
                    max_sampling_gap_seconds=max_sampling_gap_s,
                )
            else:
                write_json_atomic(
                    output_dir / "failure.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "timestamp": utc_now(),
                        "error": failure,
                    },
                )
        finally:
            try:
                if process is not None:
                    stop_result = stop_server(process)
                    append_jsonl(
                        events_path,
                        {
                            "timestamp": utc_now(),
                            "kind": "server_stopped",
                            "result": stop_result,
                        },
                    )
            finally:
                if log_handle is not None:
                    log_handle.close()

    if failure:
        print(f"AXQ endurance run {status}: {failure}", file=sys.stderr)
    else:
        print(f"AXQ endurance run {status}: {output_dir}")
    return (
        0 if status in {"completed", "preflight_passed"} else 130 if status == "interrupted" else 1
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit conservative CLI for the endurance runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument(
        "--expected-server-version",
        default=None,
        help="Require the server --version output to contain this exact version token.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen3.6-27b-axq-6bit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31418)
    parser.add_argument("--duration-hours", type=positive_float, default=DEFAULT_DURATION_HOURS)
    parser.add_argument(
        "--report-interval-hours", type=positive_float, default=DEFAULT_REPORT_INTERVAL_HOURS
    )
    parser.add_argument("--baseline-hours", type=positive_float, default=DEFAULT_BASELINE_HOURS)
    parser.add_argument(
        "--request-interval-seconds", type=positive_float, default=DEFAULT_REQUEST_INTERVAL_S
    )
    parser.add_argument(
        "--resource-interval-seconds", type=positive_float, default=DEFAULT_RESOURCE_INTERVAL_S
    )
    parser.add_argument(
        "--max-sampling-gap-seconds",
        type=positive_float,
        default=None,
        help=(
            "Maximum tolerated wall-clock gap between resource samples; defaults to "
            "a cadence-derived value."
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds", type=positive_float, default=DEFAULT_REQUEST_TIMEOUT_S
    )
    parser.add_argument(
        "--startup-timeout-seconds", type=positive_float, default=DEFAULT_STARTUP_TIMEOUT_S
    )
    parser.add_argument(
        "--drain-timeout-seconds", type=positive_float, default=DEFAULT_DRAIN_TIMEOUT_S
    )
    parser.add_argument("--max-batch-tokens", type=positive_int, default=DEFAULT_MAX_BATCH_TOKENS)
    parser.add_argument("--block-size-tokens", type=positive_int, default=DEFAULT_BLOCK_SIZE_TOKENS)
    parser.add_argument("--total-blocks", type=positive_int, default=DEFAULT_TOTAL_BLOCKS)
    parser.add_argument(
        "--max-capacity-fill-requests",
        type=non_negative_int,
        default=DEFAULT_MAX_CAPACITY_FILL_REQUESTS,
        help=(
            "Maximum bounded fresh-long requests used to make the post-warm-up long "
            "probe prove logical-KV reclamation before measured time."
        ),
    )
    parser.add_argument("--warmup-requests", type=non_negative_int, default=DEFAULT_WARMUP_REQUESTS)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run the full warm-up and cache-capacity rehearsal, then stop before the "
            "measured interval. A passing result is a preflight pass, not a soak pass."
        ),
    )
    parser.add_argument(
        "--max-consecutive-request-failures",
        type=positive_int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
    )
    parser.add_argument(
        "--max-client-error-rate",
        type=non_negative_float,
        default=DEFAULT_MAX_ERROR_RATE,
    )
    parser.add_argument(
        "--max-ttft-p95-ratio", type=positive_float, default=DEFAULT_MAX_TTFT_P95_RATIO
    )
    parser.add_argument(
        "--min-decode-p05-ratio", type=positive_float, default=DEFAULT_MIN_DECODE_P05_RATIO
    )
    parser.add_argument(
        "--min-prefill-p05-ratio",
        type=positive_float,
        default=DEFAULT_MIN_PREFILL_P05_RATIO,
    )
    parser.add_argument(
        "--min-performance-samples", type=positive_int, default=DEFAULT_MIN_PERFORMANCE_SAMPLES
    )
    parser.add_argument(
        "--memory-growth-mib", type=positive_float, default=DEFAULT_MEMORY_GROWTH_MIB
    )
    parser.add_argument(
        "--memory-slope-mib-per-hour",
        type=positive_float,
        default=DEFAULT_MEMORY_SLOPE_MIB_PER_HOUR,
    )
    parser.add_argument(
        "--early-memory-growth-mib",
        type=positive_float,
        default=DEFAULT_EARLY_MEMORY_GROWTH_MIB,
    )
    parser.add_argument(
        "--early-memory-slope-mib-per-hour",
        type=positive_float,
        default=DEFAULT_EARLY_MEMORY_SLOPE_MIB_PER_HOUR,
    )
    parser.add_argument(
        "--early-memory-positive-step-ratio",
        type=positive_float,
        default=DEFAULT_EARLY_MEMORY_POSITIVE_STEP_RATIO,
        help="Positive quiescent sample-step fraction required for an early memory warning.",
    )
    parser.add_argument(
        "--baseline-stability-growth-mib",
        type=positive_float,
        default=DEFAULT_BASELINE_STABILITY_GROWTH_MIB,
        help="Flag a baseline that rises by this much while still trending upward.",
    )
    parser.add_argument(
        "--baseline-stability-slope-mib-per-hour",
        type=positive_float,
        default=DEFAULT_BASELINE_STABILITY_SLOPE_MIB_PER_HOUR,
        help="Require this latter-half slope as well as baseline growth before flagging drift.",
    )
    parser.add_argument(
        "--max-swap-growth-mib",
        type=positive_float,
        default=DEFAULT_MAX_SWAP_GROWTH_MIB,
    )
    parser.add_argument(
        "--max-quiescent-kv-logical-mib",
        type=positive_float,
        default=DEFAULT_MAX_QUIESCENT_KV_LOGICAL_MIB,
    )
    parser.add_argument(
        "--max-prefill-steps-per-1k-tokens",
        type=positive_float,
        default=DEFAULT_MAX_PREFILL_STEPS_PER_1K_TOKENS,
        help=(
            "Fail the post-warm-up fresh cache-capacity probes when prefill is fragmented "
            "into more scheduler turns per 1,000 prompt tokens than this limit."
        ),
    )
    parser.add_argument(
        "--server-extra-arg",
        action="append",
        default=[],
        help="Repeat to pass a deliberate additional ax-engine-server argument.",
    )
    return parser


def main_with_args_for_test(argv: list[str]) -> int:
    """Entrypoint kept separately for simple test invocation."""
    return run_endurance(build_parser().parse_args(argv))


def main() -> None:
    """Run the command-line entrypoint."""
    raise SystemExit(main_with_args_for_test(sys.argv[1:]))


if __name__ == "__main__":
    main()
